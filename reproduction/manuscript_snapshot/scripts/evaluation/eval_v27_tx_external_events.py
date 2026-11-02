#!/usr/bin/env python3
"""Freeze and evaluate TexNet strong-motion events with the locked V27 model."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
from obspy import UTCDateTime, read_events, read_inventory
from obspy.geodetics import gps2dist_azimuth
import requests


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/evaluation"))

import pilot_pr_event_diagnostic as common  # noqa: E402
from analyze_v27_ncedc_event_scores import (  # noqa: E402
    event_cluster_bootstrap,
    extract_candidates,
    matched_candidate,
    metrics,
    support_with_resets,
)
from cspp_v27_reused_core_score_cache import (  # noqa: E402
    load_ensemble,
    release_device_cache,
    score_waveform_packets,
)
from oriented_gappy_strong_motion_stream import (  # noqa: E402
    ChannelOrientation,
    stream_to_oriented_gappy_zne_array,
)
from resilient_mseed import read_resilient_mseed  # noqa: E402
from v20c_event_snr import local_p_excess_power_snr  # noqa: E402


OUTPUT_ROOT = Path(
    "workspace/tx_external_event_validation_20260824"
)
LOCKED_SCORING_PROTOCOL = common.LOCKED_SCORING_PROTOCOL
NETWORK = "TX"
SOURCE = "tx"
START = "2021-01-01T00:00:00Z"
END = "2026-01-01T00:00:00Z"
MINIMUM_MAGNITUDE = 4.0
MAXIMUM_DISTANCE_KM = 200.0
MAXIMUM_STATIONS_PER_EVENT = 12
REGION = {
    "minlatitude": 28.0,
    "maxlatitude": 37.0,
    "minlongitude": -107.0,
    "maxlongitude": -98.0,
}
CATALOG_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"
STATION_URL = common.STATION_URL
DATASELECT_URL = common.DATASELECT_URL
DETAIL_URL = common.DETAIL_URL
USER_AGENT = "CSPP-TexNet-frozen-external-event-evaluation/1.0"

common.NETWORK = NETWORK
common.USER_AGENT = USER_AGENT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run"))
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    return parser.parse_args()


def write_npz_atomic(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp-{os.getpid()}.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def preferred(value: Any, plural: list[Any]) -> Any:
    return value or plural[0]


def fetch_cached(path: Path, url: str, **kwargs: Any) -> bytes:
    if path.exists():
        payload = path.read_bytes()
        if not payload:
            raise RuntimeError(f"Cached metadata is empty: {path}")
        return payload
    payload = common.fetch(url, **kwargs)
    common.write_once(path, payload)
    return payload


def manual_tx_p_arrivals(event: Any, origin: Any) -> dict[str, list[dict[str, Any]]]:
    picks = {str(pick.resource_id): pick for pick in event.picks}
    by_station: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for arrival in origin.arrivals:
        pick = picks.get(str(arrival.pick_id))
        if pick is None or pick.waveform_id is None:
            continue
        phase = str(arrival.phase or pick.phase_hint or "").upper()
        status = str(pick.evaluation_status or "").lower()
        if (
            not phase.startswith("P")
            or pick.waveform_id.network_code != NETWORK
            or str(pick.evaluation_mode or "").lower() != "manual"
            or status not in {"", "reviewed"}
        ):
            continue
        station = pick.waveform_id.station_code or ""
        if station:
            by_station[station].append(
                {
                    "pick_id": str(pick.resource_id).rstrip("/").rsplit("/", 1)[-1],
                    "pick_time_utc": common.iso(pick.time),
                    "pick_channel": pick.waveform_id.channel_code or "",
                    "pick_evaluation_status": status or "missing",
                    "arrival_weight": float(arrival.time_weight or 0.0),
                    "time_residual_seconds": float(arrival.time_residual or 0.0),
                }
            )
    return by_station


def prepare() -> None:
    protocol_path = OUTPUT_ROOT / "protocol.json"
    if protocol_path.exists():
        print(json.dumps(common.artifact(protocol_path), indent=2))
        return

    raw_root = OUTPUT_ROOT / "metadata/raw"
    manifest_root = OUTPUT_ROOT / "metadata/manifests"
    catalog_path = raw_root / "usgs_tx_region_m4_2021_2025.geojson"
    catalog_payload = fetch_cached(
        catalog_path,
        CATALOG_URL,
        params={
            "format": "geojson",
            "starttime": START,
            "endtime": END,
            "minmagnitude": MINIMUM_MAGNITUDE,
            "orderby": "time-asc",
            **REGION,
        },
    )
    features = json.loads(catalog_payload)["features"]
    event_ids = [
        item["id"] for item in features if str(item["id"]).startswith(SOURCE)
    ]
    if not event_ids:
        raise RuntimeError("USGS catalog contains no TexNet event IDs")

    raw_artifacts = [common.artifact(catalog_path)]
    event_paths: dict[str, Path] = {}
    metadata_status: dict[str, str] = {}
    for index, event_id in enumerate(event_ids, start=1):
        detail_path = raw_root / f"{event_id}.geojson"
        detail_payload = fetch_cached(
            detail_path, DETAIL_URL.format(event_id=event_id)
        )
        raw_artifacts.append(common.artifact(detail_path))
        detail = json.loads(detail_payload)
        products = detail.get("properties", {}).get("products", {}).get(
            "phase-data", []
        )
        products = [item for item in products if item.get("source") == SOURCE]
        if not products or "quakeml.xml" not in products[0].get("contents", {}):
            metadata_status[event_id] = "missing_tx_phase_product"
            continue
        phase_path = raw_root / f"{event_id}.quakeml.xml"
        phase_url = products[0]["contents"]["quakeml.xml"]["url"]
        fetch_cached(phase_path, phase_url)
        raw_artifacts.append(common.artifact(phase_path))
        event_paths[event_id] = phase_path
        metadata_status[event_id] = "phase_product_available"
        if index % 10 == 0:
            print(f"metadata {index}/{len(event_ids)}", flush=True)
        time.sleep(0.05)

    station_path = raw_root / "TX_HN_2021_2025_response.xml"
    fetch_cached(
        station_path,
        STATION_URL,
        params={
            "network": NETWORK,
            "channel": "HN?",
            "starttime": START,
            "endtime": END,
            "level": "response",
            "format": "xml",
            "nodata": 404,
        },
    )
    raw_artifacts.append(common.artifact(station_path))
    inventory = read_inventory(str(station_path))

    event_rows: list[dict[str, Any]] = []
    station_rows: list[dict[str, Any]] = []
    orientation_rows: list[dict[str, Any]] = []
    for event_id in event_ids:
        if event_id not in event_paths:
            event_rows.append(
                {
                    "event_id": event_id,
                    "origin_time_utc": "",
                    "magnitude": "",
                    "manual_tx_p_stations": 0,
                    "eligible_hn_station_count": 0,
                    "selected_station_count": 0,
                    "status": "excluded",
                    "exclusion_reason": metadata_status[event_id],
                }
            )
            continue
        event = read_events(str(event_paths[event_id]))[0]
        origin = preferred(event.preferred_origin(), event.origins)
        magnitude = preferred(event.preferred_magnitude(), event.magnitudes)
        arrivals = manual_tx_p_arrivals(event, origin)
        groups = common.channel_groups(inventory, origin.time)
        eligible: list[dict[str, Any]] = []
        for station, picks in arrivals.items():
            if station not in groups:
                continue
            group = groups[station][0]
            distance_m, _, _ = gps2dist_azimuth(
                float(origin.latitude),
                float(origin.longitude),
                group["latitude"],
                group["longitude"],
            )
            distance_km = distance_m / 1000.0
            if distance_km > MAXIMUM_DISTANCE_KM:
                continue
            reference = common.select_reference(picks)
            eligible.append(
                {
                    "event_id": event_id,
                    "origin_time_utc": common.iso(origin.time),
                    "event_latitude": float(origin.latitude),
                    "event_longitude": float(origin.longitude),
                    "depth_km": float(origin.depth or 0.0) / 1000.0,
                    "magnitude": float(magnitude.mag),
                    "magnitude_type": str(magnitude.magnitude_type or ""),
                    "network": NETWORK,
                    "station": station,
                    "location": group["location"],
                    "channels": ",".join(item.code for item in group["channels"]),
                    "native_sample_rate_hz": group["sample_rate_hz"],
                    "station_latitude": group["latitude"],
                    "station_longitude": group["longitude"],
                    "epicentral_distance_km": distance_km,
                    "primary_pick_id": reference["pick_id"],
                    "primary_pick_time_utc": reference["pick_time_utc"],
                    "primary_pick_evaluation_status": reference[
                        "pick_evaluation_status"
                    ],
                    "primary_arrival_weight": reference["arrival_weight"],
                    "primary_time_residual_seconds": reference[
                        "time_residual_seconds"
                    ],
                    "request_start_utc": common.iso(origin.time - 60.0),
                    "request_end_utc": common.iso(origin.time + 180.0),
                    "_channels": group["channels"],
                }
            )
        eligible.sort(
            key=lambda row: (
                row["epicentral_distance_km"],
                row["station"],
                row["location"],
            )
        )
        selected = eligible[:MAXIMUM_STATIONS_PER_EVENT]
        status = "selected" if len(selected) >= 3 else "excluded"
        event_rows.append(
            {
                "event_id": event_id,
                "origin_time_utc": common.iso(origin.time),
                "magnitude": float(magnitude.mag),
                "manual_tx_p_stations": len(arrivals),
                "eligible_hn_station_count": len(eligible),
                "selected_station_count": len(selected) if status == "selected" else 0,
                "status": status,
                "exclusion_reason": (
                    "" if status == "selected" else "fewer_than_three_eligible_hn_stations"
                ),
            }
        )
        if status != "selected":
            continue
        for rank, row in enumerate(selected, start=1):
            channels = row.pop("_channels")
            row["station_rank"] = rank
            station_rows.append(row)
            location_label = row["location"] or "--"
            group_id = f"{event_id}.{NETWORK}.{row['station']}.{location_label}"
            for component_index, channel in enumerate(channels):
                sensitivity = channel.response.instrument_sensitivity
                orientation_rows.append(
                    {
                        "group_id": group_id,
                        "event_id": event_id,
                        "network": NETWORK,
                        "station": row["station"],
                        "location": row["location"],
                        "component_index": component_index,
                        "channel": channel.code,
                        "azimuth_deg": float(channel.azimuth),
                        "dip_deg": float(channel.dip),
                        "scale": float(sensitivity.value),
                        "scale_units": str(sensitivity.input_units),
                    }
                )

    if not station_rows:
        raise RuntimeError("No TexNet event has three eligible strong-motion stations")
    event_path = manifest_root / "events.csv"
    station_manifest = manifest_root / "event_station_requests.csv"
    orientation_path = manifest_root / "event_station_orientations.csv"
    common.write_csv(event_path, event_rows, list(event_rows[0]))
    common.write_csv(station_manifest, station_rows, list(station_rows[0]))
    common.write_csv(orientation_path, orientation_rows, list(orientation_rows[0]))

    locked = json.loads(LOCKED_SCORING_PROTOCOL.read_text(encoding="utf-8"))
    protocol = {
        "protocol_id": "cspp_v27_tx_external_event_validation_v1",
        "status": "frozen_before_tx_waveform_access_and_model_scoring",
        "frozen_at_utc": datetime.now(UTC).isoformat(),
        "objective": (
            "Model-frozen external-network event-record validation on TexNet "
            "strong-motion acceleration."
        ),
        "evidence_boundary": {
            "target_event_waveforms_accessed_before_freeze": False,
            "target_model_outputs_computed_before_freeze": False,
            "metadata_accessed_before_freeze": True,
            "claim": (
                "External event-record validation; not a chronological continuous "
                "false-alarm estimate or a prospectively registered trial."
            ),
            "current_pipeline_overlap": (
                "V27 training, development, and operating-point selection contain "
                "no TexNet event records or TX continuous windows."
            ),
        },
        "selection": {
            "period": [START, END],
            "region": REGION,
            "minimum_magnitude": MINIMUM_MAGNITUDE,
            "event_id_rule": "USGS preferred event ID begins with tx",
            "reference": (
                "TexNet-contributed manual P arrival. The provider product omits "
                "evaluation_status; missing values are retained as missing."
            ),
            "maximum_distance_km": MAXIMUM_DISTANCE_KM,
            "minimum_stations_per_event": 3,
            "maximum_stations_per_event": MAXIMUM_STATIONS_PER_EVENT,
            "station_order": "epicentral distance, station, location",
            "accepted_sample_rates_hz": sorted(common.ACCEPTED_SAMPLE_RATES),
            "unavailable_record_policy": "retain as unavailable; no replacement",
        },
        "counts": {
            "catalog_tx_events": len(event_ids),
            "selected_events": sum(row["status"] == "selected" for row in event_rows),
            "selected_event_station_records": len(station_rows),
        },
        "inputs": {
            "locked_scoring_protocol": common.artifact(LOCKED_SCORING_PROTOCOL),
            "events": common.artifact(event_path),
            "event_station_requests": common.artifact(station_manifest),
            "event_station_orientations": common.artifact(orientation_path),
            "raw_metadata": raw_artifacts,
        },
        "code": {
            "driver": common.artifact(Path(__file__)),
            "shared_pilot_helpers": common.artifact(common.Path(common.__file__)),
        },
        "models": locked["models"],
        "runtime": locked["runtime"],
        "operating_point": locked["operating_point"],
        "metrics": locked["metrics"],
        "bootstrap": {"seed": 20260824, "replicates": 5000},
        "prohibited": [
            "changing model, threshold, packet size, filter, or match tolerance",
            "replacing unavailable records",
            "selecting a reported seed from target performance",
            "using target output to revise this primary analysis",
            "automatic manuscript modification",
        ],
    }
    common.write_once(protocol_path, common.json_bytes(protocol))
    print(
        json.dumps(
            {
                "protocol": common.artifact(protocol_path),
                **protocol["counts"],
                "status": protocol["status"],
            },
            indent=2,
        )
    )


def verify_protocol() -> dict[str, Any]:
    path = OUTPUT_ROOT / "protocol.json"
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen_before_tx_waveform_access_and_model_scoring":
        raise RuntimeError("Unexpected TexNet protocol status")
    for spec in protocol["code"].values():
        common.verify_artifact(spec)
    for key in (
        "locked_scoring_protocol",
        "events",
        "event_station_requests",
        "event_station_orientations",
    ):
        common.verify_artifact(protocol["inputs"][key])
    for spec in protocol["inputs"]["raw_metadata"]:
        common.verify_artifact(spec)
    for spec in protocol["models"].values():
        common.verify_artifact(spec)
    return protocol


def request_body(rows: list[dict[str, str]]) -> bytes:
    lines = []
    for row in sorted(rows, key=lambda item: int(item["station_rank"])):
        for channel in row["channels"].split(","):
            lines.append(
                " ".join(
                    (
                        row["network"],
                        row["station"],
                        row["location"] or "--",
                        channel,
                        row["request_start_utc"],
                        row["request_end_utc"],
                    )
                )
            )
    return ("\n".join(lines) + "\n").encode("ascii")


def download_event(event_id: str, rows: list[dict[str, str]]) -> Path:
    root = OUTPUT_ROOT / "waveforms" / event_id
    request_path = root / "request.txt"
    mseed_path = root / "waveforms.mseed"
    body = request_body(rows)
    if mseed_path.exists():
        if not request_path.is_file() or request_path.read_bytes() != body:
            raise RuntimeError(f"Existing request differs: {event_id}")
        return mseed_path
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            response = requests.post(
                DATASELECT_URL,
                data=body,
                headers={"Content-Type": "text/plain", "User-Agent": USER_AGENT},
                timeout=(30, 300),
            )
            response.raise_for_status()
            if not response.content:
                raise RuntimeError("empty DataSelect response")
            common.write_once(request_path, body)
            common.write_once(mseed_path, response.content)
            return mseed_path
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < 3:
                time.sleep(2**attempt)
    raise RuntimeError(f"Waveform request failed: {event_id}") from last_error


def run(device: str) -> None:
    result_path = OUTPUT_ROOT / "results/summary.json"
    if result_path.exists():
        print(result_path.read_text(encoding="utf-8"))
        return
    protocol = verify_protocol()
    stations = common.read_csv(
        Path(protocol["inputs"]["event_station_requests"]["path"])
    )
    orientations = common.read_csv(
        Path(protocol["inputs"]["event_station_orientations"]["path"])
    )
    by_event: dict[str, list[dict[str, str]]] = defaultdict(list)
    by_group: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in stations:
        by_event[row["event_id"]].append(row)
    for row in orientations:
        by_group[row["group_id"]].append(row)
    event_ids = sorted(by_event, key=lambda key: by_event[key][0]["origin_time_utc"])
    ensemble = load_ensemble(
        {
            key: {
                "path": spec["path"],
                "sha256": spec["sha256"],
                "verifier_feature_mode": spec["verifier_feature_mode"],
            }
            for key, spec in protocol["models"].items()
        },
        device_name=device,
        torch_threads=4,
    )

    rows_out: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    waveforms: dict[str, dict[str, Any]] = {}
    for event_index, event_id in enumerate(event_ids, start=1):
        event_stations = by_event[event_id]
        try:
            mseed_path = download_event(event_id, event_stations)
            waveforms[event_id] = common.artifact(mseed_path)
            source = read_resilient_mseed(mseed_path).stream
        except Exception as exc:  # noqa: BLE001
            for station in event_stations:
                failures.append(
                    {
                        "event_id": event_id,
                        "station_key": station["station"],
                        "reason": f"event_download: {type(exc).__name__}: {exc}",
                    }
                )
            print(f"[{event_index}/{len(event_ids)}] {event_id}: download failed")
            continue
        for station in event_stations:
            location_label = station["location"] or "--"
            station_key = f"{NETWORK}.{station['station']}.{location_label}"
            group_id = f"{event_id}.{station_key}"
            orientation = [
                ChannelOrientation(
                    network=row["network"],
                    station=row["station"],
                    location=row["location"],
                    channel=row["channel"],
                    azimuth=float(row["azimuth_deg"]),
                    dip=float(row["dip_deg"]),
                    scale=float(row["scale"]),
                    scale_units=row["scale_units"],
                )
                for row in sorted(
                    by_group[group_id], key=lambda item: int(item["component_index"])
                )
            ]
            try:
                start = UTCDateTime(station["request_start_utc"])
                end = UTCDateTime(station["request_end_utc"])
                wave, metadata = stream_to_oriented_gappy_zne_array(
                    source,
                    station,
                    orientation,
                    start,
                    end,
                    target_sample_rate_hz=float(protocol["runtime"]["sample_rate_hz"]),
                    samples_per_packet=int(protocol["runtime"]["packet_samples"]),
                )
                reference = float(UTCDateTime(station["primary_pick_time_utc"]).timestamp)
                sample_rate = float(protocol["runtime"]["sample_rate_hz"])
                local_snr = local_p_excess_power_snr(
                    wave,
                    p_sample=int(round((reference - float(start.timestamp)) * sample_rate)),
                    sample_rate_hz=sample_rate,
                    window_seconds=float(protocol["metrics"]["local_p_snr_window_seconds"]),
                )
                cache = score_waveform_packets(
                    wave,
                    start_epoch=float(start.timestamp),
                    ensemble=ensemble,
                    runtime=protocol["runtime"],
                )
                cache_path = OUTPUT_ROOT / "scores" / event_id / f"{station_key}.npz"
                write_npz_atomic(cache_path, cache.arrays())
                support = support_with_resets(
                    cache.candidate_probability,
                    cache.candidate_allowed,
                    cache.quality_valid,
                    float(protocol["operating_point"]["candidate_threshold"]),
                    int(protocol["runtime"]["candidate_history_packets"]),
                )
                for model_key, model_spec in protocol["models"].items():
                    candidates = extract_candidates(
                        support=support,
                        valid=cache.quality_valid,
                        event_probability=cache.event_probability[model_key],
                        arrival_age=cache.arrival_age_fraction,
                        event_threshold=float(protocol["operating_point"]["event_threshold"]),
                        start_epoch=float(start.timestamp),
                        packet_seconds=float(protocol["runtime"]["packet_duration_seconds"]),
                        max_arrival_age_packets=float(
                            protocol["runtime"]["max_arrival_age_packets"]
                        ),
                    )
                    match = matched_candidate(
                        candidates,
                        reference,
                        float(protocol["metrics"]["match_tolerance_seconds"]),
                    )
                    error = (
                        float(match["estimated_p_epoch"]) - reference
                        if match is not None
                        else None
                    )
                    availability = (
                        float(match["available_epoch"]) - reference
                        if match is not None
                        else None
                    )
                    rows_out.append(
                        {
                            "event_id": event_id,
                            "station_key": station_key,
                            "model_key": model_key,
                            "seed": int(model_spec["seed"]),
                            "magnitude": float(station["magnitude"]),
                            "epicentral_distance_km": float(
                                station["epicentral_distance_km"]
                            ),
                            "native_sample_rate_hz": float(
                                station["native_sample_rate_hz"]
                            ),
                            "complete_three_component_fraction": float(
                                metadata["complete_three_component_fraction"]
                            ),
                            "snr_valid": bool((local_snr or {}).get("valid", False)),
                            "snr_db": (local_snr or {}).get("snr_db"),
                            "candidate_count": len(candidates),
                            "matched_candidate_count": int(match is not None),
                            "matched": match is not None,
                            "false_candidate_count": len(candidates) - int(match is not None),
                            "pick_error_seconds": error,
                            "absolute_pick_error_seconds": (
                                abs(error) if error is not None else None
                            ),
                            "availability_delay_seconds": availability,
                            "score_cache": str(cache_path),
                        }
                    )
            except Exception as exc:  # noqa: BLE001
                failures.append(
                    {
                        "event_id": event_id,
                        "station_key": station_key,
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                )
            release_device_cache(ensemble.device_name)
        print(f"[{event_index}/{len(event_ids)}] {event_id}: scored", flush=True)

    if not rows_out:
        raise RuntimeError("No TexNet event-station record was scorable")
    results_root = OUTPUT_ROOT / "results"
    station_result_path = results_root / "event_station_results.csv"
    common.write_csv(station_result_path, rows_out, list(rows_out[0]))
    primary_key = next(
        key
        for key, spec in protocol["models"].items()
        if int(spec["seed"])
        == int(protocol["operating_point"]["primary_operational_seed"])
    )
    by_model = {
        key: metrics([row for row in rows_out if row["model_key"] == key])
        for key in protocol["models"]
    }
    primary_rows = [row for row in rows_out if row["model_key"] == primary_key]
    cluster_ci = event_cluster_bootstrap(
        primary_rows,
        seed=int(protocol["bootstrap"]["seed"]),
        replicates=int(protocol["bootstrap"]["replicates"]),
    )
    event_rows = []
    for event_id in event_ids:
        values = [row for row in primary_rows if row["event_id"] == event_id]
        if not values:
            continue
        matched = sum(row["matched"] for row in values)
        event_rows.append(
            {
                "event_id": event_id,
                "magnitude": values[0]["magnitude"],
                "scorable_station_records": len(values),
                "matched_station_records": matched,
                "detected_by_at_least_one_station": matched >= 1,
                "detected_by_at_least_three_stations": matched >= 3,
            }
        )
    event_result_path = results_root / "event_results.csv"
    common.write_csv(event_result_path, event_rows, list(event_rows[0]))
    summary = {
        "summary_id": "cspp_v27_tx_external_event_validation_v1",
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "protocol": common.artifact(OUTPUT_ROOT / "protocol.json"),
        "requested_events": len(event_ids),
        "requested_event_station_records": len(stations),
        "scorable_event_station_records": len(
            {(row["event_id"], row["station_key"]) for row in rows_out}
        ),
        "unscorable_event_station_records": failures,
        "primary_model_key": primary_key,
        "primary_metrics": by_model[primary_key],
        "primary_event_cluster_bootstrap95": cluster_ci,
        "primary_event_detection": {
            "events": len(event_rows),
            "at_least_one_station": sum(
                row["detected_by_at_least_one_station"] for row in event_rows
            ),
            "at_least_three_stations": sum(
                row["detected_by_at_least_three_stations"] for row in event_rows
            ),
        },
        "replication_metrics": by_model,
        "waveforms": waveforms,
        "artifacts": {
            "event_station_results": common.artifact(station_result_path),
            "event_results": common.artifact(event_result_path),
        },
        "paper_or_submission_files_modified": False,
    }
    common.write_json_atomic(result_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))


def main() -> None:
    args = parse_args()
    if args.stage == "prepare":
        prepare()
    else:
        run(args.device)


if __name__ == "__main__":
    main()
