#!/usr/bin/env python3
"""Run a frozen, small Puerto Rico strong-motion event diagnostic."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Iterable

import numpy as np
from obspy import UTCDateTime, read_events, read_inventory
from obspy.geodetics import gps2dist_azimuth
import requests


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/evaluation"))

from analyze_v27_ncedc_event_scores import (  # noqa: E402
    extract_candidates,
    matched_candidate,
    metrics,
    support_with_resets,
)
from cspp_v27_reused_core_score_cache import (  # noqa: E402
    load_ensemble,
    release_device_cache,
    score_waveform_packets,
    sha256_file,
)
from oriented_gappy_strong_motion_stream import (  # noqa: E402
    ChannelOrientation,
    stream_to_oriented_gappy_zne_array,
)
from resilient_mseed import read_resilient_mseed  # noqa: E402
from v20c_event_snr import local_p_excess_power_snr  # noqa: E402


OUTPUT_ROOT = Path(
    "workspace/pr_pilot_20260824"
)
LOCKED_SCORING_PROTOCOL = Path(
    "workspace/"
    "ncedc_model_blind_20260817/event_scoring_protocol.json"
)
EVENT_IDS = ("pr2024136000", "pr2024119002", "pr2024239000")
NETWORK = "PR"
MAXIMUM_DISTANCE_KM = 200.0
MAXIMUM_STATIONS_PER_EVENT = 6
ACCEPTED_SAMPLE_RATES = {100.0, 200.0}
CHANNEL_PATTERNS = (("HNZ", "HNN", "HNE"), ("HNZ", "HN1", "HN2"))
STATION_URL = "https://service.iris.edu/fdsnws/station/1/query"
DATASELECT_URL = "https://service.iris.edu/fdsnws/dataselect/1/query"
DETAIL_URL = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/detail/{event_id}.geojson"
USER_AGENT = "CSPP-PR-diagnostic-pilot/1.0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run"))
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    return parser.parse_args()


def iso(value: Any) -> str:
    return value.datetime.replace(tzinfo=UTC).isoformat().replace("+00:00", "Z")


def artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def write_once(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_bytes(json_bytes(payload))
    temporary.replace(path)


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def fetch(url: str, *, params: dict[str, Any] | None = None) -> bytes:
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            response = requests.get(
                url,
                params=params,
                headers={"User-Agent": USER_AGENT},
                timeout=(30, 180),
            )
            response.raise_for_status()
            if not response.content:
                raise RuntimeError(f"Empty response: {response.url}")
            return response.content
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < 3:
                time.sleep(2**attempt)
    raise RuntimeError(f"Request failed: {url}") from last_error


def active(channel: Any, epoch: UTCDateTime) -> bool:
    return not (
        (channel.start_date is not None and epoch < channel.start_date)
        or (channel.end_date is not None and epoch > channel.end_date)
    )


def channel_groups(inventory: Any, epoch: UTCDateTime) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for network in inventory:
        if network.code != NETWORK:
            continue
        for station in network:
            for channel in station:
                if not channel.code.startswith("HN") or not active(channel, epoch):
                    continue
                location = channel.location_code or ""
                previous = grouped[station.code][location].get(channel.code)
                if previous is None or (channel.start_date or UTCDateTime(0)) > (
                    previous.start_date or UTCDateTime(0)
                ):
                    grouped[station.code][location][channel.code] = channel

    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for station, locations in grouped.items():
        for location, channels in locations.items():
            for pattern_index, pattern in enumerate(CHANNEL_PATTERNS):
                if not set(pattern).issubset(channels):
                    continue
                selected = [channels[code] for code in pattern]
                rates = [float(channel.sample_rate) for channel in selected]
                if max(rates) - min(rates) > 1e-6 or rates[0] not in ACCEPTED_SAMPLE_RATES:
                    continue
                sensitivities = [
                    getattr(channel.response, "instrument_sensitivity", None)
                    for channel in selected
                ]
                if any(item is None or float(item.value) <= 0.0 for item in sensitivities):
                    continue
                result[station].append(
                    {
                        "location": location,
                        "pattern_index": pattern_index,
                        "channels": selected,
                        "sample_rate_hz": rates[0],
                        "latitude": float(np.mean([item.latitude for item in selected])),
                        "longitude": float(np.mean([item.longitude for item in selected])),
                    }
                )
    for values in result.values():
        values.sort(key=lambda row: (row["pattern_index"], row["location"]))
    return result


def manual_p_arrivals(event: Any, origin: Any) -> dict[str, list[dict[str, Any]]]:
    picks = {str(pick.resource_id): pick for pick in event.picks}
    by_station: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for arrival in origin.arrivals:
        pick = picks.get(str(arrival.pick_id))
        if pick is None or pick.waveform_id is None:
            continue
        phase = str(arrival.phase or pick.phase_hint or "").upper()
        if (
            not phase.startswith("P")
            or pick.waveform_id.network_code != NETWORK
            or str(pick.evaluation_mode or "").lower() != "manual"
            or str(pick.evaluation_status or "").lower() != "reviewed"
        ):
            continue
        station = pick.waveform_id.station_code or ""
        if not station:
            continue
        by_station[station].append(
            {
                "pick_id": str(pick.resource_id).rstrip("/").rsplit("/", 1)[-1],
                "pick_time_utc": iso(pick.time),
                "pick_channel": pick.waveform_id.channel_code or "",
                "arrival_weight": float(arrival.time_weight or 0.0),
                "time_residual_seconds": float(arrival.time_residual or 0.0),
            }
        )
    return by_station


def select_reference(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return min(
        rows,
        key=lambda row: (
            -float(row["arrival_weight"]),
            abs(float(row["time_residual_seconds"])),
            row["pick_time_utc"],
        ),
    )


def prepare() -> None:
    protocol_path = OUTPUT_ROOT / "protocol.json"
    if protocol_path.exists():
        print(json.dumps(artifact(protocol_path), indent=2))
        return
    raw_root = OUTPUT_ROOT / "metadata/raw"
    manifest_root = OUTPUT_ROOT / "metadata/manifests"
    event_paths: dict[str, Path] = {}
    for event_id in EVENT_IDS:
        detail_path = raw_root / f"{event_id}.geojson"
        detail = fetch(DETAIL_URL.format(event_id=event_id))
        write_once(detail_path, detail)
        payload = json.loads(detail)
        products = payload["properties"]["products"]["phase-data"]
        product = next(item for item in products if item["source"] == "pr")
        phase_url = product["contents"]["quakeml.xml"]["url"]
        phase_path = raw_root / f"{event_id}.quakeml.xml"
        write_once(phase_path, fetch(phase_url))
        event_paths[event_id] = phase_path

    station_path = raw_root / "PR_HN_2024_response.xml"
    write_once(
        station_path,
        fetch(
            STATION_URL,
            params={
                "network": NETWORK,
                "channel": "HN?",
                "starttime": "2024-01-01T00:00:00Z",
                "endtime": "2025-01-01T00:00:00Z",
                "level": "response",
                "format": "xml",
                "nodata": 404,
            },
        ),
    )
    inventory = read_inventory(str(station_path))
    event_rows: list[dict[str, Any]] = []
    station_rows: list[dict[str, Any]] = []
    orientation_rows: list[dict[str, Any]] = []
    for event_id in EVENT_IDS:
        event = read_events(str(event_paths[event_id]))[0]
        origin = event.preferred_origin() or event.origins[0]
        magnitude = event.preferred_magnitude() or event.magnitudes[0]
        arrivals = manual_p_arrivals(event, origin)
        groups = channel_groups(inventory, origin.time)
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
            reference = select_reference(picks)
            eligible.append(
                {
                    "event_id": event_id,
                    "origin_time_utc": iso(origin.time),
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
                    "primary_arrival_weight": reference["arrival_weight"],
                    "primary_time_residual_seconds": reference[
                        "time_residual_seconds"
                    ],
                    "request_start_utc": iso(origin.time - 60.0),
                    "request_end_utc": iso(origin.time + 180.0),
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
        event_rows.append(
            {
                "event_id": event_id,
                "origin_time_utc": iso(origin.time),
                "magnitude": float(magnitude.mag),
                "manual_reviewed_pr_p_stations": len(arrivals),
                "eligible_hn_station_count": len(eligible),
                "selected_station_count": len(selected),
                "status": "selected" if len(selected) >= 3 else "insufficient_stations",
            }
        )
        for rank, row in enumerate(selected, start=1):
            channels = row.pop("_channels")
            row["station_rank"] = rank
            station_rows.append(row)
            group_id = f"{event_id}.{NETWORK}.{row['station']}.{row['location'] or '--'}"
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

    if any(row["status"] != "selected" for row in event_rows):
        raise RuntimeError(f"Pilot event lacks three eligible stations: {event_rows}")
    station_fields = [key for key in station_rows[0] if not key.startswith("_")]
    event_path = manifest_root / "events.csv"
    station_manifest = manifest_root / "event_station_requests.csv"
    orientation_path = manifest_root / "event_station_orientations.csv"
    write_csv(event_path, event_rows, list(event_rows[0]))
    write_csv(station_manifest, station_rows, station_fields)
    write_csv(orientation_path, orientation_rows, list(orientation_rows[0]))

    locked = json.loads(LOCKED_SCORING_PROTOCOL.read_text(encoding="utf-8"))
    protocol = {
        "protocol_id": "cspp_v27_pr_event_diagnostic_pilot_v1",
        "status": "frozen_before_pr_waveform_access_and_model_scoring",
        "frozen_at_utc": datetime.now(UTC).isoformat(),
        "scope": "Diagnostic pilot only; not manuscript evidence or an independent continuous-data validation.",
        "selection": {
            "event_ids": list(EVENT_IDS),
            "network": NETWORK,
            "maximum_distance_km": MAXIMUM_DISTANCE_KM,
            "maximum_stations_per_event": MAXIMUM_STATIONS_PER_EVENT,
            "order": "ascending epicentral distance, station, and location",
            "reference": "PR-contributed manual reviewed P arrival at the same station",
            "accepted_sample_rates_hz": sorted(ACCEPTED_SAMPLE_RATES),
        },
        "overlap_disclosure": (
            "No PR rows were found in local CSV evidence before preparation. "
            "InstanceGM provider overlap was not exhaustively resolvable; this remains diagnostic."
        ),
        "inputs": {
            "locked_scoring_protocol": artifact(LOCKED_SCORING_PROTOCOL),
            "station_metadata": artifact(station_path),
            "events": artifact(event_path),
            "event_station_requests": artifact(station_manifest),
            "event_station_orientations": artifact(orientation_path),
            "phase_metadata": {
                event_id: artifact(path) for event_id, path in event_paths.items()
            },
        },
        "code": artifact(Path(__file__)),
        "models": locked["models"],
        "runtime": locked["runtime"],
        "operating_point": locked["operating_point"],
        "metrics": locked["metrics"],
        "data_source": {
            "station": STATION_URL,
            "waveform": DATASELECT_URL,
            "event_products": "USGS ComCat PR phase-data products",
        },
        "prohibited": [
            "model or threshold tuning from pilot output",
            "replacement of unavailable selected records",
            "use as a continuous false-alarm estimate",
            "automatic insertion into the manuscript",
        ],
    }
    write_once(protocol_path, json_bytes(protocol))
    print(
        json.dumps(
            {
                "protocol": artifact(protocol_path),
                "events": len(event_rows),
                "station_records": len(station_rows),
                "status": protocol["status"],
            },
            indent=2,
        )
    )


def verify_artifact(spec: dict[str, Any]) -> Path:
    path = Path(spec["path"])
    if (
        not path.is_file()
        or path.stat().st_size != int(spec["bytes"])
        or sha256_file(path) != str(spec["sha256"])
    ):
        raise RuntimeError(f"Artifact drifted: {path}")
    return path


def download_event(rows: list[dict[str, str]], event_id: str) -> Path:
    event_root = OUTPUT_ROOT / "waveforms" / event_id
    mseed_path = event_root / "waveforms.mseed"
    request_path = event_root / "request.txt"
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
    body = ("\n".join(lines) + "\n").encode("ascii")
    if mseed_path.exists():
        if not request_path.is_file() or request_path.read_bytes() != body:
            raise RuntimeError(f"Existing request differs: {event_id}")
        return mseed_path
    response = requests.post(
        DATASELECT_URL,
        data=body,
        headers={"Content-Type": "text/plain", "User-Agent": USER_AGENT},
        timeout=(30, 300),
    )
    response.raise_for_status()
    if not response.content:
        raise RuntimeError(f"Empty waveform response: {event_id}")
    write_once(request_path, body)
    write_once(mseed_path, response.content)
    return mseed_path


def run(device: str) -> None:
    protocol_path = OUTPUT_ROOT / "protocol.json"
    if not protocol_path.is_file():
        raise RuntimeError("Run the prepare stage first")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen_before_pr_waveform_access_and_model_scoring":
        raise RuntimeError("Unexpected pilot protocol status")
    verify_artifact(protocol["code"])
    inputs = protocol["inputs"]
    for key in (
        "locked_scoring_protocol",
        "station_metadata",
        "events",
        "event_station_requests",
        "event_station_orientations",
    ):
        verify_artifact(inputs[key])
    for spec in inputs["phase_metadata"].values():
        verify_artifact(spec)
    for spec in protocol["models"].values():
        verify_artifact(spec)

    stations = read_csv(Path(inputs["event_station_requests"]["path"]))
    orientations = read_csv(Path(inputs["event_station_orientations"]["path"]))
    by_event: dict[str, list[dict[str, str]]] = defaultdict(list)
    by_group: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in stations:
        by_event[row["event_id"]].append(row)
    for row in orientations:
        by_group[row["group_id"]].append(row)
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
    waveform_artifacts = {}
    for event_id in EVENT_IDS:
        event_stations = by_event[event_id]
        mseed_path = download_event(event_stations, event_id)
        waveform_artifacts[event_id] = artifact(mseed_path)
        source = read_resilient_mseed(mseed_path).stream
        for station in event_stations:
            location_label = station["location"] or "--"
            group_id = f"{event_id}.{NETWORK}.{station['station']}.{location_label}"
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
                            "station_key": f"{NETWORK}.{station['station']}.{location_label}",
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
                        }
                    )
            except Exception as exc:  # noqa: BLE001
                failures.append(
                    {
                        "event_id": event_id,
                        "station_key": f"{NETWORK}.{station['station']}.{location_label}",
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                )
            release_device_cache(ensemble.device_name)
        print(f"{event_id}: scored", flush=True)

    result_path = OUTPUT_ROOT / "results/event_station_results.csv"
    if rows_out:
        write_csv(result_path, rows_out, list(rows_out[0]))
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
    summary = {
        "summary_id": "cspp_v27_pr_event_diagnostic_pilot_v1",
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "protocol": artifact(protocol_path),
        "scope": protocol["scope"],
        "requested_events": len(EVENT_IDS),
        "requested_event_station_records": len(stations),
        "scorable_event_station_records": len(
            {row["event_id"] + row["station_key"] for row in rows_out}
        ),
        "unscorable_event_station_records": failures,
        "primary_model_key": primary_key,
        "primary_metrics": by_model[primary_key],
        "replication_metrics": by_model,
        "waveforms": waveform_artifacts,
        "paper_or_submission_files_modified": False,
    }
    write_json_atomic(OUTPUT_ROOT / "results/summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))


def main() -> None:
    args = parse_args()
    if args.stage == "prepare":
        prepare()
    else:
        run(args.device)


if __name__ == "__main__":
    main()
