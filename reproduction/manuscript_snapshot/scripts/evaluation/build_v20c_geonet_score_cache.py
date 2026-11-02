#!/usr/bin/env python3
"""Build one frozen V20c GeoNet event or background score-cache unit."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import resource
import sys
import time
import zipfile
from collections import defaultdict
from typing import Any, Iterable

import numpy as np
from obspy import Stream, UTCDateTime, read


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from oriented_gappy_strong_motion_stream import (  # noqa: E402
    ChannelOrientation,
    stream_to_oriented_gappy_zne_array,
)
from v20c_causal_score_cache import (  # noqa: E402
    emit_frozen_policy_candidates,
    load_frozen_models,
    release_device_cache,
    score_waveform_packets,
    sha256,
)
from v20c_event_snr import local_p_excess_power_snr  # noqa: E402


PROTOCOL_ID = (
    "domain_robust_v20c_geonet_confirmation_v2_physical_station"
)
DEFAULT_PROTOCOL = (
    ROOT
    / "experiments/"
    "domain_robust_v20c_geonet_confirmation_protocol_20260724.json"
)
EVENT_WINDOW_BEFORE_SECONDS = 60.0
EVENT_WINDOW_AFTER_SECONDS = 120.0

CANDIDATE_FIELDS = [
    "record_key",
    "scope",
    "event_id",
    "cohort",
    "date_utc",
    "network",
    "station",
    "location",
    "station_key",
    "physical_station_id",
    "episode_index",
    "episode_start_packet",
    "episode_end_packet",
    "episode_duration_packets",
    "packet_index",
    "packet_start_epoch",
    "available_epoch",
    "estimated_p_epoch",
    "candidate_probability",
    "event_probability",
    "arrival_age_fraction",
    "artifact_probability",
    "label_source",
    "reference_p_epoch",
    "pick_match",
    "strict_false_candidate",
    "latency_from_reference_p_seconds",
    "latency_from_origin_seconds",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument(
        "--scope", choices=("event", "background"), required=True
    )
    parser.add_argument("--unit-key", required=True)
    return parser.parse_args()


def verify_event_window_contract(protocol: dict[str, Any]) -> None:
    event_window = protocol["event_window_contract"]
    if (
        float(event_window["before_origin_seconds"])
        != EVENT_WINDOW_BEFORE_SECONDS
        or float(event_window["after_origin_seconds"])
        != EVENT_WINDOW_AFTER_SECONDS
        or float(event_window["local_p_snr_pre_seconds"]) != 1.0
        or float(event_window["local_p_snr_post_seconds"]) != 1.0
    ):
        raise RuntimeError("Frozen event-window contract differs")


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def artifact(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256(resolved),
    }


def verify_artifact(spec: dict[str, Any], *, name: str) -> Path:
    path = resolve(spec["path"])
    if not path.is_file():
        raise FileNotFoundError(f"Missing {name}: {path}")
    if path.stat().st_size != int(spec["bytes"]):
        raise RuntimeError(f"Byte count drifted for {name}: {path}")
    actual = sha256(path)
    if actual != str(spec["sha256"]):
        raise RuntimeError(
            f"SHA-256 drifted for {name}: expected {spec['sha256']}, "
            f"got {actual}"
        )
    return path


def load_protocol(path: Path) -> tuple[dict[str, Any], str]:
    protocol_path = path.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("protocol_id") != PROTOCOL_ID:
        raise RuntimeError("Unexpected V20c GeoNet confirmation protocol")
    for relative, expected in protocol["code_sha256"].items():
        source = resolve(relative)
        if not source.is_file() or sha256(source) != str(expected):
            raise RuntimeError(f"Confirmation code drifted: {relative}")
    for name, spec in protocol["inputs"].items():
        verify_artifact(spec, name=f"input.{name}")
    for name, spec in protocol["model_artifacts"].items():
        verify_artifact(spec, name=f"model_artifact.{name}")
    return protocol, sha256(protocol_path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def protocol_csv(
    protocol: dict[str, Any], name: str
) -> list[dict[str, str]]:
    return read_csv(
        verify_artifact(protocol["inputs"][name], name=f"input.{name}")
    )


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_csv_atomic(
    path: Path,
    rows: Iterable[dict[str, Any]],
    fieldnames: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="raise",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def write_npz_atomic(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.stem}.tmp-{os.getpid()}.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def stable_payload_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def station_key(row: dict[str, str]) -> str:
    return ".".join(
        [row["network"], row["station"], row["location"]]
    )


def physical_station_id(row: dict[str, str]) -> str:
    return ".".join([row["network"], row["station"]])


def orientation_key(row: dict[str, str]) -> tuple[str, str, str, str]:
    return (
        row["event_id"],
        row["network"],
        row["station"],
        row["location"],
    )


def channel_orientations(
    rows: list[dict[str, str]],
    component_set: str,
) -> list[ChannelOrientation]:
    ordered = sorted(rows, key=lambda row: int(row["component_index"]))
    channels = component_set.split(";")
    if [row["channel"] for row in ordered] != channels:
        raise RuntimeError("Orientation channel order differs")
    return [
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
        for row in ordered
    ]


def verify_local_waveform(row: dict[str, str]) -> Path:
    path = Path(row["local_path"])
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != int(row["local_bytes"]):
        raise RuntimeError(f"Waveform byte count drifted: {path}")
    if sha256(path) != row["local_sha256"]:
        raise RuntimeError(f"Waveform SHA-256 drifted: {path}")
    return path


def output_is_valid(
    summary_path: Path,
    *,
    protocol_hash: str,
    scope: str,
    unit_key: str,
    input_identity: str,
) -> bool:
    if not summary_path.is_file():
        return False
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected = {
        "protocol_sha256": protocol_hash,
        "scope": scope,
        "unit_key": unit_key,
        "input_identity": input_identity,
    }
    for name, value in expected.items():
        if summary.get(name) != value:
            raise RuntimeError(
                f"Existing score-cache summary differs in {name}: "
                f"{summary_path}"
            )
    for name, spec in summary["artifacts"].items():
        verify_artifact(spec, name=f"existing_output.{name}")
    return True


def load_models(
    protocol: dict[str, Any],
) -> tuple[Any, Any, dict[str, Any]]:
    execution = protocol["execution"]
    backbone_spec = protocol["model_artifacts"]["backbone"]
    return load_frozen_models(
        resolve(backbone_spec["path"]),
        resolve(protocol["model_artifacts"]["artifact_head"]["path"]),
        device_name=str(execution["inference_device"]),
        torch_threads=int(execution["torch_threads"]),
        expected_backbone_sha256=str(backbone_spec["sha256"]),
    )


def score_station(
    *,
    source: Stream | list[Path],
    station: dict[str, str],
    orientation_rows: list[dict[str, str]],
    start: UTCDateTime,
    end: UTCDateTime,
    protocol: dict[str, Any],
    backbone: Any,
    artifact_head: Any,
    stored_head: dict[str, Any],
    reference_p_epoch: float | None = None,
) -> tuple[
    Any,
    dict[str, Any],
    list[dict[str, Any]],
    dict[str, Any] | None,
]:
    target = {
        "network": station["network"],
        "station": station["station"],
        "location": station["location"],
        "latitude": float(station["station_latitude"]),
        "longitude": float(station["station_longitude"]),
        "elevation_m": float(station["station_elevation_m"]),
    }
    orientations = channel_orientations(
        orientation_rows,
        station["component_set"],
    )
    wave, stream_metadata = stream_to_oriented_gappy_zne_array(
        source,
        target,
        orientations,
        start,
        end,
        target_sample_rate_hz=float(
            protocol["runtime_contract"]["sample_rate_hz"]
        ),
        samples_per_packet=50,
    )
    local_p_snr = None
    if reference_p_epoch is not None:
        sample_rate_hz = float(
            protocol["runtime_contract"]["sample_rate_hz"]
        )
        local_p_snr = local_p_excess_power_snr(
            wave,
            p_sample=int(
                round(
                    (
                        float(reference_p_epoch)
                        - float(start.timestamp)
                    )
                    * sample_rate_hz
                )
            ),
            sample_rate_hz=sample_rate_hz,
            window_seconds=float(
                protocol["event_window_contract"][
                    "local_p_snr_pre_seconds"
                ]
            ),
        )
    cache = score_waveform_packets(
        wave,
        start_epoch=float(start.timestamp),
        backbone=backbone,
        artifact_head=artifact_head,
        artifact_head_input_dim=int(stored_head["config"]["input_dim"]),
        runtime=protocol["runtime_contract"],
        device_name=str(protocol["execution"]["inference_device"]),
    )
    candidates = emit_frozen_policy_candidates(
        cache,
        start_epoch=float(start.timestamp),
        runtime=protocol["runtime_contract"],
    )
    release_device_cache(str(protocol["execution"]["inference_device"]))
    return cache, stream_metadata, candidates, local_p_snr


def event_input_identity(
    event: dict[str, str],
    protocol: dict[str, Any],
) -> str:
    return stable_payload_hash(
        {
            "event_id": event["event_id"],
            "delivery_sha256": event["delivery_sha256"],
            "raw_mseed_sha256": event["raw_mseed_sha256"],
            "event_stations_sha256": protocol["inputs"][
                "event_stations"
            ]["sha256"],
            "event_independent_station_representatives_sha256": (
                protocol["inputs"][
                    "event_independent_station_representatives"
                ]["sha256"]
            ),
            "network_geometry_protocol_sha256": protocol["inputs"][
                "network_geometry_protocol"
            ]["sha256"],
            "event_orientations_sha256": protocol["inputs"][
                "event_station_orientations"
            ]["sha256"],
        }
    )


def read_event_stream(event: dict[str, str]) -> Stream:
    archive_path = Path(event["delivery_file"])
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)
    if archive_path.stat().st_size != int(event["delivery_bytes"]):
        raise RuntimeError(f"Event archive byte count drifted: {archive_path}")
    if sha256(archive_path) != event["delivery_sha256"]:
        raise RuntimeError(f"Event archive SHA-256 drifted: {archive_path}")
    member = event["raw_mseed_member"]
    with zipfile.ZipFile(archive_path) as archive:
        info = archive.getinfo(member)
        if info.file_size != int(event["raw_mseed_bytes"]):
            raise RuntimeError(
                f"Raw miniSEED byte count drifted: {archive_path}:{member}"
            )
        payload = archive.read(member)
    payload_hash = hashlib.sha256(payload).hexdigest()
    if payload_hash != event["raw_mseed_sha256"]:
        raise RuntimeError(
            f"Raw miniSEED SHA-256 drifted: {archive_path}:{member}"
        )
    stream = read(io.BytesIO(payload))
    del payload
    return stream


def process_event(
    protocol: dict[str, Any],
    protocol_hash: str,
    event_id: str,
) -> Path:
    events = {
        row["event_id"]: row
        for row in protocol_csv(protocol, "events")
    }
    if event_id not in events:
        raise KeyError(f"Unknown frozen event: {event_id}")
    event = events[event_id]
    source_stations = [
        row
        for row in protocol_csv(protocol, "event_stations")
        if row["event_id"] == event_id
    ]
    source_by_key = {
        station_key(row): row for row in source_stations
    }
    if len(source_by_key) != len(source_stations):
        raise RuntimeError(f"Event {event_id} has duplicate instrument keys")
    representative_records = [
        row
        for row in protocol_csv(
            protocol, "event_independent_station_representatives"
        )
        if row["event_id"] == event_id
    ]
    representative_records.sort(
        key=lambda row: (
            int(row["representative_site_rank"]),
            row["representative_station_key"],
        )
    )
    stations = []
    for representative in representative_records:
        key = representative["representative_station_key"]
        if key not in source_by_key:
            raise RuntimeError(
                f"Representative instrument absent from source: "
                f"{event_id} {key}"
            )
        station = {
            **source_by_key[key],
            "physical_station_id": representative[
                "physical_station_id"
            ],
            "network_evaluable": representative["network_evaluable"],
        }
        if physical_station_id(station) != station["physical_station_id"]:
            raise RuntimeError(
                f"Representative physical station differs: {event_id} {key}"
            )
        if int(station["site_rank"]) != int(
            representative["representative_site_rank"]
        ):
            raise RuntimeError(
                f"Representative site rank differs: {event_id} {key}"
            )
        stations.append(station)
    if not 1 <= len(stations) <= 12:
        raise RuntimeError(
            f"Event {event_id} has {len(stations)} independent stations"
        )
    if len({row["physical_station_id"] for row in stations}) != len(
        stations
    ):
        raise RuntimeError(
            f"Event {event_id} repeats an independent physical station"
        )
    network_evaluable_values = {
        row["network_evaluable"] for row in stations
    }
    if len(network_evaluable_values) != 1:
        raise RuntimeError(
            f"Event {event_id} has inconsistent network evaluability"
        )
    network_evaluable = network_evaluable_values == {"True"}
    if network_evaluable != (len(stations) >= 3):
        raise RuntimeError(
            f"Event {event_id} network evaluability differs from count"
        )
    all_orientations = defaultdict(list)
    for row in protocol_csv(protocol, "event_station_orientations"):
        if row["event_id"] == event_id:
            all_orientations[orientation_key(row)].append(row)
    expected_keys = {
        (
            event_id,
            row["network"],
            row["station"],
            row["location"],
        )
        for row in stations
    }
    missing_orientation_keys = expected_keys - set(all_orientations)
    if missing_orientation_keys:
        raise RuntimeError(
            f"Event {event_id} lacks representative orientations: "
            f"{sorted(missing_orientation_keys)}"
        )

    identity = event_input_identity(event, protocol)
    output_dir = (
        resolve(protocol["execution"]["output_root"])
        / "events"
        / event_id
    )
    summary_path = output_dir / "summary.json"
    if output_is_valid(
        summary_path,
        protocol_hash=protocol_hash,
        scope="event",
        unit_key=event_id,
        input_identity=identity,
    ):
        return summary_path

    verify_event_window_contract(protocol)
    started = time.time()
    origin = UTCDateTime(event["origin_time_utc"])
    start = origin - EVENT_WINDOW_BEFORE_SECONDS
    end = origin + EVENT_WINDOW_AFTER_SECONDS
    raw = read_event_stream(event)
    backbone, artifact_head, stored_head = load_models(protocol)
    station_ids = []
    physical_station_ids = []
    p_rows = []
    event_rows = []
    age_rows = []
    artifact_rows = []
    valid_rows = []
    allowed_rows = []
    station_summaries = []
    candidate_rows = []
    tolerance = float(
        protocol["label_contract"][
            "station_candidate_match_tolerance_seconds"
        ]
    )
    for station in stations:
        key = (
            event_id,
            station["network"],
            station["station"],
            station["location"],
        )
        selected = raw.select(
            network=station["network"],
            station=station["station"],
            location=station["location"],
        )
        reference_p = float(
            UTCDateTime(station["p_arrival_time_utc"]).timestamp
        )
        (
            cache,
            stream_metadata,
            candidates,
            local_p_snr,
        ) = score_station(
            source=selected,
            station=station,
            orientation_rows=all_orientations[key],
            start=start,
            end=end,
            protocol=protocol,
            backbone=backbone,
            artifact_head=artifact_head,
            stored_head=stored_head,
            reference_p_epoch=reference_p,
        )
        if local_p_snr is None:
            raise RuntimeError(f"Event SNR was not computed: {event_id}")
        site = station_key(station)
        station_ids.append(site)
        physical_station_ids.append(station["physical_station_id"])
        p_rows.append(cache.p_probability)
        event_rows.append(cache.event_probability)
        age_rows.append(cache.arrival_age_fraction)
        artifact_rows.append(cache.artifact_probability)
        valid_rows.append(cache.quality_valid)
        allowed_rows.append(cache.candidate_allowed)
        matched = 0
        for serial, candidate in enumerate(candidates, start=1):
            pick_match = (
                abs(float(candidate["estimated_p_epoch"]) - reference_p)
                <= tolerance
            )
            matched += int(pick_match)
            candidate_rows.append(
                {
                    "record_key": (
                        f"event:{event_id}:{site}:{serial:06d}"
                    ),
                    "scope": "event",
                    "event_id": event_id,
                    "cohort": event["cohort"],
                    "date_utc": str(event["origin_time_utc"])[:10],
                    "network": station["network"],
                    "station": station["station"],
                    "location": station["location"],
                    "station_key": site,
                    "physical_station_id": station[
                        "physical_station_id"
                    ],
                    **candidate,
                    "label_source": station["label_source"],
                    "reference_p_epoch": reference_p,
                    "pick_match": pick_match,
                    "strict_false_candidate": not pick_match,
                    "latency_from_reference_p_seconds": (
                        float(candidate["available_epoch"]) - reference_p
                    ),
                    "latency_from_origin_seconds": (
                        float(candidate["available_epoch"])
                        - float(origin.timestamp)
                    ),
                }
            )
        station_summaries.append(
            {
                "station_key": site,
                "physical_station_id": station[
                    "physical_station_id"
                ],
                "site_rank": int(station["site_rank"]),
                "component_set": station["component_set"],
                "label_source": station["label_source"],
                "reference_p_epoch": reference_p,
                "candidate_count": len(candidates),
                "pick_matched_candidates": matched,
                "local_p_snr": local_p_snr,
                "diagnostics": cache.diagnostics,
                "stream_metadata": stream_metadata,
            }
        )

    score_path = output_dir / "scores.npz"
    candidate_path = output_dir / "station_candidates.csv"
    write_npz_atomic(
        score_path,
        station_keys=np.asarray(station_ids, dtype=np.str_),
        physical_station_ids=np.asarray(
            physical_station_ids, dtype=np.str_
        ),
        p_probability=np.stack(p_rows),
        event_probability=np.stack(event_rows),
        arrival_age_fraction=np.stack(age_rows),
        artifact_probability=np.stack(artifact_rows),
        quality_valid=np.stack(valid_rows),
        candidate_allowed=np.stack(allowed_rows),
    )
    write_csv_atomic(candidate_path, candidate_rows, CANDIDATE_FIELDS)
    summary = {
        "protocol": PROTOCOL_ID,
        "protocol_sha256": protocol_hash,
        "scope": "event",
        "unit_key": event_id,
        "input_identity": identity,
        "event_id": event_id,
        "cohort": event["cohort"],
        "origin_time_utc": event["origin_time_utc"],
        "magnitude": float(event["magnitude"]),
        "source_instrument_count": len(source_stations),
        "independent_physical_station_count": len(stations),
        "network_evaluable": network_evaluable,
        "candidate_count": len(candidate_rows),
        "pick_matched_candidates": sum(
            not bool(row["strict_false_candidate"])
            for row in candidate_rows
        ),
        "station_summaries": station_summaries,
        "worker_pid": os.getpid(),
        "worker_max_resident_set_bytes": int(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        ),
        "runtime_seconds": time.time() - started,
        "artifacts": {
            "scores": artifact(score_path),
            "station_candidates": artifact(candidate_path),
        },
    }
    write_json_atomic(summary_path, summary)
    return summary_path


def background_unit_parts(
    unit_key: str,
) -> tuple[str, str, str, str]:
    parts = unit_key.split("|")
    if len(parts) != 4:
        raise ValueError(
            "Background unit key must be DATE|NETWORK|STATION|LOCATION"
        )
    return parts[0], parts[1], parts[2], parts[3]


def process_background(
    protocol: dict[str, Any],
    protocol_hash: str,
    unit_key: str,
) -> Path:
    date_utc, network, station_name, location = background_unit_parts(
        unit_key
    )
    station_days = [
        row
        for row in protocol_csv(protocol, "background_station_days")
        if (
            row["date_utc"],
            row["network"],
            row["station"],
            row["location"],
        )
        == (date_utc, network, station_name, location)
    ]
    if len(station_days) != 1:
        raise RuntimeError(
            f"Expected one frozen background station-day: {unit_key}"
        )
    station_day = station_days[0]
    selected = [
        row
        for row in protocol_csv(protocol, "selected_background_stations")
        if (
            row["network"],
            row["station"],
            row["location"],
        )
        == (network, station_name, location)
    ]
    if len(selected) != 1:
        raise RuntimeError(
            f"Expected one selected background station: {unit_key}"
        )
    station = {
        **station_day,
        "station_latitude": selected[0]["latitude"],
        "station_longitude": selected[0]["longitude"],
        "station_elevation_m": selected[0]["elevation_m"],
    }
    objects = [
        row
        for row in protocol_csv(protocol, "background_objects")
        if (
            row["date_utc"],
            row["network"],
            row["station"],
            row["location"],
        )
        == (date_utc, network, station_name, location)
    ]
    by_channel = {row["channel"]: row for row in objects}
    channels = station["component_set"].split(";")
    if set(by_channel) != set(channels) or len(objects) != 3:
        raise RuntimeError(f"Background object triplet differs: {unit_key}")
    orientation_rows = [
        row
        for row in protocol_csv(
            protocol, "background_station_orientations"
        )
        if (
            row["network"],
            row["station"],
            row["location"],
        )
        == (network, station_name, location)
    ]
    paths = [verify_local_waveform(by_channel[channel]) for channel in channels]
    identity = stable_payload_hash(
        {
            "unit_key": unit_key,
            "objects": [
                {
                    "channel": row["channel"],
                    "local_sha256": row["local_sha256"],
                    "source_version_id": row["source_version_id"],
                }
                for row in (by_channel[channel] for channel in channels)
            ],
            "background_station_days_sha256": protocol["inputs"][
                "background_station_days"
            ]["sha256"],
            "background_orientations_sha256": protocol["inputs"][
                "background_station_orientations"
            ]["sha256"],
        }
    )
    output_dir = (
        resolve(protocol["execution"]["output_root"])
        / "background"
        / date_utc
        / station_key(station)
    )
    summary_path = output_dir / "summary.json"
    if output_is_valid(
        summary_path,
        protocol_hash=protocol_hash,
        scope="background",
        unit_key=unit_key,
        input_identity=identity,
    ):
        return summary_path

    started = time.time()
    start = UTCDateTime(f"{date_utc}T00:00:00Z")
    end = start + 86400.0
    backbone, artifact_head, stored_head = load_models(protocol)
    cache, stream_metadata, candidates, local_p_snr = score_station(
        source=paths,
        station=station,
        orientation_rows=orientation_rows,
        start=start,
        end=end,
        protocol=protocol,
        backbone=backbone,
        artifact_head=artifact_head,
        stored_head=stored_head,
    )
    if local_p_snr is not None:
        raise RuntimeError(f"Background SNR was unexpectedly computed: {unit_key}")
    site = station_key(station)
    candidate_rows = []
    for serial, candidate in enumerate(candidates, start=1):
        candidate_rows.append(
            {
                "record_key": (
                    f"background:{date_utc}:{site}:{serial:06d}"
                ),
                "scope": "background",
                "event_id": "",
                "cohort": "catalog_quiet_background",
                "date_utc": date_utc,
                "network": network,
                "station": station_name,
                "location": location,
                "station_key": site,
                "physical_station_id": ".".join(
                    [network, station_name]
                ),
                **candidate,
                "label_source": "catalog_quiet_day",
                "reference_p_epoch": "",
                "pick_match": False,
                "strict_false_candidate": True,
                "latency_from_reference_p_seconds": "",
                "latency_from_origin_seconds": "",
            }
        )

    score_path = output_dir / "scores.npz"
    candidate_path = output_dir / "station_candidates.csv"
    write_npz_atomic(score_path, **cache.arrays())
    write_csv_atomic(candidate_path, candidate_rows, CANDIDATE_FIELDS)
    exposure_days = (
        float(cache.candidate_allowed.sum())
        * float(protocol["runtime_contract"]["packet_duration_seconds"])
        / 86400.0
    )
    summary = {
        "protocol": PROTOCOL_ID,
        "protocol_sha256": protocol_hash,
        "scope": "background",
        "unit_key": unit_key,
        "input_identity": identity,
        "date_utc": date_utc,
        "station_key": site,
        "component_set": station["component_set"],
        "source_common_availability": float(
            station["common_availability"]
        ),
        "candidate_exposure_days": exposure_days,
        "candidate_count": len(candidate_rows),
        "strict_false_candidates": len(candidate_rows),
        "diagnostics": cache.diagnostics,
        "stream_metadata": stream_metadata,
        "worker_pid": os.getpid(),
        "worker_max_resident_set_bytes": int(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        ),
        "runtime_seconds": time.time() - started,
        "artifacts": {
            "scores": artifact(score_path),
            "station_candidates": artifact(candidate_path),
        },
    }
    write_json_atomic(summary_path, summary)
    return summary_path


def main() -> None:
    args = parse_args()
    protocol, protocol_hash = load_protocol(args.protocol)
    if args.scope == "event":
        summary_path = process_event(
            protocol, protocol_hash, args.unit_key
        )
    else:
        summary_path = process_background(
            protocol, protocol_hash, args.unit_key
        )
    print(summary_path, flush=True)


if __name__ == "__main__":
    main()
