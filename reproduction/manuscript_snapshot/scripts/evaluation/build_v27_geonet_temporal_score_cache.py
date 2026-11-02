#!/usr/bin/env python3
"""Build resumable C0/C1 GeoNet temporal-development score caches."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import UTC, datetime
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import sys
import time
from typing import Any, Iterable

import numpy as np
from obspy import Stream, UTCDateTime


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

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


DEFAULT_PROTOCOL = (
    ROOT
    / "experiments/"
    "cspp_v27_geonet_temporal_evaluation_protocol_v3_20260729.json"
)
EVENT_BEFORE_SECONDS = 60.0
EVENT_AFTER_SECONDS = 120.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument(
        "--scope", choices=("event", "background"), required=True
    )
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process at most this many event/date units; zero means all.",
    )
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument(
        "--finalize-only",
        action="store_true",
        help="Verify all unit summaries and write only scope completion.",
    )
    return parser.parse_args()


def artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def verify_artifact(spec: dict[str, Any]) -> Path:
    path = Path(spec["path"])
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != int(spec["bytes"]):
        raise RuntimeError(f"Artifact byte count drifted: {path}")
    if sha256_file(path) != str(spec["sha256"]):
        raise RuntimeError(f"Artifact SHA-256 drifted: {path}")
    return path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def stable_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def maximum_resident_set_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())


def write_npz_atomic(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp-{os.getpid()}.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def load_protocol(
    path: Path, expected_sha256: str
) -> tuple[dict[str, Any], str]:
    protocol_path = path.resolve()
    actual = sha256_file(protocol_path)
    if actual != expected_sha256:
        raise RuntimeError(
            f"Evaluation protocol SHA-256 differs: {actual}"
        )
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("protocol_id") != (
        "cspp_v27_geonet_temporal_evaluation_execution_v3"
    ):
        raise RuntimeError("Unexpected V27 evaluation protocol")
    for relative, expected in protocol["code_sha256"].items():
        source = ROOT / relative
        if not source.is_file() or sha256_file(source) != expected:
            raise RuntimeError(f"Evaluation code drifted: {relative}")
    for spec in protocol["inputs"].values():
        verify_artifact(spec)
    for spec in protocol["models"].values():
        verify_artifact(spec)
    return protocol, actual


def orientation_groups(
    rows: list[dict[str, str]],
) -> dict[str, list[dict[str, str]]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row["group_id"]].append(row)
    for group_id, values in groups.items():
        values.sort(key=lambda row: int(row["component_index"]))
        if len(values) != 3:
            raise RuntimeError(
                f"Orientation group does not have 3 rows: {group_id}"
            )
    return groups


def channel_orientations(
    rows: list[dict[str, str]],
    component_set: str,
) -> list[ChannelOrientation]:
    channels = component_set.split(";")
    if [row["channel"] for row in rows] != channels:
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
        for row in rows
    ]


def station_key(row: dict[str, str]) -> str:
    return ".".join(
        (row["network"], row["station"], row["location"])
    )


def station_target(station: dict[str, str]) -> dict[str, Any]:
    target: dict[str, Any] = {
        "network": station["network"],
        "station": station["station"],
        "location": station["location"],
    }
    aliases = {
        "latitude": ("station_latitude", "latitude"),
        "longitude": ("station_longitude", "longitude"),
        "elevation_m": ("station_elevation_m", "elevation_m"),
    }
    for output_name, input_names in aliases.items():
        raw = next(
            (
                station[name]
                for name in input_names
                if station.get(name) not in (None, "")
            ),
            None,
        )
        if raw is None:
            continue
        value = float(raw)
        if not math.isfinite(value):
            raise RuntimeError(
                f"Non-finite optional station coordinate {output_name}"
            )
        target[output_name] = value
    return target


def existing_unit_valid(
    summary_path: Path,
    *,
    protocol_sha256: str,
    scope: str,
    unit_key: str,
    input_identity: str,
) -> bool:
    if not summary_path.is_file():
        return False
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected = {
        "protocol_sha256": protocol_sha256,
        "scope": scope,
        "unit_key": unit_key,
        "input_identity": input_identity,
        "status": "complete",
    }
    for name, value in expected.items():
        if summary.get(name) != value:
            raise RuntimeError(
                f"Existing unit summary differs in {name}: {summary_path}"
            )
    for spec in summary["artifacts"].values():
        verify_artifact(spec)
    return True


def score_station(
    *,
    source: Stream | list[Path],
    station: dict[str, str],
    orientation_rows: list[dict[str, str]],
    start: UTCDateTime,
    end: UTCDateTime,
    ensemble: Any,
    protocol: dict[str, Any],
    reference_p_epoch: float | None,
) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, Any] | None]:
    target = station_target(station)
    orientations = channel_orientations(
        orientation_rows, station["component_set"]
    )
    wave, stream_metadata = stream_to_oriented_gappy_zne_array(
        source,
        target,
        orientations,
        start,
        end,
        target_sample_rate_hz=float(
            protocol["runtime"]["sample_rate_hz"]
        ),
        samples_per_packet=int(
            protocol["runtime"]["packet_samples"]
        ),
    )
    local_snr = None
    if reference_p_epoch is not None:
        sample_rate = float(protocol["runtime"]["sample_rate_hz"])
        local_snr = local_p_excess_power_snr(
            wave,
            p_sample=int(
                round(
                    (float(reference_p_epoch) - float(start.timestamp))
                    * sample_rate
                )
            ),
            sample_rate_hz=sample_rate,
            window_seconds=float(
                protocol["metrics"]["local_p_snr_window_seconds"]
            ),
        )
    cache = score_waveform_packets(
        wave,
        start_epoch=float(start.timestamp),
        ensemble=ensemble,
        runtime=protocol["runtime"],
    )
    metadata = {
        "stream": stream_metadata,
        "score": cache.diagnostics,
        "start_epoch": float(start.timestamp),
        "end_epoch": float(end.timestamp),
        "packet_count": int(len(cache.candidate_probability)),
    }
    del wave
    return cache.arrays(), metadata, local_snr


def checkpoint_specs(protocol: dict[str, Any]) -> dict[str, dict[str, str]]:
    return {
        key: {
            "path": spec["path"],
            "sha256": spec["sha256"],
            "verifier_feature_mode": spec["verifier_feature_mode"],
        }
        for key, spec in protocol["models"].items()
    }


def event_units(
    protocol: dict[str, Any],
    protocol_sha256: str,
    ensemble: Any,
    limit: int,
    shard_count: int,
    shard_index: int,
) -> None:
    events = read_csv(
        verify_artifact(protocol["inputs"]["event_rows"])
    )
    stations = read_csv(
        verify_artifact(protocol["inputs"]["event_station_rows"])
    )
    waveforms = read_csv(
        verify_artifact(protocol["inputs"]["event_waveform_files"])
    )
    orientations = orientation_groups(
        read_csv(
            verify_artifact(
                protocol["inputs"]["event_station_orientations"]
            )
        )
    )
    stations_by_event: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in stations:
        stations_by_event[row["event_id"]].append(row)
    for rows in stations_by_event.values():
        rows.sort(key=lambda row: int(row["site_rank"]))
    waveform_by_event = {row["event_id"]: row for row in waveforms}
    events.sort(key=lambda row: (row["origin_time_utc"], row["event_id"]))

    output_root = Path(protocol["execution"]["output_root"])
    status_path = output_root / "event_status.jsonl"
    processed = 0
    for index, event in enumerate(events, start=1):
        if (index - 1) % shard_count != shard_index:
            continue
        if limit and processed >= limit:
            break
        event_id = event["event_id"]
        waveform = waveform_by_event[event_id]
        input_identity = stable_hash(
            {
                "event": event,
                "stations": stations_by_event[event_id],
                "waveform_sha256": waveform["mseed_sha256"],
                "event_orientation_sha256": protocol["inputs"][
                    "event_station_orientations"
                ]["sha256"],
            }
        )
        unit_root = output_root / "event" / event_id
        summary_path = unit_root / "summary.json"
        if existing_unit_valid(
            summary_path,
            protocol_sha256=protocol_sha256,
            scope="event",
            unit_key=event_id,
            input_identity=input_identity,
        ):
            print(f"event {index}/{len(events)} {event_id}: cached", flush=True)
            continue
        append_jsonl(
            status_path,
            {
                "time_utc": datetime.now(UTC).isoformat(),
                "event": "unit_started",
                "scope": "event",
                "unit_key": event_id,
            },
        )
        started = time.perf_counter()
        source_path = Path(waveform["mseed_file"])
        if (
            not source_path.is_file()
            or source_path.stat().st_size != int(waveform["mseed_bytes"])
            or sha256_file(source_path) != waveform["mseed_sha256"]
        ):
            raise RuntimeError(f"Event waveform drifted: {source_path}")
        source_result = read_resilient_mseed(source_path)
        source = source_result.stream
        origin = UTCDateTime(event["origin_time_utc"])
        start = origin - EVENT_BEFORE_SECONDS
        end = origin + EVENT_AFTER_SECONDS
        station_results = []
        artifacts = {}
        for station in stations_by_event[event_id]:
            key = station_key(station)
            group_id = f"{event_id}.{key}"
            arrays, metadata, local_snr = score_station(
                source=source,
                station=station,
                orientation_rows=orientations[group_id],
                start=start,
                end=end,
                ensemble=ensemble,
                protocol=protocol,
                reference_p_epoch=float(
                    UTCDateTime(station["p_arrival_time_utc"]).timestamp
                ),
            )
            cache_path = unit_root / f"{key}.npz"
            write_npz_atomic(cache_path, arrays)
            artifacts[key] = artifact(cache_path)
            station_results.append(
                {
                    "station_key": key,
                    "site_rank": int(station["site_rank"]),
                    "development_role": station["development_role"],
                    "label_source": station["label_source"],
                    "pick_time_weight": (
                        float(station["pick_time_weight"])
                        if station["pick_time_weight"]
                        else None
                    ),
                    "reference_p_epoch": float(
                        UTCDateTime(
                            station["p_arrival_time_utc"]
                        ).timestamp
                    ),
                    "epicentral_distance_km": float(
                        station["epicentral_distance_km"]
                    ),
                    "magnitude": float(station["magnitude"]),
                    "local_p_snr": local_snr,
                    "metadata": metadata,
                    "score_cache": artifacts[key],
                }
            )
            release_device_cache(ensemble.device_name)
        del source
        summary = {
            "protocol_sha256": protocol_sha256,
            "scope": "event",
            "unit_key": event_id,
            "input_identity": input_identity,
            "status": "complete",
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "development_role": event["development_role"],
            "origin_time_utc": event["origin_time_utc"],
            "magnitude": float(event["magnitude"]),
            "station_count": len(station_results),
            "station_results": station_results,
            "event_waveform_decode": source_result.diagnostics,
            "shared_core_digest": ensemble.shared_core_digest,
            "elapsed_seconds": time.perf_counter() - started,
            "maximum_resident_set_bytes": maximum_resident_set_bytes(),
            "artifacts": artifacts,
        }
        write_json_atomic(summary_path, summary)
        append_jsonl(
            status_path,
            {
                "time_utc": datetime.now(UTC).isoformat(),
                "event": "unit_completed",
                "scope": "event",
                "unit_key": event_id,
                "summary": artifact(summary_path),
            },
        )
        processed += 1
        print(
            f"event {index}/{len(events)} {event_id}: "
            f"{len(station_results)} stations, "
            f"{summary['elapsed_seconds']:.2f}s",
            flush=True,
        )


def background_units(
    protocol: dict[str, Any],
    protocol_sha256: str,
    ensemble: Any,
    limit: int,
    shard_count: int,
    shard_index: int,
) -> None:
    dates = read_csv(
        verify_artifact(protocol["inputs"]["background_dates"])
    )
    stations = read_csv(
        verify_artifact(protocol["inputs"]["background_station_days"])
    )
    objects = read_csv(
        verify_artifact(protocol["inputs"]["background_objects"])
    )
    orientations = orientation_groups(
        read_csv(
            verify_artifact(
                protocol["inputs"]["background_station_orientations"]
            )
        )
    )
    stations_by_date: dict[str, list[dict[str, str]]] = defaultdict(list)
    objects_by_station: dict[
        tuple[str, str], list[dict[str, str]]
    ] = defaultdict(list)
    for row in stations:
        stations_by_date[row["date_utc"]].append(row)
    for row in objects:
        objects_by_station[
            (
                row["date_utc"],
                station_key(row),
            )
        ].append(row)
    for rows in stations_by_date.values():
        rows.sort(key=station_key)
    for rows in objects_by_station.values():
        rows.sort(key=lambda row: row["channel"])
    dates.sort(key=lambda row: int(row["quiet_rank"]))

    output_root = Path(protocol["execution"]["output_root"])
    status_path = output_root / "background_status.jsonl"
    processed = 0
    for index, date in enumerate(dates, start=1):
        if (index - 1) % shard_count != shard_index:
            continue
        if limit and processed >= limit:
            break
        date_utc = date["date_utc"]
        input_objects = []
        for station in stations_by_date[date_utc]:
            input_objects.extend(
                {
                    "station_key": station_key(station),
                    "channel": row["channel"],
                    "local_sha256": row["local_sha256"],
                    "source_version_id": row["source_version_id"],
                }
                for row in objects_by_station[
                    (date_utc, station_key(station))
                ]
            )
        input_identity = stable_hash(
            {
                "date": date,
                "stations": stations_by_date[date_utc],
                "objects": input_objects,
                "background_orientation_sha256": protocol["inputs"][
                    "background_station_orientations"
                ]["sha256"],
            }
        )
        unit_root = output_root / "background" / date_utc
        summary_path = unit_root / "summary.json"
        if existing_unit_valid(
            summary_path,
            protocol_sha256=protocol_sha256,
            scope="background",
            unit_key=date_utc,
            input_identity=input_identity,
        ):
            print(
                f"background {index}/{len(dates)} {date_utc}: cached",
                flush=True,
            )
            continue
        append_jsonl(
            status_path,
            {
                "time_utc": datetime.now(UTC).isoformat(),
                "event": "unit_started",
                "scope": "background",
                "unit_key": date_utc,
            },
        )
        started = time.perf_counter()
        start = UTCDateTime(f"{date_utc}T00:00:00Z")
        end = start + 86400.0
        station_results = []
        artifacts = {}
        for station in stations_by_date[date_utc]:
            key = station_key(station)
            source_rows = objects_by_station[(date_utc, key)]
            if len(source_rows) != 3:
                raise RuntimeError(
                    f"Expected three source objects: {date_utc} {key}"
                )
            source_paths = []
            for row in source_rows:
                path = Path(row["local_path"])
                if (
                    not path.is_file()
                    or path.stat().st_size != int(row["local_bytes"])
                    or sha256_file(path) != row["local_sha256"]
                ):
                    raise RuntimeError(
                        f"Background waveform drifted: {path}"
                    )
                source_paths.append(path)
            arrays, metadata, _ = score_station(
                source=source_paths,
                station=station,
                orientation_rows=orientations[key],
                start=start,
                end=end,
                ensemble=ensemble,
                protocol=protocol,
                reference_p_epoch=None,
            )
            cache_path = unit_root / f"{key}.npz"
            write_npz_atomic(cache_path, arrays)
            artifacts[key] = artifact(cache_path)
            station_results.append(
                {
                    "station_key": key,
                    "development_role": station["development_role"],
                    "common_availability": float(
                        station["common_availability"]
                    ),
                    "metadata": metadata,
                    "score_cache": artifacts[key],
                    "source_objects": [
                        {
                            "channel": row["channel"],
                            "local_sha256": row["local_sha256"],
                            "source_version_id": row["source_version_id"],
                        }
                        for row in source_rows
                    ],
                }
            )
            release_device_cache(ensemble.device_name)
        summary = {
            "protocol_sha256": protocol_sha256,
            "scope": "background",
            "unit_key": date_utc,
            "input_identity": input_identity,
            "status": "complete",
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "quiet_rank": int(date["quiet_rank"]),
            "development_role": date["development_role"],
            "station_count": len(station_results),
            "station_results": station_results,
            "shared_core_digest": ensemble.shared_core_digest,
            "elapsed_seconds": time.perf_counter() - started,
            "maximum_resident_set_bytes": maximum_resident_set_bytes(),
            "artifacts": artifacts,
        }
        write_json_atomic(summary_path, summary)
        append_jsonl(
            status_path,
            {
                "time_utc": datetime.now(UTC).isoformat(),
                "event": "unit_completed",
                "scope": "background",
                "unit_key": date_utc,
                "summary": artifact(summary_path),
            },
        )
        processed += 1
        print(
            f"background {index}/{len(dates)} {date_utc}: "
            f"{len(station_results)} stations, "
            f"{summary['elapsed_seconds']:.2f}s",
            flush=True,
        )


def write_scope_completion(
    protocol: dict[str, Any],
    protocol_sha256: str,
    scope: str,
) -> None:
    output_root = Path(protocol["execution"]["output_root"])
    input_name = "event_rows" if scope == "event" else "background_dates"
    expected_rows = read_csv(
        verify_artifact(protocol["inputs"][input_name])
    )
    unit_root = output_root / scope
    summaries = sorted(unit_root.glob("*/summary.json"))
    if len(summaries) != len(expected_rows):
        return
    unit_specs = []
    for path in summaries:
        summary = json.loads(path.read_text(encoding="utf-8"))
        if (
            summary.get("protocol_sha256") != protocol_sha256
            or summary.get("scope") != scope
            or summary.get("status") != "complete"
        ):
            raise RuntimeError(f"Scope summary differs: {path}")
        for spec in summary["artifacts"].values():
            verify_artifact(spec)
        unit_specs.append(artifact(path))
    completion = {
        "completion_id": (
            f"cspp_v27_geonet_temporal_{scope}_score_cache_v1"
        ),
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "protocol_sha256": protocol_sha256,
        "scope": scope,
        "units": len(unit_specs),
        "inference_device": protocol["execution"]["inference_device"],
        "inference_processes": int(
            protocol["execution"]["inference_processes"]
        ),
        "single_process_mps": (
            protocol["execution"]["inference_device"] == "mps"
            and int(protocol["execution"]["inference_processes"]) == 1
        ),
        "data_loader_workers": 0,
        "shared_core_digest": protocol["shared_core_integrity"][
            "tensor_digest"
        ],
        "unit_summaries": unit_specs,
    }
    write_json_atomic(
        output_root / f"{scope}_score_cache_complete.json",
        completion,
    )


def main() -> None:
    args = parse_args()
    if args.shard_count < 1:
        raise ValueError("--shard-count must be positive")
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("--shard-index is outside --shard-count")
    if args.limit and args.shard_count != 1:
        raise ValueError("--limit cannot be combined with sharding")
    protocol, protocol_sha256 = load_protocol(
        args.protocol, args.expected_protocol_sha256
    )
    if args.device != protocol["execution"]["inference_device"]:
        raise RuntimeError("Requested device differs from frozen protocol")
    output_root = Path(protocol["execution"]["output_root"]).resolve()
    if not output_root.is_relative_to(Path("external_volume")):
        raise RuntimeError("Score caches must stay on zhouhy")
    if str(output_root).startswith("external_volume"):
        raise RuntimeError("WD_BLACK access is prohibited")
    output_root.mkdir(parents=True, exist_ok=True)
    if args.finalize_only:
        if args.limit or args.shard_count != 1:
            raise ValueError("--finalize-only does not accept limit/sharding")
        write_scope_completion(protocol, protocol_sha256, args.scope)
        completion = output_root / f"{args.scope}_score_cache_complete.json"
        if not completion.is_file():
            raise RuntimeError(f"Scope is not complete: {args.scope}")
        return
    ensemble = load_ensemble(
        checkpoint_specs(protocol),
        device_name=args.device,
        torch_threads=int(protocol["execution"]["torch_threads"]),
    )
    expected_digest = protocol["shared_core_integrity"][
        "tensor_digest"
    ]
    if ensemble.shared_core_digest != expected_digest:
        raise RuntimeError("Runtime shared-core digest differs")
    if args.scope == "event":
        event_units(
            protocol,
            protocol_sha256,
            ensemble,
            int(args.limit),
            int(args.shard_count),
            int(args.shard_index),
        )
    else:
        background_units(
            protocol,
            protocol_sha256,
            ensemble,
            int(args.limit),
            int(args.shard_count),
            int(args.shard_index),
        )
    if not args.limit and args.shard_count == 1:
        write_scope_completion(protocol, protocol_sha256, args.scope)
    release_device_cache(args.device)


if __name__ == "__main__":
    main()
