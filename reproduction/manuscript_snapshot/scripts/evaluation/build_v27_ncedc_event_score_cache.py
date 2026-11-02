#!/usr/bin/env python3
"""Score frozen NCSN event records with the locked CSPP ensemble."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time
from typing import Any

import numpy as np
from obspy import UTCDateTime


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
    / "experiments/cspp_v27_ncedc_event_scoring_protocol_20260817.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--limit-events", type=int, default=0)
    return parser.parse_args()


def artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def verify_artifact(spec: dict[str, Any]) -> Path:
    path = Path(spec["path"])
    if (
        not path.is_file()
        or path.stat().st_size != int(spec["bytes"])
        or sha256_file(path) != str(spec["sha256"])
    ):
        raise RuntimeError(f"Artifact drifted: {path}")
    return path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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


def write_npz_atomic(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp-{os.getpid()}.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def stable_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def maximum_resident_set_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def orientation_groups(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row["group_id"]].append(row)
    for group_id, values in groups.items():
        values.sort(key=lambda row: int(row["component_index"]))
        if len(values) != 3:
            raise RuntimeError(f"Orientation group is incomplete: {group_id}")
    return groups


def load_protocol(path: Path, expected_sha256: str) -> tuple[dict[str, Any], str]:
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise RuntimeError(f"Scoring protocol SHA-256 differs: {actual}")
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("protocol_id") != "cspp_v27_ncedc_event_scoring_v1":
        raise RuntimeError("Unexpected NCSN scoring protocol")
    for relative, expected in protocol["code_sha256"].items():
        source = ROOT / relative
        if not source.is_file() or sha256_file(source) != expected:
            raise RuntimeError(f"Scoring code drifted: {relative}")
    for spec in protocol["inputs"].values():
        verify_artifact(spec)
    for spec in protocol["models"].values():
        verify_artifact(spec)
    return protocol, actual


def station_key(row: dict[str, str]) -> str:
    return ".".join((row["network"], row["station"], row["location"]))


def score_station(
    *,
    source: Any,
    station: dict[str, str],
    orientation_rows: list[dict[str, str]],
    ensemble: Any,
    protocol: dict[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, Any] | None]:
    start = UTCDateTime(station["request_start_utc"])
    end = UTCDateTime(station["request_end_utc"])
    orientations = [
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
        for row in orientation_rows
    ]
    wave, stream_metadata = stream_to_oriented_gappy_zne_array(
        source,
        {
            "network": station["network"],
            "station": station["station"],
            "location": station["location"],
            "latitude": float(station["station_latitude"]),
            "longitude": float(station["station_longitude"]),
        },
        orientations,
        start,
        end,
        target_sample_rate_hz=float(protocol["runtime"]["sample_rate_hz"]),
        samples_per_packet=int(protocol["runtime"]["packet_samples"]),
    )
    sample_rate = float(protocol["runtime"]["sample_rate_hz"])
    reference_p_epoch = float(UTCDateTime(station["primary_pick_time_utc"]).timestamp)
    local_snr = local_p_excess_power_snr(
        wave,
        p_sample=int(round((reference_p_epoch - float(start.timestamp)) * sample_rate)),
        sample_rate_hz=sample_rate,
        window_seconds=float(protocol["metrics"]["local_p_snr_window_seconds"]),
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
    return cache.arrays(), metadata, local_snr


def main() -> None:
    args = parse_args()
    protocol, protocol_sha256 = load_protocol(
        args.protocol.resolve(), args.expected_protocol_sha256
    )
    output_root = Path(protocol["execution"]["output_root"])
    requests = read_csv(verify_artifact(protocol["inputs"]["event_station_requests"]))
    acquisition = read_csv(verify_artifact(protocol["inputs"]["acquisition_status"]))
    orientations = orientation_groups(
        read_csv(verify_artifact(protocol["inputs"]["event_station_orientations"]))
    )
    acquisition_by_event = {row["event_id"]: row for row in acquisition}
    stations_by_event: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in requests:
        stations_by_event[row["event_id"]].append(row)
    for rows in stations_by_event.values():
        rows.sort(key=lambda row: int(row["station_rank"]))
    events = sorted(
        stations_by_event,
        key=lambda event_id: (stations_by_event[event_id][0]["origin_time_utc"], event_id),
    )
    if args.limit_events:
        events = events[: args.limit_events]

    ensemble = load_ensemble(
        {
            key: {
                "path": spec["path"],
                "sha256": spec["sha256"],
                "verifier_feature_mode": spec["verifier_feature_mode"],
            }
            for key, spec in protocol["models"].items()
        },
        device_name=args.device,
        torch_threads=int(protocol["execution"]["torch_threads"]),
    )

    unit_summaries: list[dict[str, Any]] = []
    scored_records = 0
    skipped_incomplete = 0
    for event_index, event_id in enumerate(events, start=1):
        unit_root = output_root / "event" / event_id
        summary_path = unit_root / "summary.json"
        acquisition_row = acquisition_by_event[event_id]
        source_path = Path(acquisition_row["mseed_file"])
        if (
            not source_path.is_file()
            or source_path.stat().st_size != int(acquisition_row["mseed_bytes"])
            or sha256_file(source_path) != acquisition_row["mseed_sha256"]
        ):
            raise RuntimeError(f"Acquired waveform drifted: {source_path}")
        qc_path = Path(acquisition_row["qc_file"])
        qc = json.loads(qc_path.read_text(encoding="utf-8"))
        complete_keys = {
            (row["network"], row["station"], row["location"])
            for row in qc["station_qc"]
            if row["complete"]
        }
        input_identity = stable_hash(
            {
                "event_id": event_id,
                "requests": stations_by_event[event_id],
                "mseed_sha256": acquisition_row["mseed_sha256"],
                "qc_sha256": sha256_file(qc_path),
                "orientation_sha256": protocol["inputs"][
                    "event_station_orientations"
                ]["sha256"],
            }
        )
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if (
                summary.get("protocol_sha256") != protocol_sha256
                or summary.get("input_identity") != input_identity
                or summary.get("status") != "complete"
            ):
                raise RuntimeError(f"Existing score summary differs: {summary_path}")
            for spec in summary["artifacts"].values():
                verify_artifact(spec)
            unit_summaries.append(artifact(summary_path))
            scored_records += int(summary["scored_station_count"])
            skipped_incomplete += int(summary["excluded_incomplete_station_count"])
            print(f"[{event_index}/{len(events)}] {event_id}: cached", flush=True)
            continue

        started = time.perf_counter()
        source_result = read_resilient_mseed(source_path)
        station_results: list[dict[str, Any]] = []
        artifacts: dict[str, dict[str, Any]] = {}
        for station in stations_by_event[event_id]:
            identity = (station["network"], station["station"], station["location"])
            if identity not in complete_keys:
                skipped_incomplete += 1
                continue
            key = station_key(station)
            group_id = ".".join((event_id, key))
            arrays, metadata, local_snr = score_station(
                source=source_result.stream,
                station=station,
                orientation_rows=orientations[group_id],
                ensemble=ensemble,
                protocol=protocol,
            )
            cache_path = unit_root / f"{key}.npz"
            write_npz_atomic(cache_path, arrays)
            artifacts[key] = artifact(cache_path)
            station_results.append(
                {
                    "station_key": key,
                    "station_rank": int(station["station_rank"]),
                    "reference_p_epoch": float(
                        UTCDateTime(station["primary_pick_time_utc"]).timestamp
                    ),
                    "pick_time_weight": float(station["primary_arrival_weight"]),
                    "epicentral_distance_km": float(station["epicentral_distance_km"]),
                    "magnitude": float(station["magnitude"]),
                    "native_sample_rate_hz": float(station["native_sample_rate_hz"]),
                    "local_p_snr": local_snr,
                    "metadata": metadata,
                    "score_cache": artifacts[key],
                }
            )
            scored_records += 1
            release_device_cache(ensemble.device_name)

        summary = {
            "protocol_sha256": protocol_sha256,
            "scope": "event",
            "unit_key": event_id,
            "input_identity": input_identity,
            "status": "complete",
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "origin_time_utc": stations_by_event[event_id][0]["origin_time_utc"],
            "magnitude": float(stations_by_event[event_id][0]["magnitude"]),
            "requested_station_count": len(stations_by_event[event_id]),
            "scored_station_count": len(station_results),
            "excluded_incomplete_station_count": (
                len(stations_by_event[event_id]) - len(station_results)
            ),
            "station_results": station_results,
            "event_waveform_decode": source_result.diagnostics,
            "shared_core_digest": ensemble.shared_core_digest,
            "elapsed_seconds": time.perf_counter() - started,
            "maximum_resident_set_bytes": maximum_resident_set_bytes(),
            "artifacts": artifacts,
        }
        write_json_atomic(summary_path, summary)
        unit_summaries.append(artifact(summary_path))
        print(
            f"[{event_index}/{len(events)}] {event_id}: "
            f"{len(station_results)} records, {summary['elapsed_seconds']:.2f}s",
            flush=True,
        )

    if args.limit_events:
        return
    if len(unit_summaries) != int(protocol["expected_units"]["events"]):
        raise RuntimeError("Event score-cache unit count differs")
    completion = {
        "completion_id": "cspp_v27_ncedc_event_score_cache_v1",
        "protocol_sha256": protocol_sha256,
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "events": len(unit_summaries),
        "scored_event_station_records": scored_records,
        "excluded_incomplete_event_station_records": skipped_incomplete,
        "unit_summaries": unit_summaries,
        "model_outputs_computed": True,
    }
    write_json_atomic(output_root / "event_score_cache_complete.json", completion)
    print(json.dumps(completion | {"unit_summaries": "omitted"}, indent=2))


if __name__ == "__main__":
    main()
