#!/usr/bin/env python3
"""Replay sealed PhaseNet and EQTransformer on the frozen NCSN event cohort."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import UTC, datetime
import fcntl
import gc
import json
import os
from pathlib import Path
import resource
import sys
import time
from typing import Any

import numpy as np
from obspy import UTCDateTime
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/evaluation"))

import build_v27_ncedc_event_score_cache as event_cache  # noqa: E402
from scripts.evaluation import eval_v27c_fixed_public_baselines as public  # noqa: E402
from cspp_v27_reused_core_score_cache import sha256_file  # noqa: E402
from oriented_gappy_strong_motion_stream import (  # noqa: E402
    ChannelOrientation,
    stream_to_oriented_gappy_zne_array,
)
from resilient_mseed import read_resilient_mseed  # noqa: E402
from v20c_public_window_baselines import (  # noqa: E402
    candidate_exposure_packets,
    extract_closed_candidates,
    infer_first_available_annotation,
    replay_packet_quality,
)


DEFAULT_PROTOCOL = (
    ROOT
    / "experiments/cspp_v27_ncedc_causal_public_baseline_protocol_20260817.json"
)
FIELDS = [
    "record_id",
    "event_id",
    "station_key",
    "station_rank",
    "model_key",
    "method_id",
    "reference_p_epoch",
    "origin_epoch",
    "record_start_epoch",
    "magnitude",
    "epicentral_distance_km",
    "candidate_count",
    "matched_candidate_count",
    "false_candidate_count",
    "false_negative",
    "selected_estimated_p_epoch",
    "selected_available_epoch",
    "pick_error_seconds",
    "absolute_pick_error_seconds",
    "availability_delay_seconds",
    "first_matched_availability_delay_seconds",
    "candidate_exposure_packets",
    "quality_valid_packets",
    "published_samples",
    "executed_valid_windows",
    "waveform_sha256",
    "technical_error",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument(
        "--model",
        choices=("phasenet_instance", "eqtransformer_original"),
        required=True,
    )
    parser.add_argument("--limit-events", type=int, default=0)
    parser.add_argument("--finalize-only", action="store_true")
    return parser.parse_args()


def artifact(path: Path) -> dict[str, Any]:
    path = path.resolve()
    return {
        "path": str(path),
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


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=FIELDS,
            extrasaction="raise",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_json(path: Path, payload: Any) -> None:
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


def maximum_resident_set_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def waveform_digest(wave: np.ndarray) -> str:
    values = np.ascontiguousarray(wave, dtype=np.float32)
    import hashlib

    digest = hashlib.sha256()
    digest.update(str(values.dtype).encode("ascii"))
    digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
    digest.update(values.tobytes())
    return digest.hexdigest()


def load_protocol(path: Path, expected: str) -> tuple[dict[str, Any], str]:
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(f"Public baseline protocol SHA-256 differs: {actual}")
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("protocol_id") != "cspp_v27_ncedc_causal_public_v1":
        raise RuntimeError("Unexpected NCSN public baseline protocol")
    for relative, expected_hash in protocol["code_sha256"].items():
        source = ROOT / relative
        if not source.is_file() or sha256_file(source) != expected_hash:
            raise RuntimeError(f"Public baseline code drifted: {relative}")
    for spec in protocol["inputs"].values():
        verify_artifact(spec)
    for model in protocol["models"].values():
        for spec in model["artifacts"].values():
            verify_artifact(spec)
    return protocol, actual


def load_cohort_keys(protocol: dict[str, Any]) -> dict[str, set[str]]:
    completion = json.loads(
        verify_artifact(protocol["inputs"]["cspp_event_score_completion"])
        .read_text(encoding="utf-8")
    )
    keys: dict[str, set[str]] = defaultdict(set)
    for summary_spec in completion["unit_summaries"]:
        summary_path = verify_artifact(summary_spec)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        event_id = summary_path.parent.name
        for station in summary["station_results"]:
            keys[event_id].add(str(station["station_key"]))
    if sum(map(len, keys.values())) != int(protocol["cohort"]["records"]):
        raise RuntimeError("NCSN public-baseline cohort count differs")
    return keys


def orientation(group: list[dict[str, str]]) -> list[ChannelOrientation]:
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
        for row in group
    ]


def score_row(
    record: dict[str, str],
    *,
    model_key: str,
    method_id: str,
    candidates: list[dict[str, Any]],
    exposure_packets: int,
    quality_valid_packets: int,
    diagnostics: dict[str, Any],
    wave_hash: str,
    technical_error: str,
) -> dict[str, Any]:
    reference = float(UTCDateTime(record["primary_pick_time_utc"]).timestamp)
    origin = float(UTCDateTime(record["origin_time_utc"]).timestamp)
    start = float(UTCDateTime(record["request_start_utc"]).timestamp)
    matched = [
        candidate
        for candidate in candidates
        if abs(float(candidate["estimated_p_epoch"]) - reference) <= 1.5
    ]
    selected = (
        min(
            matched,
            key=lambda row: (
                abs(float(row["estimated_p_epoch"]) - reference),
                float(row["available_epoch"]),
            ),
        )
        if matched and not technical_error
        else None
    )
    error = (
        None
        if selected is None
        else float(selected["estimated_p_epoch"]) - reference
    )
    delay = (
        None if selected is None else float(selected["available_epoch"]) - reference
    )
    first_delay = (
        None
        if not matched or technical_error
        else min(float(row["available_epoch"]) for row in matched) - reference
    )
    count = 0 if technical_error else len(candidates)
    return {
        "record_id": (
            f"{record['event_id']}:{record['network']}.{record['station']}."
            f"{record['location']}"
        ),
        "event_id": record["event_id"],
        "station_key": ".".join(
            (record["network"], record["station"], record["location"])
        ),
        "station_rank": int(record["station_rank"]),
        "model_key": model_key,
        "method_id": method_id,
        "reference_p_epoch": reference,
        "origin_epoch": origin,
        "record_start_epoch": start,
        "magnitude": float(record["magnitude"]),
        "epicentral_distance_km": float(record["epicentral_distance_km"]),
        "candidate_count": count,
        "matched_candidate_count": int(selected is not None),
        "false_candidate_count": count - int(selected is not None),
        "false_negative": int(selected is None),
        "selected_estimated_p_epoch": (
            "" if selected is None else float(selected["estimated_p_epoch"])
        ),
        "selected_available_epoch": (
            "" if selected is None else float(selected["available_epoch"])
        ),
        "pick_error_seconds": "" if error is None else error,
        "absolute_pick_error_seconds": "" if error is None else abs(error),
        "availability_delay_seconds": "" if delay is None else delay,
        "first_matched_availability_delay_seconds": (
            "" if first_delay is None else first_delay
        ),
        "candidate_exposure_packets": exposure_packets,
        "quality_valid_packets": quality_valid_packets,
        "published_samples": int(diagnostics.get("published_samples", 0)),
        "executed_valid_windows": int(
            diagnostics.get("executed_valid_windows", 0)
        ),
        "waveform_sha256": wave_hash,
        "technical_error": technical_error,
    }


def finalize(
    protocol: dict[str, Any], protocol_sha256: str, model_key: str
) -> None:
    root = Path(protocol["execution"]["output_root"]) / model_key
    summaries = []
    rows = 0
    for event_id in protocol["cohort"]["event_ids"]:
        path = root / "events" / event_id / "summary.json"
        summary = json.loads(path.read_text(encoding="utf-8"))
        if (
            summary.get("protocol_sha256") != protocol_sha256
            or summary.get("model_key") != model_key
        ):
            raise RuntimeError(f"Public baseline event summary differs: {path}")
        verify_artifact(summary["records"])
        summaries.append(artifact(path))
        rows += int(summary["row_count"])
    completion = {
        "completion_id": f"cspp_v27_ncedc_causal_public_{model_key}_v1",
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "protocol_sha256": protocol_sha256,
        "model_key": model_key,
        "events": len(summaries),
        "records": int(protocol["cohort"]["records"]),
        "rows": rows,
        "event_summaries": summaries,
    }
    write_json(root / "completion.json", completion)
    print(json.dumps(completion | {"event_summaries": "omitted"}, indent=2))


def main() -> None:
    args = parse_args()
    protocol, protocol_sha256 = load_protocol(
        args.protocol.resolve(), args.expected_protocol_sha256
    )
    if args.finalize_only:
        finalize(protocol, protocol_sha256, args.model)
        return
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable")
    lock_path = Path(protocol["execution"]["global_mps_lock"])
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = lock_path.open("a+")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError("Another public-baseline MPS job holds the lock") from error

    requests = read_csv(verify_artifact(protocol["inputs"]["event_station_requests"]))
    acquisitions = {
        row["event_id"]: row
        for row in read_csv(verify_artifact(protocol["inputs"]["acquisition_status"]))
    }
    groups = event_cache.orientation_groups(
        read_csv(verify_artifact(protocol["inputs"]["event_station_orientations"]))
    )
    cohort = load_cohort_keys(protocol)
    by_event: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in requests:
        key = ".".join((row["network"], row["station"], row["location"]))
        if key in cohort.get(row["event_id"], set()):
            by_event[row["event_id"]].append(row)
    for rows in by_event.values():
        rows.sort(key=lambda row: int(row["station_rank"]))
    event_ids = list(protocol["cohort"]["event_ids"])
    if args.limit_events:
        event_ids = event_ids[: args.limit_events]

    model_spec = protocol["models"][args.model]
    model, loaded_digest = public.load_sealed_public_model(
        model_spec, device=torch.device("mps")
    )
    methods = sorted(str(item["id"]) for item in model_spec["operating_points"])
    output_root = Path(protocol["execution"]["output_root"]) / args.model
    for position, event_id in enumerate(event_ids, start=1):
        root = output_root / "events" / event_id
        summary_path = root / "summary.json"
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if (
                summary.get("protocol_sha256") != protocol_sha256
                or summary.get("model_key") != args.model
            ):
                raise RuntimeError(f"Existing public summary differs: {summary_path}")
            verify_artifact(summary["records"])
            print(f"{args.model} [{position}/{len(event_ids)}] {event_id}: cached")
            continue
        started = time.perf_counter()
        acquisition = acquisitions[event_id]
        source_path = verify_artifact(
            {
                "path": acquisition["mseed_file"],
                "bytes": int(acquisition["mseed_bytes"]),
                "sha256": acquisition["mseed_sha256"],
            }
        )
        source_result = read_resilient_mseed(source_path)
        event_rows: list[dict[str, Any]] = []
        for record in by_event[event_id]:
            technical_error = ""
            candidates_by_method = {method: [] for method in methods}
            exposure_packets = 0
            quality_packets = 0
            diagnostics: dict[str, Any] = {}
            wave_hash = ""
            wave = None
            try:
                group_id = (
                    f"{event_id}.{record['network']}.{record['station']}."
                    f"{record['location']}"
                )
                start = UTCDateTime(record["request_start_utc"])
                end = UTCDateTime(record["request_end_utc"])
                wave, _ = stream_to_oriented_gappy_zne_array(
                    source_result.stream,
                    {
                        "network": record["network"],
                        "station": record["station"],
                        "location": record["location"],
                        "latitude": float(record["station_latitude"]),
                        "longitude": float(record["station_longitude"]),
                    },
                    orientation(groups[group_id]),
                    start,
                    end,
                    target_sample_rate_hz=100.0,
                    samples_per_packet=50,
                )
                if wave.shape != (3, 24000):
                    raise RuntimeError(f"Unexpected waveform shape: {wave.shape}")
                wave_hash = waveform_digest(wave)
                valid, allowed, _ = replay_packet_quality(
                    wave,
                    start_epoch=float(start.timestamp),
                    warmup_packets=20,
                )
                annotation = infer_first_available_annotation(
                    model,
                    model_spec,
                    wave,
                    valid,
                    device=torch.device("mps"),
                )
                exposure_packets = int(
                    candidate_exposure_packets(annotation, allowed).sum()
                )
                quality_packets = int(valid.sum())
                diagnostics = annotation.diagnostics
                for method in model_spec["operating_points"]:
                    candidates_by_method[str(method["id"])] = (
                        extract_closed_candidates(
                            annotation,
                            method=method,
                            start_epoch=float(start.timestamp),
                        )
                    )
            except Exception as error:
                technical_error = f"{type(error).__name__}: {error}"
            for method in methods:
                event_rows.append(
                    score_row(
                        record,
                        model_key=args.model,
                        method_id=method,
                        candidates=candidates_by_method[method],
                        exposure_packets=exposure_packets,
                        quality_valid_packets=quality_packets,
                        diagnostics=diagnostics,
                        wave_hash=wave_hash,
                        technical_error=technical_error,
                    )
                )
            del wave
        records_path = root / "records.csv"
        write_csv(records_path, event_rows)
        summary = {
            "summary_id": "cspp_v27_ncedc_causal_public_event_v1",
            "protocol_sha256": protocol_sha256,
            "model_key": args.model,
            "event_id": event_id,
            "record_count": len(by_event[event_id]),
            "method_count": len(methods),
            "row_count": len(event_rows),
            "technical_failure_rows": sum(
                bool(row["technical_error"]) for row in event_rows
            ),
            "elapsed_seconds": time.perf_counter() - started,
            "maximum_resident_set_bytes": maximum_resident_set_bytes(),
            "loaded_state_dict_sha256": loaded_digest,
            "records": artifact(records_path),
        }
        write_json(summary_path, summary)
        print(
            f"{args.model} [{position}/{len(event_ids)}] {event_id}: "
            f"records={len(by_event[event_id])}, "
            f"elapsed={summary['elapsed_seconds']:.1f}s",
            flush=True,
        )
        del source_result
        gc.collect()
        torch.mps.empty_cache()
        if maximum_resident_set_bytes() > int(
            protocol["execution"]["maximum_process_rss_bytes"]
        ):
            raise MemoryError("Public baseline RSS guard exceeded")
    if not args.limit_events:
        finalize(protocol, protocol_sha256, args.model)


if __name__ == "__main__":
    main()
