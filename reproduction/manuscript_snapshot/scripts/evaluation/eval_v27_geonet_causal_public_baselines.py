#!/usr/bin/env python3
"""Replay released public pickers with a no-future publication clock."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import fcntl
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import resource
import subprocess
import sys
import time
from typing import Any, Mapping

import numpy as np
from obspy import UTCDateTime
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.evaluation import (  # noqa: E402
    eval_v27_geonet_final_public_baselines as full_record,
)
from scripts.evaluation import (  # noqa: E402
    eval_v27c_fixed_public_baselines as sealed_public,
)
from scripts.evaluation import (  # noqa: E402
    freeze_v27_geonet_final_public_baseline_protocol as source_freeze,
)
from scripts.evaluation.build_v27_geonet_temporal_score_cache import (  # noqa: E402
    channel_orientations,
    station_target,
)
from oriented_gappy_strong_motion_stream import (  # noqa: E402
    stream_to_oriented_gappy_zne_array,
)
from v20c_public_window_baselines import (  # noqa: E402
    candidate_exposure_packets,
    extract_closed_candidates,
    infer_first_available_annotation,
    replay_packet_quality,
)


PROTOCOL_ID = "cspp_v27_geonet_causal_public_baselines_v1"
DEFAULT_PROTOCOL = (
    ROOT
    / "experiments/"
    "cspp_v27_geonet_causal_public_baseline_protocol_20260730.json"
)
FIELDS = [
    "record_id",
    "event_id",
    "station_key",
    "site_rank",
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
    "within_0p5_seconds",
    "within_1p0_seconds",
    "within_1p5_seconds",
    "available_by_0p5_seconds",
    "available_by_1p0_seconds",
    "available_by_2p0_seconds",
    "available_by_5p0_seconds",
    "available_by_10p0_seconds",
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
    parser.add_argument(
        "--model",
        choices=("phasenet_instance", "eqtransformer_original"),
        required=True,
    )
    parser.add_argument("--event-id")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_artifact(spec: Mapping[str, Any]) -> Path:
    path = Path(str(spec["path"])).resolve()
    if (
        not path.is_file()
        or path.stat().st_size != int(spec["bytes"])
        or sha256(path) != str(spec["sha256"])
    ):
        raise RuntimeError(f"Frozen artifact drifted: {path}")
    return path


def load_protocol(path: Path) -> tuple[dict[str, Any], str]:
    resolved = path.resolve()
    protocol = json.loads(resolved.read_text(encoding="utf-8"))
    if protocol.get("protocol_id") != PROTOCOL_ID:
        raise RuntimeError("Unexpected causal-public protocol")
    for spec in protocol["code_artifacts"].values():
        verify_artifact(spec)
    source_path = verify_artifact(
        protocol["inputs"]["source_public_protocol"]
    )
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if (
        source.get("protocol_id")
        != "cspp_v27c_geonet_final_public_baselines_v2"
        or source["record_manifest_sha256"]
        != protocol["cohort"]["record_manifest_sha256"]
    ):
        raise RuntimeError("Source public protocol differs")
    return {**protocol, "_source": source}, sha256(resolved)


def swap_used_gib() -> float:
    output = subprocess.check_output(
        ["/usr/sbin/sysctl", "vm.swapusage"], text=True
    )
    match = re.search(r"used = ([0-9.]+)([MG])", output)
    if match is None:
        raise RuntimeError(f"Cannot parse swap usage: {output.strip()}")
    value = float(match.group(1))
    return value / 1024.0 if match.group(2) == "M" else value


def maximum_resident_set_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def stable_waveform_sha256(wave: np.ndarray) -> str:
    values = np.ascontiguousarray(wave, dtype=np.float32)
    digest = hashlib.sha256()
    digest.update(str(values.dtype).encode("ascii"))
    digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
    digest.update(values.tobytes())
    return digest.hexdigest()


def output_is_valid(
    summary_path: Path,
    *,
    protocol_sha256: str,
    model_key: str,
    event_id: str,
) -> bool:
    if not summary_path.is_file():
        return False
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (
        summary.get("protocol_sha256") != protocol_sha256
        or summary.get("model_key") != model_key
        or summary.get("event_id") != event_id
    ):
        raise RuntimeError(f"Existing event summary differs: {summary_path}")
    verify_artifact(summary["records_artifact"])
    return True


def write_csv_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def score_row(
    record: Mapping[str, Any],
    *,
    model_key: str,
    method_id: str,
    candidates: list[dict[str, Any]],
    exposure_packets: int,
    quality_valid_packets: int,
    annotation_diagnostics: Mapping[str, Any],
    waveform_sha256: str,
    technical_error: str,
) -> dict[str, Any]:
    reference = float(UTCDateTime(str(record["reference_p_time_utc"])).timestamp)
    origin = float(UTCDateTime(str(record["origin_time_utc"])).timestamp)
    start = origin - 60.0
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
    first_delay = (
        min(float(row["available_epoch"]) for row in matched) - reference
        if matched and not technical_error
        else None
    )
    error = (
        float(selected["estimated_p_epoch"]) - reference
        if selected is not None
        else None
    )
    availability = (
        float(selected["available_epoch"]) - reference
        if selected is not None
        else None
    )
    candidate_count = len(candidates) if not technical_error else 0

    def available_by(seconds: float) -> int:
        return int(first_delay is not None and first_delay <= seconds)

    return {
        "record_id": record["record_id"],
        "event_id": record["event_id"],
        "station_key": record["station_key"],
        "site_rank": int(record["site_rank"]),
        "model_key": model_key,
        "method_id": method_id,
        "reference_p_epoch": reference,
        "origin_epoch": origin,
        "record_start_epoch": start,
        "magnitude": float(record["magnitude"]),
        "epicentral_distance_km": float(record["epicentral_distance_km"]),
        "candidate_count": candidate_count,
        "matched_candidate_count": int(selected is not None),
        "false_candidate_count": candidate_count - int(selected is not None),
        "false_negative": int(selected is None),
        "selected_estimated_p_epoch": (
            "" if selected is None else float(selected["estimated_p_epoch"])
        ),
        "selected_available_epoch": (
            "" if selected is None else float(selected["available_epoch"])
        ),
        "pick_error_seconds": "" if error is None else error,
        "absolute_pick_error_seconds": "" if error is None else abs(error),
        "availability_delay_seconds": (
            "" if availability is None else availability
        ),
        "first_matched_availability_delay_seconds": (
            "" if first_delay is None else first_delay
        ),
        "within_0p5_seconds": int(error is not None and abs(error) <= 0.5),
        "within_1p0_seconds": int(error is not None and abs(error) <= 1.0),
        "within_1p5_seconds": int(error is not None),
        "available_by_0p5_seconds": available_by(0.5),
        "available_by_1p0_seconds": available_by(1.0),
        "available_by_2p0_seconds": available_by(2.0),
        "available_by_5p0_seconds": available_by(5.0),
        "available_by_10p0_seconds": available_by(10.0),
        "candidate_exposure_packets": exposure_packets,
        "quality_valid_packets": quality_valid_packets,
        "published_samples": int(
            annotation_diagnostics.get("published_samples", 0)
        ),
        "executed_valid_windows": int(
            annotation_diagnostics.get("executed_valid_windows", 0)
        ),
        "waveform_sha256": waveform_sha256,
        "technical_error": technical_error,
    }


def main() -> None:
    args = parse_args()
    protocol, protocol_sha256 = load_protocol(args.protocol)
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable")
    if swap_used_gib() > float(
        protocol["execution"]["maximum_swap_used_gib_before_launch"]
    ):
        raise RuntimeError("Swap guard blocked causal public evaluation")

    output_root = Path(protocol["execution"]["output_root"]).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    lock_path = Path(protocol["execution"]["lock_path"]).resolve()
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = lock_path.open("a+")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError("Another causal public MPS job holds the lock") from error

    source_protocol = protocol["_source"]
    records = list(source_protocol["record_manifest"])
    by_event: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_event[str(record["event_id"])].append(record)
    event_ids = protocol["cohort"]["event_ids"]
    if args.event_id is not None:
        if args.event_id not in by_event:
            raise KeyError(args.event_id)
        event_ids = [args.event_id]

    torch.set_num_threads(int(protocol["execution"]["torch_threads"]))
    device = torch.device(str(protocol["execution"]["device"]))
    model_spec = source_protocol["public_models"][args.model]
    model, loaded_digest = sealed_public.load_sealed_public_model(
        model_spec, device=device
    )
    groups = full_record.orientation_groups(source_protocol)
    methods = sorted(str(row["id"]) for row in model_spec["operating_points"])
    started_all = time.time()
    completed = 0
    for position, event_id in enumerate(event_ids, start=1):
        event_root = output_root / args.model / "events" / event_id
        summary_path = event_root / "summary.json"
        if output_is_valid(
            summary_path,
            protocol_sha256=protocol_sha256,
            model_key=args.model,
            event_id=event_id,
        ):
            completed += 1
            continue
        event_root.mkdir(parents=True, exist_ok=True)
        event_records = by_event[event_id]
        response_path = verify_artifact(
            {
                "path": event_records[0]["response_file"],
                "bytes": event_records[0]["response_bytes"],
                "sha256": event_records[0]["response_sha256"],
            }
        )
        source_result, source_error = (
            full_record.read_event_source_or_technical_failure(response_path)
        )
        source = None if source_result is None else source_result.stream
        event_rows = []
        event_started = time.time()
        try:
            for record in event_records:
                error_text = source_error
                wave = None
                candidates_by_method = {method: [] for method in methods}
                exposure_packets = 0
                quality_valid_packets = 0
                diagnostics: dict[str, Any] = {}
                wave_hash = ""
                orientation_rows = groups[record["orientation_group_id"]]
                if (
                    source_freeze.stable_digest(orientation_rows)
                    != record["orientation_group_sha256"]
                ):
                    raise RuntimeError("Orientation group content differs")
                station = {
                    "network": record["network"],
                    "station": record["station"],
                    "location": record["location"],
                    "component_set": record["component_set"],
                }
                origin = UTCDateTime(str(record["origin_time_utc"]))
                if not error_text:
                    try:
                        wave, _ = stream_to_oriented_gappy_zne_array(
                            source,
                            station_target(station),
                            channel_orientations(
                                orientation_rows, record["component_set"]
                            ),
                            origin - 60.0,
                            origin + 120.0,
                            target_sample_rate_hz=100.0,
                            samples_per_packet=50,
                        )
                        if wave.shape != (3, 18000):
                            raise RuntimeError(
                                f"Unexpected waveform shape: {wave.shape}"
                            )
                        wave_hash = stable_waveform_sha256(wave)
                        valid, allowed, quality = replay_packet_quality(
                            wave,
                            start_epoch=float((origin - 60.0).timestamp),
                            warmup_packets=20,
                        )
                        annotation = infer_first_available_annotation(
                            model,
                            model_spec,
                            wave,
                            valid,
                            device=device,
                        )
                        exposure = candidate_exposure_packets(
                            annotation, allowed
                        )
                        exposure_packets = int(exposure.sum())
                        quality_valid_packets = int(valid.sum())
                        diagnostics = annotation.diagnostics
                        for method in model_spec["operating_points"]:
                            method_id = str(method["id"])
                            candidates_by_method[method_id] = (
                                extract_closed_candidates(
                                    annotation,
                                    method=method,
                                    start_epoch=float(
                                        (origin - 60.0).timestamp
                                    ),
                                )
                            )
                    except Exception as error:
                        error_text = f"{type(error).__name__}: {error}"
                for method_id in methods:
                    event_rows.append(
                        score_row(
                            record,
                            model_key=args.model,
                            method_id=method_id,
                            candidates=candidates_by_method[method_id],
                            exposure_packets=exposure_packets,
                            quality_valid_packets=quality_valid_packets,
                            annotation_diagnostics=diagnostics,
                            waveform_sha256=wave_hash,
                            technical_error=error_text,
                        )
                    )
                del wave
        finally:
            if source_result is not None:
                del source
                del source_result
        records_path = event_root / "records.csv"
        write_csv_atomic(records_path, event_rows)
        summary = {
            "summary_id": "cspp_v27_geonet_causal_public_event_v1",
            "protocol_sha256": protocol_sha256,
            "model_key": args.model,
            "event_id": event_id,
            "record_count": len(event_records),
            "method_count": len(methods),
            "output_rows": len(event_rows),
            "technical_failure_rows": sum(
                int(bool(row["technical_error"])) for row in event_rows
            ),
            "runtime_seconds": time.time() - event_started,
            "worker_max_resident_set_bytes": maximum_resident_set_bytes(),
            "loaded_state_dict_sha256": loaded_digest,
            "records_artifact": {
                "path": str(records_path),
                "bytes": records_path.stat().st_size,
                "sha256": sha256(records_path),
            },
        }
        write_json_atomic(summary_path, summary)
        completed += 1
        print(
            f"{args.model} [{position}/{len(event_ids)}] "
            f"{event_id} records={len(event_records)}",
            flush=True,
        )
        gc.collect()
        torch.mps.empty_cache()
        if maximum_resident_set_bytes() > int(
            float(protocol["execution"]["maximum_worker_resident_set_gib"])
            * 1024**3
        ):
            raise RuntimeError("Worker RSS guard exceeded after event commit")

    print(
        json.dumps(
            {
                "model": args.model,
                "completed_events_this_request": completed,
                "requested_events": len(event_ids),
                "elapsed_seconds": time.time() - started_all,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
