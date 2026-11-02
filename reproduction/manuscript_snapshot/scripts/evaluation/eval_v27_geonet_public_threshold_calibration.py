#!/usr/bin/env python3
"""Replay public pickers on the frozen GeoNet threshold-calibration cohort."""
from __future__ import annotations

import argparse
from collections import defaultdict
import fcntl
import gc
import json
from pathlib import Path
import sys
import time
from typing import Any

import torch
from obspy import UTCDateTime


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.evaluation import eval_v27_geonet_causal_public_baselines as common  # noqa: E402
from scripts.evaluation import eval_v27_geonet_final_public_baselines as full_record  # noqa: E402
from scripts.evaluation import eval_v27c_fixed_public_baselines as public  # noqa: E402
from scripts.evaluation import freeze_v27_geonet_final_public_baseline_protocol as source_freeze  # noqa: E402
from scripts.evaluation.build_v27_geonet_temporal_score_cache import (  # noqa: E402
    channel_orientations,
    station_target,
)
from oriented_gappy_strong_motion_stream import stream_to_oriented_gappy_zne_array  # noqa: E402
from v20c_public_window_baselines import (  # noqa: E402
    candidate_exposure_packets,
    extract_closed_candidates,
    infer_first_available_annotation,
    replay_packet_quality,
)


DEFAULT_PROTOCOL = (
    ROOT / "experiments/cspp_v27_geonet_public_threshold_calibration_20260824.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument(
        "--model",
        choices=("phasenet_instance", "eqtransformer_original"),
        required=True,
    )
    parser.add_argument("--event-id")
    return parser.parse_args()


def verify_artifact(spec: dict[str, Any]) -> Path:
    path = Path(spec["path"])
    if (
        not path.is_file()
        or path.stat().st_size != int(spec["bytes"])
        or common.sha256(path) != str(spec["sha256"])
    ):
        raise RuntimeError(f"Artifact drifted: {path}")
    return path


def load_protocol(path: Path, expected: str) -> tuple[dict[str, Any], str]:
    actual = common.sha256(path)
    if actual != expected:
        raise RuntimeError(f"Calibration protocol SHA-256 differs: {actual}")
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("protocol_id") != "cspp_v27_geonet_public_threshold_calibration_v1":
        raise RuntimeError("Unexpected calibration protocol")
    if source_freeze.stable_digest(protocol["record_manifest"]) != protocol["record_manifest_sha256"]:
        raise RuntimeError("Calibration record manifest differs")
    for spec in protocol["inputs"].values():
        verify_artifact(spec)
    for relative, digest in protocol["code_sha256"].items():
        source = ROOT / relative
        if not source.is_file() or common.sha256(source) != digest:
            raise RuntimeError(f"Calibration code drifted: {relative}")
    return protocol, actual


def orientation_groups(protocol: dict[str, Any]) -> dict[str, list[dict[str, str]]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in full_record.read_csv(
        verify_artifact(protocol["inputs"]["event_station_orientations"])
    ):
        groups[row["group_id"]].append(row)
    for rows in groups.values():
        rows.sort(key=lambda row: int(row["component_index"]))
    return groups


def main() -> None:
    args = parse_args()
    protocol, protocol_sha256 = load_protocol(args.protocol.resolve(), args.expected_protocol_sha256)
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable")
    if common.swap_used_gib() > float(protocol["execution"]["maximum_swap_used_gib_before_launch"]):
        raise RuntimeError("Swap guard blocked public threshold calibration")
    output_root = Path(protocol["execution"]["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    lock_path = Path(protocol["execution"]["lock_path"])
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = lock_path.open("a+")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError("Another public-model MPS job holds the lock") from error
    records_by_event: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in protocol["record_manifest"]:
        records_by_event[record["event_id"]].append(record)
    event_ids = protocol["cohort"]["event_ids"]
    if args.event_id:
        event_ids = [args.event_id]
    torch.set_num_threads(int(protocol["execution"]["torch_threads"]))
    device = torch.device(protocol["execution"]["device"])
    model_spec = protocol["public_models"][args.model]
    model, loaded_digest = public.load_sealed_public_model(model_spec, device=device)
    groups = orientation_groups(protocol)
    method_ids = [point["id"] for point in model_spec["operating_points"]]
    for position, event_id in enumerate(event_ids, start=1):
        event_root = output_root / args.model / "events" / event_id
        summary_path = event_root / "summary.json"
        if common.output_is_valid(
            summary_path,
            protocol_sha256=protocol_sha256,
            model_key=args.model,
            event_id=event_id,
        ):
            continue
        event_root.mkdir(parents=True, exist_ok=True)
        event_records = records_by_event[event_id]
        response_path = verify_artifact(
            {
                "path": event_records[0]["response_file"],
                "bytes": event_records[0]["response_bytes"],
                "sha256": event_records[0]["response_sha256"],
            }
        )
        source_result, source_error = full_record.read_event_source_or_technical_failure(response_path)
        source = None if source_result is None else source_result.stream
        rows = []
        started = time.time()
        try:
            for record in event_records:
                error_text = source_error
                candidates = {method_id: [] for method_id in method_ids}
                exposure_packets = quality_valid_packets = 0
                diagnostics: dict[str, Any] = {}
                wave_hash = ""
                orientation_rows = groups[record["orientation_group_id"]]
                if source_freeze.stable_digest(orientation_rows) != record["orientation_group_sha256"]:
                    raise RuntimeError("Orientation group content differs")
                station = {key: record[key] for key in ("network", "station", "location", "component_set")}
                origin = UTCDateTime(record["origin_time_utc"])
                wave = None
                if not error_text:
                    try:
                        wave, _ = stream_to_oriented_gappy_zne_array(
                            source,
                            station_target(station),
                            channel_orientations(orientation_rows, record["component_set"]),
                            origin - 60.0,
                            origin + 120.0,
                            target_sample_rate_hz=100.0,
                            samples_per_packet=50,
                        )
                        if wave.shape != (3, 18000):
                            raise RuntimeError(f"Unexpected waveform shape: {wave.shape}")
                        wave_hash = common.stable_waveform_sha256(wave)
                        valid, allowed, _ = replay_packet_quality(
                            wave,
                            start_epoch=float((origin - 60.0).timestamp),
                            warmup_packets=20,
                        )
                        annotation = infer_first_available_annotation(
                            model, model_spec, wave, valid, device=device
                        )
                        exposure_packets = int(candidate_exposure_packets(annotation, allowed).sum())
                        quality_valid_packets = int(valid.sum())
                        diagnostics = annotation.diagnostics
                        for point in model_spec["operating_points"]:
                            candidates[point["id"]] = extract_closed_candidates(
                                annotation,
                                method=point,
                                start_epoch=float((origin - 60.0).timestamp),
                            )
                    except Exception as error:
                        error_text = f"{type(error).__name__}: {error}"
                for method_id in method_ids:
                    rows.append(
                        common.score_row(
                            record,
                            model_key=args.model,
                            method_id=method_id,
                            candidates=candidates[method_id],
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
                del source, source_result
        records_path = event_root / "records.csv"
        common.write_csv_atomic(records_path, rows)
        common.write_json_atomic(
            summary_path,
            {
                "summary_id": "cspp_v27_geonet_public_threshold_calibration_event_v1",
                "protocol_sha256": protocol_sha256,
                "model_key": args.model,
                "event_id": event_id,
                "record_count": len(event_records),
                "method_count": len(method_ids),
                "output_rows": len(rows),
                "technical_failure_rows": sum(bool(row["technical_error"]) for row in rows),
                "runtime_seconds": time.time() - started,
                "worker_max_resident_set_bytes": common.maximum_resident_set_bytes(),
                "loaded_state_dict_sha256": loaded_digest,
                "records_artifact": {
                    "path": str(records_path),
                    "bytes": records_path.stat().st_size,
                    "sha256": common.sha256(records_path),
                },
            },
        )
        print(f"{args.model} [{position}/{len(event_ids)}] {event_id}", flush=True)
        gc.collect()
        torch.mps.empty_cache()
        if common.maximum_resident_set_bytes() > int(protocol["execution"]["maximum_worker_resident_set_gib"] * 1024**3):
            raise RuntimeError("Worker RSS guard exceeded")


if __name__ == "__main__":
    main()
