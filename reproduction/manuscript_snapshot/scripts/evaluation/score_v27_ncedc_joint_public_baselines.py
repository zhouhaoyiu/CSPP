#!/usr/bin/env python3
"""Score PhaseNet and EQTransformer on the frozen NCSN continuous cohort."""
from __future__ import annotations

import argparse
import csv
from datetime import UTC, datetime
import fcntl
import gc
import json
import os
from pathlib import Path
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

import build_v27_ncedc_continuous_score_cache as continuous_io  # noqa: E402
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


DEFAULT_ROOT = Path(
    "workspace/"
    "ncedc_joint_continuous_m3_20260904"
)
SOURCE_PROTOCOL = Path(
    "workspace/ncedc_model_blind_20260817/"
    "causal_public_baseline_protocol.json"
)
MODEL_KEYS = ("phasenet_instance", "eqtransformer_original")
CANDIDATE_FIELDS = [
    "candidate_id",
    "model_key",
    "method_id",
    "date_utc",
    "station_key",
    "episode_index",
    "estimated_p_epoch",
    "available_epoch",
    "p_probability",
    "detection_probability",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--model", choices=MODEL_KEYS)
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--finalize-only", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
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


def write_csv_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CANDIDATE_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def freeze_protocol(root: Path) -> Path:
    destination = root / "public_baseline_protocol.json"
    if destination.exists():
        return destination
    joint = json.loads((root / "joint_continuous_protocol.json").read_text())
    acquisition = json.loads((root / "waveforms/summary.json").read_text())
    cspp = json.loads((root / "analysis/result.json").read_text())
    source = json.loads(SOURCE_PROTOCOL.read_text())
    if acquisition["completed_station_days"] != joint["counts"]["station_day_requests"]:
        raise RuntimeError("Continuous acquisition is incomplete")
    payload = {
        "protocol_id": "cspp_v27_ncedc_joint_public_baselines_v1",
        "frozen_at_utc": datetime.now(UTC).isoformat(),
        "status": "frozen_before_joint-continuous public-baseline output",
        "inputs": {
            "joint_protocol": artifact(root / "joint_continuous_protocol.json"),
            "station_day_requests": joint["artifacts"]["station_day_requests"],
            "station_day_orientations": joint["artifacts"]["station_day_orientations"],
            "acquisition_status": artifact(root / "waveforms/latest_status.csv"),
            "acquisition_summary": artifact(root / "waveforms/summary.json"),
            "opened_cspp_result": artifact(root / "analysis/result.json"),
        },
        "models": {key: source["models"][key] for key in MODEL_KEYS},
        "causal_visibility": source["causal_visibility"],
        "comparison": {
            "packet_clock_seconds": 0.5,
            "association_window_seconds": 10.0,
            "association_minimum_stations": 3,
            "event_magnitude_thresholds": [3.0, 4.0],
            "event_distance_limit_km": 200.0,
            "catalog_candidate_tolerance_seconds": 5.0,
            "public_thresholds_changed": False,
            "native_windows_and_blinding_retained": True,
            "same_waveforms_as_cspp": True,
            "same_quality_gate_as_cspp": True,
        },
        "execution": {
            "output_root": str(root / "public_baselines"),
            "device": "mps",
            "workers": 1,
            "serial_models": True,
            "maximum_process_rss_bytes": 6442450944,
            "global_mps_lock": str(root / ".public_baseline_mps.lock"),
        },
        "expected_units": {
            "requested_station_days": acquisition["requested_station_days"],
            "data_bearing_station_days": acquisition["completed_station_days"]
            - acquisition["status_counts"].get("download_or_parse_error", 0),
        },
        "evidence_boundary": {
            "cspp_results_opened_before_baseline_freeze": True,
            "baseline_threshold_tuning_permitted": False,
            "matched_retraining_claim_permitted": False,
            "architecture_superiority_claim_permitted": False,
            "selection_after_baseline_output_permitted": False,
            "opened_cspp_primary_recall": cspp["event_metrics"]["m3_or_greater"]["recall"],
        },
        "source_public_protocol": artifact(SOURCE_PROTOCOL),
        "code_sha256": {
            "scripts/evaluation/score_v27_ncedc_joint_public_baselines.py": sha256_file(
                Path(__file__).resolve()
            ),
            "src/v20c_public_window_baselines.py": sha256_file(
                ROOT / "src/v20c_public_window_baselines.py"
            ),
            "src/oriented_gappy_strong_motion_stream.py": sha256_file(
                ROOT / "src/oriented_gappy_strong_motion_stream.py"
            ),
            "src/stream_quality_v5.py": sha256_file(ROOT / "src/stream_quality_v5.py"),
        },
    }
    continuous_io.write_json_atomic(destination, payload)
    return destination


def load_protocol(path: Path) -> tuple[dict[str, Any], str]:
    protocol = json.loads(path.read_text())
    if protocol.get("protocol_id") != "cspp_v27_ncedc_joint_public_baselines_v1":
        raise RuntimeError("Unexpected public-baseline protocol")
    for relative, expected in protocol["code_sha256"].items():
        source = ROOT / relative
        if sha256_file(source) != expected:
            raise RuntimeError(f"Public-baseline code drifted: {relative}")
    for spec in protocol["inputs"].values():
        verify_artifact(spec)
    for model in protocol["models"].values():
        for spec in model["artifacts"].values():
            verify_artifact(spec)
    return protocol, sha256_file(path)


def orientation(rows: list[dict[str, str]]) -> list[ChannelOrientation]:
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


def candidate_rows(
    candidates: list[dict[str, Any]],
    *,
    request: dict[str, str],
    model_key: str,
    method_id: str,
) -> list[dict[str, Any]]:
    station = continuous_io.station_key(request)
    return [
        {
            "candidate_id": (
                f"{request['date_utc']}:{station}:{model_key}:{method_id}:"
                f"{candidate['episode_index']}"
            ),
            "model_key": model_key,
            "method_id": method_id,
            "date_utc": request["date_utc"],
            "station_key": station,
            "episode_index": int(candidate["episode_index"]),
            "estimated_p_epoch": float(candidate["estimated_p_epoch"]),
            "available_epoch": float(candidate["available_epoch"]),
            "p_probability": float(candidate["p_probability"]),
            "detection_probability": (
                ""
                if candidate["detection_probability"] is None
                else float(candidate["detection_probability"])
            ),
        }
        for candidate in candidates
    ]


def summary_valid(
    path: Path, *, protocol_sha256: str, identity: str, input_identity: str
) -> bool:
    if not path.is_file():
        return False
    summary = json.loads(path.read_text())
    expected = {
        "protocol_sha256": protocol_sha256,
        "station_day_id": identity,
        "input_identity": input_identity,
        "status": "complete",
    }
    for key, value in expected.items():
        if summary.get(key) != value:
            raise RuntimeError(f"Existing summary differs in {key}: {path}")
    verify_artifact(summary["candidate_exposure"])
    verify_artifact(summary["candidates"])
    return True


def finalize(protocol: dict[str, Any], protocol_sha256: str, model_key: str) -> None:
    output = Path(protocol["execution"]["output_root"]) / model_key
    summaries = sorted(output.glob("station_day/*/*/summary.json"))
    expected = int(protocol["expected_units"]["data_bearing_station_days"])
    if len(summaries) != expected:
        raise RuntimeError(f"Expected {expected} summaries, got {len(summaries)}")
    specs = []
    exposure_packets = candidates = 0
    for path in summaries:
        summary = json.loads(path.read_text())
        if summary.get("protocol_sha256") != protocol_sha256:
            raise RuntimeError(f"Summary protocol differs: {path}")
        verify_artifact(summary["candidate_exposure"])
        verify_artifact(summary["candidates"])
        exposure_packets += int(summary["candidate_exposure_packets"])
        candidates += int(summary["candidate_count"])
        specs.append(artifact(path))
    continuous_io.write_json_atomic(
        output / "completion.json",
        {
            "completion_id": f"cspp_v27_ncedc_joint_{model_key}_v1",
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "protocol_sha256": protocol_sha256,
            "model_key": model_key,
            "data_bearing_station_days": len(specs),
            "candidate_exposure_packets": exposure_packets,
            "candidate_count_all_methods": candidates,
            "unit_summaries": specs,
        },
    )


def main() -> None:
    args = parse_args()
    protocol_path = freeze_protocol(args.root)
    if args.freeze_only:
        print(json.dumps(artifact(protocol_path), indent=2))
        return
    if not args.model:
        raise SystemExit("--model is required unless --freeze-only is used")
    protocol, protocol_sha256 = load_protocol(protocol_path)
    if args.finalize_only:
        finalize(protocol, protocol_sha256, args.model)
        return
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable")
    lock_path = Path(protocol["execution"]["global_mps_lock"])
    lock = lock_path.open("a+")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError("Another public-baseline MPS job holds the lock") from error

    requests = continuous_io.read_csv(
        verify_artifact(protocol["inputs"]["station_day_requests"])
    )
    requests.sort(key=lambda row: (row["date_utc"], int(row["availability_rank"])))
    acquisitions = {
        row["station_day_id"]: row
        for row in continuous_io.read_csv(
            verify_artifact(protocol["inputs"]["acquisition_status"])
        )
    }
    orientations = continuous_io.orientation_groups(
        continuous_io.read_csv(
            verify_artifact(protocol["inputs"]["station_day_orientations"])
        )
    )
    model_spec = protocol["models"][args.model]
    model, loaded_digest = public.load_sealed_public_model(
        model_spec, device=torch.device("mps")
    )
    methods = list(model_spec["operating_points"])
    output = Path(protocol["execution"]["output_root"]) / args.model
    processed = 0
    for index, request in enumerate(requests, start=1):
        identity = continuous_io.station_day_id(request)
        acquisition = acquisitions[identity]
        if acquisition["status"] == "download_or_parse_error":
            continue
        if args.limit and processed >= args.limit:
            break
        input_identity = continuous_io.stable_hash(
            {
                "request": request,
                "acquisition": acquisition,
                "orientation_sha256": protocol["inputs"]["station_day_orientations"]["sha256"],
                "model_key": args.model,
            }
        )
        root = continuous_io.unit_root(output, request)
        summary_path = root / "summary.json"
        if summary_valid(
            summary_path,
            protocol_sha256=protocol_sha256,
            identity=identity,
            input_identity=input_identity,
        ):
            continue
        started = time.perf_counter()
        source_path = verify_artifact(
            {
                "path": acquisition["mseed_file"],
                "bytes": int(acquisition["mseed_bytes"]),
                "sha256": acquisition["mseed_sha256"],
            }
        )
        source = read_resilient_mseed(source_path)
        start = UTCDateTime(request["request_start_utc"])
        end = UTCDateTime(request["request_end_utc"])
        wave, _ = stream_to_oriented_gappy_zne_array(
            source.stream,
            {
                "network": request["network"],
                "station": request["station"],
                "location": request["location"],
                "latitude": float(request["station_latitude"]),
                "longitude": float(request["station_longitude"]),
            },
            orientation(orientations[identity]),
            start,
            end,
            target_sample_rate_hz=100.0,
            samples_per_packet=50,
        )
        if wave.shape != (3, 8_640_000):
            raise RuntimeError(f"Unexpected station-day waveform shape: {wave.shape}")
        valid, allowed, quality = replay_packet_quality(
            wave, start_epoch=float(start.timestamp), warmup_packets=20
        )
        annotation = infer_first_available_annotation(
            model, model_spec, wave, valid, device=torch.device("mps")
        )
        exposure = candidate_exposure_packets(annotation, allowed)
        rows: list[dict[str, Any]] = []
        method_counts = {}
        for method in methods:
            method_id = str(method["id"])
            values = extract_closed_candidates(
                annotation, method=method, start_epoch=float(start.timestamp)
            )
            method_counts[method_id] = len(values)
            rows.extend(
                candidate_rows(
                    values,
                    request=request,
                    model_key=args.model,
                    method_id=method_id,
                )
            )
        exposure_path = root / "candidate_exposure.npz"
        candidates_path = root / "candidates.csv"
        continuous_io.write_npz_atomic(
            exposure_path, {"candidate_exposure": exposure.astype(bool, copy=False)}
        )
        write_csv_atomic(candidates_path, rows)
        rss = continuous_io.maximum_resident_set_bytes()
        if rss > int(protocol["execution"]["maximum_process_rss_bytes"]):
            raise MemoryError(f"RSS guard exceeded: {rss}")
        summary = {
            "protocol_sha256": protocol_sha256,
            "station_day_id": identity,
            "date_utc": request["date_utc"],
            "station_key": continuous_io.station_key(request),
            "input_identity": input_identity,
            "status": "complete",
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "acquisition_status": acquisition["status"],
            "model_key": args.model,
            "loaded_state_dict_sha256": loaded_digest,
            "candidate_count": len(rows),
            "candidate_count_by_method": method_counts,
            "candidate_exposure_packets": int(exposure.sum()),
            "quality_valid_packets": int(valid.sum()),
            "quality": quality,
            "annotation": annotation.diagnostics,
            "candidate_exposure": artifact(exposure_path),
            "candidates": artifact(candidates_path),
            "elapsed_seconds": time.perf_counter() - started,
            "maximum_resident_set_bytes": rss,
        }
        continuous_io.write_json_atomic(summary_path, summary)
        print(
            f"{args.model} [{index}/{len(requests)}] {identity}: "
            f"candidates={len(rows)}, elapsed={summary['elapsed_seconds']:.1f}s",
            flush=True,
        )
        processed += 1
        del source, wave, valid, allowed, annotation, exposure, rows
        gc.collect()
        torch.mps.empty_cache()
    if not args.limit:
        finalize(protocol, protocol_sha256, args.model)


if __name__ == "__main__":
    main()
