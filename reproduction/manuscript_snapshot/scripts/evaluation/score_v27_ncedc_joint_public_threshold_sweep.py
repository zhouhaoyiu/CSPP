#!/usr/bin/env python3
"""Post-outcome public-picker threshold sweep on the joint NCSN cohort."""
from __future__ import annotations

import argparse
from datetime import UTC, datetime
import fcntl
import gc
import json
from pathlib import Path
import sys
import time
from typing import Any

from obspy import UTCDateTime
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/evaluation"))

import build_v27_ncedc_continuous_score_cache as continuous_io  # noqa: E402
from scripts.evaluation import eval_v27c_fixed_public_baselines as public  # noqa: E402
import score_v27_ncedc_joint_public_baselines as base  # noqa: E402
from cspp_v27_reused_core_score_cache import sha256_file  # noqa: E402
from oriented_gappy_strong_motion_stream import (  # noqa: E402
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
THRESHOLDS = {
    "phasenet_instance": (
        0.08639824281948567,
        0.10,
        0.15,
        0.20,
        0.30,
        0.40,
        0.50,
        0.60,
        0.70,
        0.80,
        0.85,
        0.90,
        0.925,
        0.95,
        0.975,
        0.99,
        0.995,
        0.999,
        0.9995,
        0.9999,
        0.99995,
        0.99999,
    ),
    "eqtransformer_original": (
        0.001,
        0.002,
        0.005,
        0.01,
        0.02,
        0.03,
        0.04,
        0.05,
        0.06,
        0.07,
        0.08,
        0.09,
        0.10,
        0.12,
        0.15,
        0.20,
        0.30,
        0.50,
        0.70,
        0.90,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--model", choices=tuple(THRESHOLDS))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--finalize-only", action="store_true")
    return parser.parse_args()


def method_id(model: str, threshold: float) -> str:
    token = f"{threshold:.9f}".rstrip("0").rstrip(".").replace(".", "p")
    return f"{model}_p_t{token}"


def methods(model: str) -> list[dict[str, Any]]:
    return [
        {
            "id": method_id(model, threshold),
            "p_threshold": threshold,
            "detection_threshold": None,
        }
        for threshold in THRESHOLDS[model]
    ]


def protocol_path(root: Path) -> Path:
    return root / "public_baseline_threshold_sweep_protocol.json"


def freeze_protocol(root: Path) -> Path:
    destination = protocol_path(root)
    if destination.exists():
        return destination
    source = root / "public_baseline_protocol.json"
    payload = {
        "protocol_id": "cspp_v27_ncedc_public_threshold_sweep_v1",
        "frozen_at_utc": datetime.now(UTC).isoformat(),
        "status": "post-outcome sensitivity analysis",
        "source_public_baseline_protocol": base.artifact(source),
        "thresholds": {key: list(value) for key, value in THRESHOLDS.items()},
        "output_root": str(root / "public_baseline_threshold_sweep"),
        "claim_boundary": {
            "primary_or_blind_claim_permitted": False,
            "threshold_selection_on_ncsn_permitted": False,
            "purpose": "operating curves and matched-false-trigger sensitivity only",
        },
        "code_sha256": sha256_file(Path(__file__).resolve()),
    }
    continuous_io.write_json_atomic(destination, payload)
    return destination


def load_protocol(root: Path) -> tuple[dict[str, Any], dict[str, Any], str]:
    path = freeze_protocol(root)
    protocol = json.loads(path.read_text())
    if protocol["code_sha256"] != sha256_file(Path(__file__).resolve()):
        raise RuntimeError("Threshold-sweep scorer changed after protocol freeze")
    source_path = base.verify_artifact(protocol["source_public_baseline_protocol"])
    source, _ = base.load_protocol(source_path)
    return protocol, source, sha256_file(path)


def summary_valid(
    path: Path, *, protocol_sha256: str, identity: str, input_identity: str
) -> bool:
    if not path.is_file():
        return False
    summary = json.loads(path.read_text())
    for key, expected in {
        "protocol_sha256": protocol_sha256,
        "station_day_id": identity,
        "input_identity": input_identity,
        "status": "complete",
    }.items():
        if summary.get(key) != expected:
            raise RuntimeError(f"Existing summary differs in {key}: {path}")
    base.verify_artifact(summary["candidate_exposure"])
    base.verify_artifact(summary["candidates"])
    return True


def finalize(protocol: dict[str, Any], source: dict[str, Any], digest: str, model: str) -> None:
    output = Path(protocol["output_root"]) / model
    summaries = sorted(output.glob("station_day/*/*/summary.json"))
    expected = int(source["expected_units"]["data_bearing_station_days"])
    if len(summaries) != expected:
        raise RuntimeError(f"Expected {expected} summaries, got {len(summaries)}")
    specs = []
    counts: dict[str, int] = {item["id"]: 0 for item in methods(model)}
    for path in summaries:
        summary = json.loads(path.read_text())
        if summary["protocol_sha256"] != digest:
            raise RuntimeError(f"Summary protocol differs: {path}")
        base.verify_artifact(summary["candidate_exposure"])
        base.verify_artifact(summary["candidates"])
        for key, value in summary["candidate_count_by_method"].items():
            counts[key] += int(value)
        specs.append(base.artifact(path))
    continuous_io.write_json_atomic(
        output / "completion.json",
        {
            "completion_id": f"cspp_v27_ncedc_{model}_threshold_sweep_v1",
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "protocol_sha256": digest,
            "model_key": model,
            "data_bearing_station_days": len(specs),
            "candidate_count_by_method": counts,
            "unit_summaries": specs,
        },
    )


def main() -> None:
    args = parse_args()
    protocol, source_protocol, digest = load_protocol(args.root)
    if args.freeze_only:
        print(json.dumps(base.artifact(protocol_path(args.root)), indent=2))
        return
    if not args.model:
        raise SystemExit("--model is required")
    if args.finalize_only:
        finalize(protocol, source_protocol, digest, args.model)
        return
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable")

    lock = Path(source_protocol["execution"]["global_mps_lock"]).open("a+")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError("Another public-baseline MPS job holds the lock") from error

    requests = continuous_io.read_csv(
        base.verify_artifact(source_protocol["inputs"]["station_day_requests"])
    )
    requests.sort(key=lambda row: (row["date_utc"], int(row["availability_rank"])))
    acquisitions = {
        row["station_day_id"]: row
        for row in continuous_io.read_csv(
            base.verify_artifact(source_protocol["inputs"]["acquisition_status"])
        )
    }
    orientations = continuous_io.orientation_groups(
        continuous_io.read_csv(
            base.verify_artifact(source_protocol["inputs"]["station_day_orientations"])
        )
    )
    model_spec = source_protocol["models"][args.model]
    model, loaded_digest = public.load_sealed_public_model(
        model_spec, device=torch.device("mps")
    )
    method_specs = methods(args.model)
    output = Path(protocol["output_root"]) / args.model
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
                "orientation_sha256": source_protocol["inputs"]["station_day_orientations"]["sha256"],
                "model_key": args.model,
                "thresholds": list(THRESHOLDS[args.model]),
            }
        )
        root = continuous_io.unit_root(output, request)
        summary_path = root / "summary.json"
        if summary_valid(
            summary_path,
            protocol_sha256=digest,
            identity=identity,
            input_identity=input_identity,
        ):
            continue

        started = time.perf_counter()
        source_path = base.verify_artifact(
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
            base.orientation(orientations[identity]),
            start,
            end,
            target_sample_rate_hz=100.0,
            samples_per_packet=50,
        )
        valid, allowed, quality = replay_packet_quality(
            wave, start_epoch=float(start.timestamp), warmup_packets=20
        )
        annotation = infer_first_available_annotation(
            model, model_spec, wave, valid, device=torch.device("mps")
        )
        exposure = candidate_exposure_packets(annotation, allowed)
        rows = []
        counts = {}
        for method in method_specs:
            values = extract_closed_candidates(
                annotation, method=method, start_epoch=float(start.timestamp)
            )
            counts[method["id"]] = len(values)
            rows.extend(
                base.candidate_rows(
                    values,
                    request=request,
                    model_key=args.model,
                    method_id=method["id"],
                )
            )
        exposure_path = root / "candidate_exposure.npz"
        candidates_path = root / "candidates.csv"
        continuous_io.write_npz_atomic(
            exposure_path, {"candidate_exposure": exposure.astype(bool, copy=False)}
        )
        base.write_csv_atomic(candidates_path, rows)
        rss = continuous_io.maximum_resident_set_bytes()
        if rss > int(source_protocol["execution"]["maximum_process_rss_bytes"]):
            raise MemoryError(f"RSS guard exceeded: {rss}")
        summary = {
            "protocol_sha256": digest,
            "station_day_id": identity,
            "date_utc": request["date_utc"],
            "station_key": continuous_io.station_key(request),
            "input_identity": input_identity,
            "status": "complete",
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "model_key": args.model,
            "loaded_state_dict_sha256": loaded_digest,
            "candidate_count": len(rows),
            "candidate_count_by_method": counts,
            "candidate_exposure_packets": int(exposure.sum()),
            "quality": quality,
            "annotation": annotation.diagnostics,
            "candidate_exposure": base.artifact(exposure_path),
            "candidates": base.artifact(candidates_path),
            "elapsed_seconds": time.perf_counter() - started,
            "maximum_resident_set_bytes": rss,
        }
        continuous_io.write_json_atomic(summary_path, summary)
        print(
            f"{args.model} [{index}/{len(requests)}] {identity}: "
            f"rows={len(rows)}, counts={counts}, elapsed={summary['elapsed_seconds']:.1f}s",
            flush=True,
        )
        processed += 1
        del source, wave, valid, allowed, annotation, exposure, rows
        gc.collect()
        torch.mps.empty_cache()
    if not args.limit:
        finalize(protocol, source_protocol, digest, args.model)


if __name__ == "__main__":
    main()
