#!/usr/bin/env python3
"""Score the event-bearing NCSN continuous cohort with frozen CSPP settings."""
from __future__ import annotations

import argparse
import gc
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/evaluation"))
import build_v27_ncedc_continuous_score_cache as legacy  # noqa: E402
import build_v27_ncedc_continuous_score_cache_hybrid as hybrid  # noqa: E402
from cspp_v27_hybrid_score_cache import (  # noqa: E402
    prepare_hybrid_ensemble,
    release_hybrid_cache,
)
from cspp_v27_reused_core_score_cache import load_ensemble, sha256_file  # noqa: E402


DEFAULT_ROOT = Path(
    "workspace/"
    "ncedc_joint_continuous_m3_20260904"
)
SOURCE_PROTOCOL = (
    ROOT
    / "experiments/cspp_v27_ncedc_continuous_scoring_hybrid_amendment_20260817.json"
)
SOURCE_EQUIVALENCE = Path(
    "workspace/ncedc_model_blind_20260817/"
    "continuous_hybrid_equivalence_report.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
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


def freeze_protocol(root: Path) -> Path:
    destination = root / "scoring_protocol.json"
    if destination.exists():
        return destination
    cohort = json.loads((root / "joint_continuous_protocol.json").read_text())
    acquisition = json.loads((root / "waveforms/summary.json").read_text())
    if acquisition["completed_station_days"] != cohort["counts"]["station_day_requests"]:
        raise RuntimeError("Acquisition is incomplete")
    source = json.loads(SOURCE_PROTOCOL.read_text())
    equivalence = json.loads(SOURCE_EQUIVALENCE.read_text())
    if equivalence.get("decision") != "pass":
        raise RuntimeError("Frozen hybrid implementation equivalence did not pass")
    payload = {
        "protocol_id": "cspp_v27_ncedc_joint_continuous_scoring_v1",
        "frozen_at_utc": datetime.now(UTC).isoformat(),
        "status": "frozen_before_event-bearing continuous model scoring",
        "cohort_protocol": artifact(root / "joint_continuous_protocol.json"),
        "inputs": {
            "station_day_requests": cohort["artifacts"]["station_day_requests"],
            "station_day_orientations": cohort["artifacts"]["station_day_orientations"],
            "acquisition_status": artifact(root / "waveforms/latest_status.csv"),
            "acquisition_summary": artifact(root / "waveforms/summary.json"),
        },
        "models": source["models"],
        "operating_point": source["operating_point"],
        "runtime": source["runtime"],
        "execution": {
            "output_root": str(root / "scores"),
            "backend": "single_process_mps_packet_encoder_cpu_gru_and_heads",
            "torch_threads_per_process": 3,
            "maximum_process_rss_bytes": 6442450944,
        },
        "expected_units": {
            "requested_station_days": acquisition["requested_station_days"],
            "data_bearing_station_days": acquisition["completed_station_days"]
            - acquisition["status_counts"].get("download_or_parse_error", 0),
        },
        "implementation_equivalence": artifact(SOURCE_EQUIVALENCE),
        "source_hybrid_protocol": artifact(SOURCE_PROTOCOL),
        "code_sha256": {
            "scripts/evaluation/score_v27_ncedc_joint_continuous.py": sha256_file(
                Path(__file__).resolve()
            ),
            "scripts/evaluation/build_v27_ncedc_continuous_score_cache_hybrid.py": sha256_file(
                ROOT / "scripts/evaluation/build_v27_ncedc_continuous_score_cache_hybrid.py"
            ),
            "src/cspp_v27_hybrid_score_cache.py": sha256_file(
                ROOT / "src/cspp_v27_hybrid_score_cache.py"
            ),
        },
        "scientific_invariants": {
            "model_weights_changed": False,
            "thresholds_changed": False,
            "packet_clock_changed": False,
            "preprocessing_changed": False,
        },
    }
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return destination


def load_protocol(path: Path) -> tuple[dict[str, Any], str]:
    protocol = json.loads(path.read_text())
    for relative, expected in protocol["code_sha256"].items():
        source = ROOT / relative
        if sha256_file(source) != expected:
            raise RuntimeError(f"Scoring code drifted: {relative}")
    for spec in protocol["inputs"].values():
        legacy.verify_artifact(spec)
    for spec in protocol["models"].values():
        legacy.verify_artifact(spec)
    return protocol, sha256_file(path)


def finalize(
    protocol: dict[str, Any], protocol_sha256: str, requests: list[dict[str, str]]
) -> None:
    acquisitions = {
        row["station_day_id"]: row
        for row in legacy.read_csv(
            legacy.verify_artifact(protocol["inputs"]["acquisition_status"])
        )
    }
    specs = []
    quality_days = eligible_days = 0.0
    for request in requests:
        identity = legacy.station_day_id(request)
        if acquisitions[identity]["status"] == "download_or_parse_error":
            continue
        path = legacy.unit_root(Path(protocol["execution"]["output_root"]), request) / "summary.json"
        summary = json.loads(path.read_text())
        if summary.get("protocol_sha256") != protocol_sha256:
            raise RuntimeError(f"Score summary differs: {path}")
        legacy.verify_artifact(summary["score_cache"])
        quality_days += float(summary["duration"]["quality_valid_station_days"])
        eligible_days += float(summary["duration"]["candidate_eligible_station_days"])
        specs.append(artifact(path))
    expected = int(protocol["expected_units"]["data_bearing_station_days"])
    if len(specs) != expected:
        raise RuntimeError(f"Expected {expected} summaries, got {len(specs)}")
    legacy.write_json_atomic(
        Path(protocol["execution"]["output_root"]) / "score_cache_complete.json",
        {
            "completion_id": "cspp_v27_ncedc_joint_continuous_score_cache_v1",
            "protocol_sha256": protocol_sha256,
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "data_bearing_station_days": len(specs),
            "quality_valid_station_days": quality_days,
            "candidate_eligible_station_days": eligible_days,
            "unit_summaries": specs,
        },
    )


def main() -> None:
    args = parse_args()
    protocol_path = freeze_protocol(args.root)
    if args.freeze_only:
        print(json.dumps(artifact(protocol_path), indent=2))
        return
    protocol, protocol_sha256 = load_protocol(protocol_path)
    requests = legacy.read_csv(
        legacy.verify_artifact(protocol["inputs"]["station_day_requests"])
    )
    requests.sort(key=lambda row: (row["date_utc"], int(row["availability_rank"])))
    if args.finalize_only:
        finalize(protocol, protocol_sha256, requests)
        return
    acquisitions = {
        row["station_day_id"]: row
        for row in legacy.read_csv(
            legacy.verify_artifact(protocol["inputs"]["acquisition_status"])
        )
    }
    orientations = legacy.orientation_groups(
        legacy.read_csv(
            legacy.verify_artifact(protocol["inputs"]["station_day_orientations"])
        )
    )
    ensemble = load_ensemble(
        {
            key: {
                "path": spec["path"],
                "sha256": spec["sha256"],
                "verifier_feature_mode": spec["verifier_feature_mode"],
            }
            for key, spec in protocol["models"].items()
        },
        device_name="cpu",
        torch_threads=int(protocol["execution"]["torch_threads_per_process"]),
    )
    prepare_hybrid_ensemble(ensemble)
    output = Path(protocol["execution"]["output_root"])
    status_path = output / "status.jsonl"
    processed = 0
    for index, request in enumerate(requests, start=1):
        acquisition = acquisitions[legacy.station_day_id(request)]
        if acquisition["status"] == "download_or_parse_error":
            continue
        if args.limit and processed >= args.limit:
            break
        identity = legacy.station_day_id(request)
        input_identity = legacy.stable_hash(
            {
                "request": request,
                "acquisition": acquisition,
                "qc_sha256": sha256_file(Path(acquisition["qc_file"])),
                "orientation_sha256": protocol["inputs"]["station_day_orientations"]["sha256"],
            }
        )
        root = legacy.unit_root(output, request)
        summary_path = root / "summary.json"
        if legacy.summary_valid(
            summary_path,
            protocol_sha256=protocol_sha256,
            identity=identity,
            input_identity=input_identity,
        ):
            continue
        started = time.perf_counter()
        arrays, metadata, duration = hybrid.score_unit(
            request=request,
            acquisition=acquisition,
            orientation_rows=orientations[identity],
            ensemble=ensemble,
            protocol=protocol,
        )
        rss = legacy.maximum_resident_set_bytes()
        if rss > int(protocol["execution"]["maximum_process_rss_bytes"]):
            raise MemoryError(f"RSS guard exceeded: {rss}")
        cache_path = root / "scores.npz"
        legacy.write_npz_atomic(cache_path, arrays)
        summary = {
            "protocol_sha256": protocol_sha256,
            "station_day_id": identity,
            "date_utc": request["date_utc"],
            "station_key": legacy.station_key(request),
            "input_identity": input_identity,
            "status": "complete",
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "acquisition_status": acquisition["status"],
            "duration": duration,
            "metadata": metadata,
            "score_cache": artifact(cache_path),
            "elapsed_seconds": time.perf_counter() - started,
            "maximum_resident_set_bytes": rss,
        }
        legacy.write_json_atomic(summary_path, summary)
        status_path.parent.mkdir(parents=True, exist_ok=True)
        with status_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"station_day_id": identity, "summary": artifact(summary_path)}) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        print(
            f"[{index}/{len(requests)}] {identity}: {summary['elapsed_seconds']:.1f}s, "
            f"eligible={duration['candidate_eligible_station_days']:.4f} d",
            flush=True,
        )
        processed += 1
        del arrays, metadata
        gc.collect()
        release_hybrid_cache()
    if not args.limit:
        finalize(protocol, protocol_sha256, requests)


if __name__ == "__main__":
    main()
