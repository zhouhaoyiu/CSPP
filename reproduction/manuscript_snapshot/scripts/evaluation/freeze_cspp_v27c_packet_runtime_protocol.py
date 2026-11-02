#!/usr/bin/env python3
"""Freeze the single-thread CPU packet-runtime benchmark for selected V27C."""
from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.evaluation import (  # noqa: E402
    freeze_cspp_v27c_fixed_retrospective_protocol as retrospective,
)


WORK_ROOT = Path("workspace").resolve()
PROTOCOL_ID = "cspp_v27c_packet_runtime_benchmark_v1"
DEFAULT_SELECTION_FREEZE = retrospective.DEFAULT_SELECTION_FREEZE
DEFAULT_OUTPUT = (
    WORK_ROOT
    / "protocol_freeze/cspp_v27c_packet_runtime_protocol_20260729.json"
)
DEFAULT_RESULT = (
    WORK_ROOT
    / "evaluation/cspp_v27c_packet_runtime_20260729/result.json"
)
CODE_PATHS = (
    "scripts/evaluation/freeze_cspp_v27c_packet_runtime_protocol.py",
    "scripts/evaluation/freeze_cspp_v27c_fixed_retrospective_protocol.py",
    "scripts/evaluation/benchmark_cspp_v27c_packet_runtime.py",
    "src/cspp_v27_packet_candidate.py",
    "src/config.py",
    "src/data_streaming.py",
    "src/model_cspp_v25.py",
    "src/model_cspp_v27.py",
    "src/model_domain_robust_v5.py",
    "src/model_domain_robust_v8.py",
    "src/model_domain_robust_v9.py",
    "src/stream_quality_v5.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selection-freeze",
        type=Path,
        default=DEFAULT_SELECTION_FREEZE,
    )
    parser.add_argument("--expected-selection-sha256", required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    return parser.parse_args()


def require_output(path: str | Path) -> Path:
    return retrospective.require_output_path(
        path, allowed_root=WORK_ROOT
    )


def build_protocol(
    *,
    selection_freeze_path: Path,
    expected_selection_sha256: str,
    output_path: Path,
    result_path: Path,
) -> dict[str, Any]:
    output = require_output(output_path)
    result = require_output(result_path)
    if output.exists():
        raise FileExistsError(output)
    if result.exists() or result.parent.exists():
        raise FileExistsError(
            "Runtime result directory must not exist at protocol freeze"
        )
    selection_path = retrospective.reject_forbidden_path(
        selection_freeze_path
    )
    if retrospective.sha256_file(selection_path) != (
        expected_selection_sha256.lower()
    ):
        raise RuntimeError("Selection-freeze SHA-256 differs")
    selection = retrospective.load_json(selection_path)
    if selection.get("protocol_id") != retrospective.SELECTION_FREEZE_ID:
        raise RuntimeError("Unexpected selection-freeze identifier")
    selected = retrospective.resolve_selected_model(selection)
    for relative in CODE_PATHS:
        if not (ROOT / relative).is_file():
            raise FileNotFoundError(ROOT / relative)
    return {
        "protocol_id": PROTOCOL_ID,
        "frozen_at_utc": datetime.now(UTC).isoformat(),
        "selection_freeze": retrospective.artifact(selection_path),
        "model": {
            "model_class": "CSPPCausalPickerV27C",
            "selected_arm": selected["selected_arm"],
            "primary_seed": selected["primary_seed"],
            "checkpoint": selected["checkpoint"],
            "verifier_feature_mode": selected["checkpoint"][
                "verifier_feature_mode"
            ],
        },
        "operating_point": {
            "candidate_threshold": selected["candidate_threshold"],
            "event_threshold": selected["event_threshold"],
        },
        "measurement": {
            "device": "cpu",
            "torch_intraop_threads": 1,
            "torch_interop_threads": 1,
            "sample_rate_hz": 100.0,
            "packet_samples": 50,
            "packet_duration_seconds": 0.5,
            "candidate_history_packets": 4,
            "max_arrival_age_packets": 4.0,
            "quality_warmup_packets": 20,
            "timing_warmup_packets": 500,
            "timed_packets": 5000,
            "synthetic_seed": 20260729,
            "timed_path": (
                "quality gate, causal EWMA normalization, NumPy-to-Torch "
                "conversion, one stateful V27C forward pass, sigmoid, "
                "four-packet candidate support, event threshold, activity-"
                "episode suppression, and arrival-age P-time conversion"
            ),
        },
        "acceptance_gates": {
            "p95_end_to_end_packet_ms_max": 25.0,
            "model_parameter_count_max": 400_000,
            "model_tensor_bytes_max": 2 * 1024 * 1024,
            "bounded_station_state_bytes_max": 16 * 1024,
            "real_time_factor_at_p95_min": 20.0,
        },
        "execution": {
            "result_path": str(result),
            "allowed_output_root": str(WORK_ROOT),
            "forbidden_volume": str(retrospective.FORBIDDEN_VOLUME),
            "overwrite_permitted": False,
        },
        "code_sha256": {
            relative: retrospective.sha256_file(ROOT / relative)
            for relative in CODE_PATHS
        },
        "claim_boundary": {
            "role": (
                "Engineering runtime and memory evidence; it does not "
                "measure seismic accuracy."
            ),
            "hardware": (
                "Results describe this CPU and software environment. They "
                "are not a universal embedded-instrument benchmark."
            ),
            "input": (
                "The deterministic synthetic stream exercises the complete "
                "station packet path but is not earthquake evidence."
            ),
        },
    }


def write_json_once(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    payload = build_protocol(
        selection_freeze_path=args.selection_freeze,
        expected_selection_sha256=args.expected_selection_sha256,
        output_path=args.output,
        result_path=args.result,
    )
    output = require_output(args.output)
    write_json_once(output, payload)
    print(
        json.dumps(
            retrospective.artifact(output),
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
