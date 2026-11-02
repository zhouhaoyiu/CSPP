#!/usr/bin/env python3
"""Serialize the frozen TexNet public-baseline analysis with undefined values as null."""
from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
import math
import os
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/evaluation"))

import eval_v27_tx_causal_public_baselines as analysis  # noqa: E402


OUTPUT = analysis.TX_ROOT / "evaluation/tx_causal_public_analysis_v2"
ANALYSIS_PROTOCOL = analysis.TX_ROOT / "causal_public_analysis_protocol_v2.json"


def finite_or_none(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: finite_or_none(item) for key, item in value.items()}
    if isinstance(value, list):
        return [finite_or_none(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_once(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def json_bytes(payload: Any) -> bytes:
    value = finite_or_none(payload)
    if isinstance(value, dict):
        value["analysis_protocol"] = analysis.artifact(ANALYSIS_PROTOCOL)
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def freeze_analysis(expected_baseline_sha256: str) -> None:
    protocol, actual = analysis.load_protocol(
        analysis.PROTOCOL, expected_baseline_sha256
    )
    if ANALYSIS_PROTOCOL.exists() or OUTPUT.exists():
        raise FileExistsError("TexNet public analysis protocol or v2 output exists")
    completions = {
        key: analysis.artifact(
            Path(protocol["execution"]["output_root"]) / key / "completion.json"
        )
        for key in protocol["models"]
    }
    payload = {
        "protocol_id": "cspp_v27_tx_causal_public_analysis_v2",
        "status": "frozen_before_v2_aggregate_output",
        "frozen_at_utc": datetime.now(UTC).isoformat(),
        "baseline_protocol": analysis.artifact(analysis.PROTOCOL),
        "baseline_protocol_sha256": actual,
        "model_completions": completions,
        "analysis_code": analysis.artifact(Path(__file__)),
        "frozen_analysis_implementation": analysis.artifact(
            Path(analysis.__file__)
        ),
        "change_from_v1": (
            "Only non-finite values caused by undefined zero-denominator confidence "
            "intervals are serialized as JSON null; records, methods, metrics, "
            "thresholds, and bootstrap settings are unchanged."
        ),
        "output_root": str(OUTPUT),
    }
    write_once(ANALYSIS_PROTOCOL, analysis.json_bytes(payload))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-protocol-sha256", required=True)
    args = parser.parse_args()
    assert finite_or_none([float("nan"), 1.0]) == [None, 1.0]
    freeze_analysis(args.expected_protocol_sha256)
    analysis.ANALYSIS_ROOT = OUTPUT
    analysis.json_bytes = json_bytes
    analysis.analyze(args.expected_protocol_sha256)


if __name__ == "__main__":
    main()
