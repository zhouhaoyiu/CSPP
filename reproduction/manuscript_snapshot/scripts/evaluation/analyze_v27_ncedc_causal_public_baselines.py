#!/usr/bin/env python3
"""Aggregate the frozen NCSN causal public-baseline replay."""
from __future__ import annotations

from collections import defaultdict
import csv
from datetime import UTC, datetime
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from cspp_v27_reused_core_score_cache import sha256_file  # noqa: E402


DEFAULT_PROTOCOL = (
    ROOT
    / "experiments/cspp_v27_ncedc_causal_public_baseline_protocol_20260817.json"
)


def verify(spec: dict[str, Any]) -> Path:
    path = Path(spec["path"])
    if (
        not path.is_file()
        or path.stat().st_size != int(spec["bytes"])
        or sha256_file(path) != spec["sha256"]
    ):
        raise RuntimeError(f"Artifact drifted: {path}")
    return path


def read_rows(protocol: dict[str, Any], protocol_sha: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for model_key in sorted(protocol["models"]):
        completion_path = (
            Path(protocol["execution"]["output_root"])
            / model_key
            / "completion.json"
        )
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        if completion.get("protocol_sha256") != protocol_sha:
            raise RuntimeError(f"Completion protocol differs: {model_key}")
        for summary_spec in completion["event_summaries"]:
            summary = json.loads(verify(summary_spec).read_text(encoding="utf-8"))
            path = verify(summary["records"])
            with path.open(encoding="utf-8", newline="") as handle:
                rows.extend(csv.DictReader(handle))
    return rows


def wilson(successes: int, total: int) -> list[float]:
    if total <= 0:
        return [0.0, 0.0]
    z = 1.959963984540054
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
    half /= denominator
    return [max(0.0, center - half), min(1.0, center + half)]


def metrics(rows: list[dict[str, str]]) -> dict[str, Any]:
    valid = [row for row in rows if not row["technical_error"]]
    records = len(rows)
    technical = records - len(valid)
    matched = sum(int(row["matched_candidate_count"]) for row in valid)
    candidates = sum(int(row["candidate_count"]) for row in valid)
    recall = matched / records if records else 0.0
    precision = matched / candidates if candidates else 0.0
    f1 = 2 * recall * precision / (recall + precision) if recall + precision else 0.0
    errors = [
        float(row["absolute_pick_error_seconds"])
        for row in valid
        if row["absolute_pick_error_seconds"]
    ]
    delays = [
        float(row["availability_delay_seconds"])
        for row in valid
        if row["availability_delay_seconds"]
    ]
    return {
        "records": records,
        "technical_failures_counted_as_misses": technical,
        "matched_records": matched,
        "candidate_episodes": candidates,
        "recall": recall,
        "recall_wilson95": wilson(matched, records),
        "candidate_precision": precision,
        "candidate_precision_wilson95": wilson(matched, candidates),
        "f1": f1,
        "absolute_pick_error_median_seconds": (
            float(np.median(errors)) if errors else None
        ),
        "absolute_pick_error_p95_seconds": (
            float(np.quantile(errors, 0.95)) if errors else None
        ),
        "availability_delay_median_seconds": (
            float(np.median(delays)) if delays else None
        ),
        "availability_delay_p95_seconds": (
            float(np.quantile(delays, 0.95)) if delays else None
        ),
        "available_within_2_seconds_fraction": (
            sum(value <= 2.0 for value in delays) / records if records else 0.0
        ),
    }


def bootstrap(
    rows: list[dict[str, str]], replicates: int, seed: int
) -> dict[str, list[float]]:
    by_event: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_event[row["event_id"]].append(row)
    events = sorted(by_event)
    rng = np.random.default_rng(seed)
    values = {name: np.empty(replicates) for name in ("recall", "candidate_precision", "f1")}
    for index in range(replicates):
        sampled = rng.choice(events, size=len(events), replace=True)
        result = metrics([row for event in sampled for row in by_event[str(event)]])
        for name in values:
            values[name][index] = result[name]
    return {
        name: [float(np.quantile(value, 0.025)), float(np.quantile(value, 0.975))]
        for name, value in values.items()
    }


def main() -> None:
    expected = os.environ.get("CSPP_NCEDC_PUBLIC_BASELINE_SHA256", "")
    actual = sha256_file(DEFAULT_PROTOCOL)
    if not expected or actual != expected:
        raise RuntimeError(f"Set the frozen public protocol SHA-256; actual={actual}")
    protocol = json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))
    for relative, digest in protocol["code_sha256"].items():
        path = ROOT / relative
        if not path.is_file() or sha256_file(path) != digest:
            raise RuntimeError(f"Public analysis code drifted: {relative}")
    rows = read_rows(protocol, actual)
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["model_key"], row["method_id"])].append(row)
    results = {}
    for (model_key, method_id), method_rows in sorted(grouped.items()):
        result = metrics(method_rows)
        result["event_cluster_bootstrap95"] = bootstrap(
            method_rows,
            int(protocol["statistics"]["bootstrap_replicates"]),
            int(protocol["statistics"]["bootstrap_seed"]),
        )
        results[f"{model_key}:{method_id}"] = result
    output_root = Path(protocol["execution"]["analysis_output_root"])
    output_root.mkdir(parents=True, exist_ok=False)
    payload = {
        "result_id": "cspp_v27_ncedc_causal_public_analysis_v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "protocol_sha256": actual,
        "results": results,
        "interpretation": {
            "scope": (
                "same records and causal visibility; released checkpoints retain "
                "their native windows, preprocessing, thresholds, and pick semantics"
            ),
            "training": "released-model comparison, not matched retraining",
            "phasenet_checkpoint": "PhaseNet weights trained on INSTANCE",
            "eqtransformer_checkpoint": "original conservative EQTransformer weights",
        },
    }
    path = output_root / "result.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
