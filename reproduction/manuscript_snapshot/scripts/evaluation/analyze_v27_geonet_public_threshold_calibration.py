#!/usr/bin/env python3
"""Select public-picker operating points from frozen GeoNet development outputs."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.evaluation import eval_v27_geonet_causal_public_baselines as common  # noqa: E402
from scripts.evaluation import eval_v27_geonet_public_threshold_calibration as evaluator  # noqa: E402


DEFAULT_PROTOCOL = (
    ROOT / "experiments/cspp_v27_geonet_public_threshold_calibration_20260824.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--expected-protocol-sha256", required=True)
    return parser.parse_args()


def metrics(rows: list[dict[str, str]], threshold: float) -> dict[str, Any]:
    total = len(rows)
    matched = sum(int(row["matched_candidate_count"]) for row in rows)
    candidates = sum(int(row["candidate_count"]) for row in rows)
    recall = matched / total if total else 0.0
    precision = matched / candidates if candidates else 0.0
    f1 = 2.0 * recall * precision / (recall + precision) if recall + precision else 0.0
    residuals = [float(row["absolute_pick_error_seconds"]) for row in rows if row["absolute_pick_error_seconds"]]
    delays = [float(row["availability_delay_seconds"]) for row in rows if row["availability_delay_seconds"]]
    return {
        "method_id": rows[0]["method_id"],
        "p_threshold": threshold,
        "records": total,
        "matched_records": matched,
        "candidate_count": candidates,
        "recall": recall,
        "candidate_precision": precision,
        "f1": f1,
        "median_absolute_pick_error_seconds": statistics.median(residuals) if residuals else None,
        "median_availability_delay_seconds": statistics.median(delays) if delays else None,
        "technical_failure_records": sum(bool(row["technical_error"]) for row in rows),
    }


def main() -> None:
    args = parse_args()
    protocol, protocol_sha256 = evaluator.load_protocol(
        args.protocol.resolve(), args.expected_protocol_sha256
    )
    output_root = Path(protocol["execution"]["output_root"])
    curve_rows = []
    selected = {}
    for model_key, model_spec in protocol["public_models"].items():
        by_method: dict[str, list[dict[str, str]]] = defaultdict(list)
        for event_id in protocol["cohort"]["event_ids"]:
            summary_path = output_root / model_key / "events" / event_id / "summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if summary["protocol_sha256"] != protocol_sha256:
                raise RuntimeError(f"Event output protocol differs: {summary_path}")
            records_path = evaluator.verify_artifact(summary["records_artifact"])
            with records_path.open(encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    by_method[row["method_id"]].append(row)
        thresholds = {
            point["id"]: float(point["p_threshold"])
            for point in model_spec["operating_points"]
        }
        model_curve = [metrics(rows, thresholds[method]) for method, rows in by_method.items()]
        model_curve.sort(key=lambda row: row["p_threshold"])
        for row in model_curve:
            curve_rows.append({"model_key": model_key, **row})
        choice = max(
            model_curve,
            key=lambda row: (
                row["f1"],
                row["candidate_precision"],
                row["recall"],
                row["p_threshold"],
            ),
        )
        selected[model_key] = choice
    curve_path = output_root / "threshold_curve.csv"
    with curve_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(curve_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(curve_rows)
    selected_path = output_root / "selected_operating_points.json"
    common.write_json_atomic(
        selected_path,
        {
            "selection_source": "GeoNet 2024-2025 threshold_calibration only",
            "selection_rule": protocol["evidence_boundary"]["selection_rule"],
            "ncsn_used": False,
            "selected": selected,
        },
    )
    common.write_json_atomic(
        output_root / "result.json",
        {
            "result_id": "cspp_v27_geonet_public_threshold_calibration_v1",
            "protocol_sha256": protocol_sha256,
            "cohort": protocol["cohort"],
            "selected": selected,
            "artifacts": {
                "threshold_curve": {
                    "path": str(curve_path),
                    "bytes": curve_path.stat().st_size,
                    "sha256": common.sha256(curve_path),
                },
                "selected_operating_points": {
                    "path": str(selected_path),
                    "bytes": selected_path.stat().st_size,
                    "sha256": common.sha256(selected_path),
                },
            },
            "claim_boundary": (
                "These development-selected thresholds do not constitute matched "
                "retraining and were not selected on NCSN or GeoNet 2026."
            ),
        },
    )
    print(json.dumps(selected, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
