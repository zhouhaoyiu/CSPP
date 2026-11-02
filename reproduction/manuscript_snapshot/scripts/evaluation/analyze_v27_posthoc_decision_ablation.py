#!/usr/bin/env python3
"""Analyze frozen V27 decision-chain ablations without rescoring waveforms."""
from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
EVALUATION_DIR = ROOT / "scripts/evaluation"
sys.path.insert(0, str(EVALUATION_DIR))

import analyze_v27_geonet_final_evaluation as final_analysis  # noqa: E402


PROTOCOL_ID = "cspp_v27_posthoc_decision_ablation_v1"
DEFAULT_PROTOCOL = (
    ROOT
    / "experiments/cspp_v27_posthoc_decision_ablation_protocol_20260730.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_artifact(spec: dict[str, Any]) -> Path:
    path = Path(spec["path"]).resolve()
    if (
        not path.is_file()
        or path.stat().st_size != int(spec["bytes"])
        or sha256(path) != str(spec["sha256"])
    ):
        raise RuntimeError(f"Frozen artifact drifted: {path}")
    return path


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"Refusing to write empty table: {path}")
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def finite_median(values: Iterable[Any]) -> float | None:
    finite = [
        float(value)
        for value in values
        if value not in (None, "") and math.isfinite(float(value))
    ]
    return float(statistics.median(finite)) if finite else None


def arm_summary(
    event_rows: list[dict[str, Any]],
    background_rows: list[dict[str, Any]],
    network_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    primary_events = [
        row
        for row in event_rows
        if int(row["primary_seed"]) == 1
        and row["label_stratum"]
        == "authoritative_used_p_positive_weight"
    ]
    primary_background = [
        row for row in background_rows if int(row["primary_seed"]) == 1
    ]
    primary_network = [
        row for row in network_rows if int(row["primary_seed"]) == 1
    ]
    records = len(primary_events)
    matched = sum(
        int(row["matched_candidate_count"]) for row in primary_events
    )
    candidates = sum(int(row["candidate_count"]) for row in primary_events)
    exposure = sum(
        float(row["station_exposure_days"]) for row in primary_background
    )
    background_candidates = sum(
        int(row["candidate_count"]) for row in primary_background
    )
    network_exposure = sum(
        float(row["minimum_three_station_exposure_days"])
        for row in primary_network
    )
    confirmations = sum(
        int(row["network_confirmations"]) for row in primary_network
    )
    recall = matched / records if records else 0.0
    precision = matched / candidates if candidates else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "primary_station_records": records,
        "matched_station_records": matched,
        "event_record_candidates": candidates,
        "event_record_false_candidates": candidates - matched,
        "station_record_recall": recall,
        "candidate_precision_on_positive_reference_records": precision,
        "station_record_f1": f1,
        "median_absolute_pick_error_seconds": finite_median(
            row["absolute_pick_error_seconds"] for row in primary_events
        ),
        "median_availability_delay_seconds": finite_median(
            row["availability_delay_seconds"] for row in primary_events
        ),
        "background_station_exposure_days": exposure,
        "background_candidates": background_candidates,
        "background_candidate_rate_per_station_day": (
            background_candidates / exposure if exposure else None
        ),
        "minimum_three_station_exposure_days": network_exposure,
        "background_three_station_confirmations": confirmations,
        "background_three_station_confirmation_rate_per_day": (
            confirmations / network_exposure if network_exposure else None
        ),
    }


def main() -> None:
    args = parse_args()
    protocol_path = args.protocol.resolve()
    protocol = read_json(protocol_path)
    if protocol.get("protocol_id") != PROTOCOL_ID:
        raise RuntimeError("Unexpected decision-ablation protocol")
    for spec in protocol["code_artifacts"].values():
        verify_artifact(spec)
    final_protocol_path = verify_artifact(
        protocol["inputs"]["final_protocol"]
    )
    verify_artifact(protocol["inputs"]["final_result"])
    verify_artifact(protocol["inputs"]["event_score_completion"])
    verify_artifact(protocol["inputs"]["background_score_completion"])

    output_root = Path(protocol["outputs"]["root"]).resolve()
    if output_root.exists():
        raise FileExistsError(output_root)
    output_root.mkdir(parents=True)

    frozen = read_json(final_protocol_path)
    frozen_hash = sha256(final_protocol_path)
    event_summaries, _ = final_analysis.load_completion(
        frozen, frozen_hash, "event"
    )
    background_summaries, _ = final_analysis.load_completion(
        frozen, frozen_hash, "background"
    )

    all_event_rows: list[dict[str, Any]] = []
    all_background_rows: list[dict[str, Any]] = []
    all_network_rows: list[dict[str, Any]] = []
    summaries = []
    for arm in protocol["arms"]:
        arm_protocol = copy.deepcopy(frozen)
        arm_protocol["runtime"]["candidate_history_packets"] = int(
            arm["candidate_support_memory_packets"]
        )
        arm_protocol["operating_point"]["event_threshold"] = float(
            arm["event_threshold"]
        )
        event_rows = final_analysis.scored_event_rows(
            arm_protocol, event_summaries
        )
        background_rows, network_rows = final_analysis.background_rows(
            arm_protocol, background_summaries
        )
        for rows in (event_rows, background_rows, network_rows):
            for row in rows:
                row["arm_id"] = arm["id"]
        all_event_rows.extend(event_rows)
        all_background_rows.extend(background_rows)
        all_network_rows.extend(network_rows)
        summaries.append(
            {
                "arm_id": arm["id"],
                "change": arm["change"],
                "candidate_support_memory_packets": int(
                    arm["candidate_support_memory_packets"]
                ),
                "event_threshold": float(arm["event_threshold"]),
                **arm_summary(event_rows, background_rows, network_rows),
            }
        )

    reference = summaries[0]
    for row in summaries:
        row["delta_station_record_f1_vs_full"] = (
            float(row["station_record_f1"])
            - float(reference["station_record_f1"])
        )
        row["background_candidate_rate_ratio_vs_full"] = (
            float(row["background_candidate_rate_per_station_day"])
            / float(reference["background_candidate_rate_per_station_day"])
            if reference["background_candidate_rate_per_station_day"]
            else None
        )

    event_path = Path(protocol["outputs"]["event_rows"])
    background_path = Path(protocol["outputs"]["background_station_rows"])
    network_path = Path(protocol["outputs"]["background_network_rows"])
    write_csv(event_path, all_event_rows)
    write_csv(background_path, all_background_rows)
    write_csv(network_path, all_network_rows)
    result = {
        "result_id": "cspp_v27_posthoc_decision_ablation_result_v1",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": sha256(protocol_path),
        "status": "complete_posthoc_paired_decision_chain_ablation",
        "arms": summaries,
        "artifacts": {
            "event_rows": {
                "path": str(event_path),
                "bytes": event_path.stat().st_size,
                "sha256": sha256(event_path),
            },
            "background_station_rows": {
                "path": str(background_path),
                "bytes": background_path.stat().st_size,
                "sha256": sha256(background_path),
            },
            "background_network_rows": {
                "path": str(network_path),
                "bytes": network_path.stat().st_size,
                "sha256": sha256(network_path),
            },
        },
        "evidence_boundary": protocol["evidence_boundary"],
        "primary_final_result_changed": False,
    }
    result_path = Path(protocol["outputs"]["result"])
    write_json(result_path, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
