#!/usr/bin/env python3
"""Explore CSPP confirmation-threshold tradeoffs on the opened NCSN cohort."""
from __future__ import annotations

import csv
from datetime import UTC, datetime
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from obspy.taup import TauPyModel


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/evaluation"))

import adjudicate_v27_ncedc_continuous_catalog as catalog_match  # noqa: E402
import analyze_v27_ncedc_continuous_scores as continuous  # noqa: E402
import analyze_v27_ncedc_event_scores as event_metrics  # noqa: E402
import analyze_v27_ncedc_joint_continuous as joint  # noqa: E402
import build_v27_ncedc_continuous_score_cache as score_io  # noqa: E402
from cspp_v27_reused_core_score_cache import sha256_file  # noqa: E402


DATA_ROOT = Path(
    "workspace/"
    "ncedc_joint_continuous_m3_20260904"
)
THRESHOLDS = (0.975, 0.95, 0.90, 0.80, 0.70, 0.50)


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


def freeze_protocol() -> Path:
    path = DATA_ROOT / "threshold_sensitivity_protocol.json"
    if path.exists():
        return path
    scoring = json.loads((DATA_ROOT / "scoring_protocol.json").read_text())
    payload = {
        "protocol_id": "cspp_v27_ncedc_joint_threshold_sensitivity_v1",
        "frozen_at_utc": datetime.now(UTC).isoformat(),
        "status": "exploratory protocol frozen after primary NCSN result was opened",
        "inputs": {
            "scoring_protocol": artifact(DATA_ROOT / "scoring_protocol.json"),
            "score_completion": artifact(
                DATA_ROOT / "scores/score_cache_complete.json"
            ),
            "opened_primary_result": artifact(DATA_ROOT / "analysis/result.json"),
        },
        "fixed": {
            "model_seed": int(
                scoring["operating_point"]["primary_operational_seed"]
            ),
            "candidate_threshold": float(
                scoring["operating_point"]["candidate_threshold"]
            ),
            "association_minimum_stations": 3,
            "association_window_seconds": 10.0,
        },
        "confirmation_thresholds": list(THRESHOLDS),
        "claim_boundary": {
            "confirmatory_test_claim_permitted": False,
            "threshold_selection_on_this_cohort_permitted": False,
            "purpose": "describe recall, candidate burden, latency, and unmatched-group tradeoffs",
        },
        "code_sha256": {
            "scripts/evaluation/analyze_v27_ncedc_joint_threshold_sensitivity.py": sha256_file(
                Path(__file__).resolve()
            ),
            "scripts/evaluation/analyze_v27_ncedc_joint_continuous.py": sha256_file(
                ROOT / "scripts/evaluation/analyze_v27_ncedc_joint_continuous.py"
            ),
            "scripts/evaluation/analyze_v27_ncedc_continuous_scores.py": sha256_file(
                ROOT / "scripts/evaluation/analyze_v27_ncedc_continuous_scores.py"
            ),
        },
    }
    score_io.write_json_atomic(path, payload)
    return path


def load_protocol(path: Path) -> dict[str, Any]:
    protocol = json.loads(path.read_text())
    for relative, expected in protocol["code_sha256"].items():
        if sha256_file(ROOT / relative) != expected:
            raise RuntimeError(f"Sensitivity code drifted: {relative}")
    for spec in protocol["inputs"].values():
        verify_artifact(spec)
    return protocol


def metric(events: list[dict[str, Any]], minimum_magnitude: float) -> dict[str, Any]:
    rows = [
        row
        for row in events
        if row["eligible"] and float(row["magnitude"]) >= minimum_magnitude
    ]
    detected = sum(bool(row["detected"]) for row in rows)
    return {
        "eligible_events": len(rows),
        "detected_events": detected,
        "missed_events": len(rows) - detected,
        "recall": detected / len(rows) if rows else None,
        "recall_wilson95": (
            event_metrics.wilson(detected, len(rows)) if rows else [None, None]
        ),
    }


def evaluate_threshold(
    threshold: float,
    *,
    scoring: dict[str, Any],
    summaries: list[dict[str, Any]],
    coordinates: dict[str, tuple[float, float]],
    regional: list[dict[str, Any]],
    global_events: list[dict[str, Any]],
    taup: TauPyModel,
    base_targets: list[dict[str, Any]],
    predicted: dict[tuple[str, str], float],
    candidate_exposure: float,
    network_exposure: float,
) -> dict[str, Any]:
    operating_point = dict(scoring["operating_point"])
    operating_point["event_threshold"] = threshold
    protocol = {
        "operating_point": operating_point,
        "runtime": scoring["runtime"],
    }
    candidates, decision_chain = continuous.extract_seed_candidates(
        summaries=summaries,
        seed_key="C0_seed20261101",
        seed=20261101,
        protocol=protocol,
        coordinates=coordinates,
        keep_details=True,
    )
    groups = continuous.network_groups(candidates, 10.0, 3)
    classified = joint.classify_groups(
        groups, candidates, regional, global_events, taup
    )
    detected_groups: dict[str, dict[str, Any]] = {}
    for group in classified:
        if group["catalog_class"] != "catalog_supported":
            continue
        event_id = group["catalog_event_id"]
        previous = detected_groups.get(event_id)
        if previous is None or float(group["available_epoch"]) < float(
            previous["available_epoch"]
        ):
            detected_groups[event_id] = group
    targets = []
    latencies = []
    for source in base_targets:
        event = dict(source)
        group = detected_groups.get(event["event_id"])
        event["detected"] = group is not None
        if group is not None:
            arrivals = [
                predicted[(event["event_id"], station)]
                for station in event["eligible_stations"].split(";")
                if (event["event_id"], station) in predicted
            ]
            if arrivals:
                latencies.append(float(group["available_epoch"]) - min(arrivals))
        targets.append(event)
    unmatched = sum(
        group["catalog_class"] == "unmatched" for group in classified
    )
    supported = len(classified) - unmatched
    return {
        "confirmation_threshold": threshold,
        "m3": metric(targets, 3.0),
        "m4": metric(targets, 4.0),
        "candidate_count": len(candidates),
        "candidate_rate_per_station_day": len(candidates) / candidate_exposure,
        "network_group_count": len(classified),
        "catalog_supported_network_groups": supported,
        "unmatched_network_groups": unmatched,
        "catalog_supported_group_fraction": (
            supported / len(classified) if classified else None
        ),
        "unmatched_group_rate_per_network_day": unmatched / network_exposure,
        "latency_count": len(latencies),
        "latency_median_seconds": (
            float(np.median(latencies)) if latencies else None
        ),
        "latency_p95_seconds": (
            float(np.quantile(latencies, 0.95)) if latencies else None
        ),
        "decision_chain_counts": decision_chain,
    }


def write_table(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "confirmation_threshold",
        "m3_recall",
        "m4_recall",
        "candidate_rate_per_station_day",
        "network_group_count",
        "catalog_supported_network_groups",
        "unmatched_network_groups",
        "catalog_supported_group_fraction",
        "unmatched_group_rate_per_network_day",
        "latency_median_seconds",
        "latency_p95_seconds",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **{key: row[key] for key in fields if key in row},
                    "m3_recall": row["m3"]["recall"],
                    "m4_recall": row["m4"]["recall"],
                }
            )


def main() -> None:
    protocol_path = freeze_protocol()
    protocol = load_protocol(protocol_path)
    output = DATA_ROOT / "analysis_threshold_sensitivity"
    if output.exists():
        raise FileExistsError(output)
    output.mkdir()
    scoring = json.loads(
        verify_artifact(protocol["inputs"]["scoring_protocol"]).read_text()
    )
    completion = json.loads(
        verify_artifact(protocol["inputs"]["score_completion"]).read_text()
    )
    summaries = [
        json.loads(verify_artifact(spec).read_text())
        for spec in completion["unit_summaries"]
    ]
    requests = score_io.read_csv(
        Path(scoring["inputs"]["station_day_requests"]["path"])
    )
    coordinates = joint.station_coordinates(requests)
    selected_dates = {
        row["date_utc"]
        for row in score_io.read_csv(DATA_ROOT / "metadata/selected_event_dates.csv")
    }
    regional = joint.event_rows(sorted((DATA_ROOT / "catalogs/regional").glob("*.xml")))
    global_events = joint.event_rows(
        sorted((DATA_ROOT / "catalogs/global_m45").glob("*.xml"))
    )
    taup = TauPyModel(model="iasp91")
    base_targets, predicted = joint.eligible_events(
        joint.target_events(selected_dates, coordinates),
        summaries,
        coordinates,
        taup,
    )
    primary = json.loads(
        verify_artifact(protocol["inputs"]["opened_primary_result"]).read_text()
    )
    candidate_exposure = sum(
        float(summary["duration"]["candidate_eligible_station_days"])
        for summary in summaries
    )
    network_exposure = float(
        primary["continuous_metrics"]["network_exposure_days"]
    )
    rows = []
    for threshold in protocol["confirmation_thresholds"]:
        result = evaluate_threshold(
            float(threshold),
            scoring=scoring,
            summaries=summaries,
            coordinates=coordinates,
            regional=regional,
            global_events=global_events,
            taup=taup,
            base_targets=base_targets,
            predicted=predicted,
            candidate_exposure=candidate_exposure,
            network_exposure=network_exposure,
        )
        rows.append(result)
        print(
            f"threshold={threshold:.3f} m3={result['m3']['recall']:.3f} "
            f"m4={result['m4']['recall']:.3f} "
            f"candidate_rate={result['candidate_rate_per_station_day']:.3f} "
            f"unmatched={result['unmatched_network_groups']}",
            flush=True,
        )
    write_table(output / "tradeoff.csv", rows)
    score_io.write_json_atomic(
        output / "result.json",
        {
            "result_id": "cspp_v27_ncedc_joint_threshold_sensitivity_v1",
            "created_at_utc": datetime.now(UTC).isoformat(),
            "protocol": artifact(protocol_path),
            "results": rows,
            "claim_boundary": protocol["claim_boundary"],
        },
    )


if __name__ == "__main__":
    main()
