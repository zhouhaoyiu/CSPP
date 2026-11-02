#!/usr/bin/env python3
"""Analyze locked NCSN CSPP event scores without threshold search."""
from __future__ import annotations

from collections import defaultdict
import csv
from datetime import UTC, datetime
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from cspp_v27_reused_core_score_cache import sha256_file  # noqa: E402


DEFAULT_PROTOCOL = (
    ROOT
    / "experiments/cspp_v27_ncedc_event_analysis_protocol_20260817.json"
)


def artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
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


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_csv(
    path: Path,
    rows: Iterable[dict[str, Any]],
    fields: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            extrasaction="raise",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def active_runs(active: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(active, dtype=bool)
    changes = np.diff(
        np.concatenate(
            (
                np.zeros(1, dtype=np.int8),
                values.astype(np.int8),
                np.zeros(1, dtype=np.int8),
            )
        )
    )
    starts = np.flatnonzero(changes == 1).astype(np.int64, copy=False)
    ends = (np.flatnonzero(changes == -1) - 1).astype(np.int64, copy=False)
    return starts, ends


def support_with_resets(
    probability: np.ndarray,
    allowed: np.ndarray,
    valid: np.ndarray,
    threshold: float,
    history_packets: int,
) -> np.ndarray:
    support = np.zeros(len(probability), dtype=bool)
    starts, ends = active_runs(valid)
    for first, last in zip(starts, ends, strict=True):
        segment = slice(int(first), int(last) + 1)
        hits = (
            np.asarray(allowed[segment], dtype=bool)
            & (probability[segment] >= float(threshold))
        )
        cumulative = np.concatenate(
            (np.zeros(1, dtype=np.int64), np.cumsum(hits, dtype=np.int64))
        )
        index = np.arange(len(hits), dtype=np.int64)
        left = np.maximum(index - int(history_packets) + 1, 0)
        support[segment] = cumulative[index + 1] - cumulative[left] > 0
    return support


def extract_candidates(
    *,
    support: np.ndarray,
    valid: np.ndarray,
    event_probability: np.ndarray,
    arrival_age: np.ndarray,
    event_threshold: float,
    start_epoch: float,
    packet_seconds: float,
    max_arrival_age_packets: float,
) -> list[dict[str, float | int]]:
    active = (
        support
        & valid
        & np.isfinite(event_probability)
        & (event_probability >= float(event_threshold))
    )
    starts, ends = active_runs(active)
    age_scale = max_arrival_age_packets * packet_seconds
    candidates: list[dict[str, float | int]] = []
    for episode_index, (first, last) in enumerate(
        zip(starts, ends, strict=True), start=1
    ):
        packet = int(first)
        available = start_epoch + (packet + 1) * packet_seconds
        candidates.append(
            {
                "episode_index": episode_index,
                "episode_start_packet": packet,
                "episode_end_packet": int(last),
                "available_epoch": available,
                "estimated_p_epoch": (
                    available - float(arrival_age[packet]) * age_scale
                ),
            }
        )
    return candidates


def matched_candidate(
    candidates: list[dict[str, float | int]],
    reference_p_epoch: float,
    tolerance_seconds: float,
) -> dict[str, float | int] | None:
    if not candidates:
        return None
    selected = min(
        candidates,
        key=lambda row: (
            abs(float(row["estimated_p_epoch"]) - reference_p_epoch),
            float(row["available_epoch"]),
        ),
    )
    if abs(float(selected["estimated_p_epoch"]) - reference_p_epoch) > tolerance_seconds:
        return None
    return selected


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total <= 0:
        return [float("nan"), float("nan")]
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total))
    half /= denominator
    return [max(0.0, center - half), min(1.0, center + half)]


def percentile(values: list[float], q: float) -> float | None:
    return float(np.quantile(values, q)) if values else None


def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    records = len(rows)
    matched = sum(int(row["matched_candidate_count"]) for row in rows)
    candidates = sum(int(row["candidate_count"]) for row in rows)
    false_candidates = candidates - matched
    recall = matched / records if records else 0.0
    precision = matched / candidates if candidates else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    errors = [float(row["pick_error_seconds"]) for row in rows if row["matched"]]
    absolute_errors = [abs(value) for value in errors]
    availability = [
        float(row["availability_delay_seconds"]) for row in rows if row["matched"]
    ]
    return {
        "records": records,
        "matched_records": matched,
        "false_negative_records": records - matched,
        "candidate_episodes": candidates,
        "false_candidate_episodes": false_candidates,
        "recall": recall,
        "recall_wilson95": wilson(matched, records),
        "candidate_precision": precision,
        "candidate_precision_wilson95": wilson(matched, candidates),
        "f1": f1,
        "pick_error_median_seconds": percentile(errors, 0.5),
        "pick_absolute_error_median_seconds": percentile(absolute_errors, 0.5),
        "pick_absolute_error_p95_seconds": percentile(absolute_errors, 0.95),
        "availability_delay_median_seconds": percentile(availability, 0.5),
        "availability_delay_p95_seconds": percentile(availability, 0.95),
        "available_within_0_5_seconds_fraction": (
            sum(value <= 0.5 for value in availability) / records if records else 0.0
        ),
        "available_within_1_0_seconds_fraction": (
            sum(value <= 1.0 for value in availability) / records if records else 0.0
        ),
        "available_within_2_0_seconds_fraction": (
            sum(value <= 2.0 for value in availability) / records if records else 0.0
        ),
    }


def event_cluster_bootstrap(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    replicates: int,
) -> dict[str, list[float]]:
    by_event: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_event[row["event_id"]].append(row)
    event_ids = sorted(by_event)
    rng = np.random.default_rng(seed)
    names = ["recall", "candidate_precision", "f1"]
    values = {name: np.empty(replicates, dtype=np.float64) for name in names}
    for index in range(replicates):
        sampled = rng.choice(event_ids, size=len(event_ids), replace=True)
        sample_rows = [row for event_id in sampled for row in by_event[str(event_id)]]
        result = metrics(sample_rows)
        for name in names:
            values[name][index] = float(result[name])
    return {
        name: [
            float(np.quantile(array, 0.025)),
            float(np.quantile(array, 0.975)),
        ]
        for name, array in values.items()
    }


def magnitude_bin(value: float) -> str:
    if value < 4.5:
        return "4.0_to_lt_4.5"
    if value < 5.0:
        return "4.5_to_lt_5.0"
    return "ge_5.0"


def distance_bin(value: float) -> str:
    if value < 50.0:
        return "0_to_lt_50_km"
    if value < 100.0:
        return "50_to_lt_100_km"
    return "100_to_200_km"


def network_detected(
    candidates_by_station: dict[str, list[dict[str, float | int]]],
    *,
    span_seconds: float,
    minimum_stations: int,
) -> bool:
    points = sorted(
        (float(candidate["estimated_p_epoch"]), station)
        for station, candidates in candidates_by_station.items()
        for candidate in candidates
    )
    left = 0
    for right in range(len(points)):
        while points[right][0] - points[left][0] > span_seconds:
            left += 1
        if len({station for _, station in points[left : right + 1]}) >= minimum_stations:
            return True
    return False


def main() -> None:
    protocol_sha256 = os.environ.get("CSPP_NCEDC_ANALYSIS_PROTOCOL_SHA256", "")
    actual = sha256_file(DEFAULT_PROTOCOL)
    if not protocol_sha256 or actual != protocol_sha256:
        raise RuntimeError(f"Set the frozen analysis protocol SHA-256; actual={actual}")
    protocol = json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))
    if protocol.get("protocol_id") != "cspp_v27_ncedc_event_analysis_v1":
        raise RuntimeError("Unexpected NCSN analysis protocol")
    for relative, expected in protocol["code_sha256"].items():
        path = ROOT / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"Analysis code drifted: {relative}")
    completion_path = verify_artifact(protocol["inputs"]["score_completion"])
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    if completion.get("protocol_sha256") != protocol["scoring_protocol"]["sha256"]:
        raise RuntimeError("Score completion binds another scoring protocol")

    event_units = []
    for spec in completion["unit_summaries"]:
        path = verify_artifact(spec)
        summary = json.loads(path.read_text(encoding="utf-8"))
        for cache_spec in summary["artifacts"].values():
            verify_artifact(cache_spec)
        event_units.append(summary)

    candidate_threshold = float(protocol["operating_point"]["candidate_threshold"])
    event_threshold = float(protocol["operating_point"]["event_threshold"])
    runtime = protocol["runtime"]
    tolerance = float(protocol["metrics"]["match_tolerance_seconds"])
    model_specs = protocol["models"]
    all_rows: list[dict[str, Any]] = []
    candidates_by_model_event: dict[
        tuple[str, str], dict[str, list[dict[str, float | int]]]
    ] = defaultdict(dict)
    for unit in event_units:
        for station in unit["station_results"]:
            cache_path = verify_artifact(station["score_cache"])
            with np.load(cache_path, allow_pickle=False) as cache:
                candidate_probability = cache["candidate_probability"].astype(
                    np.float32, copy=False
                )
                arrival_age = cache["arrival_age_fraction"].astype(
                    np.float32, copy=False
                )
                valid = cache["quality_valid"].astype(bool, copy=False)
                allowed = cache["candidate_allowed"].astype(bool, copy=False)
                support = support_with_resets(
                    candidate_probability,
                    allowed,
                    valid,
                    candidate_threshold,
                    int(runtime["candidate_history_packets"]),
                )
                for model_key, spec in model_specs.items():
                    candidates = extract_candidates(
                        support=support,
                        valid=valid,
                        event_probability=cache[
                            f"event_probability__{model_key}"
                        ].astype(np.float32, copy=False),
                        arrival_age=arrival_age,
                        event_threshold=event_threshold,
                        start_epoch=float(station["metadata"]["start_epoch"]),
                        packet_seconds=float(runtime["packet_duration_seconds"]),
                        max_arrival_age_packets=float(
                            runtime["max_arrival_age_packets"]
                        ),
                    )
                    reference = float(station["reference_p_epoch"])
                    matched = matched_candidate(candidates, reference, tolerance)
                    error = (
                        float(matched["estimated_p_epoch"]) - reference
                        if matched is not None
                        else None
                    )
                    availability = (
                        float(matched["available_epoch"]) - reference
                        if matched is not None
                        else None
                    )
                    local_snr = station["local_p_snr"] or {}
                    row = {
                        "event_id": unit["unit_key"],
                        "station_key": station["station_key"],
                        "model_key": model_key,
                        "seed": int(spec["seed"]),
                        "magnitude": float(station["magnitude"]),
                        "magnitude_bin": magnitude_bin(float(station["magnitude"])),
                        "epicentral_distance_km": float(
                            station["epicentral_distance_km"]
                        ),
                        "distance_bin": distance_bin(
                            float(station["epicentral_distance_km"])
                        ),
                        "native_sample_rate_hz": float(
                            station["native_sample_rate_hz"]
                        ),
                        "snr_valid": bool(local_snr.get("valid", False)),
                        "snr_db": local_snr.get("snr_db"),
                        "snr_bin": str(local_snr.get("snr_bin", "invalid")),
                        "candidate_count": len(candidates),
                        "matched_candidate_count": int(matched is not None),
                        "matched": matched is not None,
                        "false_candidate_count": len(candidates) - int(matched is not None),
                        "pick_error_seconds": error,
                        "absolute_pick_error_seconds": abs(error) if error is not None else None,
                        "availability_delay_seconds": availability,
                    }
                    all_rows.append(row)
                    candidates_by_model_event[(model_key, unit["unit_key"])][
                        station["station_key"]
                    ] = candidates

    output_root = Path(protocol["execution"]["output_root"])
    station_fields = [
        "event_id",
        "station_key",
        "model_key",
        "seed",
        "magnitude",
        "magnitude_bin",
        "epicentral_distance_km",
        "distance_bin",
        "native_sample_rate_hz",
        "snr_valid",
        "snr_db",
        "snr_bin",
        "candidate_count",
        "matched_candidate_count",
        "matched",
        "false_candidate_count",
        "pick_error_seconds",
        "absolute_pick_error_seconds",
        "availability_delay_seconds",
    ]
    station_path = output_root / "event_station_results.csv"
    write_csv(station_path, all_rows, station_fields)

    metrics_rows: list[dict[str, Any]] = []
    result_by_model: dict[str, Any] = {}
    for model_key, spec in model_specs.items():
        rows = [row for row in all_rows if row["model_key"] == model_key]
        summary = metrics(rows)
        summary["event_cluster_bootstrap95"] = event_cluster_bootstrap(
            rows,
            seed=int(protocol["uncertainty"]["bootstrap_seed"]),
            replicates=int(protocol["uncertainty"]["bootstrap_replicates"]),
        )
        detected_events = sum(
            network_detected(
                candidates_by_model_event[(model_key, event_id)],
                span_seconds=float(protocol["network_confirmation"]["span_seconds"]),
                minimum_stations=int(
                    protocol["network_confirmation"]["minimum_stations"]
                ),
            )
            for event_id in sorted({row["event_id"] for row in rows})
        )
        event_count = len({row["event_id"] for row in rows})
        summary["three_station_detected_events"] = detected_events
        summary["three_station_event_recall"] = detected_events / event_count
        summary["three_station_event_recall_wilson95"] = wilson(
            detected_events, event_count
        )
        result_by_model[model_key] = summary
        metrics_rows.append(
            {
                "model_key": model_key,
                "seed": int(spec["seed"]),
                "records": summary["records"],
                "matched_records": summary["matched_records"],
                "candidate_episodes": summary["candidate_episodes"],
                "recall": summary["recall"],
                "candidate_precision": summary["candidate_precision"],
                "f1": summary["f1"],
                "pick_absolute_error_median_seconds": summary[
                    "pick_absolute_error_median_seconds"
                ],
                "pick_absolute_error_p95_seconds": summary[
                    "pick_absolute_error_p95_seconds"
                ],
                "availability_delay_median_seconds": summary[
                    "availability_delay_median_seconds"
                ],
                "availability_delay_p95_seconds": summary[
                    "availability_delay_p95_seconds"
                ],
                "three_station_event_recall": summary[
                    "three_station_event_recall"
                ],
            }
        )
    metrics_path = output_root / "event_metrics_by_seed.csv"
    write_csv(metrics_path, metrics_rows, list(metrics_rows[0]))

    strata_rows: list[dict[str, Any]] = []
    for model_key, spec in model_specs.items():
        model_rows = [row for row in all_rows if row["model_key"] == model_key]
        for field in (
            "magnitude_bin",
            "distance_bin",
            "snr_bin",
            "native_sample_rate_hz",
        ):
            for value in sorted({str(row[field]) for row in model_rows}):
                rows = [row for row in model_rows if str(row[field]) == value]
                summary = metrics(rows)
                strata_rows.append(
                    {
                        "model_key": model_key,
                        "seed": int(spec["seed"]),
                        "stratum": field,
                        "level": value,
                        "records": summary["records"],
                        "matched_records": summary["matched_records"],
                        "candidate_episodes": summary["candidate_episodes"],
                        "recall": summary["recall"],
                        "candidate_precision": summary["candidate_precision"],
                        "f1": summary["f1"],
                        "availability_delay_median_seconds": summary[
                            "availability_delay_median_seconds"
                        ],
                    }
                )
    strata_path = output_root / "event_stratified_metrics.csv"
    write_csv(strata_path, strata_rows, list(strata_rows[0]))

    primary_key = next(
        key
        for key, spec in model_specs.items()
        if int(spec["seed"]) == int(protocol["operating_point"]["primary_operational_seed"])
    )
    result = {
        "result_id": "cspp_v27_ncedc_event_analysis_v1",
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "analysis_protocol_sha256": protocol_sha256,
        "scoring_protocol_sha256": protocol["scoring_protocol"]["sha256"],
        "operating_point": protocol["operating_point"],
        "primary_model_key": primary_key,
        "primary": result_by_model[primary_key],
        "all_seeds": result_by_model,
        "acquisition_accounting": {
            "requested_event_station_records": 580,
            "scored_event_station_records": 578,
            "structurally_incomplete_records": 2,
        },
        "artifacts": {
            "event_station_results": artifact(station_path),
            "event_metrics_by_seed": artifact(metrics_path),
            "event_stratified_metrics": artifact(strata_path),
        },
        "evidence_boundary": {
            "threshold_sweep_performed": False,
            "seed_selected": False,
            "target_adaptation_performed": False,
            "external_to_cspp_training_sources": True,
            "continuous_false_trigger_evidence_complete": False,
            "public_baseline_comparison_complete": False,
        },
    }
    result_path = output_root / "result.json"
    write_json_atomic(result_path, result)
    print(
        json.dumps(
            {
                "result": str(result_path),
                "primary_model_key": primary_key,
                "primary": result["primary"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
