#!/usr/bin/env python3
"""Analyze sealed offline public baselines on GeoNet 2026 final records."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import UTC, datetime
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.evaluation import (  # noqa: E402
    eval_v27_geonet_final_public_baselines as evaluator,
)
from scripts.evaluation import (  # noqa: E402
    freeze_v27_geonet_final_public_baseline_protocol as freeze,
)
from scripts.evaluation import (  # noqa: E402
    run_v27_geonet_final_public_baselines as runner,
)


DEFAULT_PROTOCOL = freeze.DEFAULT_OUTPUT
METHOD_RESULT_FIELDS = [
    "model_key",
    "method_id",
    "records",
    "event_clusters",
    "technical_failures",
    "formal_inference_permitted",
    "true_positive",
    "false_positive",
    "false_negative",
    "event_record_candidate_precision",
    "record_recall",
    "median_absolute_pick_error_seconds_hits",
]
PAIRED_RESULT_FIELDS = [
    "model_key",
    "method_id",
    "records",
    "event_clusters",
    "technical_failures",
    "formal_inference_permitted",
    "public_record_recall",
    "cspp_record_recall",
    "public_minus_cspp_record_recall",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--expected-protocol-sha256", required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _strict_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise RuntimeError(f"{label} must be an integer")
    if isinstance(value, (int, np.integer)):
        parsed = int(value)
    elif isinstance(value, str) and re.fullmatch(r"[0-9]+", value):
        parsed = int(value)
    else:
        raise RuntimeError(f"{label} must be a nonnegative integer")
    if parsed < 0:
        raise RuntimeError(f"{label} cannot be negative")
    return parsed


def _finite_float(value: Any, label: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"{label} must be numeric") from error
    if not math.isfinite(parsed):
        raise RuntimeError(f"{label} must be finite")
    return parsed


def _optional_finite_float(value: Any, label: str) -> float | None:
    if value in ("", None):
        return None
    return _finite_float(value, label)


def _ratio(numerator: int | float, denominator: int | float) -> float | None:
    return float(numerator / denominator) if denominator else None


def _quantile_interval(
    values: Iterable[float],
    confidence: float = 0.95,
) -> dict[str, float | int | None]:
    finite = np.asarray(
        [value for value in values if math.isfinite(value)],
        dtype=float,
    )
    if not len(finite):
        return {"lower": None, "upper": None, "valid_replicates": 0}
    alpha = 1.0 - confidence
    return {
        "lower": float(np.quantile(finite, alpha / 2.0, method="linear")),
        "upper": float(np.quantile(finite, 1.0 - alpha / 2.0, method="linear")),
        "valid_replicates": int(len(finite)),
    }


def _record_map(
    protocol: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    records = protocol["record_manifest"]
    output = {str(row["record_id"]): row for row in records}
    if len(output) != len(records):
        raise RuntimeError("Frozen record identifiers are not unique")
    return output


def _assert_same_float(
    observed: Any,
    expected: Any,
    label: str,
) -> None:
    if not math.isclose(
        _finite_float(observed, label),
        _finite_float(expected, label),
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise RuntimeError(f"Public-baseline record differs: {label}")


def _utc_epoch(value: Any, label: str) -> float:
    text = str(value)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as error:
        raise RuntimeError(f"{label} must be an ISO-8601 UTC time") from error
    if parsed.tzinfo is None:
        raise RuntimeError(f"{label} must include a UTC offset")
    return parsed.timestamp()


def validate_public_rows(
    rows: list[dict[str, Any]],
    protocol: Mapping[str, Any],
) -> dict[tuple[str, str], dict[str, Any]]:
    records = _record_map(protocol)
    expected_pairs = {
        (record_id, str(point["id"]))
        for record_id in records
        for point in protocol["methods"]
    }
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    waveform_by_record: dict[str, set[str]] = defaultdict(set)
    edge_fill_by_record: dict[str, set[int]] = defaultdict(set)
    for raw in rows:
        row = dict(raw)
        record_id = str(row.get("record_id", ""))
        method_id = str(row.get("method_id", ""))
        key = (record_id, method_id)
        if key in indexed or key not in expected_pairs:
            raise RuntimeError(f"Unexpected or duplicate public row: {key}")
        record = records[record_id]
        point = next(item for item in protocol["methods"] if item["id"] == method_id)
        expected_text = {
            "event_id": record["event_id"],
            "station_key": record["station_key"],
            "label_source": record["label_source"],
            "label_stratum": record["label_stratum"],
            "model_key": point["model_key"],
            "comparison_mode": "released_full_record_offline_annotation",
            "reference_p_time_utc": record["reference_p_time_utc"],
            "origin_time_utc": record["origin_time_utc"],
            "response_sha256": record["response_sha256"],
            "orientation_group_sha256": record["orientation_group_sha256"],
        }
        for field, expected in expected_text.items():
            if str(row.get(field, "")) != str(expected):
                raise RuntimeError(f"Public row identity differs: {field}")
        if _strict_int(row["site_rank"], "site_rank") != int(record["site_rank"]):
            raise RuntimeError("Public row site rank differs")
        for field in (
            "pick_time_weight",
            "magnitude",
            "epicentral_distance_km",
        ):
            _assert_same_float(row[field], record[field], field)
        _assert_same_float(row["sample_rate_hz"], 100.0, "sample_rate_hz")
        origin_epoch = _utc_epoch(record["origin_time_utc"], "origin_time_utc")
        reference_epoch = _utc_epoch(
            record["reference_p_time_utc"],
            "reference_p_time_utc",
        )
        record_start_epoch = _utc_epoch(
            row.get("record_start_time_utc"),
            "record_start_time_utc",
        )
        if not math.isclose(
            record_start_epoch,
            origin_epoch - 60.0,
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise RuntimeError("Public record start is not origin minus 60 s")
        reference_seconds = _finite_float(
            row["reference_p_seconds"], "reference_p_seconds"
        )
        expected_reference_seconds = reference_epoch - record_start_epoch
        if not math.isclose(
            reference_seconds,
            expected_reference_seconds,
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise RuntimeError("Public reference seconds differ from UTC times")
        if not 0.0 <= reference_seconds <= 180.0:
            raise RuntimeError("Reference P lies outside the public record")
        counts = {
            field: _strict_int(row[field], field)
            for field in (
                "n_samples",
                "candidate_count",
                "matched_candidate_count",
                "true_positive",
                "false_positive",
                "false_negative",
                "within_0p5_seconds",
                "within_1p0_seconds",
                "within_1p5_seconds",
                "edge_filled_component_samples",
            )
        }
        if counts["n_samples"] not in {0, 18000}:
            raise RuntimeError("Public waveform sample count differs")
        if counts["edge_filled_component_samples"] > int(3 * 18000 * 0.001):
            raise RuntimeError("Public edge-fill count exceeds the frozen limit")
        for field in (
            "true_positive",
            "false_negative",
            "within_0p5_seconds",
            "within_1p0_seconds",
            "within_1p5_seconds",
        ):
            if counts[field] not in {0, 1}:
                raise RuntimeError(f"{field} must be binary")
        if counts["true_positive"] + counts["false_negative"] != 1:
            raise RuntimeError("Every public record must be a hit or miss")
        if (
            counts["false_positive"]
            != counts["candidate_count"] - counts["true_positive"]
            or counts["matched_candidate_count"] != counts["true_positive"]
            or counts["matched_candidate_count"] > counts["candidate_count"]
            or counts["within_0p5_seconds"] > counts["within_1p0_seconds"]
            or counts["within_1p0_seconds"] > counts["within_1p5_seconds"]
            or counts["within_1p5_seconds"] != counts["true_positive"]
        ):
            raise RuntimeError("Public candidate accounting differs")
        technical_error = str(row.get("technical_error", ""))
        selected = _optional_finite_float(
            row.get("selected_estimated_p_seconds"),
            "selected_estimated_p_seconds",
        )
        pick_error = _optional_finite_float(
            row.get("pick_error_seconds"), "pick_error_seconds"
        )
        absolute_error = _optional_finite_float(
            row.get("absolute_pick_error_seconds"),
            "absolute_pick_error_seconds",
        )
        waveform_sha = str(row.get("waveform_sha256", ""))
        if technical_error:
            if (
                counts["candidate_count"] != 0
                or counts["matched_candidate_count"] != 0
                or counts["true_positive"] != 0
                or counts["false_positive"] != 0
                or counts["false_negative"] != 1
                or selected is not None
                or pick_error is not None
                or absolute_error is not None
            ):
                raise RuntimeError("Technical failure accounting differs")
            if waveform_sha and not re.fullmatch(r"[0-9a-f]{64}", waveform_sha):
                raise RuntimeError("Technical-failure waveform SHA is invalid")
        else:
            if counts["n_samples"] != 18000:
                raise RuntimeError("Successful public record lacks full waveform")
            if not re.fullmatch(r"[0-9a-f]{64}", waveform_sha):
                raise RuntimeError("Successful public waveform SHA is invalid")
            if counts["true_positive"]:
                expected_pick_error = (
                    None if selected is None else selected - reference_seconds
                )
                if (
                    selected is None
                    or pick_error is None
                    or absolute_error is None
                    or expected_pick_error is None
                    or not math.isclose(
                        pick_error,
                        expected_pick_error,
                        rel_tol=0.0,
                        abs_tol=1e-9,
                    )
                    or not math.isclose(
                        abs(pick_error),
                        absolute_error,
                        rel_tol=0.0,
                        abs_tol=1e-9,
                    )
                    or absolute_error > 1.5 + 1e-9
                ):
                    raise RuntimeError("Public matched-pick fields differ")
                expected_windows = {
                    "within_0p5_seconds": int(absolute_error <= 0.5),
                    "within_1p0_seconds": int(absolute_error <= 1.0),
                    "within_1p5_seconds": int(absolute_error <= 1.5),
                }
                if any(
                    counts[field] != expected
                    for field, expected in expected_windows.items()
                ):
                    raise RuntimeError("Public hit-window flags differ")
            elif any(
                value is not None for value in (selected, pick_error, absolute_error)
            ):
                raise RuntimeError("Public miss contains a selected pick")
        if waveform_sha:
            waveform_by_record[record_id].add(waveform_sha)
        edge_fill_by_record[record_id].add(
            counts["edge_filled_component_samples"]
        )
        row.update(counts)
        row["technical_failure"] = int(bool(technical_error))
        row["pick_error_seconds"] = pick_error
        row["absolute_pick_error_seconds"] = absolute_error
        indexed[key] = row
    if set(indexed) != expected_pairs:
        missing = sorted(expected_pairs.difference(indexed))
        raise RuntimeError(f"Public method/record coverage differs: {missing[:3]}")
    if any(len(hashes) > 1 for hashes in waveform_by_record.values()):
        raise RuntimeError("Public models did not use identical waveform bytes")
    if any(len(counts) > 1 for counts in edge_fill_by_record.values()):
        raise RuntimeError("Public models did not use identical edge fills")
    return indexed


def point_estimate(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise RuntimeError("A public method has no records")
    tp = sum(int(row["true_positive"]) for row in rows)
    fp = sum(int(row["false_positive"]) for row in rows)
    fn = sum(int(row["false_negative"]) for row in rows)
    failures = sum(int(row["technical_failure"]) for row in rows)
    errors = [
        float(row["absolute_pick_error_seconds"])
        for row in rows
        if int(row["true_positive"])
    ]
    return {
        "records": len(rows),
        "event_clusters": len({str(row["event_id"]) for row in rows}),
        "technical_failures": failures,
        "technical_success_records": len(rows) - failures,
        "technical_success_rate": _ratio(
            len(rows) - failures,
            len(rows),
        ),
        "formal_inference_permitted": failures == 0,
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "event_record_candidate_precision": _ratio(tp, tp + fp),
        "event_record_candidate_precision_scope": (
            "conditional_on_emitted_candidates_from_successfully_processed_"
            "records"
        ),
        "record_recall": _ratio(tp, tp + fn),
        "mean_candidates_per_record": float(
            sum(int(row["candidate_count"]) for row in rows) / len(rows)
        ),
        "within_0p5_rate_all_records": float(
            sum(int(row["within_0p5_seconds"]) for row in rows) / len(rows)
        ),
        "within_1p0_rate_all_records": float(
            sum(int(row["within_1p0_seconds"]) for row in rows) / len(rows)
        ),
        "within_1p5_rate_all_records": float(tp / len(rows)),
        "median_absolute_pick_error_seconds_hits": (
            float(np.median(errors)) if errors else None
        ),
        "inference_status": (
            "formal_event_cluster_inference"
            if failures == 0
            else "descriptive_only_due_to_technical_failure"
        ),
    }


def event_cluster_bootstrap(
    rows: list[Mapping[str, Any]],
    *,
    seed: int,
    replicates: int,
    confidence: float = 0.95,
) -> dict[str, Any]:
    if replicates <= 0:
        raise ValueError("replicates must be positive")
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["event_id"])].append(row)
    event_ids = sorted(grouped)
    if not event_ids:
        raise RuntimeError("No event clusters are available")
    rng = np.random.default_rng(seed)
    recalls = []
    precisions = []
    medians = []
    for sampled in rng.integers(0, len(event_ids), size=(replicates, len(event_ids))):
        selected = [row for index in sampled for row in grouped[event_ids[int(index)]]]
        estimate = point_estimate(selected)
        recalls.append(float(estimate["record_recall"]))
        precision = estimate["event_record_candidate_precision"]
        precisions.append(math.nan if precision is None else float(precision))
        median = estimate["median_absolute_pick_error_seconds_hits"]
        medians.append(math.nan if median is None else float(median))
    estimate = point_estimate(rows)
    inferential = int(estimate["technical_failures"]) == 0
    unavailable = {
        "lower": None,
        "upper": None,
        "valid_replicates": 0,
    }
    return {
        "cluster": "event_id",
        "event_clusters": len(event_ids),
        "seed": int(seed),
        "replicates": int(replicates),
        "confidence_level": float(confidence),
        "record_recall": {
            "estimate": estimate["record_recall"],
            **(
                _quantile_interval(recalls, confidence)
                if inferential
                else unavailable
            ),
        },
        "event_record_candidate_precision": {
            "estimate": estimate["event_record_candidate_precision"],
            **(
                _quantile_interval(precisions, confidence)
                if inferential
                else unavailable
            ),
        },
        "median_absolute_pick_error_seconds_hits": {
            "estimate": estimate["median_absolute_pick_error_seconds_hits"],
            **(
                _quantile_interval(medians, confidence)
                if inferential
                else unavailable
            ),
        },
        "formal_inference_permitted": inferential,
        "inference_status": estimate["inference_status"],
    }


def station_cluster_bootstrap(
    rows: list[Mapping[str, Any]],
    *,
    seed: int,
    replicates: int,
    confidence: float = 0.95,
) -> dict[str, Any]:
    if replicates <= 0:
        raise ValueError("replicates must be positive")
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["station_key"])].append(row)
    station_keys = sorted(grouped)
    if not station_keys:
        raise RuntimeError("No station clusters are available")
    estimate = point_estimate(rows)
    inferential = int(estimate["technical_failures"]) == 0
    recalls = []
    rng = np.random.default_rng(seed)
    for sampled in rng.integers(
        0,
        len(station_keys),
        size=(replicates, len(station_keys)),
    ):
        selected = [
            row
            for index in sampled
            for row in grouped[station_keys[int(index)]]
        ]
        recalls.append(float(point_estimate(selected)["record_recall"]))
    return {
        "cluster": "station_key",
        "station_clusters": len(station_keys),
        "seed": int(seed),
        "replicates": int(replicates),
        "confidence_level": float(confidence),
        "record_recall": {
            "estimate": estimate["record_recall"],
            **(
                _quantile_interval(recalls, confidence)
                if inferential
                else {
                    "lower": None,
                    "upper": None,
                    "valid_replicates": 0,
                }
            ),
        },
        "formal_inference_permitted": inferential,
        "inference_status": (
            "station_cluster_dependence_sensitivity"
            if inferential
            else "descriptive_only_due_to_technical_failure"
        ),
    }


def paired_event_cluster_bootstrap(
    public_hits: list[Mapping[str, Any]],
    cspp_hits: Mapping[str, int],
    *,
    seed: int,
    replicates: int,
    confidence: float = 0.95,
) -> dict[str, Any]:
    if replicates <= 0:
        raise ValueError("replicates must be positive")
    grouped: dict[str, list[tuple[int, int]]] = defaultdict(list)
    seen = set()
    failures = 0
    for row in public_hits:
        record_id = str(row["record_id"])
        if record_id in seen or record_id not in cspp_hits:
            raise RuntimeError("Paired CSPP/public record identity differs")
        seen.add(record_id)
        grouped[str(row["event_id"])].append(
            (int(row["true_positive"]), int(cspp_hits[record_id]))
        )
        failures += int(row.get("technical_failure", 0))
    if seen != set(cspp_hits):
        raise RuntimeError("Paired CSPP/public coverage differs")
    event_ids = sorted(grouped)
    if not event_ids:
        raise RuntimeError("No paired event clusters are available")

    def difference(selected_ids: list[str]) -> tuple[float, float, float]:
        pairs = [pair for event_id in selected_ids for pair in grouped[event_id]]
        public_recall = sum(pair[0] for pair in pairs) / len(pairs)
        cspp_recall = sum(pair[1] for pair in pairs) / len(pairs)
        return public_recall, cspp_recall, public_recall - cspp_recall

    observed = difference(event_ids)
    rng = np.random.default_rng(seed)
    differences = []
    for sampled in rng.integers(0, len(event_ids), size=(replicates, len(event_ids))):
        selected = [event_ids[int(index)] for index in sampled]
        differences.append(difference(selected)[2])
    return {
        "cluster": "event_id",
        "event_clusters": len(event_ids),
        "records": len(seen),
        "seed": int(seed),
        "replicates": int(replicates),
        "confidence_level": float(confidence),
        "public_record_recall": observed[0],
        "cspp_record_recall": observed[1],
        "public_minus_cspp_record_recall": {
            "estimate": observed[2],
            **(
                _quantile_interval(differences, confidence)
                if failures == 0
                else {
                    "lower": None,
                    "upper": None,
                    "valid_replicates": 0,
                }
            ),
        },
        "technical_failures": failures,
        "formal_inference_permitted": failures == 0,
        "inference_status": (
            "formal_paired_event_cluster_inference"
            if failures == 0
            else "descriptive_only_due_to_public_technical_failure"
        ),
        "mcnemar": "not_implemented",
    }


def paired_station_cluster_bootstrap(
    public_hits: list[Mapping[str, Any]],
    cspp_hits: Mapping[str, int],
    *,
    seed: int,
    replicates: int,
    confidence: float = 0.95,
) -> dict[str, Any]:
    if replicates <= 0:
        raise ValueError("replicates must be positive")
    grouped: dict[str, list[tuple[int, int]]] = defaultdict(list)
    seen = set()
    failures = 0
    for row in public_hits:
        record_id = str(row["record_id"])
        if record_id in seen or record_id not in cspp_hits:
            raise RuntimeError("Paired CSPP/public record identity differs")
        seen.add(record_id)
        grouped[str(row["station_key"])].append(
            (int(row["true_positive"]), int(cspp_hits[record_id]))
        )
        failures += int(row.get("technical_failure", 0))
    if seen != set(cspp_hits):
        raise RuntimeError("Paired CSPP/public coverage differs")
    station_keys = sorted(grouped)
    if not station_keys:
        raise RuntimeError("No paired station clusters are available")

    def difference(selected: list[str]) -> float:
        pairs = [
            pair
            for station_key in selected
            for pair in grouped[station_key]
        ]
        return (
            sum(pair[0] for pair in pairs)
            - sum(pair[1] for pair in pairs)
        ) / len(pairs)

    observed = difference(station_keys)
    rng = np.random.default_rng(seed)
    differences = [
        difference([station_keys[int(index)] for index in sampled])
        for sampled in rng.integers(
            0,
            len(station_keys),
            size=(replicates, len(station_keys)),
        )
    ]
    return {
        "cluster": "station_key",
        "station_clusters": len(station_keys),
        "records": len(seen),
        "seed": int(seed),
        "replicates": int(replicates),
        "confidence_level": float(confidence),
        "public_minus_cspp_record_recall": {
            "estimate": observed,
            **(
                _quantile_interval(differences, confidence)
                if failures == 0
                else {
                    "lower": None,
                    "upper": None,
                    "valid_replicates": 0,
                }
            ),
        },
        "technical_failures": failures,
        "formal_inference_permitted": failures == 0,
        "inference_status": (
            "station_cluster_paired_dependence_sensitivity"
            if failures == 0
            else "descriptive_only_due_to_public_technical_failure"
        ),
    }


def load_cspp_hits(
    protocol: Mapping[str, Any],
) -> dict[str, int]:
    comparator = protocol["cspp_comparator"]
    result_path = freeze.verify_artifact(comparator["result"])
    result = json.loads(result_path.read_text(encoding="utf-8"))
    protocol_spec = result.get("protocol", {})
    if (
        result.get("result_id")
        != "cspp_v27c_geonet_final_evaluation_result_a2"
        or Path(str(protocol_spec.get("path", ""))).resolve()
        != Path(protocol["final_evaluation_protocol"]["path"]).resolve()
        or str(protocol_spec.get("sha256", ""))
        != protocol["final_evaluation_protocol_sha256"]
        or freeze.verify_artifact(protocol_spec)
        != Path(protocol["final_evaluation_protocol"]["path"]).resolve()
    ):
        raise RuntimeError("CSPP result is bound to another final protocol")
    station_spec = result.get("artifacts", {}).get("event_station_results", {})
    station_path = freeze.verify_artifact(station_spec)
    frozen_station_spec = comparator["event_station_results"]
    frozen_station_path = freeze.verify_artifact(frozen_station_spec)
    if station_path != frozen_station_path or any(
        str(station_spec.get(field)) != str(frozen_station_spec.get(field))
        for field in ("path", "bytes", "sha256")
    ):
        raise RuntimeError("CSPP event-station result artifact differs")
    primary_seed = int(comparator["primary_operational_seed"])
    manifest = _record_map(protocol)
    output: dict[str, int] = {}
    for row in read_csv(station_path):
        if (
            row["label_stratum"] != freeze.PRIMARY_LABEL_STRATUM
            or _strict_int(row["seed"], "CSPP seed") != primary_seed
        ):
            continue
        record_id = f"{row['event_id']}/{row['station_key']}"
        if record_id not in manifest or record_id in output:
            raise RuntimeError("CSPP primary record identity differs")
        if _strict_int(row["technical_available"], "technical_available") != 1:
            raise RuntimeError("A public scorable record is unavailable to CSPP")
        matched = _strict_int(row["matched_candidate_count"], "matched_candidate_count")
        output[record_id] = int(matched > 0)
    if set(output) != set(manifest):
        raise RuntimeError("CSPP/public final record coverage differs")
    return output


def write_csv(
    path: Path,
    rows: list[Mapping[str, Any]],
    fields: list[str],
) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        handle.write("\n")


def publish_directory_no_replace(attempt: Path, destination: Path) -> None:
    if attempt.parent != destination.parent:
        raise RuntimeError("Publication rename must remain in one directory")
    claim = destination.with_name(f".{destination.name}.publish-lock")
    try:
        claim.mkdir()
    except FileExistsError as error:
        raise RuntimeError(f"Publication lock already exists: {claim}") from error
    try:
        if destination.exists():
            raise FileExistsError(destination)
        os.rename(attempt, destination)
    finally:
        claim.rmdir()


def main() -> None:
    args = parse_args()
    protocol, protocol_sha256, protocol_path = evaluator.load_protocol(
        args.protocol,
        expected_sha256=args.expected_protocol_sha256,
    )
    all_rows = []
    run_artifacts = {}
    for model in sorted(freeze.EXPECTED_MODELS):
        root = evaluator.run_root(protocol, model)
        summary = runner.verify_summary(
            root / "summary.json",
            protocol=protocol,
            protocol_sha256=protocol_sha256,
            model=model,
        )
        audit = runner.verify_audit(
            root / "execution_audit.json",
            protocol=protocol,
            protocol_sha256=protocol_sha256,
            model=model,
        )
        all_rows.extend(read_csv(root / "records.csv"))
        run_artifacts[model] = {
            "summary": freeze.artifact(root / "summary.json"),
            "records": summary["records_artifact"],
            "execution_audit": freeze.artifact(root / "execution_audit.json"),
            "audit_resource_samples": int(audit["resource_sample_count"]),
        }
    indexed = validate_public_rows(all_rows, protocol)
    cspp_hits = load_cspp_hits(protocol)
    statistics = protocol["statistics"]
    confidence = float(statistics["confidence_level"])
    seed = int(statistics["bootstrap_seed"])
    replicates = int(statistics["bootstrap_replicates"])
    method_results = {}
    paired_results = {}
    method_table = []
    paired_table = []
    for method_index, point in enumerate(protocol["methods"]):
        method_id = str(point["id"])
        model = str(point["model_key"])
        rows = [
            indexed[(record["record_id"], method_id)]
            for record in protocol["record_manifest"]
        ]
        estimate = point_estimate(rows)
        bootstrap = event_cluster_bootstrap(
            rows,
            seed=seed + method_index,
            replicates=replicates,
            confidence=confidence,
        )
        paired = paired_event_cluster_bootstrap(
            rows,
            cspp_hits,
            seed=seed + 10_000 + method_index,
            replicates=replicates,
            confidence=confidence,
        )
        station_bootstrap = station_cluster_bootstrap(
            rows,
            seed=seed + 20_000 + method_index,
            replicates=replicates,
            confidence=confidence,
        )
        paired_station = paired_station_cluster_bootstrap(
            rows,
            cspp_hits,
            seed=seed + 30_000 + method_index,
            replicates=replicates,
            confidence=confidence,
        )
        key = f"{model}/{method_id}"
        method_results[key] = {
            "model_key": model,
            "method_id": method_id,
            "point_estimate": estimate,
            "event_cluster_bootstrap_95_ci": bootstrap,
            "station_cluster_recall_sensitivity": station_bootstrap,
        }
        paired_results[key] = {
            "model_key": model,
            "method_id": method_id,
            **paired,
            "station_cluster_difference_sensitivity": paired_station,
        }
        method_table.append({"model_key": model, "method_id": method_id, **estimate})
        paired_table.append(
            {
                "model_key": model,
                "method_id": method_id,
                "records": paired["records"],
                "event_clusters": paired["event_clusters"],
                "technical_failures": paired["technical_failures"],
                "formal_inference_permitted": paired["formal_inference_permitted"],
                "public_record_recall": paired["public_record_recall"],
                "cspp_record_recall": paired["cspp_record_recall"],
                "public_minus_cspp_record_recall": paired[
                    "public_minus_cspp_record_recall"
                ]["estimate"],
            }
        )
    allowed = Path(protocol["execution"]["allowed_output_root"]).resolve()
    published_root = freeze.require_output_path(
        protocol["execution"]["analysis_root"],
        allowed_root=allowed,
    )
    if published_root.exists():
        raise FileExistsError(published_root)
    attempt = published_root.with_name(f".{published_root.name}.attempt-{os.getpid()}")
    attempt.mkdir(parents=True)
    try:
        method_path = attempt / "method_results.csv"
        paired_path = attempt / "paired_cspp_results.csv"
        result_path = attempt / "result.json"
        write_csv(method_path, method_table, METHOD_RESULT_FIELDS)
        write_csv(paired_path, paired_table, PAIRED_RESULT_FIELDS)
        result = {
            "result_id": "cspp_v27_geonet_final_public_baselines_result_v2",
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "protocol": freeze.artifact(protocol_path),
            "endpoint": protocol["endpoint"],
            "method_results": method_results,
            "paired_cspp_record_recall_differences": paired_results,
            "run_artifacts": run_artifacts,
            "artifacts": {
                "method_results": {
                    **freeze.artifact(method_path),
                    "path": str(published_root / method_path.name),
                },
                "paired_cspp_results": {
                    **freeze.artifact(paired_path),
                    "path": str(published_root / paired_path.name),
                },
            },
            "claim_boundary": protocol["claim_boundary"],
            "interpretation": (
                "All public-model estimates are full-record offline "
                "event-station endpoints. No causal availability or online "
                "latency comparison was generated."
            ),
        }
        write_json(result_path, result)
        publish_directory_no_replace(attempt, published_root)
    except BaseException:
        if attempt.is_dir():
            for child in attempt.iterdir():
                child.unlink()
            attempt.rmdir()
        raise
    print(
        json.dumps(
            freeze.artifact(published_root / "result.json"),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
