#!/usr/bin/env python3
"""Analyze the frozen V27C K-NET/InstanceGM retrospective benchmark."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.evaluation.eval_cspp_v27c_fixed_retrospective import (  # noqa: E402
    ALLOWED_OUTPUT_ROOT,
    MODEL_KEY,
    OUTPUT_FIELDS,
    artifact,
    load_protocol,
    reject_forbidden_path,
    require_output_path,
    verify_artifact,
)
from scripts.evaluation import (  # noqa: E402
    eval_v27c_fixed_public_baselines as formal_public,
)
from scripts.evaluation import (  # noqa: E402
    freeze_v27c_fixed_public_baseline_protocol as formal_public_freeze,
)
from scripts.evaluation import (  # noqa: E402
    run_v27c_fixed_public_baselines as formal_public_runner,
)
from scripts.evaluation import (  # noqa: E402
    run_cspp_v27c_fixed_retrospective as fixed_runner,
)


DEFAULT_PROTOCOL = (
    ALLOWED_OUTPUT_ROOT
    / "protocol_freeze/cspp_v27c_fixed_retrospective_protocol_20260729.json"
)
NUMERIC_COLUMNS = {
    "primary_seed",
    "magnitude",
    "epicentral_distance_km",
    "p_arrival_sample",
    "p_precontext_seconds",
    "sample_rate_hz",
    "manifest_n_samples",
    "evaluated_n_samples",
    "snr_valid",
    "local_p_excess_power_snr_db",
    "packet_count",
    "quality_valid_packets",
    "quality_invalid_packets",
    "candidate_allowed_packets",
    "candidate_count",
    "matched_candidate_count",
    "true_positive",
    "false_positive",
    "false_negative",
    "within_0p5_seconds",
    "within_1p0_seconds",
    "within_1p5_seconds",
    "selected_estimated_p_seconds",
    "selected_available_seconds",
    "pick_error_seconds",
    "absolute_pick_error_seconds",
    "availability_latency_seconds",
    "technical_failure",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument(
        "--expected-protocol-sha256",
        required=True,
        help="Required SHA-256 of the frozen retrospective protocol",
    )
    parser.add_argument(
        "--public-baseline-artifact",
        action="append",
        default=[],
        metavar="DATASET:METHOD:CSV",
        help=(
            "Optional user-supplied same-manifest endpoint CSV. It is not a "
            "provenance-verified public-model run."
        ),
    )
    parser.add_argument(
        "--public-baseline-protocol",
        type=Path,
        help=(
            "Optional provenance-sealed PhaseNet/EQTransformer protocol. "
            "All four same-manifest run summaries are verified from it."
        ),
    )
    parser.add_argument(
        "--expected-public-baseline-protocol-sha256",
        help="Required when --public-baseline-protocol is supplied.",
    )
    return parser.parse_args()


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    return value


def _finite_quantile(
    values: Iterable[Any],
    probability: float,
) -> float:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return math.nan
    return float(np.quantile(array, probability, method="linear"))


def validate_record_accounting(frame: pd.DataFrame) -> None:
    if frame.empty:
        raise RuntimeError("Retrospective result contains no records")
    if not frame["true_positive"].isin([0, 1]).all():
        raise RuntimeError("true_positive must be binary")
    if not frame["false_negative"].isin([0, 1]).all():
        raise RuntimeError("false_negative must be binary")
    if not (frame["true_positive"] + frame["false_negative"]).eq(1).all():
        raise RuntimeError("Every record must be a hit or a miss")
    if not frame["false_positive"].ge(0).all():
        raise RuntimeError("false_positive cannot be negative")
    if not frame["technical_failure"].isin([0, 1]).all():
        raise RuntimeError("technical_failure must be binary")
    failed = frame["technical_failure"].eq(1)
    if not (
        frame.loc[failed, "true_positive"].eq(0).all()
        and frame.loc[failed, "false_positive"].eq(0).all()
        and frame.loc[failed, "false_negative"].eq(1).all()
    ):
        raise RuntimeError("Technical failures were not counted as misses")


def metric_values(frame: pd.DataFrame) -> dict[str, Any]:
    validate_record_accounting(frame)
    tp = int(frame["true_positive"].sum())
    fp = int(frame["false_positive"].sum())
    fn = int(frame["false_negative"].sum())
    precision = tp / (tp + fp) if tp + fp else math.nan
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1_denominator = 2 * tp + fp + fn
    f1 = 2.0 * tp / f1_denominator if f1_denominator else math.nan
    hits = frame.loc[frame["true_positive"].eq(1)]
    return {
        "records": int(len(frame)),
        "events": int(frame["event_id"].astype(str).nunique()),
        "stations": int(frame["station_id"].astype(str).nunique()),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "event_record_candidate_precision": float(precision),
        "record_recall": float(recall),
        "event_record_candidate_f1": float(f1),
        "records_with_any_candidate": int(
            frame["candidate_count"].gt(0).sum()
        ),
        "mean_candidates_per_record": float(
            frame["candidate_count"].mean()
        ),
        "within_0p5_rate_all_records": float(
            frame["within_0p5_seconds"].mean()
        ),
        "within_1p0_rate_all_records": float(
            frame["within_1p0_seconds"].mean()
        ),
        "within_1p5_rate_all_records": float(
            frame["within_1p5_seconds"].mean()
        ),
        "median_signed_pick_error_seconds_hits": (
            float(hits["pick_error_seconds"].median())
            if len(hits)
            else math.nan
        ),
        "median_abs_pick_error_seconds_hits": (
            float(hits["absolute_pick_error_seconds"].median())
            if len(hits)
            else math.nan
        ),
        "p90_abs_pick_error_seconds_hits": _finite_quantile(
            hits["absolute_pick_error_seconds"], 0.9
        ),
        "median_availability_latency_seconds_hits": (
            float(hits["availability_latency_seconds"].median())
            if len(hits)
            else math.nan
        ),
        "p90_availability_latency_seconds_hits": _finite_quantile(
            hits["availability_latency_seconds"], 0.9
        ),
        "technical_failures": int(frame["technical_failure"].sum()),
        "snr_invalid_records": int(frame["snr_valid"].eq(0).sum()),
    }


def _ratio(
    numerator: np.ndarray,
    denominator: np.ndarray,
) -> np.ndarray:
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan, dtype=float),
        where=denominator > 0,
    )


def _interval(values: np.ndarray) -> dict[str, float | int]:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return {
            "lower_95": math.nan,
            "upper_95": math.nan,
            "valid_replicates": 0,
        }
    return {
        "lower_95": float(np.quantile(finite, 0.025, method="linear")),
        "upper_95": float(np.quantile(finite, 0.975, method="linear")),
        "valid_replicates": int(len(finite)),
    }


def event_cluster_bootstrap(
    frame: pd.DataFrame,
    *,
    seed: int,
    replicates: int,
) -> dict[str, Any]:
    validate_record_accounting(frame)
    if replicates <= 0:
        raise ValueError("replicates must be positive")
    grouped_frames = [
        group
        for _, group in frame.groupby("event_id", sort=True, dropna=False)
    ]
    event_count = len(grouped_frames)
    if not event_count:
        raise RuntimeError("No event clusters are available")
    count_fields = (
        "true_positive",
        "false_positive",
        "false_negative",
        "within_0p5_seconds",
        "within_1p0_seconds",
        "within_1p5_seconds",
    )
    event_sums = {
        field: np.asarray(
            [group[field].sum() for group in grouped_frames], dtype=float
        )
        for field in count_fields
    }
    event_records = np.asarray(
        [len(group) for group in grouped_frames], dtype=float
    )
    event_abs_errors = [
        group.loc[
            group["true_positive"].eq(1),
            "absolute_pick_error_seconds",
        ]
        .dropna()
        .to_numpy(dtype=float)
        for group in grouped_frames
    ]
    event_latencies = [
        group.loc[
            group["true_positive"].eq(1),
            "availability_latency_seconds",
        ]
        .dropna()
        .to_numpy(dtype=float)
        for group in grouped_frames
    ]

    rng = np.random.default_rng(seed)
    sampled = rng.integers(
        0, event_count, size=(replicates, event_count)
    )
    totals = {
        field: values[sampled].sum(axis=1)
        for field, values in event_sums.items()
    }
    records = event_records[sampled].sum(axis=1)
    precision = _ratio(
        totals["true_positive"],
        totals["true_positive"] + totals["false_positive"],
    )
    recall = _ratio(
        totals["true_positive"],
        totals["true_positive"] + totals["false_negative"],
    )
    f1 = _ratio(
        2.0 * totals["true_positive"],
        (
            2.0 * totals["true_positive"]
            + totals["false_positive"]
            + totals["false_negative"]
        ),
    )
    within = {
        window: _ratio(totals[field], records)
        for window, field in (
            ("within_0p5_rate_all_records", "within_0p5_seconds"),
            ("within_1p0_rate_all_records", "within_1p0_seconds"),
            ("within_1p5_rate_all_records", "within_1p5_seconds"),
        )
    }

    median_abs = np.full(replicates, np.nan, dtype=float)
    median_latency = np.full(replicates, np.nan, dtype=float)
    for index, event_indexes in enumerate(sampled):
        errors = [
            event_abs_errors[event_index]
            for event_index in event_indexes
            if len(event_abs_errors[event_index])
        ]
        latencies = [
            event_latencies[event_index]
            for event_index in event_indexes
            if len(event_latencies[event_index])
        ]
        if errors:
            median_abs[index] = float(np.median(np.concatenate(errors)))
        if latencies:
            median_latency[index] = float(
                np.median(np.concatenate(latencies))
            )

    estimates = metric_values(frame)
    output = {
        "replicates": int(replicates),
        "seed": int(seed),
        "cluster": "event_id",
        "event_clusters": event_count,
        "event_record_candidate_precision": {
            "estimate": estimates["event_record_candidate_precision"],
            **_interval(precision),
        },
        "record_recall": {
            "estimate": estimates["record_recall"],
            **_interval(recall),
        },
        "event_record_candidate_f1": {
            "estimate": estimates["event_record_candidate_f1"],
            **_interval(f1),
        },
        "median_abs_pick_error_seconds_hits": {
            "estimate": estimates["median_abs_pick_error_seconds_hits"],
            **_interval(median_abs),
        },
        "median_availability_latency_seconds_hits": {
            "estimate": estimates[
                "median_availability_latency_seconds_hits"
            ],
            **_interval(median_latency),
        },
    }
    for name, values in within.items():
        output[name] = {
            "estimate": estimates[name],
            **_interval(values),
        }
    return output


def read_result_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = NUMERIC_COLUMNS.difference(frame.columns)
    if missing:
        raise RuntimeError(f"Result lacks numeric columns: {sorted(missing)}")
    for column in NUMERIC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame[list(NUMERIC_COLUMNS - {
        "local_p_excess_power_snr_db",
        "selected_estimated_p_seconds",
        "selected_available_seconds",
        "pick_error_seconds",
        "absolute_pick_error_seconds",
        "availability_latency_seconds",
        "magnitude",
        "epicentral_distance_km",
    })].isna().any().any():
        raise RuntimeError("Required result fields contain non-numeric values")
    return frame


def record_keys(frame: pd.DataFrame) -> list[tuple[str, str, str]]:
    return sorted(
        zip(
            frame["source_index"].astype(str),
            frame["trace_name"].astype(str),
            frame["event_id"].astype(str),
            strict=True,
        )
    )


def validate_result_identity(
    frame: pd.DataFrame,
    manifest: pd.DataFrame,
    *,
    protocol: Mapping[str, Any],
    dataset: str,
) -> int:
    keys = ["source_index", "trace_name", "event_id"]
    left = frame.sort_values(keys).reset_index(drop=True)
    right = manifest.sort_values(keys).reset_index(drop=True)
    for field in ("source_index", "trace_name", "event_id", "station_id"):
        if not left[field].astype(str).equals(right[field].astype(str)):
            raise RuntimeError(
                f"{dataset} result identity differs: {field}"
            )
    numeric_pairs = {
        "magnitude": "magnitude",
        "epicentral_distance_km": "epicentral_distance_km",
        "p_arrival_sample": "p_arrival_sample",
        "sample_rate_hz": "sample_rate_hz",
        "manifest_n_samples": "n_samples",
    }
    for result_field, manifest_field in numeric_pairs.items():
        result_values = pd.to_numeric(
            left[result_field], errors="raise"
        ).to_numpy(dtype=float)
        manifest_values = pd.to_numeric(
            right[manifest_field], errors="raise"
        ).to_numpy(dtype=float)
        if not np.allclose(
            result_values,
            manifest_values,
            rtol=0.0,
            atol=1e-9,
            equal_nan=False,
        ):
            raise RuntimeError(
                f"{dataset} result metadata differs: {result_field}"
            )
    expected_constants = {
        "dataset": dataset,
        "model_key": MODEL_KEY,
        "selected_arm": str(protocol["model"]["selected_arm"]),
        "primary_seed": int(protocol["model"]["primary_seed"]),
        "snr_definition_id": str(
            protocol["metrics"]["local_p_snr_definition"]
        ),
    }
    for field, expected in expected_constants.items():
        values = set(left[field].tolist())
        if values != {expected}:
            raise RuntimeError(
                f"{dataset} result constant differs: {field}"
            )
    expected_precontext = (
        left["p_arrival_sample"] / left["sample_rate_hz"]
    )
    if not np.allclose(
        left["p_precontext_seconds"],
        expected_precontext,
        rtol=0.0,
        atol=1e-9,
    ):
        raise RuntimeError(f"{dataset} P precontext differs")
    expected_stratum = np.where(
        expected_precontext < 10.0,
        "lt_10_seconds",
        "ge_10_seconds",
    )
    if not np.array_equal(
        left["p_precontext_stratum"].to_numpy(dtype=str),
        expected_stratum,
    ):
        raise RuntimeError(f"{dataset} P precontext stratum differs")
    hashes = left["waveform_sha256"].astype(str)
    hash_present = hashes.str.fullmatch(r"[0-9a-f]{64}")
    if not (
        hash_present
        | (
            left["technical_failure"].eq(1)
            & hashes.eq("")
            & left["evaluated_n_samples"].eq(0)
        )
    ).all():
        raise RuntimeError(f"{dataset} waveform digest differs")
    if (
        left.loc[left["technical_failure"].eq(0), "evaluated_n_samples"]
        .ne(left.loc[
            left["technical_failure"].eq(0), "manifest_n_samples"
        ])
        .any()
    ):
        raise RuntimeError(
            f"{dataset} evaluated sample count differs"
        )
    reference_path = verify_artifact(
        protocol["selected_waveform_hash_references"][dataset],
        require_sha256=True,
        require_mtime=False,
    )
    reference = pd.read_csv(
        reference_path,
        dtype=str,
        keep_default_na=False,
    )
    reference_required = set(keys + ["waveform_sha256"])
    if reference_required.difference(reference.columns):
        raise RuntimeError(
            f"{dataset} waveform-hash reference lacks required columns"
        )
    grouped = reference.groupby(keys, sort=True, dropna=False)[
        "waveform_sha256"
    ].agg(lambda values: sorted(set(values)))
    if any(len(values) != 1 for values in grouped):
        raise RuntimeError(
            f"{dataset} waveform-hash reference is inconsistent"
        )
    expected_hashes = {
        tuple(str(value) for value in key): values[0]
        for key, values in grouped.items()
    }
    observed_keys = {
        tuple(str(row[field]) for field in keys)
        for _, row in left.iterrows()
    }
    if set(expected_hashes) != observed_keys:
        raise RuntimeError(
            f"{dataset} waveform-hash reference identities differ"
        )
    unverified = 0
    for _, row in left.iterrows():
        key = tuple(str(row[field]) for field in keys)
        observed = str(row["waveform_sha256"])
        if not observed:
            unverified += 1
        elif observed != expected_hashes[key]:
            raise RuntimeError(
                f"{dataset} waveform content differs: {key}"
            )
    return unverified


def load_run(
    protocol: Mapping[str, Any],
    *,
    protocol_sha256: str,
    dataset: str,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    allowed_root = Path(
        protocol["execution"]["allowed_output_root"]
    ).resolve()
    run_root = require_output_path(
        Path(protocol["execution"]["score_output_root"]) / dataset,
        allowed_root,
    )
    summary_path = run_root / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected = {
        "summary_id": "cspp_v27c_fixed_retrospective_score_v1",
        "protocol_sha256": protocol_sha256,
        "dataset": dataset,
        "model_key": MODEL_KEY,
        "selected_arm": protocol["model"]["selected_arm"],
        "primary_seed": int(protocol["model"]["primary_seed"]),
        "records_requested": int(
            protocol["fixed_test_cohorts"][dataset]["records"]
        ),
        "records_emitted": int(
            protocol["fixed_test_cohorts"][dataset]["records"]
        ),
        "operating_point": protocol["operating_point"],
        "checkpoint": protocol["model"]["checkpoint"],
        "fixed_test_manifest": protocol["fixed_test_records"][dataset],
        "source_store": protocol["source_stores"][
            (
                "knet_waveforms"
                if dataset == "knet"
                else "instancegm_waveforms"
            )
        ],
        "waveform_hash_reference": protocol[
            "selected_waveform_hash_references"
        ][dataset],
        "device": "cpu",
        "torch_threads": int(protocol["execution"]["torch_threads"]),
        "workers": 1,
        "output_fields": OUTPUT_FIELDS,
    }
    for key, value in expected.items():
        if summary.get(key) != value:
            raise RuntimeError(f"Run summary differs for {dataset}:{key}")
    records_path = verify_artifact(
        summary["records_artifact"],
        require_sha256=True,
        require_mtime=False,
    )
    if records_path != (run_root / "records.csv").resolve():
        raise RuntimeError(
            f"{dataset} records artifact is outside its fixed run directory"
        )
    frame = read_result_csv(records_path)
    if len(frame) != expected["records_requested"]:
        raise RuntimeError(f"{dataset} result row count differs")
    if not frame["dataset"].eq(dataset).all():
        raise RuntimeError(f"{dataset} result contains another dataset")
    if not frame["model_key"].eq(MODEL_KEY).all():
        raise RuntimeError(f"{dataset} result contains another model")
    if frame.duplicated(
        ["source_index", "trace_name", "event_id"]
    ).any():
        raise RuntimeError(f"{dataset} result contains duplicate records")

    manifest_path = verify_artifact(
        protocol["fixed_test_records"][dataset],
        require_sha256=True,
        require_mtime=False,
    )
    manifest = pd.read_csv(
        manifest_path, dtype=str, keep_default_na=False
    )
    if record_keys(frame) != record_keys(manifest):
        raise RuntimeError(f"{dataset} result does not match its manifest")
    unverified_waveforms = validate_result_identity(
        frame,
        manifest,
        protocol=protocol,
        dataset=dataset,
    )
    validate_record_accounting(frame)
    if int(summary.get("technical_failures", -1)) != int(
        frame["technical_failure"].sum()
    ):
        raise RuntimeError(
            f"{dataset} technical-failure summary differs"
        )
    if int(
        summary.get(
            "waveform_identity_unverified_technical_failures", -1
        )
    ) != unverified_waveforms:
        raise RuntimeError(
            f"{dataset} waveform-identity summary differs"
        )
    execution_audit_path = fixed_runner.audit_path(protocol, dataset)
    fixed_runner.verify_audit(
        execution_audit_path,
        protocol=protocol,
        protocol_sha256=protocol_sha256,
        dataset=dataset,
    )
    return frame, summary, {
        "summary": artifact(summary_path),
        "execution_audit": artifact(execution_audit_path),
    }


def snr_strata(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    labels = (
        ("lt_0_db", frame["snr_bin"].eq("lt_0_db")),
        (
            "0_to_lt_10_db",
            frame["snr_bin"].eq("0_to_lt_10_db"),
        ),
        ("ge_10_db", frame["snr_bin"].eq("ge_10_db")),
        ("invalid", frame["snr_valid"].eq(0)),
    )
    output = {}
    for label, mask in labels:
        subset = frame.loc[mask].copy()
        if len(subset):
            output[label] = metric_values(subset)
    return output


def knet_precontext_strata(
    frame: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    output = {}
    for label in ("lt_10_seconds", "ge_10_seconds"):
        subset = frame.loc[
            frame["p_precontext_stratum"].eq(label)
        ].copy()
        if len(subset):
            output[label] = metric_values(subset)
    return output


def parse_baseline_spec(value: str) -> tuple[str, str, Path]:
    parts = value.split(":", maxsplit=2)
    if len(parts) != 3:
        raise ValueError(
            "Baseline artifact must be DATASET:METHOD:CSV"
        )
    dataset, method, path = parts
    if dataset not in {"knet", "instancegm"}:
        raise ValueError(f"Unknown baseline dataset: {dataset}")
    return dataset, method, reject_forbidden_path(path)


def exact_mcnemar_p(cspp_only: int, baseline_only: int) -> float:
    discordant = int(cspp_only) + int(baseline_only)
    if discordant == 0:
        return 1.0
    smaller = min(int(cspp_only), int(baseline_only))
    lower_tail = sum(
        math.comb(discordant, value)
        for value in range(smaller + 1)
    ) / (2.0**discordant)
    return min(1.0, 2.0 * lower_tail)


def paired_event_cluster_bootstrap(
    proposed: pd.DataFrame,
    baseline: pd.DataFrame,
    *,
    seed: int,
    replicates: int,
) -> dict[str, Any]:
    keys = ["source_index", "trace_name", "event_id"]
    paired = proposed[keys + ["true_positive"]].merge(
        baseline[keys + ["true_positive"]],
        on=keys,
        how="inner",
        validate="one_to_one",
        suffixes=("_cspp_v27c", "_baseline"),
    )
    if len(paired) != len(proposed) or len(paired) != len(baseline):
        raise RuntimeError("Paired public baseline record sets differ")
    paired["difference"] = (
        pd.to_numeric(
            paired["true_positive_cspp_v27c"], errors="raise"
        )
        - pd.to_numeric(
            paired["true_positive_baseline"], errors="raise"
        )
    )
    grouped = paired.groupby("event_id", sort=True)["difference"]
    sums = grouped.sum().to_numpy(dtype=float)
    sizes = grouped.size().to_numpy(dtype=float)
    if not len(sums) or replicates <= 0:
        raise RuntimeError("Paired event bootstrap has no valid clusters")
    rng = np.random.default_rng(seed)
    sampled = rng.integers(
        0, len(sums), size=(replicates, len(sums))
    )
    values = sums[sampled].sum(axis=1) / sizes[sampled].sum(axis=1)
    return {
        "estimate": float(paired["difference"].mean()),
        "lower_95": float(
            np.quantile(values, 0.025, method="linear")
        ),
        "upper_95": float(
            np.quantile(values, 0.975, method="linear")
        ),
        "replicates": int(replicates),
        "valid_replicates": int(len(values)),
        "cluster": "event_id",
        "seed": int(seed),
    }


def compare_public_endpoint(
    proposed: pd.DataFrame,
    baseline: pd.DataFrame,
    *,
    dataset: str,
    method: str,
    bootstrap_seed: int | None = None,
    bootstrap_replicates: int | None = None,
) -> dict[str, Any]:
    required = {
        "dataset",
        "method_id",
        "source_index",
        "trace_name",
        "event_id",
        "true_positive",
    }
    missing = required.difference(baseline.columns)
    if missing:
        raise RuntimeError(
            f"Public baseline lacks columns: {sorted(missing)}"
        )
    if not baseline["dataset"].astype(str).eq(dataset).all():
        raise RuntimeError("Public baseline dataset differs")
    if not baseline["method_id"].astype(str).eq(method).all():
        raise RuntimeError("Public baseline method differs")
    baseline = baseline.copy()
    baseline["true_positive"] = pd.to_numeric(
        baseline["true_positive"], errors="raise"
    )
    if not baseline["true_positive"].isin([0, 1]).all():
        raise RuntimeError("Public baseline endpoint must be binary")
    if baseline.duplicated(
        ["source_index", "trace_name", "event_id"]
    ).any():
        raise RuntimeError("Public baseline contains duplicate records")
    if record_keys(proposed) != record_keys(baseline):
        raise RuntimeError(
            "Public baseline does not use the identical fixed manifest"
        )

    keys = ["source_index", "trace_name", "event_id"]
    left = proposed[keys + ["true_positive"]].copy()
    right = baseline[keys + ["true_positive"]].copy()
    for frame in (left, right):
        for key in keys:
            frame[key] = frame[key].astype(str)
    paired = left.merge(
        right,
        on=keys,
        how="inner",
        validate="one_to_one",
        suffixes=("_cspp_v27c", "_baseline"),
    )
    cspp_only = int(
        (
            paired["true_positive_cspp_v27c"].eq(1)
            & paired["true_positive_baseline"].eq(0)
        ).sum()
    )
    baseline_only = int(
        (
            paired["true_positive_cspp_v27c"].eq(0)
            & paired["true_positive_baseline"].eq(1)
        ).sum()
    )
    result = {
        "dataset": dataset,
        "method_id": method,
        "records": int(len(paired)),
        "cspp_v27c_recall": float(
            paired["true_positive_cspp_v27c"].mean()
        ),
        "baseline_recall": float(
            paired["true_positive_baseline"].mean()
        ),
        "recall_difference_cspp_v27c_minus_baseline": float(
            (
                paired["true_positive_cspp_v27c"]
                - paired["true_positive_baseline"]
            ).mean()
        ),
        "cspp_v27c_only_hits": cspp_only,
        "baseline_only_hits": baseline_only,
        "record_level_mcnemar_exact_two_sided_p_descriptive": (
            exact_mcnemar_p(cspp_only, baseline_only)
        ),
        "endpoint_only": True,
        "availability_latency_compared": False,
    }
    if bootstrap_seed is not None or bootstrap_replicates is not None:
        if bootstrap_seed is None or bootstrap_replicates is None:
            raise ValueError(
                "Bootstrap seed and replicate count are required together"
            )
        result["paired_recall_difference_event_cluster_bootstrap_95"] = (
            paired_event_cluster_bootstrap(
                proposed,
                baseline,
                seed=bootstrap_seed,
                replicates=bootstrap_replicates,
            )
        )
    return result


def public_endpoint_metric_values(
    baseline: pd.DataFrame,
) -> dict[str, Any]:
    required = {
        "true_positive",
        "false_positive",
        "false_negative",
    }
    missing = required.difference(baseline.columns)
    if missing:
        raise RuntimeError(
            f"Public baseline lacks candidate accounting: {sorted(missing)}"
        )
    numeric = baseline.copy()
    for column in required:
        numeric[column] = pd.to_numeric(
            numeric[column], errors="raise"
        )
    if (
        not numeric["true_positive"].isin([0, 1]).all()
        or not numeric["false_negative"].isin([0, 1]).all()
        or not (
            numeric["true_positive"] + numeric["false_negative"]
        ).eq(1).all()
        or not numeric["false_positive"].ge(0).all()
    ):
        raise RuntimeError("Public candidate accounting is invalid")
    tp = int(numeric["true_positive"].sum())
    fp = int(numeric["false_positive"].sum())
    fn = int(numeric["false_negative"].sum())
    precision = tp / (tp + fp) if tp + fp else math.nan
    recall = tp / (tp + fn) if tp + fn else 0.0
    denominator = 2 * tp + fp + fn
    f1 = 2.0 * tp / denominator if denominator else math.nan
    return {
        "records": int(len(numeric)),
        "true_positive": tp,
        "extra_event_record_candidates": fp,
        "false_negative": fn,
        "event_record_candidate_precision": float(precision),
        "record_recall": float(recall),
        "event_record_candidate_f1": float(f1),
        "continuous_false_alarm_interpretation_permitted": False,
    }


def validate_formal_public_rows(
    baseline: pd.DataFrame,
    proposed: pd.DataFrame,
    *,
    dataset: str,
    model: str,
    method: str,
) -> None:
    if list(baseline.columns) != formal_public.OUTPUT_FIELDS:
        raise RuntimeError("Formal public baseline output schema differs")
    if len(baseline) != len(proposed):
        raise RuntimeError("Formal public baseline row count differs")
    expected_constants = {
        "dataset": dataset,
        "model_key": model,
        "method_id": method,
        "comparison_mode": "released_full_record_annotation",
        "availability_comparable": "0",
    }
    for column, expected in expected_constants.items():
        if not baseline[column].astype(str).eq(expected).all():
            raise RuntimeError(
                f"Formal public baseline differs: {column}"
            )
    if baseline["technical_error"].astype(str).ne("").any():
        raise RuntimeError(
            "Formal public baseline has technical failures; comparison is "
            "noninferential"
        )
    integer_columns = (
        "candidate_count",
        "matched_candidate_count",
        "true_positive",
        "false_positive",
        "false_negative",
        "within_0p5_seconds",
        "within_1p0_seconds",
        "within_1p5_seconds",
    )
    numeric = {
        column: pd.to_numeric(baseline[column], errors="raise")
        for column in integer_columns
    }
    integer_accounting = all(
        np.isfinite(values.to_numpy(dtype=float)).all()
        and values.ge(0).all()
        and np.equal(
            values.to_numpy(dtype=float),
            np.floor(values.to_numpy(dtype=float)),
        ).all()
        for values in numeric.values()
    )
    if (
        not integer_accounting
        or not numeric["matched_candidate_count"].isin([0, 1]).all()
        or not (
            numeric["matched_candidate_count"]
            <= numeric["candidate_count"]
        ).all()
        or not numeric["true_positive"].isin([0, 1]).all()
        or not numeric["false_negative"].isin([0, 1]).all()
        or not numeric["false_positive"].ge(0).all()
        or not all(
            numeric[column].isin([0, 1]).all()
            for column in (
                "within_0p5_seconds",
                "within_1p0_seconds",
                "within_1p5_seconds",
            )
        )
        or not (
            numeric["true_positive"] + numeric["false_negative"]
        ).eq(1).all()
        or not numeric["false_positive"].eq(
            numeric["candidate_count"] - numeric["true_positive"]
        ).all()
        or not numeric["true_positive"].eq(
            numeric["matched_candidate_count"].gt(0).astype(int)
        ).all()
        or not (
            numeric["within_0p5_seconds"]
            <= numeric["within_1p0_seconds"]
        ).all()
        or not (
            numeric["within_1p0_seconds"]
            <= numeric["within_1p5_seconds"]
        ).all()
        or not numeric["within_1p5_seconds"].eq(
            numeric["true_positive"]
        ).all()
    ):
        raise RuntimeError("Formal public candidate accounting differs")

    public_identity = baseline.copy()
    proposed_identity = proposed.copy()
    for frame in (public_identity, proposed_identity):
        for column in ("source_index", "trace_name", "event_id"):
            frame[column] = frame[column].astype(str)
    identity = public_identity.merge(
        proposed_identity,
        on=["source_index", "trace_name", "event_id"],
        how="inner",
        validate="one_to_one",
        suffixes=("_public", "_cspp"),
    )
    if len(identity) != len(proposed):
        raise RuntimeError("Formal public record identities differ")
    exact_pairs = (
        ("station_id_public", "station_id_cspp"),
        ("waveform_sha256_public", "waveform_sha256_cspp"),
    )
    for left, right in exact_pairs:
        if not identity[left].astype(str).eq(
            identity[right].astype(str)
        ).all():
            raise RuntimeError(
                f"Formal public metadata differs: {left}"
            )
    numeric_pairs = (
        ("magnitude_public", "magnitude_cspp"),
        (
            "epicentral_distance_km_public",
            "epicentral_distance_km_cspp",
        ),
        ("p_arrival_sample_public", "p_arrival_sample_cspp"),
        ("sample_rate_hz_public", "sample_rate_hz_cspp"),
        ("n_samples", "evaluated_n_samples"),
    )
    for left, right in numeric_pairs:
        observed = pd.to_numeric(identity[left], errors="raise").to_numpy(
            dtype=float
        )
        expected = pd.to_numeric(identity[right], errors="raise").to_numpy(
            dtype=float
        )
        if not np.allclose(observed, expected, rtol=0.0, atol=1e-9):
            raise RuntimeError(
                f"Formal public numeric metadata differs: {left}"
            )


def load_public_baselines(
    specs: list[str],
    *,
    protocol: Mapping[str, Any],
    proposed: Mapping[str, pd.DataFrame],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    allowed_methods = protocol[
        "public_baseline_endpoint_interface"
    ]["methods"]
    comparisons: dict[str, Any] = {}
    artifacts = []
    seen: set[tuple[str, str]] = set()
    for value in specs:
        dataset, method, path = parse_baseline_spec(value)
        if method not in allowed_methods:
            raise RuntimeError(f"Unreserved public method: {method}")
        key = (dataset, method)
        if key in seen:
            raise RuntimeError(f"Duplicate public baseline: {key}")
        seen.add(key)
        baseline_artifact = artifact(path)
        baseline = pd.read_csv(path, dtype=str, keep_default_na=False)
        comparisons[f"{dataset}:{method}"] = compare_public_endpoint(
            proposed[dataset],
            baseline,
            dataset=dataset,
            method=method,
        )
        artifacts.append(
            {
                "dataset": dataset,
                "method_id": method,
                "artifact": baseline_artifact,
            }
        )
    return comparisons, artifacts


def load_provenance_sealed_public_baselines(
    protocol_path: Path,
    *,
    expected_protocol_sha256: str,
    retrospective_protocol_sha256: str,
    proposed: Mapping[str, pd.DataFrame],
    bootstrap_seed: int,
    bootstrap_replicates: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    public_protocol, public_protocol_sha256 = formal_public.load_protocol(
        protocol_path,
        expected_sha256=expected_protocol_sha256,
    )
    if (
        public_protocol["retrospective_protocol_sha256"]
        != retrospective_protocol_sha256
    ):
        raise RuntimeError(
            "Public baselines bind another retrospective protocol"
        )
    comparisons: dict[str, Any] = {}
    artifacts: list[dict[str, Any]] = []
    expected_units = {
        ("knet", "phasenet_instance"),
        ("knet", "eqtransformer_original"),
        ("instancegm", "phasenet_instance"),
        ("instancegm", "eqtransformer_original"),
    }
    observed_units: set[tuple[str, str]] = set()
    for dataset, model in sorted(expected_units):
        summary_path = (
            formal_public.run_root(public_protocol, dataset, model)
            / "summary.json"
        )
        summary = formal_public_runner.verify_summary(
            summary_path,
            protocol=public_protocol,
            protocol_sha256=public_protocol_sha256,
            dataset=dataset,
            model=model,
        )
        if int(summary["technical_failure_records"]) != 0:
            raise RuntimeError(
                "Formal public baseline has technical failures; model "
                "comparison is execution_incomplete_noninferential"
            )
        execution_audit_path = formal_public_runner.unit_audit_path(
            public_protocol, dataset, model
        )
        execution_audit = formal_public_runner.verify_unit_audit(
            execution_audit_path,
            protocol=public_protocol,
            protocol_sha256=public_protocol_sha256,
            dataset=dataset,
            model=model,
        )
        records_path = verify_artifact(
            summary["records_artifact"],
            require_sha256=True,
            require_mtime=False,
        )
        baseline_rows = pd.read_csv(
            records_path,
            dtype=str,
            keep_default_na=False,
        )
        expected_methods = formal_public_freeze.EXPECTED_METHODS[model]
        if set(baseline_rows["method_id"].astype(str)) != expected_methods:
            raise RuntimeError(
                f"Public baseline methods differ: {dataset}/{model}"
            )
        if "waveform_sha256" not in baseline_rows:
            raise RuntimeError("Public baseline lacks waveform identity")
        proposed_identity = proposed[dataset][
            [
                "source_index",
                "trace_name",
                "event_id",
                "waveform_sha256",
            ]
        ].copy()
        for method in sorted(expected_methods):
            baseline = baseline_rows.loc[
                baseline_rows["method_id"].astype(str).eq(method)
            ].copy()
            validate_formal_public_rows(
                baseline,
                proposed[dataset],
                dataset=dataset,
                model=model,
                method=method,
            )
            baseline_identity = baseline[
                [
                    "source_index",
                    "trace_name",
                    "event_id",
                    "waveform_sha256",
                ]
            ].copy()
            for frame in (proposed_identity, baseline_identity):
                for column in frame.columns:
                    frame[column] = frame[column].astype(str)
            identity = proposed_identity.merge(
                baseline_identity,
                on=["source_index", "trace_name", "event_id"],
                how="inner",
                validate="one_to_one",
                suffixes=("_cspp_v27c", "_public"),
            )
            if len(identity) != len(proposed_identity) or not identity[
                "waveform_sha256_cspp_v27c"
            ].eq(identity["waveform_sha256_public"]).all():
                raise RuntimeError(
                    "Public and CSPP waveform identities differ"
                )
            comparisons[f"{dataset}:{method}"] = compare_public_endpoint(
                proposed[dataset],
                baseline,
                dataset=dataset,
                method=method,
                bootstrap_seed=(
                    bootstrap_seed
                    + 10_000
                    + len(comparisons)
                ),
                bootstrap_replicates=bootstrap_replicates,
            )
            comparisons[f"{dataset}:{method}"][
                "baseline_endpoint_metrics"
            ] = public_endpoint_metric_values(baseline)
        observed_units.add((dataset, model))
        artifacts.append(
            {
                "dataset": dataset,
                "model_key": model,
                "summary": artifact(summary_path),
                "records": artifact(records_path),
                "execution_audit": artifact(execution_audit_path),
                "observed_peak_process_tree_rss_bytes": int(
                    execution_audit[
                        "observed_peak_process_tree_rss_bytes"
                    ]
                ),
                "technical_failure_records": int(
                    summary["technical_failure_records"]
                ),
            }
        )
    if observed_units != expected_units:
        raise RuntimeError("Provenance-sealed public run set is incomplete")
    provenance = {
        "status": "provenance_sealed_same_manifest_public_baselines",
        "protocol": artifact(protocol_path),
        "protocol_sha256": public_protocol_sha256,
        "retrospective_protocol_sha256": (
            public_protocol["retrospective_protocol_sha256"]
        ),
        "public_model_artifacts": {
            model: spec["artifacts"]
            for model, spec in public_protocol["public_models"].items()
        },
        "software_versions": public_protocol["software_versions"],
        "endpoint": public_protocol["endpoint"],
        "claim_boundary": public_protocol["claim_boundary"],
    }
    return comparisons, artifacts, provenance


def write_json_once(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=False)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(
            json_ready(dict(payload)),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if bool(args.public_baseline_protocol) != bool(
        args.expected_public_baseline_protocol_sha256
    ):
        raise ValueError(
            "--public-baseline-protocol and "
            "--expected-public-baseline-protocol-sha256 are required together"
        )
    if args.public_baseline_protocol and args.public_baseline_artifact:
        raise ValueError(
            "Use either provenance-sealed public baselines or unverified "
            "user-supplied CSVs, not both"
        )
    protocol, protocol_hash = load_protocol(
        args.protocol,
        expected_sha256=args.expected_protocol_sha256,
    )
    analysis_root = require_output_path(
        protocol["execution"]["analysis_output_root"],
        Path(protocol["execution"]["allowed_output_root"]),
    )
    if analysis_root.exists():
        raise FileExistsError(analysis_root)

    frames: dict[str, pd.DataFrame] = {}
    summaries: dict[str, Any] = {}
    summary_artifacts: dict[str, Any] = {}
    metrics: dict[str, Any] = {}
    strata: dict[str, Any] = {}
    statistics = protocol["statistics"]
    seed = int(statistics["bootstrap_seed"])
    replicates = int(statistics["bootstrap_replicates"])
    for offset, dataset in enumerate(("knet", "instancegm")):
        frame, summary, summary_artifact = load_run(
            protocol,
            protocol_sha256=protocol_hash,
            dataset=dataset,
        )
        frames[dataset] = frame
        summaries[dataset] = summary
        summary_artifacts[dataset] = summary_artifact
        metrics[dataset] = {
            **metric_values(frame),
            "event_cluster_bootstrap_95": event_cluster_bootstrap(
                frame,
                seed=seed + offset,
                replicates=replicates,
            ),
        }
        strata[dataset] = {"snr": snr_strata(frame)}
        if dataset == "knet":
            strata[dataset]["p_precontext"] = knet_precontext_strata(
                frame
            )

    if args.public_baseline_protocol:
        (
            baseline_comparisons,
            baseline_artifacts,
            baseline_provenance,
        ) = load_provenance_sealed_public_baselines(
            args.public_baseline_protocol,
            expected_protocol_sha256=(
                args.expected_public_baseline_protocol_sha256
            ),
            retrospective_protocol_sha256=protocol_hash,
            proposed=frames,
            bootstrap_seed=int(
                protocol["statistics"]["bootstrap_seed"]
            ),
            bootstrap_replicates=int(
                protocol["statistics"]["bootstrap_replicates"]
            ),
        )
        baseline_status = (
            "provenance_sealed_same_manifest_public_baselines"
        )
        comparison_boundary = (
            "PhaseNet and EQTransformer use sealed released weights, "
            "public operating points, the identical fixed manifests, and "
            "verified per-record waveform identities. Comparisons are "
            "record-level P-pick endpoints only; public-model availability "
            "latency is not inferred from offline full-record annotation."
        )
    else:
        baseline_comparisons, baseline_artifacts = load_public_baselines(
            args.public_baseline_artifact,
            protocol=protocol,
            proposed=frames,
        )
        baseline_provenance = None
        baseline_status = (
            "user_supplied_same_manifest_endpoint_artifacts_compared_"
            "without_generation_provenance_verification"
            if baseline_artifacts
            else "omitted_no_provenance_sealed_artifact"
        )
        comparison_boundary = (
            "User-supplied PhaseNet/EQTransformer CSVs, when supplied, use "
            "binary same-manifest detection/P-pick endpoints only. Their "
            "generation provenance is not verified here, they cannot support "
            "a formal SOTA claim, and no availability-latency comparison is "
            "computed."
        )
    result = {
        "analysis_id": "cspp_v27c_fixed_retrospective_analysis_v1",
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_hash,
        "model": protocol["model"],
        "operating_point": protocol["operating_point"],
        "metrics": metrics,
        "stratified_metrics": strata,
        "statistics": statistics,
        "run_summaries": summaries,
        "run_summary_artifacts": summary_artifacts,
        "public_baseline_endpoint_comparisons": baseline_comparisons,
        "public_baseline_artifacts": baseline_artifacts,
        "public_baseline_provenance": baseline_provenance,
        "public_baseline_status": baseline_status,
        "comparison_boundary": comparison_boundary,
        "claim_boundary": protocol["claim_boundary"],
    }
    result_path = analysis_root / "result.json"
    write_json_once(result_path, result)
    print(
        json.dumps(
            {"result": artifact(result_path), "metrics": metrics},
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
