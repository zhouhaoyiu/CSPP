#!/usr/bin/env python3
"""Analyze one sealed CSPP V27C operating point on the GeoNet holdout."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import UTC, datetime
import json
import math
import os
from pathlib import Path
import statistics
import sys
from typing import Any, Iterable, Mapping

import numpy as np
from scipy.stats import chi2, norm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.evaluation import (  # noqa: E402
    analyze_v27_geonet_temporal_selection as development,
)
from scripts.evaluation import (  # noqa: E402
    build_v27_geonet_final_score_cache as score_builder,
)


WORK_ROOT = Path("workspace").resolve()
DEFAULT_PROTOCOL = (
    WORK_ROOT
    / "protocol_freeze/cspp_v27c_geonet_final_evaluation_protocol_20260729.json"
)
PROTOCOL_ID = "cspp_v27c_geonet_final_evaluation_v1"

EVENT_STATION_FIELDS = [
    "event_id",
    "station_key",
    "site_rank",
    "label_stratum",
    "model_key",
    "seed",
    "primary_seed",
    "technical_available",
    "magnitude",
    "epicentral_distance_km",
    "snr_bin",
    "local_p_snr_db",
    "candidate_count",
    "matched_candidate_count",
    "false_candidate_count",
    "false_negative",
    "selected_estimated_p_epoch",
    "selected_available_epoch",
    "pick_error_seconds",
    "absolute_pick_error_seconds",
    "availability_delay_seconds",
]
EVENT_LEVEL_FIELDS = [
    "event_id",
    "model_key",
    "seed",
    "primary_seed",
    "magnitude",
    "primary_reference_station_count",
    "technically_available_primary_station_count",
    "matched_primary_station_count",
    "technical_event_available",
    "catalog_event_any_station_hit",
]
BACKGROUND_STATION_FIELDS = [
    "quiet_rank",
    "date_utc",
    "station_key",
    "model_key",
    "seed",
    "primary_seed",
    "station_exposure_days",
    "candidate_count",
]
BACKGROUND_NETWORK_FIELDS = [
    "quiet_rank",
    "date_utc",
    "model_key",
    "seed",
    "primary_seed",
    "minimum_three_station_exposure_days",
    "network_confirmations",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--expected-protocol-sha256", required=True)
    return parser.parse_args()


def artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": development.sha256_file(path),
    }


def artifact_at(
    actual_path: Path,
    published_path: Path,
) -> dict[str, Any]:
    return {
        **artifact(actual_path),
        "path": str(published_path.resolve()),
    }


def verify_artifact(spec: Mapping[str, Any]) -> Path:
    return score_builder.verify_artifact(dict(spec))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv_once(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
    fields: list[str],
) -> None:
    if path.exists():
        raise FileExistsError(path)
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


def write_json_once(path: Path, payload: Any) -> None:
    if path.exists():
        raise FileExistsError(path)
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


def load_completion(
    protocol: dict[str, Any],
    protocol_sha256: str,
    scope: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output_root = Path(protocol["execution"]["score_output_root"])
    path = output_root / f"{scope}_score_cache_complete.json"
    completion = json.loads(path.read_text(encoding="utf-8"))
    execution = protocol["execution"]
    expected_units = int(
        protocol["expected_units"][
            "event_waveform_files"
            if scope == "event"
            else "background_scoring_dates"
        ]
    )
    if (
        completion.get("completion_id")
        != f"cspp_v27c_geonet_final_{scope}_score_cache_v1"
        or completion.get("protocol_sha256") != protocol_sha256
        or int(completion.get("units", -1)) != expected_units
        or len(completion.get("unit_summaries", [])) != expected_units
        or completion.get("inference_device")
        != execution["inference_device"]
        or int(completion.get("inference_processes", -1))
        != int(execution["inference_processes"])
        or completion.get("shared_core_digest")
        != protocol["shared_core_integrity"]["tensor_digest"]
    ):
        raise RuntimeError(f"Final score completion differs: {scope}")
    execution_audit_path = verify_artifact(completion["execution_audit"])
    score_builder.verify_final_execution_audit(
        protocol,
        protocol_sha256,
        scope,
        execution_audit_path,
    )
    summaries = []
    for spec in completion["unit_summaries"]:
        summary_path = verify_artifact(spec)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if (
            summary.get("protocol_sha256") != protocol_sha256
            or summary.get("scope") != scope
            or summary.get("status") != "complete"
        ):
            raise RuntimeError(f"Final unit summary differs: {summary_path}")
        for score_spec in summary["artifacts"].values():
            verify_artifact(score_spec)
        summaries.append(summary)
    return summaries, artifact(path)


def station_key(row: Mapping[str, Any]) -> str:
    return ".".join(
        (str(row["network"]), str(row["station"]), str(row["location"]))
    )


def label_stratum(row: Mapping[str, Any]) -> str:
    source = str(row["label_source"])
    weight = row.get("pick_time_weight")
    if source == "geonet_origin_associated_used_p_pick":
        if weight not in ("", None) and float(weight) > 0.0:
            return "authoritative_used_p_positive_weight"
        raise RuntimeError("Used-P station lacks a positive pick weight")
    if source == "geonet_origin_associated_zero_weight_p_pick":
        return "authoritative_p_zero_weight"
    if source == "taup_iasp91_fallback":
        return "taup_iasp91_fallback"
    raise RuntimeError(f"Unexpected final label source: {source}")


def optional_float(value: Any) -> float | None:
    if value in ("", None):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def model_identity(
    protocol: Mapping[str, Any],
) -> dict[str, tuple[int, bool]]:
    primary = int(protocol["seed_policy"]["primary_operational_seed"])
    output = {}
    for key, spec in protocol["models"].items():
        seed = int(spec["seed"])
        output[key] = (seed, seed == primary)
    if sorted(seed for seed, _ in output.values()) != [
        20261101,
        20261102,
        20261103,
    ]:
        raise RuntimeError("Final model seed identities differ")
    return output


def scored_event_rows(
    protocol: dict[str, Any],
    event_summaries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    models = model_identity(protocol)
    threshold = float(protocol["operating_point"]["event_threshold"])
    candidate_threshold = float(
        protocol["operating_point"]["candidate_threshold"]
    )
    runtime = protocol["runtime"]
    tolerance = float(protocol["metrics"]["match_tolerance_seconds"])
    rows = []
    for unit in event_summaries:
        for station in unit["station_results"]:
            cache_path = verify_artifact(station["score_cache"])
            with np.load(cache_path, allow_pickle=False) as cache:
                valid = cache["quality_valid"].astype(bool, copy=False)
                allowed = cache["candidate_allowed"].astype(
                    bool, copy=False
                )
                support = development.support_with_resets(
                    cache["candidate_probability"].astype(
                        np.float32, copy=False
                    ),
                    allowed,
                    valid,
                    candidate_threshold,
                    int(runtime["candidate_history_packets"]),
                )
                local_snr = station.get("local_p_snr")
                snr_db = (
                    optional_float(local_snr.get("snr_db"))
                    if local_snr is not None
                    else None
                )
                snr_group = snr_bin(snr_db)
                if (
                    local_snr is not None
                    and str(local_snr.get("snr_bin")) != snr_group
                ):
                    raise RuntimeError(
                        "Cached SNR bin differs from its numeric value"
                    )
                for model_key, (seed, is_primary) in models.items():
                    candidates = development.extract_candidates(
                        support=support,
                        valid=valid,
                        event_probability=cache[
                            f"event_probability__{model_key}"
                        ].astype(np.float32, copy=False),
                        arrival_age=cache[
                            "arrival_age_fraction"
                        ].astype(np.float32, copy=False),
                        event_threshold=threshold,
                        start_epoch=float(
                            station["metadata"]["start_epoch"]
                        ),
                        packet_seconds=float(
                            runtime["packet_duration_seconds"]
                        ),
                        max_arrival_age_packets=float(
                            runtime["max_arrival_age_packets"]
                        ),
                    )
                    reference = float(station["reference_p_epoch"])
                    matched = development.matched_candidate(
                        candidates, reference, tolerance
                    )
                    error = (
                        float(matched["estimated_p_epoch"]) - reference
                        if matched is not None
                        else None
                    )
                    rows.append(
                        {
                            "event_id": unit["unit_key"],
                            "station_key": station["station_key"],
                            "site_rank": int(station["site_rank"]),
                            "label_stratum": development.label_stratum(
                                station
                            ),
                            "model_key": model_key,
                            "seed": seed,
                            "primary_seed": int(is_primary),
                            "technical_available": 1,
                            "magnitude": float(station["magnitude"]),
                            "epicentral_distance_km": float(
                                station["epicentral_distance_km"]
                            ),
                            "snr_bin": snr_group,
                            "local_p_snr_db": snr_db,
                            "candidate_count": len(candidates),
                            "matched_candidate_count": int(
                                matched is not None
                            ),
                            "false_candidate_count": (
                                len(candidates)
                                - int(matched is not None)
                            ),
                            "false_negative": int(matched is None),
                            "selected_estimated_p_epoch": (
                                float(matched["estimated_p_epoch"])
                                if matched is not None
                                else None
                            ),
                            "selected_available_epoch": (
                                float(matched["available_epoch"])
                                if matched is not None
                                else None
                            ),
                            "pick_error_seconds": error,
                            "absolute_pick_error_seconds": (
                                abs(error) if error is not None else None
                            ),
                            "availability_delay_seconds": (
                                float(matched["available_epoch"])
                                - reference
                                if matched is not None
                                else None
                            ),
                        }
                    )
    return rows


def complete_event_station_denominator(
    protocol: dict[str, Any],
    scored: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    all_stations = read_csv(
        verify_artifact(protocol["inputs"]["event_all_stations"])
    )
    models = model_identity(protocol)
    all_station_keys = [
        (row["event_id"], station_key(row)) for row in all_stations
    ]
    if len(all_station_keys) != 640 or len(set(all_station_keys)) != 640:
        raise RuntimeError("Final station manifest identities differ")
    scorable_stations = read_csv(
        verify_artifact(protocol["inputs"]["event_scorable_stations"])
    )
    scorable_keys = [
        (row["event_id"], station_key(row)) for row in scorable_stations
    ]
    if (
        len(scorable_keys) != len(set(scorable_keys))
        or not set(scorable_keys).issubset(set(all_station_keys))
    ):
        raise RuntimeError("Final scorable-station identities differ")
    by_key = {
        (row["event_id"], row["station_key"], row["model_key"]): row
        for row in scored
    }
    if len(by_key) != len(scored):
        raise RuntimeError("Duplicate scored final station row")
    expected_scored = {
        (event_id, key, model_key)
        for event_id, key in scorable_keys
        for model_key in models
    }
    if set(by_key) != expected_scored:
        missing = len(expected_scored.difference(by_key))
        extra = len(set(by_key).difference(expected_scored))
        raise RuntimeError(
            "Scored final station identities differ from the frozen "
            f"scorable cohort: missing={missing}, extra={extra}"
        )
    output = []
    for station in all_stations:
        key = station_key(station)
        for model_key, (seed, is_primary) in models.items():
            existing = by_key.get(
                (station["event_id"], key, model_key)
            )
            if existing is not None:
                output.append(existing)
                continue
            output.append(
                {
                    "event_id": station["event_id"],
                    "station_key": key,
                    "site_rank": int(station["site_rank"]),
                    "label_stratum": label_stratum(station),
                    "model_key": model_key,
                    "seed": seed,
                    "primary_seed": int(is_primary),
                    "technical_available": 0,
                    "magnitude": float(station["magnitude"]),
                    "epicentral_distance_km": float(
                        station["epicentral_distance_km"]
                    ),
                    "snr_bin": "technical_unavailable",
                    "local_p_snr_db": None,
                    "candidate_count": 0,
                    "matched_candidate_count": 0,
                    "false_candidate_count": 0,
                    "false_negative": 1,
                    "selected_estimated_p_epoch": None,
                    "selected_available_epoch": None,
                    "pick_error_seconds": None,
                    "absolute_pick_error_seconds": None,
                    "availability_delay_seconds": None,
                }
            )
    if len(output) != 640 * 3:
        raise RuntimeError("Final station availability denominator differs")
    return output


def event_level_rows(
    protocol: dict[str, Any],
    station_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    events = read_csv(
        verify_artifact(protocol["inputs"]["event_all_events"])
    )
    event_ids = [row["event_id"] for row in events]
    if len(event_ids) != 64 or len(set(event_ids)) != 64:
        raise RuntimeError("Final event manifest identities differ")
    models = model_identity(protocol)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in station_rows:
        if row["label_stratum"] == (
            "authoritative_used_p_positive_weight"
        ):
            grouped[(row["event_id"], row["model_key"])].append(row)
    output = []
    for event in events:
        for model_key, (seed, is_primary) in models.items():
            rows = grouped[(event["event_id"], model_key)]
            if not rows:
                raise RuntimeError(
                    f"Final event has no primary P reference: "
                    f"{event['event_id']}"
                )
            available = sum(
                int(row["technical_available"]) for row in rows
            )
            matched = sum(
                int(row["matched_candidate_count"]) for row in rows
            )
            output.append(
                {
                    "event_id": event["event_id"],
                    "model_key": model_key,
                    "seed": seed,
                    "primary_seed": int(is_primary),
                    "magnitude": float(event["magnitude"]),
                    "primary_reference_station_count": len(rows),
                    "technically_available_primary_station_count": available,
                    "matched_primary_station_count": matched,
                    "technical_event_available": int(available > 0),
                    "catalog_event_any_station_hit": int(matched > 0),
                }
            )
    if len(output) != 64 * 3:
        raise RuntimeError("Final event denominator differs")
    return output


def wilson_interval(
    successes: int,
    trials: int,
    confidence: float = 0.95,
) -> dict[str, float | int | None]:
    if trials <= 0:
        return {
            "successes": int(successes),
            "trials": int(trials),
            "estimate": None,
            "lower": None,
            "upper": None,
        }
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    z = float(norm.ppf((1.0 + confidence) / 2.0))
    value = successes / trials
    scale = 1.0 + z * z / trials
    center = (value + z * z / (2.0 * trials)) / scale
    half = (
        z
        * math.sqrt(
            value * (1.0 - value) / trials
            + z * z / (4.0 * trials * trials)
        )
        / scale
    )
    return {
        "successes": int(successes),
        "trials": int(trials),
        "estimate": value,
        "lower": max(0.0, center - half),
        "upper": min(1.0, center + half),
        "confidence": confidence,
        "method": "Wilson score",
    }


def poisson_rate_interval(
    count: int,
    exposure: float,
    confidence: float = 0.95,
) -> dict[str, float | int | None]:
    if exposure <= 0.0:
        return {
            "count": int(count),
            "exposure": float(exposure),
            "rate": None,
            "lower": None,
            "upper": None,
        }
    alpha = 1.0 - confidence
    lower_count = (
        0.0 if count == 0 else chi2.ppf(alpha / 2.0, 2 * count) / 2.0
    )
    upper_count = chi2.ppf(
        1.0 - alpha / 2.0, 2 * (count + 1)
    ) / 2.0
    return {
        "count": int(count),
        "exposure": float(exposure),
        "rate": count / exposure,
        "lower": lower_count / exposure,
        "upper": upper_count / exposure,
        "confidence": confidence,
        "method": "exact Poisson chi-square",
    }


def finite_median(values: Iterable[Any]) -> float | None:
    finite = [
        float(value)
        for value in values
        if value not in ("", None) and math.isfinite(float(value))
    ]
    return float(statistics.median(finite)) if finite else None


def seed_metrics(
    station_rows: list[dict[str, Any]],
    event_rows: list[dict[str, Any]],
    *,
    confidence: float,
) -> dict[str, Any]:
    primary = [
        row
        for row in station_rows
        if row["label_stratum"]
        == "authoritative_used_p_positive_weight"
    ]
    technical = [
        row for row in primary if int(row["technical_available"])
    ]
    matched_technical = sum(
        int(row["matched_candidate_count"]) for row in technical
    )
    candidates = sum(int(row["candidate_count"]) for row in technical)
    matched_all = sum(
        int(row["matched_candidate_count"]) for row in primary
    )
    detected_events = sum(
        int(row["catalog_event_any_station_hit"]) for row in event_rows
    )
    available_events = [
        row for row in event_rows if int(row["technical_event_available"])
    ]
    detected_available_events = sum(
        int(row["catalog_event_any_station_hit"])
        for row in available_events
    )
    precision = wilson_interval(
        matched_technical, candidates, confidence
    )
    technical_recall = wilson_interval(
        matched_technical, len(technical), confidence
    )
    availability_recall = wilson_interval(
        matched_all, len(primary), confidence
    )
    return {
        "primary_reference_stations": len(primary),
        "technically_available_primary_stations": len(technical),
        "positive_weight_reference_candidate_count": candidates,
        "technical_station_availability": wilson_interval(
            len(technical), len(primary), confidence
        ),
        "model_recall_on_technically_available_stations_"
        "descriptive_wilson": technical_recall,
        "availability_aware_station_recall_descriptive_wilson": (
            availability_recall
        ),
        "candidate_precision_on_positive_weight_reference_stations_"
        "descriptive_wilson": precision,
        "station_f1_on_technically_available_stations": (
            2.0
            * float(technical_recall["estimate"])
            * float(precision["estimate"])
            / (
                float(technical_recall["estimate"])
                + float(precision["estimate"])
            )
            if technical_recall["estimate"] is not None
            and precision["estimate"] is not None
            and float(technical_recall["estimate"])
            + float(precision["estimate"])
            > 0.0
            else 0.0
        ),
        "event_availability": wilson_interval(
            len(available_events), len(event_rows), confidence
        ),
        "catalog_event_any_station_hit_when_technically_available": (
            wilson_interval(
                detected_available_events,
                len(available_events),
                confidence,
            )
        ),
        "availability_aware_catalog_event_any_station_hit": wilson_interval(
            detected_events, len(event_rows), confidence
        ),
        "median_absolute_pick_error_seconds": finite_median(
            row["absolute_pick_error_seconds"] for row in technical
        ),
        "median_availability_delay_seconds": finite_median(
            row["availability_delay_seconds"] for row in technical
        ),
        "false_candidate_count": sum(
            int(row["false_candidate_count"]) for row in technical
        ),
    }


def station_label_metrics(
    rows: list[dict[str, Any]],
    *,
    confidence: float = 0.95,
) -> dict[str, Any]:
    technical = [
        row for row in rows if int(row["technical_available"])
    ]
    matched_technical = sum(
        int(row["matched_candidate_count"]) for row in technical
    )
    matched_all = sum(
        int(row["matched_candidate_count"]) for row in rows
    )
    candidates = sum(int(row["candidate_count"]) for row in technical)
    return {
        "stations": len(rows),
        "technically_available_stations": len(technical),
        "technical_availability": wilson_interval(
            len(technical), len(rows), confidence
        ),
        "technical_station_recall_descriptive_wilson": wilson_interval(
            matched_technical, len(technical), confidence
        ),
        "availability_aware_station_recall_descriptive_wilson": (
            wilson_interval(matched_all, len(rows), confidence)
        ),
        "candidate_precision_on_this_reference_stratum_"
        "descriptive_wilson": wilson_interval(
            matched_technical, candidates, confidence
        ),
        "median_absolute_pick_error_seconds": finite_median(
            row["absolute_pick_error_seconds"] for row in technical
        ),
    }


def event_cluster_bootstrap(
    station_rows: list[dict[str, Any]],
    event_rows: list[dict[str, Any]],
    *,
    seed: int,
    replicates: int,
    confidence: float = 0.95,
) -> dict[str, Any]:
    event_ids = sorted({row["event_id"] for row in event_rows})
    stations_by_event: dict[str, list[dict[str, Any]]] = defaultdict(list)
    event_by_id = {row["event_id"]: row for row in event_rows}
    for row in station_rows:
        if row["label_stratum"] == (
            "authoritative_used_p_positive_weight"
        ):
            stations_by_event[row["event_id"]].append(row)
    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = defaultdict(list)
    for _ in range(replicates):
        sampled = rng.choice(event_ids, size=len(event_ids), replace=True)
        selected_stations = [
            row for event_id in sampled for row in stations_by_event[event_id]
        ]
        technical = [
            row
            for row in selected_stations
            if int(row["technical_available"])
        ]
        matches_all = sum(
            int(row["matched_candidate_count"])
            for row in selected_stations
        )
        matches_technical = sum(
            int(row["matched_candidate_count"]) for row in technical
        )
        candidates = sum(
            int(row["candidate_count"]) for row in technical
        )
        values["availability_aware_station_recall"].append(
            matches_all / len(selected_stations)
            if selected_stations
            else 0.0
        )
        if technical:
            values["technical_station_recall"].append(
                matches_technical / len(technical)
            )
        if candidates:
            values[
                "candidate_precision_on_positive_weight_reference_stations"
            ].append(matches_technical / candidates)
        absolute_errors = [
            float(value)
            for row in technical
            if (
                value := optional_float(
                    row.get("absolute_pick_error_seconds")
                )
            )
            is not None
        ]
        availability_delays = [
            float(value)
            for row in technical
            if (
                value := optional_float(
                    row.get("availability_delay_seconds")
                )
            )
            is not None
        ]
        if absolute_errors:
            values["median_absolute_pick_error_seconds"].append(
                float(statistics.median(absolute_errors))
            )
        if availability_delays:
            values["median_availability_delay_seconds"].append(
                float(statistics.median(availability_delays))
            )
        values["availability_aware_catalog_event_any_station_hit"].append(
            sum(
                int(
                    event_by_id[event_id][
                        "catalog_event_any_station_hit"
                    ]
                )
                for event_id in sampled
            )
            / len(sampled)
        )
    alpha = 1.0 - confidence
    return {
        "cluster": "event_id",
        "seed": seed,
        "replicates": replicates,
        "confidence": confidence,
        "intervals": {
            name: {
                "lower": float(np.quantile(sample, alpha / 2.0)),
                "upper": float(
                    np.quantile(sample, 1.0 - alpha / 2.0)
                ),
                "valid_replicates": len(sample),
            }
            for name, sample in sorted(values.items())
        },
    }


def station_event_cluster_bootstrap(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    replicates: int,
    confidence: float = 0.95,
) -> dict[str, Any]:
    by_event: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_event[row["event_id"]].append(row)
    event_ids = sorted(by_event)
    if not event_ids:
        return {
            "cluster": "event_id",
            "seed": seed,
            "replicates": replicates,
            "confidence": confidence,
            "event_clusters": 0,
            "intervals": {},
        }
    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = defaultdict(list)
    for _ in range(replicates):
        sampled = rng.choice(event_ids, size=len(event_ids), replace=True)
        selected = [
            row for event_id in sampled for row in by_event[event_id]
        ]
        technical = [
            row for row in selected if int(row["technical_available"])
        ]
        matched_all = sum(
            int(row["matched_candidate_count"]) for row in selected
        )
        matched_technical = sum(
            int(row["matched_candidate_count"]) for row in technical
        )
        candidates = sum(
            int(row["candidate_count"]) for row in technical
        )
        if selected:
            values["technical_availability"].append(
                len(technical) / len(selected)
            )
            values["availability_aware_station_recall"].append(
                matched_all / len(selected)
            )
        if technical:
            values["technical_station_recall"].append(
                matched_technical / len(technical)
            )
        if candidates:
            values["candidate_precision"].append(
                matched_technical / candidates
            )
    alpha = 1.0 - confidence
    return {
        "cluster": "event_id",
        "seed": seed,
        "replicates": replicates,
        "confidence": confidence,
        "event_clusters": len(event_ids),
        "intervals": {
            name: {
                "lower": float(np.quantile(sample, alpha / 2.0)),
                "upper": float(
                    np.quantile(sample, 1.0 - alpha / 2.0)
                ),
                "valid_replicates": len(sample),
            }
            for name, sample in sorted(values.items())
        },
    }


def stratum_metrics(
    rows: list[dict[str, Any]],
    *,
    field: str,
    bootstrap_seed: int,
    bootstrap_replicates: int,
    confidence: float = 0.95,
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[field])].append(row)
    output = {}
    for name, values in sorted(grouped.items()):
        available = [
            row for row in values if int(row["technical_available"])
        ]
        output[name] = {
            "stations": len(values),
            "technically_available": len(available),
            "descriptive_wilson": {
                "availability_aware_recall": wilson_interval(
                    sum(
                        int(row["matched_candidate_count"])
                        for row in values
                    ),
                    len(values),
                    confidence,
                ),
                "technical_recall": wilson_interval(
                    sum(
                        int(row["matched_candidate_count"])
                        for row in available
                    ),
                    len(available),
                    confidence,
                ),
            },
            "event_cluster_bootstrap": station_event_cluster_bootstrap(
                values,
                seed=bootstrap_seed,
                replicates=bootstrap_replicates,
                confidence=confidence,
            ),
        }
    return output


def magnitude_bin(value: float) -> str:
    if not math.isfinite(value) or value < 3.0:
        raise RuntimeError(f"Magnitude is outside the frozen scope: {value}")
    if value < 4.0:
        return "3_to_lt_4"
    if value < 5.0:
        return "4_to_lt_5"
    return "ge_5"


def distance_bin(value: float) -> str:
    if not math.isfinite(value) or not 0.0 <= value <= 200.0:
        raise RuntimeError(
            f"Epicentral distance is outside the frozen scope: {value}"
        )
    if value <= 50.0:
        return "0_to_50"
    if value <= 100.0:
        return "gt_50_to_100"
    return "gt_100_to_200"


def snr_bin(value: float | None) -> str:
    if value is None:
        return "invalid"
    if not math.isfinite(value):
        raise RuntimeError(f"Non-finite SNR value: {value}")
    if value < 0.0:
        return "lt_0_db"
    if value < 10.0:
        return "0_to_lt_10_db"
    return "ge_10_db"


def background_rows(
    protocol: dict[str, Any],
    summaries: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    models = model_identity(protocol)
    point = protocol["operating_point"]
    runtime = protocol["runtime"]
    station_output = []
    network_output = []
    network_contract = protocol["metrics"]["network_confirmation"]
    for unit in summaries:
        candidates_by_model: dict[
            str, dict[str, list[dict[str, float | int]]]
        ] = {key: {} for key in models}
        for station in unit["station_results"]:
            cache_path = verify_artifact(station["score_cache"])
            with np.load(cache_path, allow_pickle=False) as cache:
                valid = cache["quality_valid"].astype(bool, copy=False)
                support = development.support_with_resets(
                    cache["candidate_probability"].astype(
                        np.float32, copy=False
                    ),
                    cache["candidate_allowed"].astype(bool, copy=False),
                    valid,
                    float(point["candidate_threshold"]),
                    int(runtime["candidate_history_packets"]),
                )
                for model_key, (seed, is_primary) in models.items():
                    candidates = development.extract_candidates(
                        support=support,
                        valid=valid,
                        event_probability=cache[
                            f"event_probability__{model_key}"
                        ].astype(np.float32, copy=False),
                        arrival_age=cache[
                            "arrival_age_fraction"
                        ].astype(np.float32, copy=False),
                        event_threshold=float(point["event_threshold"]),
                        start_epoch=float(
                            station["metadata"]["start_epoch"]
                        ),
                        packet_seconds=float(
                            runtime["packet_duration_seconds"]
                        ),
                        max_arrival_age_packets=float(
                            runtime["max_arrival_age_packets"]
                        ),
                    )
                    candidates_by_model[model_key][
                        station["station_key"]
                    ] = candidates
                    station_output.append(
                        {
                            "quiet_rank": int(unit["quiet_rank"]),
                            "date_utc": unit["unit_key"],
                            "station_key": station["station_key"],
                            "model_key": model_key,
                            "seed": seed,
                            "primary_seed": int(is_primary),
                            "station_exposure_days": float(
                                station["common_availability"]
                            ),
                            "candidate_count": len(candidates),
                        }
                    )
        for model_key, (seed, is_primary) in models.items():
            count = development.network_confirmation_count(
                candidates_by_model[model_key],
                window_seconds=float(
                    network_contract["estimated_p_window_seconds"]
                ),
                minimum_stations=int(
                    network_contract["minimum_distinct_stations"]
                ),
            )
            network_output.append(
                {
                    "quiet_rank": int(unit["quiet_rank"]),
                    "date_utc": unit["unit_key"],
                    "model_key": model_key,
                    "seed": seed,
                    "primary_seed": int(is_primary),
                    "minimum_three_station_exposure_days": float(
                        unit[
                            "post_quarantine_minimum_three_station_"
                            "exposure_days"
                        ]
                    ),
                    "network_confirmations": count,
                }
            )
    return station_output, network_output


def background_seed_metrics(
    stations: list[dict[str, Any]],
    networks: list[dict[str, Any]],
    *,
    confidence: float = 0.95,
) -> dict[str, Any]:
    station_exposure = sum(
        float(row["station_exposure_days"]) for row in stations
    )
    candidate_count = sum(int(row["candidate_count"]) for row in stations)
    network_exposure = sum(
        float(row["minimum_three_station_exposure_days"])
        for row in networks
    )
    confirmations = sum(
        int(row["network_confirmations"]) for row in networks
    )
    return {
        "quiet_background_candidate_rate_per_station_day": (
            poisson_rate_interval(
                candidate_count, station_exposure, confidence
            )
        ),
        "quiet_background_network_confirmation_rate_per_"
        "minimum_three_station_day": (
            poisson_rate_interval(
                confirmations, network_exposure, confidence
            )
        ),
        "scoring_dates": len(networks),
        "station_date_records": len(stations),
        "station_exposure_days": station_exposure,
        "minimum_three_station_exposure_days": network_exposure,
    }


def background_date_cluster_bootstrap(
    stations: list[dict[str, Any]],
    networks: list[dict[str, Any]],
    *,
    seed: int,
    replicates: int,
    confidence: float = 0.95,
) -> dict[str, Any]:
    station_by_date: dict[str, list[dict[str, Any]]] = defaultdict(list)
    network_by_date: dict[str, dict[str, Any]] = {}
    for row in stations:
        station_by_date[row["date_utc"]].append(row)
    for row in networks:
        date = row["date_utc"]
        if date in network_by_date:
            raise RuntimeError(f"Duplicate background network date: {date}")
        network_by_date[date] = row
    dates = sorted(station_by_date)
    if not dates or set(dates) != set(network_by_date):
        raise RuntimeError("Background station/network date identities differ")
    candidate_counts = np.asarray(
        [
            sum(int(row["candidate_count"]) for row in station_by_date[date])
            for date in dates
        ],
        dtype=float,
    )
    station_exposures = np.asarray(
        [
            sum(
                float(row["station_exposure_days"])
                for row in station_by_date[date]
            )
            for date in dates
        ],
        dtype=float,
    )
    confirmation_counts = np.asarray(
        [int(network_by_date[date]["network_confirmations"]) for date in dates],
        dtype=float,
    )
    network_exposures = np.asarray(
        [
            float(
                network_by_date[date][
                    "minimum_three_station_exposure_days"
                ]
            )
            for date in dates
        ],
        dtype=float,
    )
    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = defaultdict(list)
    for _ in range(replicates):
        sampled = rng.integers(0, len(dates), size=len(dates))
        station_exposure = float(station_exposures[sampled].sum())
        network_exposure = float(network_exposures[sampled].sum())
        if station_exposure > 0.0:
            values["candidate_rate_per_station_day"].append(
                float(candidate_counts[sampled].sum()) / station_exposure
            )
        if network_exposure > 0.0:
            values[
                "network_confirmation_rate_per_minimum_three_station_day"
            ].append(
                float(confirmation_counts[sampled].sum())
                / network_exposure
            )
    alpha = 1.0 - confidence
    return {
        "cluster": "date_utc",
        "seed": seed,
        "replicates": replicates,
        "confidence": confidence,
        "intervals": {
            name: {
                "lower": float(np.quantile(sample, alpha / 2.0)),
                "upper": float(
                    np.quantile(sample, 1.0 - alpha / 2.0)
                ),
                "valid_replicates": len(sample),
            }
            for name, sample in sorted(values.items())
        },
    }


def main() -> None:
    args = parse_args()
    protocol, protocol_sha256 = score_builder.load_protocol(
        args.protocol, args.expected_protocol_sha256
    )
    published_analysis_root = Path(
        protocol["execution"]["analysis_output_root"]
    ).resolve()
    if published_analysis_root.exists():
        raise FileExistsError(published_analysis_root)

    event_summaries, event_completion = load_completion(
        protocol, protocol_sha256, "event"
    )
    background_summaries, background_completion = load_completion(
        protocol, protocol_sha256, "background"
    )
    analysis_root = published_analysis_root.with_name(
        f".{published_analysis_root.name}.attempt-{os.getpid()}"
    )
    if analysis_root.exists():
        raise FileExistsError(analysis_root)
    analysis_root.mkdir(parents=True)
    scored = scored_event_rows(protocol, event_summaries)
    station_rows = complete_event_station_denominator(protocol, scored)
    event_rows = event_level_rows(protocol, station_rows)
    background_station, background_network = background_rows(
        protocol, background_summaries
    )
    confidence = float(protocol["statistics"]["confidence_level"])
    expected_strata = {
        "snr_bin_keys": [
            "lt_0_db",
            "0_to_lt_10_db",
            "ge_10_db",
            "invalid",
        ],
        "magnitude_bin_keys": [
            "3_to_lt_4",
            "4_to_lt_5",
            "ge_5",
        ],
        "distance_bin_keys": [
            "0_to_50",
            "gt_50_to_100",
            "gt_100_to_200",
        ],
    }
    for name, expected in expected_strata.items():
        if protocol["statistics"].get(name) != expected:
            raise RuntimeError(f"Frozen reporting strata differ: {name}")

    event_station_path = analysis_root / "event_station_results.csv"
    event_level_path = analysis_root / "event_level_results.csv"
    background_station_path = (
        analysis_root / "background_station_results.csv"
    )
    background_network_path = (
        analysis_root / "background_network_results.csv"
    )
    write_csv_once(
        event_station_path, station_rows, EVENT_STATION_FIELDS
    )
    write_csv_once(event_level_path, event_rows, EVENT_LEVEL_FIELDS)
    write_csv_once(
        background_station_path,
        background_station,
        BACKGROUND_STATION_FIELDS,
    )
    write_csv_once(
        background_network_path,
        background_network,
        BACKGROUND_NETWORK_FIELDS,
    )

    seed_results = {}
    models = model_identity(protocol)
    bootstrap_seed = int(protocol["statistics"]["bootstrap_seed"])
    bootstrap_replicates = int(
        protocol["statistics"]["bootstrap_replicates"]
    )
    for model_key, (seed, is_primary) in models.items():
        current_stations = [
            row for row in station_rows if row["model_key"] == model_key
        ]
        current_events = [
            row for row in event_rows if row["model_key"] == model_key
        ]
        current_background = [
            row
            for row in background_station
            if row["model_key"] == model_key
        ]
        current_network = [
            row
            for row in background_network
            if row["model_key"] == model_key
        ]
        primary_rows = [
            {**row}
            for row in current_stations
            if row["label_stratum"]
            == "authoritative_used_p_positive_weight"
        ]
        for row in primary_rows:
            row["magnitude_bin"] = magnitude_bin(float(row["magnitude"]))
            row["distance_bin"] = distance_bin(
                float(row["epicentral_distance_km"])
            )
        result = {
            "model_key": model_key,
            "seed": seed,
            "primary_seed": is_primary,
            "event_and_station_metrics": seed_metrics(
                current_stations,
                current_events,
                confidence=confidence,
            ),
            "event_cluster_bootstrap": event_cluster_bootstrap(
                current_stations,
                current_events,
                seed=bootstrap_seed + seed,
                replicates=bootstrap_replicates,
                confidence=confidence,
            ),
            "primary_station_strata": {
                "snr": stratum_metrics(
                    primary_rows,
                    field="snr_bin",
                    bootstrap_seed=bootstrap_seed + seed + 1_000,
                    bootstrap_replicates=bootstrap_replicates,
                    confidence=confidence,
                ),
                "magnitude": stratum_metrics(
                    primary_rows,
                    field="magnitude_bin",
                    bootstrap_seed=bootstrap_seed + seed + 2_000,
                    bootstrap_replicates=bootstrap_replicates,
                    confidence=confidence,
                ),
                "distance": stratum_metrics(
                    primary_rows,
                    field="distance_bin",
                    bootstrap_seed=bootstrap_seed + seed + 3_000,
                    bootstrap_replicates=bootstrap_replicates,
                    confidence=confidence,
                ),
            },
            "diagnostic_label_strata": {
                stratum: {
                    "descriptive_metrics": station_label_metrics(
                        [
                            row
                            for row in current_stations
                            if row["label_stratum"] == stratum
                        ],
                        confidence=confidence,
                    ),
                    "event_cluster_bootstrap": (
                        station_event_cluster_bootstrap(
                            [
                                row
                                for row in current_stations
                                if row["label_stratum"] == stratum
                            ],
                            seed=(
                                bootstrap_seed
                                + seed
                                + 10_000
                                + protocol["metrics"][
                                    "diagnostic_label_strata"
                                ].index(stratum)
                            ),
                            replicates=bootstrap_replicates,
                            confidence=confidence,
                        )
                    ),
                }
                for stratum in protocol["metrics"][
                    "diagnostic_label_strata"
                ]
            },
            "quiet_background_metrics": background_seed_metrics(
                current_background,
                current_network,
                confidence=confidence,
            ),
            "quiet_background_date_cluster_bootstrap": (
                background_date_cluster_bootstrap(
                    current_background,
                    current_network,
                    seed=(
                        bootstrap_seed + seed + 100_000
                    ),
                    replicates=bootstrap_replicates,
                    confidence=confidence,
                )
            ),
        }
        seed_results[str(seed)] = result

    primary_seed = str(
        protocol["seed_policy"]["primary_operational_seed"]
    )
    if primary_seed not in seed_results:
        raise RuntimeError("Primary seed result is absent")
    event_recalls = [
        float(
            result["event_and_station_metrics"][
                "availability_aware_catalog_event_any_station_hit"
            ]["estimate"]
        )
        for result in seed_results.values()
    ]
    station_recalls = [
        float(
            result["event_and_station_metrics"][
                "model_recall_on_technically_available_stations_"
                "descriptive_wilson"
            ]["estimate"]
        )
        for result in seed_results.values()
    ]
    candidate_rates = [
        result["quiet_background_metrics"][
            "quiet_background_candidate_rate_per_station_day"
        ]["rate"]
        for result in seed_results.values()
    ]
    result = {
        "result_id": "cspp_v27c_geonet_final_evaluation_result_v1",
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "protocol": artifact(args.protocol),
        "operating_point": protocol["operating_point"],
        "seed_policy": protocol["seed_policy"],
        "primary_seed_headline": seed_results[primary_seed],
        "all_seed_results": seed_results,
        "seed_robustness": {
            "availability_aware_catalog_event_any_station_hit_range": [
                min(event_recalls),
                max(event_recalls),
            ],
            "technical_station_recall_point_estimate_range": [
                min(station_recalls),
                max(station_recalls),
            ],
            "quiet_background_candidate_rate_range": (
                [min(candidate_rates), max(candidate_rates)]
                if all(value is not None for value in candidate_rates)
                else None
            ),
            "seed_selection_performed": False,
        },
        "artifacts": {
            "event_score_completion": event_completion,
            "background_score_completion": background_completion,
            "event_station_results": artifact_at(
                event_station_path,
                published_analysis_root / event_station_path.name,
            ),
            "event_level_results": artifact_at(
                event_level_path,
                published_analysis_root / event_level_path.name,
            ),
            "background_station_results": artifact_at(
                background_station_path,
                published_analysis_root / background_station_path.name,
            ),
            "background_network_results": artifact_at(
                background_network_path,
                published_analysis_root / background_network_path.name,
            ),
        },
        "evidence_counts": {
            "event_station_rows": len(station_rows),
            "event_level_rows": len(event_rows),
            "quiet_background_station_rows": len(background_station),
            "quiet_background_network_rows": len(background_network),
        },
        "claim_boundary": protocol["claim_boundary"],
    }
    result_path = analysis_root / "result.json"
    write_json_once(result_path, result)
    os.replace(analysis_root, published_analysis_root)
    for spec in result["artifacts"].values():
        verify_artifact(spec)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
