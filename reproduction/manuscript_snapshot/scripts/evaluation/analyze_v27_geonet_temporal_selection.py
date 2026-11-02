#!/usr/bin/env python3
"""Apply the frozen V27C threshold and arm-selection rules."""
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
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL = (
    ROOT
    / "experiments/"
    "cspp_v27_geonet_temporal_evaluation_protocol_v3_20260729.json"
)

EVENT_FIELDS = [
    "development_role",
    "event_id",
    "station_key",
    "label_stratum",
    "snr_bin",
    "model_key",
    "arm",
    "seed",
    "event_threshold",
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
BACKGROUND_FIELDS = [
    "development_role",
    "quiet_rank",
    "date_utc",
    "station_key",
    "common_availability",
    "model_key",
    "arm",
    "seed",
    "event_threshold",
    "candidate_count",
]
EVENT_SUMMARY_FIELDS = [
    "development_role",
    "label_stratum",
    "snr_bin",
    "arm",
    "seed",
    "event_threshold",
    "station_count",
    "matched_station_count",
    "candidate_count",
    "false_candidate_count",
    "false_negative_count",
    "station_recall",
    "candidate_precision",
    "station_f1",
    "median_absolute_pick_error_seconds",
    "median_availability_delay_seconds",
]
BACKGROUND_SUMMARY_FIELDS = [
    "development_role",
    "arm",
    "seed",
    "event_threshold",
    "station_days",
    "station_exposure_days",
    "candidate_count",
    "candidate_rate_per_station_day",
    "network_confirmations",
    "connected_network_exposure_days",
    "network_confirmation_rate_per_day",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--expected-protocol-sha256", required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def write_csv(
    path: Path,
    rows: Iterable[dict[str, Any]],
    fields: list[str],
) -> None:
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


def write_json(path: Path, payload: Any) -> None:
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


def load_protocol(
    path: Path, expected_sha256: str
) -> tuple[dict[str, Any], str]:
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise RuntimeError("Evaluation protocol SHA-256 differs")
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("protocol_id") != (
        "cspp_v27_geonet_temporal_evaluation_execution_v3"
    ):
        raise RuntimeError("Unexpected evaluation protocol")
    for relative, expected in protocol["code_sha256"].items():
        source = ROOT / relative
        if not source.is_file() or sha256_file(source) != expected:
            raise RuntimeError(f"Evaluation code drifted: {relative}")
    for spec in protocol["inputs"].values():
        verify_artifact(spec)
    for spec in protocol["models"].values():
        verify_artifact(spec)
    return protocol, actual


def load_completion(
    output_root: Path,
    scope: str,
    protocol_sha256: str,
    expected_units: int,
) -> list[dict[str, Any]]:
    path = output_root / f"{scope}_score_cache_complete.json"
    completion = json.loads(path.read_text(encoding="utf-8"))
    if (
        completion.get("completion_id")
        != f"cspp_v27_geonet_temporal_{scope}_score_cache_v1"
        or completion.get("protocol_sha256") != protocol_sha256
        or int(completion["units"]) != expected_units
    ):
        raise RuntimeError(f"Score-cache completion differs: {scope}")
    summaries = []
    for spec in completion["unit_summaries"]:
        summary_path = verify_artifact(spec)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if (
            summary.get("protocol_sha256") != protocol_sha256
            or summary.get("scope") != scope
            or summary.get("status") != "complete"
        ):
            raise RuntimeError(f"Unit summary differs: {summary_path}")
        for artifact_spec in summary["artifacts"].values():
            verify_artifact(artifact_spec)
        summaries.append(summary)
    return summaries


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
    ends = (np.flatnonzero(changes == -1) - 1).astype(
        np.int64, copy=False
    )
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
            (
                np.zeros(1, dtype=np.int64),
                np.cumsum(hits, dtype=np.int64),
            )
        )
        index = np.arange(len(hits), dtype=np.int64)
        left = np.maximum(index - int(history_packets) + 1, 0)
        support[segment] = (
            cumulative[index + 1] - cumulative[left]
        ) > 0
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
    candidates = []
    for episode, (first, last) in enumerate(
        zip(starts, ends, strict=True), start=1
    ):
        packet = int(first)
        available = start_epoch + (packet + 1) * packet_seconds
        candidates.append(
            {
                "episode_index": episode,
                "episode_start_packet": packet,
                "episode_end_packet": int(last),
                "available_epoch": available,
                "estimated_p_epoch": (
                    available
                    - float(arrival_age[packet]) * age_scale
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
    if (
        abs(float(selected["estimated_p_epoch"]) - reference_p_epoch)
        > tolerance_seconds
    ):
        return None
    return selected


def label_stratum(station: dict[str, Any]) -> str:
    source = str(station["label_source"])
    weight = station.get("pick_time_weight")
    if source == "geonet_origin_associated_used_p_pick":
        if weight is not None and float(weight) > 0.0:
            return "authoritative_used_p_positive_weight"
        raise RuntimeError("Used-P label does not have positive weight")
    if source == "geonet_origin_associated_zero_weight_p_pick":
        return "authoritative_p_zero_weight"
    if source == "taup_iasp91_fallback":
        return "taup_iasp91_fallback"
    raise RuntimeError(f"Unexpected event label source: {source}")


def network_confirmation_count(
    candidates_by_station: dict[str, list[dict[str, float | int]]],
    *,
    window_seconds: float,
    minimum_stations: int,
) -> int:
    points = sorted(
        (
            float(candidate["estimated_p_epoch"]),
            station,
        )
        for station, candidates in candidates_by_station.items()
        for candidate in candidates
    )
    count = 0
    first = 0
    while first < len(points):
        end = first
        cutoff = points[first][0] + window_seconds
        while end < len(points) and points[end][0] <= cutoff:
            end += 1
        stations = {station for _, station in points[first:end]}
        if len(stations) >= minimum_stations:
            count += 1
            first = end
        else:
            first += 1
    return count


def finite_median(values: Iterable[Any]) -> float | None:
    finite = [
        float(value)
        for value in values
        if value not in (None, "") and math.isfinite(float(value))
    ]
    return float(statistics.median(finite)) if finite else None


def event_summary_rows(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            row["development_role"],
            row["label_stratum"],
            row["snr_bin"],
            row["arm"],
            row["seed"],
            row["event_threshold"],
        )
        groups[key].append(row)
    output = []
    for key, values in sorted(groups.items()):
        (
            role,
            stratum,
            snr_bin,
            arm,
            seed,
            threshold,
        ) = key
        station_count = len(values)
        matched = sum(int(row["matched_candidate_count"]) for row in values)
        candidate_count = sum(int(row["candidate_count"]) for row in values)
        false_candidates = sum(
            int(row["false_candidate_count"]) for row in values
        )
        false_negative = sum(int(row["false_negative"]) for row in values)
        recall = matched / station_count if station_count else 0.0
        precision = matched / candidate_count if candidate_count else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        output.append(
            {
                "development_role": role,
                "label_stratum": stratum,
                "snr_bin": snr_bin,
                "arm": arm,
                "seed": seed,
                "event_threshold": threshold,
                "station_count": station_count,
                "matched_station_count": matched,
                "candidate_count": candidate_count,
                "false_candidate_count": false_candidates,
                "false_negative_count": false_negative,
                "station_recall": recall,
                "candidate_precision": precision,
                "station_f1": f1,
                "median_absolute_pick_error_seconds": finite_median(
                    row["absolute_pick_error_seconds"] for row in values
                ),
                "median_availability_delay_seconds": finite_median(
                    row["availability_delay_seconds"] for row in values
                ),
            }
        )
    return output


def background_summary_rows(
    station_rows: list[dict[str, Any]],
    network_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    network_groups: dict[
        tuple[Any, ...], list[dict[str, Any]]
    ] = defaultdict(list)
    for row in network_rows:
        network_groups[
            (
                row["development_role"],
                row["arm"],
                row["seed"],
                row["event_threshold"],
            )
        ].append(row)
    for row in station_rows:
        key = (
            row["development_role"],
            row["arm"],
            row["seed"],
            row["event_threshold"],
        )
        groups[key].append(row)
    output = []
    for key, values in sorted(groups.items()):
        role, arm, seed, threshold = key
        exposure = sum(
            float(row["common_availability"]) for row in values
        )
        candidates = sum(int(row["candidate_count"]) for row in values)
        network = network_groups[key]
        network_exposure = sum(
            float(row["network_exposure_days"]) for row in network
        )
        confirmations = sum(
            int(row["network_confirmations"]) for row in network
        )
        output.append(
            {
                "development_role": role,
                "arm": arm,
                "seed": seed,
                "event_threshold": threshold,
                "station_days": len(values),
                "station_exposure_days": exposure,
                "candidate_count": candidates,
                "candidate_rate_per_station_day": (
                    candidates / exposure if exposure else None
                ),
                "network_confirmations": confirmations,
                "connected_network_exposure_days": network_exposure,
                "network_confirmation_rate_per_day": (
                    confirmations / network_exposure
                    if network_exposure
                    else None
                ),
            }
        )
    return output


def select_thresholds(
    protocol: dict[str, Any],
    event_summary: list[dict[str, Any]],
    background_summary: list[dict[str, Any]],
) -> dict[str, Any]:
    primary = "authoritative_used_p_positive_weight"
    event_lookup = {
        (
            row["development_role"],
            row["label_stratum"],
            row["snr_bin"],
            row["arm"],
            int(row["seed"]),
            float(row["event_threshold"]),
        ): row
        for row in event_summary
    }
    background_lookup = {
        (
            row["development_role"],
            row["arm"],
            int(row["seed"]),
            float(row["event_threshold"]),
        ): row
        for row in background_summary
    }
    seeds = [int(value) for value in protocol["selection"]["seeds"]]
    thresholds = [
        float(value)
        for value in protocol["selection"]["event_threshold_grid"]
    ]
    floor = protocol["selection"]["threshold_calibration_recall_floor"]
    selected = {}
    frontiers = {}
    for arm in protocol["selection"]["arms"]:
        candidates = []
        for threshold in thresholds:
            events = [
                event_lookup[
                    (
                        "threshold_calibration",
                        primary,
                        "all",
                        arm,
                        seed,
                        threshold,
                    )
                ]
                for seed in seeds
            ]
            backgrounds = [
                background_lookup[
                    (
                        "threshold_calibration",
                        arm,
                        seed,
                        threshold,
                    )
                ]
                for seed in seeds
            ]
            recalls = [float(row["station_recall"]) for row in events]
            background_counts = [
                int(row["candidate_count"]) for row in backgrounds
            ]
            event_false = [
                int(row["false_candidate_count"]) for row in events
            ]
            delays = [
                row["median_availability_delay_seconds"] for row in events
            ]
            entry = {
                "event_threshold": threshold,
                "seed_recalls": dict(zip(map(str, seeds), recalls)),
                "seed_background_candidate_counts": dict(
                    zip(map(str, seeds), background_counts)
                ),
                "seed_event_false_candidate_counts": dict(
                    zip(map(str, seeds), event_false)
                ),
                "mean_recall": statistics.mean(recalls),
                "minimum_seed_recall": min(recalls),
                "mean_background_candidate_count": statistics.mean(
                    background_counts
                ),
                "mean_event_false_candidate_count": statistics.mean(
                    event_false
                ),
                "mean_median_availability_delay_seconds": (
                    statistics.mean(
                        float(value)
                        for value in delays
                        if value is not None
                    )
                    if any(value is not None for value in delays)
                    else None
                ),
            }
            entry["feasible"] = (
                entry["mean_recall"] >= float(floor["mean"])
                and entry["minimum_seed_recall"]
                >= float(floor["minimum_seed"])
            )
            candidates.append(entry)
        feasible = [row for row in candidates if row["feasible"]]
        if feasible:
            chosen = min(
                feasible,
                key=lambda row: (
                    row["mean_background_candidate_count"],
                    row["mean_event_false_candidate_count"],
                    -row["mean_recall"],
                    (
                        row["mean_median_availability_delay_seconds"]
                        if row[
                            "mean_median_availability_delay_seconds"
                        ]
                        is not None
                        else math.inf
                    ),
                    -row["event_threshold"],
                ),
            )
            reason = "recall_floor_feasible_lexicographic_minimum"
        else:
            chosen = min(
                candidates,
                key=lambda row: (
                    -row["minimum_seed_recall"],
                    -row["mean_recall"],
                    row["mean_background_candidate_count"],
                    row["mean_event_false_candidate_count"],
                    row["event_threshold"],
                ),
            )
            reason = "predeclared_no_feasible_threshold_fallback"
        selected[arm] = {
            "event_threshold": chosen["event_threshold"],
            "selection_reason": reason,
            "metrics": chosen,
        }
        frontiers[arm] = candidates
    return {"selected": selected, "frontiers": frontiers}


def select_arm(
    protocol: dict[str, Any],
    thresholds: dict[str, Any],
    event_summary: list[dict[str, Any]],
    background_summary: list[dict[str, Any]],
) -> dict[str, Any]:
    primary = "authoritative_used_p_positive_weight"
    event_lookup = {
        (
            row["development_role"],
            row["label_stratum"],
            row["snr_bin"],
            row["arm"],
            int(row["seed"]),
            float(row["event_threshold"]),
        ): row
        for row in event_summary
    }
    background_lookup = {
        (
            row["development_role"],
            row["arm"],
            int(row["seed"]),
            float(row["event_threshold"]),
        ): row
        for row in background_summary
    }
    seeds = [int(value) for value in protocol["selection"]["seeds"]]
    by_seed = []
    for seed in seeds:
        row = {"seed": seed}
        for arm in ("C0", "C1"):
            threshold = float(
                thresholds["selected"][arm]["event_threshold"]
            )
            event = event_lookup[
                (
                    "arm_comparison",
                    primary,
                    "all",
                    arm,
                    seed,
                    threshold,
                )
            ]
            background = background_lookup[
                ("arm_comparison", arm, seed, threshold)
            ]
            row[arm] = {
                "event_threshold": threshold,
                "background_candidate_count": int(
                    background["candidate_count"]
                ),
                "station_recall": float(event["station_recall"]),
                "median_availability_delay_seconds": event[
                    "median_availability_delay_seconds"
                ],
            }
        by_seed.append(row)
    c0_background = [
        row["C0"]["background_candidate_count"] for row in by_seed
    ]
    c1_background = [
        row["C1"]["background_candidate_count"] for row in by_seed
    ]
    c0_recall = [row["C0"]["station_recall"] for row in by_seed]
    c1_recall = [row["C1"]["station_recall"] for row in by_seed]
    c0_delays = [
        row["C0"]["median_availability_delay_seconds"] for row in by_seed
    ]
    c1_delays = [
        row["C1"]["median_availability_delay_seconds"] for row in by_seed
    ]
    delay_defined = all(
        value is not None for value in c0_delays + c1_delays
    )
    checks = {
        "background_no_greater_every_seed": all(
            c1 <= c0
            for c0, c1 in zip(c0_background, c1_background, strict=True)
        ),
        "mean_background_reduction_at_least_20_percent": (
            statistics.mean(c1_background)
            <= 0.8 * statistics.mean(c0_background)
            if statistics.mean(c0_background) > 0.0
            else statistics.mean(c1_background) == 0.0
        ),
        "mean_recall_loss_no_more_than_0p02": (
            statistics.mean(c1_recall)
            >= statistics.mean(c0_recall) - 0.02
        ),
        "mean_median_availability_increase_no_more_than_0p25_seconds": (
            delay_defined
            and statistics.mean(map(float, c1_delays))
            <= statistics.mean(map(float, c0_delays)) + 0.25
        ),
    }
    selected_arm = "C1" if all(checks.values()) else "C0"
    return {
        "selected_arm": selected_arm,
        "default_arm": "C0",
        "checks": checks,
        "paired_seed_metrics": by_seed,
        "means": {
            "C0_background_candidate_count": statistics.mean(
                c0_background
            ),
            "C1_background_candidate_count": statistics.mean(
                c1_background
            ),
            "C0_station_recall": statistics.mean(c0_recall),
            "C1_station_recall": statistics.mean(c1_recall),
            "C0_median_availability_delay_seconds": (
                statistics.mean(map(float, c0_delays))
                if all(value is not None for value in c0_delays)
                else None
            ),
            "C1_median_availability_delay_seconds": (
                statistics.mean(map(float, c1_delays))
                if all(value is not None for value in c1_delays)
                else None
            ),
        },
    }


def main() -> None:
    args = parse_args()
    protocol, protocol_sha256 = load_protocol(
        args.protocol, args.expected_protocol_sha256
    )
    output_root = Path(protocol["execution"]["output_root"])
    analysis_root = output_root / "analysis"
    if analysis_root.exists():
        raise FileExistsError(analysis_root)
    event_units = load_completion(
        output_root,
        "event",
        protocol_sha256,
        int(protocol["expected_units"]["events"]),
    )
    background_units = load_completion(
        output_root,
        "background",
        protocol_sha256,
        int(protocol["expected_units"]["background_dates"]),
    )
    analysis_root.mkdir(parents=True)

    models = protocol["models"]
    thresholds = [
        float(value)
        for value in protocol["selection"]["event_threshold_grid"]
    ]
    candidate_threshold = float(
        protocol["selection"]["candidate_threshold"]
    )
    history_packets = int(
        protocol["runtime"]["candidate_history_packets"]
    )
    packet_seconds = float(
        protocol["runtime"]["packet_duration_seconds"]
    )
    max_age = float(protocol["runtime"]["max_arrival_age_packets"])
    tolerance = float(protocol["metrics"]["match_tolerance_seconds"])

    event_rows = []
    for unit in event_units:
        for station in unit["station_results"]:
            spec = station["score_cache"]
            cache_path = verify_artifact(spec)
            with np.load(cache_path, allow_pickle=False) as cache:
                candidate_probability = cache[
                    "candidate_probability"
                ].astype(np.float32, copy=False)
                arrival_age = cache["arrival_age_fraction"].astype(
                    np.float32, copy=False
                )
                valid = cache["quality_valid"].astype(bool, copy=False)
                allowed = cache["candidate_allowed"].astype(
                    bool, copy=False
                )
                support = support_with_resets(
                    candidate_probability,
                    allowed,
                    valid,
                    candidate_threshold,
                    history_packets,
                )
                stratum = label_stratum(station)
                local_snr = station["local_p_snr"]
                snr_bin = (
                    str(local_snr["snr_bin"])
                    if local_snr is not None
                    else "invalid"
                )
                for model_key, model in models.items():
                    event_probability = cache[
                        f"event_probability__{model_key}"
                    ].astype(np.float32, copy=False)
                    for threshold in thresholds:
                        candidates = extract_candidates(
                            support=support,
                            valid=valid,
                            event_probability=event_probability,
                            arrival_age=arrival_age,
                            event_threshold=threshold,
                            start_epoch=float(
                                station["metadata"]["start_epoch"]
                            ),
                            packet_seconds=packet_seconds,
                            max_arrival_age_packets=max_age,
                        )
                        reference = float(station["reference_p_epoch"])
                        matched = matched_candidate(
                            candidates, reference, tolerance
                        )
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
                        event_rows.append(
                            {
                                "development_role": unit[
                                    "development_role"
                                ],
                                "event_id": unit["unit_key"],
                                "station_key": station["station_key"],
                                "label_stratum": stratum,
                                "snr_bin": snr_bin,
                                "model_key": model_key,
                                "arm": model["arm"],
                                "seed": int(model["seed"]),
                                "event_threshold": threshold,
                                "candidate_count": len(candidates),
                                "matched_candidate_count": (
                                    1 if matched is not None else 0
                                ),
                                "false_candidate_count": (
                                    len(candidates)
                                    - (1 if matched is not None else 0)
                                ),
                                "false_negative": (
                                    0 if matched is not None else 1
                                ),
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
                                "availability_delay_seconds": availability,
                            }
                        )

    background_rows = []
    network_rows = []
    network_contract = protocol["metrics"]["network_confirmation"]
    for unit in background_units:
        station_caches = []
        for station in unit["station_results"]:
            path = verify_artifact(station["score_cache"])
            station_caches.append((station, path))
        for model_key, model in models.items():
            candidates_by_threshold: dict[
                float, dict[str, list[dict[str, float | int]]]
            ] = {threshold: {} for threshold in thresholds}
            for station, path in station_caches:
                with np.load(path, allow_pickle=False) as cache:
                    candidate_probability = cache[
                        "candidate_probability"
                    ].astype(np.float32, copy=False)
                    arrival_age = cache["arrival_age_fraction"].astype(
                        np.float32, copy=False
                    )
                    valid = cache["quality_valid"].astype(bool, copy=False)
                    allowed = cache["candidate_allowed"].astype(
                        bool, copy=False
                    )
                    event_probability = cache[
                        f"event_probability__{model_key}"
                    ].astype(np.float32, copy=False)
                    support = support_with_resets(
                        candidate_probability,
                        allowed,
                        valid,
                        candidate_threshold,
                        history_packets,
                    )
                    for threshold in thresholds:
                        candidates = extract_candidates(
                            support=support,
                            valid=valid,
                            event_probability=event_probability,
                            arrival_age=arrival_age,
                            event_threshold=threshold,
                            start_epoch=float(
                                station["metadata"]["start_epoch"]
                            ),
                            packet_seconds=packet_seconds,
                            max_arrival_age_packets=max_age,
                        )
                        candidates_by_threshold[threshold][
                            station["station_key"]
                        ] = candidates
                        background_rows.append(
                            {
                                "development_role": unit[
                                    "development_role"
                                ],
                                "quiet_rank": int(unit["quiet_rank"]),
                                "date_utc": unit["unit_key"],
                                "station_key": station["station_key"],
                                "common_availability": float(
                                    station["common_availability"]
                                ),
                                "model_key": model_key,
                                "arm": model["arm"],
                                "seed": int(model["seed"]),
                                "event_threshold": threshold,
                                "candidate_count": len(candidates),
                            }
                        )
            network_exposure = min(
                float(row["common_availability"])
                for row, _ in station_caches
            )
            for threshold in thresholds:
                network_rows.append(
                    {
                        "development_role": unit["development_role"],
                        "date_utc": unit["unit_key"],
                        "arm": model["arm"],
                        "seed": int(model["seed"]),
                        "event_threshold": threshold,
                        "network_confirmations": (
                            network_confirmation_count(
                                candidates_by_threshold[threshold],
                                window_seconds=float(
                                    network_contract[
                                        "estimated_p_span_seconds"
                                    ]
                                ),
                                minimum_stations=int(
                                    network_contract[
                                        "minimum_distinct_stations"
                                    ]
                                ),
                            )
                        ),
                        "network_exposure_days": network_exposure,
                    }
                )

    event_summary = event_summary_rows(event_rows)
    # Add pooled primary SNR-independent rows used by the selection contract.
    primary = "authoritative_used_p_positive_weight"
    pooled_rows = [
        {**row, "snr_bin": "all"}
        for row in event_rows
        if row["label_stratum"] == primary
    ]
    event_summary.extend(event_summary_rows(pooled_rows))
    event_summary.sort(
        key=lambda row: (
            row["development_role"],
            row["label_stratum"],
            row["snr_bin"],
            row["arm"],
            int(row["seed"]),
            float(row["event_threshold"]),
        )
    )
    background_summary = background_summary_rows(
        background_rows, network_rows
    )
    threshold_selection = select_thresholds(
        protocol, event_summary, background_summary
    )
    arm_selection = select_arm(
        protocol,
        threshold_selection,
        event_summary,
        background_summary,
    )

    event_path = analysis_root / "event_station_thresholds.csv"
    background_path = (
        analysis_root / "background_station_thresholds.csv"
    )
    event_summary_path = analysis_root / "event_threshold_summary.csv"
    background_summary_path = (
        analysis_root / "background_threshold_summary.csv"
    )
    write_csv(event_path, event_rows, EVENT_FIELDS)
    write_csv(background_path, background_rows, BACKGROUND_FIELDS)
    write_csv(
        event_summary_path, event_summary, EVENT_SUMMARY_FIELDS
    )
    write_csv(
        background_summary_path,
        background_summary,
        BACKGROUND_SUMMARY_FIELDS,
    )
    result = {
        "result_id": "cspp_v27_geonet_temporal_selection_result_v1",
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "protocol": artifact(args.protocol),
        "evidence_status": (
            "temporal_development_model_and_threshold_selection_only"
        ),
        "threshold_selection": threshold_selection,
        "arm_selection": arm_selection,
        "frozen_operating_point": {
            "selected_arm": arm_selection["selected_arm"],
            "event_threshold": threshold_selection["selected"][
                arm_selection["selected_arm"]
            ]["event_threshold"],
            "candidate_threshold": candidate_threshold,
            "primary_operational_seed": int(
                protocol["selection"]["primary_seed"]
            ),
            "all_reported_seeds": protocol["selection"]["seeds"],
        },
        "counts": {
            "event_units": len(event_units),
            "background_date_units": len(background_units),
            "event_station_threshold_rows": len(event_rows),
            "background_station_threshold_rows": len(background_rows),
        },
        "artifacts": {
            "event_station_thresholds": artifact(event_path),
            "background_station_thresholds": artifact(background_path),
            "event_threshold_summary": artifact(event_summary_path),
            "background_threshold_summary": artifact(
                background_summary_path
            ),
        },
        "next_boundary": (
            "The selected arm and event threshold are now immutable. Only "
            "after this result is frozen may the 2026 GeoNet final holdout "
            "be acquired or any final-holdout model/public-baseline score "
            "be computed."
        ),
    }
    write_json(analysis_root / "selection_result.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
