#!/usr/bin/env python3
"""Analyze the frozen NCSN continuous scores without target tuning."""
from __future__ import annotations

from collections import Counter, defaultdict
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
from scipy.stats import chi2


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/evaluation"))

import analyze_v27_ncedc_event_scores as event_analysis  # noqa: E402
from cspp_v27_reused_core_score_cache import sha256_file  # noqa: E402


DEFAULT_PROTOCOL = (
    ROOT
    / "experiments/cspp_v27_ncedc_continuous_analysis_protocol_20260817.json"
)

CANDIDATE_FIELDS = [
    "candidate_id",
    "seed",
    "date_utc",
    "station_key",
    "station_latitude",
    "station_longitude",
    "episode_index",
    "episode_start_packet",
    "episode_end_packet",
    "episode_duration_seconds",
    "available_epoch",
    "available_utc",
    "estimated_p_epoch",
    "estimated_p_utc",
    "candidate_probability",
    "event_probability",
    "arrival_age_fraction",
    "candidate_eligible_station_days",
    "acquisition_status",
    "score_cache_sha256",
    "network_group_ids",
    "cause_label",
    "review_status",
]

REVIEW_FIELDS = [
    "candidate_id",
    "date_utc",
    "station_key",
    "station_latitude",
    "station_longitude",
    "episode_start_packet",
    "episode_end_packet",
    "available_utc",
    "estimated_p_utc",
    "network_group_ids",
    "cause_label",
    "review_status",
]


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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def format_utc(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, tz=UTC).isoformat().replace(
        "+00:00", "Z"
    )


def poisson_interval(count: int, exposure: float) -> list[float | None]:
    if exposure <= 0.0:
        return [None, None]
    lower = 0.0 if count == 0 else 0.5 * chi2.ppf(0.025, 2 * count)
    upper = 0.5 * chi2.ppf(0.975, 2 * (count + 1))
    return [float(lower / exposure), float(upper / exposure)]


def load_protocol() -> tuple[dict[str, Any], str]:
    expected = os.environ.get("CSPP_NCEDC_CONTINUOUS_ANALYSIS_SHA256", "")
    actual = sha256_file(DEFAULT_PROTOCOL)
    if not expected or actual != expected:
        raise RuntimeError(f"Set the frozen analysis SHA-256; actual={actual}")
    protocol = json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))
    if protocol.get("protocol_id") != "cspp_v27_ncedc_continuous_analysis_v1":
        raise RuntimeError("Unexpected NCSN continuous analysis protocol")
    for relative, expected_hash in protocol["code_sha256"].items():
        source = ROOT / relative
        if not source.is_file() or sha256_file(source) != expected_hash:
            raise RuntimeError(f"Continuous analysis code drifted: {relative}")
    for spec in protocol["inputs"].values():
        verify_artifact(spec)
    return protocol, actual


def load_completion(
    protocol: dict[str, Any], protocol_sha256: str
) -> list[dict[str, Any]]:
    completion_path = Path(protocol["scoring"]["completion_path"])
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    if (
        completion.get("protocol_sha256")
        != protocol["scoring"]["hybrid_amendment_sha256"]
        or int(completion["data_bearing_station_days"])
        != int(protocol["expected_units"]["data_bearing_station_days"])
    ):
        raise RuntimeError("Continuous score completion differs")
    summaries: list[dict[str, Any]] = []
    for spec in completion["unit_summaries"]:
        summary_path = verify_artifact(spec)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if (
            summary.get("protocol_sha256")
            != protocol["scoring"]["hybrid_amendment_sha256"]
            or summary.get("status") != "complete"
        ):
            raise RuntimeError(f"Station-day score summary differs: {summary_path}")
        verify_artifact(summary["score_cache"])
        summaries.append(summary)
    if len(summaries) != int(protocol["expected_units"]["data_bearing_station_days"]):
        raise RuntimeError("Continuous station-day summary count differs")
    return summaries


def active_episode_count(active: np.ndarray) -> int:
    starts, _ = event_analysis.active_runs(active)
    return int(len(starts))


def extract_seed_candidates(
    *,
    summaries: list[dict[str, Any]],
    seed_key: str,
    seed: int,
    protocol: dict[str, Any],
    coordinates: dict[str, tuple[float, float]],
    keep_details: bool,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    point = protocol["operating_point"]
    runtime = protocol["runtime"]
    rows: list[dict[str, Any]] = []
    ablation = Counter()
    for summary in summaries:
        score_path = verify_artifact(summary["score_cache"])
        with np.load(score_path, allow_pickle=False) as cache:
            valid = cache["quality_valid"].astype(bool, copy=False)
            allowed = cache["candidate_allowed"].astype(bool, copy=False)
            candidate_probability = cache["candidate_probability"].astype(
                np.float32, copy=False
            )
            event_probability = cache[
                f"event_probability__{seed_key}"
            ].astype(np.float32, copy=False)
            arrival_age = cache["arrival_age_fraction"].astype(
                np.float32, copy=False
            )
            support = event_analysis.support_with_resets(
                candidate_probability,
                allowed,
                valid,
                float(point["candidate_threshold"]),
                int(runtime["candidate_history_packets"]),
            )
            candidates = event_analysis.extract_candidates(
                support=support,
                valid=valid,
                event_probability=event_probability,
                arrival_age=arrival_age,
                event_threshold=float(point["event_threshold"]),
                start_epoch=float(summary["metadata"]["start_epoch"]),
                packet_seconds=float(runtime["packet_duration_seconds"]),
                max_arrival_age_packets=float(
                    runtime["max_arrival_age_packets"]
                ),
            )
            ablation["full_decision_chain"] += len(candidates)
            ablation["candidate_support_without_event_verifier"] += (
                active_episode_count(support & valid)
            )
            current_candidate = (
                valid
                & allowed
                & (candidate_probability >= float(point["candidate_threshold"]))
            )
            ablation["single_packet_candidate_plus_event"] += active_episode_count(
                current_candidate
                & (event_probability >= float(point["event_threshold"]))
            )
            ablation["single_packet_candidate_only"] += active_episode_count(
                current_candidate
            )
            if not keep_details:
                continue
            latitude, longitude = coordinates[summary["station_key"]]
            for candidate in candidates:
                packet = int(candidate["episode_start_packet"])
                candidate_id = (
                    f"{summary['date_utc']}:{summary['station_key']}:"
                    f"{candidate['episode_index']}"
                )
                available = float(candidate["available_epoch"])
                estimated = float(candidate["estimated_p_epoch"])
                rows.append(
                    {
                        "candidate_id": candidate_id,
                        "seed": seed,
                        "date_utc": summary["date_utc"],
                        "station_key": summary["station_key"],
                        "station_latitude": latitude,
                        "station_longitude": longitude,
                        "episode_index": int(candidate["episode_index"]),
                        "episode_start_packet": packet,
                        "episode_end_packet": int(candidate["episode_end_packet"]),
                        "episode_duration_seconds": (
                            int(candidate["episode_end_packet"]) - packet + 1
                        )
                        * float(runtime["packet_duration_seconds"]),
                        "available_epoch": available,
                        "available_utc": format_utc(available),
                        "estimated_p_epoch": estimated,
                        "estimated_p_utc": format_utc(estimated),
                        "candidate_probability": float(
                            candidate_probability[packet]
                        ),
                        "event_probability": float(event_probability[packet]),
                        "arrival_age_fraction": float(arrival_age[packet]),
                        "candidate_eligible_station_days": float(
                            summary["duration"][
                                "candidate_eligible_station_days"
                            ]
                        ),
                        "acquisition_status": summary["acquisition_status"],
                        "score_cache_sha256": summary["score_cache"]["sha256"],
                        "network_group_ids": "",
                        "cause_label": "unresolved",
                        "review_status": "pending",
                    }
                )
    rows.sort(
        key=lambda row: (
            row["estimated_p_epoch"],
            row["station_key"],
            row["candidate_id"],
        )
    )
    return rows, dict(ablation)


def network_groups(
    rows: list[dict[str, Any]], window_seconds: float, minimum_stations: int
) -> list[dict[str, Any]]:
    by_date: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_date[row["date_utc"]].append(row)
    groups: list[dict[str, Any]] = []
    for date_utc, date_rows in sorted(by_date.items()):
        points = sorted(date_rows, key=lambda row: row["estimated_p_epoch"])
        first = 0
        index = 1
        while first < len(points):
            end = first
            cutoff = float(points[first]["estimated_p_epoch"]) + window_seconds
            while (
                end < len(points)
                and float(points[end]["estimated_p_epoch"]) <= cutoff
            ):
                end += 1
            window = points[first:end]
            by_station: dict[str, dict[str, Any]] = {}
            for row in window:
                by_station.setdefault(row["station_key"], row)
            if len(by_station) >= minimum_stations:
                members = sorted(
                    by_station.values(), key=lambda row: row["estimated_p_epoch"]
                )
                group_id = f"{date_utc}:G{index:04d}"
                for member in members:
                    existing = str(member["network_group_ids"])
                    member["network_group_ids"] = ";".join(
                        value for value in (existing, group_id) if value
                    )
                groups.append(
                    {
                        "group_id": group_id,
                        "date_utc": date_utc,
                        "station_count": len(members),
                        "estimated_p_start_epoch": float(
                            members[0]["estimated_p_epoch"]
                        ),
                        "estimated_p_end_epoch": float(
                            members[-1]["estimated_p_epoch"]
                        ),
                        "estimated_p_span_seconds": float(
                            members[-1]["estimated_p_epoch"]
                            - members[0]["estimated_p_epoch"]
                        ),
                        "available_epoch": max(
                            float(member["available_epoch"])
                            for member in members
                        ),
                        "member_candidate_ids": ";".join(
                            member["candidate_id"] for member in members
                        ),
                        "interpretation": (
                            "time-coincident candidate group; cause unresolved"
                        ),
                    }
                )
                index += 1
                first = end
            else:
                first += 1
    return groups


def network_exposure_days(
    summaries: list[dict[str, Any]], minimum_stations: int, packet_seconds: float
) -> tuple[float, list[dict[str, Any]]]:
    by_date: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for summary in summaries:
        by_date[summary["date_utc"]].append(summary)
    rows: list[dict[str, Any]] = []
    total = 0.0
    for date_utc, date_summaries in sorted(by_date.items()):
        eligible_count = None
        for summary in date_summaries:
            with np.load(
                verify_artifact(summary["score_cache"]), allow_pickle=False
            ) as cache:
                allowed = cache["candidate_allowed"].astype(np.int16, copy=False)
                if eligible_count is None:
                    eligible_count = np.zeros(len(allowed), dtype=np.int16)
                if len(allowed) != len(eligible_count):
                    raise RuntimeError(f"Packet clock differs on {date_utc}")
                eligible_count += allowed
        if eligible_count is None:
            exposure = 0.0
        else:
            exposure = float(
                np.count_nonzero(eligible_count >= minimum_stations)
                * packet_seconds
                / 86400.0
            )
        total += exposure
        rows.append(
            {
                "date_utc": date_utc,
                "data_bearing_stations": len(date_summaries),
                "network_eligible_days": exposure,
            }
        )
    return total, rows


def review_queue(
    candidates: list[dict[str, Any]],
    groups: list[dict[str, Any]],
    protocol: dict[str, Any],
) -> list[dict[str, Any]]:
    rule = protocol["manual_review"]
    if len(candidates) <= int(rule["review_all_if_candidate_count_at_most"]):
        return list(candidates)
    selected_ids = {
        candidate_id
        for group in groups
        for candidate_id in str(group["member_candidate_ids"]).split(";")
    }
    ranked = sorted(
        candidates,
        key=lambda row: hashlib.sha256(
            f"{rule['sampling_salt']}:{row['candidate_id']}".encode("utf-8")
        ).hexdigest(),
    )
    selected_ids.update(
        row["candidate_id"]
        for row in ranked[: int(rule["deterministic_sample_size"])]
    )
    selected_ids.update(
        row["candidate_id"]
        for row in candidates
        if float(row["candidate_probability"])
        >= float(rule["extreme_candidate_probability"])
        or float(row["event_probability"])
        >= float(rule["extreme_event_probability"])
    )
    return [row for row in candidates if row["candidate_id"] in selected_ids]


def main() -> None:
    protocol, protocol_sha256 = load_protocol()
    output_root = Path(protocol["execution"]["output_root"])
    summaries = load_completion(protocol, protocol_sha256)
    output_root.mkdir(parents=True, exist_ok=False)
    station_rows = read_csv(
        verify_artifact(protocol["inputs"]["selected_stations"])
    )
    coordinates = {
        f"{row['network']}.{row['station']}.{row['location']}": (
            float(row["latitude"]),
            float(row["longitude"]),
        )
        for row in station_rows
    }
    exposure = sum(
        float(summary["duration"]["candidate_eligible_station_days"])
        for summary in summaries
    )
    packet_seconds = float(protocol["runtime"]["packet_duration_seconds"])
    network_exposure, network_exposure_rows = network_exposure_days(
        summaries,
        int(protocol["network_confirmation"]["minimum_distinct_stations"]),
        packet_seconds,
    )
    seed_results: dict[str, Any] = {}
    primary_candidates: list[dict[str, Any]] = []
    primary_groups: list[dict[str, Any]] = []
    primary_ablation: dict[str, int] = {}
    for seed_key, seed in protocol["operating_point"]["seed_keys"].items():
        primary = int(seed) == int(
            protocol["operating_point"]["primary_operational_seed"]
        )
        candidates, ablation = extract_seed_candidates(
            summaries=summaries,
            seed_key=seed_key,
            seed=int(seed),
            protocol=protocol,
            coordinates=coordinates,
            keep_details=True,
        )
        if primary:
            primary_candidates = candidates
            primary_ablation = ablation
            primary_groups = network_groups(
                candidates,
                float(
                    protocol["network_confirmation"][
                        "estimated_p_window_seconds"
                    ]
                ),
                int(
                    protocol["network_confirmation"]["minimum_distinct_stations"]
                ),
            )
            group_count = len(primary_groups)
        else:
            group_count = len(
                network_groups(
                    candidates,
                    float(
                        protocol["network_confirmation"][
                            "estimated_p_window_seconds"
                        ]
                    ),
                    int(
                        protocol["network_confirmation"][
                            "minimum_distinct_stations"
                        ]
                    ),
                )
            )
        count = int(ablation["full_decision_chain"])
        seed_results[seed_key] = {
            "seed": int(seed),
            "candidate_count": count,
            "candidate_eligible_station_days": exposure,
            "candidate_rate_per_station_day": count / exposure,
            "candidate_rate_exact_poisson95": poisson_interval(count, exposure),
            "network_group_count": group_count,
            "network_eligible_days": network_exposure,
            "network_group_rate_per_network_day": (
                group_count / network_exposure if network_exposure > 0.0 else None
            ),
            "network_group_rate_exact_poisson95": poisson_interval(
                group_count, network_exposure
            ),
            "decision_chain_counts": ablation,
        }

    queue = review_queue(primary_candidates, primary_groups, protocol)
    write_csv(
        output_root / "primary_candidates.csv",
        primary_candidates,
        CANDIDATE_FIELDS,
    )
    write_csv(
        output_root / "manual_review_queue_blind.csv",
        [
            {field: row[field] for field in REVIEW_FIELDS}
            for row in queue
        ],
        REVIEW_FIELDS,
    )
    group_fields = [
        "group_id",
        "date_utc",
        "station_count",
        "estimated_p_start_epoch",
        "estimated_p_end_epoch",
        "estimated_p_span_seconds",
        "available_epoch",
        "member_candidate_ids",
        "interpretation",
    ]
    write_csv(output_root / "network_groups.csv", primary_groups, group_fields)
    exposure_fields = [
        "date_utc",
        "data_bearing_stations",
        "network_eligible_days",
    ]
    write_csv(
        output_root / "network_exposure_by_date.csv",
        network_exposure_rows,
        exposure_fields,
    )
    primary_key = protocol["operating_point"]["primary_model_key"]
    result = {
        "result_id": "cspp_v27_ncedc_continuous_analysis_v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "protocol_sha256": protocol_sha256,
        "summary": {
            "requested_station_days": int(
                protocol["expected_units"]["requested_station_days"]
            ),
            "data_bearing_station_days": len(summaries),
            "archive_unavailable_station_days": int(
                protocol["expected_units"]["archive_unavailable_station_days"]
            ),
            "candidate_eligible_station_days": exposure,
            "network_eligible_days": network_exposure,
            "primary_model_key": primary_key,
            "primary": seed_results[primary_key],
            "replication_seeds": seed_results,
            "manual_review_queue_count": len(queue),
            "unresolved_candidate_count": len(primary_candidates),
            "decision_chain_counts": primary_ablation,
        },
        "claim_boundary": {
            "candidate_rate": (
                "raw released-candidate rate before cause adjudication"
            ),
            "unresolved": "not classified as a false alarm",
            "network_group": (
                "three-station estimated-P time coincidence; not event "
                "association, location, or cause verification"
            ),
            "decision_chain_ablation": (
                "post-training fixed-rule analysis on identical score caches"
            ),
        },
        "artifacts": {},
    }
    write_json(output_root / "result.json", result)
    result["artifacts"] = {
        name: artifact(output_root / name)
        for name in (
            "primary_candidates.csv",
            "manual_review_queue_blind.csv",
            "network_groups.csv",
            "network_exposure_by_date.csv",
        )
    }
    write_json(output_root / "result.json", result)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
