#!/usr/bin/env python3
"""Joint event-recall and unmatched-trigger analysis on continuous NCSN data."""
from __future__ import annotations

from collections import Counter
import csv
from datetime import UTC, datetime
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
from obspy import read_events
from obspy.geodetics.base import gps2dist_azimuth
from obspy.taup import TauPyModel


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/evaluation"))
import adjudicate_v27_ncedc_continuous_catalog as catalog_match  # noqa: E402
import analyze_v27_ncedc_continuous_scores as continuous  # noqa: E402
import analyze_v27_ncedc_event_scores as event_metrics  # noqa: E402
import build_v27_ncedc_continuous_score_cache as score_io  # noqa: E402


DEFAULT_ROOT = Path(
    "workspace/"
    "ncedc_joint_continuous_m3_20260904"
)
TARGET_CATALOG = Path(
    "workspace/ncedc_model_blind_20260817/"
    "metadata/raw/ncedc_events_2024_2025_m3_screen.xml"
)


def event_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for path in paths:
        if path.name.startswith("._"):
            continue
        for event in read_events(str(path)):
            origin = event.preferred_origin() or event.origins[0]
            magnitude = event.preferred_magnitude()
            if magnitude is None and event.magnitudes:
                magnitude = event.magnitudes[0]
            row = {
                "event_id": str(event.resource_id).rsplit("/", 1)[-1],
                "origin_epoch": float(origin.time.timestamp),
                "origin_utc": str(origin.time),
                "latitude": float(origin.latitude),
                "longitude": float(origin.longitude),
                "depth_km": max(float(origin.depth or 0.0) / 1000.0, 0.0),
                "magnitude": None if magnitude is None else float(magnitude.mag),
                "event_type": str(event.event_type or "unknown"),
            }
            rows[row["event_id"]] = row
    return sorted(rows.values(), key=lambda row: row["origin_epoch"])


def station_coordinates(requests: list[dict[str, str]]) -> dict[str, tuple[float, float]]:
    coordinates: dict[str, tuple[float, float]] = {}
    for row in requests:
        key = score_io.station_key(row)
        value = (float(row["station_latitude"]), float(row["station_longitude"]))
        if key in coordinates and coordinates[key] != value:
            raise RuntimeError(f"Station coordinate drift: {key}")
        coordinates[key] = value
    return coordinates


def target_events(
    selected_dates: set[str], coordinates: dict[str, tuple[float, float]]
) -> list[dict[str, Any]]:
    rows = event_rows([TARGET_CATALOG])
    selected = []
    for row in rows:
        date = datetime.fromtimestamp(row["origin_epoch"], tz=UTC).date().isoformat()
        distances = sorted(
            gps2dist_azimuth(
                row["latitude"], row["longitude"], latitude, longitude
            )[0]
            / 1000.0
            for latitude, longitude in coordinates.values()
        )
        magnitude = row["magnitude"]
        if (
            date in selected_dates
            and magnitude is not None
            and magnitude >= 3.0
            and distances[0] <= 200.0
        ):
            selected.append(
                {
                    **row,
                    "date_utc": date,
                    "nearest_station_distance_km": distances[0],
                    "third_station_distance_km": distances[2],
                }
            )
    return selected


def eligible_events(
    events: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    coordinates: dict[str, tuple[float, float]],
    taup: TauPyModel,
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], float]]:
    by_date: dict[str, list[dict[str, Any]]] = {}
    for summary in summaries:
        by_date.setdefault(summary["date_utc"], []).append(summary)
    predicted: dict[tuple[str, str], float] = {}
    output = []
    for event in events:
        stations = []
        for summary in by_date.get(event["date_utc"], []):
            station = summary["station_key"]
            latitude, longitude = coordinates[station]
            travel = catalog_match.first_p_seconds(taup, event, latitude, longitude)
            if travel is None:
                continue
            arrival = float(event["origin_epoch"]) + travel
            packet_seconds = 0.5
            packet = int(
                math.floor((arrival - float(summary["metadata"]["start_epoch"])) / packet_seconds)
            )
            with np.load(Path(summary["score_cache"]["path"]), allow_pickle=False) as cache:
                allowed = cache["candidate_allowed"]
                if 0 <= packet < len(allowed) and bool(allowed[packet]):
                    stations.append(station)
                    predicted[(event["event_id"], station)] = arrival
        output.append(
            {
                **event,
                "eligible_station_count": len(stations),
                "eligible_stations": ";".join(sorted(stations)),
                "eligible": len(stations) >= 3,
            }
        )
    return output, predicted


def classify_groups(
    groups: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    regional: list[dict[str, Any]],
    global_events: list[dict[str, Any]],
    taup: TauPyModel,
) -> list[dict[str, Any]]:
    by_id = {row["candidate_id"]: row for row in candidates}
    regional_epochs = [row["origin_epoch"] for row in regional]
    global_epochs = [row["origin_epoch"] for row in global_events]
    cache: dict[tuple[str, float, float], float | None] = {}
    candidate_matches: dict[str, tuple[str, str]] = {}
    for row in candidates:
        match = catalog_match.match_candidate(
            row,
            events=regional,
            event_epochs=regional_epochs,
            taup=taup,
            lookback=1800.0,
            tolerance=5.0,
            cache=cache,
        )
        source = "regional"
        if match is None:
            match = catalog_match.match_candidate(
                row,
                events=global_events,
                event_epochs=global_epochs,
                taup=taup,
                lookback=1800.0,
                tolerance=5.0,
                cache=cache,
            )
            source = "global_m45"
        if match is not None:
            candidate_matches[row["candidate_id"]] = (match["event"]["event_id"], source)
    output = []
    for group in groups:
        ids = str(group["member_candidate_ids"]).split(";")
        counts = Counter(candidate_matches.get(candidate_id) for candidate_id in ids)
        counts.pop(None, None)
        (event_id, source), count = counts.most_common(1)[0] if counts else (("", ""), 0)
        supported = count >= 3
        output.append(
            {
                **group,
                "catalog_event_id": event_id if supported else "",
                "catalog_source": source if supported else "",
                "shared_catalog_station_count": count if supported else 0,
                "catalog_class": "catalog_supported" if supported else "unmatched",
            }
        )
    return output


def metric(events: list[dict[str, Any]], minimum_magnitude: float) -> dict[str, Any]:
    rows = [
        row
        for row in events
        if row["eligible"] and float(row["magnitude"]) >= minimum_magnitude
    ]
    detected = sum(bool(row["detected"]) for row in rows)
    return {
        "minimum_magnitude": minimum_magnitude,
        "eligible_events": len(rows),
        "detected_events": detected,
        "missed_events": len(rows) - detected,
        "recall": detected / len(rows) if rows else None,
        "recall_wilson95": event_metrics.wilson(detected, len(rows)) if rows else [None, None],
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    root = DEFAULT_ROOT
    output = root / "analysis"
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    scoring = json.loads((root / "scoring_protocol.json").read_text())
    completion = json.loads((root / "scores/score_cache_complete.json").read_text())
    summaries = [json.loads(Path(spec["path"]).read_text()) for spec in completion["unit_summaries"]]
    requests = score_io.read_csv(Path(scoring["inputs"]["station_day_requests"]["path"]))
    coordinates = station_coordinates(requests)
    selected_dates = {
        row["date_utc"]
        for row in score_io.read_csv(root / "metadata/selected_event_dates.csv")
    }
    analysis_protocol = {
        "operating_point": scoring["operating_point"],
        "runtime": scoring["runtime"],
    }
    candidates, ablation = continuous.extract_seed_candidates(
        summaries=summaries,
        seed_key="C0_seed20261101",
        seed=20261101,
        protocol=analysis_protocol,
        coordinates=coordinates,
        keep_details=True,
    )
    groups = continuous.network_groups(candidates, 10.0, 3)
    regional = event_rows(
        sorted((root / "catalogs/regional").glob("*.xml"))
    )
    global_events = event_rows(
        sorted((root / "catalogs/global_m45").glob("*.xml"))
    )
    taup = TauPyModel(model="iasp91")
    classified_groups = classify_groups(groups, candidates, regional, global_events, taup)
    targets, predicted = eligible_events(
        target_events(selected_dates, coordinates), summaries, coordinates, taup
    )
    detected_groups: dict[str, dict[str, Any]] = {}
    for row in classified_groups:
        if row["catalog_class"] != "catalog_supported":
            continue
        event_id = row["catalog_event_id"]
        previous = detected_groups.get(event_id)
        if previous is None or float(row["available_epoch"]) < float(
            previous["available_epoch"]
        ):
            detected_groups[event_id] = row
    for event in targets:
        group = detected_groups.get(event["event_id"])
        event["detected"] = group is not None
        if group is None:
            event["network_latency_seconds"] = ""
            continue
        arrivals = [
            predicted[(event["event_id"], station)]
            for station in event["eligible_stations"].split(";")
            if (event["event_id"], station) in predicted
        ]
        event["network_latency_seconds"] = float(group["available_epoch"]) - min(arrivals)
    exposure, _ = continuous.network_exposure_days(summaries, 3, 0.5)
    candidate_exposure = sum(
        float(summary["duration"]["candidate_eligible_station_days"])
        for summary in summaries
    )
    unmatched = sum(row["catalog_class"] == "unmatched" for row in classified_groups)
    latencies = [
        float(row["network_latency_seconds"])
        for row in targets
        if row["detected"] and row["network_latency_seconds"] != ""
    ]
    result = {
        "result_id": "cspp_v27_ncedc_joint_continuous_analysis_v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "fixed_operating_point": scoring["operating_point"],
        "target_event_counts": {
            "catalog_events_m3_within_200_km": len(targets),
            "eligible_at_three_or_more_stations": sum(row["eligible"] for row in targets),
        },
        "event_metrics": {
            "m3_or_greater": metric(targets, 3.0),
            "m4_or_greater": metric(targets, 4.0),
        },
        "continuous_metrics": {
            "candidate_count": len(candidates),
            "candidate_station_days": candidate_exposure,
            "candidate_rate_per_station_day": len(candidates) / candidate_exposure,
            "network_group_count": len(classified_groups),
            "catalog_supported_network_groups": len(classified_groups) - unmatched,
            "unmatched_network_groups": unmatched,
            "network_exposure_days": exposure,
            "unmatched_group_rate_per_network_day": unmatched / exposure,
        },
        "latency_seconds": {
            "count": len(latencies),
            "median": float(np.median(latencies)) if latencies else None,
            "p95": float(np.quantile(latencies, 0.95)) if latencies else None,
        },
        "decision_chain_counts": ablation,
        "claim_boundary": (
            "Target recall and unmatched network-group rate are measured on the same "
            "event-bearing continuous cohort at the frozen operating point. Catalog "
            "timing support is not a manual cause label."
        ),
    }
    write_csv(output / "target_events.csv", targets)
    write_csv(output / "network_groups.csv", classified_groups)
    write_csv(output / "candidates.csv", candidates)
    score_io.write_json_atomic(output / "result.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
