#!/usr/bin/env python3
"""Compare sealed public pickers on the frozen joint NCSN cohort."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
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

import analyze_v27_ncedc_continuous_scores as continuous  # noqa: E402
import analyze_v27_ncedc_joint_continuous as joint  # noqa: E402
import build_v27_ncedc_continuous_score_cache as score_io  # noqa: E402
from cspp_v27_reused_core_score_cache import sha256_file  # noqa: E402


DEFAULT_ROOT = Path(
    "workspace/"
    "ncedc_joint_continuous_m3_20260904"
)
PACKET_SECONDS = 0.5
PACKETS_PER_DAY = int(86400 / PACKET_SECONDS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--freeze-only", action="store_true")
    return parser.parse_args()


def artifact(path: Path) -> dict[str, Any]:
    path = path.resolve()
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def verify(spec: dict[str, Any]) -> Path:
    path = Path(spec["path"])
    if (
        not path.is_file()
        or path.stat().st_size != int(spec["bytes"])
        or sha256_file(path) != str(spec["sha256"])
    ):
        raise RuntimeError(f"Artifact drifted: {path}")
    return path


def freeze_protocol(root: Path) -> Path:
    destination = root / "public_baseline_analysis_protocol.json"
    if destination.exists():
        return destination
    scoring = root / "public_baseline_protocol.json"
    primary = root / "analysis/result.json"
    scoring_payload = json.loads(scoring.read_text())
    payload = {
        "protocol_id": "cspp_v27_ncedc_joint_public_baseline_analysis_v1",
        "frozen_at_utc": datetime.now(UTC).isoformat(),
        "status": (
            "frozen after PhaseNet completion and during EQTransformer scoring, "
            "before EQTransformer aggregate output"
        ),
        "inputs": {
            "public_baseline_protocol": artifact(scoring),
            "opened_cspp_result": artifact(primary),
        },
        "methods": {
            model: [point["id"] for point in spec["operating_points"]]
            for model, spec in scoring_payload["models"].items()
        },
        "fixed_metrics": {
            "target_events": "same CSPP quality-eligible M>=3 events within 200 km",
            "magnitude_thresholds": [3.0, 4.0],
            "association_minimum_stations": 3,
            "association_window_seconds": 10.0,
            "catalog_candidate_tolerance_seconds": 5.0,
            "candidate_rate_denominator": "method-native candidate exposure",
            "network_rate_denominator": "three-or-more-station method-native exposure",
        },
        "claim_boundary": {
            "released_weights_and_thresholds": True,
            "matched_retraining": False,
            "baseline_threshold_tuning": False,
            "cspp_primary_result_opened_before_comparison": True,
            "architecture_superiority_claim_permitted": False,
        },
        "code_sha256": {
            "scripts/evaluation/analyze_v27_ncedc_joint_public_baselines.py": sha256_file(
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
    score_io.write_json_atomic(destination, payload)
    return destination


def load_protocol(path: Path) -> dict[str, Any]:
    protocol = json.loads(path.read_text())
    for relative, digest in protocol["code_sha256"].items():
        if sha256_file(ROOT / relative) != digest:
            raise RuntimeError(f"Analysis code drifted: {relative}")
    for spec in protocol["inputs"].values():
        verify(spec)
    return protocol


def load_completion(
    root: Path, model: str, scoring_sha256: str, expected_units: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    path = root / "public_baselines" / model / "completion.json"
    completion = json.loads(path.read_text())
    if completion["protocol_sha256"] != scoring_sha256:
        raise RuntimeError(f"Scoring protocol differs: {model}")
    if int(completion["data_bearing_station_days"]) != expected_units:
        raise RuntimeError(f"Incomplete public baseline: {model}")
    summaries = [json.loads(verify(spec).read_text()) for spec in completion["unit_summaries"]]
    if len(summaries) != expected_units:
        raise RuntimeError(f"Unit-summary count differs: {model}")
    return completion, summaries


def load_candidates(
    summaries: list[dict[str, Any]],
    method: str,
    coordinates: dict[str, tuple[float, float]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary in summaries:
        path = verify(summary["candidates"])
        with path.open(encoding="utf-8", newline="") as handle:
            for source in csv.DictReader(handle):
                if source["method_id"] != method:
                    continue
                latitude, longitude = coordinates[source["station_key"]]
                rows.append(
                    {
                        **source,
                        "station_latitude": latitude,
                        "station_longitude": longitude,
                        "estimated_p_epoch": float(source["estimated_p_epoch"]),
                        "available_epoch": float(source["available_epoch"]),
                        "network_group_ids": "",
                    }
                )
    rows.sort(key=lambda row: (row["estimated_p_epoch"], row["station_key"]))
    return rows


def exposure_metrics(summaries: list[dict[str, Any]]) -> tuple[float, float]:
    station_packets = 0
    by_date: dict[str, list[np.ndarray]] = defaultdict(list)
    for summary in summaries:
        with np.load(verify(summary["candidate_exposure"]), allow_pickle=False) as archive:
            exposure = archive["candidate_exposure"].astype(bool, copy=False)
            if exposure.shape != (PACKETS_PER_DAY,):
                raise RuntimeError(f"Candidate clock differs: {summary['station_day_id']}")
            station_packets += int(exposure.sum())
            by_date[summary["date_utc"]].append(exposure.copy())
    network_packets = 0
    for arrays in by_date.values():
        count = np.zeros(PACKETS_PER_DAY, dtype=np.uint8)
        for exposure in arrays:
            count += exposure
        network_packets += int(np.count_nonzero(count >= 3))
    return (
        station_packets * PACKET_SECONDS / 86400.0,
        network_packets * PACKET_SECONDS / 86400.0,
    )


def evaluate_method(
    *,
    model: str,
    method: str,
    summaries: list[dict[str, Any]],
    coordinates: dict[str, tuple[float, float]],
    base_targets: list[dict[str, Any]],
    predicted: dict[tuple[str, str], float],
    regional: list[dict[str, Any]],
    global_events: list[dict[str, Any]],
    taup: TauPyModel,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    candidates = load_candidates(summaries, method, coordinates)
    groups = continuous.network_groups(candidates, 10.0, 3)
    member_ids = {
        candidate_id
        for group in groups
        for candidate_id in str(group["member_candidate_ids"]).split(";")
    }
    members = [row for row in candidates if row["candidate_id"] in member_ids]
    classified = joint.classify_groups(groups, members, regional, global_events, taup)
    detected_groups: dict[str, dict[str, Any]] = {}
    for group in classified:
        if group["catalog_class"] != "catalog_supported":
            continue
        event_id = group["catalog_event_id"]
        previous = detected_groups.get(event_id)
        if previous is None or float(group["available_epoch"]) < float(previous["available_epoch"]):
            detected_groups[event_id] = group
    targets: list[dict[str, Any]] = []
    latencies = []
    for source in base_targets:
        event = dict(source)
        group = detected_groups.get(event["event_id"])
        event["model_key"] = model
        event["method_id"] = method
        event["detected"] = group is not None
        event["network_latency_seconds"] = ""
        if group is not None:
            arrivals = [
                predicted[(event["event_id"], station)]
                for station in event["eligible_stations"].split(";")
                if (event["event_id"], station) in predicted
            ]
            if arrivals:
                latency = float(group["available_epoch"]) - min(arrivals)
                event["network_latency_seconds"] = latency
                latencies.append(latency)
        targets.append(event)
    station_days, network_days = exposure_metrics(summaries)
    unmatched = sum(group["catalog_class"] == "unmatched" for group in classified)
    result = {
        "model_key": model,
        "method_id": method,
        "event_metrics": {
            "m3_or_greater": joint.metric(targets, 3.0),
            "m4_or_greater": joint.metric(targets, 4.0),
        },
        "continuous_metrics": {
            "candidate_count": len(candidates),
            "candidate_station_days": station_days,
            "candidate_rate_per_station_day": len(candidates) / station_days,
            "network_group_count": len(classified),
            "catalog_supported_network_groups": len(classified) - unmatched,
            "unmatched_network_groups": unmatched,
            "network_exposure_days": network_days,
            "unmatched_group_rate_per_network_day": unmatched / network_days,
        },
        "latency_seconds": {
            "count": len(latencies),
            "median": float(np.median(latencies)) if latencies else None,
            "p95": float(np.quantile(latencies, 0.95)) if latencies else None,
        },
    }
    return result, targets, classified


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    protocol_path = freeze_protocol(args.root)
    protocol = load_protocol(protocol_path)
    if args.freeze_only:
        print(json.dumps(artifact(protocol_path), indent=2))
        return
    output = args.root / "analysis_public_baselines"
    if output.exists():
        raise FileExistsError(output)
    scoring_path = verify(protocol["inputs"]["public_baseline_protocol"])
    scoring_sha256 = sha256_file(scoring_path)
    scoring = json.loads(scoring_path.read_text())
    expected_units = int(scoring["expected_units"]["data_bearing_station_days"])
    cspp_scoring = json.loads((args.root / "scoring_protocol.json").read_text())
    cspp_completion = json.loads((args.root / "scores/score_cache_complete.json").read_text())
    cspp_summaries = [json.loads(verify(spec).read_text()) for spec in cspp_completion["unit_summaries"]]
    requests = score_io.read_csv(Path(cspp_scoring["inputs"]["station_day_requests"]["path"]))
    coordinates = joint.station_coordinates(requests)
    selected_dates = {
        row["date_utc"]
        for row in score_io.read_csv(args.root / "metadata/selected_event_dates.csv")
    }
    taup = TauPyModel(model="iasp91")
    base_targets, predicted = joint.eligible_events(
        joint.target_events(selected_dates, coordinates),
        cspp_summaries,
        coordinates,
        taup,
    )
    regional = joint.event_rows(sorted((args.root / "catalogs/regional").glob("*.xml")))
    global_events = joint.event_rows(sorted((args.root / "catalogs/global_m45").glob("*.xml")))
    output.mkdir()
    results = {}
    completions = {}
    all_targets = []
    for model, methods in protocol["methods"].items():
        completion, summaries = load_completion(
            args.root, model, scoring_sha256, expected_units
        )
        completions[model] = artifact(
            args.root / "public_baselines" / model / "completion.json"
        )
        for method in methods:
            result, targets, groups = evaluate_method(
                model=model,
                method=method,
                summaries=summaries,
                coordinates=coordinates,
                base_targets=base_targets,
                predicted=predicted,
                regional=regional,
                global_events=global_events,
                taup=taup,
            )
            key = f"{model}:{method}"
            results[key] = result
            all_targets.extend(targets)
            write_rows(output / f"network_groups__{method}.csv", groups)
            print(
                f"{key} m3={result['event_metrics']['m3_or_greater']['recall']:.3f} "
                f"m4={result['event_metrics']['m4_or_greater']['recall']:.3f} "
                f"candidate_rate={result['continuous_metrics']['candidate_rate_per_station_day']:.3f} "
                f"unmatched={result['continuous_metrics']['unmatched_network_groups']}",
                flush=True,
            )
    write_rows(output / "target_events.csv", all_targets)
    score_io.write_json_atomic(
        output / "result.json",
        {
            "result_id": "cspp_v27_ncedc_joint_public_baselines_v1",
            "created_at_utc": datetime.now(UTC).isoformat(),
            "analysis_protocol": artifact(protocol_path),
            "scoring_completions": completions,
            "results": results,
            "claim_boundary": protocol["claim_boundary"],
        },
    )


if __name__ == "__main__":
    main()
