"""Recalculate saved-row metrics; this does not rerun waveform inference."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def rows(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def truth(value):
    return str(value).lower() in {"true", "1"}


def point_metrics(records):
    matched = [
        r
        for r in records
        if truth(r.get("matched", r.get("matched_candidate_count", "0")))
    ]
    n, tp = len(records), len(matched)
    candidates = sum(int(r["candidate_count"]) for r in records)
    fp, fn = candidates - tp, n - tp
    assert min(tp, fp, fn) >= 0
    errors = np.asarray([float(r["pick_error_seconds"]) for r in matched])
    delays = np.asarray([float(r["availability_delay_seconds"]) for r in matched])
    result = dict(
        records=n,
        tp=tp,
        fp=fp,
        fn=fn,
        candidates=candidates,
        precision=tp / candidates if candidates else 0.0,
        recall=tp / n if n else 0.0,
        f1=2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0,
    )
    for key, values in (
        ("pick_error", errors),
        ("absolute_error", np.abs(errors)),
        ("availability_delay", delays),
    ):
        result[key + "_median_seconds"] = (
            float(np.median(values)) if len(values) else None
        )
        result[key + "_p95_seconds"] = (
            float(np.quantile(values, 0.95)) if len(values) else None
        )
    for seconds in (0.5, 1, 2, 5, 10, 20, 30, 60):
        result[f"recall_available_within_{seconds:g}s"] = (
            float(np.sum(delays <= seconds) / n) if n else 0.0
        )
    return result


def same(actual, expected, label):
    if isinstance(expected, (int, float)):
        assert actual is not None and math.isclose(
            actual, expected, rel_tol=1e-9, abs_tol=1e-9
        ), (label, actual, expected)
    else:
        assert actual == expected, (label, actual, expected)


EVENT_KEYS = {
    "records": "records",
    "tp": "matched_records",
    "fp": "false_candidate_episodes",
    "fn": "false_negative_records",
    "precision": "candidate_precision",
    "recall": "recall",
    "f1": "f1",
    "absolute_error_median_seconds": "pick_absolute_error_median_seconds",
    "absolute_error_p95_seconds": "pick_absolute_error_p95_seconds",
    "availability_delay_median_seconds": "availability_delay_median_seconds",
    "availability_delay_p95_seconds": "availability_delay_p95_seconds",
}


def check_event_metrics(actual, expected, label):
    for ours, theirs in EVENT_KEYS.items():
        same(actual[ours], expected[theirs], label + "/" + ours)


def reproduce(root=ROOT):
    output = {
        "scope": "Reaggregation of saved decisions, not waveform or training reproduction",
        "event_windows": {},
        "stead": [],
        "continuous": {},
    }
    for name in ("tx", "pr", "nc", "nc_large", "geo"):
        data = rows(root / f"evidence/event_windows/{name}.csv")
        if name == "geo":
            data = [
                r
                for r in data
                if r["label_stratum"] == "authoritative_used_p_positive_weight"
                and truth(r["technical_available"])
            ]
        groups = {
            key: point_metrics([r for r in data if r["model_key"] == key])
            for key in sorted({r["model_key"] for r in data})
        }
        source = json.loads((root / f"evidence/summaries/{name}.json").read_text())
        primary = groups["C0_seed20261101"]
        if name == "geo":
            for model_key, actual in groups.items():
                reference = source["all_seed_results"][
                    model_key.removeprefix("C0_seed")
                ]["event_and_station_metrics"]
                recall = reference[
                    "availability_aware_station_recall_descriptive_wilson"
                ]
                same(actual["records"], recall["trials"], name + "/records")
                same(actual["tp"], recall["successes"], name + "/tp")
                same(actual["fp"], reference["false_candidate_count"], name + "/fp")
        else:
            references = source.get("replication_metrics", source.get("all_seeds", {}))
            for model_key, actual in groups.items():
                reference = references[model_key]
                check_event_metrics(actual, reference, name + "/" + model_key)
        output["event_windows"][name] = groups

    stead = root / "evidence/stead/threshold_comparison"
    for expected in rows(stead / "comparison.csv"):
        path = stead / f"records_{expected['id']}.csv"
        data = rows(path)
        if expected["subset"] != "all":
            data = [r for r in data if r["split"] == expected["subset"]]
        actual = point_metrics(data)
        for key, value in actual.items():
            if key in expected and expected[key] != "":
                same(value, float(expected[key]), "stead/" + expected["id"] + "/" + key)
        output["stead"].append(
            {
                "selection": expected["selection"],
                "model": expected["model"],
                "id": expected["id"],
                "subset": expected["subset"],
                **actual,
            }
        )

    output["stead_diagnostics"] = {}
    for name in ("full_test", "m3_p10"):
        folder = root / "evidence/stead" / name
        data = rows(folder / "records.csv")
        source = json.loads((folder / "result.json").read_text())
        same(len(data), source["processed_records"], name + "/total")
        events = [r for r in data if r["category"] == "earthquake_local"]
        noise = [r for r in data if r["category"] == "noise"]
        assert len(events) + len(noise) == len(data)
        actual = point_metrics(events)
        check_event_metrics(actual, source["event_window_metrics"], name)
        for key, value in {
            "records": len(noise),
            "records_with_candidates": sum(
                int(r["candidate_count"]) > 0 for r in noise
            ),
            "false_candidate_episodes": sum(int(r["candidate_count"]) for r in noise),
        }.items():
            same(value, source["noise"][key], name + "/noise/" + key)
        output["stead_diagnostics"][name] = actual

    for path in sorted((root / "evidence/continuous").glob("*/*/result.json")):
        folder = path.parent
        saved = json.loads(path.read_text())
        candidate_rows = rows(folder / "candidates.csv")
        same(len(candidate_rows), saved["candidate_count"], str(folder) + "/candidates")
        network_days = sum(
            float(r["network_eligible_days"])
            for r in rows(folder / "network_exposure_by_date.csv")
        )
        same(network_days, saved["network_days"], str(folder) + "/network_days")
        rate = len(candidate_rows) / saved["station_days"]
        same(
            rate, saved["candidate_rate_per_station_day"], str(folder) + "/station_rate"
        )
        result = dict(
            candidate_count=len(candidate_rows),
            station_days=saved["station_days"],
            station_days_provenance="Saved exposure denominator; not rebuilt from waveforms",
            network_days=network_days,
            candidate_rate_per_station_day=rate,
            associations={},
        )
        for name, reference in saved["associations"].items():
            groups = rows(folder / (name + "_groups.csv"))
            events = rows(folder / (name + "_events.csv"))
            supported = sum(r["catalog_class"] == "catalog_supported" for r in groups)
            assoc = dict(
                group_count=len(groups),
                catalog_supported=supported,
                unmatched_groups=len(groups) - supported,
                unmatched_groups_per_network_day=(len(groups) - supported)
                / network_days,
                event_metrics={},
            )
            for key in (
                "group_count",
                "catalog_supported",
                "unmatched_groups",
                "unmatched_groups_per_network_day",
            ):
                same(assoc[key], reference[key], str(folder) + "/" + name + "/" + key)
            for key, event_reference in reference["event_metrics"].items():
                selected = [
                    r
                    for r in events
                    if truth(r["eligible"])
                    and float(r["magnitude"]) >= event_reference["minimum_magnitude"]
                ]
                detected = sum(truth(r["detected"]) for r in selected)
                metrics = dict(
                    eligible_events=len(selected),
                    detected_events=detected,
                    missed_events=len(selected) - detected,
                    recall=detected / len(selected) if selected else None,
                )
                for metric, value in metrics.items():
                    same(
                        value,
                        event_reference[metric],
                        str(folder) + "/" + name + "/" + key + "/" + metric,
                    )
                assoc["event_metrics"][key] = metrics
            result["associations"][name] = assoc
        output["continuous"][str(folder.relative_to(root / "evidence/continuous"))] = (
            result
        )
    output["all_checked_metrics_match"] = True
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = reproduce()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "all_checked_metrics_match": True,
                "event_cohorts": len(result["event_windows"]),
                "stead_comparisons": len(result["stead"]),
                "continuous_conditions": len(result["continuous"]),
            }
        )
    )


if __name__ == "__main__":
    main()
