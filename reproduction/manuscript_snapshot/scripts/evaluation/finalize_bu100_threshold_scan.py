#!/usr/bin/env python3
"""Merge verified MATLAB scan tables without replacing frozen test outputs."""
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts/evaluation")]
from cache_bu100_threshold_scan import OUT, read, write, write_csv, artifact
from verify_bu100_threshold_scan import number, equal


def rows(path):
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def key(row):
    return (row["group"], row["stratum"], float(row["tolerance_seconds"]),
            float(row["candidate_threshold"]), float(row["event_threshold"]))


def rank(row):
    return (2 * int(row["TP"]) / (int(row["records"]) + int(row["candidate_count"])),
            int(row["TP"]) / int(row["records"]), float(row["event_threshold"]), float(row["candidate_threshold"]))


def main():
    combined = {}
    for directory in [OUT, OUT / "refined"]:
        assert read(directory / "verification.json")["status"] == "PASS"
        for row in rows(directory / "threshold_summary.csv"):
            k = key(row)
            if k in combined:
                previous = combined[k]
                for name in row:
                    if name not in ("group", "stratum"):
                        assert equal(number(previous[name]), number(row[name])), (k, name)
            else:
                combined[k] = row
    all_rows = [combined[k] for k in sorted(combined)]
    pairs = sorted({k[3:] for k in combined})
    assert len(pairs) == 347 and len(all_rows) == 31230
    primary = [r for r in all_rows if r["group"] == "all" and r["stratum"] == "all_records" and float(r["tolerance_seconds"]) == 1.5]
    frozen = next(r for r in primary if float(r["candidate_threshold"]) == .1 and float(r["event_threshold"]) == .975)
    chosen = {"frozen": frozen,
              "best_event_only": max((r for r in primary if float(r["candidate_threshold"]) == .1), key=rank),
              "best_joint": max(primary, key=rank)}
    comparison = []
    for label, r in chosen.items():
        for group in ["all", "strong_motion", "intensity"]:
            for stratum in ["all_records", "reference_ge_10s"]:
                for tolerance in [1.5, .5, .25]:
                    k = (group, stratum, tolerance, float(r["candidate_threshold"]), float(r["event_threshold"]))
                    comparison.append({"selection": label, **combined[k]})
    write_csv(OUT / "combined_summary.csv", all_rows)
    write_csv(OUT / "comparison.csv", comparison)
    write_csv(OUT / "event_threshold_curve_q010.csv", sorted(
        [r for r in primary if float(r["candidate_threshold"]) == .1], key=lambda r: float(r["event_threshold"])))
    result = {"status": "PASS", "records": 2136, "distinct_threshold_pairs": len(pairs),
              "record_threshold_combinations": 2136 * len(pairs), "coarse_pairs": 153, "refined_pairs": 224,
              "overlap_pairs": 30, "combined_summary_rows": len(all_rows), "selections": chosen,
              "comparison": comparison,
              "interpretation": "Micro F1 relative to supplied P labels; same-data post-outcome selection, not independent validation. No deployment threshold changed.",
              "metric_definition": "TP=matched records, FP_proxy=candidate_count-TP, FN=records-TP; F1=2TP/(records+candidate_count)",
              "verification": [artifact(OUT / "verification.json"), artifact(OUT / "refined/verification.json")],
              "tables": [artifact(OUT / f) for f in ["combined_summary.csv", "comparison.csv", "event_threshold_curve_q010.csv"]],
              "finalization_script": artifact(Path(__file__))}
    write(OUT / "result.json", result)
    for label, row in chosen.items():
        print(label, row, flush=True)
    print(f"Merged {len(pairs)} unique pairs; {len(all_rows)} verified summary rows", flush=True)


if __name__ == "__main__":
    main()
