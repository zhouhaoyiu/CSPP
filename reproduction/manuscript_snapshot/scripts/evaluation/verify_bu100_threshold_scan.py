#!/usr/bin/env python3
"""Verify threshold scan aggregates and selected points using scalar replay."""
import csv
from datetime import datetime, timezone
import math
from pathlib import Path
import sys

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts/evaluation")]
from cache_bu100_threshold_scan import OUT, BASE, read, write, sha, artifact, write_csv, verify_seal, scalar_candidates


def rows(path):
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def number(value):
    return float(value) if value != "" else math.nan


def equal(a, b):
    return (math.isnan(a) and math.isnan(b)) or math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-10)


def main():
    p = verify_seal()
    cache_check = read(OUT / "cache_validation.json")
    assert sha(cache_check["packet_scores"]["path"]) == cache_check["packet_scores"]["sha256"]
    metadata = rows(OUT / "record_order.csv")
    pairs = [(r["candidate_threshold"], r["event_threshold"]) for r in p["threshold_pairs"]]
    tolerances = p["tolerances_seconds"]
    with h5py.File(OUT / "scan_record_metrics.h5", "r") as h:
        counts = h["candidate_count"][:]
        hit = h["matched"][:].astype(bool)
        errors = h["error_seconds"][:]
        latencies = h["latency_seconds"][:]
    assert counts.shape == (len(pairs), len(metadata))
    assert hit.shape == errors.shape == latencies.shape == (3, len(pairs), len(metadata))
    assert np.array_equal(np.isfinite(errors), hit)
    assert np.array_equal(np.isfinite(latencies), hit)
    assert np.all(hit.sum(axis=2) <= len(metadata))
    instruments = np.array([r["instrument"] for r in metadata])
    cohorts = np.array([r["cohort"] for r in metadata])
    references = np.array([float(r["reference_p_seconds"]) for r in metadata])
    summary = rows(OUT / "threshold_summary.csv")
    for row in summary:
        k = pairs.index((float(row["candidate_threshold"]), float(row["event_threshold"])))
        z = tolerances.index(float(row["tolerance_seconds"]))
        selected = np.ones(len(metadata), bool) if row["group"] == "all" else (instruments == row["group"]) | (cohorts == row["group"])
        if row["stratum"] == "reference_ge_10s":
            selected &= references >= 10
        else:
            assert row["stratum"] == "all_records"
        n = int(selected.sum()); tp = int(hit[z, k, selected].sum()); nc = int(counts[k, selected].sum())
        e = errors[z, k, selected]; e = e[np.isfinite(e)]
        d = latencies[z, k, selected]; d = d[np.isfinite(d)]
        expected = {"records": n, "TP": tp, "FP_proxy": nc - tp, "FN": n - tp, "candidate_count": nc,
                    "precision": tp / nc if nc else math.nan, "recall": tp / n if n else math.nan,
                    "F1": 2 * tp / (n + nc) if n + nc else math.nan,
                    "mean_absolute_error_seconds": float(np.mean(abs(e))) if len(e) else math.nan,
                    "median_absolute_error_seconds": float(np.median(abs(e))) if len(e) else math.nan,
                    "median_availability_latency_seconds": float(np.median(d)) if len(d) else math.nan}
        for field, value in expected.items():
            assert equal(number(row[field]), float(value)), (row, field, value)
    selections = rows(OUT / "selected_operating_points.csv")
    primary = [r for r in summary if r["group"] == "all" and r["stratum"] == "all_records" and float(r["tolerance_seconds"]) == 1.5]
    def rank(row):
        return (2 * int(row["TP"]) / (int(row["records"]) + int(row["candidate_count"])),
                int(row["TP"]) / int(row["records"]), float(row["event_threshold"]), float(row["candidate_threshold"]))
    global_best = max(primary, key=rank)
    event_best = max((r for r in primary if float(r["candidate_threshold"]) == .1), key=rank)
    expected_best = {"best_joint": global_best, "best_event_only": event_best}
    for s in selections:
        if s["selection"] in expected_best:
            for key in ("candidate_threshold", "event_threshold", "TP", "FP_proxy", "FN"):
                assert s[key] == expected_best[s["selection"]][key]
    selected_pairs = sorted({(float(r["candidate_threshold"]), float(r["event_threshold"])) for r in selections})
    record_rows, candidate_rows = [], []
    with h5py.File(OUT / "packet_scores.h5", "r") as h:
        for i, r in enumerate(metadata):
            scores = h[r["id"] + "/scores"][:]
            assert np.isfinite(scores[:3, scores[3] == 1]).all()
            assert not np.any((scores[4] == 1) & (scores[3] == 0))
            for q, e in selected_pairs:
                k = pairs.index((q, e))
                candidates = scalar_candidates(scores, q, e)
                assert len(candidates) == counts[k, i], (r["id"], q, e, "candidate_count")
                names = [s["selection"] for s in selections if float(s["candidate_threshold"]) == q and float(s["event_threshold"]) == e]
                for z, tolerance in enumerate(tolerances):
                    ref = float(r["reference_p_seconds"])
                    matched = next((c for c in candidates if abs(c["estimated_p_epoch"] - ref) <= tolerance), None)
                    assert bool(matched) == hit[z, k, i]
                    if matched:
                        assert equal(matched["estimated_p_epoch"] - ref, errors[z, k, i])
                        assert equal(matched["available_epoch"] - ref, latencies[z, k, i])
                    for name in names:
                        record_rows.append({"selection": name, "candidate_threshold": q, "event_threshold": e,
                                            "tolerance_seconds": tolerance, **r, "candidate_count": len(candidates),
                                            "matched": int(matched is not None),
                                            "error_seconds": matched["estimated_p_epoch"] - ref if matched else "",
                                            "latency_seconds": matched["available_epoch"] - ref if matched else ""})
                for name in names:
                    for c in candidates:
                        candidate_rows.append({"selection": name, "candidate_threshold": q, "event_threshold": e,
                                               "id": r["id"], "reference_p_seconds": float(r["reference_p_seconds"]), **c})
            if (i + 1) % 400 == 0 or i == len(metadata) - 1:
                print(f"Scalar verification {i + 1}/{len(metadata)}, {len(selected_pairs)} unique points", flush=True)
    write_csv(OUT / "selected_record_metrics.csv", record_rows)
    write_csv(OUT / "selected_candidates.csv", candidate_rows)
    baseline = next(r for r in selections if r["selection"] == "frozen")
    assert (int(baseline["TP"]), int(baseline["FP_proxy"]), int(baseline["FN"])) == (1212, 80, 924)
    result = {"verified_utc": datetime.now(timezone.utc).isoformat(), "status": "PASS",
              "records": len(metadata), "threshold_pairs": len(pairs), "record_threshold_combinations": len(metadata) * len(pairs),
              "summary_rows_recomputed": len(summary), "all_selected_records_scalar_checked": True,
              "selected_unique_points": len(selected_pairs), "frozen_baseline_reproduced": True,
              "all_frozen_source_and_code_hashes_unchanged": True, "selections": selections,
              "role": "Same-data exploratory scan; FP and F1 are relative to provided labels, not verified operational false-alarm metrics",
              "artifacts": [artifact(OUT / f) for f in ["threshold_summary.csv", "scan_record_metrics.h5", "selected_operating_points.csv",
                                                        "selected_record_metrics.csv", "selected_candidates.csv"]],
              "verification_script": artifact(Path(__file__))}
    write(OUT / "verification.json", result)
    print("Threshold scan verification PASS", flush=True)


if __name__ == "__main__":
    main()
