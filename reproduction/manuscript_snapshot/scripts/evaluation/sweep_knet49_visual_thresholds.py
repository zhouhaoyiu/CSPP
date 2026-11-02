#!/usr/bin/env python3
"""Exploratory threshold sweep on 49 frozen AI-reviewed K-NET P intervals."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from cspp_v27_reused_core_score_cache import V27ScoreCache
from scripts.evaluation.analyze_knet66_visual_review import agreement, TOLERANCE
from scripts.evaluation.eval_cspp_v27c_fixed_retrospective import candidates_from_cache

BASE = Path("workspace/eqanomalynet_knet_full_cases_20260905")
REVIEW = BASE / "visual_review_66"
DEFAULT_OUT = REVIEW / "threshold_sweep_49_20260905"
Q_THRESHOLDS = [0.01, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5]
E_THRESHOLDS = [0.0, 0.25, 0.5, 0.7, 0.75, 0.8, 0.85, 0.9, 0.92, 0.94, 0.95, 0.96, 0.97, 0.975, 0.98, 0.99, 0.995]


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def classify_candidate(p, lo, hi):
    if max(abs(p - lo), abs(p - hi)) <= TOLERANCE:
        return "agreement"
    if lo - TOLERANCE <= p <= hi + TOLERANCE:
        return "borderline"
    return "early" if p < lo - TOLERANCE else "late"


def record_metrics(note, candidates):
    lo, hi = float(note["onset_lower_s"]), float(note["onset_upper_s"])
    mid = (lo + hi) / 2
    labels = [classify_candidate(c["estimated_p_epoch"], lo, hi) for c in candidates]
    matched = [i for i, label in enumerate(labels) if label == "agreement"]
    status = agreement(candidates, lo, hi)
    assert (status == "agreement") == bool(matched)
    # The first emitted matching candidate represents online availability.
    selected = min(matched, key=lambda i: candidates[i]["available_epoch"]) if matched else None
    chosen = candidates[selected] if selected is not None else None
    inside = sum(label in ("agreement", "borderline") for label in labels)
    return {
        "station": note["station"], "trace_name": note["trace_name"],
        "startup_stratum": note["startup_stratum"],
        "onset_lower_s": lo, "onset_upper_s": hi,
        "comparison": status, "candidate_count": len(candidates),
        "first_candidate_agreement": int(bool(labels) and labels[0] == "agreement"),
        "outside_interval_candidates": labels.count("early") + labels.count("late"),
        "early_candidates": labels.count("early"), "late_candidates": labels.count("late"),
        "borderline_candidates": labels.count("borderline"),
        "duplicate_near_onset_candidates": max(inside - 1, 0),
        "p_minus_interval_midpoint_s": chosen["estimated_p_epoch"] - mid if chosen else None,
        "release_minus_interval_midpoint_s": chosen["available_epoch"] - mid if chosen else None,
        "selected_p_s": chosen["estimated_p_epoch"] if chosen else None,
        "selected_release_s": chosen["available_epoch"] if chosen else None,
    }, labels


def summarize(rows):
    hits = [r for r in rows if r["comparison"] == "agreement"]
    errors = [r["p_minus_interval_midpoint_s"] for r in hits]
    releases = [r["release_minus_interval_midpoint_s"] for r in hits]
    return {
        "records": len(rows), "agreement_records": len(hits),
        "agreement_fraction": len(hits) / len(rows),
        "borderline_records": sum(r["comparison"] == "borderline" for r in rows),
        "no_agreement_records": sum(r["comparison"] == "no_agreement" for r in rows),
        "after_10_s_hits": sum(r["startup_stratum"] == "after_10_s" for r in hits),
        "before_10_s_hits": sum(r["startup_stratum"] == "before_10_s" for r in hits),
        "first_candidate_agreement_records": sum(r["first_candidate_agreement"] for r in rows),
        "total_candidates": sum(r["candidate_count"] for r in rows),
        "outside_interval_candidates": sum(r["outside_interval_candidates"] for r in rows),
        "early_candidates": sum(r["early_candidates"] for r in rows),
        "late_candidates": sum(r["late_candidates"] for r in rows),
        "borderline_candidates": sum(r["borderline_candidates"] for r in rows),
        "duplicate_near_onset_candidates": sum(r["duplicate_near_onset_candidates"] for r in rows),
        "median_p_minus_midpoint_s": float(np.median(errors)) if errors else None,
        "median_abs_p_minus_midpoint_s": float(np.median(np.abs(errors))) if errors else None,
        "median_release_minus_midpoint_s": float(np.median(releases)) if releases else None,
    }


def self_checks():
    assert classify_candidate(12.0, 12.0, 13.0) == "agreement"
    assert classify_candidate(10.5, 12.0, 13.0) == "borderline"
    assert classify_candidate(10.49, 12.0, 13.0) == "early"
    assert classify_candidate(14.51, 12.0, 13.0) == "late"
    note = dict(station="test", trace_name="test", startup_stratum="after_10_s",
                onset_lower_s=12.0, onset_upper_s=13.0)
    candidates = [dict(estimated_p_epoch=2.0, available_epoch=3.0),
                  dict(estimated_p_epoch=12.5, available_epoch=13.0),
                  dict(estimated_p_epoch=12.7, available_epoch=14.0)]
    row, _ = record_metrics(note, candidates)
    assert row["comparison"] == "agreement" and row["first_candidate_agreement"] == 0
    assert row["outside_interval_candidates"] == row["duplicate_near_onset_candidates"] == 1
    assert row["selected_release_s"] == 13.0
    assert record_metrics(note, [candidates[0]])[0]["comparison"] == "no_agreement"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    self_checks()
    protocol = json.loads((BASE / "protocol.json").read_text())
    runtime, original_point = protocol["runtime"], protocol["operating_point"]
    notes = [r for r in read_csv(REVIEW / "record_review.csv") if r["visual_status"] == "clear"]
    assert len(notes) == len({r["station"] for r in notes}) == 49
    assert sum(r["startup_stratum"] == "after_10_s" for r in notes) == 41
    assert sum(r["startup_stratum"] == "before_10_s" for r in notes) == 8
    cache_paths = [BASE / "scores" / (r["trace_name"] + ".npz") for r in notes]
    source_paths = [BASE / name for name in ("protocol.json", "result.json", "candidates.csv", "records.csv")]
    source_paths += [REVIEW / name for name in ("visual_first_pass.csv", "record_review.csv", "summary.json")]
    source_paths += cache_paths + [Path(protocol["model"]["path"])]
    hashes = {str(p): digest(p) for p in source_paths}
    assert hashes[protocol["model"]["path"]] == protocol["model"]["sha256"]
    sweep_protocol = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "role": "User-requested post-outcome threshold sensitivity experiment; no replacement of frozen primary operating point",
        "records": 49, "after_10_s_records": 41, "before_10_s_records": 8,
        "reference": "Frozen prior Codex visual intervals, not independent expert picks",
        "q_thresholds": Q_THRESHOLDS, "e_thresholds": E_THRESHOLDS,
        "zero_confirmation_threshold": "Bypasses the confirmation threshold while retaining candidate support and quality/startup eligibility",
        "unchanged": ["checkpoint", "score arrays", "normalization", "packet start", "quality gate", "20-packet startup gate", "4-packet support", "first packet of each active episode"],
        "agreement": "At least one emitted P estimate is within 1.5 s of both interval endpoints",
        "borderline": "No robust match, but at least one estimate is within 1.5 s of some point in the interval",
        "outside_interval_candidate": "Estimated P outside [visual lower - 1.5 s, visual upper + 1.5 s]; cause not labeled",
        "first_match_selection": "Earliest release among candidates meeting the robust agreement criterion",
        "timing_metrics": "Relative to interval midpoint; descriptive only, not expert-reference timing accuracy",
        "formal_precision_recall_f1": False,
        "original_operating_point": original_point, "runtime": runtime,
        "source_sha256": hashes, "script_sha256": digest(Path(__file__)),
    }
    (out / "protocol.json").write_text(json.dumps(sweep_protocol, indent=2) + "\n")
    caches, originals, baseline_metrics = {}, {}, []
    for c in read_csv(BASE / "candidates.csv"):
        originals.setdefault(c["trace_name"], []).append(c)
    for note, path in zip(notes, cache_paths, strict=True):
        with np.load(path, allow_pickle=False) as saved:
            cache = V27ScoreCache(candidate_probability=saved["candidate_probability"],
                event_probability={"original": saved["event_probability__original"]},
                arrival_age_fraction=saved["arrival_age_fraction"], quality_valid=saved["quality_valid"],
                candidate_allowed=saved["candidate_allowed"], diagnostics={})
        caches[note["trace_name"]] = cache
        found = candidates_from_cache(cache, event_head_key="original", operating_point=original_point, runtime=runtime)
        old = originals.get(note["trace_name"], [])
        assert len(found) == len(old)
        for a, b in zip(found, old, strict=True):
            for key in ("estimated_p_epoch", "available_epoch", "episode_start_packet", "episode_end_packet"):
                assert float(a[key]) == float(b[key]), (note["station"], key)
        row, _ = record_metrics(note, found)
        assert row["comparison"] == note["comparison"]
        baseline_metrics.append(row)
    baseline = summarize(baseline_metrics)
    assert baseline["agreement_records"] == 35 and baseline["borderline_records"] == 0
    baseline_hits = {r["station"] for r in baseline_metrics if r["comparison"] == "agreement"}
    summaries, record_rows, candidate_rows = [], [], []
    for q in Q_THRESHOLDS:
        for e in E_THRESHOLDS:
            point = {**original_point, "candidate_threshold": q, "event_threshold": e}
            local_rows = []
            for note in notes:
                candidates = candidates_from_cache(caches[note["trace_name"]], event_head_key="original",
                                                   operating_point=point, runtime=runtime)
                row, labels = record_metrics(note, candidates)
                row = {"q_threshold": q, "e_threshold": e, **row}
                local_rows.append(row)
                record_rows.append(row)
                for candidate, label in zip(candidates, labels, strict=True):
                    candidate_rows.append(dict(q_threshold=q, e_threshold=e, station=note["station"],
                                               trace_name=note["trace_name"], visual_comparison=label, **candidate))
            summary = dict(q_threshold=q, e_threshold=e, **summarize(local_rows))
            current = {r["station"] for r in local_rows if r["comparison"] == "agreement"}
            summary["gained_vs_original"] = ";".join(sorted(current - baseline_hits))
            summary["lost_vs_original"] = ";".join(sorted(baseline_hits - current))
            assert summary["agreement_records"] + summary["borderline_records"] + summary["no_agreement_records"] == 49
            assert summary["agreement_records"] == summary["after_10_s_hits"] + summary["before_10_s_hits"]
            summaries.append(summary)
    baseline_scan = next(r for r in summaries if r["q_threshold"] == .1 and r["e_threshold"] == .975)
    assert all(baseline_scan[k] == v for k, v in baseline.items())
    assert all(digest(Path(p)) == expected for p, expected in hashes.items())
    write_csv(out / "threshold_summary.csv", summaries)
    write_csv(out / "record_results.csv", record_rows)
    write_csv(out / "candidate_results.csv", candidate_rows)
    result = {
        "baseline": baseline_scan, "grid_points": len(summaries), "record_evaluations": len(record_rows),
        "max_observed_agreement": max(r["agreement_records"] for r in summaries),
        "summary": summaries,
        "qa": {"baseline_saved_candidates_exactly_reproduced": True, "self_checks_passed": True,
               "all_source_hashes_unchanged": True, "neural_inference_repeated": False,
               "production_threshold_changed": False, "manuscript_changed": False},
    }
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "summary"}, indent=2))
    for row in summaries:
        if row["q_threshold"] == .1:
            print(json.dumps(row))
    print("COMPLETE", out)


if __name__ == "__main__":
    main()
