#!/usr/bin/env python3
"""Compare frozen visual intervals with saved predictions; no expert labels implied."""
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from scripts.evaluation.eval_cspp_v27c_fixed_retrospective import candidates_from_cache
from scripts.evaluation.analyze_v27_geonet_temporal_selection import support_with_resets
from cspp_v27_reused_core_score_cache import V27ScoreCache

BASE = Path("workspace/eqanomalynet_knet_full_cases_20260905")
OUT = BASE / "visual_review_66"
TOLERANCE = 1.5


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def agreement(candidates, lo, hi):
    times = [float(c["estimated_p_epoch"]) for c in candidates]
    robust = [t for t in times if max(abs(t-lo), abs(t-hi)) <= TOLERANCE]
    possible = [t for t in times if lo-TOLERANCE <= t <= hi+TOLERANCE]
    return "agreement" if robust else "borderline" if possible else "no_agreement"


def main():
    protocol = json.loads((BASE / "protocol.json").read_text())
    point, runtime = protocol["operating_point"], protocol["runtime"]
    notes = read_csv(OUT / "visual_first_pass.csv")
    records = {r["station"]: r for r in read_csv(BASE / "records.csv") if r["event"] == "19980903165800"}
    original = {}
    for c in read_csv(BASE / "candidates.csv"):
        original.setdefault(c["trace_name"], []).append(c)
    assert len(notes) == len(records) == len({r["station"] for r in notes}) == 66
    assert {r["station"] for r in notes} == records.keys()
    source_files = [OUT / "visual_first_pass.csv", BASE / "records.csv", BASE / "candidates.csv", BASE / "protocol.json", BASE / "result.json"]
    source_hashes = {str(p): sha(p) for p in source_files}
    results, diagnostics = [], []
    for note in notes:
        record = records[note["station"]]
        trace = record["trace_name"]
        candidates = original.get(trace, [])
        item = dict(note, trace_name=trace, epicentral_distance_km=record["epicentral_distance_km"],
                    candidate_count=len(candidates), estimated_p_seconds=";".join(c["estimated_p_epoch"] for c in candidates),
                    release_seconds=";".join(c["available_epoch"] for c in candidates),
                    comparison="uncertain_onset", startup_stratum="unknown",
                    gate_disabled_comparison="not_scored", first_allowed_close_s="")
        if note["visual_status"] == "clear":
            lo, hi = float(note["onset_lower_s"]), float(note["onset_upper_s"])
            item["comparison"] = agreement(candidates, lo, hi)
            item["startup_stratum"] = "before_10_s" if hi < 10 else "after_10_s" if lo >= 10 else "straddles_10_s"
        else:
            lo = hi = None
        with np.load(BASE / "scores" / (trace + ".npz"), allow_pickle=False) as stored:
            valid, allowed = stored["quality_valid"], stored["candidate_allowed"]
            q, e, age = stored["candidate_probability"], stored["event_probability__original"], stored["arrival_age_fraction"]
            cache = V27ScoreCache(candidate_probability=q, arrival_age_fraction=age,
                                  event_probability={"original": e}, quality_valid=valid,
                                  candidate_allowed=allowed, diagnostics={})
            replayed = candidates_from_cache(cache, event_head_key="original", operating_point=point, runtime=runtime)
            assert len(replayed) == len(candidates)
            for new, old in zip(replayed, candidates):
                for key in ("estimated_p_epoch", "available_epoch", "episode_start_packet", "episode_end_packet"):
                    assert float(new[key]) == float(old[key]), (trace, key)
            first = np.flatnonzero(allowed)
            item["first_allowed_close_s"] = float((first[0] + 1) * .5) if len(first) else ""
            if lo is not None:
                # Counterfactual changes eligibility only, not the score arrays or thresholds.
                no_gate = V27ScoreCache(candidate_probability=q, arrival_age_fraction=age,
                                        event_probability={"original": e}, quality_valid=valid,
                                        candidate_allowed=valid.copy(), diagnostics={})
                alternative = candidates_from_cache(no_gate, event_head_key="original", operating_point=point, runtime=runtime)
                item["gate_disabled_comparison"] = agreement(alternative, lo, hi)
                if item["comparison"] != "agreement":
                    close = (np.arange(len(q)) + 1) * .5
                    local = (close >= lo) & (close <= hi + 2.0) & valid
                    support = support_with_resets(q, allowed, valid, point["candidate_threshold"], runtime["candidate_history_packets"])
                    local_supported = local & support
                    diagnostic = {
                        "station": note["station"], "visual_interval_s": [lo, hi],
                        "startup_stratum": item["startup_stratum"],
                        "gate_disabled_comparison": item["gate_disabled_comparison"],
                        "gate_disabled_candidates": alternative,
                        "local_window_definition": "packet close within [visual lower bound, visual upper bound + 2 s]; quality valid only",
                        "local_valid_packets": int(local.sum()),
                        "local_supported_packets": int(local_supported.sum()),
                        "local_max_q": float(q[local].max()) if local.any() else None,
                        "local_max_e": float(e[local].max()) if local.any() else None,
                        "local_max_e_with_support": float(e[local_supported].max()) if local_supported.any() else None,
                    }
                    diagnostics.append(diagnostic)
        results.append(item)

    clear = [r for r in results if r["visual_status"] == "clear"]
    matched = [r for r in clear if r["comparison"] == "agreement"]
    misses = [r for r in clear if r["comparison"] == "no_agreement"]
    early = [r for r in clear if r["startup_stratum"] == "before_10_s"]
    later = [r for r in clear if r["startup_stratum"] == "after_10_s"]
    recovered = [r for r in misses if r["gate_disabled_comparison"] == "agreement"]
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "reviewer": "Codex visual inspection; not independent human-expert annotation",
        "review_order": "All 66 waveforms reviewed without model overlays, uncertain/early cases enlarged, visual intervals saved, then all 66 model-overlay waveforms inspected. Prior experiment context was known; this was not a fully blinded study.",
        "visual_intervals_are": "Approximate sample-relative P-onset intervals, not sample-accurate reference picks",
        "agreement_rule": "Estimated P is within 1.5 s of every point of the visual interval; overlap-only cases are borderline",
        "match_tolerance_s": TOLERANCE,
        "records_reviewed": len(results), "visually_discernible_onsets": len(clear),
        "uncertain_onsets": len(results)-len(clear),
        "records_with_original_candidate": sum(r["candidate_count"] > 0 for r in results),
        "visual_agreement_records": len(matched), "discernible_but_no_agreement": len(misses),
        "borderline_records": sum(r["comparison"] == "borderline" for r in clear),
        "visual_subset_agreement_fraction": len(matched)/len(clear),
        "before_10_s_records": len(early), "before_10_s_agreement": sum(r["comparison"] == "agreement" for r in early),
        "after_10_s_records": len(later), "after_10_s_agreement": sum(r["comparison"] == "agreement" for r in later),
        "gate_disabled_recovered_records": [r["station"] for r in recovered],
        "uncertain_records": [r["station"] for r in results if r["visual_status"] != "clear"],
        "miss_diagnostics": diagnostics,
        "model_thresholds_or_primary_results_changed": False,
        "formal_precision_recall_f1_reported": False,
        "primary_saved_candidates_exactly_reproduced": True,
        "source_sha256": source_hashes,
    }
    with (OUT / "record_review.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    assert all(sha(Path(p)) == expected for p, expected in source_hashes.items())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
