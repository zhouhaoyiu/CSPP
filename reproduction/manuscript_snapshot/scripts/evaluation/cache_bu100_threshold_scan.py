#!/usr/bin/env python3
"""Cache unchanged packet scores for an exploratory BU100 threshold sweep."""
import argparse
import copy
from datetime import datetime, timezone
from pathlib import Path
import sys
import time

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts/evaluation")]
from bu100_external_test import read, write, artifact, sha, write_csv, wave_for
from component_fault_experiment import new_cspp, cspp_candidates
from cspp_v27_hybrid_score_cache import prepare_hybrid_ensemble, score_waveform_packets_hybrid
from cspp.streaming.candidate import PacketCandidateState

BASE = ROOT / "outputs/cspp_bu100_20260910"
OUT = ROOT / "outputs/cspp_bu100_threshold_scan_20260911"
QS = [0.01, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5]
ES = [0.0, 0.25, 0.5, 0.7, 0.75, 0.8, 0.85, 0.9, 0.92, 0.94, 0.95, 0.96, 0.97, 0.975, 0.98, 0.99, 0.995]


def scalar_candidates(scores, q, e):
    state = PacketCandidateState(history_packets=4)
    result = []
    for i, (candidate, event, age, valid, allowed) in enumerate(scores.T):
        c = state.step(packet_index=i, valid=bool(valid), candidate_allowed=bool(allowed),
                       candidate_probability=float(candidate), event_probability=float(event),
                       arrival_age_fraction=float(age), candidate_threshold=q, event_threshold=e,
                       start_epoch=0, packet_seconds=0.5, max_arrival_age_packets=4)
        if c:
            result.append(c)
    return result


def freeze():
    OUT.mkdir(exist_ok=True)
    assert not (OUT / "protocol.json").exists(), "Sweep protocol already frozen"
    sources = [BASE / name for name in ["protocol.json", "conversion_status.json", "record_metrics.csv", "summary.csv",
                                        "waveforms_100hz_zne.h5", "result_verification.json"]]
    original = read(BASE / "protocol.json")
    sources.append(Path(original["models"]["cspp"]["path"]))
    sources.append(ROOT / "scripts/evaluation/sweep_knet49_visual_thresholds.py")
    code = [Path(__file__), ROOT / "scripts/evaluation/scan_bu100_matlab.m",
            ROOT / "scripts/evaluation/bu100_scan_record.m", ROOT / "scripts/evaluation/component_fault_experiment.py",
            ROOT / "scripts/evaluation/bu100_external_test.py", ROOT / "cspp/streaming/candidate.py",
            ROOT / "cspp/evaluation/score_cache.py", ROOT / "src/cspp_v27_hybrid_score_cache.py"]
    p = {"created_utc": datetime.now(timezone.utc).isoformat(), "base_directory": str(BASE),
         "role": "Post-outcome exploratory threshold sweep on the same 2136 records; not independent validation",
         "candidate_thresholds": QS, "event_thresholds": ES,
         "threshold_pairs": [{"candidate_threshold": q, "event_threshold": e} for q in QS for e in ES],
         "grid_source": "Existing sweep_knet49_visual_thresholds.py grid, reused without reference-label rules",
         "zero_event_threshold": "Diagnostic bypass of confirmation threshold; candidate support and quality/startup gate remain",
         "original_thresholds": original["cspp_thresholds"], "model": original["models"]["cspp"],
         "tolerances_seconds": [1.5, 0.5, 0.25], "primary_tolerance_seconds": 1.5,
         "metric_definition": {"TP": "Records with at least one emitted estimated P within tolerance; at most one TP per record",
                               "FP_proxy": "All emitted candidates minus TP; unmatched candidates are not independently verified false alarms",
                               "FN": "Reference records minus TP", "precision": "TP / (TP + FP_proxy); NaN if no candidates",
                               "recall": "TP / reference records", "F1": "2*TP / (reference records + candidate count)",
                               "matching": "Earliest emitted candidate within tolerance, unchanged from frozen test"},
         "selection": "Max F1 on all records at 1.5s; ties prefer higher recall, then higher event threshold, then higher candidate threshold",
         "unchanged": ["model weights", "conversion", "reference P", "quality and startup gate", "normalization", "packet history", "episode onset rule"],
         "source_artifacts": [artifact(x) for x in sources], "code_artifacts": [artifact(x) for x in code]}
    write(OUT / "protocol.json", p)
    print(f"Frozen sweep grid: {len(p['threshold_pairs'])} pairs", flush=True)


def verify_seal():
    p = read(OUT / "protocol.json")
    for s in p["source_artifacts"] + p["code_artifacts"]:
        assert sha(s["path"]) == s["sha256"], s["path"]
    return p


def fixtures():
    rng = np.random.default_rng(20260911)
    out = OUT / "scan_selftest.h5"
    assert not out.exists()
    expected = []
    with h5py.File(out, "w") as h:
        for i, n in enumerate([1, 3, 7, 45, 123]):
            scores = rng.random((5, n))
            scores[3] = rng.random(n) > 0.15
            scores[4] = (rng.random(n) > 0.3) & (scores[3] == 1)
            if n >= 7:
                scores[0, :7] = [.1, .05, .5, .1, .01, .3, .2]
                scores[1, :7] = [.975, .99, .5, .7, .95, .9, .8]
                scores[3, 2] = 0
                scores[4, 2] = 0
            scores[:3, scores[3] == 0] = np.nan
            ref = n * 0.25
            name = f"synthetic{i}"
            h.create_dataset(name + "/scores", data=scores)
            h[name].attrs["reference_p_seconds"] = ref
            for q in QS:
                for e in ES:
                    candidates = scalar_candidates(scores, q, e)
                    for tol in [1.5, .5, .25]:
                        match = next((c for c in candidates if abs(c["estimated_p_epoch"] - ref) <= tol), None)
                        expected.append(dict(id=name, candidate_threshold=q, event_threshold=e, tolerance=tol,
                                             candidate_count=len(candidates), matched=int(match is not None),
                                             error=match["estimated_p_epoch"] - ref if match else "",
                                             latency=match["available_epoch"] - ref if match else ""))
    write_csv(OUT / "scan_selftest_expected.csv", expected)


def cache():
    verify_seal()
    original = read(BASE / "protocol.json")
    status = [r for r in read(BASE / "conversion_status.json") if r["status"] == "converted"]
    ensemble = new_cspp(original)
    prepare_hybrid_ensemble(ensemble)
    cache_path = OUT / "packet_scores.h5"
    started = time.monotonic()
    with h5py.File(BASE / "waveforms_100hz_zne.h5", "r") as waves, h5py.File(cache_path, "a") as h:
        for i, r in enumerate(status):
            if r["id"] in h:
                continue
            wave = wave_for(waves, r)
            saved = score_waveform_packets_hybrid(wave, start_epoch=0, ensemble=ensemble, runtime=original["runtime"])
            candidates = cspp_candidates(saved, {"start_epoch": 0}, original)
            old = read(BASE / "record_results" / (r["id"] + ".json"))
            assert candidates == old["candidates"], r["id"]
            matrix = np.vstack([saved.candidate_probability, saved.event_probability["C0_seed20261101"],
                                saved.arrival_age_fraction, saved.quality_valid, saved.candidate_allowed]).astype(np.float64)
            dataset = h.create_dataset(r["id"] + "/scores", data=matrix, compression="gzip")
            dataset.attrs["columns_in_matlab"] = "candidate_probability,event_probability,arrival_age_fraction,quality_valid,candidate_allowed"
            if (i + 1) % 100 == 0 or i == len(status) - 1:
                h.flush()
                print(f"Cached {i + 1}/{len(status)}; frozen candidates reproduced; {time.monotonic() - started:.1f}s", flush=True)
    write(OUT / "cache_validation.json", {"records": len(status), "baseline_candidates_exactly_reproduced": True,
                                          "packet_scores": artifact(cache_path)})
    if not (OUT / "scan_selftest.h5").exists():
        fixtures()


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("stage", choices=["freeze", "cache"])
    stage = a.parse_args().stage
    {"freeze": freeze, "cache": cache}[stage]()
