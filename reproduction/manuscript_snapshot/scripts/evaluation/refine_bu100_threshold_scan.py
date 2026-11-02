#!/usr/bin/env python3
"""Freeze a disclosed second-stage grid and reuse the unchanged score cache."""
import argparse
import copy
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts/evaluation")]
import cache_bu100_threshold_scan as cache
from bu100_external_test import read, write, artifact, sha

COARSE = cache.OUT
FINE = COARSE / "refined"
QS = [0.0, 0.0001, 0.0003, 0.001, 0.003, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.05, 0.075, 0.1]
ES = [0.5, 0.525, 0.55, 0.575, 0.6, 0.625, 0.65, 0.675, 0.7, 0.725, 0.75, 0.775, 0.8, 0.825, 0.85, 0.975]


def freeze():
    original = cache.verify_seal()
    assert not (FINE / "protocol.json").exists()
    FINE.mkdir(exist_ok=True)
    p = copy.deepcopy(original)
    p.update(created_utc=datetime.now(timezone.utc).isoformat(),
             candidate_thresholds=QS, event_thresholds=ES,
             threshold_pairs=[{"candidate_threshold": q, "event_threshold": e} for q in QS for e in ES],
             grid_source="Explicit refinement after coarse grid: fill 0.50-0.85 event interval in 0.025 steps and extend q below 0.01; retain original point",
             zero_candidate_threshold="Diagnostic bypass of candidate probability cutoff; quality eligibility, 4-packet support and confirmation remain",
             role="Second-stage same-data exploratory scan informed by coarse results; no independent threshold validation")
    p["source_artifacts"] += [artifact(COARSE / name) for name in ["protocol.json", "threshold_summary.csv",
                                                                  "selected_operating_points.csv", "packet_scores.h5"]]
    p["code_artifacts"].append(artifact(Path(__file__)))
    write(FINE / "protocol.json", p)
    score = COARSE / "packet_scores.h5"
    (FINE / "packet_scores.h5").symlink_to(score)
    assert sha(score) == read(COARSE / "cache_validation.json")["packet_scores"]["sha256"]
    write(FINE / "cache_validation.json", {"records": 2136, "baseline_candidates_exactly_reproduced": True,
                                            "neural_inference_repeated": False, "packet_scores": artifact(score)})
    cache.OUT = FINE; cache.QS = QS; cache.ES = ES
    cache.fixtures()
    print(f"Refinement frozen: {len(QS) * len(ES)} pairs; same packet-score SHA", flush=True)


def verify():
    import verify_bu100_threshold_scan as verifier
    cache.OUT = FINE
    verifier.OUT = FINE
    verifier.main()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["freeze", "verify"])
    {"freeze": freeze, "verify": verify}[parser.parse_args().stage]()
