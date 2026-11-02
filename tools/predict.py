"""Packet-causal inference on an explicitly prepared 100 Hz ZNE .npy array."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cspp.evaluation.score_cache import (
    load_ensemble,
    score_waveform_packets,
    sha256_file,
)
from cspp.streaming.candidate import PacketCandidateState

RUNTIME = dict(
    sample_rate_hz=100.0,
    packet_duration_seconds=0.5,
    warmup_packets=20,
    temporal_block_packets=64,
)


def load_model(checkpoint):
    return load_ensemble(
        {
            "primary": {
                "path": str(checkpoint),
                "sha256": sha256_file(checkpoint),
                "verifier_feature_mode": "none",
            }
        },
        device_name="cpu",
        torch_threads=1,
    )


def predict(wave, ensemble, q=0.1, e=0.975):
    wave = np.asarray(wave, dtype=np.float32)
    if wave.ndim != 2 or wave.shape[0] != 3 or wave.shape[1] % 50:
        raise ValueError(
            "Input must have shape (3, N), ZNE, 100 Hz, with N divisible by 50"
        )
    cache = score_waveform_packets(
        wave, start_epoch=0.0, ensemble=ensemble, runtime=RUNTIME
    )
    state = PacketCandidateState(4)
    candidates = []
    for index in range(len(cache.quality_valid)):
        emitted = state.step(
            packet_index=index,
            valid=bool(cache.quality_valid[index]),
            candidate_allowed=bool(cache.candidate_allowed[index]),
            candidate_probability=float(cache.candidate_probability[index]),
            event_probability=float(cache.event_probability["primary"][index]),
            arrival_age_fraction=float(cache.arrival_age_fraction[index]),
            candidate_threshold=q,
            event_threshold=e,
            start_epoch=0.0,
            packet_seconds=0.5,
            max_arrival_age_packets=4.0,
        )
        if emitted is not None:
            candidates.append(emitted)
    return cache, candidates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--checkpoint", type=Path, default=ROOT / "models/cspp_seed20261101.pt"
    )
    parser.add_argument("--candidate-threshold", type=float, default=0.1)
    parser.add_argument("--event-threshold", type=float, default=0.975)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    cache, candidates = predict(
        np.load(args.input, allow_pickle=False),
        load_model(args.checkpoint),
        args.candidate_threshold,
        args.event_threshold,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "candidates": candidates,
                "diagnostics": cache.diagnostics,
                "sample_rate_hz": 100,
                "component_order": "ZNE",
                "time_origin": "start of input",
                "candidate_threshold": args.candidate_threshold,
                "event_threshold": args.event_threshold,
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"candidates": len(candidates), "output": str(args.output)}))


if __name__ == "__main__":
    main()
