"""Installed CLI for replaying prepared acceleration arrays, without downloads."""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from cspp.picker import StreamingPicker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="100 Hz ZNE acceleration .npy")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-seconds", type=float, default=0.0)
    parser.add_argument("--candidate-threshold", type=float, default=0.1)
    parser.add_argument("--event-threshold", type=float, default=0.975)
    parser.add_argument("--warmup-packets", type=int, default=20)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.input.suffix.lower() != ".npy":
        parser.error("--input must be a prepared .npy array")
    wave = np.load(args.input, allow_pickle=False, mmap_mode="r")
    picker = StreamingPicker(
        args.checkpoint,
        candidate_threshold=args.candidate_threshold,
        event_threshold=args.event_threshold,
        warmup_packets=args.warmup_packets,
    )
    counts = Counter()
    reasons = Counter()
    picks = []
    for result in picker.iter_array(wave, start_seconds=args.start_seconds):
        counts["packets"] += 1
        counts["quality_valid_packets"] += int(result.quality_valid)
        counts["candidate_allowed_packets"] += int(result.candidate_allowed)
        reasons.update(result.reason_codes)
        if result.pick is not None:
            picks.append(result.pick)
    payload = {
        "candidates": picks,
        "diagnostics": {**counts, "quality_reasons": dict(reasons)},
        "checkpoint": args.checkpoint.name,
        "sample_rate_hz": 100,
        "component_order": "ZNE",
        "packet_seconds": 0.5,
        "start_seconds": args.start_seconds,
        "candidate_threshold": args.candidate_threshold,
        "event_threshold": args.event_threshold,
        "warmup_packets": args.warmup_packets,
        "time_basis": "input waveform clock; excludes telemetry and compute time",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"candidates": len(picks), "output": str(args.output)}))


if __name__ == "__main__":
    main()
