"""Offline synthetic smoke test, not a seismic accuracy demonstration."""

import argparse
import json
from pathlib import Path

import numpy as np

from cspp import StreamingPicker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    picker = StreamingPicker(args.checkpoint)
    rng = np.random.default_rng(42)
    picks = []
    valid = allowed = 0
    for index in range(80):
        packet = rng.normal(0, 0.01, (3, 50)).astype(np.float32)
        result = picker.process_packet(packet, timestamp_seconds=index * 0.5)
        valid += int(result.quality_valid)
        allowed += int(result.candidate_allowed)
        if result.pick is not None:
            picks.append(result.pick)
    print(json.dumps({
        "input": "synthetic Gaussian noise; no earthquake labels",
        "packets": 80,
        "valid_packets": valid,
        "candidate_allowed_packets": allowed,
        "candidate_count": len(picks),
        "final_packet": result.to_dict(),
    }, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
