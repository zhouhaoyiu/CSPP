"""Replay the included public NCSN record and check its expected P-wave output."""
import argparse
import json
from pathlib import Path

import numpy as np

from cspp import StreamingPicker

DATA = Path(__file__).resolve().parent / "data/ncsn_75079016_KSM"
DEFAULT_CHECKPOINT = Path(__file__).resolve().parents[1] / "models/cspp_seed20261101.pt"


def run_demo(checkpoint=DEFAULT_CHECKPOINT):
    metadata = json.loads((DATA / "metadata.json").read_text())
    waveform_file = DATA / "waveform.npy"
    wave = np.load(waveform_file, allow_pickle=False)
    picker = StreamingPicker(checkpoint)
    results = list(picker.iter_array(wave))
    picks = [r.pick for r in results if r.pick is not None]
    reference = metadata["reference_p_seconds"]
    summary = {
        "event_id": metadata["event_id"], "station_key": metadata["station_key"],
        "checkpoint": Path(checkpoint).name, "packets": len(results),
        "valid_packets": sum(r.quality_valid for r in results),
        "candidate_allowed_packets": sum(r.candidate_allowed for r in results),
        "candidate_count": len(picks), "reference_p_seconds": reference,
        "picks": [{**p, "pick_error_seconds": p["estimated_p_epoch"] - reference,
                   "output_delay_seconds": p["available_epoch"] - reference} for p in picks],
    }
    return summary, results


def verify(summary):
    expected = json.loads((DATA / "expected_output.json").read_text())
    for key in ("event_id", "station_key", "checkpoint", "packets", "valid_packets", "candidate_allowed_packets", "candidate_count"):
        if summary[key] != expected[key]:
            raise AssertionError(f"{key}: {summary[key]!r} != {expected[key]!r}")
    for actual, target in zip(summary["picks"], expected["picks"], strict=True):
        for key, value in target.items():
            if not np.isclose(actual[key], value, rtol=0, atol=1e-5):
                raise AssertionError(f"Pick field {key} differs: {actual[key]} != {value}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summary, _ = run_demo(args.checkpoint)
    if args.verify:
        verify(summary)
    text = json.dumps(summary, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
