"""Replay CSPP on the distributed cohort using a separately obtained SeisBench STEAD HDF5."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from predict import ROOT, load_model, predict
from reproduce_metrics import point_metrics, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--waveforms", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--split", choices=("train", "dev", "test"))
    parser.add_argument("--candidate-threshold", type=float, default=0.1)
    parser.add_argument("--event-threshold", type=float, default=0.975)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    import h5py
    from seisbench.data.base import WaveformDataset

    cohort = rows(ROOT / "evidence/stead/threshold_comparison/cohort.csv")
    if args.split:
        cohort = [r for r in cohort if r["split"] == args.split]
    if args.limit is not None:
        if args.limit < 1:
            parser.error("--limit must be positive")
        cohort = cohort[: args.limit]
    ensemble = load_model(ROOT / "models/cspp_seed20261101.pt")
    output = []
    with h5py.File(args.waveforms, "r") as handle:
        for row in cohort:
            block, location = row["trace_name"].split("$", 1)
            wave = np.asarray(
                handle["data"][block][WaveformDataset._parse_location(location)],
                dtype=np.float32,
            )
            _, candidates = predict(
                wave, ensemble, args.candidate_threshold, args.event_threshold
            )
            p_time = float(row["trace_p_arrival_sample"]) / 100
            matching = [
                c for c in candidates if abs(c["estimated_p_epoch"] - p_time) <= 1.5
            ]
            selected = (
                min(
                    matching,
                    key=lambda c: (
                        abs(c["estimated_p_epoch"] - p_time),
                        c["available_epoch"],
                    ),
                )
                if matching
                else None
            )
            output.append(
                {
                    "trace_name": row["trace_name"],
                    "split": row["split"],
                    "candidate_count": len(candidates),
                    "matched": bool(selected),
                    "pick_error_seconds": selected["estimated_p_epoch"] - p_time
                    if selected
                    else "",
                    "availability_delay_seconds": selected["available_epoch"] - p_time
                    if selected
                    else "",
                }
            )
    args.output_dir.mkdir(parents=True)
    with (args.output_dir / "records.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    result = {
        "scope": "Limited engineering replay"
        if args.limit
        else "Cohort waveform replay",
        "q": args.candidate_threshold,
        "e": args.event_threshold,
        "metrics": point_metrics(output),
    }
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
