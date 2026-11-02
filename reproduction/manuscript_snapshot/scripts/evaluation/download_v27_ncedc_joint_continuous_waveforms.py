#!/usr/bin/env python3
"""Download the frozen event-bearing NCSN station-days."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/evaluation"))
import download_v27_ncedc_continuous_waveforms as base  # noqa: E402


DEFAULT_ROOT = Path(
    "workspace/"
    "ncedc_joint_continuous_m3_20260904"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument("--limit-station-days", type=int)
    parser.add_argument("--retry-errors", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 1 <= args.workers <= 8:
        raise ValueError("--workers must be between 1 and 8")
    protocol = json.loads((args.root / "joint_continuous_protocol.json").read_text())
    manifest = Path(protocol["artifacts"]["station_day_requests"]["path"])
    if base.sha256_file(manifest) != protocol["artifacts"]["station_day_requests"]["sha256"]:
        raise RuntimeError("Station-day manifest drifted")
    requests = base.load_csv(manifest)
    requests.sort(key=lambda row: (row["date_utc"], int(row["availability_rank"])))
    if args.limit_station_days is not None:
        if args.limit_station_days <= 0:
            raise ValueError("--limit-station-days must be positive")
        requests = requests[: args.limit_station_days]
    output = args.root / "waveforms"
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / "status.jsonl"
    latest = base.load_latest_status(status_path)
    terminal = {"complete", "partial"}
    pending = []
    for index, row in enumerate(requests, start=1):
        previous = latest.get(base.station_day_id(row))
        if previous and previous.get("status") in terminal:
            continue
        if previous and previous.get("status") == "download_or_parse_error" and not args.retry_errors:
            continue
        pending.append((index, row))
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                base.process_station_day,
                row,
                output,
                args.timeout_seconds,
                0.1,
            ): (index, row)
            for index, row in pending
        }
        for future in as_completed(futures):
            index, _ = futures[future]
            result = future.result()
            base.append_status(status_path, result)
            latest[result["station_day_id"]] = result
            print(
                f"[{index}/{len(requests)}] {result['station_day_id']}: "
                f"{result['status']} {result['mseed_bytes']} bytes",
                flush=True,
            )
    rows = [latest[base.station_day_id(row)] for row in requests if base.station_day_id(row) in latest]
    fields = [
        "station_day_id", "date_utc", "network", "station", "location",
        "native_sample_rate_hz", "status", "reason",
        "minimum_component_available_fraction", "mseed_file", "mseed_bytes",
        "mseed_sha256", "qc_file", "request_sha256", "checked_at_utc",
    ]
    base.write_csv_atomic(output / "latest_status.csv", rows, fields)
    summary = {
        "summary_id": "cspp_v27_ncedc_joint_continuous_acquisition_v1",
        "requested_station_days": len(requests),
        "completed_station_days": len(rows),
        "status_counts": dict(sorted(Counter(row["status"] for row in rows).items())),
        "data_bytes": sum(int(row["mseed_bytes"]) for row in rows),
        "protocol_sha256": base.sha256_file(args.root / "joint_continuous_protocol.json"),
        "model_outputs_computed": False,
    }
    base.write_json_atomic(output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
