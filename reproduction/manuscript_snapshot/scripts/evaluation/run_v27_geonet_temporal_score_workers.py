#!/usr/bin/env python3
"""Run the frozen V27 GeoNet score cache in bounded CPU shards."""
from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL = (
    ROOT
    / "experiments/"
    "cspp_v27_geonet_temporal_evaluation_protocol_v3_20260729.json"
)
BUILDER = (
    ROOT / "scripts/evaluation/build_v27_geonet_temporal_score_cache.py"
)
PYTHON = Path("external_home/miniforge3/envs/zhy/bin/python")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument(
        "--scope", choices=("event", "background"), required=True
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def combined_rss_bytes(processes: list[subprocess.Popen[Any]]) -> int:
    pids = [str(process.pid) for process in processes if process.poll() is None]
    if not pids:
        return 0
    result = subprocess.run(
        ["ps", "-o", "rss=", "-p", ",".join(pids)],
        check=True,
        capture_output=True,
        text=True,
    )
    return sum(int(value) for value in result.stdout.split()) * 1024


def terminate_all(processes: list[subprocess.Popen[Any]]) -> None:
    for process in processes:
        if process.poll() is None:
            process.send_signal(signal.SIGTERM)
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        if all(process.poll() is not None for process in processes):
            return
        time.sleep(0.2)
    for process in processes:
        if process.poll() is None:
            process.kill()


def main() -> None:
    args = parse_args()
    protocol_path = args.protocol.resolve()
    actual_sha256 = sha256_file(protocol_path)
    if actual_sha256 != args.expected_protocol_sha256:
        raise RuntimeError("Evaluation protocol SHA-256 differs")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    execution = protocol["execution"]
    if execution["inference_device"] != "cpu":
        raise RuntimeError("Bounded worker runner requires frozen CPU scoring")
    workers = int(execution["inference_processes"])
    threads = int(execution["torch_threads"])
    if workers != 4 or threads != 3:
        raise RuntimeError("Unexpected frozen CPU worker topology")
    maximum_rss = int(execution["maximum_combined_worker_rss_bytes"])
    if maximum_rss > 14 * 1024**3:
        raise RuntimeError("Frozen worker RSS ceiling exceeds 14 GiB")

    output_root = Path(execution["output_root"]).resolve()
    if (
        not output_root.is_relative_to(Path("external_volume"))
        or str(output_root).startswith("external_volume")
    ):
        raise RuntimeError("Evaluation outputs must stay on zhouhy")
    logs_root = output_root / "logs" / args.scope
    logs_root.mkdir(parents=True, exist_ok=True)
    run_path = logs_root / "worker_run.json"
    if run_path.exists():
        raise FileExistsError(run_path)

    environment = os.environ.copy()
    environment.update(
        {
            "OMP_NUM_THREADS": str(threads),
            "MKL_NUM_THREADS": str(threads),
            "OPENBLAS_NUM_THREADS": str(threads),
            "VECLIB_MAXIMUM_THREADS": str(threads),
        }
    )
    processes: list[subprocess.Popen[Any]] = []
    log_handles = []
    commands = []
    started = datetime.now(UTC).isoformat()
    try:
        for shard_index in range(workers):
            command = [
                str(PYTHON),
                str(BUILDER),
                "--protocol",
                str(protocol_path),
                "--expected-protocol-sha256",
                actual_sha256,
                "--scope",
                args.scope,
                "--device",
                "cpu",
                "--shard-count",
                str(workers),
                "--shard-index",
                str(shard_index),
            ]
            log_path = logs_root / f"shard_{shard_index}.log"
            handle = log_path.open("w", encoding="utf-8")
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            processes.append(process)
            log_handles.append(handle)
            commands.append(
                {
                    "shard_index": shard_index,
                    "pid": process.pid,
                    "command": command,
                    "log_path": str(log_path),
                }
            )

        peak_rss = 0
        last_report = 0.0
        while any(process.poll() is None for process in processes):
            rss = combined_rss_bytes(processes)
            peak_rss = max(peak_rss, rss)
            if rss > maximum_rss:
                raise MemoryError(
                    f"Combined worker RSS {rss} exceeds {maximum_rss}"
                )
            now = time.monotonic()
            if now - last_report >= 30.0:
                states = [process.poll() for process in processes]
                print(
                    f"{args.scope} workers states={states} "
                    f"combined_rss_gib={rss / 1024**3:.2f}",
                    flush=True,
                )
                last_report = now
            time.sleep(2.0)
        return_codes = [process.returncode for process in processes]
        if any(code != 0 for code in return_codes):
            raise RuntimeError(f"Worker failure: {return_codes}")
    except BaseException:
        terminate_all(processes)
        raise
    finally:
        for handle in log_handles:
            handle.close()

    finalize = [
        str(PYTHON),
        str(BUILDER),
        "--protocol",
        str(protocol_path),
        "--expected-protocol-sha256",
        actual_sha256,
        "--scope",
        args.scope,
        "--device",
        "cpu",
        "--finalize-only",
    ]
    subprocess.run(finalize, cwd=ROOT, env=environment, check=True)
    completed = datetime.now(UTC).isoformat()
    write_json(
        run_path,
        {
            "run_id": "cspp_v27_geonet_temporal_bounded_cpu_workers_v1",
            "protocol_sha256": actual_sha256,
            "scope": args.scope,
            "started_at_utc": started,
            "completed_at_utc": completed,
            "workers": workers,
            "threads_per_worker": threads,
            "maximum_combined_worker_rss_bytes": maximum_rss,
            "observed_peak_combined_worker_rss_bytes": peak_rss,
            "commands": commands,
            "return_codes": return_codes,
            "scope_completion": str(
                output_root / f"{args.scope}_score_cache_complete.json"
            ),
        },
    )
    print(run_path, flush=True)


if __name__ == "__main__":
    main()
