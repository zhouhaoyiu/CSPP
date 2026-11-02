#!/usr/bin/env python3
"""Run the sealed GeoNet holdout score cache in bounded CPU shards."""
from __future__ import annotations

import argparse
from datetime import UTC, datetime
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any, TextIO

import psutil


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.evaluation import v27_runner_lease as runner_lease  # noqa: E402


WORK_ROOT = Path("workspace").resolve()
DEFAULT_PROTOCOL = (
    WORK_ROOT
    / "protocol_freeze/cspp_v27c_geonet_final_evaluation_protocol_20260729.json"
)
BUILDER = (
    ROOT / "scripts/evaluation/build_v27_geonet_final_score_cache.py"
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
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_once(path: Path, payload: Any) -> None:
    if path.exists():
        raise FileExistsError(path)
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


def write_json_replace(path: Path, payload: Any) -> None:
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


def process_table() -> dict[int, tuple[int, int]]:
    result = subprocess.run(
        ["ps", "-axo", "pid=,ppid=,rss="],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Cannot sample the process table for the RSS guard: "
            f"{result.stderr.strip()}"
        )
    table = {}
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) == 3:
            pid, parent, rss_kib = map(int, fields)
            table[pid] = (parent, rss_kib * 1024)
    if not table:
        raise RuntimeError("RSS guard received an empty process table")
    return table


def process_tree_rss_bytes(
    root_pid: int,
    table: dict[int, tuple[int, int]],
) -> int:
    descendants = {int(root_pid)}
    changed = True
    while changed:
        changed = False
        for pid, (parent, _) in table.items():
            if parent in descendants and pid not in descendants:
                descendants.add(pid)
                changed = True
    return sum(table[pid][1] for pid in descendants if pid in table)


def worker_rss_bytes(
    processes: list[subprocess.Popen[Any]],
) -> list[int]:
    table = process_table()
    values = []
    for process in processes:
        if process.poll() is not None:
            values.append(0)
            continue
        value = process_tree_rss_bytes(process.pid, table)
        if value <= 0:
            if process.poll() is not None:
                values.append(0)
                continue
            raise RuntimeError(
                f"RSS guard cannot locate live worker PID {process.pid}"
            )
        values.append(value)
    return values


def terminate_all(processes: list[subprocess.Popen[Any]]) -> None:
    for process in processes:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        if all(process.poll() is not None for process in processes):
            return
        time.sleep(0.2)
    for process in processes:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def pid_is_alive(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def active_conflicting_processes(
    patterns: list[str],
) -> list[dict[str, Any]]:
    own_process = psutil.Process()
    excluded_pids = {
        own_process.pid,
        *(parent.pid for parent in own_process.parents()),
    }
    conflicts = []
    for process in psutil.process_iter(["pid", "cmdline"]):
        try:
            pid = int(process.info["pid"])
            command = " ".join(process.info.get("cmdline") or [])
        except (psutil.Error, TypeError, ValueError):
            continue
        if pid in excluded_pids:
            continue
        matched = [
            pattern for pattern in patterns if pattern in command
        ]
        if matched:
            conflicts.append(
                {"pid": pid, "patterns": matched, "command": command}
            )
    return conflicts


def acquire_shared_evaluation_lock(execution: dict[str, Any]) -> TextIO:
    path = Path(execution["global_lock_path"]).resolve()
    if (
        not path.is_relative_to(WORK_ROOT)
        or path.is_relative_to(Path("external_volume"))
    ):
        raise RuntimeError("Shared evaluation lock must remain on zhouhy")
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        handle.close()
        raise RuntimeError(
            "Another heavy V27 evaluation holds the shared lock"
        ) from error
    return handle


def release_shared_evaluation_lock(handle: TextIO) -> None:
    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    handle.close()


def acquire_global_lock(
    output_root: Path,
    *,
    protocol_sha256: str,
    scope: str,
    started_at_utc: str,
) -> Path:
    lock = output_root / ".active_score_run.lock"
    output_root.mkdir(parents=True, exist_ok=True)
    for _ in range(2):
        try:
            lock.mkdir()
        except FileExistsError:
            lease_path = lock / "lease.json"
            lease = (
                json.loads(lease_path.read_text(encoding="utf-8"))
                if lease_path.is_file()
                else {}
            )
            live = [
                int(pid)
                for pid in [
                    lease.get("runner_pid"),
                    *lease.get("worker_pids", []),
                ]
                if pid is not None and pid_is_alive(int(pid))
            ]
            if live:
                raise RuntimeError(
                    "Another final score run is active: "
                    f"scope={lease.get('scope')}, pids={live}"
                )
            stale = output_root / (
                ".stale_score_run_lock_"
                f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}_"
                f"{os.getpid()}"
            )
            lock.replace(stale)
            continue
        write_json_replace(
            lock / "lease.json",
            {
                "lease_id": "cspp_v27c_final_score_global_lease_v1",
                "protocol_sha256": protocol_sha256,
                "scope": scope,
                "started_at_utc": started_at_utc,
                "runner_pid": os.getpid(),
                "worker_pids": [],
            },
        )
        return lock
    raise RuntimeError("Could not acquire the final score global lock")


def release_global_lock(lock: Path) -> None:
    lease = lock / "lease.json"
    if lease.exists():
        lease.unlink()
    if lock.exists():
        lock.rmdir()


def release_run_locks(lock: Path, shared_lock: TextIO) -> None:
    try:
        release_global_lock(lock)
    finally:
        release_shared_evaluation_lock(shared_lock)


def main() -> None:
    args = parse_args()
    protocol_path = args.protocol.resolve()
    actual_sha256 = sha256_file(protocol_path)
    if actual_sha256 != args.expected_protocol_sha256:
        raise RuntimeError("Final evaluation protocol SHA-256 differs")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("protocol_id") != (
        "cspp_v27c_geonet_final_evaluation_v1"
    ):
        raise RuntimeError("Unexpected final evaluation protocol")
    execution = protocol["execution"]
    workers = int(execution["inference_processes"])
    threads = int(execution["torch_threads_per_process"])
    interop_threads = int(
        execution["torch_interop_threads_per_process"]
    )
    maximum_rss = int(
        float(execution["combined_worker_rss_limit_gib"]) * 1024**3
    )
    maximum_worker_rss = int(
        float(execution["per_worker_rss_limit_gib"]) * 1024**3
    )
    minimum_system_available = int(
        float(execution["minimum_system_available_memory_gib"])
        * 1024**3
    )
    maximum_system_swap = int(
        float(execution["maximum_system_swap_used_gib"]) * 1024**3
    )
    poll_interval = float(execution["rss_poll_interval_seconds"])
    if (
        execution["inference_device"] != "cpu"
        or workers != 4
        or threads != 3
        or interop_threads != 1
        or maximum_rss > 14 * 1024**3
        or maximum_worker_rss > 4 * 1024**3
        or minimum_system_available < 8 * 1024**3
        or maximum_system_swap > 4 * 1024**3
        or not 0.1 <= poll_interval <= 1.0
        or execution["mps_multi_process_permitted"] is not False
    ):
        raise RuntimeError("Unexpected frozen CPU worker topology")

    output_root = Path(execution["score_output_root"]).resolve()
    if (
        not output_root.is_relative_to(WORK_ROOT)
        or output_root.is_relative_to(Path("external_volume"))
    ):
        raise RuntimeError("Final score output must remain on zhouhy")
    logs_root = output_root / "logs" / args.scope
    logs_root.mkdir(parents=True, exist_ok=True)
    run_path = logs_root / "worker_run.json"
    completion_path = (
        output_root / f"{args.scope}_score_cache_complete.json"
    )
    if run_path.is_file() and completion_path.is_file():
        run = json.loads(run_path.read_text(encoding="utf-8"))
        completion = json.loads(
            completion_path.read_text(encoding="utf-8")
        )
        if (
            run.get("protocol_sha256") == actual_sha256
            and run.get("scope") == args.scope
            and run.get("status") == "workers_completed_within_limits"
            and completion.get("protocol_sha256") == actual_sha256
            and completion.get("scope") == args.scope
        ):
            audit_path = Path(
                completion["execution_audit"]["path"]
            ).resolve()
            audit_spec = completion["execution_audit"]
            completion_spec = run.get("scope_completion", {})
            if (
                not audit_path.is_file()
                or audit_path.stat().st_size
                != int(audit_spec.get("bytes", -1))
                or sha256_file(audit_path)
                != str(audit_spec.get("sha256", ""))
                or Path(str(completion_spec.get("path", ""))).resolve()
                != completion_path
                or completion_path.stat().st_size
                != int(completion_spec.get("bytes", -1))
                or sha256_file(completion_path)
                != str(completion_spec.get("sha256", ""))
            ):
                raise RuntimeError("Existing final run artifacts differ")
            print(completion_path, flush=True)
            return
        raise RuntimeError("Existing final run seal differs")
    if run_path.exists() or completion_path.exists():
        raise RuntimeError("Partial final run seal requires audit")

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
    attempt_root = logs_root / (
        "attempt_"
        f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}_{os.getpid()}"
    )
    shared_lock = acquire_shared_evaluation_lock(execution)
    lock: Path | None = None
    try:
        conflicts = active_conflicting_processes(
            list(execution["exclusive_process_patterns"])
        )
        if conflicts:
            raise RuntimeError(
                f"Another heavy V27 evaluation is active: {conflicts}"
            )
        lock = acquire_global_lock(
            output_root,
            protocol_sha256=actual_sha256,
            scope=args.scope,
            started_at_utc=started,
        )
        attempt_root.mkdir()
    except BaseException:
        if lock is not None:
            release_run_locks(lock, shared_lock)
        else:
            release_shared_evaluation_lock(shared_lock)
        raise
    assert lock is not None
    lease_path = attempt_root / "runner_lease.json"
    try:
        lease = runner_lease.create_lease(
            protocol_sha256=actual_sha256,
            unit={"scope": args.scope},
            runner_script=Path(__file__).resolve(),
            evaluator_script=BUILDER,
            attempt_root=attempt_root,
            published_root=output_root / args.scope,
        )
        runner_lease.write_json_exclusive(lease_path, lease)
    except BaseException:
        release_run_locks(lock, shared_lock)
        raise
    run_identity = {
        "runner_lease_sha256": sha256_file(lease_path),
        "lease_nonce": str(lease["nonce"]),
        "runner_pid": int(lease["runner_pid"]),
    }
    previous_sigterm = signal.getsignal(signal.SIGTERM)

    def handle_sigterm(signum: int, _frame: Any) -> None:
        raise SystemExit(f"Received signal {signum}")

    signal.signal(signal.SIGTERM, handle_sigterm)
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
                "--runner-lease",
                str(lease_path),
                "--attempt-output-root",
                str(attempt_root),
            ]
            log_path = attempt_root / f"shard_{shard_index}.log"
            handle = log_path.open("w", encoding="utf-8")
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
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
        write_json_replace(
            lock / "lease.json",
            {
                "lease_id": "cspp_v27c_final_score_global_lease_v1",
                "protocol_sha256": actual_sha256,
                "scope": args.scope,
                "started_at_utc": started,
                "runner_pid": os.getpid(),
                "worker_pids": [
                    process.pid for process in processes
                ],
            },
        )

        peak_rss = 0
        peak_worker_rss = [0] * workers
        minimum_observed_system_available = 2**63 - 1
        maximum_observed_system_swap = 0
        resource_sample_count = 0
        last_report = 0.0

        def sample_resources() -> int:
            nonlocal peak_rss
            nonlocal peak_worker_rss
            nonlocal minimum_observed_system_available
            nonlocal maximum_observed_system_swap
            nonlocal resource_sample_count
            worker_rss = worker_rss_bytes(processes)
            rss = sum(worker_rss)
            peak_rss = max(peak_rss, rss)
            peak_worker_rss = [
                max(previous, observed)
                for previous, observed in zip(
                    peak_worker_rss, worker_rss, strict=True
                )
            ]
            virtual = psutil.virtual_memory()
            swap = psutil.swap_memory()
            minimum_observed_system_available = min(
                minimum_observed_system_available,
                int(virtual.available),
            )
            maximum_observed_system_swap = max(
                maximum_observed_system_swap,
                int(swap.used),
            )
            resource_sample_count += 1
            if rss > maximum_rss:
                raise MemoryError(
                    f"Combined worker RSS {rss} exceeds {maximum_rss}"
                )
            if any(value > maximum_worker_rss for value in worker_rss):
                raise MemoryError(
                    "Per-worker RSS exceeds the frozen limit: "
                    f"{worker_rss}"
                )
            if int(virtual.available) < minimum_system_available:
                raise MemoryError(
                    "System available memory is below the frozen limit: "
                    f"{virtual.available} < {minimum_system_available}"
                )
            if int(swap.used) > maximum_system_swap:
                raise MemoryError(
                    "System swap use exceeds the frozen limit: "
                    f"{swap.used} > {maximum_system_swap}"
                )
            return rss

        sample_resources()
        while any(process.poll() is None for process in processes):
            failed = [
                process.returncode
                for process in processes
                if process.poll() not in (None, 0)
            ]
            if failed:
                raise RuntimeError(
                    f"Final score worker failed early: {failed}"
                )
            rss = sample_resources()
            now = time.monotonic()
            if now - last_report >= 30.0:
                print(
                    f"{args.scope} workers states="
                    f"{[process.poll() for process in processes]} "
                    f"combined_rss_gib={rss / 1024**3:.2f}",
                    flush=True,
                )
                last_report = now
            time.sleep(poll_interval)
        return_codes = [process.returncode for process in processes]
        if any(code != 0 for code in return_codes):
            raise RuntimeError(f"Final score worker failure: {return_codes}")
    except BaseException:
        terminate_all(processes)
        release_run_locks(lock, shared_lock)
        raise
    finally:
        for handle in log_handles:
            handle.close()
        signal.signal(signal.SIGTERM, previous_sigterm)

    try:
        audit_payload = {
            "run_id": "cspp_v27c_geonet_final_bounded_cpu_workers_v2",
            "protocol_sha256": actual_sha256,
            "scope": args.scope,
            "status": "workers_completed_within_limits",
            "runner_pid": os.getpid(),
            "started_at_utc": started,
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "workers": workers,
            "threads_per_worker": threads,
            "interop_threads_per_worker": interop_threads,
            "maximum_combined_worker_rss_bytes": maximum_rss,
            "maximum_per_worker_rss_bytes": maximum_worker_rss,
            "minimum_system_available_memory_bytes": (
                minimum_system_available
            ),
            "maximum_system_swap_used_bytes": maximum_system_swap,
            "rss_poll_interval_seconds": poll_interval,
            "observed_peak_combined_worker_rss_bytes": peak_rss,
            "observed_peak_per_worker_rss_bytes": peak_worker_rss,
            "observed_minimum_system_available_memory_bytes": (
                minimum_observed_system_available
            ),
            "observed_maximum_system_swap_used_bytes": (
                maximum_observed_system_swap
            ),
            "resource_sample_count": resource_sample_count,
            "runner_lease": {
                "path": str(lease_path.resolve()),
                "bytes": lease_path.stat().st_size,
                "sha256": sha256_file(lease_path),
            },
            "run_identity": run_identity,
            "commands": commands,
            "return_codes": return_codes,
        }
        execution_audit_path = attempt_root / "execution_audit.json"
        write_json_once(execution_audit_path, audit_payload)
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
            "--shard-count",
            str(workers),
            "--shard-index",
            "0",
            "--finalize-only",
            "--execution-audit",
            str(execution_audit_path),
            "--runner-lease",
            str(lease_path),
            "--attempt-output-root",
            str(attempt_root),
        ]
        subprocess.run(finalize, cwd=ROOT, env=environment, check=True)
        if not completion_path.is_file():
            raise RuntimeError(
                "Final scope completion was not created"
            )
        write_json_once(
            run_path,
            {
                **audit_payload,
                "execution_audit": {
                    "path": str(execution_audit_path),
                    "bytes": execution_audit_path.stat().st_size,
                    "sha256": sha256_file(execution_audit_path),
                },
                "scope_completion": {
                    "path": str(completion_path),
                    "bytes": completion_path.stat().st_size,
                    "sha256": sha256_file(completion_path),
                },
            },
        )
    finally:
        release_run_locks(lock, shared_lock)
    print(completion_path, flush=True)


if __name__ == "__main__":
    main()
