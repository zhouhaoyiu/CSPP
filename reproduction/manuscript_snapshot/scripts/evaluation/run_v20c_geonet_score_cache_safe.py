#!/usr/bin/env python3
"""Run V20c GeoNet score-cache units serially with live memory guards."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any

import psutil


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "evaluation"))

import build_v20c_geonet_score_cache as builder  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol", type=Path, default=builder.DEFAULT_PROTOCOL
    )
    parser.add_argument(
        "--scope", choices=("event", "background"), required=True
    )
    parser.add_argument("--unit-key")
    parser.add_argument(
        "--max-units",
        type=int,
        default=0,
        help="0 runs every incomplete frozen unit.",
    )
    return parser.parse_args()


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def event_units(protocol: dict[str, Any]) -> list[str]:
    rows = builder.protocol_csv(protocol, "events")
    rows.sort(
        key=lambda row: (row["origin_time_utc"], row["event_id"])
    )
    return [row["event_id"] for row in rows]


def background_units(protocol: dict[str, Any]) -> list[str]:
    rows = builder.protocol_csv(protocol, "background_station_days")
    rows.sort(
        key=lambda row: (
            int(row["quiet_rank"]),
            row["date_utc"],
            row["network"],
            row["station"],
            row["location"],
        )
    )
    return [
        "|".join(
            [
                row["date_utc"],
                row["network"],
                row["station"],
                row["location"],
            ]
        )
        for row in rows
    ]


def summary_path(
    protocol: dict[str, Any], scope: str, unit_key: str
) -> Path:
    output_root = builder.resolve(protocol["execution"]["output_root"])
    if scope == "event":
        return output_root / "events" / unit_key / "summary.json"
    date_utc, network, station, location = builder.background_unit_parts(
        unit_key
    )
    site = ".".join([network, station, location])
    return (
        output_root
        / "background"
        / date_utc
        / site
        / "summary.json"
    )


def verify_completed_summary(
    path: Path,
    *,
    protocol_hash: str,
    scope: str,
    unit_key: str,
) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"Score-cache summary is absent: {path}")
    summary = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "protocol_sha256": protocol_hash,
        "scope": scope,
        "unit_key": unit_key,
    }
    for name, value in expected.items():
        if summary.get(name) != value:
            raise RuntimeError(
                f"Score-cache summary differs in {name}: {path}"
            )
    for name, spec in summary["artifacts"].items():
        builder.verify_artifact(
            spec, name=f"completed_output.{scope}.{unit_key}.{name}"
        )
    return summary


def process_tree_rss_bytes(pid: int) -> int | None:
    try:
        process = psutil.Process(pid)
        total = int(process.memory_info().rss)
        members = process.children(recursive=True)
    except psutil.Error:
        return None
    for member in members:
        try:
            total += int(member.memory_info().rss)
        except psutil.Error:
            return None
    return total


def resource_violation(
    child_pid: int,
    execution: dict[str, Any],
) -> tuple[str | None, dict[str, int | float]]:
    virtual = psutil.virtual_memory()
    swap = psutil.swap_memory()
    tree_rss = process_tree_rss_bytes(child_pid)
    metrics: dict[str, int | float] = {
        "tree_rss_bytes": -1 if tree_rss is None else tree_rss,
        "system_available_bytes": int(virtual.available),
        "swap_used_bytes": int(swap.used),
    }
    maximum_rss = int(
        float(execution["maximum_worker_resident_set_gib"])
        * 1024**3
    )
    minimum_available = int(
        float(execution["minimum_system_available_memory_gib"])
        * 1024**3
    )
    maximum_swap = int(
        float(execution["maximum_system_swap_used_gib"])
        * 1024**3
    )
    if tree_rss is None:
        return (
            f"worker process-tree RSS could not be sampled for PID "
            f"{child_pid}",
            metrics,
        )
    if tree_rss > maximum_rss:
        return (
            f"worker process-tree RSS {tree_rss} exceeds {maximum_rss}",
            metrics,
        )
    if int(virtual.available) < minimum_available:
        return (
            f"system available memory {virtual.available} is below "
            f"{minimum_available}",
            metrics,
        )
    if int(swap.used) > maximum_swap:
        return (
            f"system swap use {swap.used} exceeds {maximum_swap}",
            metrics,
        )
    return None, metrics


def memory_guard_poll_seconds(execution: dict[str, Any]) -> float:
    value = float(execution["memory_guard_poll_seconds"])
    if not 0.0 < value <= 0.25:
        raise RuntimeError(
            "Frozen memory-guard polling must be in (0, 0.25] seconds"
        )
    return value


def terminate_process_group(child: subprocess.Popen[Any]) -> None:
    if child.poll() is not None:
        return
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        child.wait(timeout=20.0)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(child.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    child.wait()


def run_unit(
    protocol_path: Path,
    protocol: dict[str, Any],
    protocol_hash: str,
    scope: str,
    unit_key: str,
) -> dict[str, Any]:
    violation, initial_metrics = resource_violation(
        os.getpid(), protocol["execution"]
    )
    if violation is not None:
        raise RuntimeError(
            f"Refusing to start {scope} {unit_key}: {violation}; "
            f"metrics={initial_metrics}"
        )
    command = [
        sys.executable,
        str(Path(builder.__file__).resolve()),
        "--protocol",
        str(protocol_path),
        "--scope",
        scope,
        "--unit-key",
        unit_key,
    ]
    child = subprocess.Popen(
        command,
        cwd=ROOT,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
        start_new_session=True,
    )
    peak_tree_rss = 0
    minimum_available = 2**63 - 1
    maximum_swap = 0
    violation_message = None
    poll_seconds = memory_guard_poll_seconds(protocol["execution"])
    try:
        while child.poll() is None:
            time.sleep(poll_seconds)
            violation, metrics = resource_violation(
                child.pid, protocol["execution"]
            )
            peak_tree_rss = max(
                peak_tree_rss, int(metrics["tree_rss_bytes"])
            )
            minimum_available = min(
                minimum_available,
                int(metrics["system_available_bytes"]),
            )
            maximum_swap = max(
                maximum_swap, int(metrics["swap_used_bytes"])
            )
            if violation is not None:
                violation_message = violation
                terminate_process_group(child)
                break
    finally:
        if child.poll() is None:
            terminate_process_group(child)
    if violation_message is not None:
        raise RuntimeError(
            f"Stopped {scope} {unit_key}: {violation_message}; "
            f"peak_tree_rss={peak_tree_rss}, "
            f"minimum_available={minimum_available}, "
            f"maximum_swap={maximum_swap}"
        )
    if int(child.returncode) != 0:
        raise RuntimeError(
            f"Score-cache unit failed ({child.returncode}): "
            f"{scope} {unit_key}"
        )
    summary = verify_completed_summary(
        summary_path(protocol, scope, unit_key),
        protocol_hash=protocol_hash,
        scope=scope,
        unit_key=unit_key,
    )
    maximum_rss = int(
        float(protocol["execution"]["maximum_worker_resident_set_gib"])
        * 1024**3
    )
    reported_rss = int(summary["worker_max_resident_set_bytes"])
    if reported_rss > maximum_rss:
        raise RuntimeError(
            f"Completed worker exceeded frozen RSS limit: "
            f"{reported_rss} > {maximum_rss}"
        )
    return {
        "unit_key": unit_key,
        "summary_path": str(
            summary_path(protocol, scope, unit_key).resolve()
        ),
        "runtime_seconds": float(summary["runtime_seconds"]),
        "worker_max_resident_set_bytes": reported_rss,
        "observed_peak_process_tree_rss_bytes": peak_tree_rss,
        "minimum_system_available_bytes": minimum_available,
        "maximum_system_swap_used_bytes": maximum_swap,
    }


def main() -> None:
    args = parse_args()
    protocol_path = args.protocol.resolve()
    protocol, protocol_hash = builder.load_protocol(protocol_path)
    execution = protocol["execution"]
    if str(execution["inference_device"]) != "mps":
        raise RuntimeError("Safe V20c runner requires frozen MPS execution")
    if int(execution["workers"]) != 1:
        raise RuntimeError("Safe V20c runner requires exactly one worker")
    if not bool(execution["fresh_process_per_unit"]):
        raise RuntimeError("Safe V20c runner requires fresh unit processes")
    memory_guard_poll_seconds(execution)

    all_units = (
        event_units(protocol)
        if args.scope == "event"
        else background_units(protocol)
    )
    if args.unit_key:
        if args.unit_key not in all_units:
            raise KeyError(
                f"Unit is outside frozen {args.scope} cohort: "
                f"{args.unit_key}"
            )
        requested_units = [args.unit_key]
    else:
        requested_units = list(all_units)
    if args.max_units < 0:
        raise ValueError("--max-units cannot be negative")

    skipped_verified = []
    units = []
    for unit_key in requested_units:
        path = summary_path(protocol, args.scope, unit_key)
        if path.is_file():
            verify_completed_summary(
                path,
                protocol_hash=protocol_hash,
                scope=args.scope,
                unit_key=unit_key,
            )
            skipped_verified.append(unit_key)
        else:
            units.append(unit_key)
    if args.max_units:
        units = units[: args.max_units]

    output_root = builder.resolve(execution["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    lock_path = builder.resolve(execution["mps_global_lock_path"])
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_handle = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(
            lock_handle.fileno(),
            fcntl.LOCK_EX | fcntl.LOCK_NB,
        )
    except BlockingIOError as error:
        lock_handle.close()
        raise RuntimeError(
            f"Another V20c MPS score-cache run holds {lock_path}"
        ) from error

    completed = []
    started = time.time()
    try:
        for index, unit_key in enumerate(units, start=1):
            print(
                f"V20c {args.scope} [{index}/{len(units)}] {unit_key}",
                flush=True,
            )
            completed.append(
                run_unit(
                    protocol_path,
                    protocol,
                    protocol_hash,
                    args.scope,
                    unit_key,
                )
            )
    finally:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        lock_handle.close()

    verified = []
    for unit_key in all_units:
        path = summary_path(protocol, args.scope, unit_key)
        if not path.is_file():
            continue
        summary = verify_completed_summary(
            path,
            protocol_hash=protocol_hash,
            scope=args.scope,
            unit_key=unit_key,
        )
        verified.append(
            {
                "unit_key": unit_key,
                "summary": builder.artifact(path),
                "runtime_seconds": float(summary["runtime_seconds"]),
                "worker_max_resident_set_bytes": int(
                    summary["worker_max_resident_set_bytes"]
                ),
            }
        )
    status = {
        "protocol": builder.PROTOCOL_ID,
        "protocol_sha256": protocol_hash,
        "scope": args.scope,
        "expected_units": len(all_units),
        "verified_completed_units": len(verified),
        "complete": len(verified) == len(all_units),
        "skipped_verified_units": skipped_verified,
        "this_run_units": completed,
        "elapsed_seconds": time.time() - started,
        "maximum_reported_worker_resident_set_bytes": max(
            (
                row["worker_max_resident_set_bytes"]
                for row in verified
            ),
            default=0,
        ),
        "unit_summaries": {
            row["unit_key"]: row["summary"] for row in verified
        },
    }
    status_path = output_root / f"{args.scope}_score_cache_status.json"
    write_json_atomic(status_path, status)
    print(json.dumps(status, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
