#!/usr/bin/env python3
"""Run GeoNet final public baselines serially under one guarded lease."""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping

import psutil


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "evaluation"))

from scripts.evaluation import (  # noqa: E402
    eval_v27_geonet_final_public_baselines as evaluator,
)
from scripts.evaluation import (  # noqa: E402
    freeze_v27_geonet_final_public_baseline_protocol as freeze,
)
from scripts.evaluation import (  # noqa: E402
    v27_runner_lease as runner_lease,
)
import run_v20c_geonet_score_cache_safe as safe  # noqa: E402


UNITS = ("phasenet_instance", "eqtransformer_original")
RSS_SAMPLE_FAILURE_PREFIX = "worker process-tree RSS could not be sampled"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=evaluator.DEFAULT_PROTOCOL)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--model", choices=UNITS)
    parser.add_argument("--max-runs", type=int, default=0)
    return parser.parse_args()


def _strict_nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise RuntimeError(f"{label} is not an integer")
    text = str(value)
    if not text.isdigit():
        raise RuntimeError(f"{label} is not a nonnegative integer")
    return int(text)


def _read_records(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != evaluator.OUTPUT_FIELDS:
            raise RuntimeError("Public-baseline records schema differs")
        return list(reader)


def verify_summary(
    path: Path,
    *,
    protocol: Mapping[str, Any],
    protocol_sha256: str,
    model: str,
    storage_root: Path | None = None,
) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"Public-baseline summary is absent: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    methods = sorted(freeze.EXPECTED_METHODS[model])
    expected = {
        "summary_id": "cspp_v27_geonet_final_public_baseline_run_v2",
        "protocol_id": freeze.PROTOCOL_ID,
        "protocol_sha256": protocol_sha256,
        "final_evaluation_protocol_sha256": protocol[
            "final_evaluation_protocol_sha256"
        ],
        "record_manifest_sha256": protocol["record_manifest_sha256"],
        "model_key": model,
        "methods": methods,
        "records_requested": int(protocol["record_count"]),
        "records_emitted_per_method": int(protocol["record_count"]),
        "output_rows": int(protocol["record_count"]) * len(methods),
        "expected_output_rows": int(protocol["record_count"]) * len(methods),
        "comparison_mode": "released_full_record_offline_annotation",
        "availability_latency_generated": False,
        "device": protocol["execution"]["device"],
        "torch_threads": int(protocol["execution"]["torch_threads"]),
        "software_versions": protocol["software_versions"],
        "loaded_state_dict_sha256": protocol["public_models"][model][
            "loaded_state_dict_sha256"
        ],
        "resolved_weight_artifact": protocol["public_models"][model]["artifacts"][
            "weights"
        ],
        "resolved_metadata_artifact": protocol["public_models"][model]["artifacts"][
            "metadata"
        ],
        "output_fields": evaluator.OUTPUT_FIELDS,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(f"Public-baseline summary differs: {key}")
    published_root = evaluator.run_root(protocol, model)
    records_spec = payload.get("records_artifact", {})
    if (
        Path(str(records_spec.get("path", ""))).resolve()
        != (published_root / "records.csv").resolve()
    ):
        raise RuntimeError("Public-baseline records path differs")
    actual_root = storage_root or path.parent
    records_path = actual_root / "records.csv"
    if (
        not records_path.is_file()
        or records_path.stat().st_size != int(records_spec.get("bytes", -1))
        or freeze.sha256_file(records_path) != str(records_spec.get("sha256", ""))
    ):
        raise RuntimeError("Public-baseline records artifact differs")
    rows = _read_records(records_path)
    if len(rows) != expected["output_rows"]:
        raise RuntimeError("Public-baseline records count differs")
    method_counts: dict[str, int] = {}
    technical_records = set()
    for row in rows:
        if row["model_key"] != model:
            raise RuntimeError("Public-baseline record model differs")
        method_counts[row["method_id"]] = method_counts.get(row["method_id"], 0) + 1
        for field in (
            "candidate_count",
            "matched_candidate_count",
            "true_positive",
            "false_positive",
            "false_negative",
        ):
            _strict_nonnegative_int(row[field], field)
        if row["technical_error"]:
            technical_records.add(row["record_id"])
    if method_counts != {method: int(protocol["record_count"]) for method in methods}:
        raise RuntimeError("Public-baseline method coverage differs")
    if len(technical_records) != _strict_nonnegative_int(
        payload.get("technical_failure_records"),
        "technical_failure_records",
    ):
        raise RuntimeError("Public-baseline technical-failure count differs")
    filled_by_record: dict[str, int] = {}
    for row in rows:
        count = _strict_nonnegative_int(
            row["edge_filled_component_samples"],
            "edge_filled_component_samples",
        )
        record_id = row["record_id"]
        if record_id in filled_by_record and filled_by_record[record_id] != count:
            raise RuntimeError("Public-baseline edge-fill count differs by method")
        filled_by_record[record_id] = count
    if sum(value > 0 for value in filled_by_record.values()) != (
        _strict_nonnegative_int(
            payload.get("edge_filled_record_count"),
            "edge_filled_record_count",
        )
    ) or sum(filled_by_record.values()) != _strict_nonnegative_int(
        payload.get("edge_filled_component_samples"),
        "edge_filled_component_samples",
    ):
        raise RuntimeError("Public-baseline edge-fill summary differs")
    maximum_rss = int(
        float(protocol["execution"]["maximum_worker_resident_set_gib"]) * 1024**3
    )
    reported_rss = _strict_nonnegative_int(
        payload.get("worker_max_resident_set_bytes"),
        "worker_max_resident_set_bytes",
    )
    if reported_rss > maximum_rss:
        raise RuntimeError("Public-baseline evaluator exceeded RSS limit")
    return payload


def verify_audit(
    path: Path,
    *,
    protocol: Mapping[str, Any],
    protocol_sha256: str,
    model: str,
    storage_root: Path | None = None,
) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"Public-baseline execution audit is absent: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "audit_id": "cspp_v27_geonet_final_public_baseline_execution_v2",
        "protocol_sha256": protocol_sha256,
        "model": model,
        "return_code": 0,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(f"Public-baseline audit differs: {key}")
    published_root = evaluator.run_root(protocol, model)
    actual_root = storage_root or path.parent
    for name in ("runner_lease", "summary"):
        spec = payload.get(name, {})
        expected_path = published_root / (
            "runner_lease.json" if name == "runner_lease" else "summary.json"
        )
        if Path(str(spec.get("path", ""))).resolve() != expected_path.resolve():
            raise RuntimeError(f"Public-baseline audit {name} path differs")
        actual = actual_root / expected_path.name
        if (
            not actual.is_file()
            or actual.stat().st_size != int(spec.get("bytes", -1))
            or freeze.sha256_file(actual) != str(spec.get("sha256", ""))
        ):
            raise RuntimeError(f"Public-baseline audit {name} differs")
    lease_path = actual_root / "runner_lease.json"
    lease = json.loads(lease_path.read_text(encoding="utf-8"))
    runner_sha = freeze.sha256_file(Path(__file__).resolve())
    evaluator_sha = freeze.sha256_file(Path(evaluator.__file__).resolve())
    if (
        lease.get("lease_id") != runner_lease.LEASE_ID
        or lease.get("protocol_sha256") != protocol_sha256
        or lease.get("unit") != {"model": model}
        or lease.get("runner_script_sha256") != runner_sha
        or lease.get("evaluator_script_sha256") != evaluator_sha
        or payload.get("runner_code_sha256") != runner_sha
        or payload.get("evaluator_code_sha256") != evaluator_sha
    ):
        raise RuntimeError("Public-baseline runner provenance differs")
    command = payload.get("command")
    expected_command = [
        str(Path(protocol["execution"]["python_executable"]).resolve()),
        str(Path(evaluator.__file__).resolve()),
        "--protocol",
        str(command[3]) if isinstance(command, list) and len(command) > 3 else "",
        "--expected-protocol-sha256",
        protocol_sha256,
        "--model",
        model,
        "--runner-lease",
        str(Path(lease["attempt_root"]) / "runner_lease.json"),
        "--attempt-output-root",
        str(lease["attempt_root"]),
    ]
    if command != expected_command:
        raise RuntimeError("Public-baseline evaluator command differs")
    protocol_path = Path(expected_command[3]).resolve()
    if (
        not protocol_path.is_file()
        or freeze.sha256_file(protocol_path) != protocol_sha256
    ):
        raise RuntimeError("Public-baseline command protocol differs")
    summary = json.loads((actual_root / "summary.json").read_text(encoding="utf-8"))
    execution = protocol["execution"]
    reported_rss = _strict_nonnegative_int(
        payload.get("reported_worker_max_resident_set_bytes"),
        "reported_worker_max_resident_set_bytes",
    )
    observed_rss = _strict_nonnegative_int(
        payload.get("observed_peak_process_tree_rss_bytes"),
        "observed_peak_process_tree_rss_bytes",
    )
    sample_count = _strict_nonnegative_int(
        payload.get("resource_sample_count"),
        "resource_sample_count",
    )
    minimum_available = _strict_nonnegative_int(
        payload.get("minimum_system_available_bytes"),
        "minimum_system_available_bytes",
    )
    maximum_swap = _strict_nonnegative_int(
        payload.get("maximum_system_swap_used_bytes"),
        "maximum_system_swap_used_bytes",
    )
    if (
        sample_count < 1
        or reported_rss != int(summary["worker_max_resident_set_bytes"])
        or max(reported_rss, observed_rss)
        > float(execution["maximum_worker_resident_set_gib"]) * 1024**3
        or minimum_available
        < float(execution["minimum_system_available_memory_gib"]) * 1024**3
        or maximum_swap > float(execution["maximum_system_swap_used_gib"]) * 1024**3
    ):
        raise RuntimeError("Public-baseline execution exceeded resource limits")
    return payload


def active_conflicting_processes(
    protocol: Mapping[str, Any],
) -> list[dict[str, Any]]:
    patterns = tuple(protocol["execution"]["exclusive_process_patterns"])
    own = psutil.Process()
    excluded = {own.pid, *(parent.pid for parent in own.parents())}
    conflicts = []
    for process in psutil.process_iter(["pid", "cmdline"]):
        try:
            pid = int(process.info["pid"])
            command = " ".join(process.info.get("cmdline") or [])
        except (psutil.Error, TypeError, ValueError):
            continue
        if pid in excluded:
            continue
        matched = [pattern for pattern in patterns if pattern in command]
        if matched:
            conflicts.append({"pid": pid, "patterns": matched, "command": command})
    return conflicts


def write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    runner_lease.write_json_exclusive(path, payload)


def publish_directory_no_replace(attempt: Path, destination: Path) -> None:
    if attempt.parent != destination.parent:
        raise RuntimeError("Publication rename must remain in one directory")
    claim = destination.with_name(f".{destination.name}.publish-lock")
    try:
        claim.mkdir()
    except FileExistsError as error:
        raise RuntimeError(f"Publication lock already exists: {claim}") from error
    try:
        if destination.exists():
            raise FileExistsError(destination)
        os.rename(attempt, destination)
    finally:
        claim.rmdir()


def sample_child_resources(
    child: subprocess.Popen[Any],
    execution: Mapping[str, Any],
) -> tuple[str | None, dict[str, int | float] | None, bool]:
    if child.poll() is not None:
        return None, None, True
    violation, metrics = safe.resource_violation(
        child.pid, dict(execution)
    )
    exited_during_sample = child.poll() is not None
    if (
        exited_during_sample
        and violation
        and violation.startswith(RSS_SAMPLE_FAILURE_PREFIX)
    ):
        return None, None, True
    return violation, metrics, exited_during_sample


def run_unit(
    *,
    protocol_path: Path,
    protocol: dict[str, Any],
    protocol_sha256: str,
    model: str,
) -> dict[str, Any]:
    violation, initial = safe.resource_violation(os.getpid(), protocol["execution"])
    if violation is not None:
        raise RuntimeError(f"Refusing to start {model}: {violation}; {initial}")
    published_root = evaluator.run_root(protocol, model)
    if published_root.exists():
        raise FileExistsError(published_root)
    attempt_root = published_root.with_name(
        f".{published_root.name}.attempt-{os.getpid()}-{time.time_ns()}"
    )
    allowed = Path(protocol["execution"]["allowed_output_root"]).resolve()
    attempt_root = freeze.require_output_path(attempt_root, allowed_root=allowed)
    attempt_root.mkdir(parents=True)
    lease_path = attempt_root / "runner_lease.json"
    lease = runner_lease.create_lease(
        protocol_sha256=protocol_sha256,
        unit={"model": model},
        runner_script=Path(__file__).resolve(),
        evaluator_script=Path(evaluator.__file__).resolve(),
        attempt_root=attempt_root,
        published_root=published_root,
    )
    runner_lease.write_json_exclusive(lease_path, lease)
    command = [
        str(Path(protocol["execution"]["python_executable"]).resolve()),
        str(Path(evaluator.__file__).resolve()),
        "--protocol",
        str(protocol_path),
        "--expected-protocol-sha256",
        protocol_sha256,
        "--model",
        model,
        "--runner-lease",
        str(lease_path),
        "--attempt-output-root",
        str(attempt_root),
    ]
    child = subprocess.Popen(
        command,
        cwd=ROOT,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
        start_new_session=True,
    )
    peak_rss = 0
    minimum_available = sys.maxsize
    maximum_swap = 0
    sample_count = 0
    violation_message = None
    try:
        while violation_message is None:
            violation, metrics, exited = sample_child_resources(
                child, protocol["execution"]
            )
            if metrics is not None:
                sample_count += 1
                peak_rss = max(
                    peak_rss, int(metrics["tree_rss_bytes"])
                )
                minimum_available = min(
                    minimum_available,
                    int(metrics["system_available_bytes"]),
                )
                maximum_swap = max(
                    maximum_swap,
                    int(metrics["swap_used_bytes"]),
                )
            if violation is not None:
                violation_message = violation
                safe.terminate_process_group(child)
                break
            if exited:
                break
            time.sleep(
                safe.memory_guard_poll_seconds(protocol["execution"])
            )
    finally:
        if child.poll() is None:
            safe.terminate_process_group(child)
    if violation_message is not None:
        raise RuntimeError(
            f"Stopped {model}: {violation_message}; "
            f"peak_rss={peak_rss}, minimum_available={minimum_available}, "
            f"maximum_swap={maximum_swap}"
        )
    if child.returncode != 0:
        raise RuntimeError(f"Public baseline failed ({child.returncode}): {model}")
    summary_path = attempt_root / "summary.json"
    summary = verify_summary(
        summary_path,
        protocol=protocol,
        protocol_sha256=protocol_sha256,
        model=model,
        storage_root=attempt_root,
    )
    audit_path = attempt_root / "execution_audit.json"
    audit = {
        "audit_id": "cspp_v27_geonet_final_public_baseline_execution_v2",
        "protocol_sha256": protocol_sha256,
        "model": model,
        "command": command,
        "return_code": int(child.returncode),
        "reported_runtime_seconds": float(summary["runtime_seconds"]),
        "reported_worker_max_resident_set_bytes": int(
            summary["worker_max_resident_set_bytes"]
        ),
        "observed_peak_process_tree_rss_bytes": peak_rss,
        "resource_sample_count": sample_count,
        "minimum_system_available_bytes": minimum_available,
        "maximum_system_swap_used_bytes": maximum_swap,
        "runner_code_sha256": freeze.sha256_file(Path(__file__).resolve()),
        "evaluator_code_sha256": freeze.sha256_file(Path(evaluator.__file__).resolve()),
        "runner_lease": runner_lease.artifact_at(
            lease_path,
            published_root / "runner_lease.json",
        ),
        "summary": runner_lease.artifact_at(
            summary_path,
            published_root / "summary.json",
        ),
    }
    write_json_exclusive(audit_path, audit)
    verify_audit(
        audit_path,
        protocol=protocol,
        protocol_sha256=protocol_sha256,
        model=model,
        storage_root=attempt_root,
    )
    publish_directory_no_replace(attempt_root, published_root)
    verify_summary(
        published_root / "summary.json",
        protocol=protocol,
        protocol_sha256=protocol_sha256,
        model=model,
    )
    verify_audit(
        published_root / "execution_audit.json",
        protocol=protocol,
        protocol_sha256=protocol_sha256,
        model=model,
    )
    return {
        "model": model,
        "summary": freeze.artifact(published_root / "summary.json"),
        "execution_audit": freeze.artifact(published_root / "execution_audit.json"),
    }


def main() -> None:
    args = parse_args()
    if args.max_runs < 0:
        raise ValueError("--max-runs cannot be negative")
    protocol, protocol_sha256, protocol_path = evaluator.load_protocol(
        args.protocol,
        expected_sha256=args.expected_protocol_sha256,
    )
    if (
        Path(sys.executable).resolve()
        != Path(protocol["execution"]["python_executable"]).resolve()
    ):
        raise RuntimeError("Runner is not using the frozen Python executable")
    safe.memory_guard_poll_seconds(protocol["execution"])
    selected = (args.model,) if args.model else UNITS
    pending = []
    skipped = []
    for model in selected:
        root = evaluator.run_root(protocol, model)
        if root.is_dir():
            verify_summary(
                root / "summary.json",
                protocol=protocol,
                protocol_sha256=protocol_sha256,
                model=model,
            )
            verify_audit(
                root / "execution_audit.json",
                protocol=protocol,
                protocol_sha256=protocol_sha256,
                model=model,
            )
            skipped.append(model)
        else:
            pending.append(model)
    if args.max_runs:
        pending = pending[: args.max_runs]
    run_root = freeze.require_output_path(
        protocol["execution"]["run_root"],
        allowed_root=Path(protocol["execution"]["allowed_output_root"]),
    )
    run_root.mkdir(parents=True, exist_ok=True)
    lock_path = freeze.require_output_path(
        protocol["execution"]["global_lock_path"],
        allowed_root=Path(protocol["execution"]["allowed_output_root"]),
    )
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_handle = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(
            lock_handle.fileno(),
            fcntl.LOCK_EX | fcntl.LOCK_NB,
        )
    except BlockingIOError as error:
        lock_handle.close()
        raise RuntimeError(f"Heavy-evaluation lock is held: {lock_path}") from error
    completed = []
    started = time.time()
    try:
        conflicts = active_conflicting_processes(protocol)
        if conflicts:
            raise RuntimeError(
                "Another heavy V27 evaluation is active after lock "
                f"acquisition: {conflicts}"
            )
        for index, model in enumerate(pending, start=1):
            print(
                f"GeoNet final public baseline [{index}/{len(pending)}] {model}",
                flush=True,
            )
            completed.append(
                run_unit(
                    protocol_path=protocol_path,
                    protocol=protocol,
                    protocol_sha256=protocol_sha256,
                    model=model,
                )
            )
    finally:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        lock_handle.close()
    status = {
        "run_id": "cspp_v27_geonet_final_public_baseline_safe_run_v2",
        "protocol_sha256": protocol_sha256,
        "completed_this_run": completed,
        "skipped_verified": skipped,
        "runtime_seconds": time.time() - started,
    }
    status_path = run_root / f"safe_run_status_{time.time_ns()}.json"
    write_json_exclusive(status_path, status)
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
