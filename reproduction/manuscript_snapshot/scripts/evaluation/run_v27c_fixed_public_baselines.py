#!/usr/bin/env python3
"""Run sealed fixed-record public baselines serially with memory guards."""
from __future__ import annotations

import argparse
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
    eval_v27c_fixed_public_baselines as evaluator,
)
from scripts.evaluation import (  # noqa: E402
    freeze_v27c_fixed_public_baseline_protocol as freeze,
)
from scripts.evaluation import v27_runner_lease as runner_lease  # noqa: E402
import run_v20c_geonet_score_cache_safe as safe  # noqa: E402


UNITS = (
    ("knet", "phasenet_instance"),
    ("knet", "eqtransformer_original"),
    ("instancegm", "phasenet_instance"),
    ("instancegm", "eqtransformer_original"),
)
RSS_SAMPLE_FAILURE_PREFIX = "worker process-tree RSS could not be sampled"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol", type=Path, default=evaluator.DEFAULT_PROTOCOL
    )
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument(
        "--dataset", choices=("knet", "instancegm")
    )
    parser.add_argument(
        "--model",
        choices=("phasenet_instance", "eqtransformer_original"),
    )
    parser.add_argument("--max-runs", type=int, default=0)
    return parser.parse_args()


def run_root(
    protocol: Mapping[str, Any],
    dataset: str,
    model: str,
) -> Path:
    return evaluator.run_root(protocol, dataset, model)


def verify_summary(
    path: Path,
    *,
    protocol: Mapping[str, Any],
    protocol_sha256: str,
    dataset: str,
    model: str,
    storage_root: Path | None = None,
) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"Public-baseline summary is absent: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "summary_id": "cspp_v27c_fixed_public_baseline_run_v1",
        "protocol_sha256": protocol_sha256,
        "retrospective_protocol_sha256": protocol[
            "retrospective_protocol_sha256"
        ],
        "dataset": dataset,
        "model_key": model,
        "methods": sorted(freeze.EXPECTED_METHODS[model]),
        "records_requested": int(
            protocol["fixed_test_cohorts"][dataset]["records"]
        ),
        "waveform_identity_verified_records": int(
            protocol["fixed_test_cohorts"][dataset]["records"]
        ),
        "comparison_mode": "released_full_record_annotation",
        "availability_comparable": False,
        "device": protocol["execution"]["device"],
        "torch_threads": int(protocol["execution"]["torch_threads"]),
        "software_versions": protocol["software_versions"],
        "loaded_state_dict_sha256": protocol["public_models"][model][
            "loaded_state_dict_sha256"
        ],
        "resolved_weight_artifact": protocol["public_models"][model][
            "artifacts"
        ]["weights"],
        "resolved_metadata_artifact": protocol["public_models"][model][
            "artifacts"
        ]["metadata"],
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(
                f"Public-baseline summary differs for {key}: {path}"
            )
    expected_rows = expected["records_requested"] * len(
        freeze.EXPECTED_METHODS[model]
    )
    if int(payload.get("output_rows", -1)) != expected_rows:
        raise RuntimeError("Public-baseline output-row count differs")
    published_root = run_root(protocol, dataset, model)
    records_spec = payload["records_artifact"]
    if Path(records_spec["path"]).resolve() != (
        published_root / "records.csv"
    ).resolve():
        raise RuntimeError("Public-baseline records path differs")
    records_path = (storage_root or path.parent) / "records.csv"
    if (
        not records_path.is_file()
        or int(records_spec["bytes"]) != records_path.stat().st_size
        or str(records_spec["sha256"])
        != evaluator.retrospective.sha256_file(records_path)
    ):
        raise RuntimeError("Public-baseline records artifact differs")
    with records_path.open(encoding="utf-8", newline="") as handle:
        reader = evaluator.csv.DictReader(handle)
        if reader.fieldnames != evaluator.OUTPUT_FIELDS:
            raise RuntimeError("Public-baseline output schema differs")
        rows = list(reader)
    if len(rows) != expected_rows:
        raise RuntimeError("Public-baseline CSV row count differs")
    observed_methods = {
        str(row["method_id"]) for row in rows
    }
    if observed_methods != freeze.EXPECTED_METHODS[model]:
        raise RuntimeError("Public-baseline CSV method set differs")
    technical_records = {
        (row["source_index"], row["trace_name"], row["event_id"])
        for row in rows
        if row["technical_error"]
    }
    if len(technical_records) != int(
        payload["technical_failure_records"]
    ):
        raise RuntimeError("Public-baseline technical-failure count differs")
    maximum_rss = int(
        float(protocol["execution"]["maximum_worker_resident_set_gib"])
        * 1024**3
    )
    if int(payload["worker_max_resident_set_bytes"]) > maximum_rss:
        raise RuntimeError("Completed public baseline exceeded RSS limit")
    return payload


def unit_audit_path(
    protocol: Mapping[str, Any],
    dataset: str,
    model: str,
) -> Path:
    return run_root(protocol, dataset, model) / "execution_audit.json"


def verify_unit_audit(
    path: Path,
    *,
    protocol: Mapping[str, Any],
    protocol_sha256: str,
    dataset: str,
    model: str,
    storage_root: Path | None = None,
) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"Public-baseline execution audit is absent: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "audit_id": "cspp_v27c_fixed_public_baseline_unit_execution_v1",
        "protocol_sha256": protocol_sha256,
        "dataset": dataset,
        "model": model,
        "return_code": 0,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(
                f"Public-baseline execution audit differs: {key}"
            )
    if int(payload.get("observed_peak_process_tree_rss_bytes", -1)) < 0:
        raise RuntimeError("Public-baseline execution RSS audit is invalid")
    published_root = run_root(protocol, dataset, model)
    summary_spec = payload["summary"]
    if Path(summary_spec["path"]).resolve() != (
        published_root / "summary.json"
    ).resolve():
        raise RuntimeError("Public-baseline audit summary path differs")
    actual_root = storage_root or path.parent
    summary_path = actual_root / "summary.json"
    if (
        not summary_path.is_file()
        or int(summary_spec["bytes"]) != summary_path.stat().st_size
        or str(summary_spec["sha256"])
        != evaluator.retrospective.sha256_file(summary_path)
    ):
        raise RuntimeError("Public-baseline audit summary artifact differs")
    lease_spec = payload.get("runner_lease", {})
    if Path(str(lease_spec.get("path", ""))).resolve() != (
        published_root / "runner_lease.json"
    ).resolve():
        raise RuntimeError("Public-baseline runner lease path differs")
    lease_path = actual_root / "runner_lease.json"
    if (
        not lease_path.is_file()
        or int(lease_spec.get("bytes", -1)) != lease_path.stat().st_size
        or str(lease_spec.get("sha256", ""))
        != evaluator.retrospective.sha256_file(lease_path)
    ):
        raise RuntimeError("Public-baseline runner lease artifact differs")
    lease_payload = json.loads(lease_path.read_text(encoding="utf-8"))
    expected_runner_sha256 = evaluator.retrospective.sha256_file(
        Path(__file__).resolve()
    )
    expected_evaluator_sha256 = evaluator.retrospective.sha256_file(
        Path(evaluator.__file__).resolve()
    )
    if (
        lease_payload.get("lease_id") != runner_lease.LEASE_ID
        or lease_payload.get("protocol_sha256") != protocol_sha256
        or lease_payload.get("unit")
        != {"dataset": dataset, "model": model}
        or lease_payload.get("runner_script_sha256")
        != expected_runner_sha256
        or lease_payload.get("evaluator_script_sha256")
        != expected_evaluator_sha256
        or payload.get("runner_code_sha256")
        != expected_runner_sha256
        or payload.get("evaluator_code_sha256")
        != expected_evaluator_sha256
    ):
        raise RuntimeError("Public-baseline runner identity differs")
    command = payload.get("command")
    expected_command = [
        sys.executable,
        str(Path(evaluator.__file__).resolve()),
        "--protocol",
        (
            str(command[3])
            if isinstance(command, list) and len(command) > 3
            else ""
        ),
        "--expected-protocol-sha256",
        protocol_sha256,
        "--dataset",
        dataset,
        "--model",
        model,
        "--runner-lease",
        str(Path(lease_payload["attempt_root"]) / "runner_lease.json"),
        "--attempt-output-root",
        str(lease_payload["attempt_root"]),
    ]
    if (
        command != expected_command
        or not Path(expected_command[3]).is_file()
        or evaluator.retrospective.sha256_file(expected_command[3])
        != protocol_sha256
    ):
        raise RuntimeError("Public-baseline evaluator command differs")
    summary_payload = json.loads(
        summary_path.read_text(encoding="utf-8")
    )
    execution = protocol["execution"]
    reported_rss = int(payload.get(
        "reported_worker_max_resident_set_bytes", -1
    ))
    if (
        int(payload.get("resource_sample_count", 0)) < 1
        or int(payload.get("observed_peak_process_tree_rss_bytes", -1)) < 0
        or reported_rss
        != int(summary_payload["worker_max_resident_set_bytes"])
        or reported_rss
        > float(execution["maximum_worker_resident_set_gib"]) * 1024**3
        or int(payload["observed_peak_process_tree_rss_bytes"])
        > float(execution["maximum_worker_resident_set_gib"]) * 1024**3
        or int(payload["minimum_system_available_bytes"])
        < float(execution["minimum_system_available_memory_gib"]) * 1024**3
        or int(payload["maximum_system_swap_used_bytes"])
        > float(execution["maximum_system_swap_used_gib"]) * 1024**3
    ):
        raise RuntimeError("Public-baseline execution exceeded resource limits")
    return payload


def active_conflicting_processes(
    protocol: Mapping[str, Any],
) -> list[dict[str, Any]]:
    patterns = tuple(
        str(value)
        for value in protocol["execution"]["exclusive_process_patterns"]
    )
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
        matched = [pattern for pattern in patterns if pattern in command]
        if matched:
            conflicts.append(
                {"pid": pid, "patterns": matched, "command": command}
            )
    return conflicts


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
    dataset: str,
    model: str,
) -> dict[str, Any]:
    violation, initial = safe.resource_violation(
        os.getpid(), protocol["execution"]
    )
    if violation is not None:
        raise RuntimeError(
            f"Refusing to start {dataset}/{model}: {violation}; "
            f"metrics={initial}"
        )
    published_root = run_root(protocol, dataset, model)
    if published_root.exists():
        raise FileExistsError(published_root)
    attempt_root = evaluator.retrospective.require_output_path(
        published_root.with_name(
            f".{published_root.name}.attempt-{os.getpid()}"
        ),
        Path(protocol["execution"]["allowed_output_root"]),
    )
    if attempt_root.exists():
        raise FileExistsError(attempt_root)
    attempt_root.mkdir(parents=True)
    lease_path = attempt_root / "runner_lease.json"
    lease_payload = runner_lease.create_lease(
        protocol_sha256=protocol_sha256,
        unit={"dataset": dataset, "model": model},
        runner_script=Path(__file__).resolve(),
        evaluator_script=Path(evaluator.__file__).resolve(),
        attempt_root=attempt_root,
        published_root=published_root,
    )
    runner_lease.write_json_exclusive(lease_path, lease_payload)
    command = [
        sys.executable,
        str(Path(evaluator.__file__).resolve()),
        "--protocol",
        str(protocol_path),
        "--expected-protocol-sha256",
        protocol_sha256,
        "--dataset",
        dataset,
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
                    maximum_swap, int(metrics["swap_used_bytes"])
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
            f"Stopped {dataset}/{model}: {violation_message}; "
            f"peak_rss={peak_rss}, minimum_available={minimum_available}, "
            f"maximum_swap={maximum_swap}"
        )
    if int(child.returncode) != 0:
        raise RuntimeError(
            f"Public baseline failed ({child.returncode}): {dataset}/{model}"
        )
    summary_path = attempt_root / "summary.json"
    payload = verify_summary(
        summary_path,
        protocol=protocol,
        protocol_sha256=protocol_sha256,
        dataset=dataset,
        model=model,
        storage_root=attempt_root,
    )
    audit_path = attempt_root / "execution_audit.json"
    audit_payload = {
        "audit_id": "cspp_v27c_fixed_public_baseline_unit_execution_v1",
        "protocol_sha256": protocol_sha256,
        "dataset": dataset,
        "model": model,
        "command": command,
        "return_code": int(child.returncode),
        "reported_runtime_seconds": float(payload["runtime_seconds"]),
        "reported_worker_max_resident_set_bytes": int(
            payload["worker_max_resident_set_bytes"]
        ),
        "observed_peak_process_tree_rss_bytes": peak_rss,
        "resource_sample_count": sample_count,
        "minimum_system_available_bytes": minimum_available,
        "maximum_system_swap_used_bytes": maximum_swap,
        "runner_code_sha256": evaluator.retrospective.sha256_file(
            Path(__file__).resolve()
        ),
        "evaluator_code_sha256": evaluator.retrospective.sha256_file(
            Path(evaluator.__file__).resolve()
        ),
        "runner_lease": runner_lease.artifact_at(
            lease_path,
            published_root / "runner_lease.json",
        ),
        "summary": runner_lease.artifact_at(
            summary_path,
            published_root / "summary.json",
        ),
    }
    write_json_atomic(audit_path, audit_payload)
    verify_unit_audit(
        audit_path,
        protocol=protocol,
        protocol_sha256=protocol_sha256,
        dataset=dataset,
        model=model,
        storage_root=attempt_root,
    )
    os.replace(attempt_root, published_root)
    verify_summary(
        published_root / "summary.json",
        protocol=protocol,
        protocol_sha256=protocol_sha256,
        dataset=dataset,
        model=model,
    )
    verify_unit_audit(
        published_root / "execution_audit.json",
        protocol=protocol,
        protocol_sha256=protocol_sha256,
        dataset=dataset,
        model=model,
    )
    return {
        "dataset": dataset,
        "model": model,
        "summary": evaluator.retrospective.artifact(summary_path),
        "reported_runtime_seconds": float(payload["runtime_seconds"]),
        "reported_worker_max_resident_set_bytes": int(
            payload["worker_max_resident_set_bytes"]
        ),
        "observed_peak_process_tree_rss_bytes": peak_rss,
        "minimum_system_available_bytes": minimum_available,
        "maximum_system_swap_used_bytes": maximum_swap,
        "return_code": int(child.returncode),
        "execution_audit": evaluator.retrospective.artifact(
            published_root / "execution_audit.json"
        ),
    }


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    if bool(args.dataset) != bool(args.model):
        raise ValueError("--dataset and --model must be supplied together")
    if args.max_runs < 0:
        raise ValueError("--max-runs cannot be negative")
    protocol_path = args.protocol.resolve()
    protocol, protocol_sha256 = evaluator.load_protocol(
        protocol_path,
        expected_sha256=args.expected_protocol_sha256,
    )
    safe.memory_guard_poll_seconds(protocol["execution"])

    selected = (
        ((args.dataset, args.model),)
        if args.dataset
        else UNITS
    )
    pending: list[tuple[str, str]] = []
    skipped = []
    for dataset, model in selected:
        summary_path = run_root(
            protocol, dataset, model
        ) / "summary.json"
        if summary_path.is_file():
            verify_summary(
                summary_path,
                protocol=protocol,
                protocol_sha256=protocol_sha256,
                dataset=dataset,
                model=model,
            )
            verify_unit_audit(
                unit_audit_path(protocol, dataset, model),
                protocol=protocol,
                protocol_sha256=protocol_sha256,
                dataset=dataset,
                model=model,
            )
            skipped.append(f"{dataset}/{model}")
        else:
            pending.append((dataset, model))
    if args.max_runs:
        pending = pending[: args.max_runs]

    output_root = freeze.require_external_output(
        protocol["execution"]["output_root"]
    )
    output_root.mkdir(parents=True, exist_ok=True)
    lock_path = freeze.require_external_output(
        protocol["execution"]["global_lock_path"]
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
        raise RuntimeError(
            f"Another public-baseline scorer holds {lock_path}"
        ) from error

    completed = []
    started = time.time()
    try:
        conflicts = active_conflicting_processes(protocol)
        if conflicts:
            raise RuntimeError(
                "Another heavy V27 evaluation is active; refusing "
                f"concurrent public-baseline scoring: {conflicts}"
            )
        for index, (dataset, model) in enumerate(pending, start=1):
            print(
                f"V27C public baseline [{index}/{len(pending)}] "
                f"{dataset}/{model}",
                flush=True,
            )
            completed.append(
                run_unit(
                    protocol_path=protocol_path,
                    protocol=protocol,
                    protocol_sha256=protocol_sha256,
                    dataset=dataset,
                    model=model,
                )
            )
    finally:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        lock_handle.close()

    status = {
        "run_id": "cspp_v27c_fixed_public_baseline_safe_run_v1",
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_sha256,
        "completed_this_run": completed,
        "skipped_verified": skipped,
        "runtime_seconds": time.time() - started,
    }
    write_json_atomic(output_root / "safe_run_status.json", status)
    print(json.dumps(status, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
