#!/usr/bin/env python3
"""Run the two sealed V27C fixed retrospective datasets safely."""
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


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "evaluation"))

from scripts.evaluation import (  # noqa: E402
    eval_cspp_v27c_fixed_retrospective as evaluator,
)
from scripts.evaluation import (  # noqa: E402
    run_v27c_fixed_public_baselines as shared_guard,
)
from scripts.evaluation import v27_runner_lease as runner_lease  # noqa: E402
import run_v20c_geonet_score_cache_safe as safe  # noqa: E402


UNITS = ("knet", "instancegm")
RSS_SAMPLE_FAILURE_PREFIX = "worker process-tree RSS could not be sampled"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=evaluator.DEFAULT_PROTOCOL)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--dataset", choices=UNITS)
    return parser.parse_args()


def run_root(protocol: Mapping[str, Any], dataset: str) -> Path:
    allowed = Path(protocol["execution"]["allowed_output_root"]).resolve()
    return evaluator.require_output_path(
        Path(protocol["execution"]["score_output_root"]) / dataset,
        allowed,
    )


def verify_summary(
    path: Path,
    *,
    protocol: Mapping[str, Any],
    protocol_sha256: str,
    dataset: str,
    storage_root: Path | None = None,
) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"Retrospective summary is absent: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected_records = int(
        protocol["fixed_test_cohorts"][dataset]["records"]
    )
    expected = {
        "summary_id": "cspp_v27c_fixed_retrospective_score_v1",
        "protocol_sha256": protocol_sha256,
        "dataset": dataset,
        "model_key": evaluator.MODEL_KEY,
        "selected_arm": protocol["model"]["selected_arm"],
        "primary_seed": int(protocol["model"]["primary_seed"]),
        "records_requested": expected_records,
        "records_emitted": expected_records,
        "device": "cpu",
        "torch_threads": int(protocol["execution"]["torch_threads"]),
        "workers": 1,
        "output_fields": evaluator.OUTPUT_FIELDS,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(
                f"Retrospective summary differs: {dataset}:{key}"
            )
    published_root = run_root(protocol, dataset)
    records_spec = payload["records_artifact"]
    if Path(records_spec["path"]).resolve() != (
        published_root / "records.csv"
    ).resolve():
        raise RuntimeError("Retrospective records path differs")
    records = (storage_root or path.parent) / "records.csv"
    if (
        not records.is_file()
        or int(records_spec["bytes"]) != records.stat().st_size
        or str(records_spec["sha256"]) != evaluator.sha256_file(records)
    ):
        raise RuntimeError("Retrospective records artifact differs")
    with records.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != evaluator.OUTPUT_FIELDS:
            raise RuntimeError("Retrospective output schema differs")
        rows = list(reader)
    if len(rows) != expected_records:
        raise RuntimeError("Retrospective CSV row count differs")
    technical = sum(
        str(row["technical_failure"]) == "1" for row in rows
    )
    if technical != int(payload["technical_failures"]):
        raise RuntimeError("Retrospective technical-failure count differs")
    return payload


def audit_path(protocol: Mapping[str, Any], dataset: str) -> Path:
    return run_root(protocol, dataset) / "execution_audit.json"


def verify_audit(
    path: Path,
    *,
    protocol: Mapping[str, Any],
    protocol_sha256: str,
    dataset: str,
    storage_root: Path | None = None,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "audit_id": "cspp_v27c_fixed_retrospective_unit_execution_v1",
        "protocol_sha256": protocol_sha256,
        "dataset": dataset,
        "return_code": 0,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(f"Retrospective audit differs: {key}")
    published_root = run_root(protocol, dataset)
    summary_spec = payload["summary"]
    if Path(summary_spec["path"]).resolve() != (
        published_root / "summary.json"
    ).resolve():
        raise RuntimeError("Retrospective audit summary path differs")
    actual_root = storage_root or path.parent
    summary = actual_root / "summary.json"
    if (
        not summary.is_file()
        or int(summary_spec["bytes"]) != summary.stat().st_size
        or str(summary_spec["sha256"]) != evaluator.sha256_file(summary)
    ):
        raise RuntimeError("Retrospective audit summary artifact differs")
    lease_spec = payload.get("runner_lease", {})
    if Path(str(lease_spec.get("path", ""))).resolve() != (
        published_root / "runner_lease.json"
    ).resolve():
        raise RuntimeError("Retrospective runner lease path differs")
    lease_path = actual_root / "runner_lease.json"
    if (
        not lease_path.is_file()
        or int(lease_spec.get("bytes", -1)) != lease_path.stat().st_size
        or str(lease_spec.get("sha256", ""))
        != evaluator.sha256_file(lease_path)
    ):
        raise RuntimeError("Retrospective runner lease artifact differs")
    lease_payload = json.loads(lease_path.read_text(encoding="utf-8"))
    expected_runner_sha256 = evaluator.sha256_file(
        Path(__file__).resolve()
    )
    expected_evaluator_sha256 = evaluator.sha256_file(
        Path(evaluator.__file__).resolve()
    )
    if (
        lease_payload.get("lease_id") != runner_lease.LEASE_ID
        or lease_payload.get("protocol_sha256") != protocol_sha256
        or lease_payload.get("unit") != {"dataset": dataset}
        or lease_payload.get("runner_script_sha256")
        != expected_runner_sha256
        or lease_payload.get("evaluator_script_sha256")
        != expected_evaluator_sha256
        or payload.get("runner_code_sha256")
        != expected_runner_sha256
        or payload.get("evaluator_code_sha256")
        != expected_evaluator_sha256
    ):
        raise RuntimeError("Retrospective runner identity differs")
    command = payload.get("command")
    expected_command = [
        sys.executable,
        str(Path(evaluator.__file__).resolve()),
        "--protocol",
        str(command[3]) if isinstance(command, list) and len(command) > 3 else "",
        "--expected-protocol-sha256",
        protocol_sha256,
        "--dataset",
        dataset,
        "--runner-lease",
        str(Path(lease_payload["attempt_root"]) / "runner_lease.json"),
        "--attempt-output-root",
        str(lease_payload["attempt_root"]),
    ]
    if (
        command != expected_command
        or not Path(expected_command[3]).is_file()
        or evaluator.sha256_file(expected_command[3]) != protocol_sha256
    ):
        raise RuntimeError("Retrospective evaluator command differs")
    summary_payload = json.loads(summary.read_text(encoding="utf-8"))
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
        raise RuntimeError("Retrospective execution exceeded resource limits")
    return payload


def write_json_once(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


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
) -> dict[str, Any]:
    violation, initial = safe.resource_violation(
        os.getpid(), protocol["execution"]
    )
    if violation:
        raise RuntimeError(
            f"Refusing to start {dataset}: {violation}; metrics={initial}"
        )
    published_root = run_root(protocol, dataset)
    if published_root.exists():
        raise FileExistsError(published_root)
    attempt_root = evaluator.require_output_path(
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
        unit={"dataset": dataset},
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
    stopped = None
    try:
        while stopped is None:
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
            if violation:
                stopped = violation
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
    if stopped:
        raise RuntimeError(f"Stopped {dataset}: {stopped}")
    if int(child.returncode) != 0:
        raise RuntimeError(
            f"Retrospective unit failed ({child.returncode}): {dataset}"
        )
    summary_path = attempt_root / "summary.json"
    summary = verify_summary(
        summary_path,
        protocol=protocol,
        protocol_sha256=protocol_sha256,
        dataset=dataset,
        storage_root=attempt_root,
    )
    unit_audit = {
        "audit_id": "cspp_v27c_fixed_retrospective_unit_execution_v1",
        "protocol_sha256": protocol_sha256,
        "dataset": dataset,
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
        "runner_code_sha256": evaluator.sha256_file(
            Path(__file__).resolve()
        ),
        "evaluator_code_sha256": evaluator.sha256_file(
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
    unit_audit_path = attempt_root / "execution_audit.json"
    write_json_once(unit_audit_path, unit_audit)
    verify_audit(
        unit_audit_path,
        protocol=protocol,
        protocol_sha256=protocol_sha256,
        dataset=dataset,
        storage_root=attempt_root,
    )
    os.replace(attempt_root, published_root)
    verify_summary(
        published_root / "summary.json",
        protocol=protocol,
        protocol_sha256=protocol_sha256,
        dataset=dataset,
    )
    verify_audit(
        published_root / "execution_audit.json",
        protocol=protocol,
        protocol_sha256=protocol_sha256,
        dataset=dataset,
    )
    return {
        "dataset": dataset,
        "summary": evaluator.artifact(
            published_root / "summary.json"
        ),
        "execution_audit": evaluator.artifact(
            published_root / "execution_audit.json"
        ),
    }


def main() -> None:
    args = parse_args()
    protocol_path = args.protocol.resolve()
    protocol, protocol_sha256 = evaluator.load_protocol(
        protocol_path,
        expected_sha256=args.expected_protocol_sha256,
    )
    if not protocol["execution"].get("fresh_process_per_dataset"):
        raise RuntimeError("Retrospective fresh-process contract differs")
    selected = (args.dataset,) if args.dataset else UNITS
    pending = []
    skipped = []
    for dataset in selected:
        summary_path = run_root(protocol, dataset) / "summary.json"
        if summary_path.is_file():
            verify_summary(
                summary_path,
                protocol=protocol,
                protocol_sha256=protocol_sha256,
                dataset=dataset,
            )
            verify_audit(
                audit_path(protocol, dataset),
                protocol=protocol,
                protocol_sha256=protocol_sha256,
                dataset=dataset,
            )
            skipped.append(dataset)
        else:
            pending.append(dataset)

    lock_path = evaluator.require_output_path(
        protocol["execution"]["global_lock_path"],
        Path(protocol["execution"]["allowed_output_root"]),
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
        raise RuntimeError("Another heavy V27 evaluation holds the lock") from error
    completed = []
    try:
        conflicts = shared_guard.active_conflicting_processes(protocol)
        if conflicts:
            raise RuntimeError(
                "Another heavy V27 evaluation is active: "
                f"{conflicts}"
            )
        for dataset in pending:
            completed.append(
                run_unit(
                    protocol_path=protocol_path,
                    protocol=protocol,
                    protocol_sha256=protocol_sha256,
                    dataset=dataset,
                )
            )
    finally:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        lock_handle.close()
    print(
        json.dumps(
            {
                "run_id": "cspp_v27c_fixed_retrospective_safe_run_v1",
                "protocol_sha256": protocol_sha256,
                "completed": completed,
                "skipped_verified": skipped,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
