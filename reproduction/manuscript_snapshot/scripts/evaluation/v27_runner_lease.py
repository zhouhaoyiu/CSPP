#!/usr/bin/env python3
"""Create and verify parent-bound leases for formal V27 evaluators."""
from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import secrets
import sys
from typing import Any, Mapping

import psutil


LEASE_ID = "cspp_v27_formal_evaluator_runner_lease_v1"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact_at(
    actual_path: str | Path,
    published_path: str | Path,
) -> dict[str, Any]:
    actual = Path(actual_path).resolve()
    if not actual.is_file():
        raise FileNotFoundError(actual)
    return {
        "path": str(Path(published_path).resolve()),
        "bytes": int(actual.stat().st_size),
        "sha256": sha256_file(actual),
    }


def write_json_exclusive(
    path: str | Path,
    payload: Mapping[str, Any],
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        output,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o644,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    indent=2,
                    allow_nan=False,
                )
                + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        output.unlink(missing_ok=True)
        raise


def command_executes_runner(
    command: list[str],
    runner_script: str | Path,
) -> bool:
    if len(command) < 2:
        return False
    try:
        executable = Path(command[0]).resolve()
        entrypoint = Path(command[1]).resolve()
    except (OSError, RuntimeError):
        return False
    return (
        executable == Path(sys.executable).resolve()
        and entrypoint == Path(runner_script).resolve()
    )


def create_lease(
    *,
    protocol_sha256: str,
    unit: Mapping[str, str],
    runner_script: str | Path,
    evaluator_script: str | Path,
    attempt_root: str | Path,
    published_root: str | Path,
) -> dict[str, Any]:
    runner = Path(runner_script).resolve()
    evaluator = Path(evaluator_script).resolve()
    return {
        "lease_id": LEASE_ID,
        "nonce": secrets.token_hex(32),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "protocol_sha256": protocol_sha256,
        "unit": dict(unit),
        "runner_pid": os.getpid(),
        "runner_script": str(runner),
        "runner_script_sha256": sha256_file(runner),
        "evaluator_script": str(evaluator),
        "evaluator_script_sha256": sha256_file(evaluator),
        "attempt_root": str(Path(attempt_root).resolve()),
        "published_root": str(Path(published_root).resolve()),
    }


def verify_lease(
    path: str | Path,
    *,
    protocol_sha256: str,
    unit: Mapping[str, str],
    evaluator_script: str | Path,
    attempt_root: str | Path,
    published_root: str | Path,
) -> dict[str, Any]:
    lease_path = Path(path).resolve()
    attempt = Path(attempt_root).resolve()
    published = Path(published_root).resolve()
    if lease_path != attempt / "runner_lease.json":
        raise RuntimeError("Runner lease is outside the attempt directory")
    payload = json.loads(lease_path.read_text(encoding="utf-8"))
    evaluator = Path(evaluator_script).resolve()
    expected = {
        "lease_id": LEASE_ID,
        "protocol_sha256": protocol_sha256,
        "unit": dict(unit),
        "evaluator_script": str(evaluator),
        "evaluator_script_sha256": sha256_file(evaluator),
        "attempt_root": str(attempt),
        "published_root": str(published),
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(f"Runner lease differs: {key}")
    nonce = str(payload.get("nonce", ""))
    if len(nonce) != 64 or any(
        character not in "0123456789abcdef" for character in nonce
    ):
        raise RuntimeError("Runner lease nonce is invalid")
    runner_pid = int(payload.get("runner_pid", -1))
    if runner_pid != os.getppid():
        raise RuntimeError("Evaluator parent is not the leased runner")
    runner_script = Path(str(payload["runner_script"])).resolve()
    if (
        not runner_script.is_file()
        or sha256_file(runner_script)
        != str(payload.get("runner_script_sha256"))
    ):
        raise RuntimeError("Leased runner code identity differs")
    try:
        command = psutil.Process(runner_pid).cmdline()
    except psutil.Error as error:
        raise RuntimeError("Leased runner process is unavailable") from error
    if not command_executes_runner(command, runner_script):
        raise RuntimeError("Leased parent command is not the runner")
    return payload
