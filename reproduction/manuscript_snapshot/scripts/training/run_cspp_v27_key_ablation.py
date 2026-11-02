#!/usr/bin/env python3
"""Run the frozen CSPP V27 key ablations serially with swap guards."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL = (
    ROOT / "experiments/cspp_v27_key_ablation_protocol_20260730.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_artifact(spec: dict[str, Any]) -> None:
    path = Path(spec["path"])
    if (
        not path.is_file()
        or path.stat().st_size != int(spec["bytes"])
        or sha256(path) != spec["sha256"]
    ):
        raise RuntimeError(f"Frozen artifact drifted: {path}")


def swap_used_gib() -> float:
    output = subprocess.check_output(
        ["/usr/sbin/sysctl", "vm.swapusage"],
        text=True,
    )
    match = re.search(r"used = ([0-9.]+)([MG])", output)
    if match is None:
        raise RuntimeError(f"Cannot parse swap usage: {output.strip()}")
    value = float(match.group(1))
    return value / 1024.0 if match.group(2) == "M" else value


def final_path(command: list[str]) -> Path:
    index = command.index("--save-dir")
    return Path(command[index + 1]) / "final.pt"


def verify_checkpoint(path: Path, expected_epoch: int) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if int(checkpoint["epoch"]) != expected_epoch:
        raise RuntimeError(f"Unexpected final epoch: {path}")
    if checkpoint.get("development_evaluation") not in {None, "disabled"}:
        raise RuntimeError(f"Unexpected development result in {path}")
    training = checkpoint["training_config"]
    if not bool(training["no_development_evaluation"]):
        raise RuntimeError(f"Development evaluation was enabled: {path}")
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "model_class": checkpoint["model_class"],
        "model_config": checkpoint["model_config"],
        "epoch": int(checkpoint["epoch"]),
        "parameter_count": int(checkpoint["parameter_count"]),
        "training_seed": int(training["seed"]),
        "training_packet_localization_loss_weight": float(
            training["packet_localization_loss_weight"]
        ),
    }


def command_float(command: list[str], name: str) -> float:
    index = command.index(name)
    return float(command[index + 1])


def verify_run_identity(
    result: dict[str, Any],
    *,
    run: dict[str, Any],
    command: list[str],
    expected_switches: dict[str, Any],
    stage: str,
) -> None:
    if int(result["training_seed"]) != int(run["seed"]):
        raise RuntimeError(
            f"Checkpoint seed differs: {run['run_id']} {stage}"
        )
    expected_weight = command_float(
        command,
        "--packet-localization-loss-weight",
    )
    if (
        abs(
            float(result["training_packet_localization_loss_weight"])
            - expected_weight
        )
        > 1e-12
    ):
        raise RuntimeError(
            f"Localization-loss weight differs: {run['run_id']} {stage}"
        )
    for key, expected in expected_switches.items():
        if result["model_config"].get(key) != expected:
            raise RuntimeError(
                f"Model switch {key} differs: {run['run_id']} {stage}"
            )


def main() -> None:
    args = parse_args()
    protocol_path = args.protocol.resolve()
    protocol = read_json(protocol_path)
    if protocol.get("protocol_id") != "cspp_v27_key_ablation_v2":
        raise RuntimeError("Unexpected protocol")
    for spec in protocol["code_artifacts"].values():
        verify_artifact(spec)
    maximum_swap = float(
        protocol["resource_policy"]["maximum_swap_used_gib_before_launch"]
    )
    arm_by_id = {
        arm["id"]: arm for arm in protocol["design"]["arms"]
    }
    completed = []
    for run in protocol["design"]["runs"]:
        arm = arm_by_id[run["arm_id"]]
        flags = arm["flags"]
        expected_switches = {
            "temporal_mode": (
                flags[flags.index("--v27-temporal-mode") + 1]
                if "--v27-temporal-mode" in flags
                else "gru"
            ),
            "dense_phase_evidence": (
                "--v27-disable-dense-phase-evidence" not in flags
            ),
            "spectral_features": (
                "--v27-disable-spectral-features" not in flags
            ),
        }
        arm_result: dict[str, Any] = {
            "run_id": run["run_id"],
            "seed": int(run["seed"]),
            "arm_id": run["arm_id"],
        }
        for stage in ("candidate", "verifier"):
            command = run[f"{stage}_command"]
            output = final_path(command)
            if output.exists():
                arm_result[stage] = verify_checkpoint(output, expected_epoch=3)
            else:
                current_swap = swap_used_gib()
                if current_swap > maximum_swap:
                    raise RuntimeError(
                        f"Swap guard blocked {run['run_id']} {stage}: "
                        f"{current_swap:.3f} GiB > {maximum_swap:.3f} GiB"
                    )
                subprocess.run(command, cwd=ROOT, check=True)
                arm_result[stage] = verify_checkpoint(
                    output,
                    expected_epoch=3,
                )
            verify_run_identity(
                arm_result[stage],
                run=run,
                command=command,
                expected_switches=expected_switches,
                stage=stage,
            )
        completed.append(arm_result)

    result = {
        "result_id": "cspp_v27_key_ablation_training_result_v2",
        "protocol": str(protocol_path),
        "protocol_sha256": sha256(protocol_path),
        "status": "all_fixed_epoch_models_complete",
        "arms": completed,
        "evaluation_status": "pending_frozen_cross_network_and_posthoc_geonet_scoring",
    }
    output_path = (
        Path(protocol["resource_policy"]["output_root"])
        / "training_result.json"
    )
    if output_path.exists():
        raise FileExistsError(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    print(f"result={output_path}")
    print(f"result_sha256={sha256(output_path)}")


if __name__ == "__main__":
    sys.exit(main())
