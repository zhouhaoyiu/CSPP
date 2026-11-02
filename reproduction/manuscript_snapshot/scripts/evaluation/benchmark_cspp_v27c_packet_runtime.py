#!/usr/bin/env python3
"""Benchmark the sealed V27C station packet path on one CPU thread."""
from __future__ import annotations

import argparse
from collections import deque
from datetime import UTC, datetime
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Mapping

import numpy as np
import psutil
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.evaluation import (  # noqa: E402
    freeze_cspp_v27c_fixed_retrospective_protocol as retrospective,
)
from scripts.evaluation import (  # noqa: E402
    freeze_cspp_v27c_packet_runtime_protocol as freeze,
)
from data_streaming import normalize_packet_ewma_causal  # noqa: E402
from cspp_v27_packet_candidate import (  # noqa: E402
    PacketCandidateState,
)
from model_cspp_v27 import CSPPCausalPickerV27C  # noqa: E402
from stream_quality_v5 import (  # noqa: E402
    PacketQualityConfig,
    PacketQualityGateV5,
)


DEFAULT_PROTOCOL = freeze.DEFAULT_OUTPUT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--expected-protocol-sha256", required=True)
    return parser.parse_args()


def cpu_name() -> str:
    run = subprocess.run(
        ["sysctl", "-n", "machdep.cpu.brand_string"],
        capture_output=True,
        text=True,
        check=False,
    )
    return (
        run.stdout.strip()
        or platform.processor()
        or platform.machine()
    )


def payload_bytes(value: Any, seen: set[int] | None = None) -> int:
    if seen is None:
        seen = set()
    identity = id(value)
    if identity in seen:
        return 0
    seen.add(identity)
    if isinstance(value, torch.Tensor):
        return int(value.numel() * value.element_size())
    if isinstance(value, np.ndarray):
        return int(value.nbytes)
    if isinstance(value, Mapping):
        return sum(payload_bytes(item, seen) for item in value.values())
    if isinstance(value, (tuple, list, set, deque)):
        return sum(payload_bytes(item, seen) for item in value)
    if hasattr(value, "__dict__"):
        return payload_bytes(vars(value), seen)
    if isinstance(value, (float, int, bool, np.generic)):
        return 8
    return 0


def packet_suite(seed: int, count: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    packets = rng.normal(
        0.0, 0.03, size=(count, 3, 50)
    ).astype(np.float32)
    samples = np.arange(50, dtype=np.float32) / 100.0
    for start in range(80, count, 257):
        for index in range(start, min(start + 16, count)):
            age = index - start
            envelope = np.float32(math.exp(-age / 7.0))
            packets[index, 0] += (
                0.50
                * envelope
                * np.sin(2.0 * np.pi * 7.0 * samples)
            )
            packets[index, 1] += (
                0.35
                * envelope
                * np.sin(2.0 * np.pi * 5.0 * samples + 0.4)
            )
            packets[index, 2] += (
                0.30
                * envelope
                * np.sin(2.0 * np.pi * 9.0 * samples + 0.9)
            )
    return packets


def load_protocol(
    path: Path,
    *,
    expected_sha256: str,
) -> tuple[dict[str, Any], str]:
    protocol_path = retrospective.reject_forbidden_path(path)
    actual = retrospective.sha256_file(protocol_path)
    if actual != expected_sha256.lower():
        raise RuntimeError("Runtime protocol SHA-256 differs")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("protocol_id") != freeze.PROTOCOL_ID:
        raise RuntimeError("Unexpected runtime protocol")
    measurement = protocol["measurement"]
    if (
        measurement["device"] != "cpu"
        or int(measurement["torch_intraop_threads"]) != 1
        or int(measurement["torch_interop_threads"]) != 1
        or int(measurement["packet_samples"]) != 50
        or not math.isclose(
            float(measurement["packet_duration_seconds"]),
            0.5,
            abs_tol=1e-12,
        )
    ):
        raise RuntimeError("Runtime measurement contract differs")
    result_path = retrospective.require_output_path(
        protocol["execution"]["result_path"],
        allowed_root=freeze.WORK_ROOT,
    )
    if result_path.exists():
        raise FileExistsError(result_path)
    for relative, expected in protocol["code_sha256"].items():
        source = ROOT / relative
        if (
            not source.is_file()
            or retrospective.sha256_file(source) != str(expected)
        ):
            raise RuntimeError(f"Frozen runtime code drifted: {relative}")
    retrospective.verify_artifact(
        protocol["selection_freeze"],
        require_sha256=True,
        require_mtime=False,
    )
    retrospective.verify_artifact(
        protocol["model"]["checkpoint"],
        require_sha256=True,
        require_mtime=False,
    )
    return protocol, actual


def timing_summary(values: list[float], packet_seconds: float) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array) or not np.isfinite(array).all():
        raise RuntimeError("Runtime timing samples are invalid")
    p95 = float(np.quantile(array, 0.95, method="linear"))
    return {
        "timed_packets": int(len(array)),
        "mean_ms": float(array.mean()),
        "median_ms": float(np.median(array)),
        "p95_ms": p95,
        "p99_ms": float(np.quantile(array, 0.99, method="linear")),
        "maximum_ms": float(array.max()),
        "real_time_factor_at_p95": float(
            packet_seconds * 1000.0 / p95
        ),
    }


def model_from_protocol(
    protocol: Mapping[str, Any],
) -> tuple[CSPPCausalPickerV27C, dict[str, Any]]:
    checkpoint_path = retrospective.verify_artifact(
        protocol["model"]["checkpoint"],
        require_sha256=True,
        require_mtime=False,
    )
    stored = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    model = CSPPCausalPickerV27C.from_config(stored["model_config"])
    model.load_state_dict(stored["model"], strict=True)
    if (
        stored.get("model_class") != "CSPPCausalPickerV27C"
        or int(stored.get("epoch", -1)) != 3
        or int(stored.get("training_config", {}).get("seed", -1))
        != int(protocol["model"]["primary_seed"])
        or str(model.config.verifier_feature_mode)
        != str(protocol["model"]["verifier_feature_mode"])
    ):
        raise RuntimeError("Runtime checkpoint metadata differs")
    return model.eval(), stored


@torch.inference_mode()
def benchmark(
    protocol: Mapping[str, Any],
    model: CSPPCausalPickerV27C,
) -> tuple[dict[str, Any], dict[str, Any]]:
    measurement = protocol["measurement"]
    warmup = int(measurement["timing_warmup_packets"])
    timed = int(measurement["timed_packets"])
    packet_seconds = float(measurement["packet_duration_seconds"])
    packets = packet_suite(
        int(measurement["synthetic_seed"]), warmup + timed
    )
    gate = PacketQualityGateV5(
        PacketQualityConfig(
            warmup_packets=int(
                measurement["quality_warmup_packets"]
            )
        )
    )
    normalization_state = None
    model_state = None
    candidate_state = PacketCandidateState(
        history_packets=int(measurement["candidate_history_packets"])
    )
    timings = []
    candidate_count = 0
    quality_allowed = 0
    maximum_state_bytes = 0
    last_probabilities = None
    candidate_threshold = float(
        protocol["operating_point"]["candidate_threshold"]
    )
    event_threshold = float(
        protocol["operating_point"]["event_threshold"]
    )
    for index, packet in enumerate(packets):
        started = time.perf_counter_ns()
        decision = gate.evaluate(packet, index * packet_seconds)
        if decision.reset_required:
            normalization_state = None
            model_state = None
            candidate_state.reset()
            probabilities = None
        else:
            normalized, normalization_state = (
                normalize_packet_ewma_causal(
                    packet, normalization_state
                )
            )
            tensor = torch.from_numpy(normalized)
            candidate, event, delay, model_state = (
                model.forward_streaming_packet(tensor, model_state)
            )
            probabilities = (
                float(torch.sigmoid(candidate)),
                float(torch.sigmoid(event)),
                float(torch.sigmoid(delay)),
            )
            allowed = bool(decision.candidate_allowed)
            quality_allowed += int(allowed)
            emitted = candidate_state.step(
                packet_index=index,
                valid=True,
                candidate_allowed=allowed,
                candidate_probability=probabilities[0],
                event_probability=probabilities[1],
                arrival_age_fraction=probabilities[2],
                candidate_threshold=candidate_threshold,
                event_threshold=event_threshold,
                start_epoch=0.0,
                packet_seconds=packet_seconds,
                max_arrival_age_packets=float(
                    measurement["max_arrival_age_packets"]
                ),
            )
            candidate_count += int(emitted is not None)
        elapsed_ms = (time.perf_counter_ns() - started) / 1e6
        if index >= warmup:
            timings.append(elapsed_ms)
        maximum_state_bytes = max(
            maximum_state_bytes,
            payload_bytes(
                {
                    "normalization": normalization_state,
                    "model": model_state,
                    "quality": gate,
                    "candidate": candidate_state,
                }
            ),
        )
        last_probabilities = probabilities
    if len(timings) != timed:
        raise RuntimeError("Timed packet count differs")

    reset_gate = PacketQualityGateV5(
        PacketQualityConfig(warmup_packets=20)
    )
    reset_gate.evaluate(packets[0], 0.0)
    invalid = packets[1].copy()
    invalid[0, 0] = np.nan
    reset_decision = reset_gate.evaluate(invalid, packet_seconds)
    if not reset_decision.reset_required:
        raise RuntimeError("Quality reset smoke test failed")
    return (
        timing_summary(timings, packet_seconds),
        {
            "quality_allowed_packets_including_timing_warmup": (
                quality_allowed
            ),
            "emitted_candidates_including_timing_warmup": candidate_count,
            "maximum_bounded_station_state_bytes": maximum_state_bytes,
            "last_probabilities": last_probabilities,
            "invalid_packet_reset_verified": True,
        },
    )


def write_json_once(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=False)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    protocol, protocol_sha256 = load_protocol(
        args.protocol,
        expected_sha256=args.expected_protocol_sha256,
    )
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    process = psutil.Process()
    rss_before = int(process.memory_info().rss)
    load_started = time.perf_counter_ns()
    model, _ = model_from_protocol(protocol)
    load_ms = (time.perf_counter_ns() - load_started) / 1e6
    rss_after_load = int(process.memory_info().rss)
    timings, diagnostics = benchmark(protocol, model)
    rss_after_benchmark = int(process.memory_info().rss)

    parameters = sum(value.numel() for value in model.parameters())
    model_bytes = sum(
        value.numel() * value.element_size()
        for value in model.state_dict().values()
    )
    gates = protocol["acceptance_gates"]
    acceptance = {
        "latency_pass": timings["p95_ms"]
        <= float(gates["p95_end_to_end_packet_ms_max"]),
        "parameter_count_pass": parameters
        <= int(gates["model_parameter_count_max"]),
        "model_tensor_bytes_pass": model_bytes
        <= int(gates["model_tensor_bytes_max"]),
        "bounded_station_state_pass": diagnostics[
            "maximum_bounded_station_state_bytes"
        ]
        <= int(gates["bounded_station_state_bytes_max"]),
        "real_time_factor_pass": timings["real_time_factor_at_p95"]
        >= float(gates["real_time_factor_at_p95_min"]),
    }
    acceptance["all_runtime_gates_pass"] = all(acceptance.values())
    result = {
        "result_id": "cspp_v27c_packet_runtime_result_v1",
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "protocol": retrospective.artifact(args.protocol),
        "protocol_sha256": protocol_sha256,
        "evidence_status": "engineering_runtime_not_seismic_accuracy",
        "environment": {
            "cpu": cpu_name(),
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "torch": torch.__version__,
            "torch_intraop_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
        },
        "model": {
            **protocol["model"],
            "parameter_count": int(parameters),
            "tensor_bytes": int(model_bytes),
            "tensor_mib": float(model_bytes / 1024**2),
            "declared_state_scalars": int(model.state_size_scalars),
        },
        "measurement": protocol["measurement"],
        "checkpoint_load_ms": float(load_ms),
        "end_to_end_packet_runtime": timings,
        "diagnostics": diagnostics,
        "process_rss": {
            "before_model_load_bytes": rss_before,
            "after_model_load_bytes": rss_after_load,
            "after_benchmark_bytes": rss_after_benchmark,
            "increment_after_model_load_bytes": rss_after_load - rss_before,
        },
        "acceptance_gates": gates,
        "acceptance": acceptance,
        "claim_boundary": protocol["claim_boundary"],
    }
    result_path = retrospective.require_output_path(
        protocol["execution"]["result_path"],
        allowed_root=freeze.WORK_ROOT,
    )
    write_json_once(result_path, result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if not acceptance["all_runtime_gates_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
