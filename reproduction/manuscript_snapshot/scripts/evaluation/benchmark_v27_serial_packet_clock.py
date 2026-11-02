#!/usr/bin/env python3
"""Benchmark frozen pickers under one causal 0.5 s packet clock."""
from __future__ import annotations

from datetime import UTC, datetime
import gc
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any, Callable

import numpy as np
from obspy import UTCDateTime
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/evaluation"))

import build_v27_ncedc_continuous_score_cache as score_io  # noqa: E402
import eval_v27c_fixed_public_baselines as public_eval  # noqa: E402
from cspp_v27_reused_core_score_cache import sha256_file  # noqa: E402
from data_streaming import normalize_packet_ewma_causal  # noqa: E402
from model_causal_classical_v1 import CausalMultiBandSTALTAPickerV1  # noqa: E402
from model_cspp_v27 import CSPPCausalPickerV27C  # noqa: E402
from oriented_gappy_strong_motion_stream import (  # noqa: E402
    ChannelOrientation,
    stream_to_oriented_gappy_zne_array,
)
from resilient_mseed import read_resilient_mseed  # noqa: E402
from stream_quality_v5 import PacketQualityConfig, PacketQualityGateV5  # noqa: E402
from v20c_public_window_baselines import (  # noqa: E402
    preprocess_windows,
    public_outputs,
    replay_packet_quality,
)


EXPERIMENT = Path(
    "workspace/"
    "ncedc_joint_continuous_m3_20260904"
)
PACKET_SAMPLES = 50
PACKET_SECONDS = 0.5
SEGMENT_PACKETS = 360
WARMUP_PACKETS = 120
TIMED_PACKETS = SEGMENT_PACKETS - WARMUP_PACKETS
REPETITIONS = 5


def load_real_segment() -> tuple[np.ndarray, float, dict[str, Any]]:
    scoring_path = EXPERIMENT / "scoring_protocol.json"
    scoring = json.loads(scoring_path.read_text())
    profile = json.loads(
        (EXPERIMENT / "runtime_profile/cspp_station_day_stage_profile.json").read_text()
    )
    station_day_id = str(profile["station_day_id"])
    requests = score_io.read_csv(Path(scoring["inputs"]["station_day_requests"]["path"]))
    request = next(row for row in requests if score_io.station_day_id(row) == station_day_id)
    acquisitions = {
        row["station_day_id"]: row
        for row in score_io.read_csv(Path(scoring["inputs"]["acquisition_status"]["path"]))
    }
    orientation_rows = score_io.orientation_groups(
        score_io.read_csv(Path(scoring["inputs"]["station_day_orientations"]["path"]))
    )[station_day_id]
    orientations = [
        ChannelOrientation(
            network=row["network"],
            station=row["station"],
            location=row["location"],
            channel=row["channel"],
            azimuth=float(row["azimuth_deg"]),
            dip=float(row["dip_deg"]),
            scale=float(row["scale"]),
            scale_units=row["scale_units"],
        )
        for row in orientation_rows
    ]
    source = read_resilient_mseed(Path(acquisitions[station_day_id]["mseed_file"]))
    start = UTCDateTime(request["request_start_utc"])
    end = UTCDateTime(request["request_end_utc"])
    wave, metadata = stream_to_oriented_gappy_zne_array(
        source.stream,
        {
            "network": request["network"],
            "station": request["station"],
            "location": request["location"],
            "latitude": float(request["station_latitude"]),
            "longitude": float(request["station_longitude"]),
        },
        orientations,
        start,
        end,
        target_sample_rate_hz=100.0,
        samples_per_packet=PACKET_SAMPLES,
    )
    valid, _, quality = replay_packet_quality(
        wave,
        start_epoch=float(start.timestamp),
        warmup_packets=int(scoring["runtime"]["warmup_packets"]),
    )
    edges = np.diff(np.r_[False, valid, False].astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    stops = np.flatnonzero(edges == -1)
    runs = [(int(a), int(b)) for a, b in zip(starts, stops, strict=True) if b - a >= SEGMENT_PACKETS]
    if not runs:
        raise RuntimeError("No quality-valid 180 s interval is available")
    run_start, run_stop = max(runs, key=lambda pair: pair[1] - pair[0])
    first_packet = run_start + (run_stop - run_start - SEGMENT_PACKETS) // 2
    first_sample = first_packet * PACKET_SAMPLES
    segment = np.ascontiguousarray(
        wave[:, first_sample : first_sample + SEGMENT_PACKETS * PACKET_SAMPLES],
        dtype=np.float32,
    )
    if segment.shape != (3, SEGMENT_PACKETS * PACKET_SAMPLES):
        raise RuntimeError(f"Unexpected segment shape: {segment.shape}")
    segment_epoch = float(start.timestamp) + first_packet * PACKET_SECONDS
    provenance = {
        "station_day_id": station_day_id,
        "station": ".".join([request["network"], request["station"], request["location"]]),
        "segment_start_utc": datetime.fromtimestamp(segment_epoch, UTC).isoformat(),
        "segment_seconds": SEGMENT_PACKETS * PACKET_SECONDS,
        "segment_sha256": hashlib.sha256(segment.tobytes()).hexdigest(),
        "source_mseed": acquisitions[station_day_id]["mseed_file"],
        "source_mseed_sha256": acquisitions[station_day_id].get("mseed_sha256", ""),
        "quality_diagnostics": quality,
        "orientation_metadata": metadata,
        "scoring_protocol_sha256": sha256_file(scoring_path),
    }
    return segment, segment_epoch, provenance


def summarize(samples_ms: list[float], checksum: float) -> dict[str, Any]:
    values = np.asarray(samples_ms, dtype=np.float64)
    expected = REPETITIONS * TIMED_PACKETS
    if len(values) != expected or not np.isfinite(values).all():
        raise RuntimeError(f"Expected {expected} finite timings, got {len(values)}")
    p95 = float(np.percentile(values, 95))
    return {
        "updates": int(len(values)),
        "repetitions": REPETITIONS,
        "timed_updates_per_repetition": TIMED_PACKETS,
        "median_ms": float(np.median(values)),
        "mean_ms": float(np.mean(values)),
        "p95_ms": p95,
        "p99_ms": float(np.percentile(values, 99)),
        "max_ms": float(np.max(values)),
        "p95_realtime_margin": float(500.0 / p95),
        "checksum": float(checksum),
        "samples_ms": values.tolist(),
    }


def timed_packets(step: Callable[[int], float]) -> tuple[list[float], float]:
    samples: list[float] = []
    checksum = 0.0
    for packet_index in range(SEGMENT_PACKETS):
        started = time.perf_counter_ns()
        checksum += float(step(packet_index))
        elapsed_ms = (time.perf_counter_ns() - started) / 1e6
        if packet_index >= WARMUP_PACKETS:
            samples.append(elapsed_ms)
    return samples, checksum


def benchmark_cspp(
    wave: np.ndarray,
    start_epoch: float,
    model: CSPPCausalPickerV27C,
    warmup_packets: int,
) -> dict[str, Any]:
    all_samples: list[float] = []
    checksum = 0.0
    for _ in range(REPETITIONS):
        gate = PacketQualityGateV5(PacketQualityConfig(warmup_packets=warmup_packets))
        normalization_state = None
        model_state = None

        def step(packet_index: int) -> float:
            nonlocal normalization_state, model_state
            first = packet_index * PACKET_SAMPLES
            packet = wave[:, first : first + PACKET_SAMPLES]
            decision = gate.evaluate(packet, start_epoch + packet_index * PACKET_SECONDS)
            if not decision.valid:
                normalization_state = None
                model_state = None
                return 0.0
            normalized, normalization_state = normalize_packet_ewma_causal(
                packet, normalization_state
            )
            candidate, event, delay, model_state = model.forward_streaming_packet(
                torch.from_numpy(normalized), model_state
            )
            return candidate.item() + event.item() + delay.item()

        samples, value = timed_packets(step)
        all_samples.extend(samples)
        checksum += value
    return summarize(all_samples, checksum)


def benchmark_public(
    wave: np.ndarray,
    start_epoch: float,
    model: torch.nn.Module,
    spec: dict[str, Any],
    warmup_packets: int,
) -> dict[str, Any]:
    in_samples = int(spec["in_samples"])
    right_blind = int(spec["blinding_samples"][1])
    all_samples: list[float] = []
    checksum = 0.0
    for _ in range(REPETITIONS):
        gate = PacketQualityGateV5(PacketQualityConfig(warmup_packets=warmup_packets))

        def step(packet_index: int) -> float:
            first = packet_index * PACKET_SAMPLES
            packet = wave[:, first : first + PACKET_SAMPLES]
            decision = gate.evaluate(packet, start_epoch + packet_index * PACKET_SECONDS)
            stop = (packet_index + 1) * PACKET_SAMPLES
            if not decision.valid or stop < in_samples:
                return 0.0
            window = wave[:, stop - in_samples : stop][None, ...]
            prepared = preprocess_windows(model, window, spec, device=torch.device("cpu"))
            p_output, detection_output = public_outputs(model, prepared)
            output_index = p_output.shape[-1] - right_blind - 1
            value = p_output[0, output_index].item()
            if detection_output is not None:
                value += detection_output[0, output_index].item()
            return value

        samples, value = timed_packets(step)
        all_samples.extend(samples)
        checksum += value
    return summarize(all_samples, checksum)


def benchmark_sta_lta(
    wave: np.ndarray,
    start_epoch: float,
    model: CausalMultiBandSTALTAPickerV1,
    warmup_packets: int,
) -> dict[str, Any]:
    all_samples: list[float] = []
    checksum = 0.0
    for _ in range(REPETITIONS):
        gate = PacketQualityGateV5(PacketQualityConfig(warmup_packets=warmup_packets))
        normalization_state = None
        model_state = None

        def step(packet_index: int) -> float:
            nonlocal normalization_state, model_state
            first = packet_index * PACKET_SAMPLES
            packet = wave[:, first : first + PACKET_SAMPLES]
            decision = gate.evaluate(packet, start_epoch + packet_index * PACKET_SECONDS)
            if not decision.valid:
                normalization_state = None
                model_state = None
                return 0.0
            normalized, normalization_state = normalize_packet_ewma_causal(
                packet, normalization_state
            )
            peak, sustained, offset, model_state = model.forward_streaming_packet(
                torch.from_numpy(normalized), model_state
            )
            return peak.item() + sustained.item() + offset.item()

        samples, value = timed_packets(step)
        all_samples.extend(samples)
        checksum += value
    return summarize(all_samples, checksum)


def load_cspp(spec: dict[str, Any]) -> CSPPCausalPickerV27C:
    checkpoint = Path(spec["path"])
    if sha256_file(checkpoint) != spec["sha256"]:
        raise RuntimeError("CSPP checkpoint hash differs from the frozen protocol")
    stored = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = CSPPCausalPickerV27C.from_config(stored["model_config"])
    model.load_state_dict(stored["model"], strict=True)
    return model.eval()


def main() -> None:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    scoring_path = EXPERIMENT / "scoring_protocol.json"
    public_path = EXPERIMENT / "public_baseline_protocol.json"
    classical_path = EXPERIMENT / "classical_baseline_protocol.json"
    scoring = json.loads(scoring_path.read_text())
    public_protocol = json.loads(public_path.read_text())
    classical = json.loads(classical_path.read_text())
    wave, start_epoch, provenance = load_real_segment()
    warmup_packets = int(scoring["runtime"]["warmup_packets"])

    cspp_key = sorted(scoring["models"])[0]
    cspp = load_cspp(scoring["models"][cspp_key])
    results = {
        "CSPP": benchmark_cspp(wave, start_epoch, cspp, warmup_packets),
    }
    del cspp
    gc.collect()

    for result_key, protocol_key in (
        ("PhaseNet", "phasenet_instance"),
        ("EQTransformer", "eqtransformer_original"),
    ):
        spec = public_protocol["models"][protocol_key]
        model, _ = public_eval.load_sealed_public_model(spec, device=torch.device("cpu"))
        results[result_key] = benchmark_public(
            wave, start_epoch, model, spec, warmup_packets
        )
        del model
        gc.collect()

    classical_model = CausalMultiBandSTALTAPickerV1.from_config(
        classical["model"]["configuration"]
    ).eval()
    results["STA_LTA"] = benchmark_sta_lta(
        wave, start_epoch, classical_model, warmup_packets
    )

    payload = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "protocol": {
            "clock_seconds": PACKET_SECONDS,
            "segment_packets": SEGMENT_PACKETS,
            "warmup_packets_not_timed": WARMUP_PACKETS,
            "timed_packets_per_repetition": TIMED_PACKETS,
            "repetitions": REPETITIONS,
            "device": "cpu",
            "torch_threads": 1,
            "batch_size": 1,
            "synchronization": "Every selected scalar output is materialized with Tensor.item().",
            "phase_net": "At each packet, recompute the latest fully historical 3001-sample window.",
            "eqtransformer": "At each packet, recompute the latest fully historical 6000-sample window, including its frozen filter.",
            "cspp": "Process the new 50-sample packet once and retain causal normalization and recurrent state.",
            "sta_lta": "Process the new 50-sample packet once and retain causal normalization and recursive energy state.",
        },
        "software": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "inputs": {
            "scoring_protocol_sha256": sha256_file(scoring_path),
            "public_protocol_sha256": sha256_file(public_path),
            "classical_protocol_sha256": sha256_file(classical_path),
            "cspp_checkpoint_key": cspp_key,
            **provenance,
        },
        "results": results,
        "boundary": (
            "Controlled engineering latency benchmark on one real 180 s record. "
            "It compares serial causal wrappers, not published batch throughput or detection accuracy."
        ),
    }
    output = EXPERIMENT / "runtime_profile/serial_packet_clock_cpu1.json"
    score_io.write_json_atomic(output, payload)
    print(json.dumps({"output": str(output), "results": results}, indent=2))


if __name__ == "__main__":
    main()
