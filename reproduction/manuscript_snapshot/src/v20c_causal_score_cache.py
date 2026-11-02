"""Frozen-policy causal packet scoring shared by event and background replay."""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from causal_stream_veto_features import CausalStreamVetoState
from current_packet_artifact_features import FEATURE_NAMES, packet_feature_row
from current_packet_artifact_head import CurrentPacketArtifactHead
from data_streaming import normalize_packet_ewma_causal
from domain_robust_model_io import load_checkpoint_model
from stream_quality_v5 import PacketQualityConfig, PacketQualityGateV5
from v15_reused_packet_features import (
    score_segmented_packets_with_reused_embeddings,
)


SAMPLES_PER_PACKET = 50


@dataclass(frozen=True)
class CausalScoreCache:
    p_probability: np.ndarray
    event_probability: np.ndarray
    arrival_age_fraction: np.ndarray
    artifact_probability: np.ndarray
    quality_valid: np.ndarray
    candidate_allowed: np.ndarray
    diagnostics: dict[str, Any]

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            "p_probability": self.p_probability,
            "event_probability": self.event_probability,
            "arrival_age_fraction": self.arrival_age_fraction,
            "artifact_probability": self.artifact_probability,
            "quality_valid": self.quality_valid,
            "candidate_allowed": self.candidate_allowed,
        }


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def release_device_cache(device_name: str) -> None:
    if device_name == "mps" and torch.backends.mps.is_available():
        torch.mps.synchronize()
        torch.mps.empty_cache()


def load_frozen_models(
    backbone_path: str | Path,
    artifact_head_path: str | Path,
    *,
    device_name: str,
    torch_threads: int,
    expected_backbone_sha256: str,
) -> tuple[torch.nn.Module, CurrentPacketArtifactHead, dict[str, Any]]:
    """Load and cross-check the frozen backbone and artifact head."""
    backbone_path = Path(backbone_path)
    artifact_head_path = Path(artifact_head_path)
    actual_backbone_sha256 = sha256(backbone_path)
    if actual_backbone_sha256 != expected_backbone_sha256:
        raise RuntimeError(
            "Backbone SHA-256 differs: "
            f"{actual_backbone_sha256} != {expected_backbone_sha256}"
        )
    if device_name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS scoring requested but MPS is unavailable")
    torch.set_num_threads(int(torch_threads))
    backbone, _ = load_checkpoint_model(backbone_path, device_name)
    stored = torch.load(
        artifact_head_path,
        map_location="cpu",
        weights_only=True,
    )
    artifact_head = CurrentPacketArtifactHead(**stored["config"])
    artifact_head.load_state_dict(stored["state_dict"])
    artifact_head.eval()
    expected_feature_names = list(FEATURE_NAMES) + [
        f"target_embedding_{index:03d}" for index in range(64)
    ]
    if list(stored["feature_names"]) != expected_feature_names:
        raise RuntimeError("Artifact-head feature identity differs")
    if int(stored["config"]["input_dim"]) != len(expected_feature_names):
        raise RuntimeError("Artifact-head input width differs")
    if str(stored["backbone_sha256"]) != actual_backbone_sha256:
        raise RuntimeError("Artifact head points to another backbone")
    return backbone.eval(), artifact_head, stored


def active_runs(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    active = np.asarray(values, dtype=bool)
    if not len(active):
        empty = np.empty(0, dtype=np.int64)
        return empty, empty.copy()
    changes = np.diff(
        np.concatenate(
            (
                np.zeros(1, dtype=np.int8),
                active.astype(np.int8),
                np.zeros(1, dtype=np.int8),
            )
        )
    )
    starts = np.flatnonzero(changes == 1).astype(np.int64, copy=False)
    ends = (np.flatnonzero(changes == -1) - 1).astype(
        np.int64, copy=False
    )
    return starts, ends


def candidate_support(
    probability: np.ndarray,
    allowed: np.ndarray,
    threshold: float,
    history_packets: int,
) -> np.ndarray:
    raw = np.asarray(allowed, dtype=bool) & (
        np.asarray(probability) >= float(threshold)
    )
    cumulative = np.concatenate(
        (np.zeros(1, dtype=np.int64), np.cumsum(raw, dtype=np.int64))
    )
    index = np.arange(len(raw), dtype=np.int64)
    left = np.maximum(index - int(history_packets) + 1, 0)
    return np.asarray(allowed, dtype=bool) & (
        (cumulative[index + 1] - cumulative[left]) > 0
    )


def support_with_resets(
    probability: np.ndarray,
    allowed: np.ndarray,
    valid: np.ndarray,
    threshold: float,
    history_packets: int,
) -> np.ndarray:
    support = np.zeros(len(probability), dtype=bool)
    starts, ends = active_runs(valid)
    for first, last in zip(starts, ends, strict=True):
        segment = slice(int(first), int(last) + 1)
        support[segment] = candidate_support(
            probability[segment],
            allowed[segment],
            threshold,
            history_packets,
        )
    return support


def _predict_artifact_head(
    model: CurrentPacketArtifactHead,
    features: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    if not len(features):
        return np.empty(0, dtype=np.float32)
    parts = []
    with torch.inference_mode():
        for start in range(0, len(features), int(batch_size)):
            values = torch.from_numpy(
                features[start : start + int(batch_size)]
            )
            parts.append(torch.sigmoid(model(values)).cpu().numpy())
    probability = np.concatenate(parts).astype(np.float32, copy=False)
    if not np.isfinite(probability).all():
        raise RuntimeError("Artifact probabilities are non-finite")
    return probability


def _required_runtime(runtime: Mapping[str, Any], name: str) -> Any:
    if name not in runtime:
        raise ValueError(f"Runtime contract is missing {name}")
    return runtime[name]


def score_waveform_packets(
    wave: np.ndarray,
    *,
    start_epoch: float,
    backbone: torch.nn.Module,
    artifact_head: CurrentPacketArtifactHead,
    artifact_head_input_dim: int,
    runtime: Mapping[str, Any],
    device_name: str,
) -> CausalScoreCache:
    """Score one causal stream while resetting every state at invalid gaps."""
    values = np.asarray(wave, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] != 3:
        raise ValueError(f"Expected ZNE wave shape (3, N), got {values.shape}")
    if values.shape[1] % SAMPLES_PER_PACKET:
        raise ValueError("Waveform length is not an exact packet multiple")
    packet_count = values.shape[1] // SAMPLES_PER_PACKET
    if packet_count <= 0:
        raise ValueError("Waveform contains no complete packet")

    sample_rate = float(_required_runtime(runtime, "sample_rate_hz"))
    packet_seconds = float(
        _required_runtime(runtime, "packet_duration_seconds")
    )
    if not math.isclose(
        sample_rate * packet_seconds,
        SAMPLES_PER_PACKET,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError("Runtime sample rate and packet duration differ")

    quality_gate = PacketQualityGateV5(
        PacketQualityConfig(
            warmup_packets=int(_required_runtime(runtime, "warmup_packets"))
        )
    )
    normalized_packets = np.empty(
        (packet_count, 3, SAMPLES_PER_PACKET),
        dtype=np.float32,
    )
    valid = np.zeros(packet_count, dtype=bool)
    allowed = np.zeros(packet_count, dtype=bool)
    normalization_state = None
    quality_reasons: Counter[str] = Counter()
    quality_flags: Counter[str] = Counter()
    for packet_index in range(packet_count):
        timestamp = float(start_epoch) + packet_index * packet_seconds
        first = packet_index * SAMPLES_PER_PACKET
        raw = values[:, first : first + SAMPLES_PER_PACKET]
        quality = quality_gate.evaluate(raw, timestamp)
        quality_reasons.update(quality.reason_codes)
        quality_flags.update(quality.quality_flags)
        valid[packet_index] = bool(quality.valid)
        allowed[packet_index] = bool(quality.candidate_allowed)
        if not quality.valid:
            normalization_state = None
            continue
        normalized, normalization_state = normalize_packet_ewma_causal(
            raw, normalization_state
        )
        normalized_packets[packet_index] = normalized

    p_probability = np.full(packet_count, np.nan, dtype=np.float32)
    event_probability = np.full(packet_count, np.nan, dtype=np.float32)
    arrival_age_fraction = np.full(
        packet_count, np.nan, dtype=np.float32
    )
    artifact_probability = np.full(
        packet_count, np.inf, dtype=np.float32
    )
    valid_starts, valid_ends = active_runs(valid)
    segment_lengths = valid_ends - valid_starts + 1
    valid_normalized = normalized_packets[valid]
    encoder_batches = 0
    temporal_blocks = 0
    artifact_scored_packets = 0
    if len(valid_normalized):
        scored = score_segmented_packets_with_reused_embeddings(
            backbone,
            valid_normalized,
            segment_lengths,
            device=torch.device(device_name),
            embedding_batch_size=int(
                _required_runtime(runtime, "embedding_batch_size")
            ),
            block_packets=int(
                _required_runtime(runtime, "temporal_block_packets")
            ),
            maximum_padded_packets_per_temporal_batch=int(
                _required_runtime(
                    runtime,
                    "maximum_padded_packets_per_temporal_batch",
                )
            ),
        )
        scored_p = scored.p_probability
        scored_event = scored.event_probability
        scored_age = scored.arrival_age_fraction
        target_embedding = scored.target_embedding
        encoder_batches = int(scored.encoder_batches)
        temporal_blocks = int(scored.temporal_blocks)
        del scored
        release_device_cache(device_name)

        score_start = 0
        for first_packet, last_packet in zip(
            valid_starts, valid_ends, strict=True
        ):
            first = int(first_packet)
            last = int(last_packet)
            length = last - first + 1
            score_slice = slice(score_start, score_start + length)
            absolute = np.arange(first, last + 1, dtype=np.int64)
            normalized = normalized_packets[first : last + 1]
            p_local = scored_p[score_slice]
            event_local = scored_event[score_slice]
            age_local = scored_age[score_slice]
            embedding_local = target_embedding[score_slice]
            p_probability[absolute] = p_local
            event_probability[absolute] = event_local
            arrival_age_fraction[absolute] = age_local

            support = candidate_support(
                p_local,
                allowed[first : last + 1],
                float(_required_runtime(runtime, "candidate_threshold")),
                int(
                    _required_runtime(
                        runtime, "candidate_history_packets"
                    )
                ),
            )
            eligible = support & (
                event_local
                >= float(_required_runtime(runtime, "event_threshold"))
            )
            eligible_offsets = np.flatnonzero(eligible)
            stream_state = CausalStreamVetoState(
                slow_alpha=float(
                    _required_runtime(runtime, "state_slow_alpha")
                ),
                fast_alpha=float(
                    _required_runtime(runtime, "state_fast_alpha")
                ),
            )
            features = []
            for offset in range(length):
                state_features, previous_energy = stream_state.step(
                    normalized[offset],
                    float(p_local[offset]),
                    float(event_local[offset]),
                )
                if not eligible[offset]:
                    continue
                scalar = packet_feature_row(
                    normalized,
                    p_local,
                    event_local,
                    age_local,
                    offset,
                    int(
                        _required_runtime(
                            runtime, "candidate_history_packets"
                        )
                    ),
                    state_features,
                    previous_energy,
                )
                features.append(
                    np.concatenate(
                        (
                            np.asarray(
                                [
                                    scalar[name]
                                    for name in FEATURE_NAMES
                                ],
                                dtype=np.float32,
                            ),
                            embedding_local[offset],
                        )
                    )
                )
            if len(features) != len(eligible_offsets):
                raise RuntimeError("Eligible artifact feature count differs")
            if features:
                feature_values = np.stack(features).astype(
                    np.float32, copy=False
                )
                if feature_values.shape[1] != int(
                    artifact_head_input_dim
                ):
                    raise RuntimeError("Artifact feature width differs")
                predicted = _predict_artifact_head(
                    artifact_head,
                    feature_values,
                    int(
                        _required_runtime(
                            runtime, "artifact_inference_batch_size"
                        )
                    ),
                )
                artifact_probability[
                    absolute[eligible_offsets]
                ] = predicted
                artifact_scored_packets += len(predicted)
            score_start += length
        if score_start != len(valid_normalized):
            raise RuntimeError("Segmented score coverage differs")

    diagnostics = {
        "packet_count": int(packet_count),
        "quality_valid_packets": int(valid.sum()),
        "candidate_allowed_packets": int(allowed.sum()),
        "scored_packets": int(valid.sum()),
        "artifact_scored_packets": int(artifact_scored_packets),
        "valid_segments": int(len(valid_starts)),
        "encoder_batches": encoder_batches,
        "temporal_blocks": temporal_blocks,
        "quality_reasons": dict(sorted(quality_reasons.items())),
        "quality_flags": dict(sorted(quality_flags.items())),
    }
    return CausalScoreCache(
        p_probability=p_probability,
        event_probability=event_probability,
        arrival_age_fraction=arrival_age_fraction,
        artifact_probability=artifact_probability,
        quality_valid=valid,
        candidate_allowed=allowed,
        diagnostics=diagnostics,
    )


def emit_frozen_policy_candidates(
    cache: CausalScoreCache,
    *,
    start_epoch: float,
    runtime: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Emit the first accepted packet of each cooldown-eligible run."""
    support = support_with_resets(
        cache.p_probability,
        cache.candidate_allowed,
        cache.quality_valid,
        float(_required_runtime(runtime, "candidate_threshold")),
        int(_required_runtime(runtime, "candidate_history_packets")),
    )
    accepted = (
        support
        & (
            cache.event_probability
            >= float(_required_runtime(runtime, "event_threshold"))
        )
        & (
            cache.artifact_probability
            < float(_required_runtime(runtime, "artifact_veto_threshold"))
        )
    )
    starts, ends = active_runs(accepted)
    persistence = int(
        _required_runtime(runtime, "minimum_consecutive_packets")
    )
    packet_seconds = float(
        _required_runtime(runtime, "packet_duration_seconds")
    )
    cooldown_packets = int(
        math.ceil(
            float(_required_runtime(runtime, "emission_cooldown_seconds"))
            / packet_seconds
        )
    )
    age_scale_seconds = (
        float(_required_runtime(runtime, "max_arrival_age_packets"))
        * packet_seconds
    )
    candidates = []
    last_emitted = -10**12
    for episode, (first, last) in enumerate(
        zip(starts, ends, strict=True), start=1
    ):
        first_i = int(first)
        last_i = int(last)
        completion = first_i + persistence - 1
        if completion > last_i:
            continue
        if completion - last_emitted < cooldown_packets:
            continue
        packet_start_epoch = (
            float(start_epoch) + completion * packet_seconds
        )
        available_epoch = packet_start_epoch + packet_seconds
        arrival_age = float(cache.arrival_age_fraction[completion])
        candidates.append(
            {
                "episode_index": episode,
                "episode_start_packet": first_i,
                "episode_end_packet": last_i,
                "episode_duration_packets": last_i - first_i + 1,
                "packet_index": completion,
                "packet_start_epoch": packet_start_epoch,
                "available_epoch": available_epoch,
                "estimated_p_epoch": (
                    available_epoch - arrival_age * age_scale_seconds
                ),
                "candidate_probability": float(
                    cache.p_probability[completion]
                ),
                "event_probability": float(
                    cache.event_probability[completion]
                ),
                "arrival_age_fraction": arrival_age,
                "artifact_probability": float(
                    cache.artifact_probability[completion]
                ),
            }
        )
        last_emitted = completion
    return candidates


__all__ = [
    "CausalScoreCache",
    "SAMPLES_PER_PACKET",
    "active_runs",
    "candidate_support",
    "emit_frozen_policy_candidates",
    "load_frozen_models",
    "release_device_cache",
    "score_waveform_packets",
    "sha256",
    "support_with_resets",
]
