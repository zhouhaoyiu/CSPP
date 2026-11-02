"""Numerically equivalent V27 scoring with an MPS encoder and CPU state model."""
from __future__ import annotations

from collections import Counter
import math
from typing import Any, Mapping

import numpy as np
import torch

from cspp_v27_reused_core_score_cache import (
    SAMPLES_PER_PACKET,
    V27Ensemble,
    V27ScoreCache,
    _active_runs,
    _normalize_and_gate,
)


def prepare_hybrid_ensemble(ensemble: V27Ensemble) -> None:
    """Move only the stateless packet encoder to MPS."""
    if ensemble.device_name != "cpu":
        raise RuntimeError("Hybrid scoring must start from a CPU ensemble")
    if not torch.backends.mps.is_available():
        raise RuntimeError("Hybrid scoring requested but MPS is unavailable")
    ensemble.core.encoder.to(torch.device("mps")).eval()
    for name, parameter in ensemble.core.named_parameters():
        expected = "mps" if name.startswith("encoder.") else "cpu"
        if parameter.device.type != expected:
            raise RuntimeError(
                f"Unexpected hybrid parameter placement: {name}={parameter.device}"
            )
    for key, head in ensemble.event_heads.items():
        if any(parameter.device.type != "cpu" for parameter in head.parameters()):
            raise RuntimeError(f"Event head is not on CPU: {key}")


def release_hybrid_cache() -> None:
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
        torch.mps.empty_cache()


@torch.inference_mode()
def score_waveform_packets_hybrid(
    wave: np.ndarray,
    *,
    start_epoch: float,
    ensemble: V27Ensemble,
    runtime: Mapping[str, Any],
) -> V27ScoreCache:
    """Score a waveform while preserving the frozen packet and state semantics."""
    values = np.asarray(wave, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] != 3:
        raise ValueError(f"Expected ZNE wave shape (3, N), got {values.shape}")
    if values.shape[1] % SAMPLES_PER_PACKET:
        raise ValueError("Waveform length is not an exact packet multiple")
    packet_count = values.shape[1] // SAMPLES_PER_PACKET
    if packet_count <= 0:
        raise ValueError("Waveform contains no complete packet")
    sample_rate = float(runtime["sample_rate_hz"])
    packet_seconds = float(runtime["packet_duration_seconds"])
    if not math.isclose(
        sample_rate * packet_seconds,
        SAMPLES_PER_PACKET,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise RuntimeError("Runtime packet clock differs")

    normalized, valid, allowed, quality_reasons, quality_flags = (
        _normalize_and_gate(
            values,
            start_epoch=float(start_epoch),
            packet_seconds=packet_seconds,
            warmup_packets=int(runtime["warmup_packets"]),
        )
    )
    candidate = np.full(packet_count, np.nan, dtype=np.float32)
    arrival_age = np.full(packet_count, np.nan, dtype=np.float32)
    event = {
        key: np.full(packet_count, np.nan, dtype=np.float32)
        for key in ensemble.event_heads
    }

    core = ensemble.core
    block_packets = int(runtime["temporal_block_packets"])
    if block_packets <= 0:
        raise RuntimeError("temporal_block_packets must be positive")
    starts, ends = _active_runs(valid)
    scored_blocks = 0
    for segment_start, segment_end in zip(starts, ends, strict=True):
        previous_feature = None
        hidden = None
        first = int(segment_start)
        stop = int(segment_end) + 1
        for block_start in range(first, stop, block_packets):
            block_stop = min(block_start + block_packets, stop)
            chunks = (
                torch.from_numpy(normalized[block_start:block_stop])
                .unsqueeze(0)
                .to(torch.device("mps"))
            )
            features_mps, phase_mps, packet_features_mps = (
                core._encode_packets_with_verifier(chunks)
            )
            widths = (
                features_mps.shape[-1],
                phase_mps.shape[-1],
                packet_features_mps.shape[-1],
            )
            combined = torch.cat(
                (features_mps, phase_mps, packet_features_mps), dim=-1
            ).cpu()
            features, phase_logits, packet_features = torch.split(
                combined, widths, dim=-1
            )
            tokens, last_feature = core._tokens_from_features(
                features, previous_feature
            )
            sequence, next_hidden = core.gru(tokens, hidden)
            base_evidence = torch.cat(
                (sequence, tokens, core._phase_evidence(phase_logits)), dim=-1
            )
            candidate[block_start:block_stop] = (
                torch.sigmoid(core.candidate_head(base_evidence).squeeze(-1)[0])
                .numpy()
            )
            arrival_age[block_start:block_stop] = (
                torch.sigmoid(core.delay_head(sequence).squeeze(-1)[0]).numpy()
            )
            for key, head in ensemble.event_heads.items():
                mode = ensemble.event_head_modes[key]
                if mode == "none":
                    head_input = base_evidence
                elif mode == "current":
                    head_input = torch.cat(
                        (base_evidence, packet_features), dim=-1
                    )
                else:
                    raise RuntimeError(
                        f"Unsupported frozen verifier mode: {mode}"
                    )
                event[key][block_start:block_stop] = (
                    torch.sigmoid(head(head_input.detach()).squeeze(-1)[0])
                    .numpy()
                )
            previous_feature = last_feature.detach()
            hidden = next_hidden.detach()
            scored_blocks += 1
            del (
                chunks,
                features_mps,
                phase_mps,
                packet_features_mps,
                combined,
                features,
                phase_logits,
                packet_features,
                tokens,
                sequence,
                base_evidence,
                last_feature,
                next_hidden,
            )
        del previous_feature, hidden
    release_hybrid_cache()

    if not np.isfinite(candidate[valid]).all():
        raise RuntimeError("Candidate probability contains non-finite values")
    if not np.isfinite(arrival_age[valid]).all():
        raise RuntimeError("Arrival-age fraction contains non-finite values")
    for key, values_by_head in event.items():
        if not np.isfinite(values_by_head[valid]).all():
            raise RuntimeError(f"Event probability contains non-finite values: {key}")
    return V27ScoreCache(
        candidate_probability=candidate,
        event_probability=event,
        arrival_age_fraction=arrival_age,
        quality_valid=valid,
        candidate_allowed=allowed,
        diagnostics={
            "packet_count": int(packet_count),
            "quality_valid_packets": int(valid.sum()),
            "candidate_allowed_packets": int(allowed.sum()),
            "valid_segments": int(len(starts)),
            "temporal_blocks": int(scored_blocks),
            "quality_reasons": dict(sorted(Counter(quality_reasons).items())),
            "quality_flags": dict(sorted(Counter(quality_flags).items())),
            "backend": "mps_packet_encoder_cpu_gru_and_heads",
        },
    )


__all__ = [
    "prepare_hybrid_ensemble",
    "release_hybrid_cache",
    "score_waveform_packets_hybrid",
]
