"""Memory-bounded V27C scoring with one shared frozen candidate core."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from cspp.streaming.normalization import normalize_packet_ewma_causal
from cspp.model import CSPPCausalPickerV27C
from cspp.streaming.quality import PacketQualityConfig, PacketQualityGate


SAMPLES_PER_PACKET = 50


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def release_device_cache(device_name: str) -> None:
    if device_name == "mps" and torch.backends.mps.is_available():
        torch.mps.synchronize()
        torch.mps.empty_cache()


def tensor_state_digest(
    state: Mapping[str, torch.Tensor],
    *,
    exclude_prefixes: tuple[str, ...] = (),
) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        if any(name.startswith(prefix) for prefix in exclude_prefixes):
            continue
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


@dataclass
class ScoringEnsemble:
    core: CSPPCausalPickerV27C
    event_heads: dict[str, torch.nn.Module]
    event_head_modes: dict[str, str]
    checkpoint_sha256: dict[str, str]
    shared_core_digest: str
    device_name: str


@dataclass(frozen=True)
class ScoreCache:
    candidate_probability: np.ndarray
    event_probability: dict[str, np.ndarray]
    arrival_age_fraction: np.ndarray
    quality_valid: np.ndarray
    candidate_allowed: np.ndarray
    diagnostics: dict[str, Any]

    def arrays(self) -> dict[str, np.ndarray]:
        arrays = {
            "candidate_probability": self.candidate_probability,
            "arrival_age_fraction": self.arrival_age_fraction,
            "quality_valid": self.quality_valid,
            "candidate_allowed": self.candidate_allowed,
        }
        arrays.update(
            {
                f"event_probability__{key}": value
                for key, value in self.event_probability.items()
            }
        )
        return arrays


def load_ensemble(
    checkpoints: Mapping[str, Mapping[str, str]],
    *,
    device_name: str,
    torch_threads: int = 4,
) -> ScoringEnsemble:
    if device_name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS scoring requested but MPS is unavailable")
    torch.set_num_threads(int(torch_threads))
    device = torch.device(device_name)
    loaded: dict[str, tuple[CSPPCausalPickerV27C, dict[str, Any]]] = {}
    checkpoint_hashes = {}
    core_digests = {}
    base_configs = {}
    for key, spec in checkpoints.items():
        path = Path(spec["path"])
        actual = sha256_file(path)
        if actual != str(spec["sha256"]):
            raise RuntimeError(f"Checkpoint SHA-256 differs: {path}")
        stored = torch.load(path, map_location="cpu", weights_only=False)
        if stored.get("model_class") != "CSPPCausalPickerV27C":
            raise RuntimeError(f"Unexpected model class in {path}")
        model = CSPPCausalPickerV27C.from_config(stored["model_config"])
        model.load_state_dict(stored["model"], strict=True)
        model.eval()
        mode = str(model.config.verifier_feature_mode)
        expected_mode = str(spec["verifier_feature_mode"])
        if mode != expected_mode:
            raise RuntimeError(
                f"Verifier mode differs for {key}: {mode} != {expected_mode}"
            )
        core_digests[key] = tensor_state_digest(
            stored["model"], exclude_prefixes=("event_head.",)
        )
        base_configs[key] = {
            name: value
            for name, value in stored["model_config"].items()
            if name not in {"verifier_hidden_dim", "verifier_feature_mode"}
        }
        loaded[key] = (model, stored)
        checkpoint_hashes[key] = actual
    if len(set(core_digests.values())) != 1:
        raise RuntimeError(f"V27C non-verifier tensors differ: {core_digests}")
    if len({repr(sorted(config.items())) for config in base_configs.values()}) != 1:
        raise RuntimeError("V27C shared-core configurations differ")

    reference_key = sorted(loaded)[0]
    core = loaded[reference_key][0].to(device).eval()
    heads = {}
    modes = {}
    for key, (model, _) in loaded.items():
        heads[key] = model.event_head.to(device).eval()
        modes[key] = str(model.config.verifier_feature_mode)
    return ScoringEnsemble(
        core=core,
        event_heads=heads,
        event_head_modes=modes,
        checkpoint_sha256=checkpoint_hashes,
        shared_core_digest=next(iter(core_digests.values())),
        device_name=device_name,
    )


def _active_runs(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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
    ends = (np.flatnonzero(changes == -1) - 1).astype(np.int64, copy=False)
    return starts, ends


def _normalize_and_gate(
    wave: np.ndarray,
    *,
    start_epoch: float,
    packet_seconds: float,
    warmup_packets: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Counter[str],
    Counter[str],
]:
    packet_count = wave.shape[1] // SAMPLES_PER_PACKET
    normalized = np.zeros((packet_count, 3, SAMPLES_PER_PACKET), dtype=np.float32)
    valid = np.zeros(packet_count, dtype=bool)
    allowed = np.zeros(packet_count, dtype=bool)
    gate = PacketQualityGate(PacketQualityConfig(warmup_packets=int(warmup_packets)))
    state = None
    reasons: Counter[str] = Counter()
    flags: Counter[str] = Counter()
    for packet_index in range(packet_count):
        first = packet_index * SAMPLES_PER_PACKET
        packet = wave[:, first : first + SAMPLES_PER_PACKET]
        decision = gate.evaluate(
            packet,
            float(start_epoch) + packet_index * packet_seconds,
        )
        reasons.update(decision.reason_codes)
        flags.update(decision.quality_flags)
        valid[packet_index] = bool(decision.valid)
        allowed[packet_index] = bool(decision.candidate_allowed)
        if not decision.valid:
            state = None
            continue
        normalized[packet_index], state = normalize_packet_ewma_causal(packet, state)
    return normalized, valid, allowed, reasons, flags


@torch.inference_mode()
def score_waveform_packets(
    wave: np.ndarray,
    *,
    start_epoch: float,
    ensemble: ScoringEnsemble,
    runtime: Mapping[str, Any],
) -> ScoreCache:
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

    (
        normalized,
        valid,
        allowed,
        quality_reasons,
        quality_flags,
    ) = _normalize_and_gate(
        values,
        start_epoch=float(start_epoch),
        packet_seconds=packet_seconds,
        warmup_packets=int(runtime["warmup_packets"]),
    )
    candidate = np.full(packet_count, np.nan, dtype=np.float32)
    arrival_age = np.full(packet_count, np.nan, dtype=np.float32)
    event = {
        key: np.full(packet_count, np.nan, dtype=np.float32)
        for key in ensemble.event_heads
    }

    device = torch.device(ensemble.device_name)
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
                .to(device)
            )
            features, phase_logits, packet_features = (
                core._encode_packets_with_verifier(chunks)
            )
            tokens, last_feature = core._tokens_from_features(
                features, previous_feature
            )
            sequence, next_hidden = core.gru(tokens, hidden)
            base_evidence = torch.cat(
                (
                    sequence,
                    tokens,
                    core._phase_evidence(phase_logits),
                ),
                dim=-1,
            )
            candidate[block_start:block_stop] = (
                torch.sigmoid(core.candidate_head(base_evidence).squeeze(-1)[0])
                .cpu()
                .numpy()
            )
            arrival_age[block_start:block_stop] = (
                torch.sigmoid(core.delay_head(sequence).squeeze(-1)[0]).cpu().numpy()
            )
            for key, head in ensemble.event_heads.items():
                mode = ensemble.event_head_modes[key]
                if mode == "none":
                    head_input = base_evidence
                elif mode == "current":
                    head_input = torch.cat((base_evidence, packet_features), dim=-1)
                else:
                    raise RuntimeError(f"Unsupported frozen verifier mode: {mode}")
                event[key][block_start:block_stop] = (
                    torch.sigmoid(head(head_input.detach()).squeeze(-1)[0])
                    .cpu()
                    .numpy()
                )
            previous_feature = last_feature.detach()
            hidden = next_hidden.detach()
            scored_blocks += 1
            del (
                chunks,
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
    release_device_cache(ensemble.device_name)

    if not np.isfinite(candidate[valid]).all():
        raise RuntimeError("Candidate probability contains non-finite values")
    if not np.isfinite(arrival_age[valid]).all():
        raise RuntimeError("Arrival-age fraction contains non-finite values")
    for key, values_by_head in event.items():
        if not np.isfinite(values_by_head[valid]).all():
            raise RuntimeError(f"Event probability contains non-finite values: {key}")
    return ScoreCache(
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
            "quality_reasons": dict(sorted(quality_reasons.items())),
            "quality_flags": dict(sorted(quality_flags.items())),
        },
    )


__all__ = [
    "ScoringEnsemble",
    "ScoreCache",
    "load_ensemble",
    "release_device_cache",
    "score_waveform_packets",
    "sha256_file",
    "tensor_state_digest",
]
