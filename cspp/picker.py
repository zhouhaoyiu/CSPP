"""One-station packet API using the frozen model and release trigger rules."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch

from cspp.checkpoint import load_checkpoint_model
from cspp.streaming.candidate import PacketCandidateState
from cspp.streaming.normalization import normalize_packet_ewma_causal
from cspp.streaming.quality import PacketQualityConfig, PacketQualityGate


@dataclass(frozen=True)
class PacketResult:
    packet_index: int
    packet_start_seconds: float
    available_seconds: float
    quality_valid: bool
    candidate_allowed: bool
    warmup_remaining: int
    reason_codes: tuple[str, ...]
    quality_flags: tuple[str, ...]
    candidate_probability: float | None
    event_probability: float | None
    arrival_age_fraction: float | None
    pick: dict[str, Any] | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class StreamingPicker:
    """Process complete (3, 50) acceleration packets in ZNE order at 100 Hz.

    Create one instance per station. Timestamps refer to the first sample of
    each packet and use one clock (relative seconds or Unix epoch seconds).
    There is no waveform history buffer. Invalid packets reset inference state;
    a timestamp gap discards its current packet as in the frozen quality gate.
    """

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        candidate_threshold: float = 0.1,
        event_threshold: float = 0.975,
        warmup_packets: int = 20,
        torch_threads: int = 1,
    ):
        for name, value in (
            ("candidate_threshold", candidate_threshold),
            ("event_threshold", event_threshold),
        ):
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be finite and between 0 and 1")
        if isinstance(warmup_packets, bool) or not isinstance(warmup_packets, int):
            raise ValueError("warmup_packets must be a nonnegative integer")
        if warmup_packets < 0:
            raise ValueError("warmup_packets must be a nonnegative integer")
        if isinstance(torch_threads, bool) or not isinstance(torch_threads, int) or torch_threads < 1:
            raise ValueError("torch_threads must be a positive integer")
        torch.set_num_threads(torch_threads)
        self.model, stored = load_checkpoint_model(checkpoint, device="cpu")
        config = self.model.config
        if stored["model_class"] != "CSPPCausalPickerV27C" or config.verifier_feature_mode != "none":
            raise ValueError("StreamingPicker supports the released C0 checkpoints only")
        if config.temporal_mode != "gru" or config.max_arrival_age_packets != 4.0:
            raise ValueError("Checkpoint is incompatible with the frozen release policy")
        if config.packet_samples != 50 or config.sample_rate_hz != 100.0:
            raise ValueError("Checkpoint does not use 50-sample, 100 Hz packets")
        self.candidate_threshold = float(candidate_threshold)
        self.event_threshold = float(event_threshold)
        self.warmup_packets = warmup_packets
        self.reset()

    def reset(self) -> None:
        """Start an independent stream; numbering and warmup restart at zero."""
        self._gate = PacketQualityGate(PacketQualityConfig(warmup_packets=self.warmup_packets))
        self._candidates = PacketCandidateState(history_packets=4)
        self._normalization = None
        self._model_state = None
        self._packet_index = 0

    @torch.inference_mode()
    def process_packet(self, packet: np.ndarray, *, timestamp_seconds: float) -> PacketResult:
        """Score one complete packet; never pad, interpolate, resample or reorder."""
        raw = np.asarray(packet)
        if raw.shape != (3, 50) or raw.dtype.kind not in "fiu":
            raise ValueError("packet must be a real numeric array with shape (3, 50)")
        timestamp = float(timestamp_seconds)
        if not math.isfinite(timestamp):
            raise ValueError("timestamp_seconds must be finite")
        values = raw.astype(np.float32, copy=False)
        decision = self._gate.evaluate(values, timestamp)
        index = self._packet_index
        self._packet_index += 1
        candidate = event = age = None
        pick = None
        if not decision.valid:
            self._normalization = None
            self._model_state = None
            self._candidates.reset()
        else:
            normalized, self._normalization = normalize_packet_ewma_causal(
                values, self._normalization
            )
            outputs = self.model.forward_streaming_packet(
                torch.from_numpy(normalized), self._model_state
            )
            self._model_state = outputs[-1]
            candidate, event, age = [float(torch.sigmoid(value).item()) for value in outputs[:3]]
            # The candidate policy uses packet indices; anchor each call to the
            # actual packet clock so resets and telemetry gaps cannot shift picks.
            pick = self._candidates.step(
                packet_index=index,
                valid=True,
                candidate_allowed=decision.candidate_allowed,
                candidate_probability=candidate,
                event_probability=event,
                arrival_age_fraction=age,
                candidate_threshold=self.candidate_threshold,
                event_threshold=self.event_threshold,
                start_epoch=timestamp - index * 0.5,
                packet_seconds=0.5,
                max_arrival_age_packets=4.0,
            )
        return PacketResult(
            packet_index=index,
            packet_start_seconds=timestamp,
            available_seconds=timestamp + 0.5,
            quality_valid=decision.valid,
            candidate_allowed=decision.candidate_allowed,
            warmup_remaining=decision.warmup_remaining,
            reason_codes=decision.reason_codes,
            quality_flags=decision.quality_flags,
            candidate_probability=candidate,
            event_probability=event,
            arrival_age_fraction=age,
            pick=pick,
        )

    def iter_array(self, wave: np.ndarray, *, start_seconds: float = 0.0) -> Iterator[PacketResult]:
        """Replay a prepared array without resetting an existing stream."""
        values = np.asarray(wave)
        if values.ndim != 2 or values.shape[0] != 3 or values.shape[1] == 0 or values.shape[1] % 50:
            raise ValueError("wave must have shape (3, N), with positive N divisible by 50")
        for index, first in enumerate(range(0, values.shape[1], 50)):
            yield self.process_packet(
                values[:, first:first + 50], timestamp_seconds=start_seconds + index * 0.5
            )
