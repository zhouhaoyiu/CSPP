"""Deployable packet-quality state machine for the v5 causal picker."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PacketQualityConfig:
    components: int = 3
    samples_per_component: int = 50
    packet_seconds: float = 0.5
    timestamp_tolerance_seconds: float = 0.05
    warmup_packets: int = 20
    impulse_crest_factor: float = 12.0
    clipping_repeat_samples: int = 3


@dataclass(frozen=True)
class PacketQualityDecision:
    valid: bool
    candidate_allowed: bool
    reset_required: bool
    reason_codes: tuple[str, ...]
    quality_flags: tuple[str, ...]
    valid_streak: int
    warmup_remaining: int


class PacketQualityGateV5:
    """Validate packets using only the current and preceding packet state."""

    def __init__(self, config: PacketQualityConfig | None = None):
        self.config = config or PacketQualityConfig()
        self.reset()

    def reset(self) -> None:
        self.previous_packet: np.ndarray | None = None
        self.previous_timestamp: float | None = None
        self.valid_streak = 0

    def evaluate(
        self,
        packet: np.ndarray,
        timestamp_seconds: float | None = None,
    ) -> PacketQualityDecision:
        config = self.config
        wave = np.asarray(packet)
        reasons: list[str] = []
        flags: list[str] = []
        expected_shape = (config.components, config.samples_per_component)

        if wave.shape != expected_shape:
            reasons.append("invalid_shape")
        elif not np.isfinite(wave).all():
            reasons.append("nonfinite_value")
        else:
            if np.any(np.ptp(wave, axis=1) == 0):
                reasons.append("constant_component")
            if self.previous_packet is not None and np.array_equal(wave, self.previous_packet):
                reasons.append("duplicate_packet")

            if wave.size:
                absolute = np.abs(wave.astype(np.float64, copy=False))
                peak = float(absolute.max())
                rms = float(np.sqrt(np.mean(absolute**2)))
                if peak > 0 and int(np.count_nonzero(absolute >= peak * (1.0 - 1e-12))) >= config.clipping_repeat_samples:
                    flags.append("possible_clipping")
                if rms > 0 and peak / rms >= config.impulse_crest_factor:
                    flags.append("isolated_impulse")

        if timestamp_seconds is not None:
            timestamp = float(timestamp_seconds)
            if not np.isfinite(timestamp):
                reasons.append("invalid_timestamp")
            elif self.previous_timestamp is not None:
                delta = timestamp - self.previous_timestamp
                if abs(delta - config.packet_seconds) > config.timestamp_tolerance_seconds:
                    reasons.append("timestamp_gap")

        if reasons:
            self.reset()
            return PacketQualityDecision(
                valid=False,
                candidate_allowed=False,
                reset_required=True,
                reason_codes=tuple(dict.fromkeys(reasons)),
                quality_flags=tuple(dict.fromkeys(flags)),
                valid_streak=0,
                warmup_remaining=config.warmup_packets,
            )

        self.valid_streak += 1
        self.previous_packet = wave.copy()
        if timestamp_seconds is not None:
            self.previous_timestamp = float(timestamp_seconds)
        candidate_allowed = self.valid_streak >= config.warmup_packets
        return PacketQualityDecision(
            valid=True,
            candidate_allowed=candidate_allowed,
            reset_required=False,
            reason_codes=(),
            quality_flags=tuple(dict.fromkeys(flags)),
            valid_streak=self.valid_streak,
            warmup_remaining=max(config.warmup_packets - self.valid_streak, 0),
        )


__all__ = [
    "PacketQualityConfig",
    "PacketQualityDecision",
    "PacketQualityGateV5",
]
