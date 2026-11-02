"""Stateful one-packet candidate formation matching the V27 replay rule."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import math


@dataclass
class PacketCandidateState:
    history_packets: int
    candidate_hits: deque[bool] = field(init=False)
    active: bool = False
    episode_index: int = 0

    def __post_init__(self) -> None:
        if self.history_packets < 1:
            raise ValueError("history_packets must be positive")
        self.candidate_hits = deque(maxlen=self.history_packets)

    def reset(self) -> None:
        self.candidate_hits.clear()
        self.active = False

    def step(
        self,
        *,
        packet_index: int,
        valid: bool,
        candidate_allowed: bool,
        candidate_probability: float,
        event_probability: float,
        arrival_age_fraction: float,
        candidate_threshold: float,
        event_threshold: float,
        start_epoch: float,
        packet_seconds: float,
        max_arrival_age_packets: float,
    ) -> dict[str, Any] | None:
        if packet_index < 0:
            raise ValueError("packet_index cannot be negative")
        if packet_seconds <= 0.0 or max_arrival_age_packets <= 0.0:
            raise ValueError("packet timing must be positive")
        if not valid:
            self.reset()
            return None
        values = (
            candidate_probability,
            event_probability,
            arrival_age_fraction,
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("valid packet probabilities must be finite")
        self.candidate_hits.append(
            bool(candidate_allowed and candidate_probability >= candidate_threshold)
        )
        current_active = bool(
            any(self.candidate_hits) and event_probability >= event_threshold
        )
        emitted = current_active and not self.active
        self.active = current_active
        if not emitted:
            return None
        self.episode_index += 1
        available_epoch = start_epoch + (packet_index + 1) * packet_seconds
        return {
            "episode_index": self.episode_index,
            "episode_start_packet": packet_index,
            "available_epoch": available_epoch,
            "estimated_p_epoch": (
                available_epoch
                - arrival_age_fraction * max_arrival_age_packets * packet_seconds
            ),
        }
