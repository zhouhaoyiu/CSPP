"""Lightweight runtime construction of the frozen 139 artifact features."""
from __future__ import annotations

import numpy as np

from causal_packet_veto_features import (
    PHYSICS_FEATURE_NAMES,
    causal_packet_veto_features,
)
from causal_stream_veto_features import CAUSAL_STREAM_FEATURE_NAMES


MODEL_FEATURE_NAMES = (
    "candidate_probability",
    "candidate_history_max",
    "candidate_age_packets",
    "verifier_probability",
    "verifier_previous",
    "verifier_delta",
    "verifier_mean_2",
    "arrival_age_fraction",
)
FEATURE_NAMES = (
    MODEL_FEATURE_NAMES + PHYSICS_FEATURE_NAMES + CAUSAL_STREAM_FEATURE_NAMES
)


def packet_feature_row(
    chunks: np.ndarray,
    candidate: np.ndarray,
    verifier: np.ndarray,
    offsets: np.ndarray,
    packet: int,
    history_packets: int,
    state_features: dict[str, float],
    previous_energy: float | None,
) -> dict[str, float]:
    """Construct the frozen current-package artifact feature mapping."""
    history_start = max(0, packet - history_packets + 1)
    history = candidate[history_start : packet + 1]
    history_peak = int(np.argmax(history))
    previous_verifier = float(verifier[packet - 1]) if packet else 0.0
    return {
        "candidate_probability": float(candidate[packet]),
        "candidate_history_max": float(history.max()),
        "candidate_age_packets": float(len(history) - 1 - history_peak),
        "verifier_probability": float(verifier[packet]),
        "verifier_previous": previous_verifier,
        "verifier_delta": float(verifier[packet]) - previous_verifier,
        "verifier_mean_2": 0.5
        * (float(verifier[packet]) + previous_verifier),
        "arrival_age_fraction": float(offsets[packet]),
        **causal_packet_veto_features(chunks[packet], previous_energy),
        **state_features,
    }


def streaming_packet_feature_row(
    *,
    normalized_packet: np.ndarray,
    candidate_probability: float,
    candidate_history: tuple[float, ...],
    verifier_probability: float,
    previous_verifier_probability: float,
    arrival_age_fraction: float,
    state_features: dict[str, float],
    previous_energy: float | None,
) -> dict[str, float]:
    """Construct the same feature row from bounded streaming state."""
    history = np.asarray(candidate_history, dtype=np.float64)
    if history.ndim != 1 or not len(history):
        raise ValueError("candidate_history must include the current packet")
    history_peak = int(np.argmax(history))
    return {
        "candidate_probability": float(candidate_probability),
        "candidate_history_max": float(history.max()),
        "candidate_age_packets": float(len(history) - 1 - history_peak),
        "verifier_probability": float(verifier_probability),
        "verifier_previous": float(previous_verifier_probability),
        "verifier_delta": float(
            verifier_probability - previous_verifier_probability
        ),
        "verifier_mean_2": 0.5
        * float(verifier_probability + previous_verifier_probability),
        "arrival_age_fraction": float(arrival_age_fraction),
        **causal_packet_veto_features(
            normalized_packet,
            previous_energy,
        ),
        **state_features,
    }


__all__ = [
    "FEATURE_NAMES",
    "MODEL_FEATURE_NAMES",
    "packet_feature_row",
    "streaming_packet_feature_row",
]
