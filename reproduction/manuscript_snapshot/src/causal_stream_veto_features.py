"""Bounded-state causal background features for a packet-once artifact veto."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


STREAM_SUMMARY_NAMES = (
    "log1p_rms",
    "derivative_ratio",
    "zero_crossing_fraction",
    "max_component_energy_fraction",
    "z_energy_fraction",
    "lag1_correlation",
    "edge_jump_over_rms",
    "second_half_energy_fraction",
)


def _packet_summary(packet: np.ndarray) -> tuple[np.ndarray, float]:
    values = np.asarray(packet, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != 3 or values.shape[1] < 8:
        raise ValueError(f"Expected packet shape (3, L>=8), got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("Packet contains non-finite values")

    eps = 1e-12
    samples = values.shape[1]
    centered = values - values.mean(axis=1, keepdims=True)
    energy = max(float(np.mean(centered**2)), eps)
    rms = math.sqrt(energy)
    derivative = np.diff(centered, axis=1)
    component_energy = np.mean(centered**2, axis=1)
    component_total = max(float(component_energy.sum()), eps)
    zero_crossings = np.count_nonzero(centered[:, 1:] * centered[:, :-1] < 0.0)
    lag_numerator = float(np.sum(centered[:, 1:] * centered[:, :-1]))
    lag_denominator = math.sqrt(
        max(float(np.sum(centered[:, 1:] ** 2)), eps)
        * max(float(np.sum(centered[:, :-1] ** 2)), eps)
    )
    edge = max(2, min(10, samples // 4))
    half = samples // 2
    first_energy = float(np.mean(centered[:, :half] ** 2))
    second_energy = float(np.mean(centered[:, half:] ** 2))
    summary = np.asarray(
        [
            math.log1p(rms),
            float(np.sqrt(np.mean(derivative**2))) / max(rms, eps),
            float(zero_crossings) / (3.0 * (samples - 1)),
            float(component_energy.max()) / component_total,
            float(component_energy[0]) / component_total,
            lag_numerator / max(lag_denominator, eps),
            float(
                np.max(
                    np.abs(
                        centered[:, -edge:].mean(axis=1)
                        - centered[:, :edge].mean(axis=1)
                    )
                )
            )
            / max(rms, eps),
            second_energy / max(first_energy + second_energy, eps),
        ],
        dtype=np.float64,
    )
    summary = np.nan_to_num(summary, nan=0.0, posinf=1e6, neginf=-1e6)
    return summary, energy


def _feature_names() -> tuple[str, ...]:
    names: list[str] = []
    for name in STREAM_SUMMARY_NAMES:
        names.extend(
            (
                f"state_{name}_minus_slow",
                f"state_{name}_z_slow",
                f"state_{name}_minus_fast",
                f"state_{name}_fast_minus_slow",
            )
        )
    names.extend(
        (
            "state_candidate_minus_slow",
            "state_candidate_minus_fast",
            "state_verifier_minus_slow",
            "state_verifier_minus_fast",
            "state_candidate_slow",
            "state_verifier_slow",
            "state_age_log1p",
        )
    )
    return tuple(names)


CAUSAL_STREAM_FEATURE_NAMES = _feature_names()


def _finite_float(value: float) -> float:
    """Sanitize one scalar without constructing NumPy arrays per feature."""
    result = float(value)
    if math.isnan(result):
        return 0.0
    if result == math.inf:
        return 1e6
    if result == -math.inf:
        return -1e6
    return result


@dataclass
class CausalStreamVetoState:
    """Maintain fixed-size summaries; raw packets are never retained."""

    slow_alpha: float = 0.05
    fast_alpha: float = 0.20
    count: int = 0
    slow_mean: np.ndarray = field(
        default_factory=lambda: np.zeros(len(STREAM_SUMMARY_NAMES), dtype=np.float64)
    )
    slow_variance: np.ndarray = field(
        default_factory=lambda: np.zeros(len(STREAM_SUMMARY_NAMES), dtype=np.float64)
    )
    fast_mean: np.ndarray = field(
        default_factory=lambda: np.zeros(len(STREAM_SUMMARY_NAMES), dtype=np.float64)
    )
    candidate_slow: float = 0.0
    candidate_fast: float = 0.0
    verifier_slow: float = 0.0
    verifier_fast: float = 0.0
    previous_energy: float | None = None

    def __post_init__(self) -> None:
        if not 0.0 < self.slow_alpha <= 1.0:
            raise ValueError("slow_alpha must be in (0, 1]")
        if not 0.0 < self.fast_alpha <= 1.0:
            raise ValueError("fast_alpha must be in (0, 1]")
        expected = (len(STREAM_SUMMARY_NAMES),)
        for name in ("slow_mean", "slow_variance", "fast_mean"):
            values = np.asarray(getattr(self, name), dtype=np.float64)
            if values.shape != expected:
                raise ValueError(f"{name} must have shape {expected}, got {values.shape}")
            setattr(self, name, values.copy())

    def step(
        self,
        packet: np.ndarray,
        candidate_probability: float,
        verifier_probability: float,
    ) -> tuple[dict[str, float], float | None]:
        summary, energy = _packet_summary(packet)
        previous_energy = self.previous_energy
        if self.count == 0:
            delta_slow = np.zeros_like(summary)
            z_slow = np.zeros_like(summary)
            delta_fast = np.zeros_like(summary)
            fast_minus_slow = np.zeros_like(summary)
            candidate_minus_slow = 0.0
            candidate_minus_fast = 0.0
            verifier_minus_slow = 0.0
            verifier_minus_fast = 0.0
        else:
            delta_slow = summary - self.slow_mean
            scale = np.sqrt(np.maximum(self.slow_variance, 1e-6)) + 1e-3
            z_slow = np.clip(delta_slow / scale, -20.0, 20.0)
            delta_fast = summary - self.fast_mean
            fast_minus_slow = self.fast_mean - self.slow_mean
            candidate_minus_slow = candidate_probability - self.candidate_slow
            candidate_minus_fast = candidate_probability - self.candidate_fast
            verifier_minus_slow = verifier_probability - self.verifier_slow
            verifier_minus_fast = verifier_probability - self.verifier_fast

        values: dict[str, float] = {}
        for index, name in enumerate(STREAM_SUMMARY_NAMES):
            values[f"state_{name}_minus_slow"] = float(delta_slow[index])
            values[f"state_{name}_z_slow"] = float(z_slow[index])
            values[f"state_{name}_minus_fast"] = float(delta_fast[index])
            values[f"state_{name}_fast_minus_slow"] = float(
                fast_minus_slow[index]
            )
        values.update(
            {
                "state_candidate_minus_slow": float(candidate_minus_slow),
                "state_candidate_minus_fast": float(candidate_minus_fast),
                "state_verifier_minus_slow": float(verifier_minus_slow),
                "state_verifier_minus_fast": float(verifier_minus_fast),
                "state_candidate_slow": float(self.candidate_slow),
                "state_verifier_slow": float(self.verifier_slow),
                "state_age_log1p": math.log1p(self.count),
            }
        )

        if self.count == 0:
            self.slow_mean = summary.copy()
            self.fast_mean = summary.copy()
            self.candidate_slow = float(candidate_probability)
            self.candidate_fast = float(candidate_probability)
            self.verifier_slow = float(verifier_probability)
            self.verifier_fast = float(verifier_probability)
        else:
            slow_delta = summary - self.slow_mean
            self.slow_mean += self.slow_alpha * slow_delta
            self.slow_variance = (1.0 - self.slow_alpha) * (
                self.slow_variance + self.slow_alpha * slow_delta**2
            )
            self.fast_mean += self.fast_alpha * (summary - self.fast_mean)
            self.candidate_slow += self.slow_alpha * (
                candidate_probability - self.candidate_slow
            )
            self.candidate_fast += self.fast_alpha * (
                candidate_probability - self.candidate_fast
            )
            self.verifier_slow += self.slow_alpha * (
                verifier_probability - self.verifier_slow
            )
            self.verifier_fast += self.fast_alpha * (
                verifier_probability - self.verifier_fast
            )
        self.previous_energy = energy
        self.count += 1
        return {
            key: _finite_float(value) for key, value in values.items()
        }, previous_energy


__all__ = [
    "CAUSAL_STREAM_FEATURE_NAMES",
    "CausalStreamVetoState",
    "STREAM_SUMMARY_NAMES",
]
