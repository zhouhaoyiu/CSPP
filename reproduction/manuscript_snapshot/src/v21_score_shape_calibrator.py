"""Tiny causal calibration gate over frozen station-model score traces."""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np


ROLLING_WINDOWS = (2, 4, 8, 16)
BASE_FEATURE_NAMES = (
    "p_current",
    "event_current",
    "artifact_current",
    "age_current",
    "p_logit",
    "event_logit",
    "artifact_logit",
    "p_event_product",
    "event_minus_p",
    "p_delta_1",
    "event_delta_1",
    "artifact_delta_1",
    "age_delta_1",
)
WINDOW_FEATURE_SUFFIXES = (
    "p_mean",
    "p_std",
    "p_min",
    "p_max",
    "event_mean",
    "event_std",
    "event_min",
    "event_max",
    "artifact_mean",
    "artifact_std",
    "artifact_min",
    "artifact_max",
    "age_mean",
    "age_std",
    "age_range",
    "raw_p_fraction",
    "event_fraction",
    "artifact_scored_fraction",
    "candidate_allowed_fraction",
    "quality_valid_fraction",
)
FEATURE_NAMES = BASE_FEATURE_NAMES + tuple(
    f"w{window}_{suffix}"
    for window in ROLLING_WINDOWS
    for suffix in WINDOW_FEATURE_SUFFIXES
)


def _clip_probability(value: float | np.ndarray) -> float | np.ndarray:
    return np.clip(value, 1e-6, 1.0 - 1e-6)


def _logit(value: float) -> float:
    clipped = float(_clip_probability(value))
    return math.log(clipped / (1.0 - clipped))


def _previous(values: np.ndarray, index: int) -> float:
    return float(values[max(0, index - 1)])


def score_shape_feature_row(
    *,
    p_probability: np.ndarray,
    event_probability: np.ndarray,
    artifact_probability: np.ndarray,
    arrival_age_fraction: np.ndarray,
    candidate_allowed: np.ndarray,
    quality_valid: np.ndarray,
    packet_index: int,
    candidate_threshold: float,
    event_threshold: float,
) -> np.ndarray:
    """Return bounded past-only features at one packet index."""
    p = np.asarray(p_probability, dtype=np.float64)
    event = np.asarray(event_probability, dtype=np.float64)
    artifact_raw = np.asarray(
        artifact_probability, dtype=np.float64
    )
    age = np.asarray(arrival_age_fraction, dtype=np.float64)
    allowed = np.asarray(candidate_allowed, dtype=bool)
    valid = np.asarray(quality_valid, dtype=bool)
    lengths = {
        len(p),
        len(event),
        len(artifact_raw),
        len(age),
        len(allowed),
        len(valid),
    }
    if len(lengths) != 1:
        raise ValueError("Score-shape arrays have different lengths")
    index = int(packet_index)
    if not 0 <= index < len(p):
        raise IndexError(packet_index)
    artifact = np.where(
        np.isfinite(artifact_raw), artifact_raw, 1.0
    )
    current = (
        float(p[index]),
        float(event[index]),
        float(artifact[index]),
        float(age[index]),
    )
    values = [
        *current,
        _logit(current[0]),
        _logit(current[1]),
        _logit(current[2]),
        current[0] * current[1],
        current[1] - current[0],
        current[0] - _previous(p, index),
        current[1] - _previous(event, index),
        current[2] - _previous(artifact, index),
        current[3] - _previous(age, index),
    ]
    for window in ROLLING_WINDOWS:
        first = max(0, index - int(window) + 1)
        selection = slice(first, index + 1)
        p_window = p[selection]
        event_window = event[selection]
        artifact_window = artifact[selection]
        age_window = age[selection]
        allowed_window = allowed[selection]
        valid_window = valid[selection]
        artifact_scored = np.isfinite(artifact_raw[selection])
        values.extend(
            (
                float(np.mean(p_window)),
                float(np.std(p_window)),
                float(np.min(p_window)),
                float(np.max(p_window)),
                float(np.mean(event_window)),
                float(np.std(event_window)),
                float(np.min(event_window)),
                float(np.max(event_window)),
                float(np.mean(artifact_window)),
                float(np.std(artifact_window)),
                float(np.min(artifact_window)),
                float(np.max(artifact_window)),
                float(np.mean(age_window)),
                float(np.std(age_window)),
                float(np.max(age_window) - np.min(age_window)),
                float(
                    np.mean(
                        allowed_window
                        & (p_window >= float(candidate_threshold))
                    )
                ),
                float(
                    np.mean(
                        event_window >= float(event_threshold)
                    )
                ),
                float(np.mean(artifact_scored)),
                float(np.mean(allowed_window)),
                float(np.mean(valid_window)),
            )
        )
    row = np.asarray(values, dtype=np.float32)
    if row.shape != (len(FEATURE_NAMES),):
        raise RuntimeError(
            f"Score-shape width differs: {row.shape} "
            f"!= {(len(FEATURE_NAMES),)}"
        )
    if not np.isfinite(row).all():
        raise RuntimeError("Score-shape features are non-finite")
    return row


@dataclass(frozen=True)
class LinearScoreShapeCalibrator:
    feature_names: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray
    intercept: float
    threshold: float

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "LinearScoreShapeCalibrator":
        feature_names = tuple(str(x) for x in payload["feature_names"])
        if feature_names != FEATURE_NAMES:
            raise ValueError("Calibrator feature identity differs")
        width = len(feature_names)
        mean = np.asarray(payload["mean"], dtype=np.float64)
        scale = np.asarray(payload["scale"], dtype=np.float64)
        coefficients = np.asarray(
            payload["coefficients"], dtype=np.float64
        )
        if any(
            value.shape != (width,)
            for value in (mean, scale, coefficients)
        ):
            raise ValueError("Calibrator vector width differs")
        if (
            not np.isfinite(mean).all()
            or not np.isfinite(scale).all()
            or not np.isfinite(coefficients).all()
            or np.any(scale <= 0.0)
        ):
            raise ValueError("Calibrator vectors are invalid")
        threshold = float(payload["threshold"])
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Calibrator threshold is outside [0, 1]")
        return cls(
            feature_names=feature_names,
            mean=mean,
            scale=scale,
            coefficients=coefficients,
            intercept=float(payload["intercept"]),
            threshold=threshold,
        )

    @classmethod
    def load(
        cls, path: str | Path
    ) -> "LinearScoreShapeCalibrator":
        return cls.from_dict(
            json.loads(Path(path).read_text(encoding="utf-8"))
        )

    def predict_logit(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float64)
        if values.shape[-1] != len(self.feature_names):
            raise ValueError("Calibrator feature width differs")
        normalized = (values - self.mean) / self.scale
        return normalized @ self.coefficients + self.intercept

    def predict_probability(self, features: np.ndarray) -> np.ndarray:
        logits = self.predict_logit(features)
        positive = logits >= 0.0
        result = np.empty_like(logits, dtype=np.float64)
        result[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
        exponent = np.exp(logits[~positive])
        result[~positive] = exponent / (1.0 + exponent)
        return result

    def accepts(self, features: np.ndarray) -> np.ndarray:
        return self.predict_probability(features) >= self.threshold


def calibrated_run_completions(
    packet_indices: np.ndarray,
    probabilities: np.ndarray,
    *,
    threshold: float,
    persistence_packets: int,
    cooldown_packets: int,
) -> list[int]:
    """Emit completion packets from thresholded contiguous accepted runs."""
    indices = np.asarray(packet_indices, dtype=np.int64)
    scores = np.asarray(probabilities, dtype=np.float64)
    if indices.shape != scores.shape:
        raise ValueError("Packet indices and probabilities differ")
    if len(indices) and np.any(np.diff(indices) <= 0):
        raise ValueError("Packet indices must be strictly increasing")
    if persistence_packets < 1 or cooldown_packets < 0:
        raise ValueError("Persistence/cooldown is invalid")
    accepted = indices[scores >= float(threshold)]
    if not len(accepted):
        return []
    split = np.flatnonzero(np.diff(accepted) != 1) + 1
    runs = np.split(accepted, split)
    completions = []
    last_emitted = -10**12
    for run in runs:
        if len(run) < int(persistence_packets):
            continue
        completion = int(run[int(persistence_packets) - 1])
        if completion - last_emitted < int(cooldown_packets):
            continue
        completions.append(completion)
        last_emitted = completion
    return completions


def self_test() -> None:
    packets = 20
    p = np.linspace(0.1, 0.9, packets, dtype=np.float32)
    event = np.linspace(0.2, 0.95, packets, dtype=np.float32)
    artifact = np.full(packets, np.inf, dtype=np.float32)
    artifact[10:] = np.linspace(0.8, 0.1, 10, dtype=np.float32)
    age = np.linspace(0.0, 1.0, packets, dtype=np.float32)
    allowed = np.ones(packets, dtype=bool)
    valid = np.ones(packets, dtype=bool)
    row = score_shape_feature_row(
        p_probability=p,
        event_probability=event,
        artifact_probability=artifact,
        arrival_age_fraction=age,
        candidate_allowed=allowed,
        quality_valid=valid,
        packet_index=12,
        candidate_threshold=0.5,
        event_threshold=0.8,
    )
    model = LinearScoreShapeCalibrator(
        feature_names=FEATURE_NAMES,
        mean=np.zeros(len(FEATURE_NAMES)),
        scale=np.ones(len(FEATURE_NAMES)),
        coefficients=np.zeros(len(FEATURE_NAMES)),
        intercept=0.0,
        threshold=0.5,
    )
    if float(model.predict_probability(row[None, :])[0]) != 0.5:
        raise RuntimeError("Calibrator probability selftest failed")
    completions = calibrated_run_completions(
        np.asarray([1, 2, 3, 7, 8, 12]),
        np.asarray([0.8, 0.9, 0.2, 0.9, 0.9, 0.9]),
        threshold=0.5,
        persistence_packets=2,
        cooldown_packets=4,
    )
    if completions != [2, 8]:
        raise RuntimeError("Calibrated emission selftest failed")
    print(
        json.dumps(
            {
                "decision": "PASS_V21_SCORE_SHAPE_SELFTEST",
                "feature_count": len(FEATURE_NAMES),
                "completion_packets": completions,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    self_test()


__all__ = [
    "FEATURE_NAMES",
    "LinearScoreShapeCalibrator",
    "calibrated_run_completions",
    "score_shape_feature_row",
    "self_test",
]
