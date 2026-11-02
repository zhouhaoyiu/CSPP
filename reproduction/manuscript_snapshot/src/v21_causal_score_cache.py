"""Offline V21 calibration and emission over frozen causal score traces."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np

from v20c_causal_score_cache import CausalScoreCache, active_runs
from v21_score_shape_calibrator import (
    LinearScoreShapeCalibrator,
    ROLLING_WINDOWS,
    score_shape_feature_row,
)


@dataclass(frozen=True)
class V21CalibratedCache:
    calibrator_probability: np.ndarray
    score_shape_eligible: np.ndarray
    diagnostics: dict[str, Any]


def _required(runtime: Mapping[str, Any], name: str) -> Any:
    if name not in runtime:
        raise ValueError(f"V21 runtime contract is missing {name}")
    return runtime[name]


def calibrate_score_cache(
    cache: CausalScoreCache,
    *,
    runtime: Mapping[str, Any],
    calibrator: LinearScoreShapeCalibrator,
) -> V21CalibratedCache:
    """Apply the calibrator with exact valid-segment history resets."""
    lengths = {
        len(cache.p_probability),
        len(cache.event_probability),
        len(cache.arrival_age_fraction),
        len(cache.artifact_probability),
        len(cache.quality_valid),
        len(cache.candidate_allowed),
    }
    if len(lengths) != 1:
        raise ValueError("V20c score-cache arrays have different lengths")
    packet_count = lengths.pop()
    history_packets = int(_required(runtime, "score_history_packets"))
    if history_packets != max(ROLLING_WINDOWS):
        raise ValueError("V21 score-history length differs")
    if not math.isclose(
        float(_required(runtime, "calibrator_threshold")),
        calibrator.threshold,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("V21 calibrator threshold differs")

    eligible = np.asarray(
        np.isfinite(cache.artifact_probability)
        & np.asarray(cache.quality_valid, dtype=bool),
        dtype=bool,
    )
    probabilities = np.full(packet_count, np.nan, dtype=np.float64)
    feature_rows = 0
    valid_starts, valid_ends = active_runs(cache.quality_valid)
    for first, last in zip(valid_starts, valid_ends, strict=True):
        segment_first = int(first)
        segment_last = int(last)
        for packet_index in np.flatnonzero(
            eligible[segment_first : segment_last + 1]
        ):
            absolute = segment_first + int(packet_index)
            history_first = max(
                segment_first, absolute - history_packets + 1
            )
            history = slice(history_first, absolute + 1)
            features = score_shape_feature_row(
                p_probability=cache.p_probability[history],
                event_probability=cache.event_probability[history],
                artifact_probability=cache.artifact_probability[history],
                arrival_age_fraction=cache.arrival_age_fraction[history],
                candidate_allowed=cache.candidate_allowed[history],
                quality_valid=cache.quality_valid[history],
                packet_index=absolute - history_first,
                candidate_threshold=float(
                    _required(runtime, "candidate_threshold")
                ),
                event_threshold=float(
                    _required(runtime, "event_threshold")
                ),
            )
            probabilities[absolute] = float(
                calibrator.predict_probability(features[None, :])[0]
            )
            feature_rows += 1
    if feature_rows != int(eligible.sum()):
        raise RuntimeError("V21 eligible feature coverage differs")
    return V21CalibratedCache(
        calibrator_probability=probabilities,
        score_shape_eligible=eligible,
        diagnostics={
            "packet_count": packet_count,
            "score_shape_eligible_packets": int(eligible.sum()),
            "calibrator_scored_packets": feature_rows,
            "valid_segments": len(valid_starts),
            "score_history_packets": history_packets,
        },
    )


def emit_v21_candidates(
    cache: CausalScoreCache,
    *,
    start_epoch: float,
    runtime: Mapping[str, Any],
    calibrator: LinearScoreShapeCalibrator,
) -> tuple[V21CalibratedCache, list[dict[str, Any]]]:
    """Emit V21 candidates with online-equivalent run and cooldown state."""
    calibrated = calibrate_score_cache(
        cache,
        runtime=runtime,
        calibrator=calibrator,
    )
    persistence = int(_required(runtime, "persistence_packets"))
    if persistence < 1:
        raise ValueError("V21 persistence must be positive")
    packet_seconds = float(_required(runtime, "packet_duration_seconds"))
    cooldown_packets = int(
        math.ceil(
            float(_required(runtime, "emission_cooldown_seconds"))
            / packet_seconds
        )
    )
    accepted = (
        calibrated.score_shape_eligible
        & (
            calibrated.calibrator_probability
            >= calibrator.threshold
        )
    )
    starts, ends = active_runs(accepted)
    age_scale_seconds = (
        float(_required(runtime, "max_arrival_age_packets"))
        * packet_seconds
    )
    candidates = []
    last_emitted = -(10**12)
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
        packet_start_epoch = float(start_epoch) + completion * packet_seconds
        available_epoch = packet_start_epoch + packet_seconds
        arrival_age = float(cache.arrival_age_fraction[completion])
        calibrated_probability = float(
            calibrated.calibrator_probability[completion]
        )
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
                "candidate_probability": calibrated_probability,
                "base_p_probability": float(
                    cache.p_probability[completion]
                ),
                "calibrator_probability": calibrated_probability,
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
    calibrated.diagnostics.update(
        {
            "accepted_packets": int(accepted.sum()),
            "accepted_runs": len(starts),
            "emitted_candidates": len(candidates),
            "persistence_packets": persistence,
            "cooldown_packets": cooldown_packets,
        }
    )
    return calibrated, candidates


def self_test() -> None:
    packet_count = 40
    cache = CausalScoreCache(
        p_probability=np.linspace(0.1, 0.9, packet_count).astype(
            np.float32
        ),
        event_probability=np.full(packet_count, 0.9, dtype=np.float32),
        arrival_age_fraction=np.full(
            packet_count, 0.5, dtype=np.float32
        ),
        artifact_probability=np.full(
            packet_count, 0.1, dtype=np.float32
        ),
        quality_valid=np.ones(packet_count, dtype=bool),
        candidate_allowed=np.ones(packet_count, dtype=bool),
        diagnostics={},
    )
    width = 93
    calibrator = LinearScoreShapeCalibrator(
        feature_names=tuple(
            __import__(
                "v21_score_shape_calibrator"
            ).FEATURE_NAMES
        ),
        mean=np.zeros(width),
        scale=np.ones(width),
        coefficients=np.zeros(width),
        intercept=20.0,
        threshold=0.95,
    )
    runtime = {
        "candidate_threshold": 0.5,
        "event_threshold": 0.8,
        "calibrator_threshold": 0.95,
        "score_history_packets": 16,
        "persistence_packets": 1,
        "packet_duration_seconds": 0.5,
        "emission_cooldown_seconds": 20.0,
        "max_arrival_age_packets": 4.0,
    }
    calibrated, candidates = emit_v21_candidates(
        cache,
        start_epoch=1000.0,
        runtime=runtime,
        calibrator=calibrator,
    )
    if len(candidates) != 1 or candidates[0]["packet_index"] != 0:
        raise RuntimeError("V21 candidate emission selftest failed")
    if int(calibrated.score_shape_eligible.sum()) != packet_count:
        raise RuntimeError("V21 calibration coverage selftest failed")
    print("V21 causal score-cache self-test: PASS")


if __name__ == "__main__":
    self_test()


__all__ = [
    "V21CalibratedCache",
    "calibrate_score_cache",
    "emit_v21_candidates",
]
