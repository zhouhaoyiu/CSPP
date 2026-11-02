"""Model-independent local P-window SNR diagnostics for V20c."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


DEFINITION_ID = "local_p_excess_power_1s_v1"
SNR_BIN_LT_0 = "lt_0_db"
SNR_BIN_0_TO_10 = "0_to_lt_10_db"
SNR_BIN_GE_10 = "ge_10_db"
SNR_BIN_INVALID = "invalid"
SNR_FLOOR_DB = -120.0


def snr_bin(snr_db: float) -> str:
    """Map a finite SNR value to the frozen reporting bins."""
    value = float(snr_db)
    if not math.isfinite(value):
        raise ValueError("SNR must be finite")
    if value < 0.0:
        return SNR_BIN_LT_0
    if value < 10.0:
        return SNR_BIN_0_TO_10
    return SNR_BIN_GE_10


def _invalid_result(
    *,
    reason: str,
    p_sample: int,
    sample_rate_hz: float,
    window_seconds: float,
    window_samples: int,
    pre_complete_samples: int,
    post_complete_samples: int,
) -> dict[str, Any]:
    return {
        "definition_id": DEFINITION_ID,
        "valid": False,
        "invalid_reason": reason,
        "p_sample": int(p_sample),
        "sample_rate_hz": float(sample_rate_hz),
        "window_seconds": float(window_seconds),
        "window_samples": int(window_samples),
        "pre_complete_samples": int(pre_complete_samples),
        "post_complete_samples": int(post_complete_samples),
        "noise_power": None,
        "post_power": None,
        "excess_signal_power": None,
        "excess_power_censored": None,
        "snr_db": None,
        "snr_bin": SNR_BIN_INVALID,
    }


def local_p_excess_power_snr(
    wave: np.ndarray,
    *,
    p_sample: int,
    sample_rate_hz: float,
    window_seconds: float = 1.0,
) -> dict[str, Any]:
    """Estimate local P SNR from equal pre-P and post-P windows.

    Component means are estimated from the pre-P window. Noise power is the
    mean squared pre-P residual over all three components. Signal power is the
    nonnegative excess of post-P power over noise power. The reported ratio is
    ``10 log10(excess signal power / noise power)``. Nonpositive excess power
    is left-censored at -120 dB. Any missing sample invalidates the record.
    """
    values = np.asarray(wave)
    if values.ndim != 2 or values.shape[0] != 3:
        raise ValueError("wave must have shape (3, samples)")
    sample_rate_hz = float(sample_rate_hz)
    window_seconds = float(window_seconds)
    p_sample = int(p_sample)
    if sample_rate_hz <= 0.0 or not math.isfinite(sample_rate_hz):
        raise ValueError("sample_rate_hz must be finite and positive")
    if window_seconds <= 0.0 or not math.isfinite(window_seconds):
        raise ValueError("window_seconds must be finite and positive")
    window_samples = max(1, int(round(window_seconds * sample_rate_hz)))
    pre_start = p_sample - window_samples
    post_end = p_sample + window_samples
    if pre_start < 0 or post_end > values.shape[1]:
        return _invalid_result(
            reason="reference_window_outside_waveform",
            p_sample=p_sample,
            sample_rate_hz=sample_rate_hz,
            window_seconds=window_seconds,
            window_samples=window_samples,
            pre_complete_samples=0,
            post_complete_samples=0,
        )

    pre = values[:, pre_start:p_sample].astype(np.float64, copy=False)
    post = values[:, p_sample:post_end].astype(np.float64, copy=False)
    pre_complete = np.isfinite(pre).all(axis=0)
    post_complete = np.isfinite(post).all(axis=0)
    pre_count = int(pre_complete.sum())
    post_count = int(post_complete.sum())
    if pre_count != window_samples or post_count != window_samples:
        return _invalid_result(
            reason="missing_sample_in_reference_window",
            p_sample=p_sample,
            sample_rate_hz=sample_rate_hz,
            window_seconds=window_seconds,
            window_samples=window_samples,
            pre_complete_samples=pre_count,
            post_complete_samples=post_count,
        )

    baseline = pre.mean(axis=1, keepdims=True)
    noise_power = float(np.mean((pre - baseline) ** 2))
    post_power = float(np.mean((post - baseline) ** 2))
    if not math.isfinite(noise_power) or noise_power <= 0.0:
        return _invalid_result(
            reason="nonpositive_pre_p_noise_power",
            p_sample=p_sample,
            sample_rate_hz=sample_rate_hz,
            window_seconds=window_seconds,
            window_samples=window_samples,
            pre_complete_samples=pre_count,
            post_complete_samples=post_count,
        )
    if not math.isfinite(post_power):
        return _invalid_result(
            reason="nonfinite_post_p_power",
            p_sample=p_sample,
            sample_rate_hz=sample_rate_hz,
            window_seconds=window_seconds,
            window_samples=window_samples,
            pre_complete_samples=pre_count,
            post_complete_samples=post_count,
        )

    excess = max(post_power - noise_power, 0.0)
    censored = excess <= 0.0
    ratio = max(excess / noise_power, 10.0 ** (SNR_FLOOR_DB / 10.0))
    value = float(10.0 * math.log10(ratio))
    return {
        "definition_id": DEFINITION_ID,
        "valid": True,
        "invalid_reason": "",
        "p_sample": p_sample,
        "sample_rate_hz": sample_rate_hz,
        "window_seconds": window_seconds,
        "window_samples": window_samples,
        "pre_complete_samples": pre_count,
        "post_complete_samples": post_count,
        "noise_power": noise_power,
        "post_power": post_power,
        "excess_signal_power": excess,
        "excess_power_censored": censored,
        "snr_db": value,
        "snr_bin": snr_bin(value),
    }


__all__ = [
    "DEFINITION_ID",
    "SNR_BIN_0_TO_10",
    "SNR_BIN_GE_10",
    "SNR_BIN_INVALID",
    "SNR_BIN_LT_0",
    "SNR_FLOOR_DB",
    "local_p_excess_power_snr",
    "snr_bin",
]
