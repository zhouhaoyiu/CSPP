"""Fixed current-packet features for a constant-state causal artifact veto."""
from __future__ import annotations

import math

import numpy as np


SAMPLE_RATE_HZ = 100.0
BAND_EDGES_HZ = (1.0, 4.0, 8.0, 16.0, 30.0, 50.1)


def normalized_packet_energy(packet: np.ndarray) -> float:
    values = np.asarray(packet, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != 3:
        raise ValueError(f"Expected packet shape (3, L), got {values.shape}")
    centered = values - values.mean(axis=1, keepdims=True)
    return float(np.mean(centered**2))


def causal_packet_veto_features(
    packet: np.ndarray,
    previous_energy: float | None = None,
) -> dict[str, float]:
    """Extract scale-stable features from one arrived normalized packet."""
    values = np.asarray(packet, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != 3 or values.shape[1] < 8:
        raise ValueError(f"Expected packet shape (3, L>=8), got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("Packet contains non-finite values")

    eps = 1e-12
    samples = values.shape[1]
    centered = values - values.mean(axis=1, keepdims=True)
    absolute = np.abs(centered)
    component_energy = np.mean(centered**2, axis=1)
    total_component_energy = max(float(component_energy.sum()), eps)
    energy = max(float(np.mean(centered**2)), eps)
    rms = math.sqrt(energy)
    peak = float(absolute.max())
    positive_peak = float(np.maximum(centered, 0.0).max(initial=0.0))
    negative_peak = float(np.maximum(-centered, 0.0).max(initial=0.0))
    sample_norm = np.sqrt(np.sum(centered**2, axis=0))
    derivative = np.diff(centered, axis=1)
    zero_crossings = np.count_nonzero(centered[:, 1:] * centered[:, :-1] < 0.0)

    half = samples // 2
    quarter_edges = np.linspace(0, samples, 5, dtype=int)
    quarter_energy = np.asarray(
        [
            np.mean(centered[:, start:stop] ** 2)
            for start, stop in zip(
                quarter_edges[:-1], quarter_edges[1:], strict=True
            )
        ],
        dtype=np.float64,
    )
    quarter_fraction = quarter_energy / max(float(quarter_energy.sum()), eps)
    first_energy = float(np.mean(centered[:, :half] ** 2))
    second_energy = float(np.mean(centered[:, half:] ** 2))

    spectrum = np.abs(np.fft.rfft(centered, axis=1)) ** 2
    frequencies = np.fft.rfftfreq(samples, d=1.0 / SAMPLE_RATE_HZ)
    spectrum[:, 0] = 0.0
    total_spectrum = spectrum.sum(axis=0)
    spectral_sum = max(float(total_spectrum.sum()), eps)
    spectral_fraction = total_spectrum / spectral_sum
    nonzero = frequencies > 0.0
    entropy_values = spectral_fraction[nonzero]
    spectral_entropy = float(
        -(entropy_values * np.log(entropy_values + eps)).sum()
        / math.log(max(int(nonzero.sum()), 2))
    )
    spectral_centroid = float(
        (spectral_fraction * frequencies).sum() / (0.5 * SAMPLE_RATE_HZ)
    )
    band_energy = []
    for low, high in zip(BAND_EDGES_HZ[:-1], BAND_EDGES_HZ[1:], strict=True):
        band = (frequencies >= low) & (frequencies < high)
        band_energy.append(float(total_spectrum[band].sum()))
    band_fraction = np.asarray(band_energy, dtype=np.float64)
    band_fraction /= max(float(band_fraction.sum()), eps)

    lag_numerator = float(np.sum(centered[:, 1:] * centered[:, :-1]))
    lag_denominator = math.sqrt(
        max(float(np.sum(centered[:, 1:] ** 2)), eps)
        * max(float(np.sum(centered[:, :-1] ** 2)), eps)
    )
    edge = max(2, min(10, samples // 4))
    previous = energy if previous_energy is None else max(float(previous_energy), eps)
    features = {
        "log1p_peak": math.log1p(peak),
        "log1p_rms": math.log1p(rms),
        "crest_factor": peak / max(rms, eps),
        "width_norm_gt_2": float(np.mean(sample_norm >= 2.0)),
        "width_norm_gt_4": float(np.mean(sample_norm >= 4.0)),
        "width_norm_gt_8": float(np.mean(sample_norm >= 8.0)),
        "polarity_asymmetry": abs(
            math.log((positive_peak + eps) / (negative_peak + eps))
        ),
        "signed_area_fraction": abs(float(centered.sum()))
        / max(float(absolute.sum()), eps),
        "mean_shift_over_rms": float(np.max(np.abs(values.mean(axis=1))))
        / max(rms, eps),
        "second_half_energy_fraction": second_energy
        / max(first_energy + second_energy, eps),
        "edge_jump_over_rms": float(
            np.max(
                np.abs(
                    centered[:, -edge:].mean(axis=1)
                    - centered[:, :edge].mean(axis=1)
                )
            )
        )
        / max(rms, eps),
        "derivative_ratio": float(np.sqrt(np.mean(derivative**2)))
        / max(rms, eps),
        "zero_crossing_fraction": float(zero_crossings)
        / (3.0 * (samples - 1)),
        "max_component_energy_fraction": float(component_energy.max())
        / total_component_energy,
        "z_energy_fraction": float(component_energy[0]) / total_component_energy,
        "spectral_entropy": spectral_entropy,
        "spectral_centroid_nyquist": spectral_centroid,
        "lag1_correlation": lag_numerator / max(lag_denominator, eps),
        "log_energy_change": math.log(energy) - math.log(previous),
        "quarter_1_energy_fraction": float(quarter_fraction[0]),
        "quarter_2_energy_fraction": float(quarter_fraction[1]),
        "quarter_3_energy_fraction": float(quarter_fraction[2]),
        "quarter_4_energy_fraction": float(quarter_fraction[3]),
    }
    features.update(
        {
            f"spectral_band_{index + 1}_fraction": float(value)
            for index, value in enumerate(band_fraction)
        }
    )
    return {
        key: float(np.nan_to_num(value, nan=0.0, posinf=1e6, neginf=-1e6))
        for key, value in features.items()
    }


PHYSICS_FEATURE_NAMES = tuple(
    causal_packet_veto_features(np.zeros((3, 50), dtype=np.float32)).keys()
)


__all__ = [
    "BAND_EDGES_HZ",
    "PHYSICS_FEATURE_NAMES",
    "causal_packet_veto_features",
    "normalized_packet_energy",
]
