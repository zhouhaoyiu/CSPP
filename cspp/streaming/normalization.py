"""Packet normalization using the preceding EWMA scale."""

from __future__ import annotations

import numpy as np


def normalize_packet_ewma_causal(
    wave_chunk: np.ndarray,
    running_stats: dict | None = None,
    valid_samples: int | None = None,
    half_life_packets: float = 20.0,
    clip: float = 12.0,
) -> tuple[np.ndarray, dict]:
    """Normalize one packet with a robust scale estimated before the packet.

    The previous EWMA scale is used for the current packet, so an onset keeps
    its amplitude contrast. The current packet updates only future scale.
    Statistics are shared across channels to preserve Z/H amplitude ratios.
    """
    chunk = np.asarray(wave_chunk, dtype=np.float64)
    if chunk.ndim != 2 or chunk.shape[0] != 3:
        raise ValueError(f"Expected wave_chunk shape (3, L), got {chunk.shape}")
    if half_life_packets <= 0:
        raise ValueError("half_life_packets must be positive")

    n_valid = chunk.shape[1] if valid_samples is None else int(valid_samples)
    n_valid = max(0, min(n_valid, chunk.shape[1]))
    if n_valid == 0:
        state = dict(running_stats or {"rms_sq": 1.0, "count": 0})
        return np.zeros_like(chunk, dtype=np.float32), state

    valid = chunk[:, :n_valid]
    centered = chunk - valid.mean(axis=1, keepdims=True)
    valid_centered = centered[:, :n_valid]

    # Winsorized RMS is stable under isolated telemetry spikes while retaining
    # more amplitude information than a per-packet standard deviation.
    abs_values = np.abs(valid_centered).reshape(-1)
    cap = float(np.quantile(abs_values, 0.95))
    cap = max(cap, 1e-8)
    robust_energy = float(np.mean(np.minimum(valid_centered**2, cap**2)))
    robust_energy = max(robust_energy, 1e-16)

    state = dict(running_stats or {})
    initialized = int(state.get("count", 0)) > 0
    previous_energy = max(float(state.get("rms_sq", robust_energy)), 1e-16)
    scale_energy = previous_energy if initialized else robust_energy

    normalized = centered / (np.sqrt(scale_energy) + 1e-8)
    if n_valid < chunk.shape[1]:
        normalized[:, n_valid:] = 0.0
    normalized = np.clip(normalized, -clip, clip)

    alpha = 1.0 - 0.5 ** (1.0 / half_life_packets)
    if initialized:
        # A single bad packet may raise the running scale gradually, but it
        # cannot erase the contrast of the following packets immediately.
        update_energy = min(robust_energy, previous_energy * 16.0)
        next_energy = (1.0 - alpha) * previous_energy + alpha * update_energy
    else:
        next_energy = robust_energy
    state["rms_sq"] = float(max(next_energy, 1e-16))
    state["count"] = int(state.get("count", 0)) + n_valid
    return normalized.astype(np.float32), state
