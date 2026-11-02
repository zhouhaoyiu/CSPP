"""Rebase native causal windows at each contiguous valid packet run."""
from __future__ import annotations

import numpy as np

from v20c_public_window_baselines import (
    PublicAnnotation, SAMPLES_PER_PACKET, infer_first_available_annotation,
    published_runs,
)


def infer_valid_segment_annotation(model, spec, wave, valid_packets, *, device):
    values = np.asarray(wave, dtype=np.float32)
    valid = np.asarray(valid_packets, dtype=bool)
    if valid.ndim != 1 or values.shape != (3, len(valid) * SAMPLES_PER_PACKET):
        raise ValueError("Waveform and packet validity shapes differ")
    n = values.shape[1]
    probability = np.zeros(n, dtype=np.float32)
    detection = np.zeros(n, dtype=np.float32) if spec['class'] == 'EQTransformer' else None
    published = np.zeros(n, dtype=bool)
    available = np.full(n, -1, dtype=np.int32)
    windows = 0
    segments = published_runs(valid)
    for first, stop in segments:
        a, b = first * SAMPLES_PER_PACKET, stop * SAMPLES_PER_PACKET
        result = infer_first_available_annotation(
            model, spec, values[:, a:b], np.ones(stop - first, dtype=bool), device=device,
        )
        probability[a:b] = result.p_probability
        published[a:b] = result.published
        # Preserve the absolute record clock, including discarded startup/gaps.
        available[a:b] = np.where(result.published, result.available_packet + first, -1)
        if detection is not None:
            detection[a:b] = result.detection_probability
        windows += result.diagnostics['executed_valid_windows']
    return PublicAnnotation(probability, detection, available, published, {
        'native_windows': windows, 'executed_valid_windows': windows,
        'skipped_invalid_windows': 0, 'valid_segments': len(segments),
        'segment_start_packets': [int(a) for a, _ in segments],
        'published_samples': int(published.sum()),
        'fully_published_packets': int(published.reshape(-1, SAMPLES_PER_PACKET).all(axis=1).sum()),
        'window_origin': 'first packet of each contiguous valid segment',
        'future_samples_used': False, 'zero_prefill_used': False,
    })
