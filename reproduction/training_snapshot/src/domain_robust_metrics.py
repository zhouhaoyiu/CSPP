"""Frozen-threshold metrics for causal first-fire evaluation."""
from __future__ import annotations

import math

import numpy as np
import torch

from config import CHUNK_SEC


@torch.inference_mode()
def collect_predictions(model, loader, device: torch.device) -> dict:
    model.eval()
    probabilities = []
    targets = []
    events = []
    valid_lengths = []
    sources: list[str] = []
    snr_db = []
    for batch in loader:
        chunks = batch["chunks"].to(device)
        logits_p, _ = model(chunks)
        probabilities.append(torch.sigmoid(logits_p).cpu().numpy())
        targets.append(batch["target_chunk"].numpy())
        events.append(batch["is_event"].numpy())
        valid_lengths.append(batch["mask"].sum(dim=1).long().numpy())
        sources.extend(batch["source"])
        snr_db.append(batch["snr_db"].numpy())
    if not probabilities:
        return {
            "probabilities": np.empty((0, 0), dtype=np.float32),
            "targets": np.empty(0, dtype=np.int64),
            "events": np.empty(0, dtype=bool),
            "valid_lengths": np.empty(0, dtype=np.int64),
            "sources": np.empty(0, dtype=object),
            "snr_db": np.empty(0, dtype=np.float32),
        }
    return {
        "probabilities": np.concatenate(probabilities),
        "targets": np.concatenate(targets),
        "events": np.concatenate(events).astype(bool),
        "valid_lengths": np.concatenate(valid_lengths),
        "sources": np.asarray(sources, dtype=object),
        "snr_db": np.concatenate(snr_db),
    }


def subset_predictions(predictions: dict, mask: np.ndarray) -> dict:
    mask = np.asarray(mask, dtype=bool)
    return {key: value[mask] for key, value in predictions.items()}


def evaluate_threshold(
    predictions: dict,
    threshold: float,
    tolerance_chunks: int = 3,
) -> dict:
    probs = predictions["probabilities"]
    targets = predictions["targets"]
    events = predictions["events"]
    valid_lengths = predictions["valid_lengths"]
    hits = early = late = missed = detected = 0
    errors = []
    noise_false = 0
    noise_duration_hours = 0.0

    for index in range(len(events)):
        valid = int(valid_lengths[index])
        crossings = np.flatnonzero(probs[index, :valid] >= threshold)
        first = int(crossings[0]) if len(crossings) else None
        if events[index]:
            target = int(targets[index])
            if first is None:
                missed += 1
                continue
            detected += 1
            error = first - target
            errors.append(error)
            if abs(error) <= tolerance_chunks:
                hits += 1
            elif error < -tolerance_chunks:
                early += 1
            else:
                late += 1
        else:
            noise_duration_hours += valid * CHUNK_SEC / 3600.0
            if first is not None:
                noise_false += 1

    event_count = int(events.sum())
    noise_count = int((~events).sum())
    usable = hits / event_count if event_count else math.nan
    detection = detected / event_count if event_count else math.nan
    specificity = 1.0 - noise_false / noise_count if noise_count else math.nan
    if event_count and noise_count:
        score = 2.0 * usable * specificity / max(usable + specificity, 1e-12)
    elif event_count:
        score = usable
    else:
        score = specificity
    return {
        "threshold": float(threshold),
        "score": float(score),
        "event_n": event_count,
        "event_detected": detected,
        "event_hits": hits,
        "event_early": early,
        "event_late": late,
        "event_missed": missed,
        "detection_rate": float(detection),
        "usable_rate": float(usable),
        "median_error_s": float(np.median(errors) * CHUNK_SEC) if errors else math.nan,
        "median_abs_error_s": float(np.median(np.abs(errors)) * CHUNK_SEC) if errors else math.nan,
        "noise_n": noise_count,
        "noise_false_traces": noise_false,
        "noise_specificity": float(specificity),
        "noise_duration_hours": float(noise_duration_hours),
        "false_fires_per_hour": float(noise_false / noise_duration_hours) if noise_duration_hours else math.nan,
    }


def select_threshold(
    predictions: dict,
    thresholds: np.ndarray | None = None,
    tolerance_chunks: int = 3,
) -> tuple[float, dict, list[dict]]:
    if thresholds is None:
        thresholds = np.arange(0.10, 0.951, 0.025)
    sweep = [
        evaluate_threshold(predictions, float(threshold), tolerance_chunks)
        for threshold in thresholds
    ]
    best = max(
        sweep,
        key=lambda row: (
            row["score"],
            row["usable_rate"],
            row["noise_specificity"],
            -abs(row["threshold"] - 0.55),
        ),
    )
    return float(best["threshold"]), best, sweep


__all__ = [
    "collect_predictions",
    "evaluate_threshold",
    "select_threshold",
    "subset_predictions",
]
