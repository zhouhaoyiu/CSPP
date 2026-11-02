"""Frozen-threshold timing and false-candidate metrics for v5."""
from __future__ import annotations

import math

import numpy as np
import torch

from config import CHUNK_SEC


@torch.inference_mode()
def collect_predictions_v5(model, loader, device: torch.device) -> dict:
    model.eval()
    probabilities = []
    event_probabilities = []
    offsets = []
    targets = []
    target_offsets = []
    events = []
    valid_lengths = []
    masks = []
    trace_indices = []
    sources: list[str] = []
    snr_db = []
    achieved_snr_db = []
    empirical_snr_db = []
    for batch in loader:
        chunks = batch["chunks"].to(device)
        logits_p, logits_event, logits_offset = model(chunks)
        probabilities.append(torch.sigmoid(logits_p).cpu().numpy())
        event_probabilities.append(torch.sigmoid(logits_event).cpu().numpy())
        offsets.append(torch.sigmoid(logits_offset).cpu().numpy())
        targets.append(batch["target_chunk"].numpy())
        target_offsets.append(batch["target_offset"].numpy())
        events.append(batch["is_event"].numpy())
        positions = torch.arange(
            1,
            batch["mask"].shape[1] + 1,
            dtype=torch.long,
        ).unsqueeze(0)
        valid_lengths.append(
            (batch["mask"].bool() * positions).amax(dim=1).numpy()
        )
        masks.append(batch["mask"].numpy())
        trace_indices.append(batch["trace_index"].numpy())
        sources.extend(batch["source"])
        snr_db.append(batch["snr_db"].numpy())
        achieved_snr_db.append(batch["achieved_snr_db"].numpy())
        empirical_snr_db.append(batch["empirical_snr_db"].numpy())
    if not probabilities:
        return {
            "probabilities": np.empty((0, 0), dtype=np.float32),
            "event_probabilities": np.empty((0, 0), dtype=np.float32),
            "offsets": np.empty((0, 0), dtype=np.float32),
            "targets": np.empty(0, dtype=np.int64),
            "target_offsets": np.empty(0, dtype=np.float32),
            "events": np.empty(0, dtype=bool),
            "valid_lengths": np.empty(0, dtype=np.int64),
            "masks": np.empty((0, 0), dtype=np.float32),
            "trace_indices": np.empty(0, dtype=np.int64),
            "sources": np.empty(0, dtype=object),
            "snr_db": np.empty(0, dtype=np.float32),
            "achieved_snr_db": np.empty(0, dtype=np.float32),
            "empirical_snr_db": np.empty(0, dtype=np.float32),
        }
    return {
        "probabilities": np.concatenate(probabilities),
        "event_probabilities": np.concatenate(event_probabilities),
        "offsets": np.concatenate(offsets),
        "targets": np.concatenate(targets),
        "target_offsets": np.concatenate(target_offsets),
        "events": np.concatenate(events).astype(bool),
        "valid_lengths": np.concatenate(valid_lengths),
        "masks": np.concatenate(masks),
        "trace_indices": np.concatenate(trace_indices),
        "sources": np.asarray(sources, dtype=object),
        "snr_db": np.concatenate(snr_db),
        "achieved_snr_db": np.concatenate(achieved_snr_db),
        "empirical_snr_db": np.concatenate(empirical_snr_db),
    }


def subset_predictions_v5(predictions: dict, mask: np.ndarray) -> dict:
    mask = np.asarray(mask, dtype=bool)
    return {key: value[mask] for key, value in predictions.items()}


def candidate_active_mask_v5(
    probabilities: np.ndarray,
    event_probabilities: np.ndarray | None,
    valid_length: int,
    threshold: float,
    event_threshold: float,
    candidate_history_packets: int,
    warmup_packets: int,
) -> np.ndarray:
    """Apply the frozen packet policy to one uninterrupted causal run."""
    if candidate_history_packets < 1:
        raise ValueError("candidate_history_packets must be positive")
    if warmup_packets < 0:
        raise ValueError("warmup_packets cannot be negative")
    valid = int(valid_length)
    first_allowed = max(warmup_packets - 1, 0)
    candidate_hits = np.asarray(probabilities[:valid]) >= threshold
    candidate_hits[: min(first_allowed, valid)] = False
    support = np.zeros(valid, dtype=bool)
    for packet_index in range(valid):
        history_start = max(first_allowed, packet_index - candidate_history_packets + 1)
        if history_start <= packet_index:
            support[packet_index] = candidate_hits[history_start : packet_index + 1].any()
    if event_probabilities is not None:
        support &= np.asarray(event_probabilities[:valid]) >= event_threshold
    support[: min(first_allowed, valid)] = False
    return support


def evaluate_threshold_v5(
    predictions: dict,
    threshold: float,
    event_threshold: float = 0.0,
    tolerance_seconds: float = 1.5,
    candidate_history_packets: int = 1,
    warmup_packets: int = 0,
    timing_mode: str = "packet_offset",
    max_arrival_age_packets: float = 4.0,
) -> dict:
    if candidate_history_packets < 1:
        raise ValueError("candidate_history_packets must be positive")
    if warmup_packets < 0:
        raise ValueError("warmup_packets cannot be negative")
    if timing_mode not in {"packet_offset", "arrival_age"}:
        raise ValueError(f"Unknown timing mode: {timing_mode}")
    probs = predictions["probabilities"]
    event_probs = predictions.get("event_probabilities")
    offsets = predictions["offsets"]
    targets = predictions["targets"]
    target_offsets = predictions["target_offsets"]
    events = predictions["events"]
    valid_lengths = predictions["valid_lengths"]
    packet_masks = predictions.get("masks")
    hits = early = late = missed = detected = 0
    timing_errors = []
    availability_delays = []
    noise_false = 0
    noise_duration_hours = 0.0

    for index in range(len(events)):
        if packet_masks is None:
            valid = int(valid_lengths[index])
            supervised = np.ones(valid, dtype=bool)
        else:
            supervised_full = np.asarray(packet_masks[index], dtype=bool)
            supervised_indices = np.flatnonzero(supervised_full)
            valid = int(supervised_indices[-1] + 1) if len(supervised_indices) else 0
            supervised = supervised_full[:valid]
        active = candidate_active_mask_v5(
            probs[index],
            None if event_probs is None else event_probs[index],
            valid,
            threshold,
            event_threshold,
            candidate_history_packets,
            warmup_packets,
        )
        crossings = np.flatnonzero(active & supervised)
        first = int(crossings[0]) if len(crossings) else None
        if events[index]:
            if first is None:
                missed += 1
                continue
            detected += 1
            actual_position = float(targets[index]) + float(target_offsets[index])
            if timing_mode == "arrival_age":
                estimated_position = float(first + 1) - (
                    float(offsets[index, first]) * max_arrival_age_packets
                )
            else:
                estimated_position = float(first) + float(offsets[index, first])
            error_seconds = (estimated_position - actual_position) * CHUNK_SEC
            availability_seconds = (float(first) + 1.0 - actual_position) * CHUNK_SEC
            timing_errors.append(error_seconds)
            availability_delays.append(availability_seconds)
            if abs(error_seconds) <= tolerance_seconds:
                hits += 1
            elif error_seconds < -tolerance_seconds:
                early += 1
            else:
                late += 1
        else:
            first_allowed = max(warmup_packets - 1, 0)
            noise_duration_hours += float(supervised[first_allowed:].sum()) * (
                CHUNK_SEC / 3600.0
            )
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
    errors = np.asarray(timing_errors, dtype=np.float64)
    availability = np.asarray(availability_delays, dtype=np.float64)
    return {
        "threshold": float(threshold),
        "event_threshold": float(event_threshold),
        "candidate_history_packets": int(candidate_history_packets),
        "warmup_packets": int(warmup_packets),
        "timing_mode": timing_mode,
        "score": float(score),
        "event_n": event_count,
        "event_detected": detected,
        "event_hits": hits,
        "event_early": early,
        "event_late": late,
        "event_missed": missed,
        "detection_rate": float(detection),
        "usable_rate": float(usable),
        "median_error_s": float(np.median(errors)) if len(errors) else math.nan,
        "median_abs_error_s": float(np.median(np.abs(errors))) if len(errors) else math.nan,
        "p90_abs_error_s": float(np.percentile(np.abs(errors), 90)) if len(errors) else math.nan,
        "median_availability_delay_s": float(np.median(availability)) if len(availability) else math.nan,
        "p90_availability_delay_s": float(np.percentile(availability, 90)) if len(availability) else math.nan,
        "noise_n": noise_count,
        "noise_false_traces": noise_false,
        "noise_specificity": float(specificity),
        "noise_duration_hours": float(noise_duration_hours),
        "false_positive_traces_per_hour_proxy": (
            float(noise_false / noise_duration_hours) if noise_duration_hours else math.nan
        ),
    }


def select_threshold_v5(
    predictions: dict,
    thresholds: np.ndarray | None = None,
    tolerance_seconds: float = 1.5,
) -> tuple[float, dict, list[dict]]:
    if thresholds is None:
        thresholds = np.arange(0.10, 0.951, 0.025)
    sweep = [
        evaluate_threshold_v5(
            predictions,
            float(threshold),
            tolerance_seconds=tolerance_seconds,
        )
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
    "candidate_active_mask_v5",
    "collect_predictions_v5",
    "evaluate_threshold_v5",
    "select_threshold_v5",
    "subset_predictions_v5",
]
