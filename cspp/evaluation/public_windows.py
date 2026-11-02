"""Causal publication interface for frozen PhaseNet and EQTransformer baselines."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np
from obspy.signal.filter import bandpass
from obspy.signal.trigger import trigger_onset
import torch

from cspp.streaming.quality import PacketQualityConfig, PacketQualityGate


SAMPLE_RATE_HZ = 100.0
SAMPLES_PER_PACKET = 50
PACKET_DURATION_SECONDS = 0.5


@dataclass(frozen=True)
class PublicAnnotation:
    """First-available public-model annotations and their causal clocks."""

    p_probability: np.ndarray
    detection_probability: np.ndarray | None
    available_packet: np.ndarray
    published: np.ndarray
    diagnostics: dict[str, Any]

    def packet_publication(self) -> np.ndarray:
        if len(self.published) % SAMPLES_PER_PACKET:
            raise ValueError("Published samples do not fill packet slots")
        return self.published.reshape(-1, SAMPLES_PER_PACKET).all(axis=1)


def replay_packet_quality(
    wave: np.ndarray,
    *,
    start_epoch: float,
    warmup_packets: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Replay the frozen packet-quality state machine without model scoring."""
    values = np.asarray(wave)
    if values.ndim != 2 or values.shape[0] != 3:
        raise ValueError(f"Expected ZNE wave shape (3, N), got {values.shape}")
    if values.shape[1] % SAMPLES_PER_PACKET:
        raise ValueError("Waveform length is not an exact packet multiple")
    packet_count = values.shape[1] // SAMPLES_PER_PACKET
    gate = PacketQualityGate(PacketQualityConfig(warmup_packets=int(warmup_packets)))
    valid = np.zeros(packet_count, dtype=bool)
    candidate_allowed = np.zeros(packet_count, dtype=bool)
    reason_counts: dict[str, int] = {}
    flag_counts: dict[str, int] = {}
    for packet_index in range(packet_count):
        first = packet_index * SAMPLES_PER_PACKET
        decision = gate.evaluate(
            values[:, first : first + SAMPLES_PER_PACKET],
            float(start_epoch) + packet_index * PACKET_DURATION_SECONDS,
        )
        valid[packet_index] = bool(decision.valid)
        candidate_allowed[packet_index] = bool(decision.candidate_allowed)
        for reason in decision.reason_codes:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        for flag in decision.quality_flags:
            flag_counts[flag] = flag_counts.get(flag, 0) + 1
    diagnostics = {
        "packets": packet_count,
        "quality_valid_packets": int(valid.sum()),
        "candidate_allowed_packets": int(candidate_allowed.sum()),
        "quality_reasons": dict(sorted(reason_counts.items())),
        "quality_flags": dict(sorted(flag_counts.items())),
    }
    return valid, candidate_allowed, diagnostics


def model_labels(model: torch.nn.Module) -> str:
    labels = getattr(model, "labels", "")
    return labels if isinstance(labels, str) else "".join(labels)


def validate_public_model(
    model: torch.nn.Module,
    spec: Mapping[str, Any],
) -> None:
    actual_version = str(
        getattr(
            model,
            "_weights_version",
            getattr(model, "weights_version", ""),
        )
    )
    expected = {
        "class": str(spec["class"]),
        "in_samples": int(spec["in_samples"]),
        "labels": str(spec["labels"]),
        "normalization": str(spec["normalization"]),
        "weights_version": str(spec["weights_version"]),
        "default_args": dict(spec["required_weight_default_args"]),
        "overlap": int(spec["native_overlap_samples"]),
    }
    actual = {
        "class": model.__class__.__name__,
        "in_samples": int(model.in_samples),
        "labels": model_labels(model),
        "normalization": str(getattr(model, "norm", "")),
        "weights_version": actual_version,
        "default_args": dict(getattr(model, "default_args", {})),
        "overlap": int(model._annotate_args["overlap"][1]),
    }
    if actual != expected:
        raise RuntimeError(
            f"Loaded public model differs from frozen specification: "
            f"expected={expected}, actual={actual}"
        )


def load_public_model(
    spec: Mapping[str, Any],
    *,
    device: torch.device,
) -> torch.nn.Module:
    import seisbench.models as sbm

    classes = {
        "PhaseNet": sbm.PhaseNet,
        "EQTransformer": sbm.EQTransformer,
    }
    model_class = classes[str(spec["class"])]
    model = model_class.from_pretrained(
        str(spec["weights_name"]),
        version_str=str(spec["weights_version"]),
        update=False,
    )
    validate_public_model(model, spec)
    model.to(device).eval()
    return model


def public_outputs(
    model: torch.nn.Module,
    batch: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    output = model(batch)
    if isinstance(output, (tuple, list)):
        if len(output) < 2:
            raise RuntimeError("Public model tuple lacks a P output")
        return output[1], output[0]

    labels = model_labels(model)
    if "P" not in labels or output.ndim != 3:
        raise RuntimeError(
            f"Unexpected public model output: labels={labels}, "
            f"shape={tuple(output.shape)}"
        )
    p_index = labels.index("P")
    if output.shape[1] <= 10 and output.shape[2] > output.shape[1]:
        return output[:, p_index, :], None
    return output[:, :, p_index], None


def preprocess_windows(
    model: torch.nn.Module,
    values: np.ndarray,
    spec: Mapping[str, Any],
    *,
    device: torch.device,
) -> torch.Tensor:
    windows = values
    filter_spec = spec.get("trailing_window_filter")
    if filter_spec is not None:
        windows = bandpass(
            windows,
            freqmin=float(filter_spec["freqmin_hz"]),
            freqmax=float(filter_spec["freqmax_hz"]),
            df=SAMPLE_RATE_HZ,
            corners=int(filter_spec["corners"]),
            zerophase=bool(filter_spec["zerophase"]),
            axis=-1,
        ).astype(np.float32, copy=False)
    tensor = torch.from_numpy(np.ascontiguousarray(windows, dtype=np.float32)).to(
        device
    )
    return model.annotate_batch_pre(tensor, {})


def native_offsets(
    sample_count: int,
    spec: Mapping[str, Any],
) -> np.ndarray:
    in_samples = int(spec["in_samples"])
    stride = int(spec["native_stride_samples"])
    if in_samples <= 0 or stride <= 0:
        raise ValueError("Native window length and stride must be positive")
    if sample_count < in_samples:
        return np.empty(0, dtype=np.int64)
    return np.arange(
        0,
        sample_count - in_samples + 1,
        stride,
        dtype=np.int64,
    )


def valid_offset_mask(
    offsets: np.ndarray,
    valid_packets: np.ndarray,
    in_samples: int,
) -> np.ndarray:
    offsets = np.asarray(offsets, dtype=np.int64)
    valid_packets = np.asarray(valid_packets, dtype=bool)
    if offsets.ndim != 1 or valid_packets.ndim != 1:
        raise ValueError("Offsets and packet validity must be one-dimensional")
    if np.any(offsets < 0) or int(in_samples) <= 0:
        raise ValueError("Offsets and input length must be nonnegative")
    packet_starts = offsets // SAMPLES_PER_PACKET
    packet_stops = (
        offsets + int(in_samples) + SAMPLES_PER_PACKET - 1
    ) // SAMPLES_PER_PACKET
    if len(offsets) and int(packet_stops.max()) > len(valid_packets):
        raise ValueError("Native window extends beyond packet validity mask")
    invalid_prefix = np.concatenate(
        (
            np.zeros(1, dtype=np.int64),
            np.cumsum(~valid_packets, dtype=np.int64),
        )
    )
    return (invalid_prefix[packet_stops] - invalid_prefix[packet_starts]) == 0


@torch.inference_mode()
def infer_first_available_annotation(
    model: torch.nn.Module,
    spec: Mapping[str, Any],
    wave: np.ndarray,
    valid_packets: np.ndarray,
    *,
    device: torch.device,
) -> PublicAnnotation:
    """Publish each valid annotation sample once at native-window close."""
    values = np.asarray(wave, dtype=np.float32)
    valid_packets = np.asarray(valid_packets, dtype=bool)
    if values.ndim != 2 or values.shape[0] != 3:
        raise ValueError(f"Expected ZNE wave shape (3, N), got {values.shape}")
    if values.shape[1] != len(valid_packets) * SAMPLES_PER_PACKET:
        raise ValueError("Waveform and packet-validity lengths differ")

    sample_count = values.shape[1]
    probability = np.zeros(sample_count, dtype=np.float32)
    detection = (
        np.zeros(sample_count, dtype=np.float32)
        if str(spec["class"]) == "EQTransformer"
        else None
    )
    available_packet = np.full(sample_count, -1, dtype=np.int32)
    published = np.zeros(sample_count, dtype=bool)
    offsets = native_offsets(sample_count, spec)
    offset_is_valid = valid_offset_mask(
        offsets,
        valid_packets,
        int(spec["in_samples"]),
    )
    batch_size = int(spec["execution_batch_size"])
    if batch_size <= 0:
        raise ValueError("Execution batch size must be positive")
    left_blind, right_blind = (int(value) for value in spec["blinding_samples"])
    in_samples = int(spec["in_samples"])
    if left_blind < 0 or right_blind < 0 or left_blind + right_blind >= in_samples:
        raise ValueError("Frozen blinding interval is invalid")

    published_stop = 0
    executed_windows = 0
    for batch_start in range(0, len(offsets), batch_size):
        batch_offsets = offsets[batch_start : batch_start + batch_size]
        batch_valid = offset_is_valid[batch_start : batch_start + batch_size]
        selected_offsets = batch_offsets[batch_valid]
        p_by_offset: dict[int, np.ndarray] = {}
        det_by_offset: dict[int, np.ndarray] = {}
        if len(selected_offsets):
            windows = np.stack(
                [values[:, offset : offset + in_samples] for offset in selected_offsets]
            ).astype(np.float32, copy=False)
            prepared = preprocess_windows(
                model,
                windows,
                spec,
                device=device,
            )
            p_output, det_output = public_outputs(model, prepared)
            if p_output.shape[-1] != in_samples:
                raise RuntimeError(f"Public P output length differs: {p_output.shape}")
            p_values = p_output.detach().cpu().numpy()
            det_values = (
                None if det_output is None else det_output.detach().cpu().numpy()
            )
            for index, offset in enumerate(selected_offsets):
                p_by_offset[int(offset)] = p_values[index]
                if det_values is not None:
                    det_by_offset[int(offset)] = det_values[index]
            executed_windows += len(selected_offsets)

        for offset, is_valid in zip(
            batch_offsets,
            batch_valid,
            strict=True,
        ):
            if not bool(is_valid):
                continue
            offset_int = int(offset)
            valid_start = offset_int + left_blind
            valid_stop = offset_int + in_samples - right_blind
            publish_start = max(valid_start, published_stop)
            if publish_start >= valid_stop:
                continue
            local_start = publish_start - offset_int
            local_stop = valid_stop - offset_int
            probability[publish_start:valid_stop] = p_by_offset[offset_int][
                local_start:local_stop
            ]
            if detection is not None:
                detection[publish_start:valid_stop] = det_by_offset[offset_int][
                    local_start:local_stop
                ]
            window_end = offset_int + in_samples
            window_available_packet = math.ceil(window_end / SAMPLES_PER_PACKET) - 1
            available_packet[publish_start:valid_stop] = window_available_packet
            published[publish_start:valid_stop] = True
            published_stop = valid_stop

    diagnostics = {
        "native_windows": int(len(offsets)),
        "executed_valid_windows": int(executed_windows),
        "skipped_invalid_windows": int(len(offsets) - executed_windows),
        "published_samples": int(published.sum()),
        "fully_published_packets": int(
            published.reshape(-1, SAMPLES_PER_PACKET).all(axis=1).sum()
        ),
        "future_samples_used": False,
        "zero_prefill_used": False,
    }
    return PublicAnnotation(
        p_probability=probability,
        detection_probability=detection,
        available_packet=available_packet,
        published=published,
        diagnostics=diagnostics,
    )


def published_runs(published: np.ndarray) -> list[tuple[int, int]]:
    """Return half-open contiguous runs of observed annotation samples."""
    mask = np.asarray(published, dtype=bool)
    if mask.ndim != 1:
        raise ValueError("Published-sample mask must be one-dimensional")
    padded = np.concatenate((np.array([False]), mask, np.array([False])))
    starts = np.flatnonzero(padded[1:] & ~padded[:-1])
    stops = np.flatnonzero(~padded[1:] & padded[:-1])
    return [(int(start), int(stop)) for start, stop in zip(starts, stops, strict=True)]


def extract_closed_candidates(
    annotation: PublicAnnotation,
    *,
    method: Mapping[str, Any],
    start_epoch: float,
) -> list[dict[str, Any]]:
    """Extract causally closed P episodes with fully published peak packets."""
    probability = np.asarray(annotation.p_probability)
    available_packet = np.asarray(annotation.available_packet)
    published = np.asarray(annotation.published, dtype=bool)
    detection = annotation.detection_probability
    if not (
        probability.ndim == available_packet.ndim == published.ndim == 1
        and len(probability) == len(available_packet) == len(published)
    ):
        raise ValueError("Public annotation arrays have incompatible shapes")
    if len(published) % SAMPLES_PER_PACKET:
        raise ValueError("Published samples do not fill packet slots")
    if detection is not None:
        detection = np.asarray(detection)
        if detection.shape != probability.shape:
            raise ValueError("Detection annotation shape differs")

    threshold = float(method["p_threshold"])
    if not 0.0 < threshold <= 1.0:
        raise ValueError("P threshold must be in (0, 1]")
    release_threshold = threshold / 2.0
    packet_publication = published.reshape(-1, SAMPLES_PER_PACKET).all(axis=1)
    retained: list[dict[str, Any]] = []

    for run_start, run_stop in published_runs(published):
        run_values = probability[run_start:run_stop]
        triggers = trigger_onset(
            run_values,
            threshold,
            release_threshold,
        )
        for local_start, local_end in triggers:
            closure_local = int(local_end) + 1
            if (
                closure_local >= len(run_values)
                or float(run_values[closure_local]) >= release_threshold
            ):
                continue
            first = run_start + int(local_start)
            last = run_start + int(local_end)
            closure_sample = run_start + closure_local
            segment = probability[first : last + 1]
            if not len(segment):
                continue
            peak_sample = first + int(np.argmax(segment))
            peak_packet = peak_sample // SAMPLES_PER_PACKET
            if not bool(packet_publication[peak_packet]):
                continue
            episode_availability = int(
                available_packet[first : closure_sample + 1].max()
            )
            if episode_availability < 0:
                raise RuntimeError("Closed public episode lacks an availability clock")
            detection_probability = (
                None if detection is None else float(detection[peak_sample])
            )
            detection_threshold = method.get("detection_threshold")
            if detection_threshold is not None and (
                detection_probability is None
                or detection_probability < float(detection_threshold)
            ):
                continue

            retained.append(
                {
                    "episode_index": len(retained) + 1,
                    "episode_start_sample": first,
                    "episode_end_sample": last,
                    "closure_sample": closure_sample,
                    "peak_sample": peak_sample,
                    "peak_packet": peak_packet,
                    "packet_index": episode_availability,
                    "packet_start_epoch": (
                        float(start_epoch)
                        + episode_availability * PACKET_DURATION_SECONDS
                    ),
                    "available_epoch": (
                        float(start_epoch)
                        + (episode_availability + 1) * PACKET_DURATION_SECONDS
                    ),
                    "estimated_p_epoch": (
                        float(start_epoch) + peak_sample / SAMPLE_RATE_HZ
                    ),
                    "p_probability": float(probability[peak_sample]),
                    "detection_probability": detection_probability,
                }
            )
    return retained


def candidate_exposure_packets(
    annotation: PublicAnnotation,
    candidate_allowed: np.ndarray,
) -> np.ndarray:
    allowed = np.asarray(candidate_allowed, dtype=bool)
    publication = annotation.packet_publication()
    if allowed.shape != publication.shape:
        raise ValueError("Candidate allowance and publication lengths differ")
    return publication & allowed


__all__ = [
    "PACKET_DURATION_SECONDS",
    "SAMPLE_RATE_HZ",
    "SAMPLES_PER_PACKET",
    "PublicAnnotation",
    "candidate_exposure_packets",
    "extract_closed_candidates",
    "infer_first_available_annotation",
    "load_public_model",
    "model_labels",
    "native_offsets",
    "preprocess_windows",
    "public_outputs",
    "published_runs",
    "replay_packet_quality",
    "valid_offset_mask",
    "validate_public_model",
]
