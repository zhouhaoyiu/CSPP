"""Released SeisBench full-record scoring without external prefiltering."""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np
from obspy import Stream, Trace, UTCDateTime
import torch


SAMPLE_RATE_HZ = 100.0
CHANNELS = ("HNZ", "HNN", "HNE")
EDGE_GAP_MAX_SAMPLES = 100
EDGE_GAP_MAX_FRACTION = 0.001
EDGE_GAP_FILL_VALUE = 0.0


def released_annotation_preprocessing_contract(
    spec: Mapping[str, Any],
) -> dict[str, Any]:
    """Describe the one-owner preprocessing contract for a public model."""
    filter_spec = spec.get("trailing_window_filter")
    if filter_spec is None:
        filter_args: list[str] = []
        filter_kwargs: dict[str, Any] = {}
    else:
        if str(filter_spec.get("type")) != "butterworth_bandpass":
            raise RuntimeError("Unsupported released-model filter contract")
        filter_args = ["bandpass"]
        filter_kwargs = {
            "freqmin": float(filter_spec["freqmin_hz"]),
            "freqmax": float(filter_spec["freqmax_hz"]),
            "corners": int(filter_spec["corners"]),
            "zerophase": bool(filter_spec["zerophase"]),
        }
    return {
        "interface": "SeisBench WaveformModel.annotate",
        "input_scope": "one_complete_180_second_record",
        "preprocessing_owner": "seisbench_model_annotate_stream_pre",
        "external_prefilter": False,
        "annotate_calls_per_record": 1,
        "model_filter_args": filter_args,
        "model_filter_kwargs": filter_kwargs,
        "expected_model_filter_applications_per_record": int(bool(filter_args)),
        "temporal_access": (
            "full_record_zero_phase_filter_uses_future_samples"
            if bool(filter_kwargs.get("zerophase"))
            else "full_record_annotation"
        ),
    }


def _normalised_filter_args(model: torch.nn.Module) -> list[str]:
    value = getattr(model, "filter_args", None)
    if value is None:
        return []
    return [str(item) for item in value]


def _normalised_filter_kwargs(model: torch.nn.Module) -> dict[str, Any]:
    value = getattr(model, "filter_kwargs", None)
    if value is None:
        return {}
    output = dict(value)
    for key in ("freqmin", "freqmax"):
        if key in output:
            output[key] = float(output[key])
    if "corners" in output:
        output["corners"] = int(output["corners"])
    if "zerophase" in output:
        output["zerophase"] = bool(output["zerophase"])
    return output


def validate_released_annotation_preprocessing(
    model: torch.nn.Module,
    spec: Mapping[str, Any],
) -> dict[str, Any]:
    """Fail closed if model-owned filtering differs from the frozen contract."""
    expected = spec.get("released_annotation_preprocessing")
    if expected is None:
        raise RuntimeError("Released annotation preprocessing is not frozen")
    canonical = released_annotation_preprocessing_contract(spec)
    if dict(expected) != canonical:
        raise RuntimeError("Released annotation preprocessing contract differs")
    actual_args = _normalised_filter_args(model)
    actual_kwargs = _normalised_filter_kwargs(model)
    if (
        actual_args != canonical["model_filter_args"]
        or actual_kwargs != canonical["model_filter_kwargs"]
    ):
        raise RuntimeError(
            "Loaded model-owned filter differs from frozen contract: "
            f"expected=({canonical['model_filter_args']}, "
            f"{canonical['model_filter_kwargs']}), "
            f"actual=({actual_args}, {actual_kwargs})"
        )
    return canonical


def fill_small_edge_gaps(
    wave: np.ndarray,
    *,
    max_edge_samples: int = EDGE_GAP_MAX_SAMPLES,
    max_fraction: float = EDGE_GAP_MAX_FRACTION,
    fill_value: float = EDGE_GAP_FILL_VALUE,
) -> tuple[np.ndarray, int]:
    """Fill only sparse nonfinite samples confined to record edges."""
    values = np.asarray(wave, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] != 3 or values.shape[1] == 0:
        raise ValueError(f"Expected nonempty ZNE wave shape (3, N), got {values.shape}")
    if max_edge_samples < 1 or max_edge_samples * 2 >= values.shape[1]:
        raise ValueError("Edge-gap sample limit is invalid")
    if not 0.0 < max_fraction <= 1.0 or not math.isfinite(max_fraction):
        raise ValueError("Edge-gap fraction limit is invalid")
    if not math.isfinite(fill_value):
        raise ValueError("Edge-gap fill value must be finite")
    nonfinite = ~np.isfinite(values)
    count = int(nonfinite.sum())
    if count == 0:
        return values, 0
    sample_indices = np.flatnonzero(nonfinite.any(axis=0))
    edge_only = np.logical_or(
        sample_indices < max_edge_samples,
        sample_indices >= values.shape[1] - max_edge_samples,
    )
    fraction = count / values.size
    if not bool(edge_only.all()) or fraction > max_fraction:
        raise ValueError(
            "Released full-record waveform has nonfinite samples outside "
            "the frozen sparse-edge remediation contract"
        )
    output = values.copy()
    output[nonfinite] = np.float32(fill_value)
    if not np.isfinite(output).all():
        raise RuntimeError("Edge-gap remediation did not produce a finite waveform")
    return output, count


def _raw_stream(
    wave: np.ndarray,
    *,
    start_time_epoch: float,
    station: str,
) -> Stream:
    values = np.asarray(wave, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] != 3:
        raise ValueError(f"Expected ZNE wave shape (3, N), got {values.shape}")
    if values.shape[1] == 0 or not np.isfinite(values).all():
        raise ValueError("Released full-record waveform must be finite and nonempty")
    stream = Stream()
    for component, channel in zip(values, CHANNELS, strict=True):
        trace = Trace(np.ascontiguousarray(component))
        trace.stats.starttime = UTCDateTime(float(start_time_epoch))
        trace.stats.sampling_rate = SAMPLE_RATE_HZ
        trace.stats.network = "XX"
        trace.stats.station = str(station)
        trace.stats.location = ""
        trace.stats.channel = channel
        stream += trace
    return stream


def score_released_full_record(
    wave: np.ndarray,
    spec: Mapping[str, Any],
    model: torch.nn.Module,
    *,
    start_time_epoch: float = 0.0,
    station: str = "V27",
) -> dict[str, list[dict[str, Any]]]:
    """Call the released annotation interface once on the unfiltered record."""
    validate_released_annotation_preprocessing(model, spec)
    stream = _raw_stream(
        wave,
        start_time_epoch=start_time_epoch,
        station=station,
    )
    annotations = model.annotate(
        stream,
        copy=True,
        batch_size=int(spec["execution_batch_size"]),
        overlap=int(spec["native_overlap_samples"]),
        blinding=tuple(int(value) for value in spec["blinding_samples"]),
        strict=True,
        flexible_horizontal_components=False,
    )
    p_annotations = annotations.select(channel=f"{spec['class']}_P")
    if len(p_annotations) != 1:
        raise RuntimeError(
            f"Expected one public P annotation, got {len(p_annotations)}"
        )
    detection_annotations = annotations.select(channel=f"{spec['class']}_Detection")
    if len(detection_annotations) > 1:
        raise RuntimeError("Expected at most one detection annotation")

    by_method: dict[str, list[dict[str, Any]]] = {}
    for method in spec["operating_points"]:
        picks = model.picks_from_annotations(
            p_annotations,
            float(method["p_threshold"]),
            "P",
        )
        candidates = []
        for pick in picks:
            estimated = float(pick.peak_time) - float(start_time_epoch)
            detection_probability = None
            if detection_annotations:
                detection_trace = detection_annotations[0]
                sample = int(
                    round(
                        (pick.peak_time - detection_trace.stats.starttime)
                        * detection_trace.stats.sampling_rate
                    )
                )
                if 0 <= sample < len(detection_trace.data):
                    detection_probability = float(detection_trace.data[sample])
            detection_threshold = method.get("detection_threshold")
            if detection_threshold is not None and (
                detection_probability is None
                or detection_probability < float(detection_threshold)
            ):
                continue
            candidates.append(
                {
                    "estimated_p_epoch": estimated,
                    "available_epoch": math.nan,
                    "p_probability": float(pick.peak_value),
                    "detection_probability": detection_probability,
                }
            )
        by_method[str(method["id"])] = candidates
    return by_method
