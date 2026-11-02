"""Assemble arbitrarily oriented strong-motion channels on a gappy ZNE clock."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from obspy import Stream, UTCDateTime
from obspy.signal.rotate import rotate2zne

from cspp.io.strong_motion import (
    _contiguous_trace_groups,
    _resample,
    _resampling_metadata,
    _trace_resampling_clock,
)
from cspp.io.gaps import stream_gap_summary
from cspp.io.miniseed import read_resilient_mseed


@dataclass(frozen=True)
class ChannelOrientation:
    """Frozen channel orientation and sensitivity at the waveform epoch."""

    network: str
    station: str
    location: str
    channel: str
    azimuth: float
    dip: float
    scale: float
    scale_units: str

    @classmethod
    def from_value(
        cls, value: ChannelOrientation | Mapping[str, Any]
    ) -> ChannelOrientation:
        if isinstance(value, cls):
            return value
        return cls(
            network=str(value["network"]),
            station=str(value["station"]),
            location=str(value.get("location", "")),
            channel=str(value["channel"]),
            azimuth=float(value["azimuth"]),
            dip=float(value["dip"]),
            scale=float(value["scale"]),
            scale_units=str(value["scale_units"]),
        )


def orientation_matrix(
    orientations: Sequence[ChannelOrientation],
) -> np.ndarray:
    """Return native-component unit vectors expressed in the ZNE basis."""
    rows = []
    for item in orientations:
        dip = np.deg2rad(float(item.dip))
        azimuth = np.deg2rad(float(item.azimuth))
        rows.append(
            [
                -np.sin(dip),
                np.cos(azimuth) * np.cos(dip),
                np.sin(azimuth) * np.cos(dip),
            ]
        )
    return np.asarray(rows, dtype=np.float64)


def _target_value(target: Any, name: str) -> Any:
    if isinstance(target, Mapping):
        return target[name]
    return getattr(target, name)


def _read_sources(
    source: str | Path | Stream | Sequence[str | Path],
) -> tuple[Stream, list[str], list[dict[str, Any]]]:
    if isinstance(source, Stream):
        return (
            source.copy(),
            ["<in-memory Stream>"],
            [{"decode_mode": "in_memory_stream"}],
        )
    if isinstance(source, (str, Path)):
        paths = [Path(source)]
    else:
        paths = [Path(path) for path in source]
    if not paths:
        raise ValueError("At least one waveform source is required")
    stream = Stream()
    diagnostics = []
    for path in paths:
        result = read_resilient_mseed(path)
        stream.extend(result.stream)
        diagnostics.append(result.diagnostics)
    return (
        stream,
        [str(path.resolve()) for path in paths],
        diagnostics,
    )


def _validate_orientations(
    values: Sequence[ChannelOrientation | Mapping[str, Any]],
    *,
    network: str,
    station: str,
    location: str,
) -> tuple[list[ChannelOrientation], float, float]:
    orientations = [ChannelOrientation.from_value(value) for value in values]
    if len(orientations) != 3:
        raise ValueError(
            f"Exactly three channel orientations are required, got {len(orientations)}"
        )
    channels = [item.channel for item in orientations]
    if len(set(channels)) != 3:
        raise ValueError(f"Channel orientations must be unique: {channels}")
    identities = {(item.network, item.station, item.location) for item in orientations}
    expected = (network, station, location)
    if identities != {expected}:
        raise ValueError(
            f"Orientation identity {sorted(identities)} does not match "
            f"target {expected}"
        )
    units = {item.scale_units for item in orientations}
    if len(units) != 1:
        raise ValueError(f"Channel scale units differ: {sorted(units)}")
    if any(not np.isfinite(item.scale) or item.scale <= 0.0 for item in orientations):
        raise ValueError("Channel scales must be finite and positive")
    if any(
        not np.isfinite(item.azimuth) or not np.isfinite(item.dip)
        for item in orientations
    ):
        raise ValueError("Channel azimuths and dips must be finite")

    matrix = orientation_matrix(orientations)
    determinant = float(np.linalg.det(matrix))
    condition_number = float(np.linalg.cond(matrix))
    if abs(determinant) <= 1e-6 or not np.isfinite(condition_number):
        raise ValueError(
            "Channel orientations are not linearly independent: "
            f"determinant={determinant:.12g}, "
            f"condition_number={condition_number:.12g}"
        )
    return orientations, determinant, condition_number


def _assemble_native_component(
    traces: Stream,
    *,
    channel: str,
    start: UTCDateTime,
    sample_count: int,
    target_sample_rate_hz: float,
) -> tuple[np.ndarray, int, list[float], list[float]]:
    destination = np.full(sample_count, np.nan, dtype=np.float32)
    overlap_samples = 0
    sampling_rates = []
    clock_alignment_delays = []
    ordered = sorted(
        traces.select(channel=channel),
        key=lambda trace: (
            float(trace.stats.starttime),
            float(trace.stats.endtime),
        ),
    )
    if not ordered:
        raise RuntimeError(f"Missing native channel {channel}")

    for trace_start, native_rate, values in _contiguous_trace_groups(ordered):
        sampling_rates.append(native_rate)
        (
            phase,
            destination_start,
            clock_alignment_delay,
        ) = _trace_resampling_clock(
            trace_start=trace_start,
            output_start=start,
            native_rate=native_rate,
            target_rate=target_sample_rate_hz,
        )
        clock_alignment_delays.append(clock_alignment_delay)
        resampled = _resample(
            values,
            native_rate,
            target_sample_rate_hz,
            decimation_phase=phase,
        )
        source_start = max(-destination_start, 0)
        destination_start = max(destination_start, 0)
        count = min(
            len(resampled) - source_start,
            sample_count - destination_start,
        )
        if count <= 0:
            continue
        source_values = resampled[source_start : source_start + count]
        destination_slice = slice(destination_start, destination_start + count)
        destination_view = destination[destination_slice]
        finite = np.isfinite(source_values)
        overlap_samples += int((np.isfinite(destination_view) & finite).sum())
        destination_view[finite] = source_values[finite]
    return (
        destination,
        overlap_samples,
        sampling_rates,
        clock_alignment_delays,
    )


def stream_to_oriented_gappy_zne_array(
    source: str | Path | Stream | Sequence[str | Path],
    target: Any,
    orientations: Sequence[ChannelOrientation | Mapping[str, Any]],
    start: UTCDateTime,
    end: UTCDateTime,
    *,
    target_sample_rate_hz: float = 100.0,
    samples_per_packet: int = 50,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Place native traces on one clock, calibrate gains, and rotate to ZNE.

    The response ``scale`` is treated as counts per ``scale_units``. Native
    samples are divided by their channel scales before rotation, then all ZNE
    outputs are multiplied by the geometric-mean scale. This corrects relative
    component gains while retaining the count-like amplitude range expected by
    the existing causally normalized model.
    """
    network = str(_target_value(target, "network"))
    station = str(_target_value(target, "station"))
    location = str(_target_value(target, "location") or "")
    checked, determinant, condition_number = _validate_orientations(
        orientations,
        network=network,
        station=station,
        location=location,
    )

    target_sample_rate_hz = float(target_sample_rate_hz)
    samples_per_packet = int(samples_per_packet)
    if target_sample_rate_hz <= 0.0:
        raise ValueError("target_sample_rate_hz must be positive")
    if samples_per_packet <= 0:
        raise ValueError("samples_per_packet must be positive")
    sample_count = int(round((float(end) - float(start)) * target_sample_rate_hz))
    sample_count = (sample_count // samples_per_packet) * samples_per_packet
    if sample_count <= 0:
        raise RuntimeError("Requested stream has no complete packets")

    raw, source_paths, source_decode = _read_sources(source)
    selected = raw.select(
        network=network,
        station=station,
        location=location,
    )
    if not selected:
        raise RuntimeError(f"No traces found for {network}.{station}.{location}")
    raw_gap_count, raw_gap_seconds = stream_gap_summary(selected.copy())

    native_arrays = []
    overlap_by_channel = {}
    available_by_channel = {}
    sampling_rates = []
    clock_alignment_delays = []
    for item in checked:
        array, overlap, rates, alignment_delays = _assemble_native_component(
            selected,
            channel=item.channel,
            start=start,
            sample_count=sample_count,
            target_sample_rate_hz=target_sample_rate_hz,
        )
        native_arrays.append(array)
        overlap_by_channel[item.channel] = overlap
        available_by_channel[item.channel] = int(np.isfinite(array).sum())
        sampling_rates.extend(rates)
        clock_alignment_delays.extend(alignment_delays)

    scales = np.asarray([item.scale for item in checked], dtype=np.float64)
    reference_scale = float(np.exp(np.log(scales).mean()))
    calibrated = [
        array.astype(np.float64, copy=False) * reference_scale / item.scale
        for array, item in zip(native_arrays, checked, strict=True)
    ]
    z, n, e = rotate2zne(
        calibrated[0],
        checked[0].azimuth,
        checked[0].dip,
        calibrated[1],
        checked[1].azimuth,
        checked[1].dip,
        calibrated[2],
        checked[2].azimuth,
        checked[2].dip,
    )
    wave = np.stack([z, n, e], axis=0).astype(np.float32, copy=False)
    complete_samples = np.isfinite(wave).all(axis=0)

    metadata = {
        "network": network,
        "station": station,
        "location": location,
        "output_component_order": "Z,N,E",
        "native_channels": ",".join(item.channel for item in checked),
        "channel_orientations": [asdict(item) for item in checked],
        "orientation_matrix_determinant": determinant,
        "orientation_matrix_condition_number": condition_number,
        "scale_units": checked[0].scale_units,
        "reference_scale": reference_scale,
        "gain_correction": ("native_counts / channel_scale * geometric_mean_scale"),
        "start_time": str(start),
        "end_time": str(end),
        "samples": sample_count,
        "duration_hours": (sample_count / target_sample_rate_hz / 3600.0),
        "target_sample_rate_hz": target_sample_rate_hz,
        "native_sampling_rates": ",".join(
            f"{value:.3f}" for value in sorted(set(sampling_rates))
        ),
        "selected_raw_trace_count": int(len(selected)),
        "selected_raw_gap_count": int(raw_gap_count),
        "selected_raw_gap_seconds": float(raw_gap_seconds),
        "source_paths": source_paths,
        "source_decode": source_decode,
        "missing_value_representation": "NaN",
        "gap_imputation": False,
        "available_samples_by_native_channel": available_by_channel,
        "overlap_samples_by_native_channel": overlap_by_channel,
        "complete_three_component_samples": int(complete_samples.sum()),
        "complete_three_component_fraction": float(complete_samples.mean()),
    }
    for name in ("latitude", "longitude", "elevation_m"):
        try:
            metadata[name] = _target_value(target, name)
        except (AttributeError, KeyError):
            pass
    metadata.update(
        _resampling_metadata(
            sampling_rates,
            target_sample_rate_hz,
        )
    )
    metadata["resampling_max_clock_alignment_delay_seconds"] = float(
        max(clock_alignment_delays, default=0.0)
    )
    return wave, metadata


__all__ = [
    "ChannelOrientation",
    "orientation_matrix",
    "stream_to_oriented_gappy_zne_array",
]
