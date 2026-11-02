"""Assemble gappy three-component strong-motion streams without imputation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from obspy import Stream, UTCDateTime, read

from cspp.io.decimation import (
    GROUP_DELAY_SECONDS,
    NUM_TAPS,
    SOURCE_SAMPLING_RATE_HZ,
    TARGET_SAMPLING_RATE_HZ,
    causal_decimate_200_to_100,
    design_manifest,
)


TARGET_CHANNELS = ("HNZ", "HNN", "HNE")


def _select_location(stream: Stream, preferred: str) -> tuple[str, Stream]:
    locations = sorted(
        {str(trace.stats.location or "") for trace in stream},
        key=lambda value: (
            value != str(preferred or ""),
            value not in {"", "00"},
            value,
        ),
    )
    for location in locations:
        selected = stream.select(location=location)
        if all(len(selected.select(channel=channel)) for channel in TARGET_CHANNELS):
            return location, selected
    raise RuntimeError("No location contains an HNZ/HNN/HNE triplet")


def _resample(
    values: np.ndarray,
    native_rate: float,
    target_rate: float,
    *,
    decimation_phase: int = 0,
) -> np.ndarray:
    """Resample one trace without reading future source samples."""
    if np.isclose(native_rate, target_rate):
        return values.astype(np.float32, copy=False)
    if np.isclose(native_rate, SOURCE_SAMPLING_RATE_HZ) and np.isclose(
        target_rate,
        TARGET_SAMPLING_RATE_HZ,
    ):
        return causal_decimate_200_to_100(
            values,
            decimation_phase=decimation_phase,
            require_full_history=True,
        )
    raise ValueError(
        "Strictly causal stream assembly currently supports unchanged-rate "
        "input or 200-to-100 Hz FIR decimation; got "
        f"{native_rate:g}-to-{target_rate:g} Hz"
    )


def _trace_resampling_clock(
    *,
    trace_start: UTCDateTime,
    output_start: UTCDateTime,
    native_rate: float,
    target_rate: float,
) -> tuple[int, int, float]:
    """Return source parity, output index, and non-negative clock delay."""
    relative_seconds = float(trace_start) - float(output_start)
    if not np.isfinite(relative_seconds):
        raise ValueError("Trace start offset is non-finite")

    def delayed_target_index(source_time_seconds: float) -> tuple[int, float]:
        target_position = source_time_seconds * target_rate
        destination = int(np.ceil(target_position - 1e-9))
        delay = destination / target_rate - source_time_seconds
        if delay < -1e-10:
            raise RuntimeError("Output clock precedes the consumed source sample")
        return destination, max(delay, 0.0)

    if np.isclose(native_rate, target_rate):
        destination, delay = delayed_target_index(relative_seconds)
        return 0, destination, delay
    if np.isclose(native_rate, SOURCE_SAMPLING_RATE_HZ) and np.isclose(
        target_rate,
        TARGET_SAMPLING_RATE_HZ,
    ):
        candidates = []
        for phase in range(2):
            destination, delay = delayed_target_index(
                relative_seconds + phase / native_rate
            )
            candidates.append((delay, phase, destination))
        delay, phase, destination = min(candidates)
        return phase, destination, delay
    raise ValueError(
        "Unsupported strictly causal rate conversion "
        f"{native_rate:g}-to-{target_rate:g} Hz"
    )


def _resampling_metadata(
    native_sampling_rates: list[float],
    target_rate: float,
) -> dict:
    converted = any(not np.isclose(rate, target_rate) for rate in native_sampling_rates)
    metadata = {
        "resampling_strictly_causal": True,
        "resampling_future_samples_read": 0,
        "resampling_gap_state": (
            "reset_per_discontinuous_trace_group_or_nonfinite_run"
        ),
        "resampling_incomplete_history": (
            "NaN until complete FIR support is available"
        ),
        "resampling_group_delay_seconds": (GROUP_DELAY_SECONDS if converted else 0.0),
        "resampling_clock_alignment_policy": (
            "minimum_nonnegative_delay_on_target_clock"
        ),
    }
    if converted:
        metadata["resampling_method"] = design_manifest()["method"]
        metadata["resampling_fir_taps"] = NUM_TAPS
    else:
        metadata["resampling_method"] = "identity"
        metadata["resampling_fir_taps"] = 1
    return metadata


def _trace_values(trace) -> np.ndarray:
    values = np.ma.asarray(trace.data)
    if np.ma.isMaskedArray(values):
        values = values.filled(np.nan)
    return np.asarray(values, dtype=np.float32)


def _contiguous_trace_groups(
    traces,
) -> list[tuple[UTCDateTime, float, np.ndarray]]:
    """Join adjacent same-rate traces without joining gaps or overlaps."""
    groups: list[tuple[UTCDateTime, float, np.ndarray]] = []
    group_start = None
    group_rate = None
    group_values: list[np.ndarray] = []
    expected_next_start = None

    def flush() -> None:
        nonlocal group_start, group_rate, group_values
        if group_start is None or group_rate is None:
            return
        groups.append(
            (
                group_start,
                group_rate,
                (
                    group_values[0]
                    if len(group_values) == 1
                    else np.concatenate(group_values)
                ),
            )
        )
        group_start = None
        group_rate = None
        group_values = []

    for trace in traces:
        values = _trace_values(trace)
        rate = float(trace.stats.sampling_rate)
        trace_start = trace.stats.starttime
        contiguous = (
            group_start is not None
            and np.isclose(rate, group_rate)
            and expected_next_start is not None
            and abs(float(trace_start) - expected_next_start) <= 0.25 / rate
        )
        if not contiguous:
            flush()
            group_start = trace_start
            group_rate = rate
        group_values.append(values)
        expected_next_start = float(trace_start) + len(values) / rate
    flush()
    return groups


def stream_to_gappy_zne_array(
    mseed_path: Path,
    target,
    start: UTCDateTime,
    end: UTCDateTime,
    *,
    target_sample_rate_hz: float = 100.0,
    samples_per_packet: int = 50,
) -> tuple[np.ndarray, dict]:
    """Place every available segment on one clock and leave gaps as NaN."""
    raw = read(str(mseed_path))
    raw_gaps = raw.get_gaps()
    location, selected = _select_location(raw, str(target.location or ""))
    sample_count = int(
        round((float(end) - float(start)) * float(target_sample_rate_hz))
    )
    sample_count = (sample_count // int(samples_per_packet)) * int(samples_per_packet)
    if sample_count <= 0:
        raise RuntimeError("Requested stream has no complete packets")

    arrays = []
    channel_ids = []
    sampling_rates = []
    available_samples = {}
    overlap_samples = {}
    for channel in TARGET_CHANNELS:
        destination = np.full(sample_count, np.nan, dtype=np.float32)
        overlap = 0
        traces = sorted(
            selected.select(channel=channel),
            key=lambda trace: float(trace.stats.starttime),
        )
        if not traces:
            raise RuntimeError(
                f"Missing {channel} for {target.network}.{target.station}"
            )
        for trace_start, native_rate, values in _contiguous_trace_groups(traces):
            phase, destination_start, _ = _trace_resampling_clock(
                trace_start=trace_start,
                output_start=start,
                native_rate=native_rate,
                target_rate=float(target_sample_rate_hz),
            )
            resampled = _resample(
                values,
                native_rate,
                float(target_sample_rate_hz),
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
            source = resampled[source_start : source_start + count]
            target_slice = slice(destination_start, destination_start + count)
            existing = np.isfinite(destination[target_slice])
            overlap += int(existing.sum())
            finite = np.isfinite(source)
            destination_view = destination[target_slice]
            destination_view[finite] = source[finite]
        arrays.append(destination)
        channel_ids.append(str(traces[0].id))
        sampling_rates.extend(float(trace.stats.sampling_rate) for trace in traces)
        available_samples[channel] = int(np.isfinite(destination).sum())
        overlap_samples[channel] = overlap

    wave = np.stack(arrays, axis=0)
    finite_by_component = np.isfinite(wave)
    complete_samples = finite_by_component.all(axis=0)
    meta = {
        "network": str(target.network),
        "station": str(target.station),
        "location": location,
        "channels": ",".join(channel_ids),
        "start_time": str(start),
        "end_time": str(end),
        "samples": sample_count,
        "duration_hours": (sample_count / float(target_sample_rate_hz) / 3600.0),
        "native_sampling_rates": ",".join(
            f"{value:.3f}" for value in sorted(set(sampling_rates))
        ),
        "raw_trace_count": int(len(raw)),
        "raw_gap_count": int(len(raw_gaps)),
        "raw_gap_seconds": (
            float(sum(max(float(gap[6]), 0.0) for gap in raw_gaps)) if raw_gaps else 0.0
        ),
        "mseed_path": str(mseed_path),
        "latitude": target.latitude,
        "longitude": target.longitude,
        "elevation_m": target.elevation_m,
        "missing_value_representation": "NaN",
        "gap_imputation": False,
        "available_samples_by_component": available_samples,
        "overlap_samples_by_component": overlap_samples,
        "complete_three_component_samples": int(complete_samples.sum()),
        "complete_three_component_fraction": float(complete_samples.mean()),
    }
    return wave, meta


__all__ = ["stream_to_gappy_zne_array"]
