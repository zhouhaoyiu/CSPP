"""Assemble gappy strong-motion streams without ObsPy's quadratic gap scan."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
from obspy import Stream, UTCDateTime, read
from obspy.core import compatibility

from cspp.io.strong_motion import TARGET_CHANNELS, _resample, _select_location


def stream_gap_summary(stream: Stream) -> tuple[int, float]:
    """Return ObsPy-compatible gap count and positive-gap duration.

    ``Stream.get_gaps`` scans every earlier trace when deciding whether a gap
    is already covered.  Keeping the maximum prior end time per trace id gives
    the same decision in one sorted pass.
    """
    gap_count = 0
    positive_gap_seconds = 0.0
    traces_by_id: dict[str, list] = defaultdict(list)

    for trace in stream:
        traces_by_id[trace.id].append(trace)
        if isinstance(trace.data, np.ma.masked_array):
            internal = Stream([trace]).get_gaps()
            gap_count += len(internal)
            positive_gap_seconds += sum(max(float(gap[6]), 0.0) for gap in internal)

    for traces in traces_by_id.values():
        traces.sort(
            key=lambda trace: (
                float(trace.stats.starttime),
                float(trace.stats.endtime),
            )
        )
        strict_prior_max_end = -np.inf
        same_start_prior_max_end = -np.inf
        current_start = None

        for index in range(len(traces) - 1):
            trace = traces[index]
            following = traces[index + 1]
            trace_start = float(trace.stats.starttime)
            trace_end = float(trace.stats.endtime)

            if current_start is None:
                current_start = trace_start
            elif trace_start != current_start:
                strict_prior_max_end = max(
                    strict_prior_max_end,
                    same_start_prior_max_end,
                )
                same_start_prior_max_end = -np.inf
                current_start = trace_start

            stime = min(trace_end, float(following.stats.endtime))
            etime = float(following.stats.starttime)
            delta = etime - (stime + float(trace.stats.delta))
            if delta < 0.0:
                following_duration = float(following.stats.endtime) - float(
                    following.stats.starttime
                )
                if -delta > following_duration:
                    delta = -following_duration

            nsamples = int(
                compatibility.round_away(abs(delta) * float(trace.stats.sampling_rate))
            )
            if delta < 0.0:
                nsamples = -nsamples
            if trace.stats.delta == following.stats.delta and nsamples == 0:
                same_start_prior_max_end = max(
                    same_start_prior_max_end,
                    trace_end,
                )
                continue

            prior_max_end = strict_prior_max_end
            if stime > trace_start:
                prior_max_end = max(
                    prior_max_end,
                    same_start_prior_max_end,
                )
            covered = stime < etime and prior_max_end > etime
            if not covered:
                gap_count += 1
                positive_gap_seconds += max(delta, 0.0)

            same_start_prior_max_end = max(
                same_start_prior_max_end,
                trace_end,
            )

    return gap_count, positive_gap_seconds


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
    raw_gap_count, raw_gap_seconds = stream_gap_summary(raw)
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
        for trace in traces:
            values = np.ma.asarray(trace.data)
            if np.ma.isMaskedArray(values):
                values = values.filled(np.nan)
            values = np.asarray(values, dtype=np.float32)
            native_rate = float(trace.stats.sampling_rate)
            resampled = _resample(values, native_rate, float(target_sample_rate_hz))
            destination_start = int(
                round(
                    (float(trace.stats.starttime) - float(start))
                    * float(target_sample_rate_hz)
                )
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
        "raw_gap_count": int(raw_gap_count),
        "raw_gap_seconds": float(raw_gap_seconds),
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


__all__ = ["stream_gap_summary", "stream_to_gappy_zne_array"]
