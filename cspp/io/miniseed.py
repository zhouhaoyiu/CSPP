"""Decode miniSEED while quarantining isolated corrupt fixed-size records."""

from __future__ import annotations

from dataclasses import dataclass
import io
from pathlib import Path
import warnings
from typing import Any

from obspy import Stream, read
from obspy.io.mseed import InternalMSEEDError
from obspy.io.mseed.headers import InternalMSEEDWarning
from obspy.io.mseed.util import get_record_information


MAXIMUM_QUARANTINED_RECORD_FRACTION = 0.001


class MiniSeedIntegrityWarningError(RuntimeError):
    pass


@dataclass(frozen=True)
class ResilientMiniSeedResult:
    stream: Stream
    diagnostics: dict[str, Any]


def _read_without_integrity_warnings(
    source: str | io.BytesIO,
    *,
    record_length: int,
) -> Stream:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", InternalMSEEDWarning)
        stream = read(
            source,
            format="MSEED",
            reclen=record_length,
            headonly=False,
        )
    integrity_warnings = [
        str(item.message)
        for item in caught
        if issubclass(item.category, InternalMSEEDWarning)
    ]
    if integrity_warnings:
        raise MiniSeedIntegrityWarningError(" | ".join(integrity_warnings))
    return stream


def _record_header(
    record: bytes,
    *,
    record_length: int,
) -> dict[str, Any]:
    try:
        stream = read(
            io.BytesIO(record),
            format="MSEED",
            reclen=record_length,
            headonly=True,
            details=True,
        )
    except Exception as error:
        return {"header_error": f"{type(error).__name__}: {error}"}
    if not stream:
        return {"header_error": "record decoded to no header"}
    trace = stream[0]
    return {
        "trace_id": trace.id,
        "start_time_utc": str(trace.stats.starttime),
        "sample_rate_hz": float(trace.stats.sampling_rate),
        "declared_samples": int(trace.stats.npts),
        "encoding": str(trace.stats.mseed.encoding),
    }


def read_resilient_mseed(
    path: str | Path,
    *,
    maximum_quarantined_record_fraction: float = (MAXIMUM_QUARANTINED_RECORD_FRACTION),
) -> ResilientMiniSeedResult:
    source = Path(path).resolve()
    info = get_record_information(str(source))
    record_length = int(info["record_length"])
    record_count = int(info["number_of_records"])
    excess_bytes = int(info["excess_bytes"])
    if record_length <= 0 or record_count <= 0 or excess_bytes:
        raise RuntimeError(
            f"Unsupported miniSEED record layout: {source}: "
            f"record_length={record_length}, records={record_count}, "
            f"excess_bytes={excess_bytes}"
        )
    expected_bytes = record_length * record_count
    if source.stat().st_size != expected_bytes:
        raise RuntimeError(f"miniSEED fixed-record byte count differs: {source}")

    try:
        stream = _read_without_integrity_warnings(
            str(source), record_length=record_length
        )
        return ResilientMiniSeedResult(
            stream=stream,
            diagnostics={
                "source_path": str(source),
                "decode_mode": "whole_file",
                "record_length_bytes": record_length,
                "record_count": record_count,
                "quarantined_record_count": 0,
                "quarantined_record_fraction": 0.0,
                "quarantined_records": [],
            },
        )
    except (InternalMSEEDError, MiniSeedIntegrityWarningError) as error:
        whole_file_error = f"{type(error).__name__}: {error}"

    payload = source.read_bytes()
    output = Stream()
    quarantined: list[dict[str, Any]] = []

    def decode_range(first: int, stop: int) -> None:
        first_byte = first * record_length
        stop_byte = stop * record_length
        try:
            decoded = _read_without_integrity_warnings(
                io.BytesIO(payload[first_byte:stop_byte]),
                record_length=record_length,
            )
        except (
            InternalMSEEDError,
            MiniSeedIntegrityWarningError,
        ) as error:
            if stop - first > 1:
                middle = first + (stop - first) // 2
                decode_range(first, middle)
                decode_range(middle, stop)
                return
            record = payload[first_byte:stop_byte]
            quarantined.append(
                {
                    "record_index_zero_based": first,
                    "byte_offset": first_byte,
                    "error": f"{type(error).__name__}: {error}",
                    **_record_header(record, record_length=record_length),
                }
            )
            return
        output.extend(decoded)

    decode_range(0, record_count)
    fraction = len(quarantined) / record_count
    if fraction > float(maximum_quarantined_record_fraction):
        raise RuntimeError(
            f"miniSEED corrupt-record fraction exceeds policy: {source}: "
            f"{len(quarantined)}/{record_count}={fraction:.9g}"
        )
    if not output:
        raise RuntimeError(
            f"miniSEED has no decodable records after quarantine: {source}"
        )
    return ResilientMiniSeedResult(
        stream=output,
        diagnostics={
            "source_path": str(source),
            "decode_mode": "record_quarantine",
            "record_length_bytes": record_length,
            "record_count": record_count,
            "quarantined_record_count": len(quarantined),
            "quarantined_record_fraction": fraction,
            "maximum_quarantined_record_fraction": float(
                maximum_quarantined_record_fraction
            ),
            "whole_file_error": whole_file_error,
            "quarantined_records": quarantined,
        },
    )


__all__ = [
    "MAXIMUM_QUARANTINED_RECORD_FRACTION",
    "ResilientMiniSeedResult",
    "read_resilient_mseed",
]
