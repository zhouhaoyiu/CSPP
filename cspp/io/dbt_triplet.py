"""Stateful assembly of DB/T 114 single-channel continuous triplets."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

from cspp.io.dbt_packet import (
    DBT114WaveformHeader,
    PACKET_SEQUENCE_MODULUS,
    parse_waveform_header,
    packet_sequence_from_prefix,
)


@dataclass(frozen=True)
class DBT114SequenceGap:
    expected: int
    actual: int


class DBT114SequenceTracker:
    """Track the unified 29-bit sequence of data-transfer packets."""

    def __init__(self) -> None:
        self.last_sequence: int | None = None

    def reset(self) -> None:
        self.last_sequence = None

    def observe(self, packet: bytes) -> DBT114SequenceGap | None:
        actual = packet_sequence_from_prefix(packet)
        gap = None
        if self.last_sequence is not None:
            expected = (self.last_sequence + 1) % PACKET_SEQUENCE_MODULUS
            if actual != expected:
                gap = DBT114SequenceGap(expected=expected, actual=actual)
        self.last_sequence = actual
        return gap


@dataclass(frozen=True)
class DBT114Triplet:
    station_id: str
    location: str
    start_header_hex: str
    sample_count: int
    sample_rate_factor: int
    sample_rate_multiplier: int
    records_zne: tuple[bytes, bytes, bytes]
    sequences_zne: tuple[int, int, int]

    @property
    def body(self) -> bytes:
        return b"".join(self.records_zne)


class DBT114TripletAssembler:
    """Group one Z, N, and E ``wc`` record without guessing orientation."""

    def __init__(self, *, maximum_pending_groups: int = 64):
        if maximum_pending_groups < 1:
            raise ValueError("maximum_pending_groups must be positive")
        self.maximum_pending_groups = int(maximum_pending_groups)
        self._pending: OrderedDict[
            tuple[str, str, bytes, int, int, int],
            dict[str, tuple[bytes, DBT114WaveformHeader]],
        ] = OrderedDict()

    @property
    def pending_group_count(self) -> int:
        return len(self._pending)

    @property
    def pending_station_ids(self) -> set[str]:
        return {key[0] for key in self._pending}

    def clear(self) -> None:
        self._pending.clear()

    def feed(
        self,
        record: bytes,
        *,
        header: DBT114WaveformHeader | None = None,
    ) -> DBT114Triplet | None:
        parsed = header or parse_waveform_header(record)
        if parsed.identifier != "wc":
            raise ValueError("triplet assembler accepts wc packets only")
        if parsed.merged_three_component_encoding:
            raise NotImplementedError(
                "DB/T 114 merged three-component records are unsupported"
            )
        component = (
            parsed.channel[-1]
            if parsed.channel and parsed.channel[-1] in "ZNE"
            else None
        )
        if component is None:
            raise ValueError("single-channel record requires channel suffix Z, N, or E")
        key = (
            parsed.physical_station_id,
            parsed.location,
            record[20:30],
            parsed.sample_count,
            parsed.sample_rate_factor,
            parsed.sample_rate_multiplier,
        )
        if key not in self._pending:
            if len(self._pending) >= self.maximum_pending_groups:
                self.clear()
                raise BufferError("DB/T 114 incomplete triplet buffer limit exceeded")
            self._pending[key] = {}
        group = self._pending[key]
        if component in group:
            self._pending.pop(key, None)
            raise ValueError(f"DB/T 114 triplet repeats component {component}")
        group[component] = (record, parsed)
        self._pending.move_to_end(key)
        if set(group) != {"Z", "N", "E"}:
            return None

        self._pending.pop(key)
        ordered = tuple(group[item] for item in "ZNE")
        return DBT114Triplet(
            station_id=parsed.physical_station_id,
            location=parsed.location,
            start_header_hex=record[20:30].hex(),
            sample_count=parsed.sample_count,
            sample_rate_factor=parsed.sample_rate_factor,
            sample_rate_multiplier=parsed.sample_rate_multiplier,
            records_zne=tuple(item[0] for item in ordered),
            sequences_zne=tuple(item[1].packet_sequence for item in ordered),
        )


__all__ = [
    "DBT114SequenceGap",
    "DBT114SequenceTracker",
    "DBT114Triplet",
    "DBT114TripletAssembler",
]
