"""DB/T 114-2026 waveform packet framing and single-channel decoding."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import io
import struct
from typing import Any


WAVEFORM_IDENTIFIERS = {b"wc", b"wt", b"ws"}
LENGTH_INDEX_IDENTIFIERS = {
    b"re",
    b"rr",
    b"si",
    b"ti",
    *WAVEFORM_IDENTIFIERS,
}
PACKET_SEQUENCE_MODULUS = 1 << 29
DIMENSION_NAMES = {
    0: "dimensionless",
    1: "displacement",
    2: "velocity",
    3: "acceleration",
}


def packet_length_from_index(length_index: int) -> int:
    index = int(length_index)
    if not 0 <= index <= 7:
        raise ValueError("DB/T 114 packet length index must be 0..7")
    return 32 << index


def packet_length_from_prefix(prefix: bytes | bytearray) -> int:
    if len(prefix) < 3:
        raise ValueError("DB/T 114 packet prefix requires three bytes")
    identifier = bytes(prefix[:2])
    if identifier not in LENGTH_INDEX_IDENTIFIERS:
        raise ValueError(f"unsupported DB/T 114 length-index identifier {identifier!r}")
    return packet_length_from_index(prefix[2] >> 5)


def packet_sequence_from_prefix(prefix: bytes | bytearray) -> int:
    if len(prefix) < 6:
        raise ValueError("DB/T 114 packet sequence requires six bytes")
    identifier = bytes(prefix[:2])
    if identifier not in LENGTH_INDEX_IDENTIFIERS:
        raise ValueError(f"unsupported DB/T 114 length-index identifier {identifier!r}")
    return ((prefix[2] & 0x1F) << 24) | int.from_bytes(prefix[3:6], "big")


class DBT114LengthIndexFramer:
    """Split an arbitrary TCP byte stream into length-index packets."""

    def __init__(self, *, maximum_buffer_bytes: int = 1 << 20):
        if maximum_buffer_bytes < 4096:
            raise ValueError("maximum_buffer_bytes must be at least 4096")
        self.maximum_buffer_bytes = int(maximum_buffer_bytes)
        self.buffer = bytearray()

    def reset(self) -> None:
        self.buffer.clear()

    def feed(self, data: bytes | bytearray) -> list[bytes]:
        if not data:
            return []
        self.buffer.extend(data)
        if len(self.buffer) > self.maximum_buffer_bytes:
            self.reset()
            raise ValueError("DB/T 114 TCP frame buffer limit exceeded")
        packets = []
        while len(self.buffer) >= 3:
            length = packet_length_from_prefix(self.buffer)
            if len(self.buffer) < length:
                break
            packets.append(bytes(self.buffer[:length]))
            del self.buffer[:length]
        return packets


def _ascii(packet: bytes, start: int, stop: int, name: str) -> str:
    try:
        return packet[start:stop].decode("ascii").strip()
    except UnicodeDecodeError as exc:
        raise ValueError(f"DB/T 114 {name} is not ASCII") from exc


@dataclass(frozen=True)
class DBT114WaveformHeader:
    identifier: str
    packet_length_index: int
    packet_length_bytes: int
    packet_sequence: int
    quality_indicator: str
    network: str
    station: str
    location: str
    channel: str
    sample_count: int
    sample_rate_factor: int
    sample_rate_multiplier: int
    activity_flags: int
    input_output_clock_flags: int
    data_quality_flags: int
    following_blockette_count: int
    time_correction: int
    data_offset: int
    first_blockette_offset: int
    blockette_type: int
    next_blockette_offset: int
    encoding_format: int
    data_word_order: int
    record_length_exponent: int
    channel_order: str
    dimension_code: int
    sensitivity_factor: int

    @property
    def physical_station_id(self) -> str:
        return f"{self.network}.{self.station}"

    @property
    def dimension(self) -> str:
        return DIMENSION_NAMES[self.dimension_code]

    @property
    def data_status_unavailable(self) -> bool:
        return bool(self.activity_flags & 0x80)

    @property
    def calibration_signal_present(self) -> bool:
        return bool(self.activity_flags & 0x01)

    @property
    def calibration_type(self) -> str:
        code = (self.input_output_clock_flags >> 6) & 0x03
        return {
            0: "normal",
            1: "pulse",
            2: "sine",
            3: "other",
        }[code]

    @property
    def merged_three_component_encoding(self) -> bool:
        return self.channel.endswith(".")

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "physical_station_id": self.physical_station_id,
            "dimension": self.dimension,
            "data_status_unavailable": self.data_status_unavailable,
            "calibration_signal_present": (self.calibration_signal_present),
            "calibration_type": self.calibration_type,
            "merged_three_component_encoding": (self.merged_three_component_encoding),
        }


def parse_waveform_header(packet: bytes) -> DBT114WaveformHeader:
    if len(packet) < 64:
        raise ValueError("DB/T 114 waveform packet is shorter than 64 bytes")
    identifier_bytes = packet[:2]
    if identifier_bytes not in WAVEFORM_IDENTIFIERS:
        raise ValueError(f"not a DB/T 114 waveform packet: {identifier_bytes!r}")
    length_index = packet[2] >> 5
    expected_length = packet_length_from_index(length_index)
    if len(packet) != expected_length:
        raise ValueError(
            f"DB/T 114 waveform length differs: {len(packet)} != {expected_length}"
        )
    packet_sequence = packet_sequence_from_prefix(packet)
    record_length_exponent = packet[54]
    if (1 << record_length_exponent) != expected_length:
        raise ValueError("DB/T 114 length index and blockette record length differ")
    data_offset = struct.unpack(">H", packet[44:46])[0]
    first_blockette_offset = struct.unpack(">H", packet[46:48])[0]
    blockette_type = struct.unpack(">H", packet[48:50])[0]
    if expected_length not in {256, 512}:
        raise ValueError("DB/T 114 waveform records must be 256 or 512 bytes")
    if data_offset != 64:
        raise ValueError("DB/T 114 waveform data must begin at byte 65")
    if first_blockette_offset != 48 or blockette_type != 1000:
        raise ValueError("DB/T 114 waveform requires blockette 1000 at byte 49")
    if packet[39] != 1 or struct.unpack(">H", packet[50:52])[0] != 0:
        raise ValueError("DB/T 114 waveform requires one terminal blockette 1000")
    if packet[53] != 1:
        raise ValueError("DB/T 114 waveform data word order must be big endian")
    dimension_code = packet[60] & 0x03
    sensitivity_factor = ((packet[60] >> 2) << 24) | int.from_bytes(
        packet[61:64], "big"
    )
    return DBT114WaveformHeader(
        identifier=identifier_bytes.decode("ascii"),
        packet_length_index=length_index,
        packet_length_bytes=expected_length,
        packet_sequence=packet_sequence,
        quality_indicator=_ascii(
            packet,
            6,
            7,
            "quality indicator",
        ),
        network=_ascii(packet, 18, 20, "network"),
        station=_ascii(packet, 8, 13, "station"),
        location=_ascii(packet, 13, 15, "location"),
        channel=_ascii(packet, 15, 18, "channel"),
        sample_count=struct.unpack(">H", packet[30:32])[0],
        sample_rate_factor=struct.unpack(">h", packet[32:34])[0],
        sample_rate_multiplier=struct.unpack(">h", packet[34:36])[0],
        activity_flags=packet[36],
        input_output_clock_flags=packet[37],
        data_quality_flags=packet[38],
        following_blockette_count=packet[39],
        time_correction=struct.unpack(">i", packet[40:44])[0],
        data_offset=data_offset,
        first_blockette_offset=first_blockette_offset,
        blockette_type=blockette_type,
        next_blockette_offset=struct.unpack(">H", packet[50:52])[0],
        encoding_format=packet[52],
        data_word_order=packet[53],
        record_length_exponent=record_length_exponent,
        channel_order=_ascii(packet, 56, 59, "channel order"),
        dimension_code=dimension_code,
        sensitivity_factor=sensitivity_factor,
    )


def to_standard_miniseed_record(packet: bytes) -> bytes:
    """Restore a single-channel DB/T 114 record's miniSEED sequence bytes."""
    header = parse_waveform_header(packet)
    if header.merged_three_component_encoding:
        raise NotImplementedError(
            "DB/T 114 merged three-component channel codes such as BH. "
            "require the referenced DB/T 2 merged-data decoder"
        )
    restored = bytearray(packet)
    restored[:6] = f"{header.packet_sequence % 1_000_000:06d}".encode("ascii")
    return bytes(restored)


def decode_single_channel_trace(
    packet: bytes,
) -> tuple[Any, DBT114WaveformHeader]:
    """Decode one non-merged waveform record with ObsPy."""
    from obspy import read

    header = parse_waveform_header(packet)
    record = to_standard_miniseed_record(packet)
    stream = read(io.BytesIO(record), format="MSEED")
    if len(stream) != 1:
        raise ValueError("single-channel DB/T 114 packet decoded to multiple traces")
    trace = stream[0]
    if (
        str(trace.stats.network).strip() != header.network
        or str(trace.stats.station).strip() != header.station
        or str(trace.stats.location).strip() != header.location
        or str(trace.stats.channel).strip() != header.channel
        or int(trace.stats.npts) != header.sample_count
    ):
        raise ValueError(
            "decoded miniSEED identity or sample count differs from the DB/T 114 header"
        )
    return trace, header


__all__ = [
    "DBT114LengthIndexFramer",
    "DBT114WaveformHeader",
    "DIMENSION_NAMES",
    "LENGTH_INDEX_IDENTIFIERS",
    "PACKET_SEQUENCE_MODULUS",
    "WAVEFORM_IDENTIFIERS",
    "decode_single_channel_trace",
    "packet_length_from_index",
    "packet_length_from_prefix",
    "packet_sequence_from_prefix",
    "parse_waveform_header",
    "to_standard_miniseed_record",
]
