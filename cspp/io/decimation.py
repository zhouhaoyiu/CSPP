"""Fixed-delay causal 200-to-100 Hz FIR decimation for streaming data."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


SOURCE_SAMPLING_RATE_HZ = 200.0
TARGET_SAMPLING_RATE_HZ = 100.0
SOURCE_PACKET_SAMPLES = 100
TARGET_PACKET_SAMPLES = 50
DECIMATION_FACTOR = 2
NUM_TAPS = 31
CUTOFF_HZ = 40.0
MAX_PROCESS_CHUNK_SAMPLES = 1_000_000


def _design_coefficients() -> np.ndarray:
    """Return a normalized symmetric Hamming-windowed sinc low-pass FIR."""
    center = (NUM_TAPS - 1) / 2.0
    positions = np.arange(NUM_TAPS, dtype=np.float64) - center
    normalized_cutoff = CUTOFF_HZ / SOURCE_SAMPLING_RATE_HZ
    coefficients = (
        2.0
        * normalized_cutoff
        * np.sinc(2.0 * normalized_cutoff * positions)
        * np.hamming(NUM_TAPS)
    )
    coefficients /= coefficients.sum()
    return coefficients


FIR_COEFFICIENTS = _design_coefficients()
GROUP_DELAY_SOURCE_SAMPLES = (NUM_TAPS - 1) // 2
GROUP_DELAY_SECONDS = GROUP_DELAY_SOURCE_SAMPLES / SOURCE_SAMPLING_RATE_HZ


@dataclass
class CausalFirDecimator200To100:
    """Stateful causal FIR filtering followed by fixed-phase decimation.

    ``decimation_phase`` is the source-sample parity retained in the output.
    Chunk boundaries do not change that phase or reset the FIR history.
    """

    state: np.ndarray = field(
        default_factory=lambda: np.zeros((3, NUM_TAPS - 1), dtype=np.float64)
    )
    decimation_phase: int = 0
    source_samples_seen: int = 0

    def reset(self) -> None:
        self.state.fill(0.0)
        self.source_samples_seen = 0

    @property
    def component_count(self) -> int:
        return int(self.state.shape[0])

    def __post_init__(self) -> None:
        self.state = np.asarray(self.state, dtype=np.float64)
        if (
            self.state.ndim != 2
            or self.state.shape[1] != NUM_TAPS - 1
            or self.state.shape[0] <= 0
        ):
            raise ValueError(
                "state must have shape (components, "
                f"{NUM_TAPS - 1}), got {self.state.shape}"
            )
        self.decimation_phase = int(self.decimation_phase)
        if self.decimation_phase not in range(DECIMATION_FACTOR):
            raise ValueError(
                "decimation_phase must be in "
                f"[0, {DECIMATION_FACTOR}), got {self.decimation_phase}"
            )
        self.source_samples_seen = int(self.source_samples_seen)
        if self.source_samples_seen < 0:
            raise ValueError("source_samples_seen must be non-negative")

    @classmethod
    def for_components(
        cls,
        component_count: int,
        *,
        decimation_phase: int = 0,
        source_samples_seen: int = 0,
    ) -> CausalFirDecimator200To100:
        component_count = int(component_count)
        if component_count <= 0:
            raise ValueError("component_count must be positive")
        return cls(
            state=np.zeros(
                (component_count, NUM_TAPS - 1),
                dtype=np.float64,
            ),
            decimation_phase=decimation_phase,
            source_samples_seen=source_samples_seen,
        )

    def output_source_indices(self, chunk_samples: int) -> np.ndarray:
        """Return source indices emitted by the next chunk."""
        chunk_samples = int(chunk_samples)
        if chunk_samples < 0:
            raise ValueError("chunk_samples must be non-negative")
        indices = self.source_samples_seen + np.arange(
            chunk_samples,
            dtype=np.int64,
        )
        return indices[indices % DECIMATION_FACTOR == self.decimation_phase]

    def process_chunk(self, chunk: np.ndarray) -> np.ndarray:
        """Process an arbitrary non-empty source chunk without resetting state."""
        values = np.asarray(chunk, dtype=np.float64)
        if values.ndim == 1:
            values = values[np.newaxis, :]
        if values.ndim != 2 or values.shape[0] != self.component_count:
            raise ValueError(
                "Expected a chunk with shape "
                f"({self.component_count}, samples), got {values.shape}"
            )
        if values.shape[1] == 0:
            return np.empty((self.component_count, 0), dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError("Source chunk contains non-finite values")

        output_indices = self.output_source_indices(values.shape[1])
        combined = np.concatenate([self.state, values], axis=1)
        windows = np.lib.stride_tricks.sliding_window_view(
            combined,
            NUM_TAPS,
            axis=1,
        )
        filtered = np.einsum(
            "ctk,k->ct",
            windows,
            FIR_COEFFICIENTS[::-1],
            optimize=True,
        )
        local_indices = output_indices - self.source_samples_seen
        self.state = combined[:, -(NUM_TAPS - 1) :].copy()
        self.source_samples_seen += values.shape[1]
        return filtered[:, local_indices].astype(np.float32)

    def process_packet(self, packet: np.ndarray) -> np.ndarray:
        values = np.asarray(packet, dtype=np.float64)
        if values.shape != (3, SOURCE_PACKET_SAMPLES):
            raise ValueError(
                "Expected one complete 200 Hz packet with shape "
                f"(3, {SOURCE_PACKET_SAMPLES}), got {values.shape}"
            )
        if not np.isfinite(values).all():
            raise ValueError("Source packet contains non-finite values")
        converted = self.process_chunk(values)
        if converted.shape != (3, TARGET_PACKET_SAMPLES):
            raise RuntimeError(
                f"FIR decimator produced unexpected shape {converted.shape}"
            )
        return converted


def causal_decimate_200_to_100(
    values: np.ndarray,
    *,
    decimation_phase: int = 0,
    require_full_history: bool = True,
) -> np.ndarray:
    """Causally decimate one trace while preserving gaps as NaN.

    FIR state is reset at every non-finite run. When ``require_full_history``
    is true, the first ``NUM_TAPS - 1`` source samples after each reset are
    withheld because their complete anti-alias support is unavailable.
    """
    source = np.ma.asarray(values)
    if np.ma.isMaskedArray(source):
        source = source.filled(np.nan)
    source = np.asarray(source, dtype=np.float64)
    if source.ndim != 1:
        raise ValueError(f"Expected one source trace, got {source.shape}")
    decimation_phase = int(decimation_phase)
    if decimation_phase not in range(DECIMATION_FACTOR):
        raise ValueError(
            "decimation_phase must be in "
            f"[0, {DECIMATION_FACTOR}), got {decimation_phase}"
        )

    selected_source_indices = np.arange(
        decimation_phase,
        source.size,
        DECIMATION_FACTOR,
        dtype=np.int64,
    )
    output = np.full(
        selected_source_indices.size,
        np.nan,
        dtype=np.float32,
    )
    finite = np.isfinite(source)
    boundaries = np.flatnonzero(
        np.diff(np.concatenate(([False], finite, [False])).astype(np.int8))
    )
    for run_start, run_stop in boundaries.reshape(-1, 2):
        processor = CausalFirDecimator200To100.for_components(
            1,
            decimation_phase=decimation_phase,
            source_samples_seen=int(run_start),
        )
        chunk_start = int(run_start)
        while chunk_start < run_stop:
            chunk_stop = min(
                chunk_start + MAX_PROCESS_CHUNK_SAMPLES,
                int(run_stop),
            )
            chunk_output_indices = processor.output_source_indices(
                chunk_stop - chunk_start
            )
            chunk_output = processor.process_chunk(source[chunk_start:chunk_stop])[0]
            if require_full_history:
                chunk_output[chunk_output_indices - run_start < NUM_TAPS - 1] = np.nan
            output_positions = (
                chunk_output_indices - decimation_phase
            ) // DECIMATION_FACTOR
            output[output_positions] = chunk_output
            chunk_start = chunk_stop
    return output


def design_manifest() -> dict:
    return {
        "method": "stateful_causal_fir_then_fixed_phase_decimation",
        "source_sampling_rate_hz": SOURCE_SAMPLING_RATE_HZ,
        "target_sampling_rate_hz": TARGET_SAMPLING_RATE_HZ,
        "source_packet_samples": SOURCE_PACKET_SAMPLES,
        "target_packet_samples": TARGET_PACKET_SAMPLES,
        "decimation_factor": DECIMATION_FACTOR,
        "num_taps": NUM_TAPS,
        "cutoff_hz": CUTOFF_HZ,
        "window": "numpy_hamming",
        "group_delay_source_samples": GROUP_DELAY_SOURCE_SAMPLES,
        "group_delay_seconds": GROUP_DELAY_SECONDS,
        "decimation_phase_source_sample": 0,
        "initial_state": "zeros",
        "state_continuity": "carried_across_all_packets_in_each_record",
        "gap_policy": "reset_state_and_withhold_incomplete_fir_support",
        "maximum_internal_source_chunk_samples": MAX_PROCESS_CHUNK_SAMPLES,
        "coefficients": FIR_COEFFICIENTS.tolist(),
    }


def self_test() -> None:
    if not np.allclose(FIR_COEFFICIENTS, FIR_COEFFICIENTS[::-1], atol=1e-15):
        raise AssertionError("FIR coefficients are not symmetric")
    if not np.isclose(FIR_COEFFICIENTS.sum(), 1.0, atol=1e-15):
        raise AssertionError("FIR coefficients are not normalized")

    rng = np.random.default_rng(20260718)
    source = rng.normal(size=(3, 4 * SOURCE_PACKET_SAMPLES))
    streaming = CausalFirDecimator200To100()
    packet_outputs = [
        streaming.process_packet(source[:, start : start + SOURCE_PACKET_SAMPLES])
        for start in range(0, source.shape[1], SOURCE_PACKET_SAMPLES)
    ]
    actual = np.concatenate(packet_outputs, axis=1)

    padded = np.pad(source, ((0, 0), (NUM_TAPS - 1, 0)))
    windows = np.lib.stride_tricks.sliding_window_view(padded, NUM_TAPS, axis=1)
    expected = np.einsum(
        "ctk,k->ct",
        windows,
        FIR_COEFFICIENTS[::-1],
        optimize=True,
    )[:, ::DECIMATION_FACTOR].astype(np.float32)
    if not np.array_equal(actual, expected):
        raise AssertionError("Packet and whole-record causal FIR outputs differ")

    chunked = CausalFirDecimator200To100()
    chunks = [17, 83, 1, 119, 180]
    offset = 0
    chunk_outputs = []
    for size in chunks:
        chunk_outputs.append(chunked.process_chunk(source[:, offset : offset + size]))
        offset += size
    if offset != source.shape[1]:
        raise AssertionError("Self-test chunks do not cover the source")
    if not np.array_equal(np.concatenate(chunk_outputs, axis=1), expected):
        raise AssertionError("Random chunks and whole-record outputs differ")

    impulse = np.zeros((3, 2 * SOURCE_PACKET_SAMPLES), dtype=np.float64)
    impulse[:, 40] = 1.0
    streaming = CausalFirDecimator200To100()
    impulse_output = np.concatenate(
        [
            streaming.process_packet(impulse[:, :SOURCE_PACKET_SAMPLES]),
            streaming.process_packet(impulse[:, SOURCE_PACKET_SAMPLES:]),
        ],
        axis=1,
    )
    peak = int(np.argmax(np.abs(impulse_output[0])))
    expected_peak = int(round((40 + GROUP_DELAY_SOURCE_SAMPLES) / 2.0))
    if abs(peak - expected_peak) > 1:
        raise AssertionError("FIR impulse delay does not match the frozen delay")

    gappy = source[0].copy()
    gappy[137:191] = np.nan
    converted = causal_decimate_200_to_100(gappy)
    output_indices = np.arange(0, gappy.size, DECIMATION_FACTOR)
    after_gap = output_indices >= 191
    warmup = output_indices < 191 + NUM_TAPS - 1
    if not np.isnan(converted[after_gap & warmup]).all():
        raise AssertionError("Post-gap FIR warmup was not withheld")


if __name__ == "__main__":
    self_test()
    print("causal FIR decimator self-test: PASS")
