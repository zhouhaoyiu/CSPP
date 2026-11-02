"""Causality-aligned data pipeline for domain-robust v5 experiments."""
from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from config import SAMPLES_PER_CHUNK, TARGET_SR
from data_streaming import normalize_packet_ewma_causal
from domain_robust_data import (
    KNETSource,
    PrePNoiseSource,
    SeisBenchSource,
    StreamRecord,
    _augment_components,
    _ensure_three_components,
    _rms,
    deterministic_limit,
    event_indices,
    find_p_column,
    instancegm_acceleration_indices,
    knet_factorized_indices,
    make_records,
    stable_bucket,
    station_key,
)

ONSET_RESPONSE_PACKETS = 3
VERIFIER_RESPONSE_PACKETS = 6
V7_ONSET_RESPONSE_PACKETS = 3
V7_VERIFIER_RESPONSE_PACKETS = 4
V25_ONSET_RESPONSE_PACKETS = 2
CATALOG_SAFE_COLUMN = "catalog_safe_for_negative_supervision"


def _catalog_safe_store_indices(
    store_metadata,
    split: str,
    catalog_screen_path: str | Path,
) -> tuple[np.ndarray, int]:
    """Return HDF5 indices that remain eligible for negative supervision."""
    import pandas as pd

    required_store = {"index", "split"}
    missing_store = sorted(required_store - set(store_metadata.columns))
    if missing_store:
        raise ValueError(f"Continuous metadata missing columns: {missing_store}")
    split_rows = store_metadata.loc[store_metadata["split"].eq(split)].copy()
    if split_rows["index"].duplicated().any():
        raise ValueError(f"Continuous metadata contains duplicate {split!r} indices")

    screen_path = Path(catalog_screen_path)
    screen = pd.read_csv(screen_path)
    required_screen = {"index", "split", CATALOG_SAFE_COLUMN}
    missing_screen = sorted(required_screen - set(screen.columns))
    if missing_screen:
        raise ValueError(f"Catalog screen missing columns: {missing_screen}")
    if screen["index"].duplicated().any():
        raise ValueError("Catalog screen contains duplicate HDF5 indices")
    screen = screen.loc[screen["split"].eq(split)].set_index("index")
    expected = split_rows["index"].astype(int).to_numpy()
    missing = sorted(set(expected) - set(screen.index.astype(int)))
    if missing:
        raise ValueError(
            f"Catalog screen does not cover {len(missing)} {split!r} windows; "
            f"first missing indices={missing[:10]}"
        )

    raw = screen.loc[expected, CATALOG_SAFE_COLUMN]
    if pd.api.types.is_bool_dtype(raw.dtype):
        safe = raw.to_numpy(dtype=bool)
    else:
        normalized = raw.astype(str).str.strip().str.lower()
        valid = normalized.isin({"true", "false", "1", "0"})
        if not valid.all():
            raise ValueError("Catalog-safe column contains non-boolean values")
        safe = normalized.isin({"true", "1"}).to_numpy()
    return expected[safe].astype(np.int64, copy=False), int((~safe).sum())


class ContinuousNoiseWindowSource:
    """Read catalog-screened presumed non-event windows from opened 2024 CI data."""

    preserve_continuity = True

    def __init__(
        self,
        name: str,
        hdf5_path: str | Path,
        split: str,
        catalog_screen_path: str | Path,
    ):
        import h5py
        import pandas as pd

        self.name = name
        self.hdf5_path = Path(hdf5_path)
        self.catalog_screen_path = Path(catalog_screen_path)
        self.metadata = pd.read_csv(self.hdf5_path.with_suffix(".csv"))
        self.noise_indices, self.catalog_excluded_count = _catalog_safe_store_indices(
            self.metadata,
            split,
            self.catalog_screen_path,
        )
        self._h5py = h5py
        self._handle = None
        if not len(self.noise_indices):
            raise ValueError(f"No {split!r} windows in {self.hdf5_path}")

    @property
    def handle(self):
        if self._handle is None:
            self._handle = self._h5py.File(self.hdf5_path, "r")
        return self._handle

    def read(self, index: int) -> tuple[np.ndarray, None]:
        wave = self.handle["waveforms"][int(index)].astype(np.float32)
        return wave, None


class ReviewedHardNegativeWindowSource:
    """Read reviewed continuous triggers with a local negative-loss mask."""

    preserve_continuity = True
    reviewed_hard_negative = True

    def __init__(
        self,
        name: str,
        hdf5_path: str | Path,
        review_manifest_path: str | Path,
        catalog_screen_path: str | Path,
        split: str = "train",
    ):
        import h5py
        import pandas as pd

        self.name = name
        self.hdf5_path = Path(hdf5_path)
        self.review_manifest_path = Path(review_manifest_path)
        self.catalog_screen_path = Path(catalog_screen_path)
        self.metadata = pd.read_csv(self.review_manifest_path)
        required = {
            "index",
            "review_status",
            "supervision_start_packet",
            "supervision_stop_packet_exclusive",
        }
        missing = sorted(required - set(self.metadata.columns))
        if missing:
            raise ValueError(f"Hard-negative manifest missing columns: {missing}")
        if "split" in self.metadata.columns:
            self.metadata = self.metadata.loc[self.metadata["split"].eq(split)].copy()
        elif split != "train":
            raise ValueError("A split column is required for hard-negative development rows")
        expected_status = {
            "train": "approved_provisional_training_only",
            "dev": "approved_provisional_development_calibration_only",
        }.get(split)
        if expected_status is None:
            raise ValueError(f"Unsupported hard-negative split: {split!r}")
        if self.metadata.empty or not self.metadata["review_status"].eq(
            expected_status
        ).all():
            raise ValueError(
                f"Hard-negative {split!r} rows require review_status={expected_status!r}"
            )
        if self.metadata["index"].duplicated().any():
            raise ValueError("Hard-negative manifest contains duplicate HDF5 indices")

        raw_store_metadata = pd.read_csv(self.hdf5_path.with_suffix(".csv"))
        safe_indices, _ = _catalog_safe_store_indices(
            raw_store_metadata,
            split,
            self.catalog_screen_path,
        )
        store_metadata = raw_store_metadata.set_index("index")
        indices = self.metadata["index"].astype(int).to_numpy()
        if not set(indices).issubset(store_metadata.index):
            raise ValueError("Hard-negative manifest references unknown HDF5 indices")
        if not store_metadata.loc[indices, "split"].eq(split).all():
            raise ValueError(f"Hard-negative manifest crosses the requested {split!r} split")
        unsafe = sorted(set(indices) - set(safe_indices.tolist()))
        if unsafe:
            raise ValueError(
                "Hard-negative manifest contains catalog-unsafe windows: "
                f"{unsafe}"
            )
        self.noise_indices = indices.astype(np.int64, copy=False)
        self.split = split
        self._rows = self.metadata.set_index("index")
        self._h5py = h5py
        self._handle = None
        with h5py.File(self.hdf5_path, "r") as handle:
            state_fields = {
                "normalization_initial_rms_sq",
                "normalization_initial_count",
            }
            present = state_fields.intersection(handle.keys())
            if present and present != state_fields:
                raise ValueError("Hard-negative store has an incomplete normalization state")
            self.preserve_operational_normalization = present == state_fields

    @property
    def handle(self):
        if self._handle is None:
            self._handle = self._h5py.File(self.hdf5_path, "r")
        return self._handle

    def read(self, index: int) -> tuple[np.ndarray, None]:
        wave = self.handle["waveforms"][int(index)].astype(np.float32)
        return wave, None

    def initial_normalization_state(self, index: int) -> dict | None:
        if not self.preserve_operational_normalization:
            return None
        count = int(self.handle["normalization_initial_count"][int(index)])
        if count <= 0:
            return None
        return {
            "rms_sq": float(
                self.handle["normalization_initial_rms_sq"][int(index)]
            ),
            "count": count,
        }

    def supervision_crop_start_sample(self, index: int, max_samples: int) -> int:
        total_samples = int(self.handle["waveforms"].shape[2])
        if total_samples <= max_samples:
            return 0
        row = self._rows.loc[int(index)]
        start_packet = int(row["supervision_start_packet"])
        stop_packet = int(row["supervision_stop_packet_exclusive"])
        max_packets = max_samples // SAMPLES_PER_CHUNK
        if self.preserve_operational_normalization:
            crop_packet = int(
                np.clip(
                    stop_packet - max_packets,
                    0,
                    total_samples // SAMPLES_PER_CHUNK - max_packets,
                )
            )
        else:
            center_packet = (start_packet + stop_packet) // 2
            crop_packet = int(
                np.clip(
                    center_packet - max_packets // 2,
                    0,
                    total_samples // SAMPLES_PER_CHUNK - max_packets,
                )
            )
        return crop_packet * SAMPLES_PER_CHUNK

    def supervision_packet_mask(
        self,
        index: int,
        max_chunks: int,
        crop_start_packet: int = 0,
    ) -> np.ndarray:
        row = self._rows.loc[int(index)]
        start = max(int(row["supervision_start_packet"]) - crop_start_packet, 0)
        stop = min(
            int(row["supervision_stop_packet_exclusive"]) - crop_start_packet,
            max_chunks,
        )
        mask = np.zeros(max_chunks, dtype=np.float32)
        if stop <= start:
            raise ValueError(
                f"Hard-negative supervision is outside the selected crop for index={index}"
            )
        mask[start:stop] = 1.0
        return mask


def _center(wave: np.ndarray) -> np.ndarray:
    wave = _ensure_three_components(wave).astype(np.float32, copy=True)
    return wave - wave.mean(axis=1, keepdims=True)


def assemble_independent_noise(
    next_segment: Callable[[], np.ndarray],
    samples: int,
    rng: np.random.Generator,
    crossfade_samples: int = 200,
) -> np.ndarray:
    """Build a noise bed from independent records without periodic tiling."""
    if samples <= 0:
        return np.zeros((3, 0), dtype=np.float32)

    output = np.zeros((3, samples), dtype=np.float32)
    filled = 0
    reference_rms: float | None = None
    attempts = 0
    while filled < samples:
        attempts += 1
        if attempts > 256:
            raise RuntimeError("Unable to assemble enough independent noise samples")
        segment = _center(next_segment())
        if segment.shape[1] < SAMPLES_PER_CHUNK:
            continue

        if reference_rms is None:
            reference_rms = max(_rms(segment), 1e-8)
        segment *= reference_rms / max(_rms(segment), 1e-8)
        segment *= float(np.exp(rng.uniform(-0.25, 0.25)))

        remaining = samples - filled
        overlap = min(crossfade_samples, filled, segment.shape[1] // 3)
        required = remaining + overlap
        if segment.shape[1] > required:
            start = int(rng.integers(0, segment.shape[1] - required + 1))
            segment = segment[:, start : start + required]

        if overlap:
            weight = np.linspace(0.0, 1.0, overlap, dtype=np.float32)[None, :]
            output[:, filled - overlap : filled] = (
                output[:, filled - overlap : filled] * (1.0 - weight)
                + segment[:, :overlap] * weight
            )
        available = min(segment.shape[1] - overlap, remaining)
        if available <= 0:
            continue
        output[:, filled : filled + available] = segment[:, overlap : overlap + available]
        filled += available

    return output


def relocate_event_without_predictive_seam(
    wave: np.ndarray,
    original_p: int,
    target_p: int,
    noise_bed: np.ndarray,
    rng: np.random.Generator,
    crossfade_samples: int = 200,
) -> tuple[np.ndarray, int]:
    """Place an event in a noise bed while matching the pre-P splice statistics.

    The splice lead is randomized and the same crossfades occur in event-free
    noise beds, preventing a fixed pre-P boundary from becoming a shortcut.
    """
    wave = _ensure_three_components(wave)
    output = noise_bed.copy()
    max_pre = min(original_p, target_p, int(20 * TARGET_SR))
    min_pre = min(max_pre, int(4 * TARGET_SR))
    if max_pre <= 0:
        pre_context = 0
    elif max_pre <= min_pre:
        pre_context = max_pre
    else:
        pre_context = int(rng.integers(min_pre, max_pre + 1))

    source_start = max(0, original_p - pre_context)
    destination_start = max(0, target_p - (original_p - source_start))
    count = min(wave.shape[1] - source_start, output.shape[1] - destination_start)
    if count <= 0:
        return output, destination_start

    event = wave[:, source_start : source_start + count].astype(np.float32, copy=True)
    original_bed = output[:, destination_start : destination_start + count].copy()
    match = min(crossfade_samples, pre_context, count, max(destination_start, 1))
    if match:
        reference = output[:, destination_start : destination_start + match]
        source_prefix = event[:, :match]
        reference_mean = reference.mean(axis=1, keepdims=True)
        source_mean = source_prefix.mean(axis=1, keepdims=True)
        reference_scale = float(
            np.sqrt(np.mean((reference - reference_mean) ** 2, dtype=np.float64))
        )
        source_scale = float(
            np.sqrt(np.mean((source_prefix - source_mean) ** 2, dtype=np.float64))
        )
        # A shared gain preserves component ratios while converting between
        # physical-unit event archives and raw-count continuous streams.
        gain = float(np.clip(reference_scale / max(source_scale, 1e-12), 1e-6, 1e6))
        event = (event - source_mean) * gain + reference_mean
        weight = np.linspace(0.0, 1.0, match, dtype=np.float32)[None, :]
        output[:, destination_start : destination_start + match] = (
            reference * (1.0 - weight) + event[:, :match] * weight
        )
    output[:, destination_start + match : destination_start + count] = event[:, match:count]

    end = destination_start + count
    tail = min(crossfade_samples, count) if end < output.shape[1] else 0
    if tail:
        weight = np.linspace(0.0, 1.0, tail, dtype=np.float32)[None, :]
        event_tail = event[:, count - tail : count]
        bed_tail = original_bed[:, count - tail : count]
        output[:, end - tail : end] = bed_tail * weight + event_tail * (1.0 - weight)
    return output, destination_start


def scale_noise_to_local_p_snr(
    event_wave: np.ndarray,
    noise_wave: np.ndarray,
    p_sample: int,
    snr_db: float,
    signal_seconds: float = 1.0,
    return_achieved: bool = False,
) -> np.ndarray | tuple[np.ndarray, float]:
    """Add only the noise power needed to reach a local P-window SNR."""
    p_sample = int(p_sample)
    window = max(int(round(signal_seconds * TARGET_SR)), SAMPLES_PER_CHUNK)
    pre_start = max(0, p_sample - window)
    post_end = min(event_wave.shape[1], p_sample + window)
    pre = event_wave[:, pre_start:p_sample]
    post = event_wave[:, p_sample:post_end]
    if pre.size == 0 or post.size == 0:
        result = np.zeros_like(noise_wave, dtype=np.float32)
        return (result, math.nan) if return_achieved else result
    baseline_mean = pre.mean(axis=1, keepdims=True)
    pre_power = float(np.mean((pre - baseline_mean) ** 2, dtype=np.float64))
    post_power = float(np.mean((post - baseline_mean) ** 2, dtype=np.float64))
    signal_power = max(post_power - pre_power, 1e-16)
    target_total_noise_power = signal_power / (10.0 ** (float(snr_db) / 10.0))
    added_noise_power = max(target_total_noise_power - pre_power, 0.0)
    achieved_snr = float(
        10.0 * math.log10(signal_power / max(pre_power + added_noise_power, 1e-16))
    )
    if added_noise_power == 0.0:
        result = np.zeros_like(noise_wave, dtype=np.float32)
        return (result, achieved_snr) if return_achieved else result

    centered = _center(noise_wave)
    noise_start = max(0, min(p_sample - window, centered.shape[1] - 1))
    noise_end = min(centered.shape[1], p_sample)
    local_noise = centered[:, noise_start:noise_end]
    if local_noise.size == 0:
        local_noise = centered
    noise_rms = _rms(local_noise)
    result = centered * (math.sqrt(added_noise_power) / max(noise_rms, 1e-8))
    return (result, achieved_snr) if return_achieved else result


def estimate_local_p_snr_db(
    wave: np.ndarray,
    p_sample: int,
    signal_seconds: float = 1.0,
) -> float:
    """Estimate P-window excess power over the immediately preceding noise."""
    p_sample = int(p_sample)
    window = max(int(round(signal_seconds * TARGET_SR)), SAMPLES_PER_CHUNK)
    pre = wave[:, max(0, p_sample - window) : p_sample]
    post = wave[:, p_sample : min(wave.shape[1], p_sample + window)]
    if pre.size == 0 or post.size == 0:
        return math.nan
    baseline_mean = pre.mean(axis=1, keepdims=True)
    noise_power = float(np.mean((pre - baseline_mean) ** 2, dtype=np.float64))
    post_power = float(np.mean((post - baseline_mean) ** 2, dtype=np.float64))
    signal_power = max(post_power - noise_power, 1e-16)
    return float(10.0 * math.log10(signal_power / max(noise_power, 1e-16)))


def add_sensor_artifacts_from_past(
    wave: np.ndarray,
    reference_sample: int,
    rng: np.random.Generator,
    include_invalid_gaps: bool = False,
) -> np.ndarray:
    """Add valid-stream drift, impulses, and clipping using only past cues."""
    wave = _ensure_three_components(wave).astype(np.float32, copy=True)
    samples = wave.shape[1]
    reference_end = min(max(int(reference_sample) - SAMPLES_PER_CHUNK, SAMPLES_PER_CHUNK), samples)
    reference = wave[:, :reference_end]
    reference_mean = reference.mean(axis=1, keepdims=True)
    reference = reference - reference_mean
    reference_rms = max(_rms(reference), 1e-8)
    time = np.arange(samples, dtype=np.float32) / TARGET_SR

    if rng.random() < 0.35:
        amplitude = reference_rms * float(rng.uniform(0.02, 0.25))
        frequency = float(rng.uniform(0.03, 0.3))
        wave += amplitude * np.sin(2 * np.pi * frequency * time)[None, :]
    if rng.random() < 0.20:
        for _ in range(int(rng.integers(1, 5))):
            position = int(rng.integers(0, samples))
            wave[:, position] += rng.normal(0.0, reference_rms * rng.uniform(5.0, 20.0), 3)
    if include_invalid_gaps and rng.random() < 0.15 and samples > 200:
        gap = int(rng.integers(10, 100))
        before = np.arange(0, max(reference_sample - 150, 1))
        after = np.arange(min(reference_sample + 150, samples), samples)
        candidates = np.concatenate((before, after))
        if len(candidates):
            start = int(rng.choice(candidates))
            wave[:, start : min(start + gap, samples)] = 0.0
    if rng.random() < 0.15:
        limit = reference_rms * float(np.exp(rng.uniform(np.log(8.0), np.log(200.0))))
        wave = reference_mean + np.clip(wave - reference_mean, -limit, limit)
    return wave


def prepend_event_warmup_noise(
    wave: np.ndarray,
    p_sample: float,
    minimum_p_sample: int,
    noise_prefix: np.ndarray,
    crossfade_samples: int = 100,
) -> tuple[np.ndarray, float, int]:
    """Add prediction-independent past noise when an archive starts too near P."""
    shift = max(int(math.ceil(minimum_p_sample - float(p_sample))), 0)
    if shift == 0:
        return wave, float(p_sample), 0
    overlap = min(crossfade_samples, shift, wave.shape[1], max(int(p_sample) - 50, 0))
    required = shift + overlap
    if noise_prefix.shape[1] < required:
        raise ValueError(f"Warm-up noise has {noise_prefix.shape[1]} samples; need {required}")
    prefix = _ensure_three_components(noise_prefix[:, :required]).astype(
        np.float32, copy=True
    )
    reference_count = min(max(overlap, 50), wave.shape[1], max(int(p_sample) - 50, 1))
    source_reference = wave[:, :reference_count].astype(np.float32, copy=False)
    source_mean = source_reference.mean(axis=1, keepdims=True)
    noise_mean = prefix.mean(axis=1, keepdims=True)
    source_scale = _rms(source_reference - source_mean)
    noise_scale = _rms(prefix - noise_mean)
    prefix = (prefix - noise_mean) * (source_scale / max(noise_scale, 1e-8)) + source_mean
    if overlap:
        weight = np.linspace(0.0, 1.0, overlap, dtype=np.float32)[None, :]
        blend = prefix[:, shift : shift + overlap] * (1.0 - weight) + wave[:, :overlap] * weight
        combined = np.concatenate((prefix[:, :shift], blend, wave[:, overlap:]), axis=1)
    else:
        combined = np.concatenate((prefix[:, :shift], wave), axis=1)
    return combined.astype(np.float32, copy=False), float(p_sample) + shift, shift


def crop_native_event_context(
    wave: np.ndarray,
    p_sample: float,
    max_samples: int,
    minimum_p_sample: int,
    trailing_guard_samples: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float] | None:
    """Take one contiguous native crop without replacing its background."""
    wave = _ensure_three_components(wave)
    p_sample = float(p_sample)
    if (
        max_samples <= 0
        or trailing_guard_samples < 0
        or not 0 <= p_sample < wave.shape[1]
    ):
        return None
    if wave.shape[1] <= max_samples:
        if (
            p_sample < minimum_p_sample
            or p_sample >= wave.shape[1] - trailing_guard_samples
        ):
            return None
        return wave.astype(np.float32, copy=True), p_sample

    earliest_start = max(
        0,
        int(math.ceil(p_sample + trailing_guard_samples - max_samples)),
    )
    latest_start = min(
        int(math.floor(p_sample - minimum_p_sample)),
        wave.shape[1] - max_samples,
    )
    if latest_start < earliest_start:
        return None
    crop_start = int(rng.integers(earliest_start, latest_start + 1))
    return (
        wave[:, crop_start : crop_start + max_samples].astype(
            np.float32,
            copy=True,
        ),
        p_sample - crop_start,
    )


def packetize_trace_v5(
    wave: np.ndarray,
    p_sample: float | None,
    max_chunks: int,
    valid_samples: int | None = None,
    onset_response_packets: int = ONSET_RESPONSE_PACKETS,
    verifier_response_packets: int = VERIFIER_RESPONSE_PACKETS,
    verifier_start_delay_packets: int = 0,
    initial_normalization_state: dict | None = None,
) -> dict[str, torch.Tensor | int | float]:
    """Packetize with a causal post-P response window and sub-packet target."""
    max_samples = max_chunks * SAMPLES_PER_CHUNK
    valid_samples = min(wave.shape[1], max_samples) if valid_samples is None else min(valid_samples, max_samples)
    padded = np.zeros((3, max_samples), dtype=np.float32)
    padded[:, :valid_samples] = wave[:, :valid_samples]
    chunks = np.zeros((max_chunks, 3, SAMPLES_PER_CHUNK), dtype=np.float32)
    mask = np.zeros(max_chunks, dtype=np.float32)
    state = (
        None
        if initial_normalization_state is None
        else dict(initial_normalization_state)
    )
    for packet_index in range(math.ceil(valid_samples / SAMPLES_PER_CHUNK)):
        start = packet_index * SAMPLES_PER_CHUNK
        count = min(SAMPLES_PER_CHUNK, valid_samples - start)
        chunks[packet_index], state = normalize_packet_ewma_causal(
            padded[:, start : start + SAMPLES_PER_CHUNK],
            state,
            valid_samples=count,
        )
        mask[packet_index] = 1.0

    labels = np.zeros(max_chunks, dtype=np.float32)
    labels_event = np.zeros(max_chunks, dtype=np.float32)
    target_chunk = -1
    target_offset = math.nan
    if p_sample is not None and 0 <= p_sample < valid_samples:
        target_position = float(p_sample) / SAMPLES_PER_CHUNK
        target_chunk = int(math.floor(target_position))
        target_offset = target_position - target_chunk
        response_end = min(target_chunk + onset_response_packets, max_chunks)
        verifier_start = min(
            target_chunk + verifier_start_delay_packets,
            max_chunks,
        )
        verifier_end = min(target_chunk + verifier_response_packets, max_chunks)
        labels[target_chunk:response_end] = mask[target_chunk:response_end]
        labels_event[verifier_start:verifier_end] = mask[verifier_start:verifier_end]
    return {
        "chunks": torch.from_numpy(chunks),
        "labels": torch.from_numpy(labels),
        "labels_event": torch.from_numpy(labels_event),
        "mask": torch.from_numpy(mask),
        "target_chunk": target_chunk,
        "target_offset": float(target_offset),
    }


class DomainRobustStreamDatasetV5(Dataset):
    def __init__(
        self,
        sources: dict[str, object],
        records: list[StreamRecord],
        max_chunks: int,
        training: bool,
        noise_source_names: tuple[str, ...],
        seed: int = 20260715,
        eval_snr_db: float | None = None,
        onset_response_packets: int = ONSET_RESPONSE_PACKETS,
        verifier_response_packets: int = VERIFIER_RESPONSE_PACKETS,
        snr_profile: str = "legacy",
        include_wave: bool = False,
        minimum_event_p_packet: int = 0,
        verifier_start_delay_packets: int = 0,
        native_event_fraction: float = 0.15,
    ):
        self.sources = sources
        self.records = records
        self.max_chunks = int(max_chunks)
        self.training = bool(training)
        self.noise_source_names = tuple(name for name in noise_source_names if name in sources)
        self.seed = int(seed)
        self.epoch = 0
        self._draw_counts: dict[int, int] = {}
        self.eval_snr_db = eval_snr_db
        self.onset_response_packets = int(onset_response_packets)
        self.verifier_response_packets = int(verifier_response_packets)
        self.snr_profile = str(snr_profile)
        self.include_wave = bool(include_wave)
        self.minimum_event_p_packet = int(minimum_event_p_packet)
        self.verifier_start_delay_packets = int(verifier_start_delay_packets)
        self.native_event_fraction = float(native_event_fraction)
        if self.onset_response_packets < 1 or self.verifier_response_packets < 1:
            raise ValueError("Response packet counts must be positive")
        if not 0 <= self.verifier_start_delay_packets < self.verifier_response_packets:
            raise ValueError("Verifier delay must be inside its response window")
        if self.snr_profile not in {
            "legacy",
            "v7_operational",
            "strong_motion_balanced",
        }:
            raise ValueError(f"Unknown SNR profile: {self.snr_profile}")
        if not 0.0 <= self.native_event_fraction <= 1.0:
            raise ValueError("native_event_fraction must be in [0, 1]")
        self.noise_indices: dict[str, np.ndarray] = {}
        self._noise_cache: dict[tuple[str, int], np.ndarray] = {}
        for name in self.noise_source_names:
            source = sources[name]
            if hasattr(source, "noise_indices"):
                indices = np.asarray(source.noise_indices, dtype=np.int64)
            else:
                indices = np.flatnonzero(source.metadata["trace_category"].eq("noise").to_numpy())
            if len(indices):
                self.noise_indices[name] = indices
        if not self.noise_indices:
            raise ValueError("v5 requires at least one real noise source")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self._draw_counts.clear()

    def _draw_training_snr(self, rng: np.random.Generator) -> float:
        if self.epoch <= 1:
            lower = 8.0 if self.snr_profile == "strong_motion_balanced" else 5.0
            return float(rng.uniform(lower, 25.0))
        if self.epoch == 2:
            upper = 15.0 if self.snr_profile == "v7_operational" else 20.0
            return float(rng.uniform(0.0, upper))
        level_draw = float(rng.random())
        if self.snr_profile == "strong_motion_balanced":
            if level_draw < 0.45:
                return float(rng.uniform(-1.0, 2.0))
            if level_draw < 0.80:
                return float(rng.uniform(2.0, 8.0))
            if level_draw < 0.95:
                return float(rng.uniform(8.0, 15.0))
            return float(rng.uniform(15.0, 25.0))
        if self.snr_profile == "v7_operational":
            # -10 dB events are retained for reporting, but training mass is
            # concentrated around the deployable 0--10 dB boundary.
            if level_draw < 0.05:
                return float(rng.uniform(-5.0, 0.0))
            if level_draw < 0.70:
                return float(rng.uniform(0.0, 8.0))
            if level_draw < 0.90:
                return float(rng.uniform(8.0, 15.0))
            return float(rng.uniform(15.0, 25.0))
        if level_draw < 0.20:
            return float(rng.uniform(-10.0, 0.0))
        if level_draw < 0.65:
            return float(rng.uniform(0.0, 10.0))
        return float(rng.uniform(10.0, 20.0))

    def __len__(self) -> int:
        return len(self.records)

    def _read_noise(self, source_name: str, index: int) -> np.ndarray:
        source = self.sources[source_name]
        if getattr(source, "preserve_continuity", False):
            wave, _ = source.read(int(index))
            return _ensure_three_components(wave).astype(np.float32, copy=True)
        key = (source_name, int(index))
        if key not in self._noise_cache:
            wave, _ = source.read(int(index))
            self._noise_cache[key] = _ensure_three_components(wave).astype(np.float32, copy=True)
        return self._noise_cache[key]

    def _noise_segment(self, rng: np.random.Generator) -> np.ndarray:
        source_name = tuple(self.noise_indices)[int(rng.integers(0, len(self.noise_indices)))]
        indices = self.noise_indices[source_name]
        index = int(indices[int(rng.integers(0, len(indices)))])
        return self._read_noise(source_name, index)

    def _noise_bed(
        self,
        rng: np.random.Generator,
        samples: int,
        primary: np.ndarray | None = None,
    ) -> np.ndarray:
        first = [primary] if primary is not None and primary.shape[1] >= SAMPLES_PER_CHUNK else []

        def next_segment() -> np.ndarray:
            if first:
                return first.pop()
            return self._noise_segment(rng)

        return assemble_independent_noise(next_segment, samples, rng)

    def __getitem__(self, item: int) -> dict:
        record = self.records[item]
        draw_index = 0
        if self.training:
            draw_index = self._draw_counts.get(item, 0)
            self._draw_counts[item] = draw_index + 1
        rng = np.random.default_rng(
            self.seed
            + self.epoch * 1_000_003
            + item * 97
            + draw_index * 65_537
        )
        source = self.sources[record.source]
        if hasattr(source, "read_packetized"):
            if not record.is_event:
                raise ValueError("Prepacketized CWA records must be events")
            if self.eval_snr_db is not None:
                raise ValueError(
                    "Prepacketized native CWA records cannot be synthetically "
                    "reused as a controlled-SNR cohort"
                )
            if self.include_wave:
                raise ValueError(
                    "Prepacketized CWA records do not expose raw waveforms"
                )
            packed = source.read_packetized(
                record.index,
                max_chunks=self.max_chunks,
                onset_response_packets=self.onset_response_packets,
                verifier_response_packets=self.verifier_response_packets,
                verifier_start_delay_packets=self.verifier_start_delay_packets,
            )
            packed.update(
                {
                    "is_event": 1,
                    "source": record.source,
                    "trace_index": record.index,
                    "snr_db": math.nan,
                    "achieved_snr_db": math.nan,
                    "empirical_snr_db": math.nan,
                    "splice_sample": -1,
                    "warmup_prefix_samples": 0,
                }
            )
            return packed
        wave, p_sample = self.sources[record.source].read(record.index)
        wave = _ensure_three_components(wave)
        max_samples = self.max_chunks * SAMPLES_PER_CHUNK
        snr_db = math.nan
        achieved_snr_db = math.nan
        empirical_snr_db = math.nan
        splice_sample = -1
        warmup_prefix_samples = 0
        crop_start_sample = 0
        normalization_state = (
            source.initial_normalization_state(record.index)
            if hasattr(source, "initial_normalization_state")
            else None
        )
        preserve_operational_normalization = bool(
            getattr(source, "preserve_operational_normalization", False)
        )

        if record.is_event:
            if p_sample is None:
                raise ValueError(f"Event record has no P label: {record}")
            original_p = int(round(p_sample))
            if self.training:
                minimum_target_packet = max(8, self.minimum_event_p_packet)
                minimum_p_sample = minimum_target_packet * SAMPLES_PER_CHUNK
                trailing_guard_samples = 8 * SAMPLES_PER_CHUNK
                native_context = None
                if rng.random() < self.native_event_fraction:
                    native_context = crop_native_event_context(
                        wave,
                        float(original_p),
                        max_samples,
                        minimum_p_sample,
                        trailing_guard_samples,
                        rng,
                    )
                if native_context is not None:
                    wave, p_sample = native_context
                    target_p = int(round(p_sample))
                    valid_samples = min(wave.shape[1], max_samples)
                else:
                    wave = _augment_components(wave, rng)
                    target_p = int(
                        rng.integers(
                            minimum_p_sample,
                            max_samples - trailing_guard_samples,
                        )
                    )
                    noise_bed = self._noise_bed(rng, max_samples)
                    wave, splice_sample = relocate_event_without_predictive_seam(
                        wave, original_p, target_p, noise_bed, rng
                    )
                    snr_db = self._draw_training_snr(rng)
                    added_noise, achieved_snr_db = scale_noise_to_local_p_snr(
                        wave,
                        self._noise_bed(rng, max_samples),
                        target_p,
                        snr_db,
                        return_achieved=True,
                    )
                    wave = add_sensor_artifacts_from_past(
                        wave + added_noise,
                        target_p,
                        rng,
                    )
                    p_sample = float(target_p)
                    empirical_snr_db = estimate_local_p_snr_db(wave, target_p)
                    valid_samples = max_samples
            else:
                valid_samples = min(wave.shape[1], max_samples)
                if self.eval_snr_db is not None:
                    noise = self._noise_bed(rng, valid_samples)
                    added_noise, achieved_snr_db = scale_noise_to_local_p_snr(
                        wave[:, :valid_samples],
                        noise,
                        original_p,
                        self.eval_snr_db,
                        return_achieved=True,
                    )
                    wave = wave[:, :valid_samples] + added_noise
                    snr_db = float(self.eval_snr_db)
                    empirical_snr_db = estimate_local_p_snr_db(wave, original_p)
                minimum_p_sample = self.minimum_event_p_packet * SAMPLES_PER_CHUNK
                if original_p < minimum_p_sample:
                    needed = int(math.ceil(minimum_p_sample - original_p)) + 100
                    wave, p_sample, warmup_prefix_samples = prepend_event_warmup_noise(
                        wave,
                        float(original_p),
                        minimum_p_sample,
                        self._noise_bed(rng, needed),
                    )
                    valid_samples = min(wave.shape[1], max_samples)
        else:
            p_sample = None
            if self.training:
                if getattr(source, "preserve_continuity", False):
                    if wave.shape[1] < max_samples:
                        raise ValueError(
                            f"Continuous noise window is shorter than {max_samples}: {record}"
                        )
                    if wave.shape[1] > max_samples:
                        if hasattr(source, "supervision_crop_start_sample"):
                            crop_start_sample = int(
                                source.supervision_crop_start_sample(
                                    record.index, max_samples
                                )
                            )
                        else:
                            crop_start_sample = int(
                                rng.integers(0, wave.shape[1] - max_samples + 1)
                            )
                        if preserve_operational_normalization and crop_start_sample:
                            for start in range(0, crop_start_sample, SAMPLES_PER_CHUNK):
                                _, normalization_state = normalize_packet_ewma_causal(
                                    wave[:, start : start + SAMPLES_PER_CHUNK],
                                    normalization_state,
                                )
                        wave = wave[
                            :, crop_start_sample : crop_start_sample + max_samples
                        ].copy()
                    if not preserve_operational_normalization and rng.random() < 0.50:
                        wave = _augment_components(wave, rng)
                    if (
                        not getattr(source, "reviewed_hard_negative", False)
                        and rng.random() < 0.20
                    ):
                        wave = add_sensor_artifacts_from_past(
                            wave, max_samples // 2, rng
                        )
                else:
                    wave = self._noise_bed(rng, max_samples, primary=wave)
                    wave = add_sensor_artifacts_from_past(
                        _augment_components(wave, rng), max_samples // 2, rng
                    )
                valid_samples = max_samples
            else:
                if (
                    wave.shape[1] > max_samples
                    and hasattr(source, "supervision_crop_start_sample")
                ):
                    crop_start_sample = int(
                        source.supervision_crop_start_sample(
                            record.index, max_samples
                        )
                    )
                    if preserve_operational_normalization and crop_start_sample:
                        for start in range(0, crop_start_sample, SAMPLES_PER_CHUNK):
                            _, normalization_state = normalize_packet_ewma_causal(
                                wave[:, start : start + SAMPLES_PER_CHUNK],
                                normalization_state,
                            )
                    wave = wave[
                        :, crop_start_sample : crop_start_sample + max_samples
                    ].copy()
                valid_samples = min(wave.shape[1], max_samples)

        packed = packetize_trace_v5(
            wave,
            p_sample,
            self.max_chunks,
            valid_samples=valid_samples,
            onset_response_packets=self.onset_response_packets,
            verifier_response_packets=self.verifier_response_packets,
            verifier_start_delay_packets=self.verifier_start_delay_packets,
            initial_normalization_state=normalization_state,
        )
        if not record.is_event and hasattr(source, "supervision_packet_mask"):
            local_mask = source.supervision_packet_mask(
                record.index,
                self.max_chunks,
                crop_start_packet=crop_start_sample // SAMPLES_PER_CHUNK,
            )
            packed["mask"] *= torch.from_numpy(local_mask).to(packed["mask"].dtype)
        packed.update(
            {
                "is_event": int(record.is_event),
                "source": record.source,
                "trace_index": record.index,
                "snr_db": float(snr_db),
                "achieved_snr_db": float(achieved_snr_db),
                "empirical_snr_db": float(empirical_snr_db),
                "splice_sample": int(splice_sample),
                "warmup_prefix_samples": int(warmup_prefix_samples),
            }
        )
        if self.include_wave:
            packed["raw_wave"] = torch.from_numpy(
                np.asarray(wave[:, :valid_samples], dtype=np.float32).copy()
            )
            packed["raw_p_sample"] = (
                math.nan if p_sample is None else float(p_sample)
            )
            packed["valid_samples"] = int(valid_samples)
        return packed


def collate_domain_robust_v5(batch: list[dict]) -> dict:
    tensor_keys = ("chunks", "labels", "labels_event", "mask")
    output = {key: torch.stack([sample[key] for sample in batch]) for key in tensor_keys}
    output["target_chunk"] = torch.tensor([sample["target_chunk"] for sample in batch], dtype=torch.long)
    output["target_offset"] = torch.tensor([sample["target_offset"] for sample in batch], dtype=torch.float32)
    output["is_event"] = torch.tensor([sample["is_event"] for sample in batch], dtype=torch.bool)
    output["trace_index"] = torch.tensor([sample["trace_index"] for sample in batch], dtype=torch.long)
    output["snr_db"] = torch.tensor([sample["snr_db"] for sample in batch], dtype=torch.float32)
    output["achieved_snr_db"] = torch.tensor(
        [sample["achieved_snr_db"] for sample in batch], dtype=torch.float32
    )
    output["empirical_snr_db"] = torch.tensor(
        [sample["empirical_snr_db"] for sample in batch], dtype=torch.float32
    )
    output["splice_sample"] = torch.tensor([sample["splice_sample"] for sample in batch], dtype=torch.long)
    output["source"] = [sample["source"] for sample in batch]
    return output


__all__ = [
    "DomainRobustStreamDatasetV5",
    "ContinuousNoiseWindowSource",
    "ReviewedHardNegativeWindowSource",
    "KNETSource",
    "ONSET_RESPONSE_PACKETS",
    "VERIFIER_RESPONSE_PACKETS",
    "V7_ONSET_RESPONSE_PACKETS",
    "V7_VERIFIER_RESPONSE_PACKETS",
    "V25_ONSET_RESPONSE_PACKETS",
    "PrePNoiseSource",
    "SeisBenchSource",
    "StreamRecord",
    "add_sensor_artifacts_from_past",
    "assemble_independent_noise",
    "collate_domain_robust_v5",
    "crop_native_event_context",
    "deterministic_limit",
    "estimate_local_p_snr_db",
    "event_indices",
    "find_p_column",
    "instancegm_acceleration_indices",
    "knet_factorized_indices",
    "make_records",
    "packetize_trace_v5",
    "prepend_event_warmup_noise",
    "relocate_event_without_predictive_seam",
    "scale_noise_to_local_p_snr",
    "stable_bucket",
    "station_key",
]
