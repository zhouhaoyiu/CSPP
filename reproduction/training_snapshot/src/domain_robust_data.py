"""Data pipeline for domain-robust causal streaming experiments."""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from config import P_ARRIVAL_COL_CANDIDATES, SAMPLES_PER_CHUNK, TARGET_SR
from data_streaming import normalize_packet_ewma_causal


def stable_bucket(value: object, seed: int = 20260715, buckets: int = 100) -> int:
    payload = f"{seed}|{value}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % buckets


def deterministic_limit(indices: np.ndarray, limit: int, seed: int) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    if limit <= 0 or len(indices) <= limit:
        return indices
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(indices, size=limit, replace=False))


def find_p_column(metadata: pd.DataFrame) -> str:
    for column in P_ARRIVAL_COL_CANDIDATES:
        if column in metadata.columns:
            return column
    if "trace_p_arrival_sample" in metadata.columns:
        return "trace_p_arrival_sample"
    raise ValueError("No P-arrival sample column found")


def event_indices(
    metadata: pd.DataFrame,
    split: str,
    p_column: str | None = None,
    max_samples: int | None = None,
) -> np.ndarray:
    p_column = p_column or find_p_column(metadata)
    mask = metadata["split"].eq(split) & metadata[p_column].notna()
    if max_samples is not None:
        mask &= metadata[p_column].astype(float).between(0, max_samples - 1)
    return np.flatnonzero(mask.to_numpy())


def station_key(metadata: pd.DataFrame) -> pd.Series:
    network = metadata.get("station_network_code", pd.Series("", index=metadata.index))
    station = metadata.get("station_code", pd.Series("UNKNOWN", index=metadata.index))
    return network.fillna("").astype(str) + "." + station.fillna("UNKNOWN").astype(str)


def instancegm_acceleration_indices(metadata: pd.DataFrame, split: str) -> np.ndarray:
    """Select labeled accelerometer traces, excluding velocity channels."""
    p_column = find_p_column(metadata)
    channels = metadata["station_channels"].fillna("").astype(str).str.upper()
    accelerometer_channel = channels.str.len().ge(2) & channels.str[1].eq("N")
    acceleration_units = metadata["trace_deconvolved_units"].fillna("").astype(str).str.lower().eq("mps2")
    mask = metadata["split"].eq(split) & metadata[p_column].notna()
    mask &= accelerometer_channel & acceleration_units
    return np.flatnonzero(mask.to_numpy())


def knet_factorized_indices(
    metadata: pd.DataFrame,
    role: str,
    seed: int = 20260715,
) -> np.ndarray:
    """Return event/station-factorized K-NET partitions.

    Original event splits remain untouched. Stations are independently hashed
    into train (0-69), dev (70-84), and test (85-99) groups.
    """
    rules = {
        "train": ("train", "train"),
        "dev_event": ("dev", "train"),
        "dev_station": ("train", "dev"),
        "dev_joint": ("dev", "dev"),
        "test_event": ("test", "train"),
        "test_station": ("train", "test"),
        "test_joint": ("test", "test"),
    }
    if role not in rules:
        raise ValueError(f"Unknown K-NET partition role: {role}")
    event_split, station_split = rules[role]
    buckets = station_key(metadata).map(lambda value: stable_bucket(value, seed=seed))
    if station_split == "train":
        station_mask = buckets < 70
    elif station_split == "dev":
        station_mask = (buckets >= 70) & (buckets < 85)
    else:
        station_mask = buckets >= 85
    p_column = find_p_column(metadata)
    mask = metadata["split"].eq(event_split) & station_mask & metadata[p_column].notna()
    if "trace_sampling_rate_hz" in metadata:
        native_rate = pd.to_numeric(metadata["trace_sampling_rate_hz"], errors="coerce")
        mask &= np.isclose(native_rate, TARGET_SR)
    return np.flatnonzero(mask.to_numpy())


class SeisBenchSource:
    def __init__(self, name: str, dataset, p_column: str | None = None):
        self.name = name
        self.dataset = dataset
        self.metadata = dataset.metadata
        self.p_column = p_column or (
            find_p_column(self.metadata) if any(c in self.metadata for c in P_ARRIVAL_COL_CANDIDATES) else None
        )

    def read(self, index: int) -> tuple[np.ndarray, float | None]:
        wave, sample_meta = self.dataset.get_sample(int(index), sampling_rate=TARGET_SR)
        p_sample = None
        if self.p_column is not None:
            value = sample_meta.get(self.p_column, self.metadata.iloc[int(index)][self.p_column])
            if value is not None and not pd.isna(value):
                p_sample = float(value)
        return np.asarray(wave, dtype=np.float32), p_sample


class KNETSource:
    def __init__(self, data_dir: str | Path):
        import h5py

        self.name = "knet"
        self.data_dir = Path(data_dir)
        self.metadata = pd.read_csv(self.data_dir / "metadata.csv")
        self.p_column = find_p_column(self.metadata)
        self.hdf5_path = self.data_dir / "waveforms.hdf5"
        manifest_path = self.data_dir / "manifest.json"
        self.strict_causal = False
        if manifest_path.is_file():
            import json

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.strict_causal = (
                manifest.get("protocol") == "knet_raw_acceleration_strict_causal_input_v1"
            )
        self._h5py = h5py
        self._handle = None

    @property
    def handle(self):
        if self._handle is None:
            self._handle = self._h5py.File(self.hdf5_path, "r")
        return self._handle

    def read(self, index: int) -> tuple[np.ndarray, float]:
        row = self.metadata.iloc[int(index)]
        wave = self.handle["data"][row["trace_name"]][:].astype(np.float32)
        p_sample = float(row[self.p_column])
        native_rate = float(row.get("trace_sampling_rate_hz", TARGET_SR))
        if not math.isclose(native_rate, TARGET_SR):
            if self.strict_causal:
                raise ValueError(
                    f"Strict-causal K-NET trace {row['trace_name']} has native rate "
                    f"{native_rate:g} Hz; select native {TARGET_SR:g} Hz traces instead "
                    "of applying a whole-record resampler"
                )
            from scipy.signal import resample

            target_samples = round(wave.shape[1] * TARGET_SR / native_rate)
            wave = resample(wave, target_samples, axis=1).astype(np.float32)
            p_sample *= TARGET_SR / native_rate
        return wave, p_sample


class PrePNoiseSource:
    """Expose only the causal pre-P portion of labeled acceleration traces."""

    def __init__(
        self,
        name: str,
        base_source,
        noise_indices: np.ndarray,
        guard_samples: int = 100,
    ):
        self.name = name
        self.base_source = base_source
        self.metadata = base_source.metadata
        self.noise_indices = np.asarray(noise_indices, dtype=np.int64)
        self.guard_samples = int(guard_samples)

    def read(self, index: int) -> tuple[np.ndarray, None]:
        wave, p_sample = self.base_source.read(int(index))
        wave = _ensure_three_components(wave)
        if p_sample is None:
            end = min(wave.shape[1], int(5 * TARGET_SR))
        else:
            end = max(int(round(p_sample)) - self.guard_samples, SAMPLES_PER_CHUNK)
            end = min(end, wave.shape[1])
        return wave[:, :end].copy(), None


@dataclass(frozen=True)
class StreamRecord:
    source: str
    index: int
    is_event: bool


def make_records(source: str, indices: np.ndarray, is_event: bool) -> list[StreamRecord]:
    return [StreamRecord(source, int(index), is_event) for index in indices]


def _ensure_three_components(wave: np.ndarray) -> np.ndarray:
    wave = np.asarray(wave, dtype=np.float32)
    if wave.ndim != 2:
        raise ValueError(f"Expected two-dimensional waveform, got {wave.shape}")
    if wave.shape[0] < 3:
        wave = np.pad(wave, ((0, 3 - wave.shape[0]), (0, 0)))
    wave = wave[:3]
    return np.nan_to_num(wave, copy=False, nan=0.0, posinf=0.0, neginf=0.0)


def _repeat_or_crop(wave: np.ndarray, samples: int, rng: np.random.Generator) -> np.ndarray:
    if wave.shape[1] >= samples:
        start = int(rng.integers(0, wave.shape[1] - samples + 1))
        return wave[:, start : start + samples].copy()
    repeats = math.ceil(samples / max(wave.shape[1], 1))
    tiled = np.tile(wave, (1, repeats))
    return tiled[:, :samples].copy()


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(values, dtype=np.float64) ** 2) + 1e-16))


def _scale_noise_to_snr(
    event_wave: np.ndarray,
    noise_wave: np.ndarray,
    p_sample: int,
    snr_db: float,
) -> np.ndarray:
    pre_end = max(p_sample - 50, 1)
    baseline = event_wave[:, :pre_end].mean(axis=1, keepdims=True)
    signal_end = min(event_wave.shape[1], p_sample + int(5 * TARGET_SR))
    signal_rms = _rms(event_wave[:, p_sample:signal_end] - baseline)
    noise_centered = noise_wave - noise_wave.mean(axis=1, keepdims=True)
    noise_rms = _rms(noise_centered)
    target_noise_rms = max(signal_rms, 1e-8) / (10.0 ** (snr_db / 20.0))
    return noise_centered * (target_noise_rms / max(noise_rms, 1e-8))


def _augment_components(wave: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    wave = wave.copy()
    angle = float(rng.uniform(0, 2 * np.pi))
    cosine, sine = math.cos(angle), math.sin(angle)
    north, east = wave[1].copy(), wave[2].copy()
    wave[1] = cosine * north - sine * east
    wave[2] = sine * north + cosine * east
    wave *= rng.choice((-1.0, 1.0), size=(3, 1)).astype(np.float32)
    wave *= np.exp(rng.uniform(-0.7, 0.7, size=(3, 1))).astype(np.float32)
    return wave


def _add_sensor_artifacts(wave: np.ndarray, p_sample: int, rng: np.random.Generator) -> np.ndarray:
    wave = wave.copy()
    samples = wave.shape[1]
    time = np.arange(samples, dtype=np.float32) / TARGET_SR
    if rng.random() < 0.35:
        amplitude = _rms(wave) * float(rng.uniform(0.02, 0.25))
        frequency = float(rng.uniform(0.03, 0.3))
        wave += amplitude * np.sin(2 * np.pi * frequency * time)[None, :]
    if rng.random() < 0.20:
        for _ in range(int(rng.integers(1, 5))):
            position = int(rng.integers(0, samples))
            wave[:, position] += rng.normal(0, max(_rms(wave), 1e-6) * rng.uniform(5, 20), 3)
    if rng.random() < 0.15 and samples > 200:
        gap = int(rng.integers(10, 100))
        candidates = np.concatenate((np.arange(0, max(p_sample - 150, 1)), np.arange(min(p_sample + 150, samples), samples)))
        if len(candidates):
            start = int(rng.choice(candidates))
            wave[:, start : min(start + gap, samples)] = 0.0
    if rng.random() < 0.15:
        limit = float(np.quantile(np.abs(wave), rng.uniform(0.96, 0.995)))
        if limit > 0:
            wave = np.clip(wave, -limit, limit)
    return wave


def _relocate_event(
    wave: np.ndarray,
    original_p: int,
    target_p: int,
    noise_bed: np.ndarray,
) -> np.ndarray:
    output = noise_bed.copy()
    shift = target_p - original_p
    source_start = max(0, -shift)
    destination_start = max(0, shift)
    count = min(wave.shape[1] - source_start, output.shape[1] - destination_start)
    if count <= 0:
        return output
    blend = min(50, count, destination_start)
    if blend:
        weight = np.linspace(0.0, 1.0, blend, dtype=np.float32)[None, :]
        output[:, destination_start : destination_start + blend] = (
            output[:, destination_start : destination_start + blend] * (1.0 - weight)
            + wave[:, source_start : source_start + blend] * weight
        )
    output[:, destination_start + blend : destination_start + count] = wave[
        :, source_start + blend : source_start + count
    ]
    return output


def packetize_trace(
    wave: np.ndarray,
    p_sample: float | None,
    max_chunks: int,
    valid_samples: int | None = None,
    label_sigma: float = 1.0,
) -> dict[str, torch.Tensor | int]:
    samples_per_chunk = SAMPLES_PER_CHUNK
    max_samples = max_chunks * samples_per_chunk
    valid_samples = min(wave.shape[1], max_samples) if valid_samples is None else min(valid_samples, max_samples)
    padded = np.zeros((3, max_samples), dtype=np.float32)
    padded[:, :valid_samples] = wave[:, :valid_samples]
    chunks = np.zeros((max_chunks, 3, samples_per_chunk), dtype=np.float32)
    mask = np.zeros(max_chunks, dtype=np.float32)
    state = None
    for packet_index in range(math.ceil(valid_samples / samples_per_chunk)):
        start = packet_index * samples_per_chunk
        count = min(samples_per_chunk, valid_samples - start)
        chunks[packet_index], state = normalize_packet_ewma_causal(
            padded[:, start : start + samples_per_chunk],
            state,
            valid_samples=count,
        )
        mask[packet_index] = 1.0

    labels = np.zeros(max_chunks, dtype=np.float32)
    labels_event = np.zeros(max_chunks, dtype=np.float32)
    target_chunk = -1
    if p_sample is not None and 0 <= p_sample < valid_samples:
        target = float(p_sample) / samples_per_chunk
        target_chunk = int(round(target))
        positions = np.arange(max_chunks, dtype=np.float32)
        labels = np.exp(-0.5 * ((positions - target) / label_sigma) ** 2).astype(np.float32)
        labels *= mask
        labels_event[positions >= math.floor(target)] = 1.0
        labels_event *= mask
    return {
        "chunks": torch.from_numpy(chunks),
        "labels": torch.from_numpy(labels),
        "labels_event": torch.from_numpy(labels_event),
        "mask": torch.from_numpy(mask),
        "target_chunk": target_chunk,
    }


class DomainRobustStreamDataset(Dataset):
    def __init__(
        self,
        sources: dict[str, object],
        records: list[StreamRecord],
        max_chunks: int,
        training: bool,
        noise_source_names: tuple[str, ...] = ("stead_noise",),
        seed: int = 20260715,
        label_sigma: float = 1.0,
        eval_snr_db: float | None = None,
    ):
        self.sources = sources
        self.records = records
        self.max_chunks = max_chunks
        self.training = training
        self.noise_source_names = tuple(name for name in noise_source_names if name in sources)
        self.noise_indices = {}
        for name in self.noise_source_names:
            source = sources[name]
            if hasattr(source, "noise_indices"):
                self.noise_indices[name] = np.asarray(source.noise_indices, dtype=np.int64)
            else:
                metadata = source.metadata
                self.noise_indices[name] = np.flatnonzero(
                    metadata["trace_category"].eq("noise").to_numpy()
                )
        self.seed = seed
        self.epoch = 0
        self.label_sigma = label_sigma
        self.eval_snr_db = eval_snr_db

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.records)

    def _noise_wave(self, rng: np.random.Generator, samples: int) -> np.ndarray:
        if not self.noise_source_names:
            return rng.normal(0, 1, size=(3, samples)).astype(np.float32)
        source_name = self.noise_source_names[int(rng.integers(0, len(self.noise_source_names)))]
        source = self.sources[source_name]
        noise_rows = self.noise_indices[source_name]
        noise_index = int(noise_rows[int(rng.integers(0, len(noise_rows)))])
        noise, _ = source.read(noise_index)
        return _repeat_or_crop(_ensure_three_components(noise), samples, rng)

    def __getitem__(self, item: int) -> dict:
        record = self.records[item]
        rng = np.random.default_rng(self.seed + self.epoch * 1_000_003 + item * 97)
        wave, p_sample = self.sources[record.source].read(record.index)
        wave = _ensure_three_components(wave)
        max_samples = self.max_chunks * SAMPLES_PER_CHUNK
        snr_db = math.nan

        if record.is_event:
            if p_sample is None:
                raise ValueError(f"Event record has no P label: {record}")
            original_p = int(round(p_sample))
            if self.training:
                wave = _augment_components(wave, rng)
                target_p = int(rng.integers(8 * SAMPLES_PER_CHUNK, max_samples - 8 * SAMPLES_PER_CHUNK))
                noise = self._noise_wave(rng, max_samples)
                pre = wave[:, : max(original_p - 50, 1)]
                noise *= max(_rms(pre), 1e-8) / max(_rms(noise), 1e-8)
                wave = _relocate_event(wave, original_p, target_p, noise)
                # Learn the onset first, then expose the model to the hardest
                # domestic-network noise levels. Later epochs retain native
                # traces so robustness does not erase clean-signal precision.
                draw = rng.random()
                if draw < 0.10:
                    added_noise = 0.0
                else:
                    if self.epoch <= 1:
                        snr_db = float(rng.uniform(5.0, 25.0))
                    elif self.epoch == 2:
                        snr_db = float(rng.uniform(0.0, 20.0))
                    elif draw < 0.45:
                        snr_db = float(rng.uniform(-10.0, 0.0))
                    elif draw < 0.75:
                        snr_db = float(rng.uniform(0.0, 10.0))
                    else:
                        snr_db = float(rng.uniform(10.0, 20.0))
                    added_noise = _scale_noise_to_snr(
                        wave, self._noise_wave(rng, max_samples), target_p, snr_db
                    )
                wave = _add_sensor_artifacts(wave + added_noise, target_p, rng)
                p_sample = float(target_p)
                valid_samples = max_samples
            else:
                valid_samples = min(wave.shape[1], max_samples)
                if self.eval_snr_db is not None:
                    noise = self._noise_wave(rng, valid_samples)
                    wave = wave[:, :valid_samples] + _scale_noise_to_snr(
                        wave[:, :valid_samples], noise, original_p, self.eval_snr_db
                    )
                    snr_db = float(self.eval_snr_db)
        else:
            p_sample = None
            if self.training:
                wave = _repeat_or_crop(wave, max_samples, rng)
                wave = _add_sensor_artifacts(_augment_components(wave, rng), max_samples // 2, rng)
                valid_samples = max_samples
            else:
                valid_samples = min(wave.shape[1], max_samples)

        packed = packetize_trace(
            wave,
            p_sample,
            self.max_chunks,
            valid_samples=valid_samples,
            label_sigma=self.label_sigma,
        )
        packed.update(
            {
                "is_event": int(record.is_event),
                "source": record.source,
                "trace_index": record.index,
                "snr_db": float(snr_db),
            }
        )
        return packed


def collate_domain_robust(batch: list[dict]) -> dict:
    tensor_keys = ("chunks", "labels", "labels_event", "mask")
    output = {key: torch.stack([sample[key] for sample in batch]) for key in tensor_keys}
    output["target_chunk"] = torch.tensor([sample["target_chunk"] for sample in batch], dtype=torch.long)
    output["is_event"] = torch.tensor([sample["is_event"] for sample in batch], dtype=torch.bool)
    output["trace_index"] = torch.tensor([sample["trace_index"] for sample in batch], dtype=torch.long)
    output["snr_db"] = torch.tensor([sample["snr_db"] for sample in batch], dtype=torch.float32)
    output["source"] = [sample["source"] for sample in batch]
    return output


__all__ = [
    "DomainRobustStreamDataset",
    "KNETSource",
    "PrePNoiseSource",
    "SeisBenchSource",
    "StreamRecord",
    "collate_domain_robust",
    "deterministic_limit",
    "event_indices",
    "find_p_column",
    "instancegm_acceleration_indices",
    "knet_factorized_indices",
    "make_records",
    "packetize_trace",
    "stable_bucket",
    "station_key",
]
