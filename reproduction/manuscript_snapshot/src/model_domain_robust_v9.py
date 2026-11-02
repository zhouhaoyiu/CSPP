"""V8 packet-once picker with fixed causal within-packet spectral cues."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn

from model_domain_robust_v8 import DomainRobustCausalPickerV8


class FixedPacketSpectrum(nn.Module):
    """DFT band shape from one fully arrived packet, implemented as fixed matrices."""

    out_features = 31

    def __init__(
        self,
        packet_samples: int = 50,
        sample_rate_hz: float = 100.0,
        band_edges_hz: tuple[float, ...] = (1.0, 4.0, 8.0, 16.0, 30.0, 50.1),
    ):
        super().__init__()
        if packet_samples < 4:
            raise ValueError("packet_samples must be at least four")
        if len(band_edges_hz) < 3:
            raise ValueError("At least two spectral bands are required")
        self.packet_samples = int(packet_samples)
        self.sample_rate_hz = float(sample_rate_hz)
        self.band_edges_hz = tuple(float(value) for value in band_edges_hz)

        samples = torch.arange(self.packet_samples, dtype=torch.float32)
        bins = torch.arange(
            self.packet_samples // 2 + 1, dtype=torch.float32
        )
        angles = 2.0 * math.pi * bins[:, None] * samples[None, :] / self.packet_samples
        window = torch.hann_window(self.packet_samples, periodic=False)
        frequencies = bins * self.sample_rate_hz / self.packet_samples
        masks = []
        centers = []
        for low, high in zip(
            self.band_edges_hz[:-1], self.band_edges_hz[1:], strict=True
        ):
            mask = ((frequencies >= low) & (frequencies < high)).to(torch.float32)
            if not bool(mask.any()):
                raise ValueError(f"Spectral band [{low}, {high}) has no DFT bins")
            masks.append(mask)
            centers.append(float(frequencies[mask.bool()].mean()))
        self.register_buffer("cosine", torch.cos(angles) * window[None, :])
        self.register_buffer("sine", -torch.sin(angles) * window[None, :])
        self.register_buffer("band_masks", torch.stack(masks))
        self.register_buffer("band_centers", torch.tensor(centers, dtype=torch.float32))
        self.register_buffer("window_energy", window.square().sum().clamp_min(1e-6))

    @property
    def band_count(self) -> int:
        return len(self.band_edges_hz) - 1

    def forward(self, packet: torch.Tensor) -> torch.Tensor:
        if packet.ndim != 3 or packet.shape[1] != 3:
            raise ValueError(f"Expected packet shape (B, 3, L), got {tuple(packet.shape)}")
        if packet.shape[-1] != self.packet_samples:
            raise ValueError(
                f"Expected {self.packet_samples} packet samples, got {packet.shape[-1]}"
            )
        eps = 1e-8
        centered = packet - packet.mean(dim=-1, keepdim=True)
        real = centered @ self.cosine.transpose(0, 1)
        imaginary = centered @ self.sine.transpose(0, 1)
        power = (real.square() + imaginary.square()) / self.window_energy
        component_bands = power @ self.band_masks.transpose(0, 1)
        z_bands = component_bands[:, 0]
        h_bands = component_bands[:, 1] + component_bands[:, 2]
        total_bands = z_bands + h_bands

        def fractions(values: torch.Tensor) -> torch.Tensor:
            return values / (values.sum(dim=-1, keepdim=True) + eps)

        z_fraction = fractions(z_bands)
        h_fraction = fractions(h_bands)
        total_fraction = fractions(total_bands)
        vertical_by_band = z_bands / (total_bands + eps)
        log_total_bands = torch.log1p(total_bands)

        def entropy(values: torch.Tensor) -> torch.Tensor:
            return -(
                values * torch.log(values + eps)
            ).sum(dim=-1) / math.log(self.band_count)

        entropies = torch.stack(
            (
                entropy(z_fraction),
                entropy(h_fraction),
                entropy(total_fraction),
            ),
            dim=-1,
        )
        centers = self.band_centers / (0.5 * self.sample_rate_hz)
        centroids = torch.stack(
            (
                (z_fraction * centers).sum(dim=-1),
                (h_fraction * centers).sum(dim=-1),
                (total_fraction * centers).sum(dim=-1),
            ),
            dim=-1,
        )
        features = torch.cat(
            (
                z_fraction,
                h_fraction,
                total_fraction,
                vertical_by_band,
                log_total_bands,
                entropies,
                centroids,
            ),
            dim=-1,
        )
        if features.shape[-1] != self.out_features:
            raise RuntimeError(f"Unexpected spectral feature width {features.shape[-1]}")
        return features


@dataclass(frozen=True)
class DomainRobustV9Config:
    packet_width: int = 56
    embedding_dim: int = 96
    sequence_width: int = 96
    temporal_bins: int = 5
    dropout: float = 0.1
    sequence_dilations: tuple[int, ...] = (1, 2, 4, 8, 16)
    verifier_dilations: tuple[int, ...] = (1, 2)
    max_arrival_age_packets: float = 4.0
    packet_samples: int = 50
    sample_rate_hz: float = 100.0
    spectral_band_edges_hz: tuple[float, ...] = (1.0, 4.0, 8.0, 16.0, 30.0, 50.1)

    def as_dict(self) -> dict:
        return {
            "packet_width": self.packet_width,
            "embedding_dim": self.embedding_dim,
            "sequence_width": self.sequence_width,
            "temporal_bins": self.temporal_bins,
            "dropout": self.dropout,
            "sequence_dilations": list(self.sequence_dilations),
            "verifier_dilations": list(self.verifier_dilations),
            "max_arrival_age_packets": self.max_arrival_age_packets,
            "packet_samples": self.packet_samples,
            "sample_rate_hz": self.sample_rate_hz,
            "spectral_band_edges_hz": list(self.spectral_band_edges_hz),
        }


class DomainRobustCausalPickerV9(DomainRobustCausalPickerV8):
    """Add a residual fixed-spectrum token without changing temporal state."""

    model_class = "DomainRobustCausalPickerV9"

    def __init__(
        self,
        packet_width: int = 56,
        embedding_dim: int = 96,
        sequence_width: int = 96,
        temporal_bins: int = 5,
        dropout: float = 0.1,
        sequence_dilations: tuple[int, ...] | list[int] = (1, 2, 4, 8, 16),
        verifier_dilations: tuple[int, ...] | list[int] = (1, 2),
        max_arrival_age_packets: float = 4.0,
        packet_samples: int = 50,
        sample_rate_hz: float = 100.0,
        spectral_band_edges_hz: tuple[float, ...] | list[float] = (
            1.0,
            4.0,
            8.0,
            16.0,
            30.0,
            50.1,
        ),
    ):
        super().__init__(
            packet_width=packet_width,
            embedding_dim=embedding_dim,
            sequence_width=sequence_width,
            temporal_bins=temporal_bins,
            dropout=dropout,
            sequence_dilations=sequence_dilations,
            verifier_dilations=verifier_dilations,
            max_arrival_age_packets=max_arrival_age_packets,
        )
        self.config = DomainRobustV9Config(
            packet_width=packet_width,
            embedding_dim=embedding_dim,
            sequence_width=sequence_width,
            temporal_bins=temporal_bins,
            dropout=dropout,
            sequence_dilations=tuple(sequence_dilations),
            verifier_dilations=tuple(verifier_dilations),
            max_arrival_age_packets=float(max_arrival_age_packets),
            packet_samples=int(packet_samples),
            sample_rate_hz=float(sample_rate_hz),
            spectral_band_edges_hz=tuple(spectral_band_edges_hz),
        )
        self.spectral_features = FixedPacketSpectrum(
            packet_samples=self.config.packet_samples,
            sample_rate_hz=self.config.sample_rate_hz,
            band_edges_hz=self.config.spectral_band_edges_hz,
        )
        self.spectral_projection = nn.Sequential(
            nn.Linear(self.spectral_features.out_features, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.SiLU(),
        )
        self.spectral_scale = nn.Parameter(torch.tensor(0.1))

    def _encode_packet_features(self, packet: torch.Tensor) -> torch.Tensor:
        morphology = self.encoder(packet)
        spectral = self.spectral_projection(self.spectral_features(packet))
        return morphology + self.spectral_scale * spectral

    @classmethod
    def from_config(cls, config: dict) -> "DomainRobustCausalPickerV9":
        keys = DomainRobustV9Config.__dataclass_fields__.keys()
        values = {key: config[key] for key in keys if key in config}
        for key in (
            "sequence_dilations",
            "verifier_dilations",
            "spectral_band_edges_hz",
        ):
            if key in values:
                values[key] = tuple(values[key])
        return cls(**values)


__all__ = [
    "DomainRobustCausalPickerV9",
    "DomainRobustV9Config",
    "FixedPacketSpectrum",
]
