"""Lightweight CSPP variant with packet-local P supervision."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_cspp_v25 import (
    CSPPCausalPickerV25,
    CSPPPacketEncoderV25,
    CSPPV25Config,
)
from model_domain_robust_v5 import _group_count


class PacketLocalizingEncoderV26(CSPPPacketEncoderV25):
    """Reuse V25 features and expose a dense P score inside the current packet."""

    def __init__(self, config: CSPPV25Config):
        super().__init__(config)
        width = config.packet_width
        self.phase_head = nn.Sequential(
            nn.GroupNorm(_group_count(width), width),
            nn.Conv1d(
                width,
                width,
                kernel_size=3,
                padding=1,
                groups=width,
            ),
            nn.SiLU(),
            nn.Conv1d(width, 1, kernel_size=1),
        )

    def forward_with_phase(
        self,
        packet: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if packet.ndim != 3 or packet.shape[1:] != (
            3,
            self.config.packet_samples,
        ):
            raise ValueError(
                "Expected packet shape "
                f"(B, 3, {self.config.packet_samples}), got {tuple(packet.shape)}"
            )
        values = F.silu(self.stem_norm(self.stem(self.physics(packet))))
        for block in self.blocks:
            values = block(values)
        average = F.adaptive_avg_pool1d(values, self.config.temporal_bins)
        maximum = F.adaptive_max_pool1d(values, self.config.temporal_bins)
        features = torch.cat(
            (
                average.flatten(1),
                maximum.flatten(1),
                self.summary(packet),
                self.spectrum(packet),
            ),
            dim=1,
        )
        return self.projection(features), self.phase_head(values).squeeze(1)

    def forward(self, packet: torch.Tensor) -> torch.Tensor:
        embedding, _ = self.forward_with_phase(packet)
        return embedding


class CSPPCausalPickerV26(CSPPCausalPickerV25):
    """V25 plus PhaseNet-style dense timing supervision inside one packet."""

    model_class = "CSPPCausalPickerV26"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.encoder = PacketLocalizingEncoderV26(self.config)

    def forward_with_aux(
        self,
        chunks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if (
            chunks.ndim != 4
            or chunks.shape[2] != 3
            or chunks.shape[3] != self.config.packet_samples
        ):
            raise ValueError(
                "Expected chunks shape "
                f"(B, T, 3, {self.config.packet_samples}), got {tuple(chunks.shape)}"
            )
        batch, packets, channels, samples = chunks.shape
        features, phase_logits = self.encoder.forward_with_phase(
            chunks.reshape(batch * packets, channels, samples)
        )
        features = features.reshape(batch, packets, -1)
        phase_logits = phase_logits.reshape(batch, packets, samples)
        tokens, _ = self._tokens_from_features(features)
        local = tokens.transpose(1, 2).contiguous()
        for block in self.sequence_blocks:
            local = block(local)
        local = self.sequence_norm(local).transpose(1, 2).contiguous()
        fused, _ = self._fuse_sequence(tokens, local, None)
        return (*self._heads(tokens, local, fused), phase_logits)

    def forward(
        self,
        chunks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        candidate, event, delay, _ = self.forward_with_aux(chunks)
        return candidate, event, delay

    @classmethod
    def from_config(cls, config: dict) -> "CSPPCausalPickerV26":
        keys = CSPPV25Config.__dataclass_fields__.keys()
        values = {key: config[key] for key in keys if key in config}
        for key in (
            "packet_kernels",
            "sequence_dilations",
            "retention_half_lives_seconds",
            "spectral_band_edges_hz",
        ):
            if key in values:
                values[key] = tuple(values[key])
        return cls(**values)


__all__ = [
    "CSPPCausalPickerV26",
    "PacketLocalizingEncoderV26",
]
