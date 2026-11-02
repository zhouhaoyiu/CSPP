"""Causal packet picker with explicit amplitude cues and sub-packet timing."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_count(channels: int, maximum: int = 8) -> int:
    for groups in range(min(maximum, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class PacketPhysicsFeatures(nn.Module):
    out_channels = 8

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1] != 3:
            raise ValueError(f"Expected (B, 3, L), got {tuple(x.shape)}")
        z, n, e = x.unbind(dim=1)
        horizontal = torch.sqrt(n.square() + e.square() + 1e-6)
        total = torch.sqrt(z.square() + horizontal.square() + 1e-6)
        z_fraction = z.square() / (total.square() + 1e-6)
        dz = F.pad(z[:, 1:] - z[:, :-1], (1, 0))
        dh = F.pad(horizontal[:, 1:] - horizontal[:, :-1], (1, 0))
        return torch.stack((z, n, e, horizontal, total, z_fraction, dz, dh), dim=1)


class PacketSummaryFeatures(nn.Module):
    """Amplitude, frequency-shape, and component cues that bypass packet norm."""

    out_features = 10

    def forward(self, packet: torch.Tensor) -> torch.Tensor:
        eps = 1e-6
        z, n, e = packet.unbind(dim=1)
        horizontal = torch.sqrt(n.square() + e.square() + eps)
        total = torch.sqrt(z.square() + horizontal.square() + eps)
        rms_total = torch.sqrt(total.square().mean(dim=-1) + eps)
        rms_z = torch.sqrt(z.square().mean(dim=-1) + eps)
        rms_h = torch.sqrt(horizontal.square().mean(dim=-1) + eps)
        peak = total.amax(dim=-1)
        crest = torch.clamp(peak / rms_total, 0.0, 20.0) / 20.0
        derivative = total[:, 1:] - total[:, :-1]
        derivative_ratio = torch.clamp(
            torch.sqrt(derivative.square().mean(dim=-1) + eps) / rms_total,
            0.0,
            5.0,
        ) / 5.0
        z_fraction = torch.clamp(rms_z.square() / (rms_total.square() + eps), 0.0, 1.0)
        rms_n = torch.sqrt(n.square().mean(dim=-1) + eps)
        rms_e = torch.sqrt(e.square().mean(dim=-1) + eps)
        horizontal_balance = (rms_n - rms_e) / (rms_n + rms_e + eps)

        total_1d = total.unsqueeze(1)
        low5 = F.avg_pool1d(F.pad(total_1d, (4, 0), mode="replicate"), 5, stride=1).squeeze(1)
        low15 = F.avg_pool1d(F.pad(total_1d, (14, 0), mode="replicate"), 15, stride=1).squeeze(1)
        high_ratio = torch.clamp(
            torch.sqrt((total - low5).square().mean(dim=-1) + eps) / rms_total,
            0.0,
            3.0,
        ) / 3.0
        mid_ratio = torch.clamp(
            torch.sqrt((low5 - low15).square().mean(dim=-1) + eps) / rms_total,
            0.0,
            3.0,
        ) / 3.0
        clipped = (packet.abs() >= 11.9).to(packet.dtype).mean(dim=(1, 2))
        return torch.stack(
            (
                torch.log1p(rms_total),
                torch.log1p(rms_z),
                torch.log1p(rms_h),
                torch.log1p(peak),
                crest,
                derivative_ratio,
                z_fraction,
                horizontal_balance,
                high_ratio + mid_ratio,
                clipped,
            ),
            dim=1,
        )


class PacketCausalBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float):
        super().__init__()
        self.pad = 2 * dilation
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            dilation=dilation,
            groups=channels,
        )
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1)
        self.norm = nn.GroupNorm(_group_count(channels), channels)
        self.dropout = nn.Dropout(dropout)
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.depthwise(F.pad(x, (self.pad, 0)))
        y = self.pointwise(y)
        return x + self.scale * self.dropout(F.silu(self.norm(y)))


class PacketEncoder(nn.Module):
    def __init__(self, width: int = 48, embedding_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.features = PacketPhysicsFeatures()
        self.summary = PacketSummaryFeatures()
        self.stem = nn.Conv1d(self.features.out_channels, width, kernel_size=1)
        self.stem_norm = nn.GroupNorm(_group_count(width), width)
        self.blocks = nn.ModuleList(
            PacketCausalBlock(width, dilation=d, dropout=dropout) for d in (1, 2, 4, 8)
        )
        self.projection = nn.Sequential(
            nn.Linear(width * 2 + self.summary.out_features, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.SiLU(),
        )

    def forward(self, packet: torch.Tensor) -> torch.Tensor:
        summary = self.summary(packet)
        x = F.silu(self.stem_norm(self.stem(self.features(packet))))
        for block in self.blocks:
            x = block(x)
        pooled = torch.cat((x.mean(dim=-1), x.amax(dim=-1), summary), dim=-1)
        return self.projection(pooled)


class CausalChannelNorm(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        centered = x - x.mean(dim=1, keepdim=True)
        variance = centered.square().mean(dim=1, keepdim=True)
        return centered * torch.rsqrt(variance + self.eps) * self.weight + self.bias


class SequenceCausalBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float):
        super().__init__()
        self.pad = 2 * dilation
        self.channels = channels
        self.norm = CausalChannelNorm(channels)
        self.temporal = nn.Conv1d(channels, channels, kernel_size=3, dilation=dilation)
        self.pointwise = nn.Conv1d(channels, channels * 2, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.temporal(F.pad(self.norm(x), (self.pad, 0)))
        value, gate = self.pointwise(y).chunk(2, dim=1)
        return x + self.scale * self.dropout(torch.tanh(value) * torch.sigmoid(gate))


@dataclass(frozen=True)
class DomainRobustV5Config:
    packet_width: int = 48
    embedding_dim: int = 64
    sequence_width: int = 64
    dropout: float = 0.1
    sequence_dilations: tuple[int, ...] = (1, 2, 4, 8, 16)

    def as_dict(self) -> dict:
        return {
            "packet_width": self.packet_width,
            "embedding_dim": self.embedding_dim,
            "sequence_width": self.sequence_width,
            "dropout": self.dropout,
            "sequence_dilations": list(self.sequence_dilations),
        }


class DomainRobustCausalPickerV5(nn.Module):
    """Packet encoder, bounded causal TCN, and onset/offset/reliability heads."""

    def __init__(
        self,
        packet_width: int = 48,
        embedding_dim: int = 64,
        sequence_width: int = 64,
        dropout: float = 0.1,
        sequence_dilations: tuple[int, ...] | list[int] = (1, 2, 4, 8, 16),
    ):
        super().__init__()
        self.config = DomainRobustV5Config(
            packet_width=packet_width,
            embedding_dim=embedding_dim,
            sequence_width=sequence_width,
            dropout=dropout,
            sequence_dilations=tuple(sequence_dilations),
        )
        self.encoder = PacketEncoder(packet_width, embedding_dim, dropout)
        self.sequence_input = nn.Linear(embedding_dim, sequence_width)
        self.sequence_blocks = nn.ModuleList(
            SequenceCausalBlock(sequence_width, dilation=d, dropout=dropout)
            for d in self.config.sequence_dilations
        )
        self.sequence_norm = CausalChannelNorm(sequence_width)
        self.head_p = nn.Conv1d(sequence_width, 1, kernel_size=1)
        self.head_event = nn.Conv1d(sequence_width, 1, kernel_size=1)
        self.head_offset = nn.Conv1d(sequence_width, 1, kernel_size=1)
        nn.init.constant_(self.head_p.bias, -4.0)
        nn.init.constant_(self.head_event.bias, -2.0)
        nn.init.zeros_(self.head_offset.bias)

    @property
    def receptive_field_packets(self) -> int:
        return 1 + 2 * sum(self.config.sequence_dilations)

    def encode_packets(self, chunks: torch.Tensor) -> torch.Tensor:
        if chunks.ndim != 4 or chunks.shape[2] != 3:
            raise ValueError(f"Expected (B, T, 3, L), got {tuple(chunks.shape)}")
        batch, packets, channels, samples = chunks.shape
        encoded = self.encoder(chunks.reshape(batch * packets, channels, samples))
        return encoded.reshape(batch, packets, -1)

    def forward_embeddings(self, embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.sequence_input(embeddings).transpose(1, 2).contiguous()
        for block in self.sequence_blocks:
            x = block(x)
        x = self.sequence_norm(x)
        return (
            self.head_p(x).squeeze(1),
            self.head_event(x).squeeze(1),
            self.head_offset(x).squeeze(1),
        )

    def forward(self, chunks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.forward_embeddings(self.encode_packets(chunks))

    @torch.inference_mode()
    def forward_streaming_packet(
        self,
        packet: torch.Tensor,
        state: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        if packet.ndim == 2:
            packet = packet.unsqueeze(0)
        if packet.ndim != 3 or packet.shape[0] != 1:
            raise ValueError("Streaming inference expects (3, L) or (1, 3, L)")
        current = self.encoder(packet).unsqueeze(1)
        cached = None if state is None else state.get("embeddings")
        if cached is not None:
            current = torch.cat((cached.to(current.device), current), dim=1)
        current = current[:, -self.receptive_field_packets :]
        logits_p, logits_event, logits_offset = self.forward_embeddings(current)
        next_state = {"embeddings": current.detach()}
        return logits_p[0, -1], logits_event[0, -1], logits_offset[0, -1], next_state

    @classmethod
    def from_config(cls, config: dict) -> "DomainRobustCausalPickerV5":
        keys = DomainRobustV5Config.__dataclass_fields__.keys()
        values = {key: config[key] for key in keys if key in config}
        if "sequence_dilations" in values:
            values["sequence_dilations"] = tuple(values["sequence_dilations"])
        return cls(**values)


__all__ = ["DomainRobustCausalPickerV5", "DomainRobustV5Config"]
