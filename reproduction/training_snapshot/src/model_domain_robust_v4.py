"""Position-free causal packet picker for domain-robust streaming experiments."""
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
    """Orientation-tolerant sample features computed inside one received packet."""

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
        y = self.dropout(F.silu(self.norm(y)))
        return x + self.scale * y


class PacketEncoder(nn.Module):
    def __init__(self, width: int = 48, embedding_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.features = PacketPhysicsFeatures()
        self.stem = nn.Conv1d(self.features.out_channels, width, kernel_size=1)
        self.stem_norm = nn.GroupNorm(_group_count(width), width)
        self.blocks = nn.ModuleList(
            PacketCausalBlock(width, dilation=d, dropout=dropout)
            for d in (1, 2, 4, 8)
        )
        self.projection = nn.Sequential(
            nn.Linear(width * 2, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.SiLU(),
        )

    def forward(self, packet: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.stem_norm(self.stem(self.features(packet))))
        for block in self.blocks:
            x = block(x)
        pooled = torch.cat((x.mean(dim=-1), x.amax(dim=-1)), dim=-1)
        return self.projection(pooled)


class CausalChannelNorm(nn.Module):
    """Normalize channels independently at each packet time without transposes."""

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
        self.temporal = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            dilation=dilation,
        )
        self.pointwise = nn.Conv1d(channels, channels * 2, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.temporal(F.pad(self.norm(x), (self.pad, 0)))
        y = self.pointwise(y)
        value = y[:, : self.channels].contiguous()
        gate = y[:, self.channels :].contiguous()
        y = (torch.tanh(value) * torch.sigmoid(gate)).contiguous()
        return x + self.scale * self.dropout(y)


@dataclass(frozen=True)
class DomainRobustV4Config:
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


class DomainRobustCausalPickerV4(nn.Module):
    """Packet encoder plus bounded-context causal TCN.

    There is no packet index, absolute position embedding, recurrent state, or
    batch-dependent normalization. Streaming keeps only the TCN receptive field.
    """

    def __init__(
        self,
        packet_width: int = 48,
        embedding_dim: int = 64,
        sequence_width: int = 64,
        dropout: float = 0.1,
        sequence_dilations: tuple[int, ...] | list[int] = (1, 2, 4, 8, 16),
    ):
        super().__init__()
        self.config = DomainRobustV4Config(
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
        nn.init.constant_(self.head_p.bias, -4.0)
        nn.init.constant_(self.head_event.bias, -2.0)

    @property
    def receptive_field_packets(self) -> int:
        return 1 + 2 * sum(self.config.sequence_dilations)

    def encode_packets(self, chunks: torch.Tensor) -> torch.Tensor:
        if chunks.ndim != 4 or chunks.shape[2] != 3:
            raise ValueError(f"Expected (B, T, 3, L), got {tuple(chunks.shape)}")
        batch, packets, channels, samples = chunks.shape
        encoded = self.encoder(chunks.reshape(batch * packets, channels, samples))
        return encoded.reshape(batch, packets, -1)

    def forward_embeddings(self, embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.sequence_input(embeddings).transpose(1, 2).contiguous()
        for block in self.sequence_blocks:
            x = block(x)
        x = self.sequence_norm(x)
        return self.head_p(x).squeeze(1), self.head_event(x).squeeze(1)

    def forward(self, chunks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward_embeddings(self.encode_packets(chunks))

    @torch.inference_mode()
    def forward_streaming_packet(
        self,
        packet: torch.Tensor,
        state: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        if packet.ndim == 2:
            packet = packet.unsqueeze(0)
        if packet.ndim != 3 or packet.shape[0] != 1:
            raise ValueError("Streaming inference expects one packet with shape (3, L) or (1, 3, L)")

        current = self.encoder(packet).unsqueeze(1)
        cached = None if state is None else state.get("embeddings")
        if cached is not None:
            current = torch.cat((cached.to(current.device), current), dim=1)
        current = current[:, -self.receptive_field_packets :]
        logits_p, logits_event = self.forward_embeddings(current)
        next_state = {"embeddings": current.detach()}
        return logits_p[0, -1], logits_event[0, -1], next_state

    @classmethod
    def from_config(cls, config: dict) -> "DomainRobustCausalPickerV4":
        keys = DomainRobustV4Config.__dataclass_fields__.keys()
        values = {key: config[key] for key in keys if key in config}
        if "sequence_dilations" in values:
            values["sequence_dilations"] = tuple(values["sequence_dilations"])
        return cls(**values)


__all__ = ["DomainRobustCausalPickerV4", "DomainRobustV4Config"]
