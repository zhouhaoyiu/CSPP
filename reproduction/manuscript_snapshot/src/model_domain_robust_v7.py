"""Two-stage causal picker with explicit packet-change and arrival-age cues."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_domain_robust_v5 import (
    CausalChannelNorm,
    PacketCausalBlock,
    PacketPhysicsFeatures,
    PacketSummaryFeatures,
    SequenceCausalBlock,
    _group_count,
)


class ChangeContextPacketEncoder(nn.Module):
    """Encode the current packet and explicit changes from its causal history."""

    def __init__(
        self,
        width: int = 64,
        embedding_dim: int = 96,
        context_packets: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        if context_packets < 2:
            raise ValueError("v7 requires at least two context packets")
        self.context_packets = int(context_packets)
        self.features = PacketPhysicsFeatures()
        self.summary = PacketSummaryFeatures()
        self.stem = nn.Conv1d(self.features.out_channels, width, kernel_size=1)
        self.stem_norm = nn.GroupNorm(_group_count(width), width)
        self.blocks = nn.ModuleList(
            PacketCausalBlock(width, dilation=dilation, dropout=dropout)
            for dilation in (1, 2, 4, 8)
        )
        summary_width = self.summary.out_features * 4
        self.projection = nn.Sequential(
            nn.Linear(width * 4 + summary_width, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.SiLU(),
        )

    def forward(
        self,
        packet: torch.Tensor,
        previous_packets: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if packet.ndim != 3 or packet.shape[1] != 3:
            raise ValueError(f"Expected packet shape (B, 3, L), got {tuple(packet.shape)}")
        batch, _, samples = packet.shape
        history_count = self.context_packets - 1
        if previous_packets is None:
            previous_packets = packet.new_zeros(batch, history_count, 3, samples)
        expected = (batch, history_count, 3, samples)
        if tuple(previous_packets.shape) != expected:
            raise ValueError(
                f"Expected previous_packets shape {expected}, got {tuple(previous_packets.shape)}"
            )

        packet_stack = torch.cat((previous_packets, packet.unsqueeze(1)), dim=1)
        summaries = self.summary(packet_stack.reshape(-1, 3, samples)).reshape(
            batch, self.context_packets, -1
        )
        current_summary = summaries[:, -1]
        history_mean = summaries[:, :-1].mean(dim=1)
        recent_summary = summaries[:, -2]
        summary_features = torch.cat(
            (
                current_summary,
                history_mean,
                current_summary - history_mean,
                current_summary - recent_summary,
            ),
            dim=1,
        )

        history = previous_packets.permute(0, 2, 1, 3).reshape(
            batch, 3, history_count * samples
        )
        context = torch.cat((history, packet), dim=-1)
        encoded = F.silu(self.stem_norm(self.stem(self.features(context))))
        for block in self.blocks:
            encoded = block(encoded)
        current = encoded[:, :, -samples:]
        pooled = torch.cat(
            (
                encoded.mean(dim=-1),
                encoded.amax(dim=-1),
                current.mean(dim=-1),
                current.amax(dim=-1),
                summary_features,
            ),
            dim=1,
        )
        return self.projection(pooled)


@dataclass(frozen=True)
class DomainRobustV7Config:
    packet_width: int = 64
    embedding_dim: int = 96
    sequence_width: int = 96
    context_packets: int = 3
    dropout: float = 0.1
    sequence_dilations: tuple[int, ...] = (1, 2, 4, 8, 16)
    verifier_dilations: tuple[int, ...] = (1, 2)
    max_arrival_age_packets: float = 4.0

    def as_dict(self) -> dict:
        return {
            "packet_width": self.packet_width,
            "embedding_dim": self.embedding_dim,
            "sequence_width": self.sequence_width,
            "context_packets": self.context_packets,
            "dropout": self.dropout,
            "sequence_dilations": list(self.sequence_dilations),
            "verifier_dilations": list(self.verifier_dilations),
            "max_arrival_age_packets": self.max_arrival_age_packets,
        }


class DomainRobustCausalPickerV7(nn.Module):
    """Candidate head, causal confirmation head, and arrival-age regression."""

    model_class = "DomainRobustCausalPickerV7"
    timing_mode = "arrival_age"

    def __init__(
        self,
        packet_width: int = 64,
        embedding_dim: int = 96,
        sequence_width: int = 96,
        context_packets: int = 3,
        dropout: float = 0.1,
        sequence_dilations: tuple[int, ...] | list[int] = (1, 2, 4, 8, 16),
        verifier_dilations: tuple[int, ...] | list[int] = (1, 2),
        max_arrival_age_packets: float = 4.0,
    ):
        super().__init__()
        self.config = DomainRobustV7Config(
            packet_width=packet_width,
            embedding_dim=embedding_dim,
            sequence_width=sequence_width,
            context_packets=context_packets,
            dropout=dropout,
            sequence_dilations=tuple(sequence_dilations),
            verifier_dilations=tuple(verifier_dilations),
            max_arrival_age_packets=float(max_arrival_age_packets),
        )
        self.encoder = ChangeContextPacketEncoder(
            packet_width,
            embedding_dim,
            context_packets,
            dropout,
        )
        self.sequence_input = nn.Linear(embedding_dim, sequence_width)
        self.sequence_blocks = nn.ModuleList(
            SequenceCausalBlock(sequence_width, dilation=dilation, dropout=dropout)
            for dilation in self.config.sequence_dilations
        )
        self.sequence_norm = CausalChannelNorm(sequence_width)
        self.verifier_blocks = nn.ModuleList(
            SequenceCausalBlock(sequence_width, dilation=dilation, dropout=dropout)
            for dilation in self.config.verifier_dilations
        )
        self.head_p = nn.Conv1d(sequence_width, 1, kernel_size=1)
        self.head_event = nn.Conv1d(sequence_width, 1, kernel_size=1)
        self.head_delay = nn.Conv1d(sequence_width, 1, kernel_size=1)
        nn.init.constant_(self.head_p.bias, -4.0)
        nn.init.constant_(self.head_event.bias, -4.0)
        nn.init.zeros_(self.head_delay.bias)

    @property
    def sequence_cache_packets(self) -> int:
        return 1 + 2 * sum(self.config.sequence_dilations) + 2 * sum(
            self.config.verifier_dilations
        )

    @property
    def receptive_field_packets(self) -> int:
        return self.sequence_cache_packets + self.config.context_packets - 1

    @property
    def max_arrival_age_packets(self) -> float:
        return self.config.max_arrival_age_packets

    @property
    def head_offset(self) -> nn.Conv1d:
        """Compatibility accessor for exporters that name the third head offset."""
        return self.head_delay

    def encode_packets(self, chunks: torch.Tensor) -> torch.Tensor:
        if chunks.ndim != 4 or chunks.shape[2] != 3:
            raise ValueError(f"Expected chunks shape (B, T, 3, L), got {tuple(chunks.shape)}")
        batch, packets, channels, samples = chunks.shape
        history_count = self.config.context_packets - 1
        padded = F.pad(chunks, (0, 0, 0, 0, history_count, 0))
        histories = [padded[:, offset : offset + packets] for offset in range(history_count)]
        previous = torch.stack(histories, dim=2)
        encoded = self.encoder(
            chunks.reshape(batch * packets, channels, samples),
            previous.reshape(batch * packets, history_count, channels, samples),
        )
        return encoded.reshape(batch, packets, -1)

    def forward_embeddings(
        self,
        embeddings: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.sequence_input(embeddings).transpose(1, 2).contiguous()
        for block in self.sequence_blocks:
            x = block(x)
        x = self.sequence_norm(x)
        verifier = x
        for block in self.verifier_blocks:
            verifier = block(verifier)
        return (
            self.head_p(x).squeeze(1),
            self.head_event(verifier).squeeze(1),
            self.head_delay(verifier).squeeze(1),
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
        history_count = self.config.context_packets - 1
        previous = None if state is None else state.get("previous_packets")
        if previous is None:
            previous = packet.new_zeros(1, history_count, 3, packet.shape[-1])
        current_embedding = self.encoder(packet, previous).unsqueeze(1)
        cached = None if state is None else state.get("embeddings")
        embeddings = (
            current_embedding
            if cached is None
            else torch.cat((cached.to(current_embedding.device), current_embedding), dim=1)
        )
        embeddings = embeddings[:, -self.sequence_cache_packets :]
        logits_p, logits_event, logits_delay = self.forward_embeddings(embeddings)
        next_previous = torch.cat((previous[:, 1:], packet.unsqueeze(1)), dim=1)
        next_state = {
            "embeddings": embeddings.detach(),
            "previous_packets": next_previous.detach(),
        }
        return (
            logits_p[0, -1],
            logits_event[0, -1],
            logits_delay[0, -1],
            next_state,
        )

    @classmethod
    def from_config(cls, config: dict) -> "DomainRobustCausalPickerV7":
        keys = DomainRobustV7Config.__dataclass_fields__.keys()
        values = {key: config[key] for key in keys if key in config}
        for key in ("sequence_dilations", "verifier_dilations"):
            if key in values:
                values[key] = tuple(values[key])
        return cls(**values)


__all__ = ["DomainRobustCausalPickerV7", "DomainRobustV7Config"]
