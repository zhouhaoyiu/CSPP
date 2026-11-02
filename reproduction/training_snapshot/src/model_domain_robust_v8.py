"""Packet-morphology causal picker with bounded past-only state."""
from __future__ import annotations

from dataclasses import dataclass
import math

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


class MorphologyPacketEncoder(nn.Module):
    """Retain coarse within-packet timing before producing one packet token."""

    def __init__(
        self,
        width: int = 56,
        embedding_dim: int = 96,
        temporal_bins: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        if temporal_bins < 2:
            raise ValueError("temporal_bins must be at least two")
        self.temporal_bins = int(temporal_bins)
        self.features = PacketPhysicsFeatures()
        self.summary = PacketSummaryFeatures()
        self.stem = nn.Conv1d(self.features.out_channels, width, kernel_size=1)
        self.stem_norm = nn.GroupNorm(_group_count(width), width)
        self.blocks = nn.ModuleList(
            PacketCausalBlock(width, dilation=dilation, dropout=dropout)
            for dilation in (1, 2, 4, 8)
        )
        pooled_width = width * self.temporal_bins * 2
        self.projection = nn.Sequential(
            nn.Linear(pooled_width + self.summary.out_features, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

    def forward(self, packet: torch.Tensor) -> torch.Tensor:
        if packet.ndim != 3 or packet.shape[1] != 3:
            raise ValueError(f"Expected packet shape (B, 3, L), got {tuple(packet.shape)}")
        summary = self.summary(packet)
        encoded = F.silu(self.stem_norm(self.stem(self.features(packet))))
        for block in self.blocks:
            encoded = block(encoded)
        average = F.adaptive_avg_pool1d(encoded, self.temporal_bins)
        maximum = F.adaptive_max_pool1d(encoded, self.temporal_bins)
        morphology = torch.cat((average, maximum), dim=1).flatten(1)
        return self.projection(torch.cat((morphology, summary), dim=1))


@dataclass(frozen=True)
class DomainRobustV8Config:
    packet_width: int = 56
    embedding_dim: int = 96
    sequence_width: int = 96
    temporal_bins: int = 5
    dropout: float = 0.1
    sequence_dilations: tuple[int, ...] = (1, 2, 4, 8, 16)
    verifier_dilations: tuple[int, ...] = (1, 2)
    max_arrival_age_packets: float = 4.0

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
        }


class DomainRobustCausalPickerV8(nn.Module):
    """Morphology-preserving packet tokens followed by a causal change TCN."""

    model_class = "DomainRobustCausalPickerV8"
    timing_mode = "arrival_age"
    packet_once = True
    raw_replay_packets = 0

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
    ):
        super().__init__()
        self.config = DomainRobustV8Config(
            packet_width=packet_width,
            embedding_dim=embedding_dim,
            sequence_width=sequence_width,
            temporal_bins=temporal_bins,
            dropout=dropout,
            sequence_dilations=tuple(sequence_dilations),
            verifier_dilations=tuple(verifier_dilations),
            max_arrival_age_packets=float(max_arrival_age_packets),
        )
        self.encoder = MorphologyPacketEncoder(
            packet_width,
            embedding_dim,
            temporal_bins,
            dropout,
        )
        self.change_projection = nn.Sequential(
            nn.Linear(embedding_dim * 2, sequence_width),
            nn.LayerNorm(sequence_width),
            nn.SiLU(),
        )
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
        return self.sequence_cache_packets + 1

    @property
    def max_arrival_age_packets(self) -> float:
        return self.config.max_arrival_age_packets

    @property
    def state_size_scalars(self) -> int:
        temporal_history = 2 * sum(self.config.sequence_dilations) + 2 * sum(
            self.config.verifier_dilations
        )
        return self.config.embedding_dim + self.config.sequence_width * temporal_history

    @property
    def head_offset(self) -> nn.Conv1d:
        return self.head_delay

    def _encode_packet_features(self, packet: torch.Tensor) -> torch.Tensor:
        """Encode one or more complete packets without consulting temporal state."""
        return self.encoder(packet)

    def encode_packets(self, chunks: torch.Tensor) -> torch.Tensor:
        if chunks.ndim != 4 or chunks.shape[2] != 3:
            raise ValueError(f"Expected chunks shape (B, T, 3, L), got {tuple(chunks.shape)}")
        batch, packets, channels, samples = chunks.shape
        features = self._encode_packet_features(
            chunks.reshape(batch * packets, channels, samples)
        ).reshape(batch, packets, -1)
        previous = F.pad(features[:, :-1], (0, 0, 1, 0))
        delta = (features - previous) / math.sqrt(2.0)
        delta[:, 0] = 0.0
        return self.change_projection(torch.cat((features, delta), dim=-1))

    def forward_embeddings(
        self,
        embeddings: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = embeddings.transpose(1, 2).contiguous()
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

    @staticmethod
    def _block_step(
        block: SequenceCausalBlock,
        current: torch.Tensor,
        history: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate one causal TCN block at one time step."""
        normalized = block.norm(current.unsqueeze(-1))
        expected = (current.shape[0], current.shape[1], block.pad)
        if history is None:
            history = current.new_zeros(expected)
        else:
            if tuple(history.shape) != expected:
                raise ValueError(
                    f"Expected block history {expected}, got {tuple(history.shape)}"
                )
            history = history.to(device=current.device, dtype=current.dtype)
        context = torch.cat((history, normalized), dim=-1)
        temporal = block.temporal(context)
        value, gate = block.pointwise(temporal).chunk(2, dim=1)
        output = current.unsqueeze(-1) + block.scale * block.dropout(
            torch.tanh(value) * torch.sigmoid(gate)
        )
        return output.squeeze(-1), context[:, :, -block.pad :].detach()

    def _representations_step(
        self,
        feature: torch.Tensor,
        state: dict | None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        if feature.ndim != 2 or feature.shape[1] != self.config.embedding_dim:
            raise ValueError(
                f"Expected feature shape (B, {self.config.embedding_dim}), "
                f"got {tuple(feature.shape)}"
            )
        previous_feature = None if state is None else state.get("previous_feature")
        if previous_feature is None:
            delta = torch.zeros_like(feature)
        else:
            expected = tuple(feature.shape)
            if tuple(previous_feature.shape) != expected:
                raise ValueError(
                    f"Expected previous feature {expected}, got {tuple(previous_feature.shape)}"
                )
            delta = (
                feature - previous_feature.to(device=feature.device, dtype=feature.dtype)
            ) / math.sqrt(2.0)
        current = self.change_projection(torch.cat((feature, delta), dim=-1))

        sequence_history = () if state is None else state.get("sequence_history", ())
        if sequence_history and len(sequence_history) != len(self.sequence_blocks):
            raise ValueError("Sequence history does not match the configured TCN")
        next_sequence_history = []
        for index, block in enumerate(self.sequence_blocks):
            history = sequence_history[index] if sequence_history else None
            current, next_history = self._block_step(block, current, history)
            next_sequence_history.append(next_history)

        current = self.sequence_norm(current.unsqueeze(-1)).squeeze(-1)
        verifier = current
        verifier_history = () if state is None else state.get("verifier_history", ())
        if verifier_history and len(verifier_history) != len(self.verifier_blocks):
            raise ValueError("Verifier history does not match the configured TCN")
        next_verifier_history = []
        for index, block in enumerate(self.verifier_blocks):
            history = verifier_history[index] if verifier_history else None
            verifier, next_history = self._block_step(block, verifier, history)
            next_verifier_history.append(next_history)

        next_state = {
            "previous_feature": feature.detach(),
            "sequence_history": tuple(next_sequence_history),
            "verifier_history": tuple(next_verifier_history),
        }
        return current, verifier, next_state

    def _feature_step(
        self,
        feature: torch.Tensor,
        state: dict | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        current, verifier, next_state = self._representations_step(feature, state)
        logits_p = self.head_p(current.unsqueeze(-1)).squeeze((1, 2))
        logits_event = self.head_event(verifier.unsqueeze(-1)).squeeze((1, 2))
        logits_delay = self.head_delay(verifier.unsqueeze(-1)).squeeze((1, 2))
        return logits_p, logits_event, logits_delay, next_state

    def forward_sequence(
        self,
        chunks: torch.Tensor,
        state: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Encode every new packet once and carry exact per-layer TCN state."""
        if chunks.ndim != 4 or chunks.shape[2] != 3:
            raise ValueError(f"Expected chunks shape (B, T, 3, L), got {tuple(chunks.shape)}")
        if chunks.shape[1] < 1:
            raise ValueError("At least one packet is required")
        batch, packets, channels, samples = chunks.shape
        features = self._encode_packet_features(
            chunks.reshape(batch * packets, channels, samples)
        ).reshape(batch, packets, -1)
        outputs = [[], [], []]
        current_state = state
        for feature in features.unbind(dim=1):
            heads = self._feature_step(feature, current_state)
            current_state = heads[3]
            for values, value in zip(outputs, heads[:3], strict=True):
                values.append(value)
        return (
            torch.stack(outputs[0], dim=1),
            torch.stack(outputs[1], dim=1),
            torch.stack(outputs[2], dim=1),
            current_state,
        )

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
        feature = self._encode_packet_features(packet)
        logits_p, logits_event, logits_delay, next_state = self._feature_step(
            feature, state
        )
        return (
            logits_p[0],
            logits_event[0],
            logits_delay[0],
            next_state,
        )

    @torch.inference_mode()
    def forward_streaming_packet_features(
        self,
        packet: torch.Tensor,
        state: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Process one packet and expose the frozen confirmation-head feature."""
        if packet.ndim == 2:
            packet = packet.unsqueeze(0)
        if packet.ndim != 3 or packet.shape[0] != 1:
            raise ValueError("Streaming inference expects (3, L) or (1, 3, L)")
        feature = self._encode_packet_features(packet)
        current, verifier, next_state = self._representations_step(feature, state)
        logits_p = self.head_p(current.unsqueeze(-1)).squeeze((1, 2))
        logits_event = self.head_event(verifier.unsqueeze(-1)).squeeze((1, 2))
        logits_delay = self.head_delay(verifier.unsqueeze(-1)).squeeze((1, 2))
        confirmation_feature = torch.cat((current, verifier), dim=1)
        return (
            logits_p[0],
            logits_event[0],
            logits_delay[0],
            confirmation_feature[0],
            next_state,
        )

    @classmethod
    def from_config(cls, config: dict) -> "DomainRobustCausalPickerV8":
        keys = DomainRobustV8Config.__dataclass_fields__.keys()
        values = {key: config[key] for key in keys if key in config}
        for key in ("sequence_dilations", "verifier_dilations"):
            if key in values:
                values[key] = tuple(values[key])
        return cls(**values)


__all__ = ["DomainRobustCausalPickerV8", "DomainRobustV8Config"]
