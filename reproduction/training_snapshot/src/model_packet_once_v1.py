"""Packet-once causal picker family with constant-size temporal state."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn

from model_domain_robust_v8 import MorphologyPacketEncoder


STATE_MODES = ("packet", "retention", "gru")


@dataclass(frozen=True)
class PacketOnceV1Config:
    packet_width: int = 64
    embedding_dim: int = 96
    sequence_width: int = 96
    temporal_bins: int = 5
    dropout: float = 0.1
    state_mode: str = "retention"
    retention_half_lives_seconds: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
    packet_seconds: float = 0.5
    max_arrival_age_packets: float = 4.0

    def as_dict(self) -> dict:
        return {
            "packet_width": self.packet_width,
            "embedding_dim": self.embedding_dim,
            "sequence_width": self.sequence_width,
            "temporal_bins": self.temporal_bins,
            "dropout": self.dropout,
            "state_mode": self.state_mode,
            "retention_half_lives_seconds": list(self.retention_half_lives_seconds),
            "packet_seconds": self.packet_seconds,
            "max_arrival_age_packets": self.max_arrival_age_packets,
        }


class PacketOnceCausalPickerV1(nn.Module):
    """One packet encoder with packet, exponential-retention, or GRU state."""

    model_class = "PacketOnceCausalPickerV1"
    timing_mode = "arrival_age"
    packet_once = True
    raw_replay_packets = 0

    def __init__(
        self,
        packet_width: int = 64,
        embedding_dim: int = 96,
        sequence_width: int = 96,
        temporal_bins: int = 5,
        dropout: float = 0.1,
        state_mode: str = "retention",
        retention_half_lives_seconds: tuple[float, ...] | list[float] = (
            0.5,
            1.0,
            2.0,
            4.0,
            8.0,
            16.0,
        ),
        packet_seconds: float = 0.5,
        max_arrival_age_packets: float = 4.0,
    ):
        super().__init__()
        state_mode = str(state_mode)
        if state_mode not in STATE_MODES:
            raise ValueError(f"state_mode must be one of {STATE_MODES}, got {state_mode!r}")
        half_lives = tuple(float(value) for value in retention_half_lives_seconds)
        if not half_lives or any(value <= 0.0 for value in half_lives):
            raise ValueError("retention half-lives must be non-empty and positive")
        if packet_seconds <= 0.0:
            raise ValueError("packet_seconds must be positive")

        self.config = PacketOnceV1Config(
            packet_width=int(packet_width),
            embedding_dim=int(embedding_dim),
            sequence_width=int(sequence_width),
            temporal_bins=int(temporal_bins),
            dropout=float(dropout),
            state_mode=state_mode,
            retention_half_lives_seconds=half_lives,
            packet_seconds=float(packet_seconds),
            max_arrival_age_packets=float(max_arrival_age_packets),
        )
        self.encoder = MorphologyPacketEncoder(
            width=packet_width,
            embedding_dim=embedding_dim,
            temporal_bins=temporal_bins,
            dropout=dropout,
        )
        self.change_projection = nn.Sequential(
            nn.Linear(embedding_dim * 2, sequence_width),
            nn.LayerNorm(sequence_width),
            nn.SiLU(),
        )

        if state_mode == "packet":
            self.temporal = nn.Sequential(
                nn.Linear(sequence_width, sequence_width * 4),
                nn.LayerNorm(sequence_width * 4),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(sequence_width * 4, sequence_width),
                nn.LayerNorm(sequence_width),
                nn.SiLU(),
            )
        elif state_mode == "retention":
            self.state_projection = nn.Linear(sequence_width, sequence_width)
            self.temporal = nn.Sequential(
                nn.Linear(sequence_width * (1 + len(half_lives)), sequence_width),
                nn.LayerNorm(sequence_width),
                nn.SiLU(),
                nn.Dropout(dropout),
            )
            alphas = [
                math.exp(math.log(0.5) * float(packet_seconds) / half_life)
                for half_life in half_lives
            ]
            self.register_buffer(
                "retention_alphas",
                torch.tensor(alphas, dtype=torch.float32).view(1, -1, 1),
                persistent=True,
            )
        else:
            self.temporal = nn.GRU(
                input_size=sequence_width,
                hidden_size=sequence_width,
                num_layers=1,
                batch_first=True,
            )
            self.gru_projection = nn.Sequential(
                nn.Linear(sequence_width, sequence_width),
                nn.LayerNorm(sequence_width),
                nn.SiLU(),
                nn.Dropout(dropout),
            )

        self.head_p = nn.Linear(sequence_width, 1)
        self.head_event = nn.Linear(sequence_width, 1)
        self.head_delay = nn.Linear(sequence_width, 1)
        nn.init.constant_(self.head_p.bias, -4.0)
        nn.init.constant_(self.head_event.bias, -4.0)
        nn.init.zeros_(self.head_delay.bias)

    @property
    def state_mode(self) -> str:
        return self.config.state_mode

    @property
    def max_arrival_age_packets(self) -> float:
        return self.config.max_arrival_age_packets

    @property
    def head_offset(self) -> nn.Linear:
        return self.head_delay

    @property
    def receptive_field_packets(self) -> float:
        if self.state_mode == "packet":
            return 2
        return math.inf

    @property
    def state_size_scalars(self) -> int:
        previous = self.config.embedding_dim
        if self.state_mode == "retention":
            return previous + len(self.config.retention_half_lives_seconds) * self.config.sequence_width
        if self.state_mode == "gru":
            return previous + self.config.sequence_width
        return previous

    def _encode_tokens(
        self,
        chunks: torch.Tensor,
        previous_feature: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if chunks.ndim != 4 or chunks.shape[2] != 3:
            raise ValueError(f"Expected chunks shape (B, T, 3, L), got {tuple(chunks.shape)}")
        batch, packets, channels, samples = chunks.shape
        if packets < 1:
            raise ValueError("At least one packet is required")
        features = self.encoder(chunks.reshape(batch * packets, channels, samples)).reshape(
            batch, packets, -1
        )
        if previous_feature is None:
            previous = torch.cat((torch.zeros_like(features[:, :1]), features[:, :-1]), dim=1)
            delta = (features - previous) / math.sqrt(2.0)
            delta[:, 0] = 0.0
        else:
            expected = (batch, self.config.embedding_dim)
            if tuple(previous_feature.shape) != expected:
                raise ValueError(
                    f"Expected previous_feature shape {expected}, got {tuple(previous_feature.shape)}"
                )
            previous = torch.cat((previous_feature.to(features.device).unsqueeze(1), features[:, :-1]), dim=1)
            delta = (features - previous) / math.sqrt(2.0)
        tokens = self.change_projection(torch.cat((features, delta), dim=-1))
        return tokens, features[:, -1]

    def _retention_sequence(
        self,
        tokens: torch.Tensor,
        retention: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = tokens.shape[0]
        scales = len(self.config.retention_half_lives_seconds)
        expected = (batch, scales, self.config.sequence_width)
        if retention is None:
            retention = tokens.new_zeros(expected)
        elif tuple(retention.shape) != expected:
            raise ValueError(f"Expected retention shape {expected}, got {tuple(retention.shape)}")
        else:
            retention = retention.to(tokens.device)

        alpha = self.retention_alphas.to(device=tokens.device, dtype=tokens.dtype)
        outputs = []
        for token in tokens.unbind(dim=1):
            context = torch.cat((token, retention.flatten(1)), dim=1)
            outputs.append(self.temporal(context))
            innovation = torch.tanh(self.state_projection(token)).unsqueeze(1)
            retention = alpha * retention + (1.0 - alpha) * innovation
        return torch.stack(outputs, dim=1), retention

    def forward_sequence(
        self,
        chunks: torch.Tensor,
        state: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        previous_feature = None if state is None else state.get("previous_feature")
        tokens, last_feature = self._encode_tokens(chunks, previous_feature)

        if self.state_mode == "packet":
            sequence = self.temporal(tokens)
            next_state = {"previous_feature": last_feature.detach()}
        elif self.state_mode == "retention":
            retention = None if state is None else state.get("retention")
            sequence, next_retention = self._retention_sequence(tokens, retention)
            next_state = {
                "previous_feature": last_feature.detach(),
                "retention": next_retention.detach(),
            }
        else:
            hidden = None if state is None else state.get("hidden")
            if hidden is not None:
                expected = (1, chunks.shape[0], self.config.sequence_width)
                if tuple(hidden.shape) != expected:
                    raise ValueError(f"Expected hidden shape {expected}, got {tuple(hidden.shape)}")
                hidden = hidden.to(tokens.device)
            sequence, next_hidden = self.temporal(tokens, hidden)
            sequence = self.gru_projection(sequence)
            next_state = {
                "previous_feature": last_feature.detach(),
                "hidden": next_hidden.detach(),
            }

        return (
            self.head_p(sequence).squeeze(-1),
            self.head_event(sequence).squeeze(-1),
            self.head_delay(sequence).squeeze(-1),
            next_state,
        )

    def forward(self, chunks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.forward_sequence(chunks)[:3]

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
        heads = self.forward_sequence(packet.unsqueeze(1), state)
        return heads[0][0, 0], heads[1][0, 0], heads[2][0, 0], heads[3]

    @classmethod
    def from_config(cls, config: dict) -> "PacketOnceCausalPickerV1":
        keys = PacketOnceV1Config.__dataclass_fields__.keys()
        values = {key: config[key] for key in keys if key in config}
        if "retention_half_lives_seconds" in values:
            values["retention_half_lives_seconds"] = tuple(
                values["retention_half_lives_seconds"]
            )
        return cls(**values)


__all__ = ["PacketOnceCausalPickerV1", "PacketOnceV1Config", "STATE_MODES"]
