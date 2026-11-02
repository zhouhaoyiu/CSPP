"""V9 candidate backbone plus an independently trained causal event verifier."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_domain_robust_v5 import CausalChannelNorm, SequenceCausalBlock
from model_domain_robust_v8 import MorphologyPacketEncoder
from model_domain_robust_v9 import DomainRobustCausalPickerV9, FixedPacketSpectrum


@dataclass(frozen=True)
class DomainRobustV10Config:
    packet_width: int = 64
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
    target_packet_width: int = 32
    target_embedding_dim: int = 64
    target_sequence_width: int = 64
    target_temporal_bins: int = 5
    target_dropout: float = 0.1
    target_sequence_dilations: tuple[int, ...] = (1, 2, 4, 8)

    def as_dict(self) -> dict:
        payload = dict(self.__dict__)
        for key in (
            "sequence_dilations",
            "verifier_dilations",
            "spectral_band_edges_hz",
            "target_sequence_dilations",
        ):
            payload[key] = list(payload[key])
        return payload


class DomainRobustCausalPickerV10(DomainRobustCausalPickerV9):
    """Preserve V9 P/timing outputs and replace only its event verification path."""

    model_class = "DomainRobustCausalPickerV10"
    integrated_verifier = True

    def __init__(
        self,
        packet_width: int = 64,
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
        target_packet_width: int = 32,
        target_embedding_dim: int = 64,
        target_sequence_width: int = 64,
        target_temporal_bins: int = 5,
        target_dropout: float = 0.1,
        target_sequence_dilations: tuple[int, ...] | list[int] = (1, 2, 4, 8),
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
            packet_samples=packet_samples,
            sample_rate_hz=sample_rate_hz,
            spectral_band_edges_hz=spectral_band_edges_hz,
        )
        self.config = DomainRobustV10Config(
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
            target_packet_width=int(target_packet_width),
            target_embedding_dim=int(target_embedding_dim),
            target_sequence_width=int(target_sequence_width),
            target_temporal_bins=int(target_temporal_bins),
            target_dropout=float(target_dropout),
            target_sequence_dilations=tuple(target_sequence_dilations),
        )
        self.target_encoder = MorphologyPacketEncoder(
            width=self.config.target_packet_width,
            embedding_dim=self.config.target_embedding_dim,
            temporal_bins=self.config.target_temporal_bins,
            dropout=self.config.target_dropout,
        )
        self.target_spectral_features = FixedPacketSpectrum(
            packet_samples=self.config.packet_samples,
            sample_rate_hz=self.config.sample_rate_hz,
            band_edges_hz=self.config.spectral_band_edges_hz,
        )
        self.target_spectral_projection = nn.Sequential(
            nn.Linear(
                self.target_spectral_features.out_features,
                self.config.target_embedding_dim,
            ),
            nn.LayerNorm(self.config.target_embedding_dim),
            nn.SiLU(),
        )
        self.target_spectral_scale = nn.Parameter(torch.tensor(0.1))
        self.target_change_projection = nn.Sequential(
            nn.Linear(
                2 * self.config.target_embedding_dim,
                self.config.target_sequence_width,
            ),
            nn.LayerNorm(self.config.target_sequence_width),
            nn.SiLU(),
        )
        self.target_sequence_blocks = nn.ModuleList(
            SequenceCausalBlock(
                self.config.target_sequence_width,
                dilation=dilation,
                dropout=self.config.target_dropout,
            )
            for dilation in self.config.target_sequence_dilations
        )
        self.target_sequence_norm = CausalChannelNorm(
            self.config.target_sequence_width
        )
        self.target_head_event = nn.Conv1d(
            self.config.target_sequence_width, 1, kernel_size=1
        )
        nn.init.constant_(self.target_head_event.bias, -4.0)
        self._backbone_frozen = False

    @property
    def target_cache_packets(self) -> int:
        return 2 * sum(self.config.target_sequence_dilations)

    @property
    def state_size_scalars(self) -> int:
        return super().state_size_scalars + self.config.target_embedding_dim + (
            self.config.target_sequence_width * self.target_cache_packets
        )

    def _encode_target_packet_features(self, packet: torch.Tensor) -> torch.Tensor:
        morphology = self.target_encoder(packet)
        spectral = self.target_spectral_projection(
            self.target_spectral_features(packet)
        )
        return morphology + self.target_spectral_scale * spectral

    def encode_target_packets(self, chunks: torch.Tensor) -> torch.Tensor:
        if chunks.ndim != 4 or chunks.shape[2] != 3:
            raise ValueError(
                f"Expected chunks shape (B, T, 3, L), got {tuple(chunks.shape)}"
            )
        batch, packets, channels, samples = chunks.shape
        features = self._encode_target_packet_features(
            chunks.reshape(batch * packets, channels, samples)
        ).reshape(batch, packets, -1)
        previous = F.pad(features[:, :-1], (0, 0, 1, 0))
        delta = (features - previous) / math.sqrt(2.0)
        delta[:, 0] = 0.0
        return self.target_change_projection(torch.cat((features, delta), dim=-1))

    def forward_target_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        current = embeddings.transpose(1, 2).contiguous()
        for block in self.target_sequence_blocks:
            current = block(current)
        current = self.target_sequence_norm(current)
        return self.target_head_event(current).squeeze(1)

    def forward(
        self, chunks: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits_p, _, logits_delay = super().forward(chunks)
        logits_event = self.forward_target_embeddings(
            self.encode_target_packets(chunks)
        )
        return logits_p, logits_event, logits_delay

    def _target_representation_step(
        self,
        feature: torch.Tensor,
        state: dict | None,
    ) -> tuple[torch.Tensor, dict]:
        previous = None if state is None else state.get("previous_feature")
        if previous is None:
            delta = torch.zeros_like(feature)
        else:
            if tuple(previous.shape) != tuple(feature.shape):
                raise ValueError("Target verifier previous-feature shape mismatch")
            delta = (
                feature - previous.to(device=feature.device, dtype=feature.dtype)
            ) / math.sqrt(2.0)
        current = self.target_change_projection(torch.cat((feature, delta), dim=-1))
        history = () if state is None else state.get("sequence_history", ())
        if history and len(history) != len(self.target_sequence_blocks):
            raise ValueError("Target verifier history does not match its causal TCN")
        next_history = []
        for index, block in enumerate(self.target_sequence_blocks):
            current, block_history = self._block_step(
                block, current, history[index] if history else None
            )
            next_history.append(block_history)
        current = self.target_sequence_norm(current.unsqueeze(-1)).squeeze(-1)
        return current, {
            "previous_feature": feature.detach(),
            "sequence_history": tuple(next_history),
        }

    def _target_feature_step(
        self,
        feature: torch.Tensor,
        state: dict | None,
    ) -> tuple[torch.Tensor, dict]:
        representation, next_state = self._target_representation_step(feature, state)
        logits = self.target_head_event(representation.unsqueeze(-1)).squeeze((1, 2))
        return logits, next_state

    def forward_sequence(
        self,
        chunks: torch.Tensor,
        state: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        base_state = None if state is None else state.get("backbone")
        target_state = None if state is None else state.get("target_verifier")
        logits_p, _, logits_delay, next_base = super().forward_sequence(
            chunks, base_state
        )
        if chunks.ndim != 4 or chunks.shape[1] < 1:
            raise ValueError("At least one packet sequence is required")
        batch, packets, channels, samples = chunks.shape
        features = self._encode_target_packet_features(
            chunks.reshape(batch * packets, channels, samples)
        ).reshape(batch, packets, -1)
        event_outputs = []
        next_target = target_state
        for feature in features.unbind(dim=1):
            event, next_target = self._target_feature_step(feature, next_target)
            event_outputs.append(event)
        return (
            logits_p,
            torch.stack(event_outputs, dim=1),
            logits_delay,
            {"backbone": next_base, "target_verifier": next_target},
        )

    @torch.inference_mode()
    def forward_streaming_packet_features(
        self,
        packet: torch.Tensor,
        state: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        base_state = None if state is None else state.get("backbone")
        target_state = None if state is None else state.get("target_verifier")
        p, _, delay, base_features, next_base = (
            super().forward_streaming_packet_features(packet, base_state)
        )
        if packet.ndim == 2:
            packet = packet.unsqueeze(0)
        feature = self._encode_target_packet_features(packet)
        event, next_target = self._target_feature_step(feature, target_state)
        return (
            p,
            event[0],
            delay,
            base_features,
            {"backbone": next_base, "target_verifier": next_target},
        )

    @torch.inference_mode()
    def forward_streaming_packet(
        self,
        packet: torch.Tensor,
        state: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        p, event, delay, _, next_state = self.forward_streaming_packet_features(
            packet, state
        )
        return p, event, delay, next_state

    def load_v9_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        result = self.load_state_dict(state_dict, strict=False)
        unexpected = list(result.unexpected_keys)
        invalid_missing = [
            key for key in result.missing_keys if not key.startswith("target_")
        ]
        if unexpected or invalid_missing:
            raise ValueError(
                "V9 initialization mismatch: "
                f"unexpected={unexpected}, missing_non_target={invalid_missing}"
            )

    def freeze_backbone(self) -> None:
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(name.startswith("target_"))
        self._backbone_frozen = True
        self.train(self.training)

    def train(self, mode: bool = True):
        super().train(mode)
        if self._backbone_frozen:
            for name, module in self.named_children():
                if not name.startswith("target_"):
                    module.eval()
        return self

    @classmethod
    def from_config(cls, config: dict) -> "DomainRobustCausalPickerV10":
        keys = DomainRobustV10Config.__dataclass_fields__.keys()
        values = {key: config[key] for key in keys if key in config}
        for key in (
            "sequence_dilations",
            "verifier_dilations",
            "spectral_band_edges_hz",
            "target_sequence_dilations",
        ):
            if key in values:
                values[key] = tuple(values[key])
        return cls(**values)


__all__ = ["DomainRobustCausalPickerV10", "DomainRobustV10Config"]
