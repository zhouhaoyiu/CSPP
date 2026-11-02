"""Frozen V9/V10 features with a small residual event-fusion head."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn

from model_domain_robust_v10 import (
    DomainRobustCausalPickerV10,
    DomainRobustV10Config,
)


@dataclass(frozen=True)
class DomainRobustV11Config(DomainRobustV10Config):
    fusion_hidden_dim: int = 96
    fusion_dropout: float = 0.1


class DomainRobustCausalPickerV11(DomainRobustCausalPickerV10):
    """Fuse frozen P-sensitive and target-domain states without waveform replay."""

    model_class = "DomainRobustCausalPickerV11"

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
        fusion_hidden_dim: int = 96,
        fusion_dropout: float = 0.1,
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
            target_packet_width=target_packet_width,
            target_embedding_dim=target_embedding_dim,
            target_sequence_width=target_sequence_width,
            target_temporal_bins=target_temporal_bins,
            target_dropout=target_dropout,
            target_sequence_dilations=target_sequence_dilations,
        )
        self.config = DomainRobustV11Config(
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
            fusion_hidden_dim=int(fusion_hidden_dim),
            fusion_dropout=float(fusion_dropout),
        )
        fusion_width = 2 * self.config.sequence_width + self.config.target_sequence_width
        self.fusion_norm = nn.LayerNorm(fusion_width)
        self.fusion_residual = nn.Sequential(
            nn.Linear(fusion_width, self.config.fusion_hidden_dim),
            nn.SiLU(),
            nn.Dropout(self.config.fusion_dropout),
            nn.Linear(
                self.config.fusion_hidden_dim,
                self.config.target_sequence_width,
            ),
        )
        nn.init.zeros_(self.fusion_residual[-1].weight)
        nn.init.zeros_(self.fusion_residual[-1].bias)
        self.fusion_head_event = nn.Conv1d(
            self.config.target_sequence_width, 1, kernel_size=1
        )
        self._copy_target_event_head()

    def _copy_target_event_head(self) -> None:
        with torch.no_grad():
            self.fusion_head_event.weight.copy_(self.target_head_event.weight)
            self.fusion_head_event.bias.copy_(self.target_head_event.bias)

    def _backbone_representations(
        self, chunks: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        current = self.encode_packets(chunks).transpose(1, 2).contiguous()
        for block in self.sequence_blocks:
            current = block(current)
        current = self.sequence_norm(current)
        verifier = current
        for block in self.verifier_blocks:
            verifier = block(verifier)
        return current, verifier

    def _target_representations(self, chunks: torch.Tensor) -> torch.Tensor:
        current = self.encode_target_packets(chunks).transpose(1, 2).contiguous()
        for block in self.target_sequence_blocks:
            current = block(current)
        return self.target_sequence_norm(current)

    def _fused_event_logits(
        self,
        current: torch.Tensor,
        verifier: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        features = torch.cat((current, verifier, target), dim=1).transpose(1, 2)
        residual = self.fusion_residual(self.fusion_norm(features)).transpose(1, 2)
        return self.fusion_head_event(target + residual).squeeze(1)

    def _fused_event_step(
        self,
        current: torch.Tensor,
        verifier: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        features = torch.cat((current, verifier, target), dim=1)
        residual = self.fusion_residual(self.fusion_norm(features))
        return self.fusion_head_event(
            (target + residual).unsqueeze(-1)
        ).squeeze((1, 2))

    def forward(
        self, chunks: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        current, verifier = self._backbone_representations(chunks)
        target = self._target_representations(chunks)
        return (
            self.head_p(current).squeeze(1),
            self._fused_event_logits(current, verifier, target),
            self.head_delay(verifier).squeeze(1),
        )

    def forward_sequence(
        self,
        chunks: torch.Tensor,
        state: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        if chunks.ndim != 4 or chunks.shape[2] != 3 or chunks.shape[1] < 1:
            raise ValueError(f"Expected chunks shape (B, T, 3, L), got {tuple(chunks.shape)}")
        batch, packets, channels, samples = chunks.shape
        flat = chunks.reshape(batch * packets, channels, samples)
        base_features = self._encode_packet_features(flat).reshape(batch, packets, -1)
        target_features = self._encode_target_packet_features(flat).reshape(
            batch, packets, -1
        )
        base_state = None if state is None else state.get("backbone")
        target_state = None if state is None else state.get("target_verifier")
        outputs = [[], [], []]
        for base_feature, target_feature in zip(
            base_features.unbind(dim=1),
            target_features.unbind(dim=1),
            strict=True,
        ):
            current, verifier, base_state = self._representations_step(
                base_feature, base_state
            )
            target, target_state = self._target_representation_step(
                target_feature, target_state
            )
            outputs[0].append(
                self.head_p(current.unsqueeze(-1)).squeeze((1, 2))
            )
            outputs[1].append(self._fused_event_step(current, verifier, target))
            outputs[2].append(
                self.head_delay(verifier.unsqueeze(-1)).squeeze((1, 2))
            )
        return (
            torch.stack(outputs[0], dim=1),
            torch.stack(outputs[1], dim=1),
            torch.stack(outputs[2], dim=1),
            {"backbone": base_state, "target_verifier": target_state},
        )

    @torch.inference_mode()
    def forward_streaming_packet_features(
        self,
        packet: torch.Tensor,
        state: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        if packet.ndim == 2:
            packet = packet.unsqueeze(0)
        if packet.ndim != 3 or packet.shape[0] != 1:
            raise ValueError("Streaming inference expects (3, L) or (1, 3, L)")
        base_state = None if state is None else state.get("backbone")
        target_state = None if state is None else state.get("target_verifier")
        base_feature = self._encode_packet_features(packet)
        target_feature = self._encode_target_packet_features(packet)
        current, verifier, next_base = self._representations_step(
            base_feature, base_state
        )
        target, next_target = self._target_representation_step(
            target_feature, target_state
        )
        candidate = self.head_p(current.unsqueeze(-1)).squeeze((1, 2))
        event = self._fused_event_step(current, verifier, target)
        delay = self.head_delay(verifier.unsqueeze(-1)).squeeze((1, 2))
        return (
            candidate[0],
            event[0],
            delay[0],
            torch.cat((current, verifier, target), dim=1)[0],
            {"backbone": next_base, "target_verifier": next_target},
        )

    @torch.inference_mode()
    def forward_streaming_packet(
        self,
        packet: torch.Tensor,
        state: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        candidate, event, delay, _, next_state = self.forward_streaming_packet_features(
            packet, state
        )
        return candidate, event, delay, next_state

    def load_v10_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        result = self.load_state_dict(state_dict, strict=False)
        invalid_missing = [
            key for key in result.missing_keys if not key.startswith("fusion_")
        ]
        if result.unexpected_keys or invalid_missing:
            raise ValueError(
                "V10 initialization mismatch: "
                f"unexpected={list(result.unexpected_keys)}, "
                f"missing_non_fusion={invalid_missing}"
            )
        self._copy_target_event_head()

    def freeze_feature_extractors(self) -> None:
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(name.startswith("fusion_"))
        self._backbone_frozen = True
        self.train(self.training)

    def train(self, mode: bool = True):
        super().train(mode)
        if self._backbone_frozen:
            for name, module in self.named_children():
                module.train(mode) if name.startswith("fusion_") else module.eval()
        return self

    @classmethod
    def from_config(cls, config: dict) -> "DomainRobustCausalPickerV11":
        keys = DomainRobustV11Config.__dataclass_fields__.keys()
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


__all__ = ["DomainRobustCausalPickerV11", "DomainRobustV11Config"]
