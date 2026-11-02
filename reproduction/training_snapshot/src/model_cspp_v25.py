"""From-scratch CSPP model with packet morphology and bounded causal state."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn

from model_domain_robust_v5 import (
    CausalChannelNorm,
    PacketPhysicsFeatures,
    PacketSummaryFeatures,
    SequenceCausalBlock,
    _group_count,
)
from model_domain_robust_v9 import FixedPacketSpectrum


@dataclass(frozen=True)
class CSPPV25Config:
    packet_width: int = 64
    embedding_dim: int = 128
    sequence_width: int = 128
    temporal_bins: int = 10
    dropout: float = 0.1
    packet_kernels: tuple[int, ...] = (3, 7, 15, 31)
    sequence_dilations: tuple[int, ...] = (1, 2, 4, 8, 16)
    retention_half_lives_seconds: tuple[float, ...] = (
        0.5,
        1.0,
        2.0,
        4.0,
        8.0,
        16.0,
    )
    packet_seconds: float = 0.5
    max_arrival_age_packets: float = 4.0
    packet_samples: int = 50
    sample_rate_hz: float = 100.0
    spectral_band_edges_hz: tuple[float, ...] = (
        1.0,
        4.0,
        8.0,
        16.0,
        30.0,
        50.1,
    )

    def as_dict(self) -> dict:
        payload = dict(self.__dict__)
        for key in (
            "packet_kernels",
            "sequence_dilations",
            "retention_half_lives_seconds",
            "spectral_band_edges_hz",
        ):
            payload[key] = list(payload[key])
        return payload


class CompletePacketResidualBlock(nn.Module):
    """Use both directions inside one fully received packet, never across packets."""

    def __init__(self, channels: int, kernel_size: int, dropout: float):
        super().__init__()
        if kernel_size < 3 or kernel_size % 2 == 0:
            raise ValueError("Packet kernels must be odd and at least three")
        self.norm = nn.GroupNorm(_group_count(channels), channels)
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=channels,
        )
        self.pointwise = nn.Conv1d(channels, 2 * channels, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        filtered = self.depthwise(self.norm(values))
        content, gate = self.pointwise(filtered).chunk(2, dim=1)
        update = torch.tanh(content) * torch.sigmoid(gate)
        return values + self.scale * self.dropout(update)


class CSPPPacketEncoderV25(nn.Module):
    """Preserve intra-packet onset shape, amplitude contrast, and spectrum."""

    def __init__(self, config: CSPPV25Config):
        super().__init__()
        self.config = config
        self.physics = PacketPhysicsFeatures()
        self.summary = PacketSummaryFeatures()
        self.spectrum = FixedPacketSpectrum(
            packet_samples=config.packet_samples,
            sample_rate_hz=config.sample_rate_hz,
            band_edges_hz=config.spectral_band_edges_hz,
        )
        self.stem = nn.Conv1d(
            self.physics.out_channels,
            config.packet_width,
            kernel_size=5,
            padding=2,
            bias=False,
        )
        self.stem_norm = nn.GroupNorm(
            _group_count(config.packet_width),
            config.packet_width,
        )
        self.blocks = nn.ModuleList(
            CompletePacketResidualBlock(
                config.packet_width,
                kernel_size=kernel,
                dropout=config.dropout,
            )
            for kernel in config.packet_kernels
        )
        pooled_features = (
            2 * config.packet_width * config.temporal_bins
            + self.summary.out_features
            + self.spectrum.out_features
        )
        self.projection = nn.Sequential(
            nn.Linear(pooled_features, config.embedding_dim),
            nn.LayerNorm(config.embedding_dim),
            nn.SiLU(),
            nn.Dropout(config.dropout),
        )

    def forward(self, packet: torch.Tensor) -> torch.Tensor:
        if packet.ndim != 3 or packet.shape[1:] != (
            3,
            self.config.packet_samples,
        ):
            raise ValueError(
                "Expected packet shape "
                f"(B, 3, {self.config.packet_samples}), got {tuple(packet.shape)}"
            )
        values = torch.nn.functional.silu(
            self.stem_norm(self.stem(self.physics(packet)))
        )
        for block in self.blocks:
            values = block(values)
        average = torch.nn.functional.adaptive_avg_pool1d(
            values, self.config.temporal_bins
        )
        maximum = torch.nn.functional.adaptive_max_pool1d(
            values, self.config.temporal_bins
        )
        features = torch.cat(
            (
                average.flatten(1),
                maximum.flatten(1),
                self.summary(packet),
                self.spectrum(packet),
            ),
            dim=1,
        )
        return self.projection(features)


class CSPPCausalPickerV25(nn.Module):
    """End-to-end packet candidate, confirmation, and P-age predictor."""

    model_class = "CSPPCausalPickerV25"
    timing_mode = "arrival_age"
    packet_once = True
    raw_replay_packets = 0

    def __init__(
        self,
        packet_width: int = 64,
        embedding_dim: int = 128,
        sequence_width: int = 128,
        temporal_bins: int = 10,
        dropout: float = 0.1,
        packet_kernels: tuple[int, ...] | list[int] = (3, 7, 15, 31),
        sequence_dilations: tuple[int, ...] | list[int] = (1, 2, 4, 8, 16),
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
        super().__init__()
        half_lives = tuple(float(value) for value in retention_half_lives_seconds)
        if not half_lives or any(value <= 0.0 for value in half_lives):
            raise ValueError("Retention half-lives must be non-empty and positive")
        if packet_seconds <= 0.0:
            raise ValueError("packet_seconds must be positive")
        self.config = CSPPV25Config(
            packet_width=int(packet_width),
            embedding_dim=int(embedding_dim),
            sequence_width=int(sequence_width),
            temporal_bins=int(temporal_bins),
            dropout=float(dropout),
            packet_kernels=tuple(int(value) for value in packet_kernels),
            sequence_dilations=tuple(int(value) for value in sequence_dilations),
            retention_half_lives_seconds=half_lives,
            packet_seconds=float(packet_seconds),
            max_arrival_age_packets=float(max_arrival_age_packets),
            packet_samples=int(packet_samples),
            sample_rate_hz=float(sample_rate_hz),
            spectral_band_edges_hz=tuple(
                float(value) for value in spectral_band_edges_hz
            ),
        )
        self.encoder = CSPPPacketEncoderV25(self.config)
        self.change_projection = nn.Sequential(
            nn.Linear(2 * self.config.embedding_dim, self.config.sequence_width),
            nn.LayerNorm(self.config.sequence_width),
            nn.SiLU(),
        )
        self.sequence_blocks = nn.ModuleList(
            SequenceCausalBlock(
                self.config.sequence_width,
                dilation=dilation,
                dropout=self.config.dropout,
            )
            for dilation in self.config.sequence_dilations
        )
        self.sequence_norm = CausalChannelNorm(self.config.sequence_width)
        self.retention_projection = nn.Linear(
            self.config.sequence_width,
            self.config.sequence_width,
        )
        scales = len(self.config.retention_half_lives_seconds)
        alphas = [
            math.exp(
                math.log(0.5) * self.config.packet_seconds / half_life
            )
            for half_life in self.config.retention_half_lives_seconds
        ]
        self.register_buffer(
            "retention_alphas",
            torch.tensor(alphas, dtype=torch.float32).view(1, scales, 1),
            persistent=True,
        )
        fusion_input = self.config.sequence_width * (2 + scales)
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input, 2 * self.config.sequence_width),
            nn.LayerNorm(2 * self.config.sequence_width),
            nn.SiLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(2 * self.config.sequence_width, self.config.sequence_width),
        )
        candidate_input = 3 * self.config.sequence_width
        self.candidate_head = nn.Sequential(
            nn.LayerNorm(candidate_input),
            nn.Linear(candidate_input, self.config.sequence_width),
            nn.SiLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.sequence_width, 1),
        )
        self.event_head = nn.Sequential(
            nn.LayerNorm(self.config.sequence_width),
            nn.Linear(self.config.sequence_width, 1),
        )
        self.delay_head = nn.Sequential(
            nn.LayerNorm(self.config.sequence_width),
            nn.Linear(self.config.sequence_width, 1),
        )
        nn.init.constant_(self.candidate_head[-1].bias, -4.0)
        nn.init.constant_(self.event_head[-1].bias, -4.0)
        nn.init.zeros_(self.delay_head[-1].bias)

    @property
    def max_arrival_age_packets(self) -> float:
        return self.config.max_arrival_age_packets

    @property
    def receptive_field_packets(self) -> float:
        return math.inf

    @property
    def state_size_scalars(self) -> int:
        local_history = (
            2
            * sum(self.config.sequence_dilations)
            * self.config.sequence_width
        )
        retention = (
            len(self.config.retention_half_lives_seconds)
            * self.config.sequence_width
        )
        return self.config.embedding_dim + local_history + retention

    @property
    def head_p(self) -> nn.Module:
        return self.candidate_head

    @property
    def head_event(self) -> nn.Module:
        return self.event_head

    @property
    def head_delay(self) -> nn.Module:
        return self.delay_head

    @property
    def head_offset(self) -> nn.Module:
        return self.delay_head

    def _encode_features(self, chunks: torch.Tensor) -> torch.Tensor:
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
        return self.encoder(
            chunks.reshape(batch * packets, channels, samples)
        ).reshape(batch, packets, -1)

    def _tokens_from_features(
        self,
        features: torch.Tensor,
        previous_feature: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = features.shape[0]
        if previous_feature is None:
            previous = torch.cat(
                (torch.zeros_like(features[:, :1]), features[:, :-1]), dim=1
            )
            delta = (features - previous) / math.sqrt(2.0)
            delta[:, 0] = 0.0
        else:
            expected = (batch, self.config.embedding_dim)
            if tuple(previous_feature.shape) != expected:
                raise ValueError(
                    f"Expected previous feature {expected}, "
                    f"got {tuple(previous_feature.shape)}"
                )
            previous = torch.cat(
                (
                    previous_feature.to(
                        device=features.device,
                        dtype=features.dtype,
                    ).unsqueeze(1),
                    features[:, :-1],
                ),
                dim=1,
            )
            delta = (features - previous) / math.sqrt(2.0)
        tokens = self.change_projection(torch.cat((features, delta), dim=-1))
        return tokens, features[:, -1]

    def _initial_retention(
        self,
        reference: torch.Tensor,
        retention: torch.Tensor | None,
    ) -> torch.Tensor:
        expected = (
            reference.shape[0],
            len(self.config.retention_half_lives_seconds),
            self.config.sequence_width,
        )
        if retention is None:
            return reference.new_zeros(expected)
        if tuple(retention.shape) != expected:
            raise ValueError(
                f"Expected retention state {expected}, got {tuple(retention.shape)}"
            )
        return retention.to(device=reference.device, dtype=reference.dtype)

    def _fuse_sequence(
        self,
        tokens: torch.Tensor,
        local: torch.Tensor,
        retention: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        current_retention = self._initial_retention(tokens[:, 0], retention)
        alpha = self.retention_alphas.to(
            device=tokens.device,
            dtype=tokens.dtype,
        )
        fused = []
        for token, local_value in zip(
            tokens.unbind(dim=1),
            local.unbind(dim=1),
            strict=True,
        ):
            context = torch.cat(
                (
                    token,
                    local_value,
                    current_retention.flatten(1),
                ),
                dim=1,
            )
            fused.append(local_value + self.fusion(context))
            innovation = torch.tanh(self.retention_projection(token)).unsqueeze(1)
            current_retention = (
                alpha * current_retention + (1.0 - alpha) * innovation
            )
        return torch.stack(fused, dim=1), current_retention

    def _heads(
        self,
        tokens: torch.Tensor,
        local: torch.Tensor,
        fused: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        candidate = self.candidate_head(
            torch.cat((tokens, local, fused), dim=-1)
        ).squeeze(-1)
        event = self.event_head(fused).squeeze(-1)
        delay = self.delay_head(fused).squeeze(-1)
        return candidate, event, delay

    def forward(
        self,
        chunks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self._encode_features(chunks)
        tokens, _ = self._tokens_from_features(features)
        local = tokens.transpose(1, 2).contiguous()
        for block in self.sequence_blocks:
            local = block(local)
        local = self.sequence_norm(local).transpose(1, 2).contiguous()
        fused, _ = self._fuse_sequence(tokens, local, None)
        return self._heads(tokens, local, fused)

    @staticmethod
    def _block_step(
        block: SequenceCausalBlock,
        current: torch.Tensor,
        history: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        content, gate = block.pointwise(temporal).chunk(2, dim=1)
        output = current.unsqueeze(-1) + block.scale * block.dropout(
            torch.tanh(content) * torch.sigmoid(gate)
        )
        return output.squeeze(-1), context[:, :, -block.pad :].detach()

    def _token_step(
        self,
        token: torch.Tensor,
        state: dict | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        histories = () if state is None else state.get("sequence_history", ())
        if histories and len(histories) != len(self.sequence_blocks):
            raise ValueError("Sequence history does not match the configured TCN")
        local = token
        next_histories = []
        for index, block in enumerate(self.sequence_blocks):
            history = histories[index] if histories else None
            local, next_history = self._block_step(block, local, history)
            next_histories.append(next_history)
        local = self.sequence_norm(local.unsqueeze(-1)).squeeze(-1)

        retention_state = None if state is None else state.get("retention")
        retention = self._initial_retention(token, retention_state)
        context = torch.cat((token, local, retention.flatten(1)), dim=1)
        fused = local + self.fusion(context)
        candidate, event, delay = self._heads(
            token.unsqueeze(1),
            local.unsqueeze(1),
            fused.unsqueeze(1),
        )
        alpha = self.retention_alphas.to(
            device=token.device,
            dtype=token.dtype,
        )
        innovation = torch.tanh(self.retention_projection(token)).unsqueeze(1)
        next_retention = alpha * retention + (1.0 - alpha) * innovation
        return (
            candidate[:, 0],
            event[:, 0],
            delay[:, 0],
            {
                "sequence_history": tuple(next_histories),
                "retention": next_retention.detach(),
            },
        )

    def forward_sequence(
        self,
        chunks: torch.Tensor,
        state: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        features = self._encode_features(chunks)
        previous_feature = None if state is None else state.get("previous_feature")
        tokens, last_feature = self._tokens_from_features(
            features,
            previous_feature,
        )
        temporal_state = None
        if state is not None:
            temporal_state = {
                key: value
                for key, value in state.items()
                if key != "previous_feature"
            }
        outputs = [[], [], []]
        for token in tokens.unbind(dim=1):
            candidate, event, delay, temporal_state = self._token_step(
                token,
                temporal_state,
            )
            outputs[0].append(candidate)
            outputs[1].append(event)
            outputs[2].append(delay)
        next_state = dict(temporal_state)
        next_state["previous_feature"] = last_feature.detach()
        return (
            torch.stack(outputs[0], dim=1),
            torch.stack(outputs[1], dim=1),
            torch.stack(outputs[2], dim=1),
            next_state,
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
        candidate, event, delay, next_state = self.forward_sequence(
            packet.unsqueeze(1),
            state,
        )
        return (
            candidate[0, 0],
            event[0, 0],
            delay[0, 0],
            next_state,
        )

    @classmethod
    def from_config(cls, config: dict) -> "CSPPCausalPickerV25":
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
    "CSPPCausalPickerV25",
    "CSPPV25Config",
    "CSPPPacketEncoderV25",
    "CompletePacketResidualBlock",
]
