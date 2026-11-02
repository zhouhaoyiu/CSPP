"""Compact packet-causal CSPP model with dense within-packet P evidence."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_cspp_v25 import CompletePacketResidualBlock
from model_domain_robust_v5 import (
    PacketPhysicsFeatures,
    PacketSummaryFeatures,
    _group_count,
)
from model_domain_robust_v9 import FixedPacketSpectrum


@dataclass(frozen=True)
class CSPPV27Config:
    packet_width: int = 64
    embedding_dim: int = 64
    sequence_width: int = 128
    gru_layers: int = 2
    temporal_bins: int = 5
    dropout: float = 0.15
    packet_kernels: tuple[int, ...] = (3, 7, 15)
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
    temporal_mode: str = "gru"
    dense_phase_evidence: bool = True
    spectral_features: bool = True
    decoder_attention_heads: int = 4
    decoder_context_packets: int = 32
    decoder_ffn_multiplier: float = 2.0

    def as_dict(self) -> dict:
        payload = dict(self.__dict__)
        payload["packet_kernels"] = list(self.packet_kernels)
        payload["spectral_band_edges_hz"] = list(
            self.spectral_band_edges_hz
        )
        return payload


@dataclass(frozen=True)
class CSPPV27BConfig(CSPPV27Config):
    verifier_hidden_dim: int = 64


@dataclass(frozen=True)
class CSPPV27CConfig(CSPPV27Config):
    verifier_hidden_dim: int = 64
    verifier_feature_mode: str = "current_delta"


class CSPPPacketEncoderV27(nn.Module):
    """Robust packet features with a dense P localization branch."""

    def __init__(self, config: CSPPV27Config):
        super().__init__()
        self.config = config
        self.physics = PacketPhysicsFeatures()
        self.summary = PacketSummaryFeatures()
        self.spectrum = FixedPacketSpectrum(
            packet_samples=config.packet_samples,
            sample_rate_hz=config.sample_rate_hz,
            band_edges_hz=config.spectral_band_edges_hz,
        )
        width = config.packet_width
        self.stem = nn.Conv1d(
            self.physics.out_channels,
            width,
            kernel_size=5,
            padding=2,
            bias=False,
        )
        self.stem_norm = nn.GroupNorm(_group_count(width), width)
        self.blocks = nn.ModuleList(
            CompletePacketResidualBlock(
                width,
                kernel_size=kernel,
                dropout=config.dropout,
            )
            for kernel in config.packet_kernels
        )
        pooled_features = (
            2 * width * config.temporal_bins
            + self.summary.out_features
            + self.spectrum.out_features
        )
        self.embedding_projection = nn.Sequential(
            nn.Linear(pooled_features, config.embedding_dim),
            nn.LayerNorm(config.embedding_dim),
            nn.SiLU(),
            nn.Dropout(config.dropout),
        )
        phase_width = max(16, width // 2)
        self.phase_head = nn.Sequential(
            nn.GroupNorm(_group_count(width), width),
            nn.Conv1d(width, phase_width, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(phase_width, 1, kernel_size=1),
        )
        nn.init.constant_(self.phase_head[-1].bias, -4.0)

    def forward_with_details(
        self,
        packet: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        expected = (3, self.config.packet_samples)
        if packet.ndim != 3 or tuple(packet.shape[1:]) != expected:
            raise ValueError(
                f"Expected packet shape (B, {expected[0]}, {expected[1]}), "
                f"got {tuple(packet.shape)}"
            )
        encoded = F.silu(self.stem_norm(self.stem(self.physics(packet))))
        for block in self.blocks:
            encoded = block(encoded)
        average = F.adaptive_avg_pool1d(
            encoded,
            self.config.temporal_bins,
        )
        maximum = F.adaptive_max_pool1d(
            encoded,
            self.config.temporal_bins,
        )
        summary = self.summary(packet)
        if self.config.spectral_features:
            spectrum = self.spectrum(packet)
        else:
            spectrum = packet.new_zeros(
                packet.shape[0],
                self.spectrum.out_features,
            )
        verifier_features = torch.cat((summary, spectrum), dim=1)
        pooled = torch.cat(
            (
                average.flatten(1),
                maximum.flatten(1),
                verifier_features,
            ),
            dim=1,
        )
        embedding = self.embedding_projection(pooled)
        phase_logits = self.phase_head(encoded)
        phase_logits = F.interpolate(
            phase_logits,
            size=self.config.packet_samples,
            mode="linear",
            align_corners=False,
        ).squeeze(1)
        return embedding, phase_logits, verifier_features

    def forward_with_phase(
        self,
        packet: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        embedding, phase_logits, _ = self.forward_with_details(packet)
        return embedding, phase_logits

    def forward(self, packet: torch.Tensor) -> torch.Tensor:
        embedding, _ = self.forward_with_phase(packet)
        return embedding


class CSPPCausalDecoderBlock(nn.Module):
    """Pre-norm causal self-attention over packet tokens."""

    def __init__(
        self,
        width: int,
        heads: int,
        dropout: float,
        ffn_multiplier: float,
    ):
        super().__init__()
        hidden = max(width, int(round(width * ffn_multiplier)))
        self.attn_norm = nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(
            width,
            heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(width)
        self.ffn = nn.Sequential(
            nn.Linear(width, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, width),
        )
        self.ffn_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        normalized = self.attn_norm(x)
        attended, _ = self.attn(
            normalized,
            normalized,
            normalized,
            attn_mask=attention_mask,
            need_weights=False,
        )
        x = x + self.attn_dropout(attended)
        return x + self.ffn_dropout(self.ffn(self.ffn_norm(x)))


class CSPPCausalPickerV27(nn.Module):
    """Small causal CNN-GRU with packet-local dense P evidence.

    Symmetric interpolation is confined to the fully received current packet.
    Cross-packet information is carried only by a unidirectional GRU and the
    previous packet embedding.
    """

    model_class = "CSPPCausalPickerV27"
    timing_mode = "arrival_age"
    packet_once = True
    raw_replay_packets = 0
    event_head_role = "auxiliary_event_evidence"

    def __init__(
        self,
        packet_width: int = 64,
        embedding_dim: int = 64,
        sequence_width: int = 128,
        gru_layers: int = 2,
        temporal_bins: int = 5,
        dropout: float = 0.15,
        packet_kernels: tuple[int, ...] | list[int] = (3, 7, 15),
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
        temporal_mode: str = "gru",
        dense_phase_evidence: bool = True,
        spectral_features: bool = True,
        decoder_attention_heads: int = 4,
        decoder_context_packets: int = 32,
        decoder_ffn_multiplier: float = 2.0,
        retention_half_lives_seconds: tuple[float, ...] | list[float] | None = None,
    ):
        super().__init__()
        del retention_half_lives_seconds
        if packet_width < 16 or embedding_dim < 16 or sequence_width < 16:
            raise ValueError("V27 widths must be at least 16")
        if gru_layers < 1:
            raise ValueError("gru_layers must be positive")
        if max_arrival_age_packets <= 0.0:
            raise ValueError("max_arrival_age_packets must be positive")
        if temporal_mode not in {"gru", "packet", "decoder"}:
            raise ValueError(
                "temporal_mode must be 'gru', 'packet', or 'decoder'"
            )
        if decoder_attention_heads < 1:
            raise ValueError("decoder_attention_heads must be positive")
        if sequence_width % decoder_attention_heads != 0:
            raise ValueError(
                "sequence_width must be divisible by decoder_attention_heads"
            )
        if decoder_context_packets < 2:
            raise ValueError("decoder_context_packets must be at least 2")
        if decoder_ffn_multiplier <= 0.0:
            raise ValueError("decoder_ffn_multiplier must be positive")
        self.config = CSPPV27Config(
            packet_width=int(packet_width),
            embedding_dim=int(embedding_dim),
            sequence_width=int(sequence_width),
            gru_layers=int(gru_layers),
            temporal_bins=int(temporal_bins),
            dropout=float(dropout),
            packet_kernels=tuple(int(value) for value in packet_kernels),
            max_arrival_age_packets=float(max_arrival_age_packets),
            packet_samples=int(packet_samples),
            sample_rate_hz=float(sample_rate_hz),
            spectral_band_edges_hz=tuple(
                float(value) for value in spectral_band_edges_hz
            ),
            temporal_mode=str(temporal_mode),
            dense_phase_evidence=bool(dense_phase_evidence),
            spectral_features=bool(spectral_features),
            decoder_attention_heads=int(decoder_attention_heads),
            decoder_context_packets=int(decoder_context_packets),
            decoder_ffn_multiplier=float(decoder_ffn_multiplier),
        )
        self.encoder = CSPPPacketEncoderV27(self.config)
        token_width = 2 * self.config.embedding_dim
        if self.config.temporal_mode == "gru":
            self.gru = nn.GRU(
                token_width,
                self.config.sequence_width,
                num_layers=self.config.gru_layers,
                batch_first=True,
                dropout=(
                    self.config.dropout
                    if self.config.gru_layers > 1
                    else 0.0
                ),
                bidirectional=False,
            )
        elif self.config.temporal_mode == "packet":
            self.packet_temporal = nn.Sequential(
                nn.LayerNorm(self.config.embedding_dim),
                nn.Linear(
                    self.config.embedding_dim,
                    self.config.sequence_width,
                ),
                nn.SiLU(),
                nn.Dropout(self.config.dropout),
            )
        else:
            self.decoder_input = nn.Sequential(
                nn.LayerNorm(token_width),
                nn.Linear(token_width, self.config.sequence_width),
            )
            self.decoder_blocks = nn.ModuleList(
                CSPPCausalDecoderBlock(
                    self.config.sequence_width,
                    self.config.decoder_attention_heads,
                    self.config.dropout,
                    self.config.decoder_ffn_multiplier,
                )
                for _ in range(self.config.gru_layers)
            )
            self.decoder_norm = nn.LayerNorm(self.config.sequence_width)
        candidate_input = self.config.sequence_width + token_width + 2
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
        nn.init.constant_(self.candidate_head[-1].bias, -3.0)
        nn.init.constant_(self.event_head[-1].bias, -3.0)
        nn.init.zeros_(self.delay_head[-1].bias)

    @property
    def max_arrival_age_packets(self) -> float:
        return self.config.max_arrival_age_packets

    @property
    def receptive_field_packets(self) -> float:
        if self.config.temporal_mode == "gru":
            return math.inf
        if self.config.temporal_mode == "decoder":
            return float(self.config.decoder_context_packets)
        return 1.0

    @property
    def state_size_scalars(self) -> int:
        if self.config.temporal_mode == "packet":
            return 0
        if self.config.temporal_mode == "decoder":
            return (
                self.config.embedding_dim
                + 1
                + (self.config.decoder_context_packets - 1)
                * 2
                * self.config.embedding_dim
            )
        return (
            self.config.embedding_dim
            + self.config.gru_layers * self.config.sequence_width
        )

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

    def _encode_packets(
        self,
        chunks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        expected = (3, self.config.packet_samples)
        if chunks.ndim != 4 or tuple(chunks.shape[2:]) != expected:
            raise ValueError(
                f"Expected chunks shape (B, T, {expected[0]}, {expected[1]}), "
                f"got {tuple(chunks.shape)}"
            )
        batch, packets, channels, samples = chunks.shape
        features, phase_logits = self.encoder.forward_with_phase(
            chunks.reshape(batch * packets, channels, samples)
        )
        return (
            features.reshape(batch, packets, -1),
            phase_logits.reshape(batch, packets, samples),
        )

    def _tokens_from_features(
        self,
        features: torch.Tensor,
        previous_feature: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.config.temporal_mode == "packet":
            return (
                torch.cat((features, torch.zeros_like(features)), dim=-1),
                features[:, -1],
            )
        if previous_feature is None:
            previous = torch.cat(
                (torch.zeros_like(features[:, :1]), features[:, :-1]),
                dim=1,
            )
            delta = (features - previous) / math.sqrt(2.0)
            delta[:, 0] = 0.0
        else:
            expected = (features.shape[0], self.config.embedding_dim)
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
        return torch.cat((features, delta), dim=-1), features[:, -1]

    @staticmethod
    def _phase_evidence(phase_logits: torch.Tensor) -> torch.Tensor:
        sample_count = phase_logits.shape[-1]
        peak = phase_logits.amax(dim=-1, keepdim=True)
        mass = (
            torch.logsumexp(phase_logits, dim=-1, keepdim=True)
            - math.log(sample_count)
        )
        return torch.cat((peak, mass), dim=-1)

    def _candidate_phase_evidence(
        self,
        phase_logits: torch.Tensor,
    ) -> torch.Tensor:
        if self.config.dense_phase_evidence:
            return self._phase_evidence(phase_logits)
        return phase_logits.new_zeros(*phase_logits.shape[:-1], 2)

    def _decoder_attention_mask(
        self,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        age = positions.unsqueeze(1) - positions.unsqueeze(0)
        invalid = (age < 0) | (age >= self.config.decoder_context_packets)
        mask = torch.zeros(
            positions.shape[0],
            positions.shape[0],
            device=positions.device,
            dtype=torch.float32,
        )
        mask = mask.masked_fill(invalid, float("-inf"))
        recency_bias = -0.02 * age.clamp_min(0).to(torch.float32)
        return mask + recency_bias.masked_fill(invalid, 0.0)

    def _decoder_step(
        self,
        tokens: torch.Tensor,
        state: dict | None = None,
    ) -> tuple[torch.Tensor, dict]:
        if tokens.shape[1] != 1:
            raise ValueError("Decoder step expects exactly one packet token")
        cached_tokens = None if state is None else state.get("decoder_tokens")
        raw_position = None if state is None else state.get(
            "decoder_position",
        )
        if raw_position is None:
            position = 0
        elif torch.is_tensor(raw_position):
            position = int(raw_position.flatten()[0].item())
        else:
            position = int(raw_position)
        if cached_tokens is not None:
            expected = (tokens.shape[0], None, tokens.shape[-1])
            if (
                cached_tokens.ndim != 3
                or cached_tokens.shape[0] != expected[0]
                or cached_tokens.shape[2] != expected[2]
                or cached_tokens.shape[1]
                > self.config.decoder_context_packets - 1
            ):
                raise ValueError(
                    "Expected decoder_tokens shape "
                    f"(B, <= {self.config.decoder_context_packets - 1}, "
                    f"{tokens.shape[-1]}), got {tuple(cached_tokens.shape)}"
                )
            cached_tokens = cached_tokens.to(
                device=tokens.device,
                dtype=tokens.dtype,
            )
            all_tokens = torch.cat((cached_tokens, tokens), dim=1)
        else:
            all_tokens = tokens
        cache_len = 0 if cached_tokens is None else cached_tokens.shape[1]
        first_position = position - cache_len
        positions = torch.arange(
            first_position,
            position + tokens.shape[1],
            device=tokens.device,
        )
        x = self.decoder_input(all_tokens)
        attention_mask = self._decoder_attention_mask(positions).to(
            dtype=x.dtype,
        )
        for block in self.decoder_blocks:
            x = block(x, attention_mask)
        x = self.decoder_norm(x)
        next_cache_count = min(
            self.config.decoder_context_packets - 1,
            all_tokens.shape[1],
        )
        next_state = {
            "decoder_tokens": all_tokens[:, -next_cache_count:].detach(),
            "decoder_position": torch.full(
                (tokens.shape[0], 1),
                position + tokens.shape[1],
                device=tokens.device,
                dtype=tokens.dtype,
            ),
        }
        return x[:, cache_len:], next_state

    def _decoder_sequence(
        self,
        tokens: torch.Tensor,
        state: dict | None = None,
    ) -> tuple[torch.Tensor, dict]:
        outputs = []
        next_state = state
        for index in range(tokens.shape[1]):
            output, next_state = self._decoder_step(
                tokens[:, index : index + 1],
                next_state,
            )
            outputs.append(output)
        return torch.cat(outputs, dim=1), next_state

    def _temporal_sequence(
        self,
        tokens: torch.Tensor,
        state: torch.Tensor | dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | dict | None]:
        if self.config.temporal_mode == "packet":
            return (
                self.packet_temporal(
                    tokens[..., : self.config.embedding_dim]
                ),
                None,
            )
        if self.config.temporal_mode == "decoder":
            decoder_state = None if state is None else dict(state)
            return self._decoder_sequence(tokens, decoder_state)
        return self.gru(tokens, state)

    def _heads(
        self,
        tokens: torch.Tensor,
        sequence: torch.Tensor,
        phase_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        candidate = self.candidate_head(
            torch.cat(
                (
                    sequence,
                    tokens,
                    self._candidate_phase_evidence(phase_logits),
                ),
                dim=-1,
            )
        ).squeeze(-1)
        event = self.event_head(sequence.detach()).squeeze(-1)
        delay = self.delay_head(sequence).squeeze(-1)
        return candidate, event, delay

    def forward_with_aux(
        self,
        chunks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        features, phase_logits = self._encode_packets(chunks)
        tokens, _ = self._tokens_from_features(features)
        sequence, _ = self._temporal_sequence(tokens)
        return (*self._heads(tokens, sequence, phase_logits), phase_logits)

    def forward(
        self,
        chunks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        candidate, event, delay, _ = self.forward_with_aux(chunks)
        return candidate, event, delay

    def forward_sequence_with_aux(
        self,
        chunks: torch.Tensor,
        state: dict | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict,
    ]:
        features, phase_logits = self._encode_packets(chunks)
        previous_feature = None if state is None else state.get("previous_feature")
        tokens, last_feature = self._tokens_from_features(
            features,
            previous_feature,
        )
        temporal_state: torch.Tensor | dict | None = None
        if self.config.temporal_mode == "gru":
            temporal_state = None if state is None else state.get("gru")
        elif self.config.temporal_mode == "decoder" and state is not None:
            temporal_state = {
                key: value
                for key, value in state.items()
                if key.startswith("decoder_")
            }
        if self.config.temporal_mode == "gru" and temporal_state is not None:
            expected = (
                self.config.gru_layers,
                chunks.shape[0],
                self.config.sequence_width,
            )
            if tuple(temporal_state.shape) != expected:
                raise ValueError(
                    "Expected GRU state "
                    f"{expected}, got {tuple(temporal_state.shape)}"
                )
            temporal_state = temporal_state.to(
                device=chunks.device,
                dtype=chunks.dtype,
            )
        sequence, next_temporal_state = self._temporal_sequence(
            tokens,
            temporal_state,
        )
        candidate, event, delay = self._heads(
            tokens,
            sequence,
            phase_logits,
        )
        next_state = {}
        if self.config.temporal_mode == "gru":
            next_state = {
                "previous_feature": last_feature.detach(),
                "gru": next_temporal_state.detach(),
            }
        elif self.config.temporal_mode == "decoder":
            next_state = {
                "previous_feature": last_feature.detach(),
                **next_temporal_state,
            }
        return candidate, event, delay, phase_logits, next_state

    def forward_sequence(
        self,
        chunks: torch.Tensor,
        state: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        candidate, event, delay, _, next_state = self.forward_sequence_with_aux(
            chunks,
            state,
        )
        return candidate, event, delay, next_state

    @torch.inference_mode()
    def forward_streaming_packet_with_aux(
        self,
        packet: torch.Tensor,
        state: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        if packet.ndim == 2:
            packet = packet.unsqueeze(0)
        if packet.ndim != 3 or packet.shape[0] != 1:
            raise ValueError("Streaming inference expects (3, L) or (1, 3, L)")
        candidate, event, delay, phase, next_state = (
            self.forward_sequence_with_aux(packet.unsqueeze(1), state)
        )
        return (
            candidate[0, 0],
            event[0, 0],
            delay[0, 0],
            phase[0, 0],
            next_state,
        )

    @torch.inference_mode()
    def forward_streaming_packet(
        self,
        packet: torch.Tensor,
        state: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        candidate, event, delay, _, next_state = (
            self.forward_streaming_packet_with_aux(packet, state)
        )
        return candidate, event, delay, next_state

    @classmethod
    def from_config(cls, config: dict) -> "CSPPCausalPickerV27":
        keys = CSPPV27Config.__dataclass_fields__.keys()
        values = {key: config[key] for key in keys if key in config}
        for key in ("packet_kernels", "spectral_band_edges_hz"):
            if key in values:
                values[key] = tuple(values[key])
        return cls(**values)


class CSPPCausalPickerV27B(CSPPCausalPickerV27):
    """V27 candidate backbone with a detached nonlinear event verifier."""

    model_class = "CSPPCausalPickerV27B"
    event_head_role = "frozen_candidate_nonlinear_event_verifier"

    def __init__(
        self,
        *args,
        verifier_hidden_dim: int = 64,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if verifier_hidden_dim < 16:
            raise ValueError("verifier_hidden_dim must be at least 16")
        self.config = CSPPV27BConfig(
            **self.config.__dict__,
            verifier_hidden_dim=int(verifier_hidden_dim),
        )
        token_width = 2 * self.config.embedding_dim
        verifier_input = self.config.sequence_width + token_width + 2
        self.event_head = nn.Sequential(
            nn.LayerNorm(verifier_input),
            nn.Linear(verifier_input, self.config.verifier_hidden_dim),
            nn.SiLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.verifier_hidden_dim, 1),
        )
        nn.init.constant_(self.event_head[-1].bias, -3.0)

    def _heads(
        self,
        tokens: torch.Tensor,
        sequence: torch.Tensor,
        phase_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        evidence = torch.cat(
            (
                sequence,
                tokens,
                self._candidate_phase_evidence(phase_logits),
            ),
            dim=-1,
        )
        candidate = self.candidate_head(evidence).squeeze(-1)
        event = self.event_head(evidence.detach()).squeeze(-1)
        delay = self.delay_head(sequence).squeeze(-1)
        return candidate, event, delay

    def load_v27_candidate_state_dict(self, state_dict: dict) -> None:
        """Load every V27 tensor except its incompatible linear event head."""
        candidate_state = {
            key: value
            for key, value in state_dict.items()
            if not key.startswith("event_head.")
        }
        incompatible = self.load_state_dict(candidate_state, strict=False)
        expected_missing = {
            key for key in self.state_dict() if key.startswith("event_head.")
        }
        if set(incompatible.missing_keys) != expected_missing:
            raise ValueError(
                "V27 initializer is missing non-verifier tensors: "
                f"{sorted(set(incompatible.missing_keys) - expected_missing)}"
            )
        if incompatible.unexpected_keys:
            raise ValueError(
                "V27 initializer contains unexpected tensors: "
                f"{sorted(incompatible.unexpected_keys)}"
            )

    def freeze_candidate(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad = False
        for parameter in self.event_head.parameters():
            parameter.requires_grad = True

    @classmethod
    def from_config(cls, config: dict) -> "CSPPCausalPickerV27B":
        keys = CSPPV27BConfig.__dataclass_fields__.keys()
        values = {key: config[key] for key in keys if key in config}
        for key in ("packet_kernels", "spectral_band_edges_hz"):
            if key in values:
                values[key] = tuple(values[key])
        return cls(**values)


class CSPPCausalPickerV27C(CSPPCausalPickerV27):
    """Frozen V27 candidate with direct packet evidence for verification."""

    model_class = "CSPPCausalPickerV27C"
    event_head_role = "episode_tail_direct_packet_evidence_verifier"
    verifier_feature_modes = {"none", "current", "current_delta"}

    def __init__(
        self,
        *args,
        verifier_hidden_dim: int = 64,
        verifier_feature_mode: str = "current_delta",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if verifier_hidden_dim < 16:
            raise ValueError("verifier_hidden_dim must be at least 16")
        if verifier_feature_mode not in self.verifier_feature_modes:
            raise ValueError(
                f"Unknown verifier_feature_mode={verifier_feature_mode!r}"
            )
        self.config = CSPPV27CConfig(
            **self.config.__dict__,
            verifier_hidden_dim=int(verifier_hidden_dim),
            verifier_feature_mode=str(verifier_feature_mode),
        )
        self.verifier_packet_features = (
            self.encoder.summary.out_features
            + self.encoder.spectrum.out_features
        )
        candidate_input = (
            self.config.sequence_width
            + 2 * self.config.embedding_dim
            + 2
        )
        if self.config.verifier_feature_mode == "current":
            verifier_input = candidate_input + self.verifier_packet_features
        elif self.config.verifier_feature_mode == "current_delta":
            verifier_input = (
                candidate_input + 2 * self.verifier_packet_features
            )
        else:
            verifier_input = candidate_input
        self.event_head = nn.Sequential(
            nn.LayerNorm(verifier_input),
            nn.Linear(verifier_input, self.config.verifier_hidden_dim),
            nn.SiLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.verifier_hidden_dim, 1),
        )
        nn.init.constant_(self.event_head[-1].bias, -3.0)

    @property
    def state_size_scalars(self) -> int:
        packet_state = (
            self.verifier_packet_features
            if self.config.verifier_feature_mode == "current_delta"
            else 0
        )
        return super().state_size_scalars + packet_state

    def _encode_packets_with_verifier(
        self,
        chunks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        expected = (3, self.config.packet_samples)
        if chunks.ndim != 4 or tuple(chunks.shape[2:]) != expected:
            raise ValueError(
                f"Expected chunks shape (B, T, {expected[0]}, {expected[1]}), "
                f"got {tuple(chunks.shape)}"
            )
        batch, packets, channels, samples = chunks.shape
        features, phase_logits, verifier_features = (
            self.encoder.forward_with_details(
                chunks.reshape(batch * packets, channels, samples)
            )
        )
        return (
            features.reshape(batch, packets, -1),
            phase_logits.reshape(batch, packets, samples),
            verifier_features.reshape(batch, packets, -1),
        )

    def _verifier_inputs(
        self,
        base_evidence: torch.Tensor,
        packet_features: torch.Tensor,
        previous_packet_features: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mode = self.config.verifier_feature_mode
        if mode == "none":
            return base_evidence.detach(), packet_features[:, -1]
        if mode == "current":
            return (
                torch.cat(
                    (base_evidence, packet_features),
                    dim=-1,
                ).detach(),
                packet_features[:, -1],
            )
        if previous_packet_features is None:
            previous = torch.cat(
                (packet_features[:, :1], packet_features[:, :-1]),
                dim=1,
            )
        else:
            expected = (
                packet_features.shape[0],
                self.verifier_packet_features,
            )
            if tuple(previous_packet_features.shape) != expected:
                raise ValueError(
                    f"Expected verifier packet state {expected}, "
                    f"got {tuple(previous_packet_features.shape)}"
                )
            previous = torch.cat(
                (
                    previous_packet_features.to(
                        device=packet_features.device,
                        dtype=packet_features.dtype,
                    ).unsqueeze(1),
                    packet_features[:, :-1],
                ),
                dim=1,
            )
        delta = torch.clamp(packet_features - previous, -5.0, 5.0)
        return (
            torch.cat(
                (base_evidence, packet_features, delta),
                dim=-1,
            ).detach(),
            packet_features[:, -1],
        )

    def _outputs(
        self,
        tokens: torch.Tensor,
        sequence: torch.Tensor,
        phase_logits: torch.Tensor,
        packet_features: torch.Tensor,
        previous_packet_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        base_evidence = torch.cat(
            (
                sequence,
                tokens,
                self._candidate_phase_evidence(phase_logits),
            ),
            dim=-1,
        )
        candidate = self.candidate_head(base_evidence).squeeze(-1)
        verifier_inputs, last_packet_features = self._verifier_inputs(
            base_evidence,
            packet_features,
            previous_packet_features,
        )
        event = self.event_head(verifier_inputs).squeeze(-1)
        delay = self.delay_head(sequence).squeeze(-1)
        return candidate, event, delay, last_packet_features

    def forward_with_aux(
        self,
        chunks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        features, phase_logits, packet_features = (
            self._encode_packets_with_verifier(chunks)
        )
        tokens, _ = self._tokens_from_features(features)
        sequence, _ = self._temporal_sequence(tokens)
        candidate, event, delay, _ = self._outputs(
            tokens,
            sequence,
            phase_logits,
            packet_features,
        )
        return candidate, event, delay, phase_logits

    def forward_sequence_with_aux(
        self,
        chunks: torch.Tensor,
        state: dict | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict,
    ]:
        features, phase_logits, packet_features = (
            self._encode_packets_with_verifier(chunks)
        )
        previous_feature = None if state is None else state.get(
            "previous_feature"
        )
        tokens, last_feature = self._tokens_from_features(
            features,
            previous_feature,
        )
        temporal_state: torch.Tensor | dict | None = None
        if self.config.temporal_mode == "gru":
            temporal_state = None if state is None else state.get("gru")
        elif self.config.temporal_mode == "decoder" and state is not None:
            temporal_state = {
                key: value
                for key, value in state.items()
                if key.startswith("decoder_")
            }
        if self.config.temporal_mode == "gru" and temporal_state is not None:
            expected = (
                self.config.gru_layers,
                chunks.shape[0],
                self.config.sequence_width,
            )
            if tuple(temporal_state.shape) != expected:
                raise ValueError(
                    "Expected GRU state "
                    f"{expected}, got {tuple(temporal_state.shape)}"
                )
            temporal_state = temporal_state.to(
                device=chunks.device,
                dtype=chunks.dtype,
            )
        sequence, next_temporal_state = self._temporal_sequence(
            tokens,
            temporal_state,
        )
        previous_packet_features = (
            None if state is None else state.get("verifier_packet_features")
        )
        candidate, event, delay, last_packet_features = self._outputs(
            tokens,
            sequence,
            phase_logits,
            packet_features,
            previous_packet_features,
        )
        next_state = {}
        if self.config.temporal_mode == "gru":
            next_state = {
                "previous_feature": last_feature.detach(),
                "gru": next_temporal_state.detach(),
            }
        elif self.config.temporal_mode == "decoder":
            next_state = {
                "previous_feature": last_feature.detach(),
                **next_temporal_state,
            }
        if self.config.verifier_feature_mode == "current_delta":
            next_state["verifier_packet_features"] = (
                last_packet_features.detach()
            )
        return candidate, event, delay, phase_logits, next_state

    def load_v27_candidate_state_dict(self, state_dict: dict) -> None:
        candidate_state = {
            key: value
            for key, value in state_dict.items()
            if not key.startswith("event_head.")
        }
        incompatible = self.load_state_dict(candidate_state, strict=False)
        expected_missing = {
            key for key in self.state_dict() if key.startswith("event_head.")
        }
        if set(incompatible.missing_keys) != expected_missing:
            raise ValueError(
                "V27 initializer is missing non-verifier tensors: "
                f"{sorted(set(incompatible.missing_keys) - expected_missing)}"
            )
        if incompatible.unexpected_keys:
            raise ValueError(
                "V27 initializer contains unexpected tensors: "
                f"{sorted(incompatible.unexpected_keys)}"
            )

    def freeze_candidate(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad = False
        for parameter in self.event_head.parameters():
            parameter.requires_grad = True

    @classmethod
    def from_config(cls, config: dict) -> "CSPPCausalPickerV27C":
        keys = CSPPV27CConfig.__dataclass_fields__.keys()
        values = {key: config[key] for key in keys if key in config}
        for key in ("packet_kernels", "spectral_band_edges_hz"):
            if key in values:
                values[key] = tuple(values[key])
        return cls(**values)


__all__ = [
    "CSPPCausalPickerV27",
    "CSPPCausalPickerV27B",
    "CSPPCausalPickerV27C",
    "CSPPPacketEncoderV27",
    "CSPPV27Config",
    "CSPPV27BConfig",
    "CSPPV27CConfig",
]
