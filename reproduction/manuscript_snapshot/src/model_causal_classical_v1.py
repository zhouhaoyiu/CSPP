"""Parameter-free packet-once recursive energy baseline."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn


@dataclass(frozen=True)
class CausalClassicalV1Config:
    sampling_rate_hz: float = 100.0
    packet_seconds: float = 0.5
    sta_half_lives_seconds: tuple[float, ...] = (0.1, 0.3)
    lta_half_life_seconds: float = 5.0
    lta_update_cap: float = 16.0
    sustained_samples: int = 10

    def as_dict(self) -> dict:
        return {
            "sampling_rate_hz": self.sampling_rate_hz,
            "packet_seconds": self.packet_seconds,
            "sta_half_lives_seconds": list(self.sta_half_lives_seconds),
            "lta_half_life_seconds": self.lta_half_life_seconds,
            "lta_update_cap": self.lta_update_cap,
            "sustained_samples": self.sustained_samples,
        }


class CausalMultiBandSTALTAPickerV1(nn.Module):
    """Recursive raw/difference energy detector with fixed-size state."""

    model_class = "CausalMultiBandSTALTAPickerV1"
    timing_mode = "packet_offset"
    packet_once = True
    raw_replay_packets = 0
    receptive_field_packets = math.inf

    def __init__(
        self,
        sampling_rate_hz: float = 100.0,
        packet_seconds: float = 0.5,
        sta_half_lives_seconds: tuple[float, ...] | list[float] = (0.1, 0.3),
        lta_half_life_seconds: float = 5.0,
        lta_update_cap: float = 16.0,
        sustained_samples: int = 10,
    ):
        super().__init__()
        sta_half_lives = tuple(float(value) for value in sta_half_lives_seconds)
        if sampling_rate_hz <= 0.0 or packet_seconds <= 0.0:
            raise ValueError("sampling_rate_hz and packet_seconds must be positive")
        if not sta_half_lives or any(value <= 0.0 for value in sta_half_lives):
            raise ValueError("STA half-lives must be non-empty and positive")
        if lta_half_life_seconds <= max(sta_half_lives):
            raise ValueError("LTA half-life must exceed all STA half-lives")
        if lta_update_cap < 1.0:
            raise ValueError("lta_update_cap must be at least one")
        if sustained_samples < 1:
            raise ValueError("sustained_samples must be positive")
        self.config = CausalClassicalV1Config(
            sampling_rate_hz=float(sampling_rate_hz),
            packet_seconds=float(packet_seconds),
            sta_half_lives_seconds=sta_half_lives,
            lta_half_life_seconds=float(lta_half_life_seconds),
            lta_update_cap=float(lta_update_cap),
            sustained_samples=int(sustained_samples),
        )
        sample_seconds = 1.0 / float(sampling_rate_hz)
        sta_alphas = [
            math.exp(math.log(0.5) * sample_seconds / half_life)
            for half_life in sta_half_lives
        ]
        lta_alpha = math.exp(
            math.log(0.5) * sample_seconds / float(lta_half_life_seconds)
        )
        self.register_buffer(
            "sta_alphas",
            torch.tensor(sta_alphas, dtype=torch.float32).view(1, -1, 1),
        )
        self.register_buffer(
            "lta_alpha",
            torch.tensor(lta_alpha, dtype=torch.float32),
        )

    @property
    def state_size_scalars(self) -> int:
        scales = len(self.config.sta_half_lives_seconds)
        return 3 + 2 + 2 * scales + 1

    def _initial_state(self, packet: torch.Tensor) -> dict:
        batch = packet.shape[0]
        scales = len(self.config.sta_half_lives_seconds)
        return {
            "previous_sample": packet.new_zeros(batch, 3),
            "sta": packet.new_zeros(batch, scales, 2),
            "lta": packet.new_zeros(batch, 2),
            "initialized": torch.zeros(batch, dtype=torch.bool, device=packet.device),
        }

    def _packet_step(
        self,
        packet: torch.Tensor,
        state: dict | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        if packet.ndim != 3 or packet.shape[1] != 3:
            raise ValueError(f"Expected packet shape (B, 3, L), got {tuple(packet.shape)}")
        current = self._initial_state(packet) if state is None else {
            key: value.to(packet.device) for key, value in state.items()
        }
        batch = packet.shape[0]
        expected_sta = (batch, len(self.config.sta_half_lives_seconds), 2)
        if tuple(current["previous_sample"].shape) != (batch, 3):
            raise ValueError("Classical state batch size does not match packet batch size")
        if tuple(current["sta"].shape) != expected_sta:
            raise ValueError(f"Expected STA state shape {expected_sta}")
        if tuple(current["lta"].shape) != (batch, 2):
            raise ValueError("Classical LTA state has an invalid shape")

        sta_alpha = self.sta_alphas.to(device=packet.device, dtype=packet.dtype)
        lta_alpha = self.lta_alpha.to(device=packet.device, dtype=packet.dtype)
        ratios = []
        previous = current["previous_sample"]
        sta = current["sta"]
        lta = current["lta"]
        initialized = current["initialized"]
        eps = torch.finfo(packet.dtype).eps

        for sample in packet.unbind(dim=2):
            difference = sample - previous
            signals = torch.stack(
                (sample.square().mean(dim=1), difference.square().mean(dim=1)),
                dim=1,
            ).clamp_min(eps)
            first = ~initialized
            if first.any():
                lta = torch.where(first.unsqueeze(1), signals, lta)
                sta = torch.where(first.view(batch, 1, 1), signals.unsqueeze(1), sta)
            sta = sta_alpha * sta + (1.0 - sta_alpha) * signals.unsqueeze(1)
            capped = torch.minimum(signals, lta.clamp_min(eps) * self.config.lta_update_cap)
            lta = lta_alpha * lta + (1.0 - lta_alpha) * capped
            ratios.append((sta / lta.clamp_min(eps).unsqueeze(1)).amax(dim=(1, 2)))
            previous = sample
            initialized = torch.ones_like(initialized)

        ratio_sequence = torch.stack(ratios, dim=1).clamp_min(eps)
        peak_ratio, peak_index = ratio_sequence.max(dim=1)
        top_count = min(self.config.sustained_samples, packet.shape[-1])
        sustained_ratio = ratio_sequence.topk(top_count, dim=1).values.mean(dim=1)
        offset = (peak_index.to(packet.dtype) + 0.5) / packet.shape[-1]
        offset = offset.clamp(1e-4, 1.0 - 1e-4)
        next_state = {
            "previous_sample": previous.detach(),
            "sta": sta.detach(),
            "lta": lta.detach(),
            "initialized": initialized.detach(),
        }
        return (
            peak_ratio.log(),
            sustained_ratio.log(),
            torch.logit(offset),
            next_state,
        )

    def forward_sequence(
        self,
        chunks: torch.Tensor,
        state: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        if chunks.ndim != 4 or chunks.shape[2] != 3:
            raise ValueError(f"Expected chunks shape (B, T, 3, L), got {tuple(chunks.shape)}")
        if chunks.shape[1] < 1:
            raise ValueError("At least one packet is required")
        outputs = [[], [], []]
        current = state
        for packet in chunks.unbind(dim=1):
            heads = self._packet_step(packet, current)
            current = heads[3]
            for values, value in zip(outputs, heads[:3], strict=True):
                values.append(value)
        return (
            torch.stack(outputs[0], dim=1),
            torch.stack(outputs[1], dim=1),
            torch.stack(outputs[2], dim=1),
            current,
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
        heads = self._packet_step(packet, state)
        return heads[0][0], heads[1][0], heads[2][0], heads[3]

    @classmethod
    def from_config(cls, config: dict) -> "CausalMultiBandSTALTAPickerV1":
        keys = CausalClassicalV1Config.__dataclass_fields__.keys()
        values = {key: config[key] for key in keys if key in config}
        if "sta_half_lives_seconds" in values:
            values["sta_half_lives_seconds"] = tuple(values["sta_half_lives_seconds"])
        return cls(**values)


__all__ = ["CausalClassicalV1Config", "CausalMultiBandSTALTAPickerV1"]
