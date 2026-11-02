"""Current-packet component and summary features."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        derivative_ratio = (
            torch.clamp(
                torch.sqrt(derivative.square().mean(dim=-1) + eps) / rms_total,
                0.0,
                5.0,
            )
            / 5.0
        )
        z_fraction = torch.clamp(rms_z.square() / (rms_total.square() + eps), 0.0, 1.0)
        rms_n = torch.sqrt(n.square().mean(dim=-1) + eps)
        rms_e = torch.sqrt(e.square().mean(dim=-1) + eps)
        horizontal_balance = (rms_n - rms_e) / (rms_n + rms_e + eps)

        total_1d = total.unsqueeze(1)
        low5 = F.avg_pool1d(
            F.pad(total_1d, (4, 0), mode="replicate"), 5, stride=1
        ).squeeze(1)
        low15 = F.avg_pool1d(
            F.pad(total_1d, (14, 0), mode="replicate"), 15, stride=1
        ).squeeze(1)
        high_ratio = (
            torch.clamp(
                torch.sqrt((total - low5).square().mean(dim=-1) + eps) / rms_total,
                0.0,
                3.0,
            )
            / 3.0
        )
        mid_ratio = (
            torch.clamp(
                torch.sqrt((low5 - low15).square().mean(dim=-1) + eps) / rms_total,
                0.0,
                3.0,
            )
            / 3.0
        )
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
