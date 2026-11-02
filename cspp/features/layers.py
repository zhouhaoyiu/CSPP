"""Shared complete-packet neural network layers."""

from __future__ import annotations

import torch
import torch.nn as nn


def _group_count(channels: int, maximum: int = 8) -> int:
    for groups in range(min(maximum, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


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
