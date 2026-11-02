"""Small artifact classifier over already-computed current-packet features."""
from __future__ import annotations

import torch
import torch.nn as nn


class CurrentPacketArtifactHead(nn.Module):
    """Standardized MLP that adds no raw-packet encoder or temporal state."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (64, 32),
        dropout: float = 0.05,
        standardized_clip: float = 10.0,
    ):
        super().__init__()
        if input_dim < 1:
            raise ValueError("input_dim must be positive")
        if any(int(width) < 1 for width in hidden_dims):
            raise ValueError("hidden dimensions must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if standardized_clip <= 0.0:
            raise ValueError("standardized_clip must be positive")
        self.input_dim = int(input_dim)
        self.hidden_dims = tuple(int(width) for width in hidden_dims)
        self.dropout = float(dropout)
        self.standardized_clip = float(standardized_clip)
        layers: list[nn.Module] = []
        width = self.input_dim
        for hidden in self.hidden_dims:
            layers.extend(
                (
                    nn.Linear(width, hidden),
                    nn.SiLU(),
                    nn.Dropout(self.dropout),
                )
            )
            width = hidden
        layers.append(nn.Linear(width, 1))
        self.network = nn.Sequential(*layers)
        self.register_buffer("standard_mean", torch.zeros(self.input_dim))
        self.register_buffer("standard_scale", torch.ones(self.input_dim))

    def set_standardization(
        self,
        mean: torch.Tensor,
        scale: torch.Tensor,
    ) -> None:
        mean = torch.as_tensor(mean, dtype=self.standard_mean.dtype)
        scale = torch.as_tensor(scale, dtype=self.standard_scale.dtype)
        if tuple(mean.shape) != (self.input_dim,) or tuple(scale.shape) != (
            self.input_dim,
        ):
            raise ValueError("Standardization vectors do not match input_dim")
        if not torch.isfinite(mean).all() or not torch.isfinite(scale).all():
            raise ValueError("Standardization vectors must be finite")
        if not (scale > 0.0).all():
            raise ValueError("Standardization scales must be positive")
        self.standard_mean.copy_(mean)
        self.standard_scale.copy_(scale)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected feature width {self.input_dim}, "
                f"got {features.shape[-1]}"
            )
        standardized = (features - self.standard_mean) / self.standard_scale
        standardized = standardized.clamp(
            -self.standardized_clip, self.standardized_clip
        )
        return self.network(standardized).squeeze(-1)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def config(self) -> dict:
        return {
            "input_dim": self.input_dim,
            "hidden_dims": list(self.hidden_dims),
            "dropout": self.dropout,
            "standardized_clip": self.standardized_clip,
        }


__all__ = ["CurrentPacketArtifactHead"]
