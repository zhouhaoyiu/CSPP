"""Load current-family state dictionaries without importing historical models."""

from __future__ import annotations

from pathlib import Path, PosixPath
from typing import Any, Mapping

import torch

from cspp.model import CSPPCausalPickerV27, CSPPCausalPickerV27B, CSPPCausalPickerV27C


MODEL_TYPES = {
    model.model_class: model
    for model in (CSPPCausalPickerV27, CSPPCausalPickerV27B, CSPPCausalPickerV27C)
}


def model_from_checkpoint(checkpoint: Mapping[str, Any]) -> torch.nn.Module:
    model_class = checkpoint.get("model_class")
    if model_class not in MODEL_TYPES:
        raise ValueError(
            f"Unsupported current-family model: {model_class!r}. "
            "Historical weights use src/domain_robust_model_io.py."
        )
    model = MODEL_TYPES[model_class].from_config(checkpoint["model_config"])
    model.load_state_dict(checkpoint["model"], strict=True)
    return model


def load_checkpoint_model(
    path: str | Path,
    device: torch.device | str = "cpu",
) -> tuple[torch.nn.Module, dict]:
    # Published research checkpoints include a PosixPath in training metadata.
    with torch.serialization.safe_globals([Path, PosixPath]):
        checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise TypeError("Expected a checkpoint dictionary")
    model = model_from_checkpoint(checkpoint).eval().to(device)
    return model, checkpoint


__all__ = ["MODEL_TYPES", "load_checkpoint_model", "model_from_checkpoint"]
