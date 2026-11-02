"""Checkpoint model factory shared by domain-robust evaluation and deployment tools."""
from __future__ import annotations

from pathlib import Path

import torch

from model_causal_classical_v1 import CausalMultiBandSTALTAPickerV1
from model_domain_robust_v5 import DomainRobustCausalPickerV5
from model_domain_robust_v6 import DomainRobustCausalPickerV6
from model_domain_robust_v7 import DomainRobustCausalPickerV7
from model_domain_robust_v8 import DomainRobustCausalPickerV8
from model_domain_robust_v9 import DomainRobustCausalPickerV9
from model_domain_robust_v10 import DomainRobustCausalPickerV10
from model_domain_robust_v11 import DomainRobustCausalPickerV11
from model_packet_once_v1 import PacketOnceCausalPickerV1
from model_cspp_v25 import CSPPCausalPickerV25
from model_cspp_v26 import CSPPCausalPickerV26
from model_cspp_v27 import (
    CSPPCausalPickerV27,
    CSPPCausalPickerV27B,
    CSPPCausalPickerV27C,
)

MODEL_TYPES = {
    "CausalMultiBandSTALTAPickerV1": CausalMultiBandSTALTAPickerV1,
    "DomainRobustCausalPickerV5": DomainRobustCausalPickerV5,
    "DomainRobustCausalPickerV6": DomainRobustCausalPickerV6,
    "DomainRobustCausalPickerV7": DomainRobustCausalPickerV7,
    "DomainRobustCausalPickerV8": DomainRobustCausalPickerV8,
    "DomainRobustCausalPickerV9": DomainRobustCausalPickerV9,
    "DomainRobustCausalPickerV10": DomainRobustCausalPickerV10,
    "DomainRobustCausalPickerV11": DomainRobustCausalPickerV11,
    "PacketOnceCausalPickerV1": PacketOnceCausalPickerV1,
    "CSPPCausalPickerV25": CSPPCausalPickerV25,
    "CSPPCausalPickerV26": CSPPCausalPickerV26,
    "CSPPCausalPickerV27": CSPPCausalPickerV27,
    "CSPPCausalPickerV27B": CSPPCausalPickerV27B,
    "CSPPCausalPickerV27C": CSPPCausalPickerV27C,
}


def model_from_checkpoint(checkpoint: dict):
    model_class = checkpoint.get("model_class", "DomainRobustCausalPickerV5")
    if model_class not in MODEL_TYPES:
        raise ValueError(f"Unsupported model class: {model_class}")
    model = MODEL_TYPES[model_class].from_config(checkpoint["model_config"])
    model.load_state_dict(checkpoint["model"])
    return model


def load_checkpoint_model(
    path: str | Path,
    device: torch.device | str = "cpu",
):
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    model = model_from_checkpoint(checkpoint).eval().to(device)
    return model, checkpoint


def candidate_policy_from_checkpoint(checkpoint: dict) -> dict:
    model_class = checkpoint.get("model_class", "DomainRobustCausalPickerV5")
    arrival_age_models = {
        "DomainRobustCausalPickerV7",
        "DomainRobustCausalPickerV8",
        "DomainRobustCausalPickerV9",
        "DomainRobustCausalPickerV10",
        "DomainRobustCausalPickerV11",
        "PacketOnceCausalPickerV1",
        "CSPPCausalPickerV25",
        "CSPPCausalPickerV26",
        "CSPPCausalPickerV27",
        "CSPPCausalPickerV27B",
        "CSPPCausalPickerV27C",
    }
    classical_model = model_class == "CausalMultiBandSTALTAPickerV1"
    defaults = {
        "candidate_history_packets": 4 if model_class in arrival_age_models else 1,
        "warmup_packets": 20 if model_class in arrival_age_models or classical_model else 0,
        "timing_mode": "arrival_age" if model_class in arrival_age_models else "packet_offset",
        "max_arrival_age_packets": float(
            checkpoint.get("model_config", {}).get("max_arrival_age_packets", 4.0)
        ),
        "onset_response_packets": 3,
        "verifier_response_packets": 4 if model_class in arrival_age_models else 6,
        "verifier_start_delay_packets": 0,
    }
    defaults.update(checkpoint.get("candidate_policy", {}))
    return defaults


__all__ = [
    "MODEL_TYPES",
    "candidate_policy_from_checkpoint",
    "load_checkpoint_model",
    "model_from_checkpoint",
]
