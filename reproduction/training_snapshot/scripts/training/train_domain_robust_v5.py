#!/usr/bin/env python3
"""Train causality-aligned, domain-balanced strong-motion picker variants."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import SAMPLES_PER_CHUNK, TARGET_SR  # noqa: E402
from domain_robust_v5_data import (  # noqa: E402
    ContinuousNoiseWindowSource,
    DomainRobustStreamDatasetV5,
    ONSET_RESPONSE_PACKETS,
    VERIFIER_RESPONSE_PACKETS,
    V7_ONSET_RESPONSE_PACKETS,
    V7_VERIFIER_RESPONSE_PACKETS,
    V25_ONSET_RESPONSE_PACKETS,
    ReviewedHardNegativeWindowSource,
    collate_domain_robust_v5,
    deterministic_limit,
    find_p_column,
    knet_factorized_indices,
    make_records,
)
from cwa_opened_packet_store import CWAOpenedPacketStoreSource  # noqa: E402
from domain_robust_v5_metrics import (  # noqa: E402
    collect_predictions_v5,
    evaluate_threshold_v5,
    subset_predictions_v5,
)
from model_domain_robust_v5 import DomainRobustCausalPickerV5  # noqa: E402
from model_domain_robust_v6 import DomainRobustCausalPickerV6  # noqa: E402
from model_domain_robust_v7 import DomainRobustCausalPickerV7  # noqa: E402
from model_domain_robust_v8 import DomainRobustCausalPickerV8  # noqa: E402
from model_domain_robust_v9 import DomainRobustCausalPickerV9  # noqa: E402
from model_domain_robust_v10 import DomainRobustCausalPickerV10  # noqa: E402
from model_domain_robust_v11 import DomainRobustCausalPickerV11  # noqa: E402
from model_packet_once_v1 import PacketOnceCausalPickerV1  # noqa: E402
from model_cspp_v25 import CSPPCausalPickerV25  # noqa: E402
from model_cspp_v26 import CSPPCausalPickerV26  # noqa: E402
from model_cspp_v27 import (  # noqa: E402
    CSPPCausalPickerV27,
    CSPPCausalPickerV27B,
    CSPPCausalPickerV27C,
)
from train_domain_robust_v4 import build_datasets as build_v4_source_records  # noqa: E402


PACKET_ONCE_ARCHITECTURES = {"packet_b1", "packet_b2", "packet_b3"}
TARGET_VERIFIER_ARCHITECTURES = {"v10", "v11"}
CSPP_V27_ARCHITECTURES = {"v27", "v27b", "v27c"}
OPERATIONAL_TRAINING_ARCHITECTURES = {
    *TARGET_VERIFIER_ARCHITECTURES,
    "v25",
    "v26",
    *CSPP_V27_ARCHITECTURES,
}
ARRIVAL_AGE_ARCHITECTURES = {
    "v7",
    "v8",
    "v9",
    *TARGET_VERIFIER_ARCHITECTURES,
    "v25",
    "v26",
    *CSPP_V27_ARCHITECTURES,
} | PACKET_ONCE_ARCHITECTURES
STRICT_CAUSAL_ARCHITECTURES = ARRIVAL_AGE_ARCHITECTURES
V10_CANDIDATE_THRESHOLD = 0.03


def resolved_data_seed(args: argparse.Namespace) -> int:
    data_seed = getattr(args, "data_seed", None)
    return int(args.seed if data_seed is None else data_seed)


def resolved_verifier_start_delay_packets(
    args: argparse.Namespace,
    uses_arrival_age: bool,
) -> int:
    configured = getattr(args, "verifier_start_delay_packets", None)
    delay = (
        0
        if getattr(args, "architecture", None)
        in {"v25", "v26", *CSPP_V27_ARCHITECTURES}
        else 1
        if uses_arrival_age
        else 0
    ) if configured is None else int(configured)
    response_packets = (
        V7_VERIFIER_RESPONSE_PACKETS
        if uses_arrival_age
        else VERIFIER_RESPONSE_PACKETS
    )
    if not 0 <= delay < response_packets:
        raise ValueError(
            "--verifier-start-delay-packets must satisfy "
            f"0 <= delay < {response_packets}; received {delay}"
        )
    return delay


def resolved_verifier_candidate_gate_threshold(
    args: argparse.Namespace,
) -> float:
    configured = getattr(
        args,
        "verifier_candidate_gate_threshold",
        None,
    )
    threshold = (
        0.1
        if getattr(args, "architecture", None) in CSPP_V27_ARCHITECTURES
        else V10_CANDIDATE_THRESHOLD
    ) if configured is None else float(configured)
    if not 0.0 < threshold < 1.0:
        raise ValueError(
            "--verifier-candidate-gate-threshold must be in (0, 1)"
        )
    return threshold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=ROOT / "models" / "checkpoints" / "packet_once_v1",
    )
    parser.add_argument(
        "--knet-dir",
        type=Path,
        default=ROOT / "data" / "knet_accel_causal_raw",
    )
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-chunks", type=int, default=160)
    parser.add_argument("--ig-limit", type=int, default=3000)
    parser.add_argument("--knet-limit", type=int, default=6000)
    parser.add_argument("--noise-limit", type=int, default=3000)
    parser.add_argument("--dev-limit", type=int, default=600)
    parser.add_argument(
        "--continuous-noise-hdf5",
        type=Path,
        default=ROOT / "data/domain_robust_v5_continuous_noise/windows.hdf5",
    )
    parser.add_argument(
        "--continuous-catalog-screen",
        type=Path,
        default=(
            ROOT
            / "data/domain_robust_v5_continuous_noise/catalog_screen_20260716"
            / "window_catalog_screen.csv"
        ),
        help="Catalog screen required before any continuous window receives a negative label.",
    )
    parser.add_argument("--continuous-train-limit", type=int, default=3000)
    parser.add_argument("--continuous-dev-limit", type=int, default=1000)
    parser.add_argument("--continuous-noise-class-fraction", type=float, default=0.60)
    parser.add_argument(
        "--hard-negative-review-manifest",
        type=Path,
        help="Approved training-only local packet masks from opened continuous dates.",
    )
    parser.add_argument(
        "--target-hard-negative-hdf5",
        type=Path,
        help="Date-split FDSN actual-trigger windows for the independent V10 verifier.",
    )
    parser.add_argument("--target-hard-negative-review-manifest", type=Path)
    parser.add_argument("--target-hard-negative-catalog-screen", type=Path)
    parser.add_argument("--target-background-hdf5", type=Path)
    parser.add_argument("--target-background-catalog-screen", type=Path)
    parser.add_argument(
        "--cwa-opened-packet-store",
        type=Path,
        help=(
            "Opened CWA 2020 event contexts packetized with the frozen "
            "operational causal preprocessing; training split only."
        ),
    )
    parser.add_argument(
        "--cwa-event-limit",
        type=int,
        default=0,
        help="Deterministic CWA training-event limit; 0 keeps the full pool.",
    )
    parser.add_argument("--instancegm-training-min-median-snr-db", type=float, default=3.0)
    parser.add_argument("--minimum-event-magnitude", type=float)
    parser.add_argument("--maximum-event-distance-km", type=float)
    parser.add_argument(
        "--use-continuous-noise",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument(
        "--data-seed",
        type=int,
        help="Fixed record/split/evaluation seed; defaults to --seed.",
    )
    parser.add_argument(
        "--selection-reference-npz",
        type=Path,
        help="Require every selected-index array to match this frozen reference.",
    )
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--onset-pos-weight", type=float, default=30.0)
    parser.add_argument("--event-loss-weight", type=float, default=0.10)
    parser.add_argument("--event-pos-weight", type=float, default=20.0)
    parser.add_argument(
        "--event-outside-loss-weight",
        "--outside-response-loss-weight",
        dest="event_outside_loss_weight",
        type=float,
        default=0.50,
    )
    parser.add_argument("--noise-topk-loss-weight", type=float, default=1.00)
    parser.add_argument("--ranking-loss-weight", type=float, default=0.50)
    parser.add_argument("--verifier-hard-negative-loss-weight", type=float, default=0.50)
    parser.add_argument("--verifier-ranking-loss-weight", type=float, default=0.50)
    parser.add_argument("--verifier-partial-auc-loss-weight", type=float, default=0.0)
    parser.add_argument("--verifier-partial-auc-negative-fraction", type=float, default=0.25)
    parser.add_argument("--verifier-partial-auc-margin", type=float, default=1.0)
    parser.add_argument(
        "--verifier-candidate-gated-loss",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--verifier-candidate-gate-threshold",
        type=float,
        help=(
            "Detached candidate probability threshold defining verifier "
            "supervision support; defaults to 0.1 for V27/V27B and 0.03 otherwise."
        ),
    )
    parser.add_argument(
        "--verifier-episode-loss",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--verifier-episode-bce-weight", type=float, default=1.0)
    parser.add_argument("--verifier-episode-tail-weight", type=float, default=1.0)
    parser.add_argument("--verifier-episode-tail-fraction", type=float, default=0.01)
    parser.add_argument("--verifier-episode-positive-packets", type=int, default=3)
    parser.add_argument("--offset-loss-weight", type=float, default=0.20)
    parser.add_argument(
        "--packet-localization-loss-weight",
        type=float,
        default=0.25,
        help="V26/V27 auxiliary weight for P location inside the current packet.",
    )
    parser.add_argument(
        "--packet-localization-sigma-samples",
        type=float,
        default=5.0,
        help="Gaussian target width for the V26 within-packet P location.",
    )
    parser.add_argument(
        "--distillation-teacher-checkpoint",
        type=Path,
        help=(
            "Optional frozen V25/V26 checkpoint. Only packet-once causal "
            "teachers are accepted."
        ),
    )
    parser.add_argument("--distillation-loss-weight", type=float, default=0.25)
    parser.add_argument("--distillation-temperature", type=float, default=2.0)
    parser.add_argument("--ranking-margin", type=float, default=1.0)
    parser.add_argument("--minimum-noise-specificity", type=float, default=0.99)
    parser.add_argument("--development-event-thresholds", default="0.0,0.1,0.2,0.4,0.6,0.8,0.9")
    parser.add_argument("--minimum-native-usable", type=float, default=0.75)
    parser.add_argument("--minimum-zero-db-usable", type=float, default=0.40)
    parser.add_argument("--minimum-ten-db-usable", type=float, default=0.80)
    parser.add_argument(
        "--minimum-source-native-usable",
        type=float,
        default=0.0,
        help=(
            "Optional minimum native usable rate required independently for "
            "InstanceGM and K-NET development events."
        ),
    )
    parser.add_argument("--candidate-history-packets", type=int, default=4)
    parser.add_argument("--warmup-packets", type=int, default=20)
    parser.add_argument("--max-arrival-age-packets", type=float, default=4.0)
    parser.add_argument(
        "--verifier-start-delay-packets",
        type=int,
        help=(
            "First packet relative to annotated P included in the verifier-positive "
            "label; defaults to 1 for arrival-age architectures and 0 otherwise."
        ),
    )
    parser.add_argument("--native-event-fraction", type=float, default=0.15)
    parser.add_argument(
        "--snr-training-profile",
        choices=("auto", "legacy", "v7_operational", "strong_motion_balanced"),
        default="auto",
    )
    parser.add_argument("--low-snr-event-weight", type=float, default=2.0)
    parser.add_argument("--low-snr-weight-upper-db", type=float, default=8.0)
    parser.add_argument("--packet-width", type=int, default=64)
    parser.add_argument("--embedding-dim", type=int, default=96)
    parser.add_argument("--sequence-width", type=int, default=96)
    parser.add_argument("--temporal-bins", type=int, default=5)
    parser.add_argument(
        "--retention-half-lives-seconds",
        default="0.5,1,2,4,8,16",
        help="Comma-separated fixed half-lives used by packet_b2.",
    )
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument(
        "--architecture",
        choices=(
            "v5",
            "v6",
            "v7",
            "v8",
            "v9",
            "v10",
            "v11",
            "v25",
            "v26",
            "v27",
            "v27b",
            "v27c",
            "packet_b1",
            "packet_b2",
            "packet_b3",
        ),
        default="packet_b2",
    )
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument(
        "--v10-train-scope",
        choices=("target_only", "joint"),
        default="target_only",
        help=(
            "Keep the V9 candidate/timing backbone frozen (target_only) or "
            "jointly adapt the complete V10 model (joint)."
        ),
    )
    parser.add_argument(
        "--v10-backbone-lr-multiplier",
        type=float,
        default=0.25,
        help="Learning-rate multiplier for non-target V10 parameters in joint mode.",
    )
    parser.add_argument(
        "--development-candidate-thresholds",
        help=(
            "Optional comma-separated candidate thresholds. Frozen V10/V11 runs "
            "continue to use the fixed 0.03 threshold."
        ),
    )
    parser.add_argument("--target-packet-width", type=int, default=32)
    parser.add_argument("--target-embedding-dim", type=int, default=64)
    parser.add_argument("--target-sequence-width", type=int, default=64)
    parser.add_argument("--fusion-hidden-dim", type=int, default=96)
    parser.add_argument("--fusion-dropout", type=float, default=0.10)
    parser.add_argument("--verifier-hidden-dim", type=int, default=64)
    parser.add_argument(
        "--verifier-feature-mode",
        choices=("none", "current", "current_delta"),
        default="current_delta",
        help=(
            "V27C verifier input ablation: compressed candidate evidence only, "
            "plus current packet features, or plus clipped one-packet deltas."
        ),
    )
    parser.add_argument(
        "--v27-temporal-mode",
        choices=("gru", "packet", "decoder"),
        default="gru",
        help=(
            "V27 temporal module: recurrent cross-packet state, packet-only "
            "mapping, or bounded causal decoder-only self-attention."
        ),
    )
    parser.add_argument("--v27-decoder-attention-heads", type=int, default=4)
    parser.add_argument("--v27-decoder-context-packets", type=int, default=32)
    parser.add_argument("--v27-decoder-ffn-multiplier", type=float, default=2.0)
    parser.add_argument(
        "--v27-disable-dense-phase-evidence",
        action="store_true",
        help=(
            "V27 ablation: remove the dense phase peak and mass from the "
            "candidate input. The localization branch may remain supervised "
            "to separate direct candidate-head evidence from auxiliary "
            "representation learning."
        ),
    )
    parser.add_argument(
        "--v27-disable-spectral-features",
        action="store_true",
        help="V27 ablation: replace fixed current-packet spectral features by zeros.",
    )
    parser.add_argument("--curriculum-epoch-offset", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--no-development-evaluation",
        action="store_true",
        default=False,
        help=(
            "Train fixed epochs without constructing development loaders, "
            "selecting checkpoints, or scanning operating thresholds. The final "
            "epoch is written as final.pt and no best.pt is created."
        ),
    )
    parser.add_argument("--device", choices=("auto", "mps", "cpu", "cuda"), default="auto")
    parser.add_argument("--strong-motion-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def select_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_selection_reference(selected: dict, reference_path: Path) -> dict:
    if not reference_path.is_file():
        raise FileNotFoundError(reference_path)
    with np.load(reference_path) as reference:
        expected_keys = set(reference.files)
        actual_keys = set(selected)
        if actual_keys != expected_keys:
            raise RuntimeError(
                "Selected-index keys differ from reference: "
                f"missing={sorted(expected_keys - actual_keys)}, "
                f"extra={sorted(actual_keys - expected_keys)}"
            )
        mismatches = [
            key
            for key in sorted(expected_keys)
            if not np.array_equal(np.asarray(selected[key]), reference[key])
        ]
        if mismatches:
            raise RuntimeError(
                f"Selected-index arrays differ from reference: {mismatches}"
            )
    return {
        "path": str(reference_path.resolve()),
        "sha256": file_sha256(reference_path),
        "arrays_verified": len(selected),
        "status": "exact_match",
    }


def weighted_onset_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    pos_weight: float,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    weights = 1.0 + (pos_weight - 1.0) * targets
    effective_mask = mask
    if sample_weights is not None:
        effective_mask = effective_mask * sample_weights.unsqueeze(1)
    return (loss * weights * effective_mask).sum() / effective_mask.sum().clamp_min(1.0)


def masked_bce(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)


def causal_candidate_support(
    logits_p: torch.Tensor,
    threshold: float,
    history_packets: int,
    warmup_packets: int,
) -> torch.Tensor:
    """Match the deployment P-history gate without propagating into the frozen P head."""
    if logits_p.ndim != 2:
        raise ValueError(f"Expected (B, T) P logits, got {tuple(logits_p.shape)}")
    if history_packets < 1 or warmup_packets < 1:
        raise ValueError("Candidate history and warmup must be positive")
    hits = torch.sigmoid(logits_p.detach()).ge(threshold).to(logits_p.dtype)
    first_allowed = min(max(warmup_packets - 1, 0), hits.shape[1])
    hits[:, :first_allowed] = 0.0
    padded = F.pad(hits.unsqueeze(1), (history_packets - 1, 0))
    support = F.max_pool1d(
        padded, kernel_size=history_packets, stride=1
    ).squeeze(1)
    support[:, :first_allowed] = 0.0
    return support


def candidate_gated_verifier_mask(
    mask: torch.Tensor,
    logits_p: torch.Tensor,
    labels_event: torch.Tensor,
    reviewed_hard_rows: torch.Tensor,
    threshold: float,
    history_packets: int,
    warmup_packets: int,
) -> torch.Tensor:
    if mask.shape != logits_p.shape or labels_event.shape != mask.shape:
        raise ValueError("Verifier mask, candidate logits, and labels must have equal shapes")
    if reviewed_hard_rows.shape != (mask.shape[0],):
        raise ValueError("Reviewed-hard row flags must have shape (batch,)")
    support = causal_candidate_support(
        logits_p,
        threshold=threshold,
        history_packets=history_packets,
        warmup_packets=warmup_packets,
    ).bool()
    supervised = support | labels_event.bool() | reviewed_hard_rows.unsqueeze(1)
    return mask * supervised.to(mask.dtype)


def event_outside_topk_loss(
    logits: torch.Tensor,
    target_chunks: torch.Tensor,
    is_event: torch.Tensor,
    mask: torch.Tensor,
    top_count: int = 4,
    response_packets: int = 3,
    response_start_delay_packets: int = 0,
) -> torch.Tensor:
    positions = torch.arange(logits.shape[1], device=logits.device).unsqueeze(0)
    targets = target_chunks.clamp_min(0).unsqueeze(1)
    event_outside = (positions < targets + response_start_delay_packets) | (
        positions >= targets + response_packets
    )
    negative = mask.bool() & is_event.unsqueeze(1) & event_outside
    masked = logits.masked_fill(~negative, -1e4)
    top = masked.topk(min(top_count, logits.shape[1]), dim=1).values
    valid = negative.any(dim=1)
    return F.softplus(top[valid]).mean() if valid.any() else logits.sum() * 0.0


def noise_topk_loss(
    logits: torch.Tensor,
    is_event: torch.Tensor,
    mask: torch.Tensor,
    top_count: int = 8,
) -> torch.Tensor:
    negative = mask.bool() & (~is_event).unsqueeze(1)
    masked = logits.masked_fill(~negative, -1e4)
    top = masked.topk(min(top_count, logits.shape[1]), dim=1).values
    valid = negative.any(dim=1)
    return F.softplus(top[valid]).mean() if valid.any() else logits.sum() * 0.0


def first_fire_ranking_loss(
    logits: torch.Tensor,
    target_chunks: torch.Tensor,
    is_event: torch.Tensor,
    mask: torch.Tensor,
    margin: float,
    response_packets: int = 3,
    response_start_delay_packets: int = 0,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    valid = is_event & target_chunks.ge(0)
    if not valid.any():
        return logits.sum() * 0.0
    positions = torch.arange(logits.shape[1], device=logits.device).unsqueeze(0)
    outside_mask = mask.bool() & (
        (
            positions
            < (
                target_chunks.clamp_min(0) + response_start_delay_packets
            ).unsqueeze(1)
        )
        | (positions >= (target_chunks.clamp_min(0) + response_packets).unsqueeze(1))
    )
    response_mask = (
        mask.bool()
        & (
            positions
            >= (
                target_chunks.clamp_min(0) + response_start_delay_packets
            ).unsqueeze(1)
        )
        & (positions < (target_chunks.clamp_min(0) + response_packets).unsqueeze(1))
    )
    outside_max = logits.masked_fill(~outside_mask, -1e4).amax(dim=1)
    response_max = logits.masked_fill(~response_mask, -1e4).amax(dim=1)
    selected = valid & outside_mask.any(dim=1) & response_mask.any(dim=1)
    if not selected.any():
        return logits.sum() * 0.0
    losses = F.softplus(outside_max[selected] - response_max[selected] + margin)
    if sample_weights is not None:
        weights = sample_weights[selected]
        return (losses * weights).sum() / weights.sum().clamp_min(1.0)
    return losses.mean()


def verifier_partial_auc_ranking_loss(
    logits: torch.Tensor,
    target_chunks: torch.Tensor,
    is_event: torch.Tensor,
    mask: torch.Tensor,
    negative_fraction: float,
    margin: float,
    response_packets: int,
    response_start_delay_packets: int,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Rank event responses above the hardest record-level noise tail."""
    if not 0.0 < negative_fraction <= 1.0:
        raise ValueError("Partial-AUC negative fraction must be in (0, 1]")
    positions = torch.arange(logits.shape[1], device=logits.device).unsqueeze(0)
    targets = target_chunks.clamp_min(0).unsqueeze(1)
    response = (
        mask.bool()
        & is_event.unsqueeze(1)
        & (positions >= targets + response_start_delay_packets)
        & (positions < targets + response_packets)
    )
    noise = mask.bool() & (~is_event).unsqueeze(1)
    event_valid = is_event & target_chunks.ge(0) & response.any(dim=1)
    noise_valid = (~is_event) & noise.any(dim=1)
    if not event_valid.any() or not noise_valid.any():
        return logits.sum() * 0.0
    event_scores = logits.masked_fill(~response, -1e4).amax(dim=1)[event_valid]
    noise_scores = logits.masked_fill(~noise, -1e4).amax(dim=1)[noise_valid]
    hard_count = max(1, math.ceil(negative_fraction * int(noise_scores.numel())))
    hard_noise = noise_scores.topk(hard_count).values
    losses = F.softplus(hard_noise.unsqueeze(1) - event_scores.unsqueeze(0) + margin)
    event_losses = losses.mean(dim=0)
    if sample_weights is not None:
        weights = sample_weights[event_valid]
        return (event_losses * weights).sum() / weights.sum().clamp_min(1.0)
    return event_losses.mean()


def verifier_candidate_episode_losses(
    logits_event: torch.Tensor,
    logits_p: torch.Tensor,
    target_chunks: torch.Tensor,
    is_event: torch.Tensor,
    mask: torch.Tensor,
    *,
    candidate_threshold: float,
    history_packets: int,
    warmup_packets: int,
    positive_packets: int,
    exclusion_packets: int,
    tail_fraction: float,
    margin: float,
    sample_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
    """Supervise record-level contiguous candidate episodes."""
    if positive_packets < 1 or exclusion_packets < positive_packets:
        raise ValueError("Invalid verifier episode response windows")
    if not 0.0 < tail_fraction <= 1.0:
        raise ValueError("Episode tail fraction must be in (0, 1]")
    support = causal_candidate_support(
        logits_p,
        threshold=candidate_threshold,
        history_packets=history_packets,
        warmup_packets=warmup_packets,
    ).bool() & mask.bool()
    positions = torch.arange(logits_event.shape[1], device=logits_event.device)
    positive_scores: list[torch.Tensor] = []
    positive_weights: list[torch.Tensor] = []
    negative_scores: list[torch.Tensor] = []

    def episode_scores(
        logits: torch.Tensor,
        selected: torch.Tensor,
    ) -> list[torch.Tensor]:
        indices = torch.nonzero(selected, as_tuple=False).flatten()
        if indices.numel() == 0:
            return []
        breaks = torch.nonzero(
            indices[1:] != indices[:-1] + 1,
            as_tuple=False,
        ).flatten() + 1
        starts = torch.cat((indices[:1], indices[breaks]))
        stops = torch.cat((indices[breaks - 1] + 1, indices[-1:] + 1))
        return [
            torch.logsumexp(logits[start:stop], dim=0)
            - math.log(int(stop - start))
            for start, stop in zip(starts, stops, strict=True)
        ]

    for row in range(logits_event.shape[0]):
        target = int(target_chunks[row].detach())
        if bool(is_event[row]) and target >= 0:
            positive_mask = (
                support[row]
                & (positions >= target)
                & (positions < target + positive_packets)
            )
            row_positive = episode_scores(logits_event[row], positive_mask)
            positive_scores.extend(row_positive)
            if row_positive:
                weight = (
                    sample_weights[row]
                    if sample_weights is not None
                    else logits_event.new_tensor(1.0)
                )
                positive_weights.extend(weight for _ in row_positive)
            negative_mask = support[row] & (
                (positions < target)
                | (positions >= target + exclusion_packets)
            )
        else:
            negative_mask = support[row]
        negative_scores.extend(
            episode_scores(logits_event[row], negative_mask)
        )

    zero = logits_event.sum() * 0.0
    if positive_scores:
        positives = torch.stack(positive_scores)
        positive_loss_values = F.softplus(-positives)
        weights = torch.stack(positive_weights)
        positive_bce = (
            positive_loss_values * weights
        ).sum() / weights.sum().clamp_min(1.0)
    else:
        positives = logits_event.new_empty(0)
        positive_bce = zero
    if negative_scores:
        negatives = torch.stack(negative_scores)
        negative_bce = F.softplus(negatives).mean()
    else:
        negatives = logits_event.new_empty(0)
        negative_bce = zero
    episode_bce = 0.5 * (positive_bce + negative_bce)

    if positives.numel() and negatives.numel():
        hard_count = max(
            1,
            math.ceil(tail_fraction * int(negatives.numel())),
        )
        hard_negative = negatives.topk(hard_count).values
        tail_ranking = F.softplus(
            hard_negative.unsqueeze(1)
            - positives.unsqueeze(0)
            + margin
        ).mean()
    else:
        tail_ranking = zero
    return episode_bce, tail_ranking, {
        "positive_episodes": len(positive_scores),
        "negative_episodes": len(negative_scores),
    }


def controlled_snr_cohort(predictions: dict, snr_db: float) -> tuple[dict, dict[str, int]]:
    events = predictions["events"]
    degradable_events = events & (predictions["achieved_snr_db"] >= snr_db - 0.25)
    cohort = (~events) | degradable_events
    counts = {
        "event_degradable_n": int(degradable_events.sum()),
        "event_below_target_before_mixing_n": int((events & ~degradable_events).sum()),
    }
    return subset_predictions_v5(predictions, cohort), counts


def parse_float_list(text: str) -> list[float]:
    return [float(value.strip()) for value in text.split(",") if value.strip()]


def is_v10_joint_training(args: argparse.Namespace) -> bool:
    return args.architecture == "v10" and args.v10_train_scope == "joint"


def select_joint_development_threshold(
    native_predictions: dict,
    snr_predictions: dict[float, dict],
    minimum_noise_specificity: float = 0.99,
    event_thresholds: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8),
    candidate_history_packets: int = 1,
    warmup_packets: int = 0,
    timing_mode: str = "packet_offset",
    max_arrival_age_packets: float = 4.0,
    minimum_native_usable: float = 0.75,
    minimum_zero_db_usable: float = 0.40,
    minimum_ten_db_usable: float = 0.80,
    minimum_source_native_usable: float = 0.0,
    candidate_thresholds: tuple[float, ...] | None = None,
) -> tuple[float, float, dict, dict, list[dict]]:
    prepared = {
        snr_db: controlled_snr_cohort(predictions, snr_db)
        for snr_db, predictions in snr_predictions.items()
    }
    sweep = []
    thresholds = (
        np.arange(0.10, 0.976, 0.025)
        if candidate_thresholds is None
        else candidate_thresholds
    )
    for threshold in thresholds:
        for event_threshold in event_thresholds:
            native = evaluate_threshold_v5(
                native_predictions,
                float(threshold),
                event_threshold=float(event_threshold),
                candidate_history_packets=candidate_history_packets,
                warmup_packets=warmup_packets,
                timing_mode=timing_mode,
                max_arrival_age_packets=max_arrival_age_packets,
            )
            low_snr = {}
            for snr_db, (predictions, counts) in prepared.items():
                metrics = evaluate_threshold_v5(
                    predictions,
                    float(threshold),
                    event_threshold=float(event_threshold),
                    candidate_history_packets=candidate_history_packets,
                    warmup_packets=warmup_packets,
                    timing_mode=timing_mode,
                    max_arrival_age_packets=max_arrival_age_packets,
                )
                metrics.update(counts)
                low_snr[f"{snr_db:+.0f}_db"] = metrics
            source_native = {}
            for source_name in ("instancegm", "knet"):
                source_mask = (
                    native_predictions["events"]
                    & (native_predictions["sources"] == source_name)
                )
                if source_mask.any():
                    source_native[source_name] = evaluate_threshold_v5(
                        subset_predictions_v5(native_predictions, source_mask),
                        float(threshold),
                        event_threshold=float(event_threshold),
                        candidate_history_packets=candidate_history_packets,
                        warmup_packets=warmup_packets,
                        timing_mode=timing_mode,
                        max_arrival_age_packets=max_arrival_age_packets,
                    )
            terms = [
                native["usable_rate"],
                native["noise_specificity"],
                low_snr["+0_db"]["usable_rate"],
                low_snr["+10_db"]["usable_rate"],
            ]
            if minimum_source_native_usable > 0.0:
                terms.extend(
                    metrics["usable_rate"]
                    for metrics in source_native.values()
                )
            score = float(
                np.prod(np.clip(terms, 1e-6, 1.0))
                ** (1.0 / max(len(terms), 1))
            )
            availability_ok = (
                np.isfinite(native["median_availability_delay_s"])
                and native["median_availability_delay_s"] <= 0.75
                and np.isfinite(native["p90_availability_delay_s"])
                and native["p90_availability_delay_s"] <= 1.5
            )
            performance_checks = [
                native["usable_rate"] >= minimum_native_usable,
                low_snr["+0_db"]["usable_rate"] >= minimum_zero_db_usable,
                low_snr["+10_db"]["usable_rate"] >= minimum_ten_db_usable,
            ]
            gate_progress_terms = [
                native["usable_rate"] / max(minimum_native_usable, 1e-12),
                low_snr["+0_db"]["usable_rate"] / max(minimum_zero_db_usable, 1e-12),
                low_snr["+10_db"]["usable_rate"] / max(minimum_ten_db_usable, 1e-12),
                native["noise_specificity"] / max(minimum_noise_specificity, 1e-12),
            ]
            if minimum_source_native_usable > 0.0:
                performance_checks.extend(
                    metrics["usable_rate"] >= minimum_source_native_usable
                    for metrics in source_native.values()
                )
                gate_progress_terms.extend(
                    metrics["usable_rate"]
                    / max(minimum_source_native_usable, 1e-12)
                    for metrics in source_native.values()
                )
            gate_progress = float(
                np.prod(np.clip(gate_progress_terms, 1e-6, 1.0))
                ** (1.0 / max(len(gate_progress_terms), 1))
            )
            sweep.append(
                {
                    "threshold": float(threshold),
                    "event_threshold": float(event_threshold),
                    "selection_score": score,
                    "noise_specificity_eligible": (
                        bool(native["noise_specificity"] >= minimum_noise_specificity)
                    ),
                    "availability_eligible": bool(availability_ok),
                    "performance_eligible": bool(all(performance_checks)),
                    "development_gate_count": int(sum(performance_checks))
                    + int(native["noise_specificity"] >= minimum_noise_specificity)
                    + int(availability_ok),
                    "development_gate_progress": gate_progress,
                    "selection_eligible": (
                        bool(native["noise_specificity"] >= minimum_noise_specificity)
                        and bool(availability_ok)
                        and bool(all(performance_checks))
                    ),
                    "native": native,
                    "source_native": source_native,
                    "low_snr": low_snr,
                }
            )
    eligible = [row for row in sweep if row["selection_eligible"]]
    operationally_valid = [
        row
        for row in sweep
        if row["noise_specificity_eligible"] and row["availability_eligible"]
    ]
    specificity_eligible = [row for row in sweep if row["noise_specificity_eligible"]]
    candidates = eligible or operationally_valid or specificity_eligible or sweep
    best = max(
        candidates,
        key=lambda row: (
            row["native"]["noise_specificity"],
            row["selection_score"],
            row["threshold"],
        ) if not (eligible or operationally_valid or specificity_eligible) else (
            row["development_gate_count"],
            row["development_gate_progress"],
            row["selection_score"],
            row["availability_eligible"],
            row["native"]["noise_specificity"],
            row["threshold"],
        ),
    )
    return (
        best["threshold"],
        best["event_threshold"],
        best["native"],
        best["low_snr"],
        sweep,
    )


def offset_regression_loss(
    offset_logits: torch.Tensor,
    target_chunks: torch.Tensor,
    target_offsets: torch.Tensor,
    is_event: torch.Tensor,
) -> torch.Tensor:
    valid = is_event & target_chunks.ge(0) & torch.isfinite(target_offsets)
    if not valid.any():
        return offset_logits.sum() * 0.0
    rows = torch.arange(offset_logits.shape[0], device=offset_logits.device)
    predictions = torch.sigmoid(offset_logits[rows, target_chunks.clamp_min(0)])
    return F.smooth_l1_loss(predictions[valid], target_offsets[valid], beta=0.1)


def arrival_age_regression_loss(
    delay_logits: torch.Tensor,
    target_chunks: torch.Tensor,
    target_offsets: torch.Tensor,
    is_event: torch.Tensor,
    mask: torch.Tensor,
    response_packets: int,
    response_start_delay_packets: int,
    max_arrival_age_packets: float,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Regress elapsed packets since P at every causal confirmation packet."""
    positions = torch.arange(delay_logits.shape[1], device=delay_logits.device).unsqueeze(0)
    safe_target_offsets = torch.nan_to_num(target_offsets, nan=0.0)
    target_position = target_chunks.to(delay_logits.dtype) + safe_target_offsets
    valid = (
        mask.bool()
        & is_event.unsqueeze(1)
        & target_chunks.ge(0).unsqueeze(1)
        & torch.isfinite(target_offsets).unsqueeze(1)
        & (
            positions
            >= (
                target_chunks.clamp_min(0) + response_start_delay_packets
            ).unsqueeze(1)
        )
        & (positions < (target_chunks.clamp_min(0) + response_packets).unsqueeze(1))
    )
    if not valid.any():
        return delay_logits.sum() * 0.0
    target_age = (positions.to(delay_logits.dtype) + 1.0) - target_position.unsqueeze(1)
    target_fraction = torch.clamp(target_age / max_arrival_age_packets, 0.0, 1.0)
    losses = F.smooth_l1_loss(
        torch.sigmoid(delay_logits),
        target_fraction,
        beta=0.05,
        reduction="none",
    )
    weights = valid.to(losses.dtype)
    if sample_weights is not None:
        weights = weights * sample_weights.unsqueeze(1)
    return (losses * weights).sum() / weights.sum().clamp_min(1.0)


def packet_localization_loss(
    phase_logits: torch.Tensor,
    target_chunks: torch.Tensor,
    target_offsets: torch.Tensor,
    is_event: torch.Tensor,
    sigma_samples: float,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Supervise the P sample inside its fully received 0.5-second packet."""
    if phase_logits.ndim != 3:
        raise ValueError("phase_logits must have shape (B, T, samples)")
    if sigma_samples <= 0.0:
        raise ValueError("sigma_samples must be positive")
    valid = (
        is_event
        & target_chunks.ge(0)
        & target_chunks.lt(phase_logits.shape[1])
        & torch.isfinite(target_offsets)
    )
    if not valid.any():
        return phase_logits.sum() * 0.0

    rows = torch.arange(phase_logits.shape[0], device=phase_logits.device)
    selected = phase_logits[rows[valid], target_chunks[valid]]
    sample_positions = torch.arange(
        phase_logits.shape[-1],
        device=phase_logits.device,
        dtype=phase_logits.dtype,
    ).unsqueeze(0)
    centers = (
        target_offsets[valid].to(phase_logits.dtype)
        * float(phase_logits.shape[-1])
    ).clamp(0.0, float(phase_logits.shape[-1] - 1)).unsqueeze(1)
    targets = torch.exp(
        -0.5 * ((sample_positions - centers) / float(sigma_samples)).square()
    )
    targets = targets / targets.sum(dim=1, keepdim=True).clamp_min(1e-8)
    losses = -(targets * F.log_softmax(selected, dim=1)).sum(dim=1)
    if sample_weights is not None:
        weights = sample_weights[valid].to(losses.dtype)
        return (losses * weights).sum() / weights.sum().clamp_min(1.0)
    return losses.mean()


def dense_phase_detection_loss(
    phase_logits: torch.Tensor,
    target_chunks: torch.Tensor,
    target_offsets: torch.Tensor,
    is_event: torch.Tensor,
    mask: torch.Tensor,
    sigma_samples: float,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Supervise dense P shape and all valid non-P samples."""
    if phase_logits.ndim != 3:
        raise ValueError("phase_logits must have shape (B, T, samples)")
    if tuple(mask.shape) != tuple(phase_logits.shape[:2]):
        raise ValueError("mask must match the packet dimensions")
    if sigma_samples <= 0.0:
        raise ValueError("sigma_samples must be positive")

    targets = torch.zeros_like(phase_logits)
    positive_records = (
        is_event
        & target_chunks.ge(0)
        & target_chunks.lt(phase_logits.shape[1])
        & torch.isfinite(target_offsets)
    )
    if positive_records.any():
        rows = torch.arange(
            phase_logits.shape[0],
            device=phase_logits.device,
        )
        sample_positions = torch.arange(
            phase_logits.shape[-1],
            device=phase_logits.device,
            dtype=phase_logits.dtype,
        ).unsqueeze(0)
        centers = (
            target_offsets[positive_records].to(phase_logits.dtype)
            * float(phase_logits.shape[-1])
        ).clamp(
            0.0,
            float(phase_logits.shape[-1] - 1),
        ).unsqueeze(1)
        targets[
            rows[positive_records],
            target_chunks[positive_records],
        ] = torch.exp(
            -0.5
            * (
                (sample_positions - centers) / float(sigma_samples)
            ).square()
        )

    valid = mask.bool().unsqueeze(-1).expand_as(phase_logits)
    if not valid.any():
        return phase_logits.sum() * 0.0
    losses = F.binary_cross_entropy_with_logits(
        phase_logits,
        targets,
        reduction="none",
    )
    record_weights = torch.ones_like(losses)
    if sample_weights is not None:
        record_weights = record_weights * sample_weights[:, None, None]
    valid_weights = valid.to(losses.dtype) * record_weights
    positive_weights = targets * valid_weights
    negative_weights = (1.0 - targets) * valid_weights
    positive_mass = positive_weights.sum()
    negative_loss = (
        losses * negative_weights
    ).sum() / negative_weights.sum().clamp_min(1.0)
    if not bool(positive_mass > 0):
        return negative_loss
    positive_loss = (
        losses * positive_weights
    ).sum() / positive_mass.clamp_min(1.0)
    return 0.5 * (positive_loss + negative_loss)


def masked_bernoulli_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Distill packet-end probabilities from a frozen causal teacher."""
    if student_logits.shape != teacher_logits.shape or mask.shape != student_logits.shape:
        raise ValueError("Student, teacher, and mask shapes must match")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    valid = mask.bool()
    if not valid.any():
        return student_logits.sum() * 0.0
    soft_targets = torch.sigmoid(teacher_logits.detach() / temperature)
    losses = F.binary_cross_entropy_with_logits(
        student_logits / temperature,
        soft_targets,
        reduction="none",
    )
    return losses[valid].mean() * temperature**2


def balanced_sampler(
    records,
    seed: int,
    continuous_noise_class_fraction: float = 0.60,
) -> tuple[WeightedRandomSampler, dict[str, int]]:
    groups = []
    for record in records:
        kind = "event" if record.is_event else "noise"
        groups.append(f"{kind}:{record.source}")
    counts = Counter(groups)
    event_groups = sorted(group for group in counts if group.startswith("event:"))
    noise_groups = sorted(group for group in counts if group.startswith("noise:"))
    masses = {}
    for group in event_groups:
        masses[group] = (2.0 / 3.0) / max(len(event_groups), 1)
    continuous_groups = [group for group in noise_groups if "continuous" in group]
    other_noise_groups = [group for group in noise_groups if group not in continuous_groups]
    noise_mass = 1.0 / 3.0
    continuous_fraction = float(np.clip(continuous_noise_class_fraction, 0.0, 1.0))
    if continuous_groups and other_noise_groups:
        for group in continuous_groups:
            masses[group] = noise_mass * continuous_fraction / len(continuous_groups)
        for group in other_noise_groups:
            masses[group] = noise_mass * (1.0 - continuous_fraction) / len(other_noise_groups)
    else:
        for group in noise_groups:
            masses[group] = noise_mass / max(len(noise_groups), 1)
    weights = torch.tensor([masses[group] / counts[group] for group in groups], dtype=torch.double)
    generator = torch.Generator().manual_seed(seed)
    sampler = WeightedRandomSampler(weights, num_samples=len(records), replacement=True, generator=generator)
    return sampler, dict(counts)


def v10_balanced_sampler(records, seed: int) -> tuple[WeightedRandomSampler, dict[str, int]]:
    """Freeze V10 family mass at event/target/opened/other = .50/.35/.10/.05."""
    groups = []
    families = []
    for record in records:
        group = f"{'event' if record.is_event else 'noise'}:{record.source}"
        groups.append(group)
        if record.is_event:
            families.append("event")
        elif "target_fdsn" in record.source:
            families.append("target_actual_trigger")
        elif "continuous" in record.source:
            families.append("opened_2024_continuous")
        else:
            families.append("other_noise")
    group_counts = Counter(groups)
    family_groups: dict[str, list[str]] = {}
    for group, family in zip(groups, families, strict=True):
        family_groups.setdefault(family, []).append(group)
    family_groups = {
        family: sorted(set(values)) for family, values in family_groups.items()
    }
    desired = {
        "event": 0.50,
        "target_actual_trigger": 0.35,
        "opened_2024_continuous": 0.10,
        "other_noise": 0.05,
    }
    missing = [family for family in desired if family not in family_groups]
    if missing:
        raise ValueError(f"V10 sampler is missing required families: {missing}")
    masses = {
        group: desired[family] / len(family_groups[family])
        for family in desired
        for group in family_groups[family]
    }
    weights = torch.tensor(
        [masses[group] / group_counts[group] for group in groups], dtype=torch.double
    )
    generator = torch.Generator().manual_seed(seed)
    sampler = WeightedRandomSampler(
        weights, num_samples=len(records), replacement=True, generator=generator
    )
    counts = dict(group_counts)
    counts["__family_mass_basis_points__"] = {
        family: int(round(mass * 10_000)) for family, mass in desired.items()
    }
    return sampler, counts


def v25_balanced_sampler(records, seed: int) -> tuple[WeightedRandomSampler, dict[str, int]]:
    """Balance V25 for cross-network event recall without dropping hard negatives."""
    groups = [
        f"{'event' if record.is_event else 'noise'}:{record.source}"
        for record in records
    ]
    group_counts = Counter(groups)
    required_event_masses = {
        "event:instancegm": 0.40,
        "event:knet": 0.15,
        "event:cwa2020_opened_train_events": 0.10,
    }
    missing = sorted(set(required_event_masses) - set(group_counts))
    if missing:
        raise ValueError(f"V25 sampler is missing required event groups: {missing}")

    target_trigger_groups = sorted(
        group
        for group in group_counts
        if group.startswith("noise:") and "target_fdsn" in group
    )
    opened_continuous_groups = sorted(
        group
        for group in group_counts
        if group.startswith("noise:")
        and "continuous" in group
        and "target_fdsn" not in group
    )
    other_noise_groups = sorted(
        group
        for group in group_counts
        if group.startswith("noise:")
        and group not in target_trigger_groups
        and group not in opened_continuous_groups
    )
    noise_families = {
        "target_actual_trigger": (target_trigger_groups, 0.20),
        "opened_2024_continuous": (opened_continuous_groups, 0.10),
        "other_noise": (other_noise_groups, 0.05),
    }
    missing_noise = [
        family for family, (family_groups, _) in noise_families.items()
        if not family_groups
    ]
    if missing_noise:
        raise ValueError(f"V25 sampler is missing noise families: {missing_noise}")

    masses = dict(required_event_masses)
    for family_groups, family_mass in noise_families.values():
        for group in family_groups:
            masses[group] = family_mass / len(family_groups)
    weights = torch.tensor(
        [masses[group] / group_counts[group] for group in groups],
        dtype=torch.double,
    )
    generator = torch.Generator().manual_seed(seed)
    sampler = WeightedRandomSampler(
        weights,
        num_samples=len(records),
        replacement=True,
        generator=generator,
    )
    counts = dict(group_counts)
    counts["__group_mass_basis_points__"] = {
        group: int(round(mass * 10_000))
        for group, mass in masses.items()
    }
    return sampler, counts


def event_scope_filter(
    records,
    sources: dict[str, object],
    minimum_magnitude: float | None,
    maximum_distance_km: float | None,
) -> tuple[list, dict]:
    """Filter only event records by declared metadata; retain every noise record."""
    if minimum_magnitude is None and maximum_distance_km is None:
        return list(records), {"enabled": False}
    kept = []
    counts: dict[str, dict[str, int]] = {}
    magnitude_columns = ("source_magnitude", "preferred_source_magnitude", "magnitude")
    distance_columns = (
        "source_distance_km",
        "path_ep_distance_km",
        "path_hyp_distance_km",
        "epicentral_distance_km",
    )
    for record in records:
        if not record.is_event:
            kept.append(record)
            continue
        source_counts = counts.setdefault(
            record.source,
            {"total": 0, "metadata_known": 0, "eligible": 0},
        )
        source_counts["total"] += 1
        metadata = sources[record.source].metadata.iloc[int(record.index)]
        magnitude = math.nan
        distance = math.nan
        for column in magnitude_columns:
            if column in metadata.index:
                value = pd.to_numeric(metadata[column], errors="coerce")
                if pd.notna(value):
                    magnitude = float(value)
                    break
        for column in distance_columns:
            if column in metadata.index:
                value = pd.to_numeric(metadata[column], errors="coerce")
                if pd.notna(value):
                    distance = float(value)
                    break
        magnitude_known = minimum_magnitude is None or math.isfinite(magnitude)
        distance_known = maximum_distance_km is None or math.isfinite(distance)
        if magnitude_known and distance_known:
            source_counts["metadata_known"] += 1
        eligible = (
            magnitude_known
            and distance_known
            and (minimum_magnitude is None or magnitude >= minimum_magnitude)
            and (
                maximum_distance_km is None
                or distance <= maximum_distance_km
            )
        )
        if eligible:
            kept.append(record)
            source_counts["eligible"] += 1
    return kept, {
        "enabled": True,
        "minimum_magnitude": minimum_magnitude,
        "maximum_distance_km": maximum_distance_km,
        "role": "training_and_opened_development_population_only_not_model_input",
        "by_source": counts,
    }


def limit_source_records(
    records,
    source: str,
    is_event: bool,
    limit: int,
    seed: int,
) -> tuple[list, np.ndarray]:
    """Limit one source/label population after all scientific filters."""
    matching_indices = np.asarray(
        [
            record.index
            for record in records
            if record.source == source and record.is_event == is_event
        ],
        dtype=np.int64,
    )
    selected_indices = deterministic_limit(matching_indices, limit, seed)
    selected_set = set(selected_indices.tolist())
    limited = [
        record
        for record in records
        if not (record.source == source and record.is_event == is_event)
        or record.index in selected_set
    ]
    return limited, selected_indices


def valid_p_indices(metadata: pd.DataFrame, indices: np.ndarray, max_samples: int) -> np.ndarray:
    p_column = find_p_column(metadata)
    values = pd.to_numeric(
        metadata.iloc[np.asarray(indices, dtype=np.int64)][p_column],
        errors="coerce",
    ).to_numpy(dtype=float)
    valid = np.isfinite(values) & (values >= 0) & (values < max_samples)
    return np.asarray(indices, dtype=np.int64)[valid]


def build_datasets(args: argparse.Namespace):
    args.data_seed = resolved_data_seed(args)
    uses_arrival_age = args.architecture in ARRIVAL_AGE_ARCHITECTURES
    knet_manifest = None
    if uses_arrival_age:
        manifest_path = args.knet_dir / "manifest.json"
        if not manifest_path.is_file():
            raise RuntimeError(
                f"{args.architecture} refuses K-NET data without a causal raw-input manifest: "
                f"{manifest_path}"
            )
        knet_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if knet_manifest.get("protocol") != "knet_raw_acceleration_strict_causal_input_v1":
            raise RuntimeError(f"Unexpected K-NET causal manifest: {manifest_path}")
    onset_response_packets = (
        V25_ONSET_RESPONSE_PACKETS
        if args.architecture in {"v25", "v26", *CSPP_V27_ARCHITECTURES}
        else V7_ONSET_RESPONSE_PACKETS
        if uses_arrival_age
        else ONSET_RESPONSE_PACKETS
    )
    verifier_response_packets = (
        V7_VERIFIER_RESPONSE_PACKETS if uses_arrival_age else VERIFIER_RESPONSE_PACKETS
    )
    verifier_start_delay_packets = resolved_verifier_start_delay_packets(
        args,
        uses_arrival_age,
    )
    snr_profile = (
        "v7_operational" if uses_arrival_age else "legacy"
    ) if args.snr_training_profile == "auto" else args.snr_training_profile
    data_args = argparse.Namespace(**vars(args))
    data_args.seed = resolved_data_seed(args)
    population_first_selection = args.architecture in {
        "v25",
        "v26",
        *CSPP_V27_ARCHITECTURES,
    }
    if population_first_selection:
        # V4 historically limits records before applying the target-population
        # magnitude/distance filter. Load the complete event pools here; V25
        # applies deterministic limits only after every scientific filter.
        data_args.ig_limit = 0
        data_args.knet_limit = 0
        data_args.dev_limit = np.iinfo(np.int32).max
    base_train, base_dev, protocol, selected = build_v4_source_records(data_args)
    train_sources = dict(base_train.sources)
    dev_sources = dict(base_dev.sources)
    train_records = list(base_train.records)
    dev_records = list(base_dev.records)
    train_noise_sources = list(base_train.noise_source_names)
    dev_noise_sources = list(base_dev.noise_source_names)
    if population_first_selection:
        knet_source = dev_sources["knet"]
        max_samples = args.max_chunks * SAMPLES_PER_CHUNK
        knet_dev_joint = valid_p_indices(
            knet_source.metadata,
            knet_factorized_indices(
                knet_source.metadata,
                "dev_joint",
                args.data_seed,
            ),
            max_samples,
        )
        knet_p_column = find_p_column(knet_source.metadata)
        knet_p_values = pd.to_numeric(
            knet_source.metadata[knet_p_column],
            errors="coerce",
        ).to_numpy(dtype=float)
        knet_dev_joint_noise = knet_dev_joint[
            knet_p_values[knet_dev_joint] >= 2 * TARGET_SR
        ]
        dev_records = [
            record
            for record in dev_records
            if not (
                (record.source == "knet" and record.is_event)
                or (record.source == "knet_prep_noise" and not record.is_event)
            )
        ]
        dev_records.extend(make_records("knet", knet_dev_joint, True))
        dev_records.extend(
            make_records("knet_prep_noise", knet_dev_joint_noise, False)
        )
        selected = dict(selected)
        selected.pop("knet_dev_event", None)
        selected.pop("knet_dev_station", None)
        selected.pop("knet_noise_dev", None)
        selected["knet_dev_joint_population"] = knet_dev_joint
        selected["knet_dev_joint_noise_population"] = knet_dev_joint_noise
        protocol = dict(protocol)
        protocol["v25_population_first_selection"] = {
            "enabled": True,
            "order": (
                "construct complete provider split and K-NET joint event/station "
                "partition; apply target-population metadata filters; apply "
                "deterministic record limits last"
            ),
            "knet_development_partition": (
                "provider event dev AND independently hashed station dev"
            ),
            "pre_limit_knet_joint_development_records": int(len(knet_dev_joint)),
        }
    instance_source = train_sources.get("instancegm")
    if instance_source is not None and not population_first_selection:
        snr_columns = ["trace_Z_snr_db", "trace_N_snr_db", "trace_E_snr_db"]
        metadata = instance_source.metadata
        if all(column in metadata.columns for column in snr_columns):
            component_snr = metadata[snr_columns].apply(
                lambda column: pd.to_numeric(column, errors="coerce")
            )
            median_snr = component_snr.median(axis=1, skipna=True).to_numpy(dtype=float)
            eligible_instance = set(
                np.flatnonzero(
                    np.isfinite(median_snr)
                    & (median_snr >= args.instancegm_training_min_median_snr_db)
                ).tolist()
            )
            before = sum(
                record.source == "instancegm" and record.is_event
                for record in train_records
            )
            train_records = [
                record
                for record in train_records
                if not (
                    record.source == "instancegm"
                    and record.is_event
                    and record.index not in eligible_instance
                )
            ]
            after = sum(
                record.source == "instancegm" and record.is_event
                for record in train_records
            )
            selected = dict(selected)
            selected["ig_train_quality_filtered"] = np.asarray(
                [
                    record.index
                    for record in train_records
                    if record.source == "instancegm" and record.is_event
                ],
                dtype=np.int64,
            )
            protocol = dict(protocol)
            protocol["instancegm_training_quality_filter"] = {
                "official_component_snr_median_min_db": (
                    args.instancegm_training_min_median_snr_db
                ),
                "before": before,
                "after": after,
                "scope": "training events only; no development or final-test exclusion",
            }
    elif instance_source is not None:
        protocol = dict(protocol)
        protocol["instancegm_training_quality_filter"] = {
            "enabled": False,
            "policy": (
                "retain every labeled acceleration event in the target population; "
                "official component SNR is used for weighting and stratified "
                "reporting, never record deletion"
            ),
        }
    if args.cwa_opened_packet_store is not None:
        if args.architecture not in OPERATIONAL_TRAINING_ARCHITECTURES:
            raise ValueError(
                "Opened CWA packet contexts require a target-verifier "
                "architecture"
            )
        cwa_source = CWAOpenedPacketStoreSource(
            "cwa2020_opened_train_events",
            args.cwa_opened_packet_store,
            split="train",
        )
        cwa_event_population = cwa_source.event_indices.copy()
        cwa_event_indices = deterministic_limit(
            cwa_event_population,
            args.cwa_event_limit,
            args.data_seed + 31,
        )
        train_sources[cwa_source.name] = cwa_source
        train_records.extend(
            make_records(
                cwa_source.name,
                cwa_event_indices,
                True,
            )
        )
        selected = dict(selected)
        selected["cwa2020_opened_train"] = cwa_event_indices
        protocol = dict(protocol)
        protocol["cwa2020_opened_event_context_training"] = {
            "store": str(args.cwa_opened_packet_store.resolve()),
            "store_sha256": file_sha256(args.cwa_opened_packet_store),
            "metadata": str(cwa_source.metadata_path.resolve()),
            "metadata_sha256": file_sha256(cwa_source.metadata_path),
            "population_records": int(len(cwa_event_population)),
            "training_records": int(len(cwa_event_indices)),
            "deterministic_limit": int(args.cwa_event_limit),
            "excluded_midstream_quality_reset_records": int(
                len(cwa_source.excluded_reset_indices)
            ),
            "excluded_midstream_quality_reset_store_indices": (
                cwa_source.excluded_reset_indices.tolist()
            ),
            "events": int(
                cwa_source.metadata.loc[
                    cwa_source.metadata["store_index"].isin(
                        cwa_source.event_indices
                    ),
                    "event_id",
                ].nunique()
            ),
            "split": (
                "event-disjoint prospective split stored in the packet-store "
                "metadata; development rows are not added to generic training "
                "or controlled-SNR loaders"
            ),
            "preprocessing": (
                "frozen 200-to-100 Hz causal FIR, PacketQualityGateV5, "
                "packet EWMA normalization, and candidate-enabled mask"
            ),
            "supervision": (
                "native long pre-P context with P+0 through P+3 verifier "
                "labels; no synthetic relocation or waveform mixing"
            ),
            "quality_reset_boundary": (
                "Rows containing a quality-invalid packet inside the stored "
                "context are explicitly excluded because the generic batched "
                "trainer cannot reset model state within one sequence."
            ),
            "evidence_boundary": (
                "CWA 2020 is already-opened development data and cannot "
                "become independent confirmation evidence"
            ),
        }
    train_records, train_scope = event_scope_filter(
        train_records,
        train_sources,
        args.minimum_event_magnitude,
        args.maximum_event_distance_km,
    )
    dev_records, dev_scope = event_scope_filter(
        dev_records,
        dev_sources,
        args.minimum_event_magnitude,
        args.maximum_event_distance_km,
    )
    if population_first_selection:
        population_counts = {
            split: {
                source: sum(
                    record.is_event and record.source == source
                    for record in records
                )
                for source in ("instancegm", "knet")
            }
            for split, records in (
                ("train", train_records),
                ("development", dev_records),
            )
        }
        train_records, ig_train = limit_source_records(
            train_records,
            "instancegm",
            True,
            args.ig_limit,
            args.data_seed + 11,
        )
        train_records, knet_train = limit_source_records(
            train_records,
            "knet",
            True,
            args.knet_limit,
            args.data_seed + 12,
        )
        dev_records, ig_dev = limit_source_records(
            dev_records,
            "instancegm",
            True,
            args.dev_limit,
            args.data_seed + 21,
        )
        dev_records, knet_dev_joint = limit_source_records(
            dev_records,
            "knet",
            True,
            args.dev_limit,
            args.data_seed + 22,
        )
        knet_dev_noise_limit = (
            0 if args.dev_limit <= 0 else max(args.dev_limit * 2 // 3, 1)
        )
        ig_dev_noise_limit = (
            0 if args.dev_limit <= 0 else max(args.dev_limit // 3, 1)
        )
        dev_records, knet_noise_dev = limit_source_records(
            dev_records,
            "knet_prep_noise",
            False,
            knet_dev_noise_limit,
            args.data_seed + 24,
        )
        dev_records, ig_noise_dev = limit_source_records(
            dev_records,
            "instancegm_prep_noise",
            False,
            ig_dev_noise_limit,
            args.data_seed + 25,
        )
        selected = dict(selected)
        selected.update(
            {
                "ig_train": ig_train,
                "knet_train": knet_train,
                "ig_dev": ig_dev,
                "knet_dev_joint": knet_dev_joint,
                "knet_noise_dev": knet_noise_dev,
                "ig_noise_dev": ig_noise_dev,
            }
        )
        protocol = dict(protocol)
        protocol["v25_population_first_selection"].update(
            {
                "post_scope_pre_limit_event_counts": population_counts,
                "record_limits": {
                    "train_instancegm": args.ig_limit,
                    "train_knet": args.knet_limit,
                    "development_per_event_source": args.dev_limit,
                },
                "selected_event_counts": {
                    "train_instancegm": int(len(ig_train)),
                    "train_knet": int(len(knet_train)),
                    "development_instancegm": int(len(ig_dev)),
                    "development_knet_joint": int(len(knet_dev_joint)),
                },
            }
        )
    if train_scope["enabled"]:
        for split, scope in (("train", train_scope), ("development", dev_scope)):
            if not sum(
                source_counts["eligible"]
                for source_counts in scope["by_source"].values()
            ):
                raise ValueError(f"Event target scope removed every {split} event")
        protocol = dict(protocol)
        protocol["event_target_scope"] = {
            "train": train_scope,
            "development": dev_scope,
            "noise_records_filtered": False,
            "deployment_boundary": (
                "Magnitude and source-station distance define the target population; "
                "they are unavailable at first-P station inference and are not model inputs."
            ),
        }
        selected = dict(selected)
        for split, records in (("train", train_records), ("development", dev_records)):
            for source in ("instancegm", "knet"):
                selected[f"event_scope_{split}_{source}"] = np.asarray(
                    [
                        record.index
                        for record in records
                        if record.is_event and record.source == source
                    ],
                    dtype=np.int64,
                )
    if args.use_continuous_noise:
        if not args.continuous_noise_hdf5.exists():
            raise FileNotFoundError(
                f"Continuous noise store missing: {args.continuous_noise_hdf5}. "
                "Run scripts/data/prepare_domain_robust_v5_continuous_noise.py first."
            )
        if not args.continuous_catalog_screen.is_file():
            raise FileNotFoundError(
                f"Continuous catalog screen missing: {args.continuous_catalog_screen}. "
                "Run scripts/data/build_continuous_catalog_screen.py first."
            )
        continuous_train = ContinuousNoiseWindowSource(
            "ci2024_continuous_train_noise",
            args.continuous_noise_hdf5,
            "train",
            args.continuous_catalog_screen,
        )
        continuous_dev = ContinuousNoiseWindowSource(
            "ci2024_continuous_dev_noise",
            args.continuous_noise_hdf5,
            "dev",
            args.continuous_catalog_screen,
        )
        train_indices = deterministic_limit(
            continuous_train.noise_indices,
            args.continuous_train_limit,
            args.data_seed + 31,
        )
        dev_indices = deterministic_limit(
            continuous_dev.noise_indices,
            args.continuous_dev_limit,
            args.data_seed + 32,
        )
        train_sources[continuous_train.name] = continuous_train
        dev_sources[continuous_dev.name] = continuous_dev
        train_noise_sources.append(continuous_train.name)
        dev_noise_sources.append(continuous_dev.name)
        train_records.extend(make_records(continuous_train.name, train_indices, False))
        dev_records.extend(make_records(continuous_dev.name, dev_indices, False))
        selected = dict(selected)
        selected["ci2024_continuous_train"] = train_indices
        selected["ci2024_continuous_dev"] = dev_indices
        protocol = dict(protocol)
        protocol["continuous_noise"] = {
            "store": str(args.continuous_noise_hdf5.resolve()),
            "catalog_screen": str(args.continuous_catalog_screen.resolve()),
            "catalog_screen_sha256": file_sha256(args.continuous_catalog_screen),
            "train_windows": len(train_indices),
            "dev_windows": len(dev_indices),
            "catalog_excluded_train_windows": continuous_train.catalog_excluded_count,
            "catalog_excluded_dev_windows": continuous_dev.catalog_excluded_count,
            "date_split": "2024 Jan-Mar train; 2024 May-Dec development calibration",
            "selection": (
                "prediction-independent QC-valid non-overlapping windows, then catalog-screened "
                "to presumed non-event status before negative supervision"
            ),
            "claim_boundary": (
                "Catalog-screened presumed non-event windows; absence from the assembled catalogs "
                "does not certify physical noise"
            ),
        }
    if args.hard_negative_review_manifest is not None:
        if not args.use_continuous_noise:
            raise ValueError("Reviewed hard negatives require the continuous noise store")
        if not args.hard_negative_review_manifest.is_file():
            raise FileNotFoundError(args.hard_negative_review_manifest)
        hard_negative = ReviewedHardNegativeWindowSource(
            "ci2024_continuous_hard_train_noise",
            args.continuous_noise_hdf5,
            args.hard_negative_review_manifest,
            args.continuous_catalog_screen,
        )
        hard_indices = hard_negative.noise_indices.copy()
        train_sources[hard_negative.name] = hard_negative
        train_records.extend(make_records(hard_negative.name, hard_indices, False))
        selected = dict(selected)
        selected["ci2024_reviewed_hard_train"] = hard_indices
        protocol = dict(protocol)
        protocol["reviewed_hard_negative_training"] = {
            "manifest": str(args.hard_negative_review_manifest.resolve()),
            "manifest_sha256": file_sha256(args.hard_negative_review_manifest),
            "records": int(len(hard_indices)),
            "supervised_packets": int(
                hard_negative.metadata["supervision_packet_count"].sum()
            ),
            "scope": (
                "training only; local packet masks around provisionally reviewed triggers; "
                "all rows also pass the catalog screen; not used as generic event noise beds; "
                "development and blind sets unchanged"
            ),
            "sampler_effect": (
                "a separate continuous-noise source group receives an equal share of the "
                "configured continuous noise-class mass"
            ),
        }
    target_paths = (
        args.target_hard_negative_hdf5,
        args.target_hard_negative_review_manifest,
        args.target_hard_negative_catalog_screen,
    )
    if any(path is not None for path in target_paths):
        if args.architecture not in OPERATIONAL_TRAINING_ARCHITECTURES:
            raise ValueError(
                "Target FDSN trigger windows require a target-verifier architecture"
            )
        if not all(path is not None and path.is_file() for path in target_paths):
            raise FileNotFoundError(
                "V10 target hard negatives require HDF5, review manifest, and catalog screen"
            )
        target_train = ReviewedHardNegativeWindowSource(
            "target_fdsn_continuous_hard_train_noise",
            args.target_hard_negative_hdf5,
            args.target_hard_negative_review_manifest,
            args.target_hard_negative_catalog_screen,
            split="train",
        )
        target_dev = ReviewedHardNegativeWindowSource(
            "target_fdsn_continuous_hard_dev_noise",
            args.target_hard_negative_hdf5,
            args.target_hard_negative_review_manifest,
            args.target_hard_negative_catalog_screen,
            split="dev",
        )
        target_train_indices = target_train.noise_indices
        target_dev_indices = target_dev.noise_indices
        if args.smoke:
            target_train_indices = deterministic_limit(
                target_train_indices, 64, args.data_seed + 41
            )
            target_dev_indices = deterministic_limit(
                target_dev_indices, 32, args.data_seed + 42
            )
        train_sources[target_train.name] = target_train
        dev_sources[target_dev.name] = target_dev
        train_records.extend(
            make_records(target_train.name, target_train_indices, False)
        )
        dev_records.extend(make_records(target_dev.name, target_dev_indices, False))
        selected = dict(selected)
        selected["target_fdsn_actual_trigger_train"] = target_train_indices
        selected["target_fdsn_actual_trigger_dev"] = target_dev_indices
        protocol = dict(protocol)
        protocol["target_fdsn_actual_trigger_windows"] = {
            "store": str(args.target_hard_negative_hdf5.resolve()),
            "store_sha256": file_sha256(args.target_hard_negative_hdf5),
            "review_manifest": str(
                args.target_hard_negative_review_manifest.resolve()
            ),
            "review_manifest_sha256": file_sha256(
                args.target_hard_negative_review_manifest
            ),
            "catalog_screen": str(args.target_hard_negative_catalog_screen.resolve()),
            "catalog_screen_sha256": file_sha256(
                args.target_hard_negative_catalog_screen
            ),
            "training_windows": int(len(target_train_indices)),
            "development_calibration_windows": int(len(target_dev_indices)),
            "split": "2026-06-02/09 train; 2026-06-16 development calibration",
            "supervision": (
                "local masks around catalog-screened presumed non-event actual triggers; "
                "causal packet context receives no negative loss"
            ),
            "operational_normalization_state_preserved": bool(
                target_train.preserve_operational_normalization
                and target_dev.preserve_operational_normalization
            ),
            "normalization_contract": (
                "window-start causal EWMA state is restored when present; stores without "
                "state retain the legacy reset-at-window behavior"
            ),
            "event_noise_bed_use": False,
            "claim_boundary": (
                "presumed non-event operational triggers, not certified physical noise"
            ),
        }
    background_paths = (
        args.target_background_hdf5,
        args.target_background_catalog_screen,
    )
    if any(path is not None for path in background_paths):
        if args.architecture not in OPERATIONAL_TRAINING_ARCHITECTURES:
            raise ValueError(
                "Target FDSN backgrounds require a target-verifier architecture"
            )
        if not all(path is not None and path.is_file() for path in background_paths):
            raise FileNotFoundError(
                "V10 target backgrounds require HDF5 and catalog screen"
            )
        target_background_train = ContinuousNoiseWindowSource(
            "target_fdsn_background_train_noise",
            args.target_background_hdf5,
            "train",
            args.target_background_catalog_screen,
        )
        target_background_dev = ContinuousNoiseWindowSource(
            "target_fdsn_background_dev_noise",
            args.target_background_hdf5,
            "dev",
            args.target_background_catalog_screen,
        )
        train_sources[target_background_train.name] = target_background_train
        dev_sources[target_background_dev.name] = target_background_dev
        train_noise_sources.append(target_background_train.name)
        dev_noise_sources.append(target_background_dev.name)
        selected = dict(selected)
        selected["target_fdsn_background_train"] = (
            target_background_train.noise_indices
        )
        selected["target_fdsn_background_dev"] = target_background_dev.noise_indices
        protocol = dict(protocol)
        protocol["target_fdsn_background_event_mixing"] = {
            "store": str(args.target_background_hdf5.resolve()),
            "store_sha256": file_sha256(args.target_background_hdf5),
            "catalog_screen": str(args.target_background_catalog_screen.resolve()),
            "catalog_screen_sha256": file_sha256(
                args.target_background_catalog_screen
            ),
            "training_windows": int(
                len(target_background_train.noise_indices)
            ),
            "development_calibration_windows": int(
                len(target_background_dev.noise_indices)
            ),
            "record_sampling_mass": 0.0,
            "event_noise_bed_use": True,
            "split": "2026-06-02/09 train; 2026-06-16 development calibration",
            "claim_boundary": (
                "QC/catalog-screened presumed non-event target-domain backgrounds; "
                "used for event mixing, not independent evidence"
            ),
        }
    train = DomainRobustStreamDatasetV5(
        train_sources,
        train_records,
        args.max_chunks,
        training=True,
        noise_source_names=tuple(train_noise_sources),
        seed=args.seed,
        onset_response_packets=onset_response_packets,
        verifier_response_packets=verifier_response_packets,
        snr_profile=snr_profile,
        minimum_event_p_packet=(args.warmup_packets if uses_arrival_age else 0),
        verifier_start_delay_packets=verifier_start_delay_packets,
        native_event_fraction=args.native_event_fraction,
    )
    dev = DomainRobustStreamDatasetV5(
        dev_sources,
        dev_records,
        args.max_chunks,
        training=False,
        noise_source_names=tuple(dev_noise_sources),
        seed=args.data_seed + 1,
        onset_response_packets=onset_response_packets,
        verifier_response_packets=verifier_response_packets,
        snr_profile=snr_profile,
        minimum_event_p_packet=(args.warmup_packets if uses_arrival_age else 0),
        verifier_start_delay_packets=verifier_start_delay_packets,
        native_event_fraction=args.native_event_fraction,
    )
    protocol = dict(protocol)
    if knet_manifest is not None:
        protocol["knet_causal_raw_input"] = knet_manifest
    protocol.update(
        {
            "protocol_id": "domain_robust_v5_engineering_20260715",
            "status": "development_training_before_blind_cwa_2021",
            "architecture": args.architecture,
            "data_pipeline": (
                "contiguous true-native event crops plus independent-noise assembly, "
                "seam-matched relocation for the synthetic branch, and 1 s local "
                "P-window excess-power SNR with nominal, constructed-achieved, and "
                "empirical values stored separately"
            ),
            "label": (
                f"causal {onset_response_packets}-packet candidate window and "
                f"confirmation packets P+{verifier_start_delay_packets} through "
                f"P+{verifier_response_packets - 1}; "
                "no pre-P positives; arrival-age models regress causal elapsed time at each "
                "confirmation packet"
            ),
            "snr_training_profile": snr_profile,
            "native_event_fraction": args.native_event_fraction,
            "native_event_policy": {
                "definition": (
                    "contiguous crop from the original waveform with its original "
                    "background; no relocation, background replacement, added noise, "
                    "component augmentation, or injected artifact"
                ),
                "requested_fraction": args.native_event_fraction,
                "eligibility": (
                    "requires enough original pre-P warmup and post-P guard inside "
                    "the selected sequence; ineligible draws use the synthetic branch"
                ),
                "replacement_sampler_nonce": (
                    "repeated draws of one training record receive deterministic "
                    "distinct augmentation seeds; development draws remain fixed"
                ),
            },
            "data_selection_seed": args.data_seed,
            "optimization_seed": args.seed,
            "training_sequence_seconds": args.max_chunks * 0.5,
            "sampler": (
                "CSPP V25/V26/V27 freeze InstanceGM/K-NET/CWA-event/target-trigger/"
                "opened-continuous/other-noise mass at "
                "0.40/0.15/0.10/0.20/0.10/0.05"
                if args.architecture
                in {"v25", "v26", *CSPP_V27_ARCHITECTURES}
                else (
                    "Target-verifier models freeze event/target-trigger/opened-2024/"
                    "other-noise mass at 0.50/0.35/0.10/0.05"
                )
                if args.architecture in TARGET_VERIFIER_ARCHITECTURES
                else (
                    "two-thirds events and one-third noise; opened 2024 continuous windows receive "
                    f"{args.continuous_noise_class_fraction:.2f} of the noise-class mass when enabled"
                )
            ),
            "threshold_selection": (
                "epoch checkpoints jointly score native, 0 dB, +10 dB, and short-noise development "
                "cohorts; the operating threshold must be frozen later with the opened 2024 continuous stream"
            ),
            "checkpoint_selection": (
                "highest geometric mean of native usable rate, native noise specificity, 0 dB usable "
                "rate, and +10 dB usable rate among thresholds meeting the minimum native-noise "
                "specificity constraint"
            ),
            "hard_negative_loss": (
                "separately weighted top probabilities on pure-noise records and on event records "
                "before P or after the three-packet response window"
            ),
            "minimum_native_noise_specificity": args.minimum_noise_specificity,
            "final_blind_test": "CWA _2021; prohibited during model and threshold development",
        }
    )
    if population_first_selection:
        protocol["counts"] = {
            "train_instancegm_event": sum(
                record.is_event and record.source == "instancegm"
                for record in train.records
            ),
            "train_knet_event": sum(
                record.is_event and record.source == "knet"
                for record in train.records
            ),
            "train_cwa2020_event": sum(
                record.is_event
                and record.source == "cwa2020_opened_train_events"
                for record in train.records
            ),
            "development_instancegm_event": sum(
                record.is_event and record.source == "instancegm"
                for record in dev.records
            ),
            "development_knet_joint_event": sum(
                record.is_event and record.source == "knet"
                for record in dev.records
            ),
            "train_noise": sum(not record.is_event for record in train.records),
            "development_noise": sum(
                not record.is_event for record in dev.records
            ),
        }
    if args.architecture in PACKET_ONCE_ARCHITECTURES:
        protocol["packet_once_contract"] = {
            "model_class": "PacketOnceCausalPickerV1",
            "state_mode": {
                "packet_b1": "packet",
                "packet_b2": "retention",
                "packet_b3": "gru",
            }[args.architecture],
            "raw_packet_encoder_calls_per_packet": 1,
            "raw_replay_packets": 0,
            "constant_size_stream_state": True,
            "retention_half_lives_seconds": parse_float_list(
                args.retention_half_lives_seconds
            ),
            "training_boundary_phase": (
                "event P is relocated to a uniformly sampled integer sample, so all 50 packet "
                "phases are eligible independently in every epoch"
            ),
            "paired_boundary_consistency_loss": (
                "not enabled in the state-only pilot; reserved as a separately frozen ablation"
            ),
        }
    elif args.architecture in {
        "v8",
        "v9",
        "v25",
        "v26",
        *CSPP_V27_ARCHITECTURES,
        *TARGET_VERIFIER_ARCHITECTURES,
    }:
        protocol["packet_once_contract"] = {
            "model_class": {
                "v8": "DomainRobustCausalPickerV8",
                "v9": "DomainRobustCausalPickerV9",
                "v10": "DomainRobustCausalPickerV10",
                "v11": "DomainRobustCausalPickerV11",
                "v25": "CSPPCausalPickerV25",
                "v26": "CSPPCausalPickerV26",
                "v27": "CSPPCausalPickerV27",
                "v27b": "CSPPCausalPickerV27B",
                "v27c": "CSPPCausalPickerV27C",
            }[args.architecture],
            "state_mode": (
                (
                    "unidirectional_gru"
                    if args.v27_temporal_mode == "gru"
                    else "packet_only_no_cross_packet_state"
                )
                if args.architecture in CSPP_V27_ARCHITECTURES
                else "exact_per_layer_dilated_tcn_history"
            ),
            "raw_packet_encoder_calls_per_packet": (
                2 if args.architecture in TARGET_VERIFIER_ARCHITECTURES else 1
            ),
            "old_temporal_tokens_recomputed_per_packet": 0,
            "raw_replay_packets": 0,
            "constant_size_stream_state": True,
            "training_boundary_phase": (
                "event P is relocated to a uniformly sampled integer sample, so all 50 packet "
                "phases are eligible independently in every epoch"
            ),
            "streaming_equivalence_test": (
                "full causal convolution, arbitrary blocked replay, and one-packet replay must "
                "agree within 1e-5"
            ),
        }
        if args.architecture in {
            "v9",
            "v25",
            "v26",
            *CSPP_V27_ARCHITECTURES,
            *TARGET_VERIFIER_ARCHITECTURES,
        }:
            protocol["packet_once_contract"]["spectral_features"] = (
                "fixed within-current-packet DFT bands, entropy, and centroid; no future packet, "
                "raw replay, or additional temporal cache"
            )
        if args.architecture in {"v25", "v26"}:
            protocol["packet_once_contract"].update(
                {
                    "training_initialization": "random_weights_from_scratch",
                    "complete_packet_morphology": (
                        "symmetric convolutions operate only inside the fully received "
                        "current 0.5-second packet"
                    ),
                    "cross_packet_context": (
                        "strictly past-only causal TCN plus fixed multi-scale "
                        "exponential retention state"
                    ),
                    "integrated_outputs": (
                        "candidate probability, event confirmation probability, "
                        "and causal P-arrival age"
                    ),
                }
            )
        if args.architecture in CSPP_V27_ARCHITECTURES:
            protocol["packet_once_contract"].update(
                {
                    "training_initialization": (
                        "frozen V27 candidate with a newly initialized verifier"
                        if args.architecture in {"v27b", "v27c"}
                        else "random_weights_from_scratch"
                    ),
                    "complete_packet_morphology": (
                        "the current 0.5-second packet is encoded once; dense "
                        "phase interpolation never crosses a packet boundary"
                    ),
                    "cross_packet_context": (
                        "unidirectional two-layer GRU plus previous-packet "
                        "embedding; no future packet or raw replay"
                        if args.v27_temporal_mode == "gru"
                        else "current packet only; no cross-packet state, "
                        "future packet, or raw replay"
                    ),
                    "within_packet_dense_p_evidence": (
                        "50-sample P logits are supervised at the annotated "
                        "sample and their peak and soft mass enter the candidate head"
                        if not args.v27_disable_dense_phase_evidence
                        else "disabled; dense phase logits do not enter the "
                        "candidate head and their loss weight is zero"
                    ),
                    "event_head_role": (
                        "detached nonlinear episode verifier over recurrent and "
                        "candidate-connected evidence"
                        if args.architecture == "v27b"
                        else "detached nonlinear episode verifier with the "
                        f"{args.verifier_feature_mode} direct packet feature ablation"
                        if args.architecture == "v27c"
                        else "auxiliary event evidence; candidate recall is evaluated "
                        "separately from any event-threshold policy"
                    ),
                    "integrated_outputs": (
                        "candidate probability, auxiliary event evidence, "
                        "causal P-arrival age, and within-packet P logits"
                    ),
                    "spectral_features": (
                        "fixed DFT band fractions, vertical fractions, entropy, "
                        "and centroid from the fully received current packet only"
                        if not args.v27_disable_spectral_features
                        else "disabled by replacing all fixed spectral "
                        "descriptors with zeros"
                    ),
                    "within_packet_p_auxiliary_target": {
                        "samples": SAMPLES_PER_CHUNK,
                        "loss_weight": args.packet_localization_loss_weight,
                        "sigma_samples": args.packet_localization_sigma_samples,
                    },
                    "architecture_change_scope": (
                        "replace only the detached linear event head; preserve the "
                        "frozen V27 candidate, timing, and dense phase branches"
                        if args.architecture == "v27b"
                        else "preserve the frozen V27 candidate, timing, and dense "
                        "phase branches; vary only direct verifier packet evidence"
                        if args.architecture == "v27c"
                        else "restore the compact legacy CNN-GRU temporal backbone, "
                        "retain post-P-only labels, and add a candidate-connected "
                        "dense within-packet P branch"
                    ),
                    "verifier_candidate_gated_loss": (
                        args.verifier_candidate_gated_loss
                    ),
                    "verifier_candidate_gate_threshold": (
                        resolved_verifier_candidate_gate_threshold(args)
                        if args.verifier_candidate_gated_loss
                        else None
                    ),
                    "verifier_supervision_rule": (
                        "detached candidate support OR positive verifier label; "
                        "retain all locally masked reviewed hard-negative packets"
                        if args.verifier_candidate_gated_loss
                        else "all valid or locally masked packets"
                    ),
                    "verifier_episode_objective": {
                        "enabled": args.verifier_episode_loss,
                        "positive_packets": (
                            args.verifier_episode_positive_packets
                        ),
                        "tail_fraction": args.verifier_episode_tail_fraction,
                        "bce_weight": args.verifier_episode_bce_weight,
                        "tail_weight": args.verifier_episode_tail_weight,
                        "unit": (
                            "contiguous detached-candidate episodes"
                            if args.verifier_episode_loss
                            else None
                        ),
                    },
                    "verifier_feature_mode": (
                        args.verifier_feature_mode
                        if args.architecture == "v27c"
                        else "compressed_candidate_evidence"
                        if args.architecture == "v27b"
                        else None
                    ),
                }
            )
        if args.architecture == "v26":
            protocol["packet_once_contract"].update(
                {
                    "within_packet_p_auxiliary_target": {
                        "samples": SAMPLES_PER_CHUNK,
                        "loss_weight": args.packet_localization_loss_weight,
                        "sigma_samples": args.packet_localization_sigma_samples,
                    },
                    "architecture_change_scope": (
                        "V25 packet features plus a depthwise dense P-location "
                        "head; the cross-packet causal backbone is unchanged"
                    ),
                }
            )
        if args.architecture in TARGET_VERIFIER_ARCHITECTURES:
            v10_joint_training = is_v10_joint_training(args)
            protocol["packet_once_contract"].update(
                {
                    "frozen_v9_candidate_and_timing": not v10_joint_training,
                    "trainable_parameters": (
                        "candidate, timing, and independent target event verifier"
                        if v10_joint_training
                        else "independent target event verifier only"
                        if args.architecture == "v10"
                        else "residual hidden-state fusion head only"
                    ),
                    "frozen_v10_target_features": args.architecture == "v11",
                    "candidate_output_preservation_required": (
                        "not required after joint optimization; initializer identity recorded"
                        if v10_joint_training
                        else "exact in eval mode"
                    ),
                    "raw_packet_contract": (
                        "each branch encodes the newly arrived packet once; neither branch "
                        "replays earlier raw packets or uses a sliding waveform window"
                    ),
                    "fixed_candidate_probability_threshold": (
                        None if v10_joint_training else V10_CANDIDATE_THRESHOLD
                    ),
                    "threshold_selection_scope": (
                        "candidate and target event-verifier thresholds"
                        if v10_joint_training
                        else "target event-verifier threshold only"
                    ),
                    "verifier_candidate_gated_loss": args.verifier_candidate_gated_loss,
                    "verifier_candidate_gate_threshold": (
                        resolved_verifier_candidate_gate_threshold(args)
                        if args.verifier_candidate_gated_loss
                        else None
                    ),
                    "verifier_supervision_rule": (
                        "valid packets with frozen-P candidate support OR positive verifier label; "
                        "all locally masked packets retained for reviewed hard-negative rows"
                        if args.verifier_candidate_gated_loss
                        else "all valid or locally masked packets"
                    ),
                    "verifier_candidate_logits_detached": True,
                    "joint_backbone_lr_multiplier": (
                        args.v10_backbone_lr_multiplier
                        if v10_joint_training
                        else None
                    ),
                    "verifier_partial_auc": {
                        "loss_weight": getattr(
                            args, "verifier_partial_auc_loss_weight", 0.0
                        ),
                        "hard_negative_record_fraction": (
                            getattr(
                                args,
                                "verifier_partial_auc_negative_fraction",
                                0.25,
                            )
                        ),
                        "logit_margin": getattr(
                            args, "verifier_partial_auc_margin", 1.0
                        ),
                        "objective": (
                            "rank event response maxima above the CVaR-like tail of "
                            "record-level noise maxima"
                        ),
                    },
                }
            )
    return train, dev, protocol, selected


def main() -> None:
    args = parse_args()
    args.data_seed = resolved_data_seed(args)
    args.verifier_candidate_gate_threshold = (
        resolved_verifier_candidate_gate_threshold(args)
    )
    v10_joint_training = is_v10_joint_training(args)
    if args.v10_train_scope == "joint" and args.architecture != "v10":
        raise ValueError("--v10-train-scope joint requires --architecture v10")
    if not 0.0 < args.v10_backbone_lr_multiplier <= 1.0:
        raise ValueError("--v10-backbone-lr-multiplier must be in (0, 1]")
    if args.architecture in {"v25", "v26", "v27"} and args.init_checkpoint is not None:
        raise ValueError(
            f"{args.architecture.upper()} is a from-scratch protocol and "
            "refuses --init-checkpoint"
        )
    if args.architecture in {"v27b", "v27c"} and args.init_checkpoint is None:
        raise ValueError(
            f"{args.architecture.upper()} requires --init-checkpoint with a "
            "frozen V27 candidate"
        )
    v27_ablation_requested = (
        args.v27_temporal_mode != "gru"
        or args.v27_disable_dense_phase_evidence
        or args.v27_disable_spectral_features
    )
    if (
        v27_ablation_requested
        and args.architecture not in CSPP_V27_ARCHITECTURES
    ):
        raise ValueError("V27 ablation flags require a V27 architecture")
    if args.distillation_loss_weight < 0.0:
        raise ValueError("--distillation-loss-weight must be non-negative")
    if args.distillation_temperature <= 0.0:
        raise ValueError("--distillation-temperature must be positive")
    if (
        args.distillation_teacher_checkpoint is not None
        and not args.distillation_teacher_checkpoint.is_file()
    ):
        raise FileNotFoundError(args.distillation_teacher_checkpoint)
    if (
        args.distillation_teacher_checkpoint is not None
        and args.architecture not in {"v25", "v26", "v27"}
    ):
        raise ValueError("Causal distillation currently supports V25/V26/V27 students")
    development_candidate_thresholds = None
    if args.development_candidate_thresholds is not None:
        development_candidate_thresholds = tuple(
            parse_float_list(args.development_candidate_thresholds)
        )
        if not development_candidate_thresholds or any(
            not 0.0 < threshold < 1.0
            for threshold in development_candidate_thresholds
        ):
            raise ValueError(
                "--development-candidate-thresholds must contain values in (0, 1)"
            )
    if (
        args.verifier_candidate_gated_loss
        and args.architecture not in OPERATIONAL_TRAINING_ARCHITECTURES
    ):
        raise ValueError(
            "Candidate-gated verifier loss requires a target-verifier architecture"
        )
    if args.verifier_candidate_gated_loss and (
        args.candidate_history_packets < 1 or args.warmup_packets < 1
    ):
        raise ValueError("Candidate-gated verifier loss requires positive history and warmup")
    if args.verifier_partial_auc_loss_weight < 0.0:
        raise ValueError("Partial-AUC loss weight must be non-negative")
    if (
        args.verifier_partial_auc_loss_weight > 0.0
        and args.architecture not in OPERATIONAL_TRAINING_ARCHITECTURES
    ):
        raise ValueError(
            "Partial-AUC verifier loss requires a target-verifier architecture"
        )
    if not 0.0 < args.verifier_partial_auc_negative_fraction <= 1.0:
        raise ValueError("Partial-AUC negative fraction must be in (0, 1]")
    if args.verifier_episode_loss and args.architecture not in {"v27b", "v27c"}:
        raise ValueError("Episode verifier loss currently requires V27B or V27C")
    if (
        args.no_development_evaluation
        and args.architecture not in CSPP_V27_ARCHITECTURES
    ):
        raise ValueError(
            "--no-development-evaluation currently requires a V27 architecture"
        )
    if args.verifier_episode_bce_weight < 0.0:
        raise ValueError("Episode BCE weight must be non-negative")
    if args.verifier_episode_tail_weight < 0.0:
        raise ValueError("Episode tail weight must be non-negative")
    if not 0.0 < args.verifier_episode_tail_fraction <= 1.0:
        raise ValueError("Episode tail fraction must be in (0, 1]")
    if not 1 <= args.verifier_episode_positive_packets <= V7_VERIFIER_RESPONSE_PACKETS:
        raise ValueError("Invalid positive episode packet count")
    if args.smoke:
        args.epochs = 1
        args.max_chunks = min(
            args.max_chunks,
            96 if args.architecture in OPERATIONAL_TRAINING_ARCHITECTURES else 64,
        )
        args.ig_limit = min(args.ig_limit, 64)
        args.knet_limit = min(args.knet_limit, 64)
        args.noise_limit = min(args.noise_limit, 32)
        args.dev_limit = min(args.dev_limit, 32)
        args.continuous_train_limit = min(args.continuous_train_limit, 64)
        args.continuous_dev_limit = min(args.continuous_dev_limit, 32)
        args.cwa_event_limit = (
            64
            if args.cwa_event_limit <= 0
            else min(args.cwa_event_limit, 64)
        )
        args.save_dir = args.save_dir / "smoke"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = select_device(args.device)
    args.save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}")
    print(f"Output: {args.save_dir}")

    train_dataset, dev_dataset, protocol, selected = build_datasets(args)
    if args.selection_reference_npz is not None:
        protocol["selection_reference"] = verify_selection_reference(
            selected, args.selection_reference_npz
        )
    uses_arrival_age = args.architecture in ARRIVAL_AGE_ARCHITECTURES
    snr_profile = (
        "v7_operational" if uses_arrival_age else "legacy"
    ) if args.snr_training_profile == "auto" else args.snr_training_profile
    onset_response_packets = (
        V25_ONSET_RESPONSE_PACKETS
        if args.architecture in {"v25", "v26", *CSPP_V27_ARCHITECTURES}
        else V7_ONSET_RESPONSE_PACKETS
        if uses_arrival_age
        else ONSET_RESPONSE_PACKETS
    )
    verifier_response_packets = (
        V7_VERIFIER_RESPONSE_PACKETS if uses_arrival_age else VERIFIER_RESPONSE_PACKETS
    )
    verifier_start_delay_packets = resolved_verifier_start_delay_packets(
        args,
        uses_arrival_age,
    )
    candidate_history_packets = args.candidate_history_packets if uses_arrival_age else 1
    warmup_packets = args.warmup_packets if uses_arrival_age else 0
    timing_mode = "arrival_age" if uses_arrival_age else "packet_offset"
    if args.architecture in OPERATIONAL_TRAINING_ARCHITECTURES:
        if args.architecture in {
            "v25",
            "v26",
            *CSPP_V27_ARCHITECTURES,
        }:
            sampler, group_counts = v25_balanced_sampler(
                train_dataset.records,
                args.seed,
            )
        else:
            sampler, group_counts = v10_balanced_sampler(
                train_dataset.records,
                args.seed,
            )
    else:
        sampler, group_counts = balanced_sampler(
            train_dataset.records,
            args.seed,
            continuous_noise_class_fraction=args.continuous_noise_class_fraction,
        )
    protocol["train_record_groups"] = group_counts
    protocol["development_evaluation"] = {
        "enabled": not args.no_development_evaluation,
        "checkpoint_selection": (
            "development gate count plus progress"
            if not args.no_development_evaluation
            else "disabled; final epoch fixed before training"
        ),
        "threshold_sweep": not args.no_development_evaluation,
        "final_artifact": (
            "best.pt"
            if not args.no_development_evaluation
            else f"final.pt copied from epoch_{args.epochs:02d}.pt"
        ),
    }
    save_json(args.save_dir / "protocol.json", protocol)
    np.savez_compressed(args.save_dir / "selected_indices.npz", **selected)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=collate_domain_robust_v5,
    )
    dev_loader = None
    snr_dev_loaders = {}
    if not args.no_development_evaluation:
        dev_loader = DataLoader(
            dev_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_domain_robust_v5,
        )
        for snr_db in (0.0, 10.0):
            dataset = DomainRobustStreamDatasetV5(
                dev_dataset.sources,
                dev_dataset.records,
                args.max_chunks,
                training=False,
                noise_source_names=dev_dataset.noise_source_names,
                seed=args.data_seed + 100 + int(snr_db),
                eval_snr_db=snr_db,
                onset_response_packets=onset_response_packets,
                verifier_response_packets=verifier_response_packets,
                snr_profile=snr_profile,
                minimum_event_p_packet=(
                    args.warmup_packets if uses_arrival_age else 0
                ),
                verifier_start_delay_packets=verifier_start_delay_packets,
                native_event_fraction=args.native_event_fraction,
            )
            snr_dev_loaders[snr_db] = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                collate_fn=collate_domain_robust_v5,
            )

    model_types = {
        "v5": DomainRobustCausalPickerV5,
        "v6": DomainRobustCausalPickerV6,
        "v7": DomainRobustCausalPickerV7,
        "v8": DomainRobustCausalPickerV8,
        "v9": DomainRobustCausalPickerV9,
        "v10": DomainRobustCausalPickerV10,
        "v11": DomainRobustCausalPickerV11,
        "v25": CSPPCausalPickerV25,
        "v26": CSPPCausalPickerV26,
        "v27": CSPPCausalPickerV27,
        "v27b": CSPPCausalPickerV27B,
        "v27c": CSPPCausalPickerV27C,
        "packet_b1": PacketOnceCausalPickerV1,
        "packet_b2": PacketOnceCausalPickerV1,
        "packet_b3": PacketOnceCausalPickerV1,
    }
    model_kwargs = {
        "packet_width": args.packet_width,
        "embedding_dim": args.embedding_dim,
        "sequence_width": args.sequence_width,
        "dropout": args.dropout,
    }
    if uses_arrival_age:
        model_kwargs["max_arrival_age_packets"] = args.max_arrival_age_packets
    if (
        args.architecture
        in {
            "v8",
            "v9",
            "v25",
            "v26",
            *CSPP_V27_ARCHITECTURES,
            *TARGET_VERIFIER_ARCHITECTURES,
        }
        or args.architecture in PACKET_ONCE_ARCHITECTURES
    ):
        model_kwargs["temporal_bins"] = args.temporal_bins
    if args.architecture in TARGET_VERIFIER_ARCHITECTURES:
        model_kwargs.update(
            {
                "target_packet_width": args.target_packet_width,
                "target_embedding_dim": args.target_embedding_dim,
                "target_sequence_width": args.target_sequence_width,
            }
        )
    if args.architecture == "v11":
        model_kwargs.update(
            {
                "fusion_hidden_dim": args.fusion_hidden_dim,
                "fusion_dropout": args.fusion_dropout,
            }
        )
    if args.architecture in {"v27b", "v27c"}:
        model_kwargs["verifier_hidden_dim"] = args.verifier_hidden_dim
    if args.architecture == "v27c":
        model_kwargs["verifier_feature_mode"] = args.verifier_feature_mode
    if args.architecture in CSPP_V27_ARCHITECTURES:
        model_kwargs.update(
            {
                "temporal_mode": args.v27_temporal_mode,
                "dense_phase_evidence": (
                    not args.v27_disable_dense_phase_evidence
                ),
                "spectral_features": (
                    not args.v27_disable_spectral_features
                ),
                "decoder_attention_heads": args.v27_decoder_attention_heads,
                "decoder_context_packets": args.v27_decoder_context_packets,
                "decoder_ffn_multiplier": args.v27_decoder_ffn_multiplier,
            }
        )
    if args.architecture in PACKET_ONCE_ARCHITECTURES:
        model_kwargs["state_mode"] = {
            "packet_b1": "packet",
            "packet_b2": "retention",
            "packet_b3": "gru",
        }[args.architecture]
        model_kwargs["retention_half_lives_seconds"] = tuple(
            parse_float_list(args.retention_half_lives_seconds)
        )
    if args.architecture in {"v25", "v26", *CSPP_V27_ARCHITECTURES}:
        model_kwargs["retention_half_lives_seconds"] = tuple(
            parse_float_list(args.retention_half_lives_seconds)
        )
    model = model_types[args.architecture](
        **model_kwargs,
    ).to(device)
    if args.architecture in TARGET_VERIFIER_ARCHITECTURES and args.init_checkpoint is None:
        required = "V9" if args.architecture == "v10" else "V10"
        raise ValueError(
            f"{args.architecture.upper()} requires --init-checkpoint with a frozen {required} model"
        )
    if args.init_checkpoint is not None:
        initial = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        expected_class = getattr(model, "model_class", "DomainRobustCausalPickerV5")
        initial_class = initial.get("model_class", "DomainRobustCausalPickerV5")
        v10_from_v9 = args.architecture == "v10" and initial_class == "DomainRobustCausalPickerV9"
        v11_from_v10 = (
            args.architecture == "v11"
            and initial_class == "DomainRobustCausalPickerV10"
        )
        v27_verifier_from_v27 = (
            args.architecture in {"v27b", "v27c"}
            and initial_class == "CSPPCausalPickerV27"
        )
        if initial_class != expected_class and not (
            v10_from_v9 or v11_from_v10 or v27_verifier_from_v27
        ):
            raise ValueError(
                f"Cannot initialize {expected_class} from {initial_class}; train the new architecture from scratch"
            )
        if args.architecture in PACKET_ONCE_ARCHITECTURES:
            initial_mode = initial.get("model_config", {}).get("state_mode")
            if initial_mode != model.config.state_mode:
                raise ValueError(
                    f"Cannot initialize packet state_mode={model.config.state_mode} "
                    f"from state_mode={initial_mode}"
                )
        if v10_from_v9:
            model.load_v9_state_dict(initial["model"])
        elif v11_from_v10:
            model.load_v10_state_dict(initial["model"])
        elif v27_verifier_from_v27:
            model.load_v27_candidate_state_dict(initial["model"])
        else:
            model.load_state_dict(initial["model"])
        if args.architecture == "v10" and not v10_joint_training:
            model.freeze_backbone()
            reference = DomainRobustCausalPickerV9.from_config(
                initial["model_config"]
            ).to(device)
            reference_keys = set(reference.state_dict())
            reference_state = {
                key: value
                for key, value in initial["model"].items()
                if key in reference_keys
            }
            missing_reference = reference_keys - set(reference_state)
            if missing_reference:
                raise ValueError(
                    "V10 initializer is missing frozen V9 tensors: "
                    f"{sorted(missing_reference)}"
                )
            reference.load_state_dict(reference_state)
            reference.eval()
            model.eval()
            probe = torch.linspace(
                -2.0, 2.0, steps=8 * 3 * 50, device=device
            ).reshape(1, 8, 3, 50)
            with torch.inference_mode():
                reference_p, _, reference_delay = reference(probe)
                v10_p, _, v10_delay = model(probe)
            candidate_max_difference = float(
                (reference_p - v10_p).abs().max().detach().cpu()
            )
            delay_max_difference = float(
                (reference_delay - v10_delay).abs().max().detach().cpu()
            )
            if candidate_max_difference != 0.0 or delay_max_difference != 0.0:
                raise RuntimeError("V10 changed frozen V9 candidate or timing outputs")
            del reference, probe
            protocol["v10_frozen_backbone"] = {
                "checkpoint": str(args.init_checkpoint.resolve()),
                "sha256": file_sha256(args.init_checkpoint),
                "model_class": initial_class,
                "candidate_and_timing_trainable": False,
                "candidate_probe_max_abs_difference": candidate_max_difference,
                "timing_probe_max_abs_difference": delay_max_difference,
            }
        elif args.architecture == "v10":
            protocol["v10_joint_initialization"] = {
                "checkpoint": str(args.init_checkpoint.resolve()),
                "sha256": file_sha256(args.init_checkpoint),
                "model_class": initial_class,
                "candidate_and_timing_trainable": True,
                "target_event_verifier_trainable": True,
                "backbone_lr_multiplier": args.v10_backbone_lr_multiplier,
            }
        elif args.architecture == "v11":
            model.freeze_feature_extractors()
            if v11_from_v10:
                reference = DomainRobustCausalPickerV10.from_config(
                    initial["model_config"]
                ).to(device)
                reference.load_state_dict(initial["model"])
                reference.eval()
                model.eval()
                probe = torch.linspace(
                    -2.0, 2.0, steps=8 * 3 * 50, device=device
                ).reshape(1, 8, 3, 50)
                with torch.inference_mode():
                    reference_outputs = reference(probe)
                    v11_outputs = model(probe)
                output_differences = {
                    name: float(
                        (reference_value - v11_value).abs().max().detach().cpu()
                    )
                    for name, reference_value, v11_value in zip(
                        ("candidate", "event", "timing"),
                        reference_outputs,
                        v11_outputs,
                        strict=True,
                    )
                }
                if any(value != 0.0 for value in output_differences.values()):
                    raise RuntimeError(
                        "V11 residual fusion did not preserve the V10 initializer: "
                        f"{output_differences}"
                    )
                del reference, probe
                protocol["v11_frozen_feature_extractors"] = {
                    "checkpoint": str(args.init_checkpoint.resolve()),
                    "sha256": file_sha256(args.init_checkpoint),
                    "model_class": initial_class,
                    "candidate_target_and_timing_trainable": False,
                    "initial_probe_max_abs_difference": output_differences,
                }
        elif args.architecture in {"v27b", "v27c"}:
            model.freeze_candidate()
            reference = CSPPCausalPickerV27.from_config(
                initial["model_config"]
            ).to(device)
            reference.load_state_dict(initial["model"])
            reference.eval()
            model.eval()
            probe = torch.linspace(
                -2.0,
                2.0,
                steps=8 * 3 * SAMPLES_PER_CHUNK,
                device=device,
            ).reshape(1, 8, 3, SAMPLES_PER_CHUNK)
            with torch.inference_mode():
                reference_candidate, _, reference_delay, reference_phase = (
                    reference.forward_with_aux(probe)
                )
                verifier_candidate, _, verifier_delay, verifier_phase = (
                    model.forward_with_aux(probe)
                )
            output_differences = {
                "candidate": float(
                    (reference_candidate - verifier_candidate)
                    .abs()
                    .max()
                    .detach()
                    .cpu()
                ),
                "timing": float(
                    (reference_delay - verifier_delay)
                    .abs()
                    .max()
                    .detach()
                    .cpu()
                ),
                "dense_phase": float(
                    (reference_phase - verifier_phase)
                    .abs()
                    .max()
                    .detach()
                    .cpu()
                ),
            }
            if any(value != 0.0 for value in output_differences.values()):
                raise RuntimeError(
                    f"{args.architecture.upper()} changed the frozen V27 candidate: "
                    f"{output_differences}"
                )
            del reference, probe
            protocol["v27_frozen_candidate"] = {
                "checkpoint": str(args.init_checkpoint.resolve()),
                "sha256": file_sha256(args.init_checkpoint),
                "model_class": initial_class,
                "candidate_timing_and_phase_trainable": False,
                "verifier_trainable": True,
                "verifier_hidden_dim": args.verifier_hidden_dim,
                "verifier_feature_mode": (
                    args.verifier_feature_mode
                    if args.architecture == "v27c"
                    else "compressed_candidate_evidence"
                ),
                "initial_probe_max_abs_difference": output_differences,
            }
        protocol["initialized_from"] = str(args.init_checkpoint.resolve())
        save_json(args.save_dir / "protocol.json", protocol)

    distillation_teacher = None
    if args.distillation_teacher_checkpoint is not None:
        cpu_rng_state = torch.random.get_rng_state()
        teacher_checkpoint = torch.load(
            args.distillation_teacher_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        teacher_class_name = teacher_checkpoint.get("model_class")
        teacher_types = {
            "CSPPCausalPickerV25": CSPPCausalPickerV25,
            "CSPPCausalPickerV26": CSPPCausalPickerV26,
            "CSPPCausalPickerV27": CSPPCausalPickerV27,
        }
        if teacher_class_name not in teacher_types:
            raise ValueError(
                "Distillation refuses a non-CSPP or unverified teacher: "
                f"{teacher_class_name!r}"
            )
        teacher_type = teacher_types[teacher_class_name]
        distillation_teacher = teacher_type.from_config(
            teacher_checkpoint["model_config"]
        ).to(device)
        distillation_teacher.load_state_dict(teacher_checkpoint["model"])
        if (
            not bool(getattr(distillation_teacher, "packet_once", False))
            or int(getattr(distillation_teacher, "raw_replay_packets", -1)) != 0
        ):
            raise ValueError("Distillation teacher violates the packet-once contract")
        distillation_teacher.eval()
        distillation_teacher.requires_grad_(False)
        torch.random.set_rng_state(cpu_rng_state)
        protocol["causal_distillation"] = {
            "teacher_checkpoint": str(
                args.distillation_teacher_checkpoint.resolve()
            ),
            "teacher_sha256": file_sha256(
                args.distillation_teacher_checkpoint
            ),
            "teacher_model_class": teacher_class_name,
            "teacher_uses_future_packets": False,
            "student_input": "identical waveform prefix ending at current packet",
            "loss_weight": args.distillation_loss_weight,
            "temperature": args.distillation_temperature,
            "outputs": ["candidate", "event_confirmation"],
            "explicit_exclusion": (
                "noncausal EQTransformer/PhaseNet soft targets are not accepted"
            ),
        }
        save_json(args.save_dir / "protocol.json", protocol)

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    history_description = (
        "constant-size recurrent state"
        if math.isinf(float(model.receptive_field_packets))
        else f"{model.receptive_field_packets:g} packets"
    )
    print(f"Parameters: {parameter_count:,}; history: {history_description}")
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not trainable_parameters:
        raise RuntimeError("Model has no trainable parameters")
    print(f"Trainable parameters: {sum(p.numel() for p in trainable_parameters):,}")
    if v10_joint_training:
        named_trainable = [
            (name, parameter)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ]
        target_parameters = [
            parameter
            for name, parameter in named_trainable
            if name.startswith("target_")
        ]
        backbone_parameters = [
            parameter
            for name, parameter in named_trainable
            if not name.startswith("target_")
        ]
        if not target_parameters or not backbone_parameters:
            raise RuntimeError("Joint V10 optimization requires target and backbone parameters")
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": target_parameters,
                    "lr": args.lr,
                    "name": "target_verifier",
                },
                {
                    "params": backbone_parameters,
                    "lr": args.lr * args.v10_backbone_lr_multiplier,
                    "name": "candidate_timing_backbone",
                },
            ],
            weight_decay=args.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: 0.1
            + 0.9
            * 0.5
            * (1.0 + math.cos(math.pi * step / max(args.epochs, 1))),
        )
    else:
        optimizer = torch.optim.AdamW(
            trainable_parameters, lr=args.lr, weight_decay=args.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(args.epochs, 1), eta_min=args.lr * 0.1
        )

    best_score = -math.inf
    history = []
    for epoch in range(1, args.epochs + 1):
        train_dataset.set_epoch(epoch + args.curriculum_epoch_offset)
        model.train()
        if args.architecture in {"v27b", "v27c"}:
            model.encoder.eval()
            gru = getattr(model, "gru", None)
            if gru is not None:
                gru.eval()
            model.candidate_head.eval()
            model.delay_head.eval()
            model.event_head.train()
        epoch_start = time.time()
        totals = {
            "loss": 0.0,
            "onset": 0.0,
            "event": 0.0,
            "event_outside": 0.0,
            "noise_topk": 0.0,
            "verifier_hard_negative": 0.0,
            "verifier_ranking": 0.0,
            "verifier_partial_auc": 0.0,
            "verifier_episode_bce": 0.0,
            "verifier_episode_tail": 0.0,
            "ranking": 0.0,
            "offset": 0.0,
            "packet_localization": 0.0,
            "distillation": 0.0,
        }
        verifier_supervision = {
            "eligible_packets": 0.0,
            "supervised_packets": 0.0,
            "positive_packets": 0.0,
            "negative_packets": 0.0,
            "weighted_positive_mass": 0.0,
            "weighted_negative_mass": 0.0,
            "positive_episodes": 0.0,
            "negative_episodes": 0.0,
        }
        snr_counts = {"native": 0, "-10_0": 0, "0_10": 0, "10_20": 0, "20_30": 0}
        for batch_index, batch in enumerate(train_loader, start=1):
            chunks = batch["chunks"].to(device)
            labels = batch["labels"].to(device)
            labels_event = batch["labels_event"].to(device)
            mask = batch["mask"].to(device)
            target_chunks = batch["target_chunk"].to(device)
            target_offsets = batch["target_offset"].to(device)
            is_event = batch["is_event"].to(device)
            snr_values = batch["snr_db"].to(device)
            sample_weights = torch.ones_like(snr_values)
            if uses_arrival_age and args.low_snr_event_weight > 1.0:
                emphasized = (
                    is_event
                    & torch.isfinite(snr_values)
                    & (snr_values >= -0.25)
                    & (snr_values <= args.low_snr_weight_upper_db)
                )
                sample_weights[emphasized] = args.low_snr_event_weight
            phase_logits = None
            if args.architecture in {"v26", *CSPP_V27_ARCHITECTURES}:
                (
                    logits_p,
                    logits_event,
                    logits_offset,
                    phase_logits,
                ) = model.forward_with_aux(chunks)
            else:
                logits_p, logits_event, logits_offset = model(chunks)
            if distillation_teacher is not None:
                with torch.inference_mode():
                    teacher_p, teacher_event, _ = distillation_teacher(chunks)
                distillation = 0.5 * (
                    masked_bernoulli_distillation_loss(
                        logits_p,
                        teacher_p,
                        mask,
                        args.distillation_temperature,
                    )
                    + masked_bernoulli_distillation_loss(
                        logits_event,
                        teacher_event,
                        mask,
                        args.distillation_temperature,
                    )
                )
            else:
                distillation = logits_p.sum() * 0.0
            verifier_mask = mask
            if args.verifier_candidate_gated_loss:
                reviewed_hard_rows = torch.tensor(
                    [
                        bool(
                            getattr(
                                train_dataset.sources[source],
                                "reviewed_hard_negative",
                                False,
                            )
                        )
                        for source in batch["source"]
                    ],
                    dtype=torch.bool,
                    device=device,
                )
                verifier_mask = candidate_gated_verifier_mask(
                    mask,
                    logits_p,
                    labels_event,
                    reviewed_hard_rows,
                    threshold=args.verifier_candidate_gate_threshold,
                    history_packets=args.candidate_history_packets,
                    warmup_packets=args.warmup_packets,
                )
            verifier_supervision["eligible_packets"] += float(mask.sum().detach())
            verifier_supervision["supervised_packets"] += float(
                verifier_mask.sum().detach()
            )
            verifier_positive = verifier_mask * labels_event
            verifier_negative = verifier_mask * (1.0 - labels_event)
            record_weights = sample_weights.unsqueeze(1)
            verifier_supervision["positive_packets"] += float(
                verifier_positive.sum().detach()
            )
            verifier_supervision["negative_packets"] += float(
                verifier_negative.sum().detach()
            )
            verifier_supervision["weighted_positive_mass"] += float(
                (verifier_positive * record_weights).sum().detach()
            )
            verifier_supervision["weighted_negative_mass"] += float(
                (verifier_negative * record_weights).sum().detach()
            )

            onset = weighted_onset_loss(
                logits_p,
                labels,
                mask,
                args.onset_pos_weight,
                sample_weights=sample_weights,
            )
            event = weighted_onset_loss(
                logits_event,
                labels_event,
                verifier_mask,
                args.event_pos_weight,
                sample_weights=sample_weights,
            )
            event_outside = event_outside_topk_loss(
                logits_p,
                target_chunks,
                is_event,
                mask,
                response_packets=onset_response_packets,
            )
            noise_topk = noise_topk_loss(logits_p, is_event, mask)
            ranking = first_fire_ranking_loss(
                logits_p,
                target_chunks,
                is_event,
                mask,
                args.ranking_margin,
                response_packets=onset_response_packets,
                sample_weights=sample_weights,
            )
            verifier_hard_negative = 0.5 * (
                event_outside_topk_loss(
                    logits_event,
                    target_chunks,
                    is_event,
                    verifier_mask,
                    response_packets=verifier_response_packets,
                    response_start_delay_packets=verifier_start_delay_packets,
                )
                + noise_topk_loss(logits_event, is_event, verifier_mask)
            )
            verifier_ranking = first_fire_ranking_loss(
                logits_event,
                target_chunks,
                is_event,
                verifier_mask,
                args.ranking_margin,
                response_packets=verifier_response_packets,
                response_start_delay_packets=verifier_start_delay_packets,
                sample_weights=sample_weights,
            )
            verifier_partial_auc = verifier_partial_auc_ranking_loss(
                logits_event,
                target_chunks,
                is_event,
                verifier_mask,
                negative_fraction=args.verifier_partial_auc_negative_fraction,
                margin=args.verifier_partial_auc_margin,
                response_packets=verifier_response_packets,
                response_start_delay_packets=verifier_start_delay_packets,
                sample_weights=sample_weights,
            )
            if args.verifier_episode_loss:
                (
                    verifier_episode_bce,
                    verifier_episode_tail,
                    verifier_episode_counts,
                ) = verifier_candidate_episode_losses(
                    logits_event,
                    logits_p,
                    target_chunks,
                    is_event,
                    mask,
                    candidate_threshold=args.verifier_candidate_gate_threshold,
                    history_packets=args.candidate_history_packets,
                    warmup_packets=args.warmup_packets,
                    positive_packets=args.verifier_episode_positive_packets,
                    exclusion_packets=verifier_response_packets,
                    tail_fraction=args.verifier_episode_tail_fraction,
                    margin=args.verifier_partial_auc_margin,
                    sample_weights=sample_weights,
                )
            else:
                verifier_episode_bce = logits_event.sum() * 0.0
                verifier_episode_tail = logits_event.sum() * 0.0
                verifier_episode_counts = {
                    "positive_episodes": 0,
                    "negative_episodes": 0,
                }
            verifier_supervision["positive_episodes"] += float(
                verifier_episode_counts["positive_episodes"]
            )
            verifier_supervision["negative_episodes"] += float(
                verifier_episode_counts["negative_episodes"]
            )
            if uses_arrival_age:
                offset = arrival_age_regression_loss(
                    logits_offset,
                    target_chunks,
                    target_offsets,
                    is_event,
                    mask,
                    response_packets=verifier_response_packets,
                    response_start_delay_packets=verifier_start_delay_packets,
                    max_arrival_age_packets=args.max_arrival_age_packets,
                    sample_weights=sample_weights,
                )
            else:
                offset = offset_regression_loss(
                    logits_offset, target_chunks, target_offsets, is_event
                )
            packet_localization = (
                (
                    dense_phase_detection_loss(
                        phase_logits,
                        target_chunks,
                        target_offsets,
                        is_event,
                        mask,
                        sigma_samples=args.packet_localization_sigma_samples,
                        sample_weights=sample_weights,
                    )
                    if args.architecture in CSPP_V27_ARCHITECTURES
                    else packet_localization_loss(
                        phase_logits,
                        target_chunks,
                        target_offsets,
                        is_event,
                        sigma_samples=args.packet_localization_sigma_samples,
                        sample_weights=sample_weights,
                    )
                )
                if phase_logits is not None
                else logits_p.sum() * 0.0
            )
            if args.verifier_episode_loss:
                loss = (
                    args.verifier_episode_bce_weight
                    * verifier_episode_bce
                    + args.verifier_episode_tail_weight
                    * verifier_episode_tail
                )
            elif (
                args.architecture in TARGET_VERIFIER_ARCHITECTURES
                and not v10_joint_training
            ) or args.architecture in {"v27b", "v27c"}:
                loss = (
                    args.event_loss_weight * event
                    + args.verifier_hard_negative_loss_weight
                    * verifier_hard_negative
                    + args.verifier_ranking_loss_weight * verifier_ranking
                    + args.verifier_partial_auc_loss_weight
                    * verifier_partial_auc
                )
            else:
                loss = (
                    onset
                    + args.event_loss_weight * event
                    + args.event_outside_loss_weight * event_outside
                    + args.noise_topk_loss_weight * noise_topk
                    + args.ranking_loss_weight * ranking
                    + args.verifier_hard_negative_loss_weight
                    * verifier_hard_negative
                    + args.verifier_ranking_loss_weight * verifier_ranking
                    + args.verifier_partial_auc_loss_weight
                    * verifier_partial_auc
                    + args.offset_loss_weight * offset
                    + args.packet_localization_loss_weight
                    * packet_localization
                    + args.distillation_loss_weight * distillation
                )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_parameters, 5.0)
            optimizer.step()

            values = {
                "loss": loss,
                "onset": onset,
                "event": event,
                "event_outside": event_outside,
                "noise_topk": noise_topk,
                "verifier_hard_negative": verifier_hard_negative,
                "verifier_ranking": verifier_ranking,
                "verifier_partial_auc": verifier_partial_auc,
                "verifier_episode_bce": verifier_episode_bce,
                "verifier_episode_tail": verifier_episode_tail,
                "ranking": ranking,
                "offset": offset,
                "packet_localization": packet_localization,
                "distillation": distillation,
            }
            for key, value in values.items():
                totals[key] += float(value.detach())
            finite_snr = batch["snr_db"][torch.isfinite(batch["snr_db"])]
            snr_counts["native"] += int((~torch.isfinite(batch["snr_db"]) & batch["is_event"]).sum())
            snr_counts["-10_0"] += int(((finite_snr >= -10) & (finite_snr < 0)).sum())
            snr_counts["0_10"] += int(((finite_snr >= 0) & (finite_snr < 10)).sum())
            snr_counts["10_20"] += int(((finite_snr >= 10) & (finite_snr <= 20)).sum())
            snr_counts["20_30"] += int(((finite_snr > 20) & (finite_snr <= 30)).sum())
            if batch_index % 50 == 0 or batch_index == len(train_loader):
                print(
                    f"epoch {epoch}/{args.epochs} batch {batch_index}/{len(train_loader)} "
                    f"loss={totals['loss'] / batch_index:.4f}",
                    flush=True,
                )

        scheduler.step()
        batches = max(len(train_loader), 1)
        row = {
            "epoch": epoch,
            "seconds": time.time() - epoch_start,
            "lr": max(group["lr"] for group in optimizer.param_groups),
            "learning_rates": {
                str(group.get("name", f"group_{index}")): group["lr"]
                for index, group in enumerate(optimizer.param_groups)
            },
            "loss": {key: value / batches for key, value in totals.items()},
            "snr_counts": snr_counts,
            "verifier_supervision": {
                **verifier_supervision,
                "fraction": (
                    verifier_supervision["supervised_packets"]
                    / max(verifier_supervision["eligible_packets"], 1.0)
                ),
                "raw_negative_to_positive_ratio": (
                    verifier_supervision["negative_packets"]
                    / max(verifier_supervision["positive_packets"], 1.0)
                ),
                "weighted_negative_to_positive_ratio": (
                    verifier_supervision["weighted_negative_mass"]
                    / max(
                        verifier_supervision["weighted_positive_mass"],
                        1.0,
                    )
                ),
            },
        }
        if args.no_development_evaluation:
            provisional_threshold = None
            provisional_event_threshold = None
            dev_metrics = None
            low_snr_dev = None
            threshold_sweep = None
            selection_score = None
            checkpoint_selection_score = None
            row["development_evaluation"] = "disabled"
        else:
            if dev_loader is None:
                raise RuntimeError("Development loader was not constructed")
            predictions = collect_predictions_v5(model, dev_loader, device)
            snr_predictions = {}
            for snr_db, loader in snr_dev_loaders.items():
                snr_predictions[snr_db] = collect_predictions_v5(
                    model,
                    loader,
                    device,
                )
            (
                provisional_threshold,
                provisional_event_threshold,
                dev_metrics,
                low_snr_dev,
                threshold_sweep,
            ) = select_joint_development_threshold(
                predictions,
                snr_predictions,
                minimum_noise_specificity=args.minimum_noise_specificity,
                event_thresholds=tuple(
                    parse_float_list(args.development_event_thresholds)
                ),
                candidate_history_packets=candidate_history_packets,
                warmup_packets=warmup_packets,
                timing_mode=timing_mode,
                max_arrival_age_packets=args.max_arrival_age_packets,
                minimum_native_usable=args.minimum_native_usable,
                minimum_zero_db_usable=args.minimum_zero_db_usable,
                minimum_ten_db_usable=args.minimum_ten_db_usable,
                minimum_source_native_usable=args.minimum_source_native_usable,
                candidate_thresholds=(
                    (V10_CANDIDATE_THRESHOLD,)
                    if args.architecture in TARGET_VERIFIER_ARCHITECTURES
                    and not v10_joint_training
                    else development_candidate_thresholds
                ),
            )
            selected_threshold_row = next(
                threshold_row
                for threshold_row in threshold_sweep
                if math.isclose(
                    threshold_row["threshold"],
                    provisional_threshold,
                )
                and math.isclose(
                    threshold_row["event_threshold"],
                    provisional_event_threshold,
                )
            )
            selection_score = selected_threshold_row["selection_score"]
            checkpoint_selection_score = (
                selected_threshold_row["development_gate_count"]
                + selected_threshold_row["development_gate_progress"]
            )
            row.update(
                {
                    "provisional_dev_threshold": provisional_threshold,
                    "provisional_dev_event_threshold": (
                        provisional_event_threshold
                    ),
                    "threshold_specificity_constraint_met": (
                        selected_threshold_row["noise_specificity_eligible"]
                    ),
                    "threshold_availability_constraint_met": (
                        selected_threshold_row["availability_eligible"]
                    ),
                    "dev": dev_metrics,
                    "low_snr_dev": low_snr_dev,
                    "selection_score": selection_score,
                    "checkpoint_selection_score": (
                        checkpoint_selection_score
                    ),
                    "development_gate_count": selected_threshold_row[
                        "development_gate_count"
                    ],
                    "development_gate_progress": selected_threshold_row[
                        "development_gate_progress"
                    ],
                }
            )
        history.append(row)
        print(json.dumps(row, indent=2), flush=True)

        checkpoint = {
            "model": model.state_dict(),
            "model_class": getattr(model, "model_class", "DomainRobustCausalPickerV5"),
            "model_config": model.config.as_dict(),
            "training_config": vars(args) | {"knet_dir": str(args.knet_dir), "save_dir": str(args.save_dir)},
            "protocol": protocol,
            "epoch": epoch,
            "provisional_dev_threshold": provisional_threshold,
            "provisional_dev_event_threshold": provisional_event_threshold,
            "dev_metrics": dev_metrics,
            "low_snr_dev_metrics": low_snr_dev,
            "selection_score": selection_score,
            "checkpoint_selection_score": checkpoint_selection_score,
            "parameter_count": parameter_count,
            "candidate_policy": {
                "candidate_history_packets": candidate_history_packets,
                "warmup_packets": warmup_packets,
                "timing_mode": timing_mode,
                "max_arrival_age_packets": args.max_arrival_age_packets,
                "onset_response_packets": onset_response_packets,
                "verifier_response_packets": verifier_response_packets,
                "verifier_start_delay_packets": verifier_start_delay_packets,
            },
        }
        epoch_checkpoint_path = args.save_dir / f"epoch_{epoch:02d}.pt"
        torch.save(checkpoint, epoch_checkpoint_path)
        if threshold_sweep is not None:
            save_json(
                args.save_dir / f"epoch_{epoch:02d}_threshold_sweep.json",
                threshold_sweep,
            )
        if args.no_development_evaluation and epoch == args.epochs:
            shutil.copyfile(epoch_checkpoint_path, args.save_dir / "final.pt")
        elif (
            checkpoint_selection_score is not None
            and checkpoint_selection_score > best_score
        ):
            best_score = checkpoint_selection_score
            torch.save(checkpoint, args.save_dir / "best.pt")
            save_json(
                args.save_dir / "best_threshold_sweep.json",
                threshold_sweep,
            )
        save_json(args.save_dir / "history.json", history)

    if args.no_development_evaluation:
        print("Training complete. Development evaluation was disabled.")
        print(f"Fixed final checkpoint: {args.save_dir / 'final.pt'}")
    else:
        print(f"Training complete. Best checkpoint gate score={best_score:.4f}")
        print(f"Best checkpoint: {args.save_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
