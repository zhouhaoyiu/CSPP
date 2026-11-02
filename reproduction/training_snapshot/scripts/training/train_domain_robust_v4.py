#!/usr/bin/env python3
"""Train the independent domain-robust v4 streaming picker."""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from config import SAMPLES_PER_CHUNK, TARGET_SR
from domain_robust_data import (
    DomainRobustStreamDataset,
    KNETSource,
    PrePNoiseSource,
    SeisBenchSource,
    collate_domain_robust,
    deterministic_limit,
    event_indices,
    find_p_column,
    instancegm_acceleration_indices,
    knet_factorized_indices,
    make_records,
    station_key,
)
from domain_robust_metrics import collect_predictions, select_threshold
from model_domain_robust_v4 import DomainRobustCausalPickerV4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--knet-dir", type=Path, default=ROOT / "data" / "knet_accel")
    parser.add_argument("--save-dir", type=Path, default=ROOT / "models" / "checkpoints" / "domain_robust_v4")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-chunks", type=int, default=160)
    parser.add_argument("--ig-limit", type=int, default=12000)
    parser.add_argument("--knet-limit", type=int, default=12000)
    parser.add_argument("--noise-limit", type=int, default=6000)
    parser.add_argument("--dev-limit", type=int, default=800)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--onset-pos-weight", type=float, default=20.0)
    parser.add_argument("--event-loss-weight", type=float, default=0.15)
    parser.add_argument("--max-negative-weight", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--device", choices=("auto", "mps", "cpu", "cuda"), default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--strong-motion-only", action="store_true")
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


def valid_window_indices(metadata, indices: np.ndarray, max_samples: int) -> np.ndarray:
    p_column = find_p_column(metadata)
    values = metadata.iloc[indices][p_column].astype(float).to_numpy()
    return np.asarray(indices, dtype=np.int64)[(values >= 0) & (values < max_samples)]


def weighted_onset_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    pos_weight: float,
) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    weights = 1.0 + (pos_weight - 1.0) * targets
    return (loss * weights * mask).sum() / mask.sum().clamp_min(1.0)


def masked_bce(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)


def max_negative_loss(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    negative = (labels < 0.05) & mask.bool()
    masked = logits.masked_fill(~negative, -1e4)
    top_count = min(4, logits.shape[1])
    top = masked.topk(top_count, dim=1).values
    valid = negative.any(dim=1)
    if not valid.any():
        return logits.sum() * 0.0
    return F.softplus(top[valid]).mean()


def save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def build_datasets(args: argparse.Namespace):
    from seisbench.data import InstanceGM

    print("Loading local waveform datasets...", flush=True)
    instance = SeisBenchSource("instancegm", InstanceGM(sampling_rate=TARGET_SR))
    knet = KNETSource(args.knet_dir)
    max_samples = args.max_chunks * SAMPLES_PER_CHUNK

    if args.strong_motion_only:
        ig_train_pool = instancegm_acceleration_indices(instance.metadata, "train")
    else:
        ig_train_pool = event_indices(instance.metadata, "train", instance.p_column)
    ig_train = ig_train_pool
    ig_train = deterministic_limit(ig_train, args.ig_limit, args.seed + 11)
    knet_train = knet_factorized_indices(knet.metadata, "train", args.seed)
    knet_train = deterministic_limit(knet_train, args.knet_limit, args.seed + 12)

    if args.strong_motion_only:
        ig_dev = valid_window_indices(
            instance.metadata,
            instancegm_acceleration_indices(instance.metadata, "dev"),
            max_samples,
        )
    else:
        ig_dev = event_indices(instance.metadata, "dev", instance.p_column, max_samples=max_samples)
    ig_dev = deterministic_limit(ig_dev, args.dev_limit, args.seed + 21)
    knet_dev_event = valid_window_indices(
        knet.metadata, knet_factorized_indices(knet.metadata, "dev_event", args.seed), max_samples
    )
    knet_dev_station = valid_window_indices(
        knet.metadata, knet_factorized_indices(knet.metadata, "dev_station", args.seed), max_samples
    )
    each_knet_dev = max(args.dev_limit // 2, 1)
    knet_dev_event = deterministic_limit(knet_dev_event, each_knet_dev, args.seed + 22)
    knet_dev_station = deterministic_limit(knet_dev_station, each_knet_dev, args.seed + 23)

    if args.strong_motion_only:
        ig_p = instance.metadata[instance.p_column].astype(float).to_numpy()
        knet_p = knet.metadata[knet.p_column].astype(float).to_numpy()
        ig_noise_pool = ig_train_pool[ig_p[ig_train_pool] >= 2 * TARGET_SR]
        knet_noise_pool = knet_factorized_indices(knet.metadata, "train", args.seed)
        knet_noise_pool = knet_noise_pool[knet_p[knet_noise_pool] >= 2 * TARGET_SR]
        knet_noise_limit = max(round(args.noise_limit * 2 / 3), 1)
        ig_noise_limit = max(args.noise_limit - knet_noise_limit, 1)
        knet_noise_train = deterministic_limit(knet_noise_pool, knet_noise_limit, args.seed + 13)
        ig_noise_train = deterministic_limit(ig_noise_pool, ig_noise_limit, args.seed + 14)

        ig_dev_pool = instancegm_acceleration_indices(instance.metadata, "dev")
        ig_dev_noise_pool = ig_dev_pool[ig_p[ig_dev_pool] >= 2 * TARGET_SR]
        knet_dev_noise_pool = np.unique(np.concatenate((knet_dev_event, knet_dev_station)))
        knet_dev_noise_pool = knet_dev_noise_pool[knet_p[knet_dev_noise_pool] >= 2 * TARGET_SR]
        knet_noise_dev = deterministic_limit(knet_dev_noise_pool, max(args.dev_limit * 2 // 3, 1), args.seed + 24)
        ig_noise_dev = deterministic_limit(ig_dev_noise_pool, max(args.dev_limit // 3, 1), args.seed + 25)

        knet_prep = PrePNoiseSource("knet_prep_noise", knet, knet_noise_pool)
        ig_prep = PrePNoiseSource("instancegm_prep_noise", instance, ig_noise_pool)
        sources = {source.name: source for source in (instance, knet, knet_prep, ig_prep)}
        noise_source_names = ("knet_prep_noise", "instancegm_prep_noise")
        train_records = (
            make_records("instancegm", ig_train, True)
            + make_records("knet", knet_train, True)
            + make_records("knet_prep_noise", knet_noise_train, False)
            + make_records("instancegm_prep_noise", ig_noise_train, False)
        )
        dev_records = (
            make_records("instancegm", ig_dev, True)
            + make_records("knet", knet_dev_event, True)
            + make_records("knet", knet_dev_station, True)
            + make_records("knet_prep_noise", knet_noise_dev, False)
            + make_records("instancegm_prep_noise", ig_noise_dev, False)
        )
        noise_counts = {
            "train_knet_prep_noise": len(knet_noise_train),
            "train_instancegm_prep_noise": len(ig_noise_train),
            "dev_knet_prep_noise": len(knet_noise_dev),
            "dev_instancegm_prep_noise": len(ig_noise_dev),
        }
        noise_selected = {
            "knet_noise_train": knet_noise_train,
            "ig_noise_train": ig_noise_train,
            "knet_noise_dev": knet_noise_dev,
            "ig_noise_dev": ig_noise_dev,
        }
    else:
        from seisbench.data import STEAD

        stead = SeisBenchSource("stead_noise", STEAD(sampling_rate=TARGET_SR))
        sources = {source.name: source for source in (instance, stead, knet)}
        noise_source_names = ("stead_noise",)
        noise_train_mask = stead.metadata["split"].eq("train") & stead.metadata["trace_category"].eq("noise")
        noise_train = deterministic_limit(
            np.flatnonzero(noise_train_mask.to_numpy()), args.noise_limit, args.seed + 13
        )
        noise_dev_mask = stead.metadata["split"].eq("dev") & stead.metadata["trace_category"].eq("noise")
        noise_dev = deterministic_limit(
            np.flatnonzero(noise_dev_mask.to_numpy()), args.dev_limit, args.seed + 24
        )
        train_records = (
            make_records("instancegm", ig_train, True)
            + make_records("knet", knet_train, True)
            + make_records("stead_noise", noise_train, False)
        )
        dev_records = (
            make_records("instancegm", ig_dev, True)
            + make_records("knet", knet_dev_event, True)
            + make_records("knet", knet_dev_station, True)
            + make_records("stead_noise", noise_dev, False)
        )
        noise_counts = {"train_stead_noise": len(noise_train), "dev_stead_noise": len(noise_dev)}
        noise_selected = {"noise_train": noise_train, "noise_dev": noise_dev}

    train_dataset = DomainRobustStreamDataset(
        sources,
        train_records,
        args.max_chunks,
        training=True,
        noise_source_names=noise_source_names,
        seed=args.seed,
    )
    dev_dataset = DomainRobustStreamDataset(
        sources,
        dev_records,
        args.max_chunks,
        training=False,
        noise_source_names=noise_source_names,
        seed=args.seed + 1,
    )

    train_stations = set(station_key(knet.metadata.iloc[knet_train]))
    dev_station_stations = set(station_key(knet.metadata.iloc[knet_dev_station]))
    if train_stations & dev_station_stations:
        raise RuntimeError("K-NET train and station-dev partitions overlap")
    protocol = {
        "status": "development_retrain_not_manuscript_evidence",
        "focus": "strong_motion_acceleration" if args.strong_motion_only else "mixed_instruments",
        "instancegm_filter": "HN/EN channels with mps2 units" if args.strong_motion_only else "all labeled channels",
        "negative_data": "pre-P acceleration windows" if args.strong_motion_only else "STEAD noise traces",
        "seed": args.seed,
        "sample_rate_hz": TARGET_SR,
        "packet_samples": SAMPLES_PER_CHUNK,
        "max_chunks": args.max_chunks,
        "low_snr_training_db": [-10.0, 20.0],
        "threshold_selection": "source dev only; harmonic mean of event usable rate and noise specificity",
        "knet_station_buckets": {"train": [0, 69], "dev": [70, 84], "test": [85, 99]},
        "counts": {
            "train_instancegm_event": len(ig_train),
            "train_knet_event": len(knet_train),
            "dev_instancegm_event": len(ig_dev),
            "dev_knet_event_held": len(knet_dev_event),
            "dev_knet_station_held": len(knet_dev_station),
        } | noise_counts,
        "station_counts": {
            "knet_train": len(train_stations),
            "knet_station_dev": len(dev_station_stations),
        },
    }
    selected_indices = {
        "ig_train": ig_train,
        "knet_train": knet_train,
        "ig_dev": ig_dev,
        "knet_dev_event": knet_dev_event,
        "knet_dev_station": knet_dev_station,
    } | noise_selected
    return train_dataset, dev_dataset, protocol, selected_indices


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.epochs = 1
        args.max_chunks = min(args.max_chunks, 64)
        args.ig_limit = min(args.ig_limit, 64)
        args.knet_limit = min(args.knet_limit, 64)
        args.noise_limit = min(args.noise_limit, 32)
        args.dev_limit = min(args.dev_limit, 32)
        args.save_dir = args.save_dir / "smoke"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = select_device(args.device)
    print(f"Device: {device}")
    print(f"Output: {args.save_dir}")
    args.save_dir.mkdir(parents=True, exist_ok=True)

    train_dataset, dev_dataset, protocol, selected = build_datasets(args)
    save_json(args.save_dir / "protocol.json", protocol)
    np.savez_compressed(args.save_dir / "selected_indices.npz", **selected)
    print(json.dumps(protocol["counts"], indent=2))

    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=args.num_workers,
        collate_fn=collate_domain_robust,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_domain_robust,
    )

    model = DomainRobustCausalPickerV4().to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"Parameters: {parameter_count:,}; receptive field: {model.receptive_field_packets} packets")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1), eta_min=args.lr * 0.1)

    best_score = -math.inf
    history = []
    for epoch in range(1, args.epochs + 1):
        train_dataset.set_epoch(epoch)
        model.train()
        epoch_start = time.time()
        totals = {"loss": 0.0, "onset": 0.0, "event": 0.0, "negative": 0.0}
        snr_counts = {"native": 0, "-10_0": 0, "0_10": 0, "10_20": 0, "20_30": 0}
        for batch_index, batch in enumerate(train_loader, start=1):
            chunks = batch["chunks"].to(device)
            labels = batch["labels"].to(device)
            labels_event = batch["labels_event"].to(device)
            mask = batch["mask"].to(device)
            logits_p, logits_event = model(chunks)
            onset = weighted_onset_loss(logits_p, labels, mask, args.onset_pos_weight)
            event = masked_bce(logits_event, labels_event, mask)
            negative = max_negative_loss(logits_p, labels, mask)
            loss = onset + args.event_loss_weight * event + args.max_negative_weight * negative

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            totals["loss"] += float(loss.detach())
            totals["onset"] += float(onset.detach())
            totals["event"] += float(event.detach())
            totals["negative"] += float(negative.detach())
            finite_snr = batch["snr_db"][torch.isfinite(batch["snr_db"])]
            snr_counts["native"] += int((~torch.isfinite(batch["snr_db"]) & batch["is_event"]).sum())
            snr_counts["-10_0"] += int(((finite_snr >= -10) & (finite_snr < 0)).sum())
            snr_counts["0_10"] += int(((finite_snr >= 0) & (finite_snr < 10)).sum())
            snr_counts["10_20"] += int(((finite_snr >= 10) & (finite_snr <= 20)).sum())
            snr_counts["20_30"] += int(((finite_snr > 20) & (finite_snr <= 30)).sum())
            if batch_index % 50 == 0 or batch_index == len(train_loader):
                mean_loss = totals["loss"] / batch_index
                print(
                    f"epoch {epoch}/{args.epochs} batch {batch_index}/{len(train_loader)} "
                    f"loss={mean_loss:.4f}",
                    flush=True,
                )

        scheduler.step()
        predictions = collect_predictions(model, dev_loader, device)
        threshold, dev_metrics, threshold_sweep = select_threshold(predictions)
        batches = max(len(train_loader), 1)
        row = {
            "epoch": epoch,
            "seconds": time.time() - epoch_start,
            "lr": optimizer.param_groups[0]["lr"],
            "loss": {key: value / batches for key, value in totals.items()},
            "snr_counts": snr_counts,
            "threshold": threshold,
            "dev": dev_metrics,
        }
        history.append(row)
        print(json.dumps(row, indent=2), flush=True)

        checkpoint = {
            "model": model.state_dict(),
            "model_config": model.config.as_dict(),
            "training_config": vars(args) | {"knet_dir": str(args.knet_dir), "save_dir": str(args.save_dir)},
            "protocol": protocol,
            "epoch": epoch,
            "threshold": threshold,
            "dev_metrics": dev_metrics,
            "parameter_count": parameter_count,
        }
        torch.save(checkpoint, args.save_dir / f"epoch_{epoch:02d}.pt")
        if dev_metrics["score"] > best_score:
            best_score = dev_metrics["score"]
            torch.save(checkpoint, args.save_dir / "best.pt")
            save_json(args.save_dir / "best_threshold_sweep.json", threshold_sweep)
        save_json(args.save_dir / "history.json", history)

    print(f"Training complete. Best dev score={best_score:.4f}")
    print(f"Best checkpoint: {args.save_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
