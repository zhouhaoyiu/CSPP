"""Reuse both V10 packet embeddings without a second raw-packet encoder pass."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch


@dataclass(frozen=True)
class ReusedPacketScores:
    p_probability: np.ndarray
    event_probability: np.ndarray
    arrival_age_fraction: np.ndarray
    base_embedding: np.ndarray
    target_embedding: np.ndarray
    encoder_batches: int
    temporal_blocks: int
    context_packets: int


def _validate_model(model: torch.nn.Module) -> None:
    required = (
        "_encode_packet_features",
        "_encode_target_packet_features",
        "_feature_step",
        "_target_feature_step",
        "change_projection",
        "target_change_projection",
        "forward_embeddings",
        "forward_target_embeddings",
        "receptive_field_packets",
        "target_cache_packets",
    )
    missing = [name for name in required if not hasattr(model, name)]
    if missing:
        raise ValueError(f"V15 requires a V10 packet-once model: missing={missing}")


def _as_packets(packets: np.ndarray) -> np.ndarray:
    values = np.asarray(packets, dtype=np.float32)
    if values.ndim != 3 or values.shape[1:] != (3, 50):
        raise ValueError(f"Expected packets (T, 3, 50), got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("Packets contain non-finite values")
    return values


@torch.inference_mode()
def encode_packet_embeddings_once(
    model: torch.nn.Module,
    packets: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Encode disjoint packet batches once through each native V10 branch."""
    _validate_model(model)
    values = _as_packets(packets)
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    base_parts = []
    target_parts = []
    batches = 0
    model.eval()
    for start in range(0, len(values), batch_size):
        tensor = torch.from_numpy(values[start : start + batch_size]).to(device)
        base_parts.append(
            model._encode_packet_features(tensor).detach().cpu().numpy()
        )
        target_parts.append(
            model._encode_target_packet_features(tensor).detach().cpu().numpy()
        )
        batches += 1
    if not base_parts:
        base_width = int(model.config.embedding_dim)
        target_width = int(model.config.target_embedding_dim)
        return (
            np.empty((0, base_width), dtype=np.float32),
            np.empty((0, target_width), dtype=np.float32),
            0,
        )
    base = np.concatenate(base_parts).astype(np.float32, copy=False)
    target = np.concatenate(target_parts).astype(np.float32, copy=False)
    if not np.isfinite(base).all() or not np.isfinite(target).all():
        raise RuntimeError("V15 packet embeddings contain non-finite values")
    return base, target, batches


def _project_change_sequence(
    projection: torch.nn.Module,
    embeddings: np.ndarray,
    previous: np.ndarray | None,
    device: torch.device,
) -> torch.Tensor:
    current = torch.from_numpy(
        np.asarray(embeddings, dtype=np.float32)
    ).to(device)
    if len(current) == 0:
        raise ValueError("Cannot project an empty embedding sequence")
    if previous is None:
        prior = torch.cat((torch.zeros_like(current[:1]), current[:-1]), dim=0)
    else:
        previous_tensor = torch.from_numpy(
            np.asarray(previous, dtype=np.float32).reshape(1, -1)
        ).to(device)
        prior = torch.cat((previous_tensor, current[:-1]), dim=0)
    delta = (current - prior) / math.sqrt(2.0)
    if previous is None:
        delta[0] = 0.0
    return projection(torch.cat((current, delta), dim=1))


@torch.inference_mode()
def score_reused_embeddings_blocked(
    model: torch.nn.Module,
    base_embedding: np.ndarray,
    target_embedding: np.ndarray,
    *,
    device: torch.device,
    block_packets: int,
    context_packets: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Score a reset-free sequence from cached embeddings with causal overlap."""
    _validate_model(model)
    base = np.asarray(base_embedding, dtype=np.float32)
    target = np.asarray(target_embedding, dtype=np.float32)
    if base.ndim != 2 or target.ndim != 2 or len(base) != len(target):
        raise ValueError("V15 embedding arrays have incompatible shapes")
    if block_packets < 1:
        raise ValueError("block_packets must be positive")
    minimum_context = max(
        int(model.receptive_field_packets),
        int(model.target_cache_packets) + 2,
    )
    context = minimum_context if context_packets is None else int(context_packets)
    if context < minimum_context:
        raise ValueError(
            f"context_packets={context} is shorter than {minimum_context}"
        )
    if len(base) == 0:
        empty = np.empty(0, dtype=np.float32)
        return empty, empty.copy(), empty.copy(), 0, context

    p_parts = []
    event_parts = []
    delay_parts = []
    blocks = 0
    model.eval()
    for start in range(0, len(base), block_packets):
        stop = min(start + block_packets, len(base))
        context_start = max(0, start - context)
        base_previous = (
            None if context_start == 0 else base[context_start - 1]
        )
        target_previous = (
            None if context_start == 0 else target[context_start - 1]
        )
        base_sequence = _project_change_sequence(
            model.change_projection,
            base[context_start:stop],
            base_previous,
            device,
        )
        target_sequence = _project_change_sequence(
            model.target_change_projection,
            target[context_start:stop],
            target_previous,
            device,
        )
        base_outputs = model.forward_embeddings(base_sequence.unsqueeze(0))
        target_event = model.forward_target_embeddings(
            target_sequence.unsqueeze(0)
        )
        discard = start - context_start
        p_parts.append(
            torch.sigmoid(base_outputs[0][0, discard:])
            .detach()
            .cpu()
            .numpy()
        )
        event_parts.append(
            torch.sigmoid(target_event[0, discard:])
            .detach()
            .cpu()
            .numpy()
        )
        delay_parts.append(
            torch.sigmoid(base_outputs[2][0, discard:])
            .detach()
            .cpu()
            .numpy()
        )
        blocks += 1
    arrays = [
        np.concatenate(parts).astype(np.float32, copy=False)
        for parts in (p_parts, event_parts, delay_parts)
    ]
    if any(len(values) != len(base) for values in arrays):
        raise RuntimeError("V15 blocked score length differs from embeddings")
    if not all(np.isfinite(values).all() for values in arrays):
        raise RuntimeError("V15 blocked scores contain non-finite values")
    return arrays[0], arrays[1], arrays[2], blocks, context


def score_packets_with_reused_embeddings(
    model: torch.nn.Module,
    packets: np.ndarray,
    *,
    device: torch.device,
    embedding_batch_size: int,
    block_packets: int,
    context_packets: int | None = None,
) -> ReusedPacketScores:
    base, target, encoder_batches = encode_packet_embeddings_once(
        model,
        packets,
        device=device,
        batch_size=embedding_batch_size,
    )
    p, event, delay, temporal_blocks, context = (
        score_reused_embeddings_blocked(
            model,
            base,
            target,
            device=device,
            block_packets=block_packets,
            context_packets=context_packets,
        )
    )
    return ReusedPacketScores(
        p_probability=p,
        event_probability=event,
        arrival_age_fraction=delay,
        base_embedding=base,
        target_embedding=target,
        encoder_batches=encoder_batches,
        temporal_blocks=temporal_blocks,
        context_packets=context,
    )


def _change_projection_batch(
    projection: torch.nn.Module,
    embeddings: np.ndarray,
    descriptors: list[tuple[int, int, int, int, int, int]],
    *,
    device: torch.device,
) -> torch.Tensor:
    width = int(embeddings.shape[1])
    maximum_length = max(stop - context_start for _, context_start, stop, _, _, _ in descriptors)
    inputs = np.zeros(
        (len(descriptors), maximum_length, 2 * width),
        dtype=np.float32,
    )
    scale = np.float32(math.sqrt(2.0))
    for row, descriptor in enumerate(descriptors):
        (
            segment_start,
            context_start,
            stop,
            _,
            _,
            _,
        ) = descriptor
        current = embeddings[context_start:stop]
        length = len(current)
        inputs[row, :length, :width] = current
        delta = inputs[row, :length, width:]
        if context_start > segment_start:
            delta[0] = (
                current[0] - embeddings[context_start - 1]
            ) / scale
        if length > 1:
            delta[1:] = (current[1:] - current[:-1]) / scale
    tensor = torch.from_numpy(inputs).to(device)
    return projection(tensor)


def _segmented_block_descriptors(
    segment_lengths: np.ndarray,
    *,
    block_packets: int,
    context_packets: int,
) -> list[tuple[int, int, int, int, int, int]]:
    descriptors = []
    segment_start = 0
    for value in segment_lengths:
        length = int(value)
        for local_start in range(0, length, block_packets):
            local_stop = min(local_start + block_packets, length)
            local_context = max(0, local_start - context_packets)
            descriptors.append(
                (
                    segment_start,
                    segment_start + local_context,
                    segment_start + local_stop,
                    local_start - local_context,
                    segment_start + local_start,
                    segment_start + local_stop,
                )
            )
        segment_start += length
    return descriptors


def _descriptor_batches(
    descriptors: list[tuple[int, int, int, int, int, int]],
    maximum_padded_packets: int,
) -> list[list[tuple[int, int, int, int, int, int]]]:
    batches = []
    current: list[tuple[int, int, int, int, int, int]] = []
    maximum_length = 0
    for descriptor in descriptors:
        length = descriptor[2] - descriptor[1]
        proposed_maximum = max(maximum_length, length)
        if (
            current
            and proposed_maximum * (len(current) + 1)
            > maximum_padded_packets
        ):
            batches.append(current)
            current = []
            maximum_length = 0
        current.append(descriptor)
        maximum_length = max(maximum_length, length)
    if current:
        batches.append(current)
    return batches


@torch.inference_mode()
def score_segmented_packets_with_reused_embeddings(
    model: torch.nn.Module,
    packets: np.ndarray,
    segment_lengths: np.ndarray,
    *,
    device: torch.device,
    embedding_batch_size: int,
    block_packets: int,
    maximum_padded_packets_per_temporal_batch: int,
    context_packets: int | None = None,
) -> ReusedPacketScores:
    """Score reset-separated packet segments with shared encoder batches.

    Packet encoders are stateless, so all valid packets are encoded together.
    Temporal blocks from different segments are batched as independent rows;
    every segment therefore keeps the same zero-state boundary as a separate
    call while avoiding allocator growth from hundreds of tiny MPS calls.
    """
    _validate_model(model)
    values = _as_packets(packets)
    lengths = np.asarray(segment_lengths, dtype=np.int64)
    if lengths.ndim != 1 or np.any(lengths <= 0):
        raise ValueError("segment_lengths must be a non-empty positive vector")
    if int(lengths.sum()) != len(values):
        raise ValueError("segment_lengths do not cover all packets")
    if block_packets < 1:
        raise ValueError("block_packets must be positive")
    minimum_context = max(
        int(model.receptive_field_packets),
        int(model.target_cache_packets) + 2,
    )
    context = minimum_context if context_packets is None else int(context_packets)
    if context < minimum_context:
        raise ValueError(
            f"context_packets={context} is shorter than {minimum_context}"
        )
    if maximum_padded_packets_per_temporal_batch < 1:
        raise ValueError(
            "maximum_padded_packets_per_temporal_batch must be positive"
        )

    base, target, encoder_batches = encode_packet_embeddings_once(
        model,
        values,
        device=device,
        batch_size=embedding_batch_size,
    )
    descriptors = _segmented_block_descriptors(
        lengths,
        block_packets=block_packets,
        context_packets=context,
    )
    batches = _descriptor_batches(
        descriptors,
        maximum_padded_packets_per_temporal_batch,
    )
    p = np.empty(len(values), dtype=np.float32)
    event = np.empty(len(values), dtype=np.float32)
    delay = np.empty(len(values), dtype=np.float32)
    model.eval()
    for batch in batches:
        base_sequence = _change_projection_batch(
            model.change_projection,
            base,
            batch,
            device=device,
        )
        target_sequence = _change_projection_batch(
            model.target_change_projection,
            target,
            batch,
            device=device,
        )
        base_outputs = model.forward_embeddings(base_sequence)
        target_event = model.forward_target_embeddings(target_sequence)
        p_batch = torch.sigmoid(base_outputs[0]).detach().cpu().numpy()
        event_batch = torch.sigmoid(target_event).detach().cpu().numpy()
        delay_batch = torch.sigmoid(base_outputs[2]).detach().cpu().numpy()
        for row, descriptor in enumerate(batch):
            _, _, _, discard, output_start, output_stop = descriptor
            length = output_stop - output_start
            selected = slice(discard, discard + length)
            p[output_start:output_stop] = p_batch[row, selected]
            event[output_start:output_stop] = event_batch[row, selected]
            delay[output_start:output_stop] = delay_batch[row, selected]
    arrays = (p, event, delay)
    if not all(np.isfinite(values).all() for values in arrays):
        raise RuntimeError("V15 segmented scores contain non-finite values")
    return ReusedPacketScores(
        p_probability=p,
        event_probability=event,
        arrival_age_fraction=delay,
        base_embedding=base,
        target_embedding=target,
        encoder_batches=encoder_batches,
        temporal_blocks=len(descriptors),
        context_packets=context,
    )


@torch.inference_mode()
def streaming_step_with_reused_embeddings(
    model: torch.nn.Module,
    packet: torch.Tensor,
    state: dict | None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    dict,
]:
    """Exact V10 streaming step exposing both embeddings from the same pass."""
    _validate_model(model)
    if packet.ndim == 2:
        packet = packet.unsqueeze(0)
    if packet.ndim != 3 or packet.shape[0] != 1:
        raise ValueError("Streaming inference expects (3, 50) or (1, 3, 50)")
    base_state = None if state is None else state.get("backbone")
    target_state = None if state is None else state.get("target_verifier")
    base_embedding = model._encode_packet_features(packet)
    p, _, delay, next_base = model._feature_step(
        base_embedding, base_state
    )
    target_embedding = model._encode_target_packet_features(packet)
    event, next_target = model._target_feature_step(
        target_embedding, target_state
    )
    return (
        p[0],
        event[0],
        delay[0],
        base_embedding[0],
        target_embedding[0],
        {"backbone": next_base, "target_verifier": next_target},
    )


__all__ = [
    "ReusedPacketScores",
    "encode_packet_embeddings_once",
    "score_packets_with_reused_embeddings",
    "score_reused_embeddings_blocked",
    "score_segmented_packets_with_reused_embeddings",
    "streaming_step_with_reused_embeddings",
]
