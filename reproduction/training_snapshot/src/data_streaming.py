"""data_streaming.py 累积归一化版 — 修复per-chunk归一化丢失振幅信息的问题

=== 修复记录 ===
[Fix-1] normalize_packet_causal: 加入 valid_samples 参数，补零部分不计入累积 RMS
[Fix-2] Welford 加权在线平均公式：添加详细注释说明正确性
"""
from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import Dataset
from config import (
    P_ARRIVAL_COL_CANDIDATES,
    SAMPLES_PER_CHUNK,
    TARGET_SR,
)

def normalize_packet_causal(
    wave_chunk: np.ndarray,
    running_stats: dict | None = None,
    valid_samples: int | None = None,
) -> tuple[np.ndarray, dict]:
    """
    累积归一化（因果安全）：
    - 减去当前包均值（去DC）
    - 除以累积RMS（保留振幅比例：P波包的振幅 > 背景包的振幅）

    running_stats: {"sum_sq": float, "count": int} 跨包累积统计量
    返回: (归一化后的chunk, 更新后的running_stats)

    因果保证：只用当前包及之前的数据，不偷看未来

    [Fix-1] valid_samples: 当前 chunk 中有效（非补零）的样本数。
        默认 None 表示全部有效。补零的最后一个 chunk 应传入实际样本数，
        避免零填充拉低累积 RMS。
    """
    if running_stats is None:
        running_stats = {"sum_sq": 0.0, "count": 0}

    # 1. 去均值（per-chunk，因果安全）
    m = wave_chunk.mean(axis=1, keepdims=True)
    centered = wave_chunk - m

    # 2. 更新累积RMS统计量
    #
    # [Fix-2] 加权在线平均公式说明：
    #
    #   new_sum_sq = old_sum_sq + (chunk_energy - old_sum_sq) * n / new_count
    #
    # 这是加权在线平均的正确实现，展开等价于：
    #   new_sum_sq = (old_sum_sq * old_count + chunk_energy * n) / new_count
    #
    # 证明：令 old_count = c, n = 新样本数, new_count = c + n
    #   old + (chunk_energy - old) * n / (c+n)
    #   = old * (1 - n/(c+n)) + chunk_energy * n/(c+n)
    #   = old * c/(c+n) + chunk_energy * n/(c+n)
    #   = (old * c + chunk_energy * n) / (c+n)  ✅
    #
    n = valid_samples if valid_samples is not None else centered.shape[1]

    # [Fix-1] 只用有效样本计算 chunk_energy
    if valid_samples is not None and valid_samples < centered.shape[1]:
        chunk_energy = float(np.mean(centered[:, :valid_samples] ** 2))
    else:
        chunk_energy = float(np.mean(centered ** 2))

    running_stats["count"] += n
    running_stats["sum_sq"] += (chunk_energy - running_stats["sum_sq"]) * n / running_stats["count"]

    # 3. 累积RMS归一化（关键：P波包振幅大，归一化后仍然比背景包大）
    cumulative_rms = np.sqrt(running_stats["sum_sq"]) + 1e-8
    normalized = centered / cumulative_rms

    return normalized.astype(np.float32), running_stats


def normalize_packet_ewma_causal(
    wave_chunk: np.ndarray,
    running_stats: dict | None = None,
    valid_samples: int | None = None,
    half_life_packets: float = 20.0,
    clip: float = 12.0,
) -> tuple[np.ndarray, dict]:
    """Normalize one packet with a robust scale estimated before the packet.

    The previous EWMA scale is used for the current packet, so an onset keeps
    its amplitude contrast. The current packet updates only future scale.
    Statistics are shared across channels to preserve Z/H amplitude ratios.
    """
    chunk = np.asarray(wave_chunk, dtype=np.float64)
    if chunk.ndim != 2 or chunk.shape[0] != 3:
        raise ValueError(f"Expected wave_chunk shape (3, L), got {chunk.shape}")
    if half_life_packets <= 0:
        raise ValueError("half_life_packets must be positive")

    n_valid = chunk.shape[1] if valid_samples is None else int(valid_samples)
    n_valid = max(0, min(n_valid, chunk.shape[1]))
    if n_valid == 0:
        state = dict(running_stats or {"rms_sq": 1.0, "count": 0})
        return np.zeros_like(chunk, dtype=np.float32), state

    valid = chunk[:, :n_valid]
    centered = chunk - valid.mean(axis=1, keepdims=True)
    valid_centered = centered[:, :n_valid]

    # Winsorized RMS is stable under isolated telemetry spikes while retaining
    # more amplitude information than a per-packet standard deviation.
    abs_values = np.abs(valid_centered).reshape(-1)
    cap = float(np.quantile(abs_values, 0.95))
    cap = max(cap, 1e-8)
    robust_energy = float(np.mean(np.minimum(valid_centered**2, cap**2)))
    robust_energy = max(robust_energy, 1e-16)

    state = dict(running_stats or {})
    initialized = int(state.get("count", 0)) > 0
    previous_energy = max(float(state.get("rms_sq", robust_energy)), 1e-16)
    scale_energy = previous_energy if initialized else robust_energy

    normalized = centered / (np.sqrt(scale_energy) + 1e-8)
    if n_valid < chunk.shape[1]:
        normalized[:, n_valid:] = 0.0
    normalized = np.clip(normalized, -clip, clip)

    alpha = 1.0 - 0.5 ** (1.0 / half_life_packets)
    if initialized:
        # A single bad packet may raise the running scale gradually, but it
        # cannot erase the contrast of the following packets immediately.
        update_energy = min(robust_energy, previous_energy * 16.0)
        next_energy = (1.0 - alpha) * previous_energy + alpha * update_energy
    else:
        next_energy = robust_energy
    state["rms_sq"] = float(max(next_energy, 1e-16))
    state["count"] = int(state.get("count", 0)) + n_valid
    return normalized.astype(np.float32), state


def trace_to_stream_packets(
    wave: np.ndarray,
    p_arrival_sample: float,
    max_chunks: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """
    wave: (3, npts)
    返回 chunks (T, 3, L), labels (T,), T
    label=1 当且仅当 P 初至样本落在该包索引对应的时间窗内。

    使用累积归一化：从第1个包开始逐包累积统计量，
    P波到达后振幅增大 → 归一化后仍然保持较大值
    """
    npts = wave.shape[1]
    spc = SAMPLES_PER_CHUNK
    n_chunks = min((npts + spc - 1) // spc, max_chunks)
    p_idx = int(round(float(p_arrival_sample)))
    chunks = np.zeros((n_chunks, 3, spc), dtype=np.float32)
    labels = np.zeros(n_chunks, dtype=np.float32)

    running_stats = None  # 每条trace独立初始化累积统计量
    for k in range(n_chunks):
        a, b = k * spc, min((k + 1) * spc, npts)
        seg = wave[:, a:b]
        is_padded = (b - a < spc)
        valid_samples = (b - a) if is_padded else None  # [Fix-1]
        if is_padded:
            pad = np.zeros((3, spc - (b - a)), dtype=wave.dtype)
            seg = np.concatenate([seg, pad], axis=1)
        chunks[k], running_stats = normalize_packet_causal(
            seg.astype(np.float64), running_stats,
            valid_samples=valid_samples,  # [Fix-1] 传入有效样本数
        )
        if 0 <= p_idx < npts and a <= p_idx < b:
            labels[k] = 1.0
    return torch.from_numpy(chunks), torch.from_numpy(labels), n_chunks


# ==================== 【核心修复】完整的 Dataset 类 ====================
class InstanceGMStreamPacketDataset(Dataset):
    """
    每条样本 = 一条完整 trace 从开始起的包序列（模拟自记录起点起的实时流）。
    可选：仅保留 P 初至落在前 max_chunks 个包内的事件，避免标签全 0 的无效样本。
    适配 InstanceGM 数据集：直接使用官方预定义 train/dev/test 划分
    """
    def __init__(
        self,
        dataset,
        split: str,
        max_chunks: int = 256,
        indices: list[int] | None = None,
        require_p_in_window: bool = True,
    ):
        self.ds = dataset
        self.max_chunks = max_chunks

        # ========== 自动检测并适配P波到时列名 ==========
        self.p_arrival_col = None
        for col in P_ARRIVAL_COL_CANDIDATES:
            if col in dataset.metadata.columns:
                self.p_arrival_col = col
                break
        if self.p_arrival_col is None:
            raise ValueError(f"未找到P波到时列，候选：{P_ARRIVAL_COL_CANDIDATES}")

        # 如果是秒单位，自动转采样点
        if self.p_arrival_col == "trace_P_arrival":
            print("Detected P arrival in seconds, converting to sample index")
            dataset.metadata["trace_P_arrival_sample"] = (
                dataset.metadata[self.p_arrival_col] * dataset.metadata["trace_sampling_rate_hz"]
            ).round().astype("Int64")
            self.p_arrival_col = "trace_P_arrival_sample"

        print(f"Using P column: {self.p_arrival_col}")

        # 1. 基础筛选
        print(f"   [{split}] Step 1/3: split & notna filter ...", flush=True)
        m = dataset.metadata["split"] == split
        m &= dataset.metadata[self.p_arrival_col].notna()
        cand = np.where(m)[0]
        print(f"   [{split}]   -> {len(cand)} candidates after split filter", flush=True)

        # 2. 处理自定义 indices
        if indices is not None:
            idx_set = set(indices)
            cand = np.array([i for i in cand if i in idx_set], dtype=np.int64)
            print(f"   [{split}]   -> {len(cand)} after indices filter", flush=True)

        # 3. 筛选P波在时间窗内的样本（向量化，避免逐行iloc，200k条从10分钟→秒级）
        if require_p_in_window and len(cand) > 0:
            print(f"   [{split}] Step 2/3: P-in-window filter (vectorized) ...", flush=True)
            spc = SAMPLES_PER_CHUNK
            p_arr = dataset.metadata.iloc[cand][self.p_arrival_col].astype(float).values
            keep_mask = (np.round(p_arr).astype(int) // spc) < max_chunks
            cand = cand[keep_mask]
            print(f"   [{split}]   -> {len(cand)}/{len(keep_mask)} retained", flush=True)

        self.row_indices = cand.astype(np.int64)
        print(f"   [{split}] Step 3/3: done. {len(self.row_indices)} valid traces", flush=True)

    def __len__(self) -> int:
        return len(self.row_indices)

    def __getitem__(self, i: int) -> dict:
        idx = int(self.row_indices[i])
        wave, meta = self.ds.get_sample(idx, sampling_rate=TARGET_SR)
        p = meta[self.p_arrival_col]
        chunks, labels, n_chunks = trace_to_stream_packets(
            np.asarray(wave, dtype=np.float32),
            float(p),
            self.max_chunks,
        )
        return {
            "chunks": chunks,
            "labels": labels,
            "n_chunks": n_chunks,
            "trace_idx": idx,
        }


def collate_stream_batch(batch: list[dict]) -> dict:
    max_t = max(b["chunks"].shape[0] for b in batch)
    bsz = len(batch)
    _, channels, packet_length = batch[0]["chunks"].shape
    chunks = torch.zeros(bsz, max_t, channels, packet_length)
    labels = torch.zeros(bsz, max_t)
    mask = torch.zeros(bsz, max_t)
    meta = []
    for i, b in enumerate(batch):
        t = b["chunks"].shape[0]
        chunks[i, :t] = b["chunks"]
        labels[i, :t] = b["labels"]
        mask[i, :t] = 1.0
        meta.append({"trace_idx": b["trace_idx"], "valid_t": t})
    return {"chunks": chunks, "labels": labels, "mask": mask, "meta": meta}
