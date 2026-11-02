"""流式 P 波捡拾：与近实时数据包一致的超参数。"""
# 导师描述：约 0.5 s 一个数据包（与 100 Hz × 0.5 = 50 点一致）
CHUNK_SEC = 0.5
TARGET_SR = 100.0
SAMPLES_PER_CHUNK = int(CHUNK_SEC * TARGET_SR)

# 【修复】InstanceGM 兼容：优先匹配大写P列名，兼容多数据集
P_ARRIVAL_COL_CANDIDATES = [
    "trace_P_arrival_sample",  # InstanceGM 官方列名（大写P）
    "trace_p_arrival_sample",  # Iquique 等数据集兼容（小写p）
    "trace_P_arrival"          # 秒单位列名兜底
]
# 默认主列名，运行时自动检测适配
P_ARRIVAL_COL = P_ARRIVAL_COL_CANDIDATES[0]