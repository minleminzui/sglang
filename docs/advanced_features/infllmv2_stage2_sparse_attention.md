### InfLLM v2 Stage-2 Sparse Attention (Triton)

本页说明 `Stage-2` 稀疏注意力的 Triton 内核使用方法与接口契约。

#### 文件位置
- `python/sglang/srt/layers/attention/triton_ops/infllmv2_stage2.py`

#### 接口
```python
from sglang.srt.layers.attention.triton_ops.infllmv2_stage2 import (
    InfLLM2Config,
    infllmv2_sparse_attn_fwd,
)

cfg = InfLLM2Config(
    topk=8,
    block_size=64,
    BLOCK_SQ=64,
    BLOCK_K=64,
    num_warps=4,
    num_stages=2,
)

out = infllmv2_sparse_attn_fwd(
    q,          # [B, H, SQ, D]
    k_all,      # [B, HK, SC, D]
    v_all,      # [B, HK, SC, D]
    topk_idx,   # [B, HK, SQ, K]（block 索引，单位=64 tokens）
    cfg,
    sink_bias=None,  # 可选，可广播到 [B, HK, SQ, K]
    causal=False,    # Week‑1：decode SQ=1 或上游已保证因果
)
# out: [B, H, SQ, D]
```

#### 关键约束
- 头分组：`H = HK * HG`，同一 `HK` 下 `HG` 个 `q` head 共享同一 `topk_idx`。
- `topk_idx` 的单位是 block（`block_size=64`），即 `k_start = topk_idx * 64`。
- `sink_bias`（若提供）在逐 block 的累加前加到 score 上，形状可广播到 `[B, HK, SQ, K]`。
- 因果：Week‑1 提供两种安全路径（decode 单 token 或上游已裁剪未来 token 的 blocks）。

#### 基准与单测
- 基准：`benchmark/bench_infllmv2_sparse_stage2.py`
  - 随机输入，比较 Triton 内核与 dense 参考实现的数值误差与耗时。
- 单测：`test/srt/test_infllmv2_stage2.py`
  - fp16 下相对 L2 误差 < 1e-2。

#### 注意
- 当前实现主要面向正确性验证，性能调优（tile、warps、stages）放在 Week‑2。
- CUDA/HIP 环境下运行；若本机无 GPU，可阅读脚本代码与接口作为参考。


