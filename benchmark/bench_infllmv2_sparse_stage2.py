from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from sglang.srt.layers.attention.triton_ops.infllmv2_stage2 import (
    InfLLM2Config,
    infllmv2_sparse_attn_fwd,
)


@dataclass
class BenchCfg:
    B: int = 2
    H: int = 8
    HK: int = 2
    D: int = 128
    SQ: int = 256
    SC: int = 4096
    TOPK: int = 8
    BLOCK: int = 64
    dtype: torch.dtype = torch.float16


def build_inputs(cfg: BenchCfg, device: torch.device):
    torch.manual_seed(0)
    B, H, HK, D, SQ, SC = cfg.B, cfg.H, cfg.HK, cfg.D, cfg.SQ, cfg.SC
    assert H % HK == 0
    q = torch.randn(B, H, SQ, D, device=device, dtype=cfg.dtype)
    k = torch.randn(B, HK, SC, D, device=device, dtype=cfg.dtype)
    v = torch.randn_like(k)

    num_blocks = SC // cfg.BLOCK
    topk = torch.randint(0, num_blocks, (B, HK, SQ, cfg.TOPK), device=device, dtype=torch.int32)
    return q, k, v, topk


@torch.no_grad()
def dense_reference(q, k, v, topk, block: int):
    # q: [B,H,SQ,D], k/v: [B,HK,SC,D], topk: [B,HK,SQ,K]
    B, H, SQ, D = q.shape
    Bk, HK, SC, _ = k.shape
    assert B == Bk and H % HK == 0
    HG = H // HK
    K = topk.shape[-1]

    out = torch.empty_like(q)
    for b in range(B):
        for h in range(H):
            hk = h // HG
            for s in range(SQ):
                idx_blocks = topk[b, hk, s]  # [K]
                token_idx = (idx_blocks[:, None] * block + torch.arange(block, device=q.device)[None, :]).flatten()
                token_idx = token_idx[token_idx < SC]
                k_sel = k[b, hk, token_idx]  # [T, D]
                v_sel = v[b, hk, token_idx]
                scores = (q[b, h, s].to(torch.float32) @ k_sel.to(torch.float32).T) / (D ** 0.5)
                w = torch.softmax(scores, dim=-1)
                out[b, h, s] = (w @ v_sel.to(torch.float32)).to(q.dtype)
    return out


def benchmark_once(device: torch.device, cfg: BenchCfg):
    q, k, v, topk = build_inputs(cfg, device)
    kern_cfg = InfLLM2Config(
        topk=cfg.TOPK,
        block_size=cfg.BLOCK,
        dtype=cfg.dtype,
        BLOCK_SQ=64,
        BLOCK_K=cfg.BLOCK,
        num_warps=4,
        num_stages=2,
    )

    # warmup
    for _ in range(3):
        _ = infllmv2_sparse_attn_fwd(q, k, v, topk, kern_cfg, sink_bias=None, causal=False)
    torch.cuda.synchronize()

    # time kernel
    iters = 10
    t0 = time.time()
    for _ in range(iters):
        out = infllmv2_sparse_attn_fwd(q, k, v, topk, kern_cfg, sink_bias=None, causal=False)
    torch.cuda.synchronize()
    t1 = time.time()

    # correctness vs dense reference (small shapes recommended)
    with torch.no_grad():
        ref = dense_reference(q, k, v, topk, cfg.BLOCK)
        l2 = torch.linalg.vector_norm((out - ref).float()) / torch.linalg.vector_norm(ref.float())
    print(f"time/iter: {(t1 - t0)/iters*1000:.3f} ms, rel L2: {l2.item():.3e}")


if __name__ == "__main__":
    device = torch.device("cuda")
    bench_cfg = BenchCfg(SQ=128, SC=2048, TOPK=4)  # smaller default for quick check
    benchmark_once(device, bench_cfg)


