from __future__ import annotations

import torch
import pytest

from sglang.srt.layers.attention.triton_ops.infllmv2_stage2 import (
    InfLLM2Config,
    infllmv2_sparse_attn_fwd,
)


def dense_reference(q, k, v, topk, block: int):
    B, H, SQ, D = q.shape
    Bk, HK, SC, _ = k.shape
    assert B == Bk and H % HK == 0
    HG = H // HK
    out = torch.empty_like(q)
    for b in range(B):
        for h in range(H):
            hk = h // HG
            for s in range(SQ):
                idx_blocks = topk[b, hk, s]  # [K]
                token_idx = (idx_blocks[:, None] * block + torch.arange(block, device=q.device)[None, :]).flatten()
                token_idx = token_idx[token_idx < k.shape[2]]
                k_sel = k[b, hk, token_idx]
                v_sel = v[b, hk, token_idx]
                scores = (q[b, h, s].to(torch.float32) @ k_sel.to(torch.float32).T) / (D ** 0.5)
                w = torch.softmax(scores, dim=-1)
                out[b, h, s] = (w @ v_sel.to(torch.float32)).to(q.dtype)
    return out


@pytest.mark.parametrize("B,H,HK,D,SQ,SC,TOPK,BLK", [
    (2, 8, 2, 64, 64, 1024, 4, 64),
])
def test_sparse_stage2_fp16_close_to_dense(B, H, HK, D, SQ, SC, TOPK, BLK):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    torch.manual_seed(0)
    device = torch.device("cuda")
    q = torch.randn(B, H, SQ, D, device=device, dtype=torch.float16)
    k = torch.randn(B, HK, SC, D, device=device, dtype=torch.float16)
    v = torch.randn_like(k)
    num_blocks = SC // BLK
    topk = torch.randint(0, num_blocks, (B, HK, SQ, TOPK), device=device, dtype=torch.int32)
    cfg = InfLLM2Config(topk=TOPK, block_size=BLK, BLOCK_SQ=64, BLOCK_K=BLK)

    out = infllmv2_sparse_attn_fwd(q, k, v, topk, cfg, sink_bias=None, causal=False)
    ref = dense_reference(q, k, v, topk, BLK)
    rel_l2 = torch.linalg.vector_norm((out - ref).float()) / torch.linalg.vector_norm(ref.float())
    assert rel_l2.item() < 1e-2


