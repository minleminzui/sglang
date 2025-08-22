from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import triton
import triton.language as tl


@dataclass
class InfLLM2Config:
    topk: int
    block_size: int = 64
    use_triton: bool = True
    dtype: torch.dtype = torch.float16
    num_warps: int = 4
    num_stages: int = 2
    BLOCK_SQ: int = 64
    BLOCK_K: int = 64


@triton.jit
def _sparse_attn_fwd_kernel(
    Q_ptr,  # [B, H, SQ, D]
    K_ptr,  # [B, HK, SC, D]
    V_ptr,  # [B, HK, SC, D]
    IDX_ptr,  # [B, HK, SQ, TOPK] (block indices)
    O_ptr,  # [B, H, SQ, D]
    Bias_ptr,  # optional, may be 0
    B: tl.constexpr,
    H: tl.constexpr,
    HK: tl.constexpr,
    SQ: tl.constexpr,
    SC: tl.constexpr,
    D: tl.constexpr,
    TOPK: tl.constexpr,
    HG: tl.constexpr,
    BLOCK_SQ: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
    sm_scale,
    causal: tl.constexpr,
    stride_q_b: tl.constexpr,
    stride_q_h: tl.constexpr,
    stride_q_s: tl.constexpr,
    stride_q_d: tl.constexpr,
    stride_k_b: tl.constexpr,
    stride_k_h: tl.constexpr,
    stride_k_s: tl.constexpr,
    stride_k_d: tl.constexpr,
    stride_v_b: tl.constexpr,
    stride_v_h: tl.constexpr,
    stride_v_s: tl.constexpr,
    stride_v_d: tl.constexpr,
    stride_idx_b: tl.constexpr,
    stride_idx_h: tl.constexpr,
    stride_idx_s: tl.constexpr,
    stride_idx_k: tl.constexpr,
    stride_o_b: tl.constexpr,
    stride_o_h: tl.constexpr,
    stride_o_s: tl.constexpr,
    stride_o_d: tl.constexpr,
    stride_bias_b: tl.constexpr,
    stride_bias_h: tl.constexpr,
    stride_bias_s: tl.constexpr,
    stride_bias_k: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_s = tl.program_id(1)

    b = pid_bh // H
    h = pid_bh % H
    # map q-head to kv-head by hg grouping
    hk = h // HG

    sq_start = pid_s * BLOCK_SQ
    q_rows = sq_start + tl.arange(0, BLOCK_SQ)
    q_mask = q_rows < SQ

    d_cols = tl.arange(0, BLOCK_D)
    d_mask = d_cols < D

    # pointers to Q and O tiles
    q_tile_ptr = Q_ptr + (
        b * stride_q_b + h * stride_q_h + q_rows[:, None] * stride_q_s + d_cols[None, :] * stride_q_d
    )
    o_tile_ptr = O_ptr + (
        b * stride_o_b + h * stride_o_h + q_rows[:, None] * stride_o_s + d_cols[None, :] * stride_o_d
    )

    # load Q tile [BLOCK_SQ, D]
    q_tile = tl.load(q_tile_ptr, mask=q_mask[:, None] & d_mask[None, :], other=0.0)

    # online softmax state per query row
    m_i = tl.full((BLOCK_SQ,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_SQ,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_SQ, BLOCK_D), dtype=tl.float32)

    # base ptrs for K/V/IDX/Bias of this (b, hk)
    idx_rows_ptr = IDX_ptr + b * stride_idx_b + hk * stride_idx_h + q_rows[:, None] * stride_idx_s
    has_bias = Bias_ptr != 0
    if has_bias:
        bias_rows_ptr = Bias_ptr + b * stride_bias_b + hk * stride_bias_h + q_rows[:, None] * stride_bias_s

    for i in range(TOPK):
        blk_idx = tl.load(idx_rows_ptr + i * stride_idx_k, mask=q_mask, other=0)
        # token indices of this block
        k_token_idx = blk_idx[:, None] * BLOCK_K + tl.arange(0, BLOCK_K)[None, :]
        k_token_mask = k_token_idx < SC

        # gather K/V blocks: [BLOCK_K, D]
        k_block_ptr = K_ptr + (
            b * stride_k_b + hk * stride_k_h + k_token_idx[:, :, None] * stride_k_s + d_cols[None, None, :] * stride_k_d
        )
        v_block_ptr = V_ptr + (
            b * stride_v_b + hk * stride_v_h + k_token_idx[:, :, None] * stride_v_s + d_cols[None, None, :] * stride_v_d
        )

        # load blocks -> [BLOCK_SQ, BLOCK_K, D]
        k_block = tl.load(k_block_ptr, mask=k_token_mask[:, :, None] & d_mask[None, None, :], other=0.0)
        v_block = tl.load(v_block_ptr, mask=k_token_mask[:, :, None] & d_mask[None, None, :], other=0.0)

        # attention scores per row: [BLOCK_SQ, BLOCK_K]
        # scores[r, :] = (q_tile[r] * k_block[r]).sum(axis=-1)
        q_expanded = q_tile[:, None, :]
        scores = tl.sum(q_expanded.to(tl.float32) * k_block.to(tl.float32), axis=2) * sm_scale

        if has_bias:
            bias_vals = tl.load(bias_rows_ptr + i * stride_bias_k, mask=q_mask, other=0.0)
            scores = scores + bias_vals[:, None]

        # causal handling (Week-1 safe paths): when decode (SQ==1) or IDX already causal
        if causal:
            # No-op here; assume upstream ensures indices are causal
            pass

        # masked scores for out-of-range k tokens (per row mask)
        scores = tl.where(k_token_mask, scores, -float("inf"))

        # online softmax update over blocks
        block_max = tl.max(scores, axis=1)
        new_m = tl.maximum(m_i, block_max)
        # exp(prev)
        exp_scale_prev = tl.exp(m_i - new_m)
        # exp(current)
        scores_shifted = scores - new_m[:, None]
        exp_scores = tl.exp(scores_shifted)
        block_l = tl.sum(exp_scores, axis=1)
        new_l = l_i * exp_scale_prev + block_l

        # update accumulator
        # prev contribution
        acc = acc * exp_scale_prev[:, None]
        # current contribution: sum_j exp_scores[:, j] * v_block[:, j, :]
        acc = acc + tl.sum(exp_scores[:, :, None].to(tl.float32) * v_block.to(tl.float32), axis=1)

        # commit new state
        m_i = new_m
        l_i = new_l

    # normalize
    out = acc / l_i[:, None]

    # store O tile
    tl.store(o_tile_ptr, out.to(q_tile.dtype), mask=q_mask[:, None] & d_mask[None, :])


def _compute_strides(t: torch.Tensor) -> tuple[int, int, int, int]:
    s = t.stride()
    # ensure 4D tensor strides (B, H, S, D)
    if len(s) != 4:
        raise ValueError("Expected a 4D tensor with layout [B, H, S, D].")
    return tuple(int(x) for x in s)


def infllmv2_sparse_attn_fwd(
    q: torch.Tensor,  # [B, H, SQ, D]
    k_all: torch.Tensor,  # [B, HK, SC, D]
    v_all: torch.Tensor,  # [B, HK, SC, D]
    topk_idx: torch.Tensor,  # [B, HK, SQ, K] (block indices)
    cfg: InfLLM2Config,
    sink_bias: Optional[torch.Tensor] = None,  # broadcastable to [B, HK, SQ, K]
    causal: bool = False,
) -> torch.Tensor:
    """
    Sparse attention forward over selected K/V blocks per kv-head.

    Returns: [B, H, SQ, D]
    """
    assert q.is_cuda and k_all.is_cuda and v_all.is_cuda and topk_idx.is_cuda, "Inputs must be CUDA tensors."
    B, H, SQ, D = q.shape
    Bk, HK, SC, Dk = k_all.shape
    assert B == Bk and D == Dk, "Shape mismatch between q and k_all/v_all."
    assert v_all.shape == k_all.shape, "k_all and v_all must have the same shape."
    assert topk_idx.shape[0] == B and topk_idx.shape[1] == HK and topk_idx.shape[2] == SQ, "topk_idx shape mismatch."
    TOPK = topk_idx.shape[3]
    assert H % HK == 0, "H must be divisible by HK to infer head-group size (HG)."
    HG = H // HK

    block_size = int(cfg.block_size)
    assert block_size == cfg.BLOCK_K, "cfg.block_size must equal cfg.BLOCK_K for now."

    if sink_bias is not None:
        # Make sure it is on device and broadcastable; create a contiguous copy for indexing
        sink_bias = sink_bias.to(q.device)
        # We will pass an actual pointer only if exact shape matches [B, HK, SQ, TOPK]
        if list(sink_bias.shape) != [B, HK, SQ, TOPK]:
            sink_bias = sink_bias.expand(B, HK, SQ, TOPK).contiguous()

    # allocate output
    o = torch.empty_like(q)

    # scale factor
    sm_scale = 1.0 / (D ** 0.5)

    # strides
    stride_q_b, stride_q_h, stride_q_s, stride_q_d = _compute_strides(q)
    stride_k_b, stride_k_h, stride_k_s, stride_k_d = _compute_strides(k_all)
    stride_v_b, stride_v_h, stride_v_s, stride_v_d = _compute_strides(v_all)
    stride_o_b, stride_o_h, stride_o_s, stride_o_d = _compute_strides(o)
    stride_idx_b, stride_idx_h, stride_idx_s, stride_idx_k = _compute_strides(topk_idx)

    if sink_bias is None:
        bias_ptr = tl.pointer_type[tl.float32](0)
        stride_bias_b = stride_bias_h = stride_bias_s = stride_bias_k = 0
    else:
        stride_bias_b, stride_bias_h, stride_bias_s, stride_bias_k = _compute_strides(sink_bias)
        bias_ptr = sink_bias

    grid = (B * H, triton.cdiv(SQ, cfg.BLOCK_SQ))

    _sparse_attn_fwd_kernel[
        grid
    ](
        q,
        k_all,
        v_all,
        topk_idx,
        o,
        bias_ptr if sink_bias is not None else 0,
        B,
        H,
        HK,
        SQ,
        SC,
        D,
        TOPK,
        HG,
        cfg.BLOCK_SQ,
        cfg.block_size,
        D,
        sm_scale,
        causal,
        stride_q_b,
        stride_q_h,
        stride_q_s,
        stride_q_d,
        stride_k_b,
        stride_k_h,
        stride_k_s,
        stride_k_d,
        stride_v_b,
        stride_v_h,
        stride_v_s,
        stride_v_d,
        stride_idx_b,
        stride_idx_h,
        stride_idx_s,
        stride_idx_k,
        stride_o_b,
        stride_o_h,
        stride_o_s,
        stride_o_d,
        stride_bias_b,
        stride_bias_h,
        stride_bias_s,
        stride_bias_k,
        num_warps=cfg.num_warps,
        num_stages=cfg.num_stages,
    )

    return o


