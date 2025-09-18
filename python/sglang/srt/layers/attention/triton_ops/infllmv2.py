# -*- coding: utf-8 -*-
from __future__ import annotations

import torch
import triton
import triton.language as tl


# ---------- 工具：把 Hq -> Hk 的 q 做组聚合（GQA） ----------
def _group_q_to_k_heads(q_bhd: torch.Tensor, hk: int) -> torch.Tensor:
    # q_bhd: [B, Hq, D]
    B, Hq, D = q_bhd.shape
    if Hq == hk:
        return q_bhd
    g = Hq // hk
    assert Hq % hk == 0 and g > 0
    return q_bhd.view(B, hk, g, D).mean(dim=2)


# ======================
# Stage-1: TopK block select
# ======================
@torch.no_grad()
def infllm2_stage1_select_blocks(
    q_bhd: torch.Tensor,  # [B,Hk,D]
    c1: torch.Tensor,  # [B,Hk,Sc1,D]  (均值池化的块中心)
    c2: torch.Tensor,  # [B,Hk,Sc2,D]  (max/其他辅助，当前实现可选)
    offs_b1n: torch.Tensor,  # [B,1,n+1]     (块前缀索引，末元素=Sk)
    valid_sc1_len: torch.Tensor,  # [B,Hk]        (每个(B,Hk) 的有效块数，尾部为 padding)
    topk: int,
    softmax_scale: float = 1.0,
) -> torch.Tensor:
    """
    简洁实现：使用 torch.bmm 做 GEMM，并在 PyTorch 里 topk。
    这部分理论上可改为 Triton bitonic-topk；为了易用、先给出稳妥版本。
    返回：topk_idx [B,Hk,topk] （在 Sc1 维度上的索引）
    """
    B, Hk, D = q_bhd.shape
    Sc1 = c1.shape[2]

    # scores = <q, c1> / sqrt(D)
    qs = q_bhd.reshape(B * Hk, 1, D)  # [B*Hk,1,D]
    c1s = c1.reshape(B * Hk, Sc1, D).transpose(1, 2)  # [B*Hk,D,Sc1]
    scores = torch.bmm(qs, c1s).squeeze(1) * (softmax_scale)  # [B*Hk,Sc1]

    # 对超出 valid_sc1_len 的尾部块打 -inf 掩码
    lens = valid_sc1_len.reshape(B * Hk)  # [B*Hk]
    arange = torch.arange(Sc1, device=scores.device)[None, :].expand(B * Hk, -1)
    mask = arange >= lens[:, None]
    scores = scores.masked_fill(mask, float("-inf"))

    # 取 topk
    topk_idx = torch.topk(
        scores, k=min(topk, Sc1), dim=1, largest=True, sorted=False
    ).indices
    return topk_idx.reshape(B, Hk, -1).contiguous()


# ======================
# Stage-2: ranges -> kv_indices/kv_indptr
# ======================


@triton.jit
def _ranges_to_indices_kernel(
    req_to_token_ptr,  # int32 / int64 table [pool, pool_w]
    req_stride_m,
    req_stride_n,
    req_pool_indices_ptr,  # int32 [B]
    range_indptr_ptr,  # int32 [B+1]
    range_starts_ptr,  # int32 [R]     (token起点，按样本串联)
    range_lens_ptr,  # int32 [R]
    kv_indptr_out_ptr,  # int32 [B+1]
    kv_indices_out_ptr,  # int64 [sum tokens]
    pool_w: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """
    对每个样本 b，遍历其所有区间，把 req_to_token[row_b, t] 复制到 kv_indices。
    grid = (B,)
    """
    b = tl.program_id(0)
    # 读 b 的 range 范围
    r_lo = tl.load(range_indptr_ptr + b)
    r_hi = tl.load(range_indptr_ptr + b + 1)
    row = tl.load(req_pool_indices_ptr + b)

    # 写 kv_indptr_out[b] = 累积 token 和
    token_count = 0

    # 累积写指针起点
    base_out = tl.load(kv_indptr_out_ptr + b)

    for r in range(r_lo, r_hi):
        start = tl.load(range_starts_ptr + r)
        length = tl.load(range_lens_ptr + r)

        # 分块 copy
        off = 0
        while off < length:
            step = tl.minimum(BLOCK, length - off)
            idx = start + off + tl.arange(0, BLOCK)
            m = idx < (start + length)

            # req_to_token[row, idx] -> kv_indices
            src_ptr = req_to_token_ptr + row * req_stride_m + idx * req_stride_n
            vals = tl.load(src_ptr, mask=m, other=0).to(tl.int64)
            dst_ptr = (
                kv_indices_out_ptr + base_out + token_count + off + tl.arange(0, BLOCK)
            )
            tl.store(dst_ptr, vals, mask=m)
            off += step
        token_count += length

    # kv_indptr_out[b+1] = base + token_count (已经由前缀和提供；这里可不写)
    # 为安全起见，这里不写，前缀和在 Python 侧完成。


@torch.no_grad()
def infllm2_build_kv_indices_from_ranges(
    ranges: list[list[tuple[int, int]]],  # len=B，元素是[(s,e),...], 半开区间 [s,e)
    req_to_token: torch.Tensor,  # [pool, pool_w] int32
    req_pool_indices: torch.Tensor,  # [B] int32
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    把“样本级 token 区间并集”转成 kernel 接口需要的 kv_indptr/kv_indices。
    """
    device = req_to_token.device
    B = len(ranges)

    # 展平 ranges
    starts, lens, r_indptr = [], [], [0]
    kv_tokens_per_b = []
    for b in range(B):
        tot = 0
        for s, e in ranges[b]:
            if e <= s:  # 跳过空段
                continue
            starts.append(s)
            lens.append(e - s)
            tot += e - s
        kv_tokens_per_b.append(tot)
        r_indptr.append(len(starts))
    starts = torch.tensor(starts, dtype=torch.int32, device=device)
    lens = torch.tensor(lens, dtype=torch.int32, device=device)
    r_indptr = torch.tensor(r_indptr, dtype=torch.int32, device=device)

    # kv_indptr 由 kv_tokens_per_b 做前缀和
    kv_indptr = torch.zeros((B + 1,), dtype=torch.int32, device=device)
    if B > 0:
        kv_indptr[1:] = torch.cumsum(
            torch.tensor(kv_tokens_per_b, dtype=torch.int32, device=device), dim=0
        )
    kv_indices = torch.empty(
        (int(kv_indptr[-1].item()),), dtype=torch.int64, device=device
    )

    if kv_indices.numel() == 0:
        return kv_indptr, kv_indices

    # 启动 kernel；把 kv_indptr 作为每个样本写入偏移（base_out）
    grid = (B,)
    _ranges_to_indices_kernel[grid](
        req_to_token,
        req_to_token.stride(0),
        req_to_token.stride(1),
        req_pool_indices.to(torch.int32),
        r_indptr,
        starts,
        lens,
        kv_indptr,
        kv_indices,
        pool_w=req_to_token.shape[1],
        BLOCK=256,
        num_warps=4,
    )
    return kv_indptr, kv_indices
