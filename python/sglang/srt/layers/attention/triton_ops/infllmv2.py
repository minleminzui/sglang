# /sgl-workspace/sglang/python/sglang/srt/layers/attention/triton_ops/infllmv2.py
from __future__ import annotations

import torch
import triton
import triton.language as tl


def _iter_segments_safe(ranges_b):
    """统一解析成合法 (s,e) 且 s<e"""
    if ranges_b is None:
        return
    for seg in ranges_b:
        # 展平一层像 [(s,e)] 这种
        if (
            isinstance(seg, (list, tuple))
            and len(seg) == 1
            and isinstance(seg[0], (list, tuple, torch.Tensor))
        ):
            seg = seg[0]

        if torch.is_tensor(seg):
            if seg.numel() >= 2:
                s = seg.reshape(-1)[0]
                e = seg.reshape(-1)[1]
            else:
                continue
        elif isinstance(seg, (list, tuple)):
            if len(seg) >= 2:
                s, e = seg[0], seg[1]
            else:
                continue
        else:
            continue

        try:
            s = int(s)
            e = int(e)
        except Exception:
            continue

        if s < e:
            yield s, e


@torch.no_grad()
def infllm2_build_kv_indices_from_ranges(ranges, req_to_token, req_pool_indices):
    """
    ranges: List[List[(s,e)] 或含有轻微嵌套的结构]  -> 清洗为扁平区间
    返回:
      kv_indptr: [B+1] int32
      kv_indices: [sum_len] int64
    """
    device = req_to_token.device
    B = len(ranges)
    kv_indptr = torch.zeros((B + 1,), dtype=torch.int32, device=device)
    total = 0
    cleaned = []
    for b in range(B):
        segs = list(_iter_segments_safe(ranges[b]))
        cleaned.append(segs)
        for s, e in segs:
            total += e - s
        kv_indptr[b + 1] = total

    kv_indices = torch.empty((total,), dtype=torch.int64, device=device)

    write_ptr = 0
    for b in range(B):
        ridx = (
            int(req_pool_indices[b].item())
            if torch.is_tensor(req_pool_indices)
            else int(req_pool_indices[b])
        )
        for s, e in cleaned[b]:
            seg = req_to_token[ridx, s:e].to(
                dtype=torch.int64
            )  # 按你的实现把 (s,e) 映射为池内 token 索引
            n = seg.numel()
            kv_indices[write_ptr : write_ptr + n] = seg
            write_ptr += n

    return kv_indptr, kv_indices


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


# ==== NEW: Stage-1 scores kernel (Trition) ====


@triton.jit
def _stage1_scores_kernel(
    q_ptr,  # [B,Hk,D]
    c1_ptr,  # [B,Hk,Sc1,D]
    valid_len_ptr,  # [B,Hk]
    out_ptr,  # [B,Hk,Sc1]  (scores)
    B: tl.constexpr,
    Hk: tl.constexpr,
    Sc1: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_SC1: tl.constexpr,
    scale: tl.constexpr,  # softmax_scale
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    if pid_b >= B or pid_h >= Hk:
        return

    # 取有效块上限（尾部 padding 打 -inf）
    valid = tl.load(valid_len_ptr + pid_b * Hk + pid_h)

    # 指向 q[b,h,:]
    q_base = q_ptr + (pid_b * Hk + pid_h) * D
    # 输出指针起点
    out_base = out_ptr + (pid_b * Hk + pid_h) * Sc1

    # 分块遍历 Sc1 维
    off_sc1 = tl.arange(0, BLOCK_SC1)
    for col0 in range(0, Sc1, BLOCK_SC1):
        sc1_idx = col0 + off_sc1
        sc1_mask = sc1_idx < Sc1

        # 计算 <q, c1[b,h,sc1_idx,:]>  (逐块、逐 D tile 做累加)
        acc = tl.zeros([BLOCK_SC1], dtype=tl.float32)
        for d0 in range(0, D, BLOCK_D):
            d_idx = d0 + tl.arange(0, BLOCK_D)
            d_mask = d_idx < D

            qv = tl.load(q_base + d_idx, mask=d_mask, other=0.0).to(tl.float32)

            # c1 索引：(((b*Hk + h)*Sc1 + sc1_idx)*D + d_idx)
            c1_base = ((pid_b * Hk + pid_h) * Sc1) * D
            c1_ptr_tile = c1_ptr + c1_base + sc1_idx[:, None] * D + d_idx[None, :]
            c1v = tl.load(
                c1_ptr_tile, mask=sc1_mask[:, None] & d_mask[None, :], other=0.0
            ).to(tl.float32)

            acc += tl.sum(c1v * qv[None, :], axis=1)

        # scale
        acc *= scale

        # 尾部 padding 打 -inf
        pad_mask = sc1_idx >= valid
        acc = tl.where(pad_mask, -float("inf"), acc)

        # 写回
        tl.store(out_base + sc1_idx, acc, mask=sc1_mask)


def infllm2_stage1_select_blocks_triton(
    q_bhd: torch.Tensor,  # [B,Hk,D]
    c1: torch.Tensor,  # [B,Hk,Sc1,D]
    valid_sc1_len: torch.Tensor,  # [B,Hk]
    softmax_scale: float,
    topk: int,
    block_sc1: int = 128,
    block_d: int = 64,
) -> torch.Tensor:
    """
    用 Triton 并行计算 scores，再在 PyTorch 里 topk（稳定 & 简洁）。
    返回: [B,Hk,topk]
    """
    B, Hk, D = q_bhd.shape
    Sc1 = c1.shape[2]
    assert c1.shape[:2] == (B, Hk)
    assert c1.shape[-1] == D

    scores = torch.empty((B, Hk, Sc1), dtype=torch.float32, device=q_bhd.device)
    grid = (B, Hk)

    _stage1_scores_kernel[grid](
        q_bhd,
        c1,
        valid_sc1_len.contiguous(),
        scores,
        B=B,
        Hk=Hk,
        Sc1=Sc1,
        D=D,
        BLOCK_D=triton.next_power_of_2(block_d),
        BLOCK_SC1=triton.next_power_of_2(block_sc1),
        scale=softmax_scale,
    )
    # 在 PyTorch 里做 topk（sorted=False 即可）
    topk_idx = torch.topk(
        scores, k=min(topk, Sc1), dim=-1, largest=True, sorted=False
    ).indices
    return topk_idx.contiguous()


# ==== NEW: Stage-2 ranges -> kv_indices 封装（Triton 版本） ====


def infllm2_build_kv_indices_from_ranges_triton(ranges, req_to_token, req_pool_indices):
    """
    与 infllm2_build_kv_indices_from_ranges 等价，但数据多时更快。
    """
    device = req_to_token.device
    B = len(ranges)

    # 先清洗成扁平段
    flat_starts = []
    flat_lens = []
    range_indptr = [0]
    for b in range(B):
        count = 0
        for s, e in _iter_segments_safe(ranges[b]):
            flat_starts.append(int(s))
            flat_lens.append(int(e - s))
            count += 1
        range_indptr.append(range_indptr[-1] + count)

    R = range_indptr[-1]
    range_indptr = torch.tensor(range_indptr, dtype=torch.int32, device=device)
    range_starts = (
        torch.tensor(flat_starts, dtype=torch.int32, device=device)
        if R > 0
        else torch.empty((0,), dtype=torch.int32, device=device)
    )
    range_lens = (
        torch.tensor(flat_lens, dtype=torch.int32, device=device)
        if R > 0
        else torch.empty((0,), dtype=torch.int32, device=device)
    )

    # 计算每个样本的 token 总数 -> kv_indptr
    tok_counts = torch.zeros((B,), dtype=torch.int32, device=device)
    for b in range(B):
        lo = range_indptr[b].item()
        hi = range_indptr[b + 1].item()
        if hi > lo:
            tok_counts[b] = range_lens[lo:hi].sum()
    kv_indptr = torch.zeros((B + 1,), dtype=torch.int32, device=device)
    kv_indptr[1:] = torch.cumsum(tok_counts, dim=0)
    total = int(kv_indptr[-1].item())
    kv_indices = torch.empty((total,), dtype=torch.int64, device=device)

    if total == 0:
        return kv_indptr, kv_indices

    # 启动 kernel：按样本并行拷贝
    BLOCK = 256
    grid = (B,)
    _ranges_to_indices_kernel[grid](
        req_to_token,
        req_to_token.stride(0),
        req_to_token.stride(1),
        req_pool_indices.contiguous(),
        range_indptr,
        range_starts,
        range_lens,
        kv_indptr,
        kv_indices,
        req_to_token.shape[1],
        BLOCK,
    )
    return kv_indptr, kv_indices
