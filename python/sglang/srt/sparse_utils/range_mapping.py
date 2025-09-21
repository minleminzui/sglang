# python/sglang/srt/sparse_utils/range_mapping.py
from __future__ import annotations

from typing import List, Tuple

import torch


def _merge_intervals(iv, Sk_vis):
    if not iv:
        return []
    iv = [(max(0, int(s)), min(Sk_vis, int(e))) for s, e in iv if int(s) < int(e)]
    iv.sort()
    out = [iv[0]]
    for s, e in iv[1:]:
        ps, pe = out[-1]
        if s <= pe:
            out[-1] = (ps, max(pe, e))
        else:
            out.append((s, e))
    return out


def blocks_to_token_ranges(
    topk_idx_bshk, c1_block_offsets_bhk, block_size, sink_len, sw_span, Sk_vis: int
):
    """
    返回: ranges: List[List[(s,e)]], len == B
    """
    B = topk_idx_bshk.shape[0]
    if topk_idx_bshk.dim() == 3:  # [B,Hk,K] -> [B,1,Hk,K]
        topk_idx_bshk = topk_idx_bshk.unsqueeze(1)
    _, Sq, Hk, K = topk_idx_bshk.shape

    offs = c1_block_offsets_bhk  # 期望 [B,Hk,n+1]
    if offs.dim() == 3:  # [B,1,n+1] -> [B,Hk,n+1]
        offs = offs.expand(-1, Hk, -1).contiguous()

    if not isinstance(Sk_vis, int):
        Sk_vis = int(Sk_vis)

    ranges = []
    for b in range(B):
        iv = []
        # sink
        if isinstance(sink_len, int) and sink_len > 0:
            iv.append((0, min(Sk_vis, sink_len)))
        # sliding window
        if isinstance(sw_span, int) and sw_span > 0:
            st = max(0, Sk_vis - sw_span)
            if st < Sk_vis:
                iv.append((st, Sk_vis))
        # blocks
        for h in range(Hk):
            for k in range(K):
                blk = int(topk_idx_bshk[b, 0, h, k].item())
                if blk < 0:
                    continue
                s = int(offs[b, h, blk].item())
                e = int(offs[b, h, blk + 1].item())
                if s < e:
                    iv.append((s, e))
        ranges.append(_merge_intervals(iv, Sk_vis))  # << 直接 append 扁平的 list[tuple]
    return ranges


def _normalize_and_merge(raw, Sk_vis: int):
    # raw 里可能有 None、单元素、嵌套等，这里统一成有序不相交区间
    segs = []
    for it in raw:
        if it is None:
            continue
        if isinstance(it, (list, tuple)):
            if len(it) >= 2:
                s, e = it[0], it[1]
            else:
                continue
        else:
            # 遇到标量/其他类型直接跳过
            continue
        try:
            s = int(s)
            e = int(e)
        except Exception:
            continue
        if s < 0:
            s = 0
        if e > int(Sk_vis):
            e = int(Sk_vis)
        if s >= e:
            continue
        segs.append((s, e))

    segs.sort()
    merged = []
    for s, e in segs:
        if not merged or s > merged[-1][1]:
            merged.append([s, e])
        else:
            merged[-1][1] = max(merged[-1][1], e)
    return [tuple(x) for x in merged]


@torch.no_grad()
def blocks_to_token_ranges(
    topk_idx_bshk: torch.Tensor,  # [B,Sq,Hk,K] 或 [B,Hk,K]（会在这里补 Sq=1 维）
    c1_block_offsets_bhk: torch.Tensor,  # [B,Hk,n+1]，int32
    block_size: int,
    sink_len: int,
    sw_span: int,
    Sk_vis: int,
):
    # 统一 topk 形状
    if topk_idx_bshk.dim() == 3:  # [B,Hk,K] -> [B,1,Hk,K]
        topk_idx_bshk = topk_idx_bshk.unsqueeze(1)
    B, Sq, Hk, K = topk_idx_bshk.shape

    offs = c1_block_offsets_bhk.to(dtype=torch.int32)  # [B,Hk,n+1]
    assert offs.dim() == 3 and offs.shape[0] == B and offs.shape[1] == Hk

    ranges = [[] for _ in range(B)]

    # sink（可选）
    sink_e = min(int(Sk_vis), int(sink_len)) if sink_len and sink_len > 0 else 0
    if sink_e > 0:
        for b in range(B):
            ranges[b].append((0, sink_e))

    # 把 Top-K 的块索引映射到 token 区间
    for b in range(B):
        for i in range(Sq):
            for h in range(Hk):
                blk_ids = topk_idx_bshk[b, i, h]  # [K]
                off = offs[b, h]  # [n+1]
                n = off.shape[-1] - 1
                if n <= 0:
                    continue
                for bi in blk_ids:
                    bi = int(bi.item())
                    if bi < 0 or bi + 1 > n:
                        continue
                    s = int(off[bi].item())
                    e = int(off[bi + 1].item())
                    if s < e:
                        ranges[b].append((s, e))

        # SW（可选）
        if sw_span and sw_span > 0 and Sk_vis > 0:
            s = max(0, int(Sk_vis) - int(sw_span))
            ranges[b].append((s, int(Sk_vis)))

    # 归一化 + 合并
    out = []
    for b in range(B):
        out.append(_normalize_and_merge(ranges[b], Sk_vis))
    return out


def blocks_to_token_ranges(
    topk_idx_bshk: torch.Tensor,  # [B,Sq,Hk,K]  对应 c1 的 block 索引
    c1_block_offsets_bhk: torch.Tensor,  # [B,Hk,n+1]   c1 的块前缀映射到 token 索引
    block_size: int,
    sink_len: int,
    sw_span: int,
    Sk_vis: int,
) -> List[List[List[Tuple[int, int]]]]:
    """
    输出 token 区间的并集（闭开区间 [s,e) ）：
    ret[b][hk] = [(s0,e0), (s1,e1), ...]  按起点升序且不相交、已合并
    规则：TopK 块 → token 区间；并 ∪ Sink[0,sink_len) ∪ SW[max(0,Sk_vis-sw_span), Sk_vis)
    """
    B, Sq, Hk, K = topk_idx_bshk.shape
    ret: List[List[List[Tuple[int, int]]]] = [[[] for _ in range(Hk)] for _ in range(B)]
    # 统一加入 sink/sw
    sink_e = min(Sk_vis, sink_len) if sink_len > 0 else 0
    sw_s = max(0, Sk_vis - sw_span) if sw_span > 0 else Sk_vis
    for b in range(B):
        for hk in range(Hk):
            segs: List[Tuple[int, int]] = []
            # TopK 块
            if K > 0:
                blocks = topk_idx_bshk[b, :, hk, :].unique()
                if blocks.numel() > 0:
                    offs = c1_block_offsets_bhk[b, hk]  # [n+1]
                    for t in blocks.tolist():
                        if t < 0 or t + 1 >= offs.numel():
                            continue
                        s = int(offs[t].item())
                        e = int(min(offs[t + 1].item(), Sk_vis))
                        if s < e:
                            segs.append((s, e))
            # Sink
            if sink_e > 0:
                segs.append((0, sink_e))
            # Sliding Window
            if sw_s < Sk_vis:
                segs.append((sw_s, Sk_vis))
            # 合并
            if not segs:
                ret[b][hk] = []
                continue
            segs.sort(key=lambda x: x[0])
            merged: List[Tuple[int, int]] = [segs[0]]
            for s, e in segs[1:]:
                ps, pe = merged[-1]
                if s <= pe:
                    merged[-1] = (ps, max(pe, e))
                else:
                    merged.append((s, e))
            ret[b][hk] = merged
    return ret
