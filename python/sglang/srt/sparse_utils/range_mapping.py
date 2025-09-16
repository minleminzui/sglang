# python/sglang/srt/sparse_utils/range_mapping.py
from __future__ import annotations

from typing import List, Tuple

import torch


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
