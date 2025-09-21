# python/sglang/srt/mem_cache/infllmv2_context.py
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


# ---------- 公共配置 ----------
@dataclass
class KVViewsConfig:
    # c1: avg(32,16) → max(5,4,1) ≈ 64-token block
    c1_avg_kernel: int = 32
    c1_avg_stride: int = 16
    c1_max_kernel: int = 5
    c1_max_stride: int = 4
    c1_max_padding: int = 1

    # c2: 更粗的 avg(128,64) 近似 LSE
    c2_avg_kernel: int = 128
    c2_avg_stride: int = 64

    # stage-2 的 token 块大小（注意力稀疏映射）
    block_size: int = 64


# ---------- 轻页表 / 工具 ----------
def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def out_len(L: int | torch.Tensor, K: int, S: int, P: int = 0):
    if isinstance(L, int):
        num = L + 2 * P - K
        return 0 if num < 0 else num // S + 1
    num = L + 2 * P - K
    return torch.where(num >= 0, num // S + 1, torch.zeros_like(num))


def build_block_offsets(Sk: int, block_size: int, device="cuda") -> torch.Tensor:
    n = ceil_div(Sk, block_size)
    offs = torch.arange(n + 1, dtype=torch.int32, device=device)
    offs[:-1] *= block_size
    offs[-1] = Sk  # 末位=Sk，便于映射半块
    return offs  # [n+1]


def to_ncl_from_bhsd(x_bhsd: torch.Tensor) -> torch.Tensor:
    # [B,H,S,D] -> [B*H, D, S]
    B, H, S, D = x_bhsd.shape
    return x_bhsd.permute(0, 1, 3, 2).reshape(B * H, D, S)


def from_ncl_to_bhsd(y_ncl: torch.Tensor, B: int, H: int) -> torch.Tensor:
    # [B*H, D, L] -> [B,H,L,D]
    N, D, L = y_ncl.shape
    return y_ncl.reshape(B, H, D, L).permute(0, 1, 3, 2).contiguous()


def gather_seq_from_pages(
    K_pool: torch.Tensor, kv_indices: torch.Tensor, s: int, e: int
) -> torch.Tensor:
    # K_pool:[Pool,Hk,D]  kv_indices:[sum Sk]  返回 [Sk,Hk,D]
    return K_pool.index_select(0, kv_indices[s:e])


# ---------- Block Context Memory（每 request/layer 一份） ----------
class BlockCtxMem:
    """管理 c1/c2 侧缓存、长度、轻页表，支持快照/回滚。"""

    def __init__(self, cfg: KVViewsConfig, device="cuda", dtype=torch.float16):
        self.cfg = cfg
        self.device = torch.device(device)
        self.dtype = dtype
        self.state: Dict[str, Any] = {
            "Sk": 0,  # 可见 token 数 (int)
            "Hk": None,  # kv-head 数
            "c1": None,  # [1,Hk,Sc1,Dqk]
            "c1_avg": None,  # [1,Hk,Sc1_avg,Dqk]
            "c2": None,  # [1,Hk,Sc2,Dqk]
            "sc1_len": torch.tensor([0], device=self.device, dtype=torch.int32),
            "sc2_len": torch.tensor([0], device=self.device, dtype=torch.int32),
            "offsets": torch.tensor(
                [[[0]]], device=self.device, dtype=torch.int32
            ),  # [1,1,n+1]
            "_snap": None,  # 快照
        }

    def _pool_c1c2_from_K(self, K_bhsd: torch.Tensor):
        """
        K_bhsd: [B,H,S,D] (通常 B=1)
        返回:
        c1: [B,H,L1,D], c2: [B,H,L2,D],
        offs: [B,1,n+1] （按 c1 的块边界构造，最后一个值是 S）
        sc1_len: int(L1), sc2_len: int(L2)
        """
        B, H, S, D = K_bhsd.shape
        device = K_bhsd.device
        dtype_i = torch.int32

        # 变成 [B*H*D, 1, S] 便于 avg_pool1d
        x = K_bhsd.permute(0, 1, 3, 2).reshape(B * H * D, 1, S)

        # 你的块配置（如无则给默认值）
        k1 = getattr(self, "c1_kernel", getattr(self, "c1_block", 16))
        s1 = getattr(self, "c1_stride", k1)
        k2 = getattr(self, "c2_kernel", getattr(self, "c2_block", 64))
        s2 = getattr(self, "c2_stride", k2)

        def safe_pool(inp, kernel, stride):
            # S 太短：退化为单块
            if S < kernel:
                out = F.adaptive_avg_pool1d(inp, output_size=1)
                return out, 1
            out = F.avg_pool1d(inp, kernel_size=kernel, stride=stride)
            return out, out.shape[-1]

        c1_avg, L1 = safe_pool(x, k1, s1)  # [B*H*D, 1, L1]
        c2_avg, L2 = safe_pool(x, k2, s2)  # [B*H*D, 1, L2]

        # 还原 [B,H,L,D]
        c1 = c1_avg.reshape(B, H, D, L1).permute(0, 1, 3, 2).contiguous()
        c2 = c2_avg.reshape(B, H, D, L2).permute(0, 1, 3, 2).contiguous()

        # === 构造 offs（与 c1 对齐）===
        if S < k1:
            offs = torch.tensor([0, S], device=device, dtype=dtype_i).view(1, 1, 2)
        else:
            starts = torch.arange(
                0, max(S - k1 + 1, 1), step=s1, device=device, dtype=dtype_i
            )
            ends = torch.clamp(starts + k1, max=S)
            offs = torch.cat(
                [torch.tensor([0], device=device, dtype=dtype_i), ends], dim=0
            )
            if offs[-1].item() != S:
                offs = torch.cat(
                    [offs, torch.tensor([S], device=device, dtype=dtype_i)], dim=0
                )
            offs = offs.view(1, 1, -1)

        return c1, c2, offs, int(L1), int(L2)

    def build_full(self, K_seq: torch.Tensor):
        """
        K_seq 支持 [S,H,D] / [H,S,D] / [S,D]，统一成 [1,H,S,D] 再池化。
        """
        if K_seq.dim() == 3:
            # 假定两个 3D 排列之一
            # 通过比较前两维猜测 [S,H,D] vs [H,S,D]，或者显式记录你的布局
            if K_seq.shape[0] < K_seq.shape[1]:  # [S,H,D]
                K_bhsd = (
                    K_seq.unsqueeze(0).permute(0, 1, 0, 2).contiguous()
                )  # -> [1,H,S,D]
            else:  # [H,S,D]
                K_bhsd = K_seq.unsqueeze(0).contiguous()  # 已是 [1,H,S,D]
        elif K_seq.dim() == 2:  # [S,D]
            K_bhsd = K_seq.unsqueeze(0).unsqueeze(0).contiguous()  # [1,1,S,D]
        else:
            raise ValueError(f"Unexpected K_seq shape: {tuple(K_seq.shape)}")

        return self._pool_c1c2_from_K(K_bhsd)

    def append_from_tail_plus_new(
        self, K_tail_plus_new: torch.Tensor, Sk_new: int, tail_len: int
    ):
        """
        增量：K_tail_plus_new:[tail+add,Hk,D]，Sk_new = 旧Sk+add，tail_len <= kernels-1
        只对“尾巴+新增”重算并追加 c1/c2；更新 offsets[-1]=Sk_new
        """
        st = self.state
        cfg = self.cfg
        assert st["c1"] is not None, "call build_full first"
        Hk = st["Hk"]
        B = 1
        x = K_tail_plus_new.unsqueeze(0).permute(0, 1, 0, 2).contiguous()  # [1,Hk,L,D]
        x = to_ncl_from_bhsd(x)  # [Hk,D,L]

        # 这段的 avg 池化
        c1_avg_piece = F.avg_pool1d(
            x, kernel_size=cfg.c1_avg_kernel, stride=cfg.c1_avg_stride, ceil_mode=False
        )
        c2_piece = F.avg_pool1d(
            x, kernel_size=cfg.c2_avg_kernel, stride=cfg.c2_avg_stride, ceil_mode=False
        )
        c1_avg_piece_bhsd = from_ncl_to_bhsd(c1_avg_piece, B, Hk)
        c2_piece_bhsd = from_ncl_to_bhsd(c2_piece, B, Hk)

        # 计算增量长度
        L1_old = int(st["sc1_len"].item())
        L1_new = int(out_len(Sk_new, cfg.c1_avg_kernel, cfg.c1_avg_stride, 0))
        d1 = max(0, L1_new - L1_old)
        if d1 > 0:
            y1_piece_len = int(
                out_len(
                    tail_len + (Sk_new - st["Sk"]),
                    cfg.c1_avg_kernel,
                    cfg.c1_avg_stride,
                    0,
                )
            )
            y1_tail = c1_avg_piece_bhsd[:, :, y1_piece_len - d1 : y1_piece_len, :]
            st["c1_avg"] = (
                torch.cat([st["c1_avg"]] if st["c1_avg"] is not None else [], dim=1)
                if False
                else (
                    st["c1_avg"]
                    if st["c1_avg"] is not None
                    else c1_avg_piece_bhsd[:, :, :0, :]
                )
            )
            st["c1_avg"] = torch.cat([st["c1_avg"], y1_tail], dim=2)

        L2_old = int(st["sc2_len"].item())
        L2_new = int(out_len(Sk_new, cfg.c2_avg_kernel, cfg.c2_avg_stride, 0))
        d2 = max(0, L2_new - L2_old)
        if d2 > 0:
            y2_piece_len = int(
                out_len(
                    tail_len + (Sk_new - st["Sk"]),
                    cfg.c2_avg_kernel,
                    cfg.c2_avg_stride,
                    0,
                )
            )
            y2_tail = c2_piece_bhsd[:, :, y2_piece_len - d2 : y2_piece_len, :]
            st["c2"] = torch.cat([st["c2"], y2_tail], dim=2)

        # c1 = max_pool(c1_avg) 的增量
        c1_tail_in = min(L1_old, cfg.c1_max_kernel - 1)
        c1_in = torch.cat(
            [
                st["c1_avg"][:, :, L1_old - c1_tail_in : L1_old, :],
                st["c1_avg"][:, :, L1_old:, :],
            ],
            dim=2,
        )
        c1_piece = F.max_pool1d(
            to_ncl_from_bhsd(c1_in),
            kernel_size=cfg.c1_max_kernel,
            stride=cfg.c1_max_stride,
            padding=cfg.c1_max_padding,
            ceil_mode=False,
        )
        c1_piece_bhsd = from_ncl_to_bhsd(c1_piece, B, Hk)

        Lc1_old = int(st["c1"].shape[2])
        Lc1_new = int(
            out_len(L1_new, cfg.c1_max_kernel, cfg.c1_max_stride, cfg.c1_max_padding)
        )
        dc1 = max(0, Lc1_new - Lc1_old)
        if dc1 > 0:
            y3_piece_len = int(
                out_len(
                    c1_tail_in + d1,
                    cfg.c1_max_kernel,
                    cfg.c1_max_stride,
                    cfg.c1_max_padding,
                )
            )
            y3_tail = c1_piece_bhsd[:, :, y3_piece_len - dc1 : y3_piece_len, :]
            st["c1"] = torch.cat([st["c1"], y3_tail], dim=2)

        st["Sk"] = Sk_new
        st["sc1_len"] = torch.tensor(
            [st["c1"].shape[2]], device=st["c1"].device, dtype=torch.int32
        )
        st["sc2_len"] = torch.tensor(
            [st["c2"].shape[2]], device=st["c2"].device, dtype=torch.int32
        )
        st["offsets"] = build_block_offsets(
            Sk_new, cfg.block_size, device=st["c1"].device
        )[None, None, :].contiguous()

    def get(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """返回 c1, c2, offsets, sc1_len, Sk（均为单元素或 [1, ...] 形状）"""
        st = self.state
        Sk = torch.tensor(
            [st["Sk"]],
            device=st["c1"].device if st["c1"] is not None else "cuda",
            dtype=torch.int32,
        )
        return st["c1"], st["c2"], st["offsets"], st["sc1_len"], Sk

    # 快照 / 回滚（spec verify 预览场景）
    def snapshot(self):
        st = self.state
        st["_snap"] = {
            "Sk": st["Sk"],
            "Hk": st["Hk"],
            "c1": None if st["c1"] is None else st["c1"].clone(),
            "c1_avg": None if st["c1_avg"] is None else st["c1_avg"].clone(),
            "c2": None if st["c2"] is None else st["c2"].clone(),
            "sc1_len": st["sc1_len"].clone(),
            "sc2_len": st["sc2_len"].clone(),
            "offsets": st["offsets"].clone(),
        }

    def rollback(self):
        snap = self.state.get("_snap")
        if snap is None:
            return
        self.state.update(
            {k: (v.clone() if torch.is_tensor(v) else v) for k, v in snap.items()}
        )
        self.state["_snap"] = None


# ---------- 每 request 的全层内存管理 ----------
class ContextMemoryManager:
    def __init__(
        self,
        cfg: KVViewsConfig,
        ttl_seconds: int = 900,
        max_entries: int = 4096,
        device="cuda",
        dtype=torch.float16,
    ):
        self.cfg, self.ttl, self.max = cfg, ttl_seconds, max_entries
        self.device, self.dtype = device, dtype
        self.pool: Dict[str, Dict[int, BlockCtxMem]] = (
            {}
        )  # request_id -> {layer_id: BlockCtxMem}
        self.meta: Dict[str, float] = {}

    def _sweep(self):
        now = time.time()
        remove = [
            rid for rid, ts in self.meta.items() if self.ttl > 0 and now - ts > self.ttl
        ]
        for rid in remove:
            self.pool.pop(rid, None)
            self.meta.pop(rid, None)
        if len(self.pool) > self.max:
            # LRU
            for rid in list(sorted(self.meta, key=self.meta.get))[
                : len(self.pool) - self.max
            ]:
                self.pool.pop(rid, None)
                self.meta.pop(rid, None)

    def get_mem(self, request_id: str) -> Dict[int, BlockCtxMem]:
        self._sweep()
        self.meta[request_id] = time.time()
        if request_id not in self.pool:
            self.pool[request_id] = {}
        return self.pool[request_id]

    def get_layer_mem(self, request_id: str, layer_id: int) -> BlockCtxMem:
        lay = self.get_mem(request_id)
        if layer_id not in lay:
            lay[layer_id] = BlockCtxMem(self.cfg, device=self.device, dtype=self.dtype)
        return lay[layer_id]

    def clear_request(self, request_id: str):
        self.pool.pop(request_id, None)
        self.meta.pop(request_id, None)

    def clear_all(self):
        self.pool.clear()
        self.meta.clear()


# ---------- 顶层：与 paged-KV 的对接 ----------
def ensure_c1c2_for_sample(
    ctx_mgr,
    request_id,
    layer_id,
    *,
    K_seq: Optional[torch.Tensor] = None,
    K_pool: Optional[torch.Tensor] = None,
    kv_indptr: Optional[torch.Tensor] = None,
    kv_indices: Optional[torch.Tensor] = None,
    b: Optional[int] = None,
    block_size: int = 16,
    preview_visible_sc: Optional[int] = None,
):
    """
    生成 / 更新 C1、C2 以及块前缀 offs。
    - 仅允许关键字传参（避免 `got multiple values`）。
    - 输入可为 K_seq（[S,Hk,D]），或 K_pool+[kv_indices]（从池按索引取出 seq）。
    - 若给出 preview_visible_sc，则只对可见前缀做视图（不落地到持久状态）。

    返回：
      c1:  [1, Hk, L1, D]
      c2:  [1, Hk, L2, D]（此处与 c1 同阶数，按需你可改成别的构造）
      offs:[1, 1, n+1]  （块前缀边界）
      sc1_len: n        （L1 的块数，int 标量 tensor）
      Sk_vis: 可见 token 长度（int 标量 tensor）
    """
    # 统一得到 K_seq_full
    if K_seq is None:
        assert (K_pool is not None) and (
            kv_indices is not None
        ), "Provide either K_seq or (K_pool + kv_indices)."
        idx = kv_indices
        if idx.dtype != torch.int64:
            idx = idx.to(torch.int64)
        K_seq = K_pool.index_select(0, idx)  # [S_full, Hk, D]

    assert K_seq.dim() == 3, f"K_seq must be [S,Hk,D], got {tuple(K_seq.shape)}"
    S_full, Hk, D = K_seq.shape
    device, dtype = K_seq.device, K_seq.dtype

    # 计算可见长度 Sk_vis
    if preview_visible_sc is not None:
        Sk_vis_val = min(int(preview_visible_sc), int(S_full))
    elif kv_indptr is not None:
        # 常规：用 indptr 提供的这条样本的可见长度
        # 约定 kv_indptr 是 [start, end]（两元素）或全 batch 的片段，这里取差值
        if kv_indptr.numel() == 2:
            Sk_vis_val = int((kv_indptr[1] - kv_indptr[0]).item())
        else:
            Sk_vis_val = int(S_full)
    else:
        Sk_vis_val = int(S_full)

    # 兜底：空序列也返回合法结构，避免 None
    if Sk_vis_val <= 0:
        c1 = K_seq.new_zeros((1, Hk, 1, D))
        c2 = K_seq.new_zeros((1, Hk, 1, D))
        offs = torch.tensor([[[0, 0]]], device=device, dtype=torch.int32)
        sc1_len = torch.tensor(0, device=device, dtype=torch.int32)
        Sk_vis = torch.tensor(0, device=device, dtype=torch.int32)
        return c1, c2, offs, sc1_len, Sk_vis

    # 取可见前缀
    K_vis = K_seq[:Sk_vis_val]  # [Sk_vis, Hk, D]

    # 计算块数与边界（至少 1 块）
    n_blocks = max(1, (Sk_vis_val + block_size - 1) // block_size)
    edges = torch.arange(0, n_blocks + 1, device=device, dtype=torch.int32) * block_size
    edges = torch.clamp(edges, max=Sk_vis_val)  # [n+1]，最后一段可能短
    offs = edges.view(1, 1, -1)  # [1,1,n+1]

    # 构造 C1：把时间维池化到 n_blocks
    # K_vis: [Sk_vis, Hk, D] -> [Hk*D, 1, Sk_vis] 做 1D 池化
    x = K_vis.permute(1, 2, 0).contiguous().view(Hk * D, 1, Sk_vis_val)
    c1_avg = F.adaptive_avg_pool1d(x, output_size=n_blocks)  # [Hk*D,1,n]
    c1 = (
        c1_avg.view(Hk, D, n_blocks).permute(0, 2, 1).contiguous().unsqueeze(0)
    )  # [1,Hk,n,D]
    # 简化：c2 先与 c1 一致（按需可改为其它摘要方式）
    c2 = c1

    # 预览模式：不落地；否则可选地写回 ctx_mgr（如果你有持久化接口）
    # if preview_visible_sc is None and hasattr(ctx_mgr, "update"):
    #     ctx_mgr.update(request_id, layer_id, c1, c2, offs, Sk_vis_val, block_size)

    sc1_len = torch.tensor(n_blocks, device=device, dtype=torch.int32)
    Sk_vis = torch.tensor(Sk_vis_val, device=device, dtype=torch.int32)
    return c1, c2, offs, sc1_len, Sk_vis
