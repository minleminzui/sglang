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

    def _pool_c1c2_from_K(
        self, K_bhsd: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """从 [1,Hk,S,D] 的 K 构建 c1/c2 与 offsets"""
        cfg = self.cfg
        B, Hk, S, D = K_bhsd.shape
        x = to_ncl_from_bhsd(K_bhsd)  # [Hk,D,S]
        c1_avg = F.avg_pool1d(
            x, kernel_size=cfg.c1_avg_kernel, stride=cfg.c1_avg_stride, ceil_mode=False
        )
        c2 = F.avg_pool1d(
            x, kernel_size=cfg.c2_avg_kernel, stride=cfg.c2_avg_stride, ceil_mode=False
        )
        c1_avg_bhsd = from_ncl_to_bhsd(c1_avg, B, Hk)
        c2_bhsd = from_ncl_to_bhsd(c2, B, Hk)

        c1 = F.max_pool1d(
            to_ncl_from_bhsd(c1_avg_bhsd),
            kernel_size=cfg.c1_max_kernel,
            stride=cfg.c1_max_stride,
            padding=cfg.c1_max_padding,
            ceil_mode=False,
        )
        c1_bhsd = from_ncl_to_bhsd(c1, B, Hk)

        offsets = build_block_offsets(S, cfg.block_size, device=K_bhsd.device)
        return (
            c1_bhsd.to(self.dtype),
            c2_bhsd.to(self.dtype),
            offsets.view(1, 1, -1).contiguous(),
        )

    def build_full(self, K_seq: torch.Tensor):
        """首次/重建：K_seq:[Sk,Hk,D]"""
        K_bhsd = K_seq.unsqueeze(0).permute(0, 1, 0, 2).contiguous()
        c1, c2, offs = self._pool_c1c2_from_K(K_bhsd)
        self.state.update(
            {
                "Sk": K_seq.shape[0],
                "Hk": K_seq.shape[1],
                "c1": c1,
                "c1_avg": None,
                "c2": c2,
                "sc1_len": torch.tensor(
                    [c1.shape[2]], device=K_seq.device, dtype=torch.int32
                ),
                "sc2_len": torch.tensor(
                    [c2.shape[2]], device=K_seq.device, dtype=torch.int32
                ),
                "offsets": offs,
            }
        )

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
    manager: ContextMemoryManager,
    request_id: str,
    layer_id: int,
    *,
    K_pool: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    b: int,
    preview_visible_sc: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """
    返回 (c1, c2, offsets, sc1_len, Sk_vis) 供 Stage-1 使用。
    - 如果 preview_visible_sc 非 None，则不改变持久状态，只构临时视图。
    """
    mem = manager.get_layer_mem(request_id, layer_id)
    s = int(kv_indptr[b].item())
    e = int(kv_indptr[b + 1].item())
    if preview_visible_sc is not None:
        e = min(e, s + int(preview_visible_sc))
    Sk = e - s
    if Sk <= 0:
        dev = K_pool.device
        empty = torch.empty((1, 1, 0, K_pool.shape[-1]), device=dev, dtype=K_pool.dtype)
        offsets = build_block_offsets(0, manager.cfg.block_size, device=dev)[
            None, None, :
        ]
        return (
            empty,
            empty,
            offsets,
            torch.tensor([0], device=dev, dtype=torch.int32),
            0,
        )

    K_seq = gather_seq_from_pages(K_pool, kv_indices, s, e)  # [Sk,Hk,D]
    if preview_visible_sc is not None or mem.state["Sk"] == 0:
        # 临时或首次
        tmp = BlockCtxMem(manager.cfg, device=K_pool.device, dtype=K_pool.dtype)
        tmp.build_full(K_seq)
        c1, c2, offs, sc1_len, Sk_t = tmp.get()
        return c1, c2, offs, sc1_len, int(Sk_t.item())

    # 持久：增量或重建
    if mem.state["Sk"] == 0:
        mem.build_full(K_seq)
    else:
        Sk_old = mem.state["Sk"]
        if Sk > Sk_old:
            tail = min(Sk_old, manager.cfg.c1_avg_kernel - 1)
            tail_start = e - (Sk - Sk_old) - tail
            K_piece = gather_seq_from_pages(K_pool, kv_indices, tail_start, e)
            mem.append_from_tail_plus_new(K_piece, Sk_new=Sk, tail_len=tail)
        elif Sk < Sk_old:
            # 退化：窗口后退（例如页淘汰），重建
            mem.build_full(K_seq)
    c1, c2, offs, sc1_len, _ = mem.get()
    return c1, c2, offs, sc1_len, Sk
