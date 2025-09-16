# python/sglang/test/test_infllmv2_context_and_ranges.py
import pytest
import torch

from sglang.srt.mem_cache.infllmv2_context import (
    ContextMemoryManager,
    KVViewsConfig,
    ensure_c1c2_for_sample,
    gather_seq_from_pages,
)
from sglang.srt.sparse_utils.range_mapping import blocks_to_token_ranges


@pytest.mark.parametrize("dev", ["cuda" if torch.cuda.is_available() else "cpu"])
@torch.no_grad()
def test_build_and_append(dev):
    torch.manual_seed(0)
    cfg = KVViewsConfig()
    mgr = ContextMemoryManager(cfg, device=dev, dtype=torch.float16)
    Hk, D = 2, 64
    Sk1, Sk2 = 256, 320

    # 假的 paged-KV 池（线性编号）
    K_pool = torch.randn(Sk2, Hk, D, device=dev, dtype=torch.float16)
    kv_indices = torch.arange(Sk2, device=dev, dtype=torch.int32)
    kv_indptr = torch.tensor([0, Sk1], device=dev, dtype=torch.int32)

    rid, layer = "reqA", 0
    # 首次
    c1, c2, offs, sc1_len, Sk_vis = ensure_c1c2_for_sample(
        mgr, rid, layer, K_pool=K_pool, kv_indptr=kv_indptr, kv_indices=kv_indices, b=0
    )
    assert Sk_vis == Sk1 and c1.shape[0] == 1 and c1.shape[1] == Hk
    # 追加
    kv_indptr2 = torch.tensor([0, Sk2], device=dev, dtype=torch.int32)
    c1b, c2b, offsb, sc1_lenb, Sk_vis2 = ensure_c1c2_for_sample(
        mgr, rid, layer, K_pool=K_pool, kv_indptr=kv_indptr2, kv_indices=kv_indices, b=0
    )
    assert Sk_vis2 == Sk2 and sc1_lenb.item() >= sc1_len.item()


@pytest.mark.parametrize("dev", ["cuda" if torch.cuda.is_available() else "cpu"])
@torch.no_grad()
def test_preview_and_rollback(dev):
    torch.manual_seed(1)
    cfg = KVViewsConfig()
    mgr = ContextMemoryManager(cfg, device=dev, dtype=torch.float16)
    Hk, D = 2, 64
    Sk = 192

    K_pool = torch.randn(Sk, Hk, D, device=dev, dtype=torch.float16)
    kv_indices = torch.arange(Sk, device=dev, dtype=torch.int32)
    kv_indptr = torch.tensor([0, Sk], device=dev, dtype=torch.int32)

    rid, layer = "reqB", 1
    # 建好持久
    c1, c2, offs, sc1_len, _ = ensure_c1c2_for_sample(
        mgr, rid, layer, K_pool=K_pool, kv_indptr=kv_indptr, kv_indices=kv_indices, b=0
    )
    # 预览可见长度（不落地）
    c1p, c2p, offsp, sc1_lenp, Sk_visp = ensure_c1c2_for_sample(
        mgr,
        rid,
        layer,
        K_pool=K_pool,
        kv_indptr=kv_indptr,
        kv_indices=kv_indices,
        b=0,
        preview_visible_sc=128,
    )
    assert Sk_visp == 128
    # 再取一次持久，确保没被覆盖
    c1c, c2c, offsc, sc1_lenc, Sk_visc = ensure_c1c2_for_sample(
        mgr, rid, layer, K_pool=K_pool, kv_indptr=kv_indptr, kv_indices=kv_indices, b=0
    )
    assert sc1_lenc.item() == sc1_len.item()


@torch.no_grad()
def test_blocks_to_ranges_cpu():
    B, Sq, Hk, K = 1, 3, 2, 2
    # 3个 query，各选两个块
    topk = torch.tensor(
        [
            [
                [[0, 2], [1, 3], [0, 1]],
                [[1, 2], [2, 3], [0, 3]],
                [[2, 3], [0, 1], [1, 2]],
            ]
        ],
        dtype=torch.long,
    )[:, :, :, :2]
    # c1 block offsets：每个块 64 token
    offs = torch.tensor([[[0, 64, 128, 192, 256]] * Hk], dtype=torch.int32)
    ranges = blocks_to_token_ranges(
        topk, offs, block_size=64, sink_len=64, sw_span=128, Sk_vis=224
    )
    assert len(ranges[0][0]) >= 1
