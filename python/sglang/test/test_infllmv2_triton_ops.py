# tests/test_infllmv2_triton_ops.py
import torch

from sglang.srt.layers.attention.triton_ops.infllmv2 import (
    infllm2_build_kv_indices_from_ranges_triton,
    infllm2_stage1_select_blocks_triton,
)


def test_stage1_scores_topk():
    B, Hk, Sc1, D = 2, 4, 32, 64
    q = torch.randn(B, Hk, D, device="cuda", dtype=torch.float16)
    c1 = torch.randn(B, Hk, Sc1, D, device="cuda", dtype=torch.float16)
    valid = torch.full((B, Hk), Sc1, device="cuda", dtype=torch.int32)
    idx = infllm2_stage1_select_blocks_triton(q, c1, valid, softmax_scale=1.0, topk=8)
    assert idx.shape == (B, Hk, 8)


def test_stage2_ranges_to_indices():
    pool_w = 4096
    req_to_token = torch.arange(pool_w, device="cuda", dtype=torch.int32)[
        None, :
    ].repeat(3, 1)
    ranges = [[(0, 10), (20, 25)], [(5, 8)], []]  # 三个样本
    req_pool_indices = torch.tensor([0, 1, 2], device="cuda", dtype=torch.int32)
    kv_indptr, kv_indices = infllm2_build_kv_indices_from_ranges_triton(
        ranges, req_to_token, req_pool_indices
    )
    assert kv_indptr.tolist() == [0, 15, 18, 18]
    assert kv_indices.numel() == 18


if __name__ == "__main__":
    test_stage1_scores_topk()
    test_stage2_ranges_to_indices()
