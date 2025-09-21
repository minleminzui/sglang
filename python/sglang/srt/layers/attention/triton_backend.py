# /sgl-workspace/sglang/python/sglang/srt/layers/attention/triton_backend.py
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Union

import torch
import triton
import triton.language as tl

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.utils import create_flashinfer_kv_indices_triton
from sglang.srt.layers.dp_attention import get_attention_tp_size
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.utils import (
    get_bool_env_var,
    get_device_core_count,
    get_int_env_var,
    next_power_of_2,
)

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.speculative.eagle_utils import EagleDraftInput, EagleVerifyInput

# === InfLLM-v2 稀疏路径依赖 ===
try:
    from sglang.srt.layers.attention.triton_ops.infllmv2 import (
        infllm2_build_kv_indices_from_ranges_triton,  # NEW: Triton 封装
    )
    from sglang.srt.layers.attention.triton_ops.infllmv2 import (
        infllm2_stage1_select_blocks,  # 你原来的 PyTorch 版
    )
    from sglang.srt.layers.attention.triton_ops.infllmv2 import (
        infllm2_stage1_select_blocks_triton,  # NEW: Triton 版（scores kernel）
    )
    from sglang.srt.layers.attention.triton_ops.infllmv2 import (
        _iter_segments_safe,
        infllm2_build_kv_indices_from_ranges,
    )
except Exception:
    infllm2_stage1_select_blocks = None
    infllm2_stage1_select_blocks_triton = None
    infllm2_build_kv_indices_from_ranges = None
    infllm2_build_kv_indices_from_ranges_triton = None

try:
    from sglang.srt.mem_cache.infllmv2_context import (
        ContextMemoryManager,
        KVViewsConfig,
        ensure_c1c2_for_sample,
    )
except Exception:
    ContextMemoryManager = None
    KVViewsConfig = None
    ensure_c1c2_for_sample = None

try:
    from sglang.srt.sparse_utils.range_mapping import blocks_to_token_ranges
except Exception:
    blocks_to_token_ranges = None


def logit_capping_mod(logit_capping_method, logit_cap):
    # positive logit_cap -> tanh cap
    if logit_capping_method == "tanh":
        return logit_cap
    else:
        raise ValueError()


@dataclass
class ForwardMetadata:
    attn_logits: torch.Tensor
    attn_lse: torch.Tensor
    max_extend_len: int
    num_kv_splits: torch.Tensor
    kv_indptr: torch.Tensor
    kv_indices: torch.Tensor
    qo_indptr: torch.Tensor
    custom_mask: torch.Tensor
    mask_indptr: torch.Tensor
    # Sliding window
    window_kv_indptr: torch.Tensor
    window_kv_indices: torch.Tensor
    window_num_kv_splits: torch.Tensor
    window_kv_offsets: torch.Tensor


class TritonAttnBackend(AttentionBackend):
    def __init__(
        self,
        model_runner: ModelRunner,
        skip_prefill: bool = False,
        kv_indptr_buf: Optional[torch.Tensor] = None,
    ):
        # Lazy import to avoid the initialization of cuda context
        from sglang.srt.layers.attention.triton_ops.decode_attention import (
            decode_attention_fwd,
        )
        from sglang.srt.layers.attention.triton_ops.extend_attention import (
            extend_attention_fwd,
        )

        super().__init__()

        self.decode_attention_fwd = torch.compiler.disable(decode_attention_fwd)
        self.extend_attention_fwd = torch.compiler.disable(extend_attention_fwd)

        # Parse args
        self.skip_prefill = skip_prefill
        max_bs = model_runner.req_to_token_pool.size
        self.sliding_window_size = model_runner.sliding_window_size
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.token_to_kv_pool_allocator = model_runner.token_to_kv_pool_allocator
        self.num_draft_tokens = model_runner.server_args.speculative_num_draft_tokens
        self.speculative_num_steps = model_runner.server_args.speculative_num_steps
        self.num_head = (
            model_runner.model_config.num_attention_heads // get_attention_tp_size()
        )
        self.num_kv_head = model_runner.model_config.get_num_kv_heads(
            get_attention_tp_size()
        )
        if model_runner.is_hybrid_gdn:
            # For hybrid linear models, layer_id = 0 may not be full attention
            self.v_head_dim = model_runner.token_to_kv_pool.get_v_head_dim()
        else:
            self.v_head_dim = model_runner.token_to_kv_pool.get_value_buffer(0).shape[
                -1
            ]
        self.max_context_len = model_runner.model_config.context_len
        self.device = model_runner.device
        self.device_core_count = get_device_core_count(model_runner.gpu_id)
        self.static_kv_splits = get_bool_env_var(
            "SGLANG_TRITON_DECODE_ATTN_STATIC_KV_SPLITS", "false"
        )
        self.max_kv_splits = model_runner.server_args.triton_attention_num_kv_splits

        # Decide whether enable deterministic inference with batch-invariant operations
        self.enable_deterministic = (
            model_runner.server_args.enable_deterministic_inference
        )

        # Configure deterministic inference settings
        if self.enable_deterministic:
            # Use fixed split tile size for batch invariance
            self.split_tile_size = get_int_env_var(
                "SGLANG_TRITON_DECODE_SPLIT_TILE_SIZE", 256
            )
            # Set static_kv_splits to False to use deterministic logic instead
            self.static_kv_splits = False
        else:
            self.split_tile_size = (
                model_runner.server_args.triton_attention_split_tile_size
            )

        if self.split_tile_size is not None:
            self.max_kv_splits = (
                self.max_context_len + self.split_tile_size - 1
            ) // self.split_tile_size

        # Check arguments
        assert not (
            model_runner.sliding_window_size is not None
            and model_runner.model_config.is_encoder_decoder
        ), "Sliding window and cross attention are not supported together"

        # Initialize buffers
        # TODO(Jianan Ji): Make sure it behaves as expected when kv_indptr_buf is provided and sliding window is enabled
        if kv_indptr_buf is None:
            self.kv_indptr = torch.zeros(
                (max_bs + 1,), dtype=torch.int32, device=model_runner.device
            )
        else:
            self.kv_indptr = kv_indptr_buf

        # If sliding window is enabled, we might need two sets of buffers
        # because of interleaved attention types (e.g. for Gemma3)
        self.window_kv_indptr = None
        if self.sliding_window_size is not None and self.sliding_window_size > 0:
            if kv_indptr_buf is None:
                self.window_kv_indptr = torch.zeros(
                    (max_bs + 1,), dtype=torch.int32, device=model_runner.device
                )
            else:
                # When provided a buffer, create a clone for the second buffer
                self.window_kv_indptr = torch.zeros_like(kv_indptr_buf)

        if not self.skip_prefill:
            self.qo_indptr = torch.zeros(
                (max_bs + 1,), dtype=torch.int32, device=model_runner.device
            )

            self.mask_indptr = torch.zeros(
                (max_bs + 1,), dtype=torch.int64, device=model_runner.device
            )

        # Initialize forward metadata
        self.forward_metadata: ForwardMetadata = None

        # === InfLLM-v2 开关（不存在也不报错）===
        srv = getattr(model_runner, "server_args", None)
        # 解释：
        #   - Hk*TopK 块的并集，再加上 sink / sliding-window 等冗余，预留一些余量
        #   - 64 是兜底，按需你可以再放大

        self.enable_infllmv2 = bool(getattr(srv, "enable_infllmv2", False))
        if self.enable_infllmv2 and (
            ContextMemoryManager is None
            or KVViewsConfig is None
            or blocks_to_token_ranges is None
        ):
            # 自动降级
            self.enable_infllmv2 = False

        if self.enable_infllmv2:
            self.infllm2_topk_blocks = int(getattr(srv, "infllmv2_topk_blocks", 4))
            self.infllm2_sink_tokens = int(getattr(srv, "infllmv2_sink_tokens", 64))
            self.infllm2_sw_span = int(getattr(srv, "infllmv2_sw_span", 256))
            self.infllm2_cfg = KVViewsConfig(block_size=64)
            self.infllm2_ctx_mgr = ContextMemoryManager(
                self.infllm2_cfg, device=self.device, dtype=torch.float16
            )
            self.infllm2_R_cap = max(
                self.num_kv_head * self.infllm2_topk_blocks * 2 + 8, 64
            )
        else:
            self.infllm2_topk_blocks = 0
            self.infllm2_sink_tokens = 0
            self.infllm2_sw_span = 0
            self.infllm2_cfg = None
            self.infllm2_ctx_mgr = None

    def get_num_kv_splits(
        self,
        num_kv_splits: torch.Tensor,
        seq_lens: torch.Tensor,
    ):
        num_token, num_seq = num_kv_splits.shape[0], seq_lens.shape[0]
        # NOTE(alcanderian): Considering speculative_decodeing,
        # num_kv_splits.shape[0] will be topk * real_num_token.
        # And the real_num_token is num_seq in decoding phase.
        num_group = num_token // num_seq

        assert (
            num_group * num_seq == num_token
        ), f"num_seq({num_seq}), num_token({num_token}), something goes wrong!"

        # Legacy dynamic splitting logic (non-deterministic)
        if (
            self.static_kv_splits or self.device_core_count <= 0
        ) and not self.enable_deterministic:
            num_kv_splits.fill_(self.max_kv_splits)
            return

        # deterministic
        if self.split_tile_size is not None and self.enable_deterministic:
            # expand seq_lens to match num_token
            if num_group > 1:
                expanded_seq_lens = seq_lens.repeat_interleave(num_group)
            else:
                expanded_seq_lens = seq_lens

            num_kv_splits[:] = (
                expanded_seq_lens + self.split_tile_size - 1
            ) // self.split_tile_size
            return

        if num_seq < 256:
            SCHEDULE_SEQ = 256
        else:
            SCHEDULE_SEQ = triton.next_power_of_2(num_seq)

        get_num_kv_splits_triton[(1,)](
            num_kv_splits,
            seq_lens,
            num_seq,
            num_group,
            self.num_head,
            self.num_kv_head,
            self.max_kv_splits,
            self.device_core_count,
            MAX_NUM_SEQ=SCHEDULE_SEQ,
        )

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Init auxiliary variables for triton attention backend."""

        bs = forward_batch.batch_size
        kv_indptr = self.kv_indptr
        window_kv_indptr = self.window_kv_indptr
        window_kv_indices = None
        window_num_kv_splits = None
        window_kv_offsets = None
        spec_info = forward_batch.spec_info

        if forward_batch.forward_mode.is_decode_or_idle():
            if spec_info is None:
                kv_indptr[1 : bs + 1] = torch.cumsum(forward_batch.seq_lens, dim=0)
                kv_indptr = kv_indptr[: bs + 1]
                kv_indices = torch.empty(
                    forward_batch.seq_lens_sum, dtype=torch.int64, device=self.device
                )
                create_flashinfer_kv_indices_triton[(bs,)](
                    self.req_to_token,
                    forward_batch.req_pool_indices,
                    forward_batch.seq_lens,
                    kv_indptr,
                    None,
                    kv_indices,
                    self.req_to_token.stride(0),
                )
                # Sliding window
                if (
                    self.sliding_window_size is not None
                    and self.sliding_window_size > 0
                ):
                    window_kv_indptr, window_kv_indices, window_kv_lens, _ = (
                        update_sliding_window_buffer(
                            self.window_kv_indptr,
                            self.req_to_token,
                            self.sliding_window_size,
                            forward_batch.seq_lens,
                            forward_batch.req_pool_indices,
                            bs,
                            self.device,
                            self.token_to_kv_pool_allocator,
                        )
                    )
                    window_num_kv_splits = torch.empty(
                        (bs,), dtype=torch.int32, device=self.device
                    )
                    self.get_num_kv_splits(window_num_kv_splits, window_kv_lens)
            else:
                kv_indptr, kv_indices = spec_info.kv_indptr, spec_info.kv_indices
                bs = kv_indptr.shape[0] - 1

            attn_logits = torch.empty(
                (bs, self.num_head, self.max_kv_splits, self.v_head_dim),
                dtype=torch.float32,
                device=self.device,
            )
            attn_lse = torch.empty(
                (bs, self.num_head, self.max_kv_splits),
                dtype=torch.float32,
                device=self.device,
            )
            num_kv_splits = torch.empty((bs,), dtype=torch.int32, device=self.device)
            self.get_num_kv_splits(num_kv_splits, forward_batch.seq_lens)

            qo_indptr = None
            custom_mask = None
            mask_indptr = None
            max_extend_len = None
        elif forward_batch.forward_mode.is_target_verify():
            bs = len(forward_batch.req_pool_indices)
            qo_indptr = torch.arange(
                0,
                (1 + bs) * self.num_draft_tokens,
                step=self.num_draft_tokens,
                dtype=torch.int32,
                device=self.device,
            )
            # Different with flashinfer kv_indptr and kv_indices construction
            kv_indptr[1 : bs + 1] = torch.cumsum(forward_batch.seq_lens, dim=0)
            kv_indptr = kv_indptr[: bs + 1]
            kv_indices = torch.empty(
                kv_indptr[-1], dtype=torch.int64, device=self.device
            )
            create_flashinfer_kv_indices_triton[(bs,)](
                self.req_to_token,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                kv_indptr,
                None,
                kv_indices,
                self.req_to_token.stride(0),
            )

            if self.sliding_window_size is not None and self.sliding_window_size > 0:
                # window_kv_offsets is used to calculate the start position in custom mask
                (
                    window_kv_indptr,
                    window_kv_indices,
                    window_kv_lens,
                    window_kv_offsets,
                ) = update_sliding_window_buffer(
                    self.window_kv_indptr,
                    self.req_to_token,
                    self.sliding_window_size,
                    forward_batch.seq_lens,
                    forward_batch.req_pool_indices,
                    bs,
                    self.device,
                    self.token_to_kv_pool_allocator,
                )

            custom_mask = spec_info.custom_mask
            seq_mask_len = self.num_draft_tokens * (
                forward_batch.seq_lens + self.num_draft_tokens
            )
            mask_indptr = self.mask_indptr
            mask_indptr[1 : bs + 1] = torch.cumsum(seq_mask_len[:bs], dim=0)
            mask_indptr = mask_indptr[: bs + 1]
            max_extend_len = self.num_draft_tokens
            num_kv_splits = None
            attn_logits = None
            attn_lse = None

        elif forward_batch.forward_mode.is_draft_extend():
            kv_indices, kv_indptr, qo_indptr, custom_mask = (
                spec_info.generate_attn_arg_prefill(
                    forward_batch.req_pool_indices,
                    forward_batch.seq_lens,
                    None,
                    self.req_to_token,
                )
            )
            kv_indices = kv_indices.to(torch.int64)
            mask_indptr = None
            # TODO(FIXME): This will trigger an invalid Eagle tree when using
            # `max(spec_info.accept_length_cpu)`.
            # It might have been forgotten to update somewhere.
            max_extend_len = torch.max(spec_info.accept_length).item()
            num_kv_splits = None
            attn_logits = None
            attn_lse = None
        else:
            kv_indptr[1 : bs + 1] = torch.cumsum(
                forward_batch.extend_prefix_lens, dim=0
            )
            kv_indptr = kv_indptr[: bs + 1]
            kv_indices = torch.empty(
                forward_batch.extend_prefix_lens.sum().item(),
                dtype=torch.int64,
                device=self.device,
            )
            create_flashinfer_kv_indices_triton[(bs,)](
                self.req_to_token,
                forward_batch.req_pool_indices,
                forward_batch.extend_prefix_lens,
                kv_indptr,
                None,
                kv_indices,
                self.req_to_token.stride(0),
            )
            # Sliding window
            if self.sliding_window_size is not None and self.sliding_window_size > 0:
                window_kv_indptr, window_kv_indices, _, _ = (
                    update_sliding_window_buffer(
                        self.window_kv_indptr,
                        self.req_to_token,
                        self.sliding_window_size,
                        forward_batch.extend_prefix_lens,
                        forward_batch.req_pool_indices,
                        bs,
                        self.device,
                        self.token_to_kv_pool_allocator,
                    )
                )

            qo_indptr = self.qo_indptr
            qo_indptr[1 : bs + 1] = torch.cumsum(forward_batch.extend_seq_lens, dim=0)
            qo_indptr = qo_indptr[: bs + 1]
            custom_mask = None
            mask_indptr = None
            attn_logits = None
            attn_lse = None
            max_extend_len = max(forward_batch.extend_seq_lens_cpu)
            num_kv_splits = None

        self.forward_metadata = ForwardMetadata(
            attn_logits,
            attn_lse,
            max_extend_len,
            num_kv_splits,
            kv_indptr,
            kv_indices,
            qo_indptr,
            custom_mask,
            mask_indptr,
            window_kv_indptr,
            window_kv_indices,
            window_num_kv_splits,
            window_kv_offsets,
        )

    def init_cuda_graph_state(
        self,
        max_bs: int,
        max_num_tokens: int,
        kv_indices_buf: Optional[torch.Tensor] = None,
    ):
        self.cuda_graph_attn_logits = torch.zeros(
            (max_num_tokens, self.num_head, self.max_kv_splits, self.v_head_dim),
            dtype=torch.float32,
            device=self.device,
        )
        self.cuda_graph_attn_lse = torch.zeros(
            (max_num_tokens, self.num_head, self.max_kv_splits),
            dtype=torch.float32,
            device=self.device,
        )
        self.cuda_graph_num_kv_splits = torch.full(
            (max_num_tokens,), self.max_kv_splits, dtype=torch.int32, device=self.device
        )
        if kv_indices_buf is None:
            self.cuda_graph_kv_indices = torch.zeros(
                (max_num_tokens * self.max_context_len),
                dtype=torch.int64,
                device=self.device,
            )
        else:
            self.cuda_graph_kv_indices = kv_indices_buf

        if not self.skip_prefill:
            self.cuda_graph_custom_mask = torch.zeros(
                (max_num_tokens * self.max_context_len),
                dtype=torch.uint8,
                device=self.device,
            )

        if self.sliding_window_size is not None and self.sliding_window_size > 0:
            if kv_indices_buf is None:
                self.cuda_graph_window_kv_indices = torch.zeros(
                    (max_num_tokens * self.sliding_window_size),
                    dtype=torch.int64,
                    device=self.device,
                )
            else:
                self.cuda_graph_window_kv_indices = torch.zeros_like(kv_indices_buf)

            self.cuda_graph_window_num_kv_splits = torch.full(
                (max_num_tokens,),
                self.max_kv_splits,
                dtype=torch.int32,
                device=self.device,
            )

            self.cuda_graph_window_kv_offsets = torch.zeros(
                (max_bs,),
                dtype=torch.int32,
                device=self.device,
            )

        if self.enable_infllmv2:
            self.infllm2_range_starts = torch.zeros(
                (max_bs * self.infllm2_R_cap,), dtype=torch.int32, device=self.device
            )
            self.infllm2_range_lens = torch.zeros_like(self.infllm2_range_starts)
            self.infllm2_range_indptr = torch.zeros(
                (max_bs + 1,), dtype=torch.int32, device=self.device
            )

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]],
    ):
        assert encoder_lens is None, "Not supported"
        window_kv_indptr = self.window_kv_indptr
        window_kv_indices = None
        window_num_kv_splits = None
        window_kv_offsets = None

        if forward_mode.is_decode_or_idle():
            if spec_info is None:
                kv_indptr = self.kv_indptr
                kv_indptr[1 : bs + 1] = torch.cumsum(seq_lens, dim=0)
                kv_indptr = kv_indptr[: bs + 1]
                kv_indices = self.cuda_graph_kv_indices
                create_flashinfer_kv_indices_triton[(bs,)](
                    self.req_to_token,
                    req_pool_indices,
                    seq_lens,
                    kv_indptr,
                    None,
                    kv_indices,
                    self.req_to_token.stride(0),
                )
                if (
                    self.sliding_window_size is not None
                    and self.sliding_window_size > 0
                ):
                    window_kv_indices = self.cuda_graph_window_kv_indices
                    window_num_kv_splits = self.cuda_graph_window_num_kv_splits
                    window_kv_indptr, window_kv_indices, _, _ = (
                        update_sliding_window_buffer_cuda_graph(
                            self.window_kv_indptr,
                            window_kv_indices,
                            self.req_to_token,
                            self.sliding_window_size,
                            seq_lens[:bs],
                            req_pool_indices,
                            bs,
                            self.token_to_kv_pool_allocator,
                        )
                    )
            else:
                kv_indptr, kv_indices = spec_info.kv_indptr, spec_info.kv_indices

            attn_logits = self.cuda_graph_attn_logits
            attn_lse = self.cuda_graph_attn_lse
            max_extend_len = None
            num_kv_splits = self.cuda_graph_num_kv_splits
            qo_indptr = None
            custom_mask = None
            mask_indptr = None
        elif forward_mode.is_target_verify():
            qo_indptr = self.qo_indptr[: bs + 1]
            qo_indptr[: bs + 1] = torch.arange(
                0,
                (1 + bs) * self.num_draft_tokens,
                step=self.num_draft_tokens,
                dtype=torch.int32,
                device=self.device,
            )
            kv_indptr = self.kv_indptr[: bs + 1]
            kv_indptr[1 : bs + 1] = torch.cumsum(seq_lens, dim=0)
            kv_indices = self.cuda_graph_kv_indices
            create_flashinfer_kv_indices_triton[(bs,)](
                self.req_to_token,
                req_pool_indices,
                seq_lens,
                kv_indptr,
                None,
                kv_indices,
                self.req_to_token.stride(0),
            )

            if self.sliding_window_size is not None and self.sliding_window_size > 0:
                window_kv_indices = self.cuda_graph_window_kv_indices
                window_num_kv_splits = self.cuda_graph_window_num_kv_splits
                window_kv_offsets = self.cuda_graph_window_kv_offsets
                window_kv_indptr, window_kv_indices, _, window_kv_offsets[:bs] = (
                    update_sliding_window_buffer_cuda_graph(
                        self.window_kv_indptr,
                        window_kv_indices,
                        self.req_to_token,
                        self.sliding_window_size,
                        seq_lens[:bs],
                        req_pool_indices,
                        bs,
                        self.token_to_kv_pool_allocator,
                    )
                )

            custom_mask = self.cuda_graph_custom_mask
            custom_mask[: spec_info.custom_mask.shape[0]] = spec_info.custom_mask
            seq_mask_len = self.num_draft_tokens * (seq_lens + self.num_draft_tokens)
            mask_indptr = self.mask_indptr[: bs + 1]
            mask_indptr[1 : bs + 1] = torch.cumsum(seq_mask_len, dim=0)
            max_extend_len = self.num_draft_tokens
            num_kv_splits = None
            attn_logits = None
            attn_lse = None
        elif forward_mode.is_draft_extend():
            num_tokens_per_bs = self.speculative_num_steps + 1
            qo_indptr = self.qo_indptr[: bs + 1]
            qo_indptr[: bs + 1] = torch.arange(
                0,
                bs * num_tokens_per_bs + 1,
                step=num_tokens_per_bs,
                dtype=torch.int32,
                device=self.device,
            )
            kv_indptr = self.kv_indptr[: bs + 1]
            kv_indptr[1 : bs + 1] = torch.cumsum(seq_lens, dim=0)
            kv_indices = self.cuda_graph_kv_indices
            create_flashinfer_kv_indices_triton[(bs,)](
                self.req_to_token,
                req_pool_indices,
                seq_lens,
                kv_indptr,
                None,
                kv_indices,
                self.req_to_token.stride(0),
            )
            custom_mask = None
            mask_indptr = None
            max_extend_len = num_tokens_per_bs
            num_kv_splits = None
            attn_logits = None
            attn_lse = None
        else:
            raise ValueError(
                f"Invalid forward mode: {forward_mode=} for CUDA Graph capture."
            )

        self.forward_metadata = ForwardMetadata(
            attn_logits,
            attn_lse,
            max_extend_len,
            num_kv_splits,
            kv_indptr,
            kv_indices,
            qo_indptr,
            custom_mask,
            mask_indptr,
            window_kv_indptr,
            window_kv_indices,
            window_num_kv_splits,
            window_kv_offsets,
        )

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]],
        seq_lens_cpu: Optional[torch.Tensor],
    ):
        # NOTE: encoder_lens expected to be zeros or None
        if forward_mode.is_decode_or_idle():
            # Update kv_indptr, kv_indices
            kv_indptr = self.kv_indptr
            kv_indices = self.cuda_graph_kv_indices
            num_kv_splits = self.cuda_graph_num_kv_splits
            if spec_info is None:
                kv_indptr[1 : bs + 1] = torch.cumsum(seq_lens[:bs], dim=0)
                kv_indptr = kv_indptr[: bs + 1]
                create_flashinfer_kv_indices_triton[(bs,)](
                    self.req_to_token,
                    req_pool_indices[:bs],
                    seq_lens[:bs],
                    kv_indptr,
                    None,
                    kv_indices,
                    self.req_to_token.stride(0),
                )
                num_token = bs
                if (
                    self.sliding_window_size is not None
                    and self.sliding_window_size > 0
                ):
                    window_num_kv_splits = self.cuda_graph_window_num_kv_splits
                    window_kv_indices = self.cuda_graph_window_kv_indices
                    _, _, window_kv_lens, _ = update_sliding_window_buffer_cuda_graph(
                        self.window_kv_indptr,
                        window_kv_indices,
                        self.req_to_token,
                        self.sliding_window_size,
                        seq_lens[:bs],
                        req_pool_indices[:bs],
                        bs,
                        self.token_to_kv_pool_allocator,
                    )
                    self.get_num_kv_splits(
                        window_num_kv_splits[:num_token], window_kv_lens[:bs]
                    )

            else:
                kv_indptr[: spec_info.kv_indptr.shape[0]] = spec_info.kv_indptr
                kv_indices[: spec_info.kv_indices.shape[0]] = spec_info.kv_indices
                num_token = spec_info.kv_indptr.shape[0] - 1
            self.get_num_kv_splits(num_kv_splits[:num_token], seq_lens[:bs])

        elif forward_mode.is_target_verify():
            # Update qo_indptr, kv_indptr, kv_indices, custom_mask, mask_indptr
            bs = len(req_pool_indices)
            qo_indptr = self.qo_indptr[: bs + 1]
            qo_indptr[: bs + 1] = torch.arange(
                0,
                (1 + bs) * self.num_draft_tokens,
                step=self.num_draft_tokens,
                dtype=torch.int32,
                device=self.device,
            )
            kv_indptr = self.kv_indptr[: bs + 1]
            kv_indptr[1 : bs + 1] = torch.cumsum(seq_lens, dim=0)
            kv_indices = self.cuda_graph_kv_indices
            create_flashinfer_kv_indices_triton[(bs,)](
                self.req_to_token,
                req_pool_indices,
                seq_lens,
                kv_indptr,
                None,
                kv_indices,
                self.req_to_token.stride(0),
            )
            if self.sliding_window_size is not None and self.sliding_window_size > 0:
                window_num_kv_splits = self.cuda_graph_window_num_kv_splits
                window_kv_indices = self.cuda_graph_window_kv_indices
                window_kv_offsets = self.cuda_graph_window_kv_offsets
                _, _, window_kv_lens, window_kv_offsets[:bs] = (
                    update_sliding_window_buffer_cuda_graph(
                        self.window_kv_indptr,
                        window_kv_indices,
                        self.req_to_token,
                        self.sliding_window_size,
                        seq_lens[:bs],
                        req_pool_indices,
                        bs,
                        self.token_to_kv_pool_allocator,
                    )
                )
            custom_mask = self.cuda_graph_custom_mask
            custom_mask[: spec_info.custom_mask.shape[0]] = spec_info.custom_mask
            seq_mask_len = self.num_draft_tokens * (seq_lens + self.num_draft_tokens)
            mask_indptr = self.mask_indptr[: bs + 1]
            mask_indptr[1 : bs + 1] = torch.cumsum(seq_mask_len, dim=0)
        elif forward_mode.is_draft_extend():
            seq_lens = seq_lens[:bs]
            accept_lens = spec_info.accept_length[:bs]
            qo_indptr = self.qo_indptr[: bs + 1]
            qo_indptr[1 : bs + 1] = torch.cumsum(accept_lens, dim=0)
            kv_indptr = self.kv_indptr[: bs + 1]
            kv_indptr[1 : bs + 1] = torch.cumsum(seq_lens, dim=0)
            kv_indices = self.cuda_graph_kv_indices
            create_flashinfer_kv_indices_triton[(bs,)](
                self.req_to_token,
                req_pool_indices,
                seq_lens,
                kv_indptr,
                None,
                kv_indices,
                self.req_to_token.stride(0),
            )
        else:
            raise ValueError(
                f"Invalid forward mode: {forward_mode=} for CUDA Graph replay."
            )

    def get_cuda_graph_seq_len_fill_value(self):
        return 1

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        sinks=None,
    ):
        # TODO: reuse the buffer across layers
        if layer.qk_head_dim != layer.v_head_dim:
            o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
        else:
            o = torch.empty_like(q)

        if save_kv_cache:
            t2k = forward_batch.token_to_kv_pool
            cache_label = getattr(forward_batch, "cache_label", 0)
            try:
                t2k.set_kv_buffer(layer, forward_batch.out_cache_loc, k, v, cache_label)
            except TypeError:
                t2k.set_kv_buffer(layer, forward_batch.out_cache_loc, k, v)

        logits_soft_cap = logit_capping_mod(layer.logit_capping_method, layer.logit_cap)

        causal = True
        if layer.attn_type == AttentionType.ENCODER_ONLY:
            causal = False

        if layer.sliding_window_size is not None and layer.sliding_window_size > -1:
            sliding_window_size = (
                layer.sliding_window_size
            )  # Needed for sliding window mask
            kv_indptr = self.forward_metadata.window_kv_indptr
            kv_indices = self.forward_metadata.window_kv_indices
            window_kv_offsets = self.forward_metadata.window_kv_offsets
        else:
            sliding_window_size = -1
            kv_indptr = self.forward_metadata.kv_indptr
            kv_indices = self.forward_metadata.kv_indices
            window_kv_offsets = None

        if self._should_use_infllm2(layer, forward_batch, mode="extend"):
            return self._infllm2_forward_triton(
                q, k, v, layer, forward_batch, sinks=sinks, mode="extend"
            )

        t2k = forward_batch.token_to_kv_pool
        get_k = getattr(t2k, "get_key_buffer")
        get_v = getattr(t2k, "get_value_buffer")
        try:
            kbuf = get_k(layer.layer_id)
            vbuf = get_v(layer.layer_id)
        except TypeError:
            cache_label = getattr(forward_batch, "cache_label", 0)
            kbuf = get_k(layer.layer_id, cache_label)
            vbuf = get_v(layer.layer_id, cache_label)

        self.extend_attention_fwd(
            q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
            k.contiguous(),
            v.contiguous(),
            o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
            kbuf,
            vbuf,
            self.forward_metadata.qo_indptr,
            kv_indptr,
            kv_indices,
            self.forward_metadata.custom_mask,
            causal,
            self.forward_metadata.mask_indptr,
            self.forward_metadata.max_extend_len,
            layer.scaling,
            logit_cap=logits_soft_cap,
            sliding_window_size=sliding_window_size,
            sinks=sinks,
            window_kv_offsets=window_kv_offsets,
            xai_temperature_len=layer.xai_temperature_len,
        )
        return o

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        sinks=None,
    ):
        # During torch.compile, there is a bug in rotary_emb that causes the
        # output value to have a 3D tensor shape. This reshapes the output correctly.
        q = q.reshape(-1, layer.tp_q_head_num * layer.qk_head_dim)

        # TODO: reuse the buffer across layers
        if layer.qk_head_dim != layer.v_head_dim:
            o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
        else:
            o = torch.empty_like(q)

        logits_soft_cap = logit_capping_mod(layer.logit_capping_method, layer.logit_cap)

        if save_kv_cache:
            t2k = forward_batch.token_to_kv_pool
            cache_label = getattr(forward_batch, "cache_label", 0)
            try:
                t2k.set_kv_buffer(layer, forward_batch.out_cache_loc, k, v, cache_label)
            except TypeError:
                t2k.set_kv_buffer(layer, forward_batch.out_cache_loc, k, v)

        if layer.sliding_window_size is not None and layer.sliding_window_size > -1:
            kv_indptr = self.forward_metadata.window_kv_indptr
            kv_indices = self.forward_metadata.window_kv_indices
        else:
            kv_indptr = self.forward_metadata.kv_indptr
            kv_indices = self.forward_metadata.kv_indices

        # --- InfLLM-v2 稀疏路径 ---
        if self._should_use_infllm2(layer, forward_batch, mode="decode"):
            return self._infllm2_forward_triton(
                q, k, v, layer, forward_batch, sinks=sinks, mode="decode"
            )

        t2k = forward_batch.token_to_kv_pool
        get_k = getattr(t2k, "get_key_buffer")
        get_v = getattr(t2k, "get_value_buffer")
        try:
            kbuf = get_k(layer.layer_id)
            vbuf = get_v(layer.layer_id)
        except TypeError:
            cache_label = getattr(forward_batch, "cache_label", 0)
            kbuf = get_k(layer.layer_id, cache_label)
            vbuf = get_v(layer.layer_id, cache_label)

        self.decode_attention_fwd(
            q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
            kbuf,
            vbuf,
            o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
            kv_indptr,
            kv_indices,
            self.forward_metadata.attn_logits,
            self.forward_metadata.attn_lse,
            self.forward_metadata.num_kv_splits,
            self.max_kv_splits,
            layer.scaling,
            logit_cap=logits_soft_cap,
            sinks=sinks,
            xai_temperature_len=layer.xai_temperature_len,
        )
        return o

    def _should_use_infllm2(
        self, layer, forward_batch: ForwardBatch, mode: str
    ) -> bool:
        if not self.enable_infllmv2:
            return False
        if layer.attn_type == AttentionType.ENCODER_ONLY:  # cross-attn 暂不走稀疏
            return False
        if (
            forward_batch.spec_info is not None
            and self.num_draft_tokens
            and self.num_draft_tokens > 0
        ):
            # 先不支持 Spec-Decode 的级联，在这条后端回退 dense
            return False
        if (
            infllm2_stage1_select_blocks is None
            or infllm2_build_kv_indices_from_ranges is None
        ):
            return False
        return True

    @torch.no_grad()
    def _infllm2_forward_triton(
        self, q, k, v, layer, forward_batch: ForwardBatch, *, sinks=None, mode: str
    ):
        device = q.device
        bs = forward_batch.batch_size

        # --- 0) 池句柄（兼容 DoubleSparse 的 cache_label）---
        t2k = forward_batch.token_to_kv_pool
        get_k = getattr(t2k, "get_key_buffer")
        get_v = getattr(t2k, "get_value_buffer")
        try:
            key_pool = get_k(layer.layer_id)
            value_pool = get_v(layer.layer_id)
        except TypeError:
            cache_label = getattr(forward_batch, "cache_label", 0)
            key_pool = get_k(layer.layer_id, cache_label)
            value_pool = get_v(layer.layer_id, cache_label)

        # 捕获期：严禁新分配，直接回退 dense
        def _is_capturing() -> bool:
            f = getattr(torch.cuda, "is_current_stream_capturing", None)
            if callable(f):
                return bool(f())
            g = getattr(torch.cuda, "_is_in_graph_capture_mode", None)
            return bool(g()) if callable(g) else False

        if _is_capturing():
            kv_indptr = self.forward_metadata.kv_indptr
            kv_indices = self.forward_metadata.kv_indices
            num_kv_splits = self.forward_metadata.num_kv_splits

            if layer.qk_head_dim != layer.v_head_dim:
                o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
            else:
                o = torch.empty_like(q)

            logits_soft_cap = logit_capping_mod(
                layer.logit_capping_method, layer.logit_cap
            )
            if mode == "decode":
                self.decode_attention_fwd(
                    q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
                    key_pool,
                    value_pool,
                    o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
                    kv_indptr,
                    kv_indices,
                    self.forward_metadata.attn_logits,
                    self.forward_metadata.attn_lse,
                    num_kv_splits,
                    self.max_kv_splits,
                    layer.scaling,
                    logit_cap=logits_soft_cap,
                    sinks=sinks,
                    xai_temperature_len=layer.xai_temperature_len,
                )
            else:
                causal = layer.attn_type != AttentionType.ENCODER_ONLY
                self.extend_attention_fwd(
                    q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
                    k.contiguous(),
                    v.contiguous(),
                    o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
                    key_pool,
                    value_pool,
                    self.forward_metadata.qo_indptr,
                    kv_indptr,
                    kv_indices,
                    self.forward_metadata.custom_mask,
                    causal,
                    self.forward_metadata.mask_indptr,
                    self.forward_metadata.max_extend_len,
                    layer.scaling,
                    logit_cap=logits_soft_cap,
                    sliding_window_size=-1,
                    sinks=sinks,
                    window_kv_offsets=None,
                    xai_temperature_len=layer.xai_temperature_len,
                )
            return o

        # === 非捕获期：InfLLM-v2 ===

        # 1) 逐样本拿“完整历史”的 token 索引 -> 直接从 req_to_token 切片（不写旧 kv_indices 缓冲）
        seq_lens = forward_batch.seq_lens  # [B]
        req_row = forward_batch.req_pool_indices.to(torch.int64)

        # 2) C1/C2/offs（由 ctx_mgr 维护；要求不返回 None）
        c1_list, c2_list, offs_list = [], [], []
        sc1_lens, sc2_lens = [], []
        req_id = getattr(forward_batch, "request_ids", None)

        for b in range(bs):
            rid = int(req_row[b].item())
            S = int(seq_lens[b].item())

            if S > 0:
                idx = self.req_to_token[rid, :S].to(torch.int64)
                # Page/SWA 映射（若启用）
                if hasattr(
                    self.token_to_kv_pool_allocator, "translate_loc_from_full_to_swa"
                ):
                    idx = (
                        self.token_to_kv_pool_allocator.translate_loc_from_full_to_swa(
                            idx
                        )
                    )
                K_seq = key_pool.index_select(0, idx)  # [S,Hk,D]
            else:
                K_seq = key_pool[:0]  # 空视图

            rid_str = str(req_id[b]) if req_id is not None else f"req_{rid}"
            c1, c2, offs, L1, L2 = ensure_c1c2_for_sample(
                self.infllm2_ctx_mgr, rid_str, layer.layer_id, K_seq=K_seq
            )
            assert (
                (c1 is not None) and (c2 is not None) and (offs is not None)
            ), "ensure_c1c2_for_sample must not return None"

            c1_list.append(c1)
            c2_list.append(c2)
            offs_list.append(offs)
            sc1_lens.append(int(L1))
            sc2_lens.append(int(L2))

        # 2.5) 右侧 pad & cat（不在捕获期，安全）
        L1_max = max(1, max(sc1_lens) if sc1_lens else 1)
        L2_max = max(1, max(sc2_lens) if sc2_lens else 1)
        nmax = max(o.shape[-1] for o in offs_list) if offs_list else 2

        def _pad_blocks(x, L_max):
            if x.shape[2] == L_max:
                return x
            return torch.nn.functional.pad(x, (0, 0, 0, L_max - x.shape[2]))

        def _pad_offs(o, n_max):
            if o.shape[-1] == n_max:
                return o
            last = o[..., -1:]
            return torch.cat([o, last.repeat(1, 1, n_max - o.shape[-1])], dim=-1)

        c1 = torch.cat([_pad_blocks(t, L1_max) for t in c1_list], dim=0)  # [B,Hk,L1,D]
        c2 = torch.cat([_pad_blocks(t, L2_max) for t in c2_list], dim=0)  # [B,Hk,L2,D]
        offs = torch.cat([_pad_offs(o, nmax) for o in offs_list], dim=0)  # [B,1,n]
        sc1_len = torch.tensor(sc1_lens, device=device, dtype=torch.int32)  # [B]

        # 3) Stage-1 Top-K（多 token 段聚合 + GQA 聚合）
        Hq, Dq = layer.tp_q_head_num, layer.qk_head_dim
        Hk = layer.tp_k_head_num
        q_tok_hqd = q.view(-1, Hq, Dq)
        qo = getattr(self.forward_metadata, "qo_indptr", None)

        if qo is None:
            assert (
                q_tok_hqd.shape[0] == bs
            ), "decode path expects T==B when no qo_indptr"
            q_seg_bhd = q_tok_hqd
        else:
            qo = qo[: bs + 1]
            chunks = []
            T = q_tok_hqd.shape[0]
            for b in range(bs):
                s = int(qo[b].item())
                e = int(qo[b + 1].item())
                if e <= s:
                    chunks.append(
                        torch.zeros((1, Hq, Dq), device=q.device, dtype=q.dtype)
                    )
                else:
                    s = max(0, min(s, T))
                    e = max(0, min(e, T))
                    chunks.append(q_tok_hqd[s:e].sum(dim=0, keepdim=True))
            q_seg_bhd = torch.cat(chunks, dim=0)

        assert Hq % Hk == 0, f"GQA mismatch: Hq={Hq}, Hk={Hk}"
        group = Hq // Hk
        q_hkd = q_seg_bhd.view(bs, Hk, group, Dq).mean(dim=2)  # [B,Hk,D]
        valid_sc1_bh = sc1_len.view(-1, 1).expand(-1, c1.shape[1]).contiguous()

        fn_stage1 = globals().get("infllm2_stage1_select_blocks_triton", None)
        if callable(fn_stage1):
            topk_idx = fn_stage1(
                q_hkd,
                c1,
                valid_sc1_bh,
                softmax_scale=layer.scaling,
                topk=self.infllm2_topk_blocks,
            )
        else:
            topk_idx = infllm2_stage1_select_blocks(
                q_hkd,
                c1,
                None,
                offs,
                valid_sc1_bh,
                topk=self.infllm2_topk_blocks,
                softmax_scale=layer.scaling,
            )  # [B,Hk,K]

        # 4) 块 -> 样本级 token 区间（并 ∪ sink / sliding window）
        if topk_idx.dim() == 3:
            topk_idx_bshk = topk_idx.unsqueeze(1)  # [B,1,Hk,K]
        else:
            topk_idx_bshk = topk_idx  # [B,Sq,Hk,K]

        offs_bhn = (
            offs.view(bs, 1, -1).expand(bs, c1.shape[1], -1).contiguous()
        )  # [B,Hk,n+1]
        ranges = blocks_to_token_ranges(
            topk_idx_bshk=topk_idx_bshk,
            c1_block_offsets_bhk=offs_bhn,
            block_size=self.infllm2_cfg.block_size,
            sink_len=int(self.infllm2_sink_tokens),
            sw_span=int(self.infllm2_sw_span),
            Sk_vis=int(seq_lens.max().item()),
        )

        # 5) ranges -> indices （落到预分配 workspace）
        Rcap = getattr(self, "infllm2_R_cap", 64)
        if not hasattr(self, "infllm2_range_starts"):
            # 非捕获期允许延迟分配
            self.infllm2_range_starts = torch.zeros(
                (max(bs, 1) * Rcap,), dtype=torch.int32, device=device
            )
            self.infllm2_range_lens = torch.zeros_like(self.infllm2_range_starts)
            self.infllm2_range_indptr = torch.zeros(
                (max(bs, 1) + 1,), dtype=torch.int32, device=device
            )

        self.infllm2_range_indptr.zero_()
        self.infllm2_range_starts.zero_()
        self.infllm2_range_lens.zero_()

        kv_indptr_out = self.forward_metadata.kv_indptr  # 用它承载“稀疏”的 indptr
        kv_indptr_out.zero_()

        tok_total, r_ptr = 0, 0
        for b in range(bs):
            self.infllm2_range_indptr[b] = r_ptr
            for s, e in _iter_segments_safe(ranges[b]):
                n = e - s
                if n <= 0:
                    continue
                if r_ptr >= (b + 1) * Rcap:
                    break
                self.infllm2_range_starts[r_ptr] = int(s)
                self.infllm2_range_lens[r_ptr] = int(n)
                r_ptr += 1
                tok_total += n
            kv_indptr_out[b + 1] = tok_total
        self.infllm2_range_indptr[bs] = r_ptr

        from sglang.srt.layers.attention.triton_ops.infllmv2 import (
            _ranges_to_indices_kernel,
        )

        req = self.req_to_token
        _ranges_to_indices_kernel[(bs,)](
            req,
            req.stride(0),
            req.stride(1),
            forward_batch.req_pool_indices.to(dtype=torch.int32, device=req.device),
            self.infllm2_range_indptr,
            self.infllm2_range_starts,
            self.infllm2_range_lens,
            kv_indptr_out,  # << 稀疏 indptr 已写好
            self.cuda_graph_kv_indices,  # << 稀疏 indices 写到预分配大缓冲
            pool_w=req.shape[1],
            BLOCK=128,
        )

        # ---------- 只在 decode 需要 num_kv_splits ----------
        num_kv_splits_local = None
        if mode == "decode":
            # 稀疏长度 -> splits
            seq_lens_sparse = (kv_indptr_out[1:] - kv_indptr_out[:-1]).to(torch.int32)

            # 优先复用 forward_metadata 的 buffer；没有就临时分配一个（非捕获期安全）
            num_kv_splits_local = self.forward_metadata.num_kv_splits
            if (num_kv_splits_local is None) or (num_kv_splits_local.numel() < bs):
                num_kv_splits_local = torch.empty(
                    (bs,), dtype=torch.int32, device=device
                )

            self.get_num_kv_splits(num_kv_splits_local[:bs], seq_lens_sparse[:bs])
        # -----------------------------------------------

        # 统一使用我们刚写好的稀疏 indptr/indices
        kv_indptr = kv_indptr_out
        kv_indices = self.cuda_graph_kv_indices
        logits_soft_cap = logit_capping_mod(layer.logit_capping_method, layer.logit_cap)

        # 输出缓冲
        if layer.qk_head_dim != layer.v_head_dim:
            o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
        else:
            o = torch.empty_like(q)

        if mode == "decode":
            # 必须把 num_kv_splits_local 传给 decode kernel
            self.decode_attention_fwd(
                q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
                key_pool,
                value_pool,
                o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
                kv_indptr,
                kv_indices,
                self.forward_metadata.attn_logits,
                self.forward_metadata.attn_lse,
                num_kv_splits_local,  # <<< 这里用 local
                self.max_kv_splits,
                layer.scaling,
                logit_cap=logits_soft_cap,
                sinks=sinks,
                xai_temperature_len=layer.xai_temperature_len,
            )
        else:
            # extend 路径不需要 num_kv_splits
            causal = layer.attn_type != AttentionType.ENCODER_ONLY
            self.extend_attention_fwd(
                q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
                k.contiguous(),
                v.contiguous(),
                o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
                key_pool,
                value_pool,
                self.forward_metadata.qo_indptr,
                kv_indptr,
                kv_indices,
                self.forward_metadata.custom_mask,
                causal,
                self.forward_metadata.mask_indptr,
                self.forward_metadata.max_extend_len,
                layer.scaling,
                logit_cap=logits_soft_cap,
                sliding_window_size=-1,
                sinks=sinks,
                window_kv_offsets=None,
                xai_temperature_len=layer.xai_temperature_len,
            )
        return o


class TritonMultiStepDraftBackend:
    """
    Wrap multiple triton attention backends as one for multiple consecutive
    draft decoding steps.
    """

    def __init__(
        self,
        model_runner: ModelRunner,
        topk: int,
        speculative_num_steps: int,
    ):
        from sglang.srt.speculative.eagle_utils import generate_draft_decode_kv_indices

        self.topk = topk
        self.speculative_num_steps = speculative_num_steps
        self.generate_draft_decode_kv_indices = generate_draft_decode_kv_indices
        max_bs = model_runner.req_to_token_pool.size * self.topk
        self.kv_indptr = torch.zeros(
            (
                self.speculative_num_steps,
                max_bs + 1,
            ),
            dtype=torch.int32,
            device=model_runner.device,
        )
        self.attn_backends = []
        for i in range(self.speculative_num_steps):
            self.attn_backends.append(
                TritonAttnBackend(
                    model_runner,
                    skip_prefill=True,
                    kv_indptr_buf=self.kv_indptr[i],
                )
            )
        self.max_context_len = self.attn_backends[0].max_context_len
        self.num_head = (
            model_runner.model_config.num_attention_heads // get_attention_tp_size()
        )
        self.device = model_runner.device
        # Cached variables for generate_draft_decode_kv_indices
        self.pool_len = model_runner.req_to_token_pool.req_to_token.shape[1]
        self.page_size = model_runner.server_args.page_size

    def common_template(
        self, forward_batch: ForwardBatch, kv_indices_buffer: torch.Tensor, call_fn: int
    ):
        num_seqs = forward_batch.batch_size
        bs = self.topk * num_seqs
        seq_lens_sum = forward_batch.seq_lens_sum

        self.generate_draft_decode_kv_indices[
            (self.speculative_num_steps, num_seqs, self.topk)
        ](
            forward_batch.req_pool_indices,
            forward_batch.req_to_token_pool.req_to_token,
            forward_batch.seq_lens,
            kv_indices_buffer,
            self.kv_indptr,
            forward_batch.positions,
            self.pool_len,
            kv_indices_buffer.shape[1],
            self.kv_indptr.shape[1],
            next_power_of_2(num_seqs),
            next_power_of_2(self.speculative_num_steps),
            next_power_of_2(bs),
            self.page_size,
        )

        for i in range(self.speculative_num_steps):
            forward_batch.spec_info.kv_indptr = self.kv_indptr[i, : bs + 1]
            forward_batch.spec_info.kv_indices = kv_indices_buffer[i][
                : seq_lens_sum * self.topk + bs * (i + 1)
            ]
            call_fn(i, forward_batch)

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        kv_indices = torch.empty(
            (
                self.speculative_num_steps,
                forward_batch.batch_size * self.topk * self.max_context_len,
            ),
            dtype=torch.int64,
            device=self.device,
        )

        def call_fn(i, forward_batch):
            forward_batch.spec_info.kv_indptr = (
                forward_batch.spec_info.kv_indptr.clone()
            )
            forward_batch.spec_info.kv_indices = (
                forward_batch.spec_info.kv_indices.clone()
            )
            self.attn_backends[i].init_forward_metadata(forward_batch)

        self.common_template(forward_batch, kv_indices, call_fn)

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        self.cuda_graph_kv_indices = torch.zeros(
            (self.speculative_num_steps, max_num_tokens * self.max_context_len),
            dtype=torch.int64,
            device=self.device,
        )
        for i in range(self.speculative_num_steps):
            self.attn_backends[i].init_cuda_graph_state(
                max_bs, max_num_tokens, kv_indices_buf=self.cuda_graph_kv_indices[i]
            )

    def init_forward_metadata_capture_cuda_graph(self, forward_batch: ForwardBatch):
        def call_fn(i, forward_batch):
            self.attn_backends[i].init_forward_metadata_capture_cuda_graph(
                forward_batch.batch_size,
                forward_batch.batch_size * self.topk,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                encoder_lens=None,
                forward_mode=ForwardMode.DECODE,
                spec_info=forward_batch.spec_info,
            )

        self.common_template(forward_batch, self.cuda_graph_kv_indices, call_fn)

    def init_forward_metadata_replay_cuda_graph(
        self, forward_batch: ForwardBatch, bs: int
    ):
        def call_fn(i, forward_batch):
            self.attn_backends[i].init_forward_metadata_replay_cuda_graph(
                bs,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                seq_lens_sum=-1,
                encoder_lens=None,
                forward_mode=ForwardMode.DECODE,
                spec_info=forward_batch.spec_info,
                seq_lens_cpu=None,
            )

        self.common_template(forward_batch, self.cuda_graph_kv_indices, call_fn)


@triton.jit
def get_num_kv_splits_triton(
    num_kv_splits_ptr,
    seq_lens_ptr,
    num_seq,
    num_group,
    num_head,
    num_kv_head,
    max_kv_splits,
    device_core_count,
    MAX_NUM_SEQ: tl.constexpr,
):
    # TODO: this method is tunable, we need more online serving data to tune it
    offs_seq = tl.arange(0, MAX_NUM_SEQ)
    mask_seq = offs_seq < num_seq

    seq_lens = tl.load(seq_lens_ptr + offs_seq, mask=mask_seq, other=0)
    max_seq_len = tl.max(seq_lens)
    seq_lens = tl.load(seq_lens_ptr + offs_seq, mask=mask_seq, other=max_seq_len)
    min_seq_len = tl.min(seq_lens)
    if max_seq_len * 8 < min_seq_len * 10:
        min_seq_len = max_seq_len
    max_kv_splits_1 = tl.minimum(tl.cdiv(max_seq_len, min_seq_len), max_kv_splits)
    kv_chunk_size_1 = tl.cdiv(max_seq_len, max_kv_splits_1)

    # NOTE: this is a hack to let num_kv_split grows up with seqlen gradually
    ext_seq_len = tl.cast(max_seq_len, tl.float32) / 64.0
    ext_device_core_count = tl.cast(
        device_core_count * tl.maximum(tl.log2(ext_seq_len), 1.0), tl.int32
    )
    block_h, num_kv_group = 16, num_head // num_kv_head
    if num_kv_group == 1:
        token_grid = num_seq * num_group * num_head
    else:
        # from triton_ops/decode_attention.py:_decode_grouped_att_m_fwd
        block_h = tl.minimum(block_h, num_kv_group)
        token_grid = num_seq * num_group * tl.cdiv(num_head, block_h)
    max_kv_splits_2 = tl.minimum(
        tl.cdiv(ext_device_core_count, token_grid), max_kv_splits
    )
    kv_chunk_size_2 = tl.cdiv(max_seq_len, max_kv_splits_2)

    num_kv_splits = tl.maximum(
        tl.cdiv(seq_lens, kv_chunk_size_1), tl.cdiv(seq_lens, kv_chunk_size_2)
    )

    offs_token = offs_seq * num_group
    mask_token = offs_token < num_seq * num_group
    for i in range(0, num_group):
        tl.store(num_kv_splits_ptr + i + offs_token, num_kv_splits, mask=mask_token)


def update_sliding_window_buffer(
    window_kv_indptr,
    req_to_token,
    sliding_window_size,
    seq_lens,
    req_pool_indices,
    bs,
    device,
    token_to_kv_pool_allocator=None,
):
    window_kv_lens = torch.minimum(
        seq_lens,
        torch.tensor(sliding_window_size),
    )
    window_kv_indptr[1 : bs + 1] = torch.cumsum(window_kv_lens, dim=0)
    window_kv_indptr = window_kv_indptr[: bs + 1]
    window_kv_indices = torch.empty(
        window_kv_indptr[-1], dtype=torch.int64, device=device
    )
    window_kv_start_idx = seq_lens - window_kv_lens
    create_flashinfer_kv_indices_triton[(bs,)](
        req_to_token,
        req_pool_indices,
        window_kv_lens,
        window_kv_indptr,
        window_kv_start_idx,
        window_kv_indices,
        req_to_token.stride(0),
    )
    # full to swa index mapping
    if hasattr(token_to_kv_pool_allocator, "translate_loc_from_full_to_swa"):
        kv_last_index = window_kv_indptr[-1]
        window_kv_indices[:kv_last_index] = (
            token_to_kv_pool_allocator.translate_loc_from_full_to_swa(
                window_kv_indices[:kv_last_index]
            )
        )
    return window_kv_indptr, window_kv_indices, window_kv_lens, window_kv_start_idx


def update_sliding_window_buffer_cuda_graph(
    window_kv_indptr,
    window_kv_indices,
    req_to_token,
    sliding_window_size,
    seq_lens,
    req_pool_indices,
    bs,
    token_to_kv_pool_allocator=None,
):
    window_kv_lens = torch.minimum(
        seq_lens,
        torch.tensor(sliding_window_size),
    )
    window_kv_indptr[1 : bs + 1] = torch.cumsum(window_kv_lens, dim=0)
    window_kv_indptr = window_kv_indptr[: bs + 1]
    window_kv_start_idx = seq_lens - window_kv_lens
    create_flashinfer_kv_indices_triton[(bs,)](
        req_to_token,
        req_pool_indices,
        window_kv_lens,
        window_kv_indptr,
        window_kv_start_idx,
        window_kv_indices,
        req_to_token.stride(0),
    )
    # full to swa index mapping
    if hasattr(token_to_kv_pool_allocator, "translate_loc_from_full_to_swa"):
        kv_last_index = window_kv_indptr[-1]
        window_kv_indices[:kv_last_index] = (
            token_to_kv_pool_allocator.translate_loc_from_full_to_swa(
                window_kv_indices[:kv_last_index]
            )
        )
    return window_kv_indptr, window_kv_indices, window_kv_lens, window_kv_start_idx
