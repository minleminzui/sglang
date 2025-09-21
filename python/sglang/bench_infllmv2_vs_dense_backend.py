# python/sglang/bench_infllmv2_vs_dense_backend.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import json
import math
from types import SimpleNamespace
from typing import Any, Dict, Tuple
from urllib.parse import urlparse

from sglang.bench_serving import run_benchmark as _run_one


def _mk_args(**kw) -> SimpleNamespace:
    # run_benchmark 里会读 args.port / args.host；即使我们用 base_url，也要占个位
    kw.setdefault("port", None)
    kw.setdefault("host", None)
    kw.setdefault("base_url", None)
    return SimpleNamespace(**kw)


def _host_port_from_url(url: str):
    u = urlparse(url)
    host = u.hostname or "127.0.0.1"
    port = u.port or (443 if (u.scheme or "http") == "https" else 80)
    return host, port


def _fmt(x, digits=2):
    if x is None:
        return "-"
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return "-"
    return f"{x:.{digits}f}" if isinstance(x, (float, int)) else str(x)


def _speedup(new, base):
    if new is None or base is None or base == 0:
        return "-"
    return f"{(new/base):.2f}×"


def _print(title, rows: Tuple[Tuple[str, Any, Any]]):
    w = 78
    print("\n" + "=" * w)
    print(title.center(w))
    print("=" * w)
    print(f"{'Metric':<35} {'Dense':>12} {'Sparse':>12} {'Speedup':>12}")
    print("-" * w)
    for n, dv, sv in rows:
        print(f"{n:<35} {_fmt(dv):>12} {_fmt(sv):>12} {_speedup(sv,dv):>12}")
    print("=" * w)


def main():
    ap = argparse.ArgumentParser("Dense vs InfLLM-v2 (backend-agnostic)")
    ap.add_argument("--dense-base-url", required=True)
    ap.add_argument("--sparse-base-url", required=True)
    ap.add_argument("--backend", default="fa3", choices=["fa3", "triton"])
    ap.add_argument("--enable-infllmv2", action="store_true")
    ap.add_argument("--model", default=None)
    ap.add_argument(
        "--dataset-name",
        default="random",
        choices=[
            "sharegpt",
            "random",
            "random-ids",
            "generated-shared-prefix",
            "mmmu",
            "random-image",
        ],
    )
    ap.add_argument("--dataset-path", default="")
    ap.add_argument("--num-prompts", type=int, default=1000)
    ap.add_argument("--sharegpt-output-len", type=int, default=None)
    ap.add_argument("--sharegpt-context-len", type=int, default=None)
    ap.add_argument("--random-input-len", type=int, default=1024)
    ap.add_argument("--random-output-len", type=int, default=1024)
    ap.add_argument("--random-range-ratio", type=float, default=0.5)
    ap.add_argument("--request-rate", type=float, default=float("inf"))
    ap.add_argument("--max-concurrency", type=int, default=64)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--save-json", type=str, default=None)
    args = ap.parse_args()

    common = dict(
        backend=args.backend,
        model=args.model,
        tokenizer=None,
        dataset_name=args.dataset_name,
        dataset_path=args.dataset_path,
        num_prompts=args.num_prompts,
        sharegpt_output_len=args.sharegpt_output_len,
        sharegpt_context_len=args.sharegpt_context_len,
        random_input_len=args.random_input_len,
        random_output_len=args.random_output_len,
        random_range_ratio=args.random_range_ratio,
        request_rate=args.request_rate,
        max_concurrency=args.max_concurrency,
        output_file=None,
        output_details=False,
        disable_tqdm=True,
        disable_stream=False,
        return_logprob=False,
        seed=args.seed,
        disable_ignore_eos=False,
        extra_request_body=None,
        apply_chat_template=False,
        profile=False,
        lora_name=None,
        prompt_suffix="",
        pd_separated=False,
        flush_cache=False,
        warmup_requests=1,
        random_image_num_images=1,
        random_image_resolution="1080p",
        tokenize_prompt=False,
        gsp_num_groups=64,
        gsp_prompts_per_group=16,
        gsp_system_prompt_len=2048,
        gsp_question_len=128,
        gsp_output_len=256,
    )

    d_host, d_port = _host_port_from_url(args.dense_base_url)
    s_host, s_port = _host_port_from_url(args.sparse_base_url)

    dense_ns = _mk_args(
        **common, base_url=args.dense_base_url, host=d_host, port=d_port
    )
    print("\n====== DENSE run ======")
    dense_res: Dict[str, Any] = _run_one(dense_ns)

    # 稀疏端：通过环境开关或服务端参数打开 InfLLM-v2
    sparse_ns = _mk_args(
        **common, base_url=args.sparse_base_url, host=s_host, port=s_port
    )
    print("\n====== SPARSE run ======")
    sparse_res: Dict[str, Any] = _run_one(sparse_ns)

    rows = (
        (
            "Request throughput (req/s)",
            dense_res.get("request_throughput"),
            sparse_res.get("request_throughput"),
        ),
        (
            "Input tok/s",
            dense_res.get("input_throughput"),
            sparse_res.get("input_throughput"),
        ),
        (
            "Output tok/s",
            dense_res.get("output_throughput"),
            sparse_res.get("output_throughput"),
        ),
        (
            "Total tok/s",
            dense_res.get("total_throughput"),
            sparse_res.get("total_throughput"),
        ),
        (
            "Mean TTFT (ms)",
            dense_res.get("mean_ttft_ms"),
            sparse_res.get("mean_ttft_ms"),
        ),
        ("P99 TTFT (ms)", dense_res.get("p99_ttft_ms"), sparse_res.get("p99_ttft_ms")),
        ("Mean ITL (ms)", dense_res.get("mean_itl_ms"), sparse_res.get("mean_itl_ms")),
        ("P95 ITL (ms)", dense_res.get("p95_itl_ms"), sparse_res.get("p95_itl_ms")),
        (
            "Mean E2E Latency (ms)",
            dense_res.get("mean_e2e_latency_ms"),
            sparse_res.get("mean_e2e_latency_ms"),
        ),
        ("Concurrency", dense_res.get("concurrency"), sparse_res.get("concurrency")),
        ("Completed requests", dense_res.get("completed"), sparse_res.get("completed")),
        (
            "Total output tokens",
            dense_res.get("total_output_tokens"),
            sparse_res.get("total_output_tokens"),
        ),
    )
    _print(f"Dense vs InfLLM-v2 — {args.backend}", rows)

    if args.save_json:
        payload = {
            "dense": dense_res,
            "sparse": sparse_res,
            "compare": {n: {"dense": dv, "sparse": sv} for (n, dv, sv) in rows},
        }
        with open(args.save_json, "w") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\nSaved to: {args.save_json}")


if __name__ == "__main__":
    main()
