#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    num_prompts: int
    prompt_tokens: int
    max_tokens: int
    n: int


@dataclass(frozen=True)
class BenchmarkResult:
    name: str
    label: str
    tensor_parallel_size: int
    decode_context_parallel_size: int
    enable_dbo: bool
    dbo_decode_token_threshold: int | None
    dbo_prefill_token_threshold: int | None
    disable_hybrid_kv_cache_manager: bool
    load_format: str
    language_model_only: bool
    reasoning_parser: str | None
    gdn_prefill_backend: str | None
    speculative_config: str | None
    num_prompts: int
    prompt_tokens: int
    max_tokens: int
    n: int
    elapsed_seconds: float
    output_tokens: int
    output_tokens_per_second: float
    choices: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone vLLM decode throughput benchmark.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--tensor-parallel-size", type=int, required=True)
    parser.add_argument("--decode-context-parallel-size", type=int, default=1)
    parser.add_argument("--enable-dbo", action="store_true")
    parser.add_argument("--dbo-decode-token-threshold", type=int)
    parser.add_argument("--dbo-prefill-token-threshold", type=int)
    parser.add_argument("--disable-hybrid-kv-cache-manager", action="store_true")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.82)
    parser.add_argument("--max-model-len", type=int, default=34816)
    parser.add_argument("--max-num-batched-tokens", type=int, default=524288)
    parser.add_argument("--max-num-seqs", type=int, default=4096)
    parser.add_argument("--load-format", default="auto")
    parser.add_argument("--language-model-only", action="store_true")
    parser.add_argument("--reasoning-parser")
    parser.add_argument("--gdn-prefill-backend", choices=("flashinfer", "triton", "cutedsl"))
    parser.add_argument("--speculative-config-json")
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--case-set", choices=("quick", "full"), default="quick")
    return parser.parse_args()


def prompt_with_token_count(tokenizer: object, token_count: int) -> str:
    text = "Solve this math problem step by step. " + ("x + y = 3. " * token_count)
    encode = getattr(tokenizer, "encode")
    decode = getattr(tokenizer, "decode")
    token_ids = encode(text, add_special_tokens=False)
    while len(token_ids) < token_count:
        token_ids.extend(token_ids)
    return decode(token_ids[:token_count])


def benchmark_cases(case_set: str) -> list[BenchmarkCase]:
    if case_set == "full":
        return [
            BenchmarkCase(name="long_decode", num_prompts=64, prompt_tokens=512, max_tokens=512, n=1),
            BenchmarkCase(name="rl_decode_chunk", num_prompts=128, prompt_tokens=4096, max_tokens=512, n=1),
            BenchmarkCase(name="candidate_pool_n50", num_prompts=16, prompt_tokens=4096, max_tokens=512, n=50),
        ]

    return [
        BenchmarkCase(name="decode_512p128g", num_prompts=64, prompt_tokens=512, max_tokens=128, n=1),
        BenchmarkCase(name="decode_4096p128g", num_prompts=64, prompt_tokens=4096, max_tokens=128, n=1),
        BenchmarkCase(name="candidate_pool_n50", num_prompts=8, prompt_tokens=4096, max_tokens=16, n=50),
    ]


def build_llm(args: argparse.Namespace) -> LLM:
    engine_kwargs: dict[str, Any] = {}
    if args.gdn_prefill_backend is not None:
        engine_kwargs["additional_config"] = {"gdn_prefill_backend": args.gdn_prefill_backend}
    if args.decode_context_parallel_size > 1:
        engine_kwargs["decode_context_parallel_size"] = args.decode_context_parallel_size
    if args.enable_dbo:
        engine_kwargs["enable_dbo"] = True
    if args.dbo_decode_token_threshold is not None:
        engine_kwargs["dbo_decode_token_threshold"] = args.dbo_decode_token_threshold
    if args.dbo_prefill_token_threshold is not None:
        engine_kwargs["dbo_prefill_token_threshold"] = args.dbo_prefill_token_threshold
    if args.disable_hybrid_kv_cache_manager:
        engine_kwargs["disable_hybrid_kv_cache_manager"] = True
    if args.reasoning_parser is not None:
        engine_kwargs["reasoning_parser"] = args.reasoning_parser
    if args.speculative_config_json is not None:
        engine_kwargs["speculative_config"] = json.loads(args.speculative_config_json)
    compilation_config = {"cudagraph_mode": "FULL_AND_PIECEWISE"}
    if args.decode_context_parallel_size > 1:
        compilation_config["cudagraph_mode"] = "PIECEWISE"
    return LLM(
        model=args.model,
        tokenizer=args.model,
        dtype="bfloat16",
        trust_remote_code=args.trust_remote_code,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        load_format=args.load_format,
        language_model_only=args.language_model_only,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_chunked_prefill=True,
        enable_prefix_caching=True,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        compilation_config=compilation_config,
        **engine_kwargs,
    )


def run_case(llm: LLM, prompts: list[str], case: BenchmarkCase, args: argparse.Namespace) -> BenchmarkResult:
    sampling_params = SamplingParams(
        max_tokens=case.max_tokens,
        n=case.n,
        temperature=0.7,
        top_p=0.95,
        top_k=-1,
        ignore_eos=True,
    )
    start = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params=sampling_params, use_tqdm=False)
    elapsed_seconds = time.perf_counter() - start
    output_tokens = sum(len(choice.token_ids) for output in outputs for choice in output.outputs)
    choices = sum(len(output.outputs) for output in outputs)
    return BenchmarkResult(
        name=case.name,
        label=args.label,
        tensor_parallel_size=args.tensor_parallel_size,
        decode_context_parallel_size=args.decode_context_parallel_size,
        enable_dbo=args.enable_dbo,
        dbo_decode_token_threshold=args.dbo_decode_token_threshold,
        dbo_prefill_token_threshold=args.dbo_prefill_token_threshold,
        disable_hybrid_kv_cache_manager=args.disable_hybrid_kv_cache_manager,
        load_format=args.load_format,
        language_model_only=args.language_model_only,
        reasoning_parser=args.reasoning_parser,
        gdn_prefill_backend=args.gdn_prefill_backend,
        speculative_config=args.speculative_config_json,
        num_prompts=case.num_prompts,
        prompt_tokens=case.prompt_tokens,
        max_tokens=case.max_tokens,
        n=case.n,
        elapsed_seconds=elapsed_seconds,
        output_tokens=output_tokens,
        output_tokens_per_second=output_tokens / elapsed_seconds,
        choices=choices,
    )


def run_warmup(llm: LLM, prompt: str) -> float:
    sampling_params = SamplingParams(max_tokens=16, n=1, temperature=0.7, top_p=0.95, top_k=-1, ignore_eos=True)
    start = time.perf_counter()
    llm.generate([prompt] * 8, sampling_params=sampling_params, use_tqdm=False)
    return time.perf_counter() - start


def main() -> None:
    args = parse_args()
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    cases = benchmark_cases(case_set=args.case_set)
    prompts_by_tokens = {
        case.prompt_tokens: prompt_with_token_count(tokenizer=tokenizer, token_count=case.prompt_tokens)
        for case in cases
    }
    llm = build_llm(args=args)
    warmup_seconds = run_warmup(llm=llm, prompt=next(iter(prompts_by_tokens.values())))
    with output_path.open("a", encoding="utf-8") as handle:
        for case in cases:
            prompts = [prompts_by_tokens[case.prompt_tokens]] * case.num_prompts
            result = run_case(llm=llm, prompts=prompts, case=case, args=args)
            payload = {
                "case_set": args.case_set,
                "warmup_seconds": warmup_seconds,
                **asdict(result),
            }
            print(json.dumps(payload, sort_keys=True), flush=True)
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
            handle.flush()
    del llm
    gc.collect()


if __name__ == "__main__":
    main()
