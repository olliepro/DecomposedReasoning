"""One-GPU benchmark job entrypoint."""

from __future__ import annotations

import argparse
import csv
import importlib
import importlib.metadata
import importlib.util
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import median, quantiles
from typing import Any, cast

from vllm_experimental.benchmark_config import build_config, write_rows
from vllm_experimental.hardware import write_hardware_manifest
from vllm_experimental.native_event_metrics import read_native_event_metrics
from vllm_experimental.native_frontier import (
    NativeLeafOutput,
    run_native_frontier_chunk,
)
from vllm_experimental.progress import ProgressContext, ProgressLogger
from vllm_experimental.prompting import (
    ASSISTANT_PREFILL,
    DEFAULT_MATH_SYSTEM_PROMPT,
    attach_prompt_prefill,
    attach_token_payloads,
    chat_prompts,
    ensure_assistant_prefill,
    maybe_pad_raw_prompts,
    verbalized_token_scripts,
)
from vllm_experimental.types import BenchmarkMode, DEFAULT_SCRATCH_ROOT


@dataclass(frozen=True)
class RowRuntime:
    """Resolved runtime objects for one benchmark row."""

    vllm_mod: Any
    sampling_params: Any
    row_prompts: list[str]
    tree_search: dict[str, object]
    event_path: Path
    frontier_path: Path
    progress: ProgressLogger
    request_count: int
    batch_size: int
    use_native_frontier: bool


@dataclass(frozen=True)
class ChunkResult:
    """One completed row chunk result."""

    token_count: int
    latency_s: float
    memory_high_mib: int


def branch_mode_enabled(*, mode: BenchmarkMode) -> bool:
    """Return whether a mode must use native scheduler branching."""

    return mode in {"eps_on_policy_diverse", "eps_off_policy_verbalized"}


def vllm_version_payload() -> dict[str, str]:
    """Return vLLM version details from the active Python path."""

    spec = importlib.util.find_spec("vllm")
    return {
        "version": importlib.metadata.version("vllm"),
        "file": str(spec.origin if spec is not None else "unknown"),
    }


def cuda_version_payload() -> dict[str, str]:
    """Return CUDA and driver details for manifests."""

    code = (
        "import json; "
        "import torch; "
        "print(json.dumps({'torch_cuda': str(torch.version.cuda)}))"
    )
    payload = dict(
        json.loads(subprocess.check_output([sys.executable, "-c", code], text=True))
    )
    try:
        driver = (
            subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=driver_version",
                    "--format=csv,noheader",
                ],
                text=True,
            )
            .strip()
            .splitlines()[0]
        )
    except (IndexError, subprocess.CalledProcessError):
        driver = "unknown"
    payload["nvidia_driver"] = driver
    return payload


def write_manifest(
    *,
    output_dir: Path,
    run_name: str,
    mode: BenchmarkMode,
    rows: list[dict[str, object]],
) -> None:
    """Write benchmark manifest before any expensive GPU work."""

    hardware = write_hardware_manifest(path=output_dir / "hardware.json")
    tree_search = rows[0]["tree_search"]
    assert isinstance(tree_search, dict), "tree_search must be a dict"
    native_scheduler_kv_fork = bool(tree_search.get("native_scheduler_kv_fork", False))
    implementation_surface = (
        "native_vllm_scheduler_kv_fork"
        if branch_mode_enabled(mode=mode)
        else "vllm_logits_processor"
    )
    if branch_mode_enabled(mode=mode):
        assert native_scheduler_kv_fork, "branch modes require native scheduler KV fork"
    payload: dict[str, Any] = {
        "run_name": run_name,
        "mode": mode,
        "created_unix": time.time(),
        "hardware": {
            "node": hardware.node,
            "gpu_name": hardware.gpu_name,
            "gpu_memory_total_mib": hardware.gpu_memory_total_mib,
            "cuda_visible_devices": hardware.cuda_visible_devices,
        },
        "vllm": vllm_version_payload(),
        "cuda": cuda_version_payload(),
        "model_path": str(rows[0]["model_path"]),
        "max_model_len": rows[0]["max_model_len"],
        "max_num_batched_tokens": rows[0]["max_num_batched_tokens"],
        "max_num_seqs": rows[0]["max_num_seqs"],
        "effective_mode_config": tree_search,
        "implementation_surface": implementation_surface,
        "enable_prefix_caching": True,
        "native_scheduler_kv_fork": native_scheduler_kv_fork,
        "rows": rows,
        "env": {
            "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        },
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def gpu_memory_used_mib() -> int:
    """Return first visible GPU memory use in MiB."""

    query = [
        "nvidia-smi",
        "--query-gpu=memory.used",
        "--format=csv,noheader,nounits",
    ]
    raw = subprocess.check_output(query, text=True).strip().splitlines()
    assert raw, "nvidia-smi returned no memory rows"
    return int(raw[0].strip())


def load_prompts(*, task_name: str, doc_ids: list[int]) -> list[str]:
    """Load prompts through the existing branching_eval lm_eval adapter."""

    try:
        return load_prompts_in_process(task_name=task_name, doc_ids=doc_ids)
    except ModuleNotFoundError as exc:
        if exc.name != "lm_eval":
            raise
    analysis_python = Path(
        os.environ.get(
            "ANALYSIS_PYTHON",
            "/users/PAA0201/ollieproudman/work/DecomposedReasoning/"
            "Analysis/.venv/bin/python",
        )
    )
    code = (
        "import json; "
        "from vllm_experimental.run_benchmark_job import load_prompts_in_process; "
        "payload=json.loads(__import__('os').environ['PROMPT_PAYLOAD']); "
        "print(json.dumps(load_prompts_in_process("
        "task_name=payload['task_name'], doc_ids=payload['doc_ids'])))"
    )
    env = {
        **os.environ,
        "PROMPT_PAYLOAD": json.dumps({"task_name": task_name, "doc_ids": doc_ids}),
    }
    raw = subprocess.check_output(
        [str(analysis_python), "-c", code], env=env, text=True
    )
    return [str(prompt) for prompt in json.loads(raw)]


def load_prompts_in_process(*, task_name: str, doc_ids: list[int]) -> list[str]:
    """Load prompts through branching_eval in the current Python process."""

    adapter_mod = importlib.import_module("branching_eval.lm_eval_adapter")
    matrix_mod = importlib.import_module("branching_eval.run_matrix")
    adapter = adapter_mod.LmEvalAdapter(task_name=task_name)
    docs = adapter.docs(limit=None)
    filtered = matrix_mod.filter_docs_by_ids(docs=docs, doc_ids=tuple(doc_ids))
    return [str(doc.prompt_text) for doc in filtered]


def load_chat_prompts(*, row: dict[str, object], tokenizer: Any) -> list[str]:
    """Load task prompts, apply optional raw padding, then chat-template them."""

    raw_prompts = load_prompts(
        task_name=str(row["task_name"]),
        doc_ids=row_doc_ids(row=row),
    )
    return chat_prompts(
        prompts=maybe_pad_raw_prompts(raw_prompts=raw_prompts),
        tokenizer=tokenizer,
    )


def cycled(items: list[str], *, count: int) -> list[str]:
    """Return exactly count prompts by cycling over items."""

    assert items, "cannot cycle an empty prompt list"
    return [items[index % len(items)] for index in range(count)]


def p90(values: list[float]) -> float:
    """Return p90 for latency values."""

    if len(values) < 2:
        return values[0] if values else 0.0
    return quantiles(values, n=10, method="inclusive")[8]


def output_token_count(*, outputs: list[Any]) -> int:
    """Count generated tokens from vLLM RequestOutput objects."""

    total = 0
    for request_output in outputs:
        for candidate in request_output.outputs:
            total += len(candidate.token_ids)
    return total


def leaf_token_count(*, leaves: list[NativeLeafOutput]) -> int:
    """Count visible tokens in final native branch products."""

    return sum(len(leaf.token_ids) for leaf in leaves)


def sample_outputs_path(*, output_dir: Path) -> Path:
    """Return the generated trajectory sample path."""

    return output_dir / "sample_outputs.jsonl"


def job_progress_path(*, output_dir: Path) -> Path:
    """Return the job-stage progress path."""

    return output_dir / "job_progress.jsonl"


def write_job_progress(
    *, output_dir: Path, event: str, extra: dict[str, object] | None = None
) -> None:
    """Append one durable job-stage progress event and print it."""

    payload = {
        "event": event,
        "time_unix": time.time(),
        **(extra or {}),
    }
    path = job_progress_path(output_dir=output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
    print(json.dumps({"native_job_progress": payload}, sort_keys=True), flush=True)


def native_frontier_path(*, output_dir: Path, row: dict[str, object]) -> Path:
    """Return the native recursive branch frontier trace path."""

    return output_dir / (
        "native_frontier_"
        f"pc{row_int(row=row, key='prompt_concurrency')}_"
        f"bs{row_int(row=row, key='request_prompt_batch_size')}.jsonl"
    )


def native_frontier_batch_size(*, row: dict[str, object]) -> int:
    """Return the max recursive frontier batch size."""

    raw = os.environ.get("NATIVE_FRONTIER_BATCH_SIZE")
    if raw is not None and raw.strip():
        return int(raw)
    return row_int(row=row, key="max_num_seqs")


def tree_search_int(*, tree_search: dict[str, object], key: str, default: int) -> int:
    """Read an integer from a tree-search payload."""

    raw = tree_search.get(key, default)
    assert isinstance(raw, (int, str)), f"{key} must be an integer scalar"
    return int(raw)


def write_native_sample_outputs(
    *,
    path: Path,
    row: dict[str, object],
    prompts: list[str],
    outputs: list[Any],
    request_offset: int,
) -> None:
    """Append visible vLLM outputs for one native row chunk."""

    with path.open("a", encoding="utf-8") as handle:
        for local_index, request_output in enumerate(outputs):
            prompt = prompts[local_index]
            for choice_index, choice in enumerate(request_output.outputs):
                handle.write(
                    json.dumps(
                        {
                            "mode": row["mode"],
                            "prompt_concurrency": row["prompt_concurrency"],
                            "request_prompt_batch_size": row[
                                "request_prompt_batch_size"
                            ],
                            "request_index": request_offset + local_index,
                            "choice_index": choice_index,
                            "prompt": prompt,
                            "prompt_char_count": len(prompt),
                            "prompt_tail": prompt[-1200:],
                            "assistant_prefill": ASSISTANT_PREFILL,
                            "text": str(choice.text),
                            "logical_text": ASSISTANT_PREFILL + str(choice.text),
                            "token_count": len(choice.token_ids),
                            "token_ids": [
                                int(token_id) for token_id in choice.token_ids
                            ],
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )


def write_native_leaf_outputs(
    *,
    path: Path,
    row: dict[str, object],
    leaves: list[NativeLeafOutput],
    prompts_by_request_index: dict[int, str],
) -> None:
    """Append final recursive native branch products."""

    with path.open("a", encoding="utf-8") as handle:
        for leaf in leaves:
            payload = leaf.sample_payload(
                row=row,
                prompt=prompts_by_request_index[leaf.request_index],
            )
            payload["assistant_prefill"] = ASSISTANT_PREFILL
            payload["logical_text"] = ASSISTANT_PREFILL + leaf.text
            handle.write(json.dumps(payload, sort_keys=True) + "\n")


def native_event_path(*, output_dir: Path, row: dict[str, object]) -> Path:
    """Return the native scheduler event path for one load row."""

    return output_dir / (
        "native_events_"
        f"pc{row_int(row=row, key='prompt_concurrency')}_"
        f"bs{row_int(row=row, key='request_prompt_batch_size')}.jsonl"
    )


def progress_path(*, output_dir: Path, row: dict[str, object]) -> Path:
    """Return the durable benchmark progress path for one load row."""

    return output_dir / (
        "progress_"
        f"pc{row_int(row=row, key='prompt_concurrency')}_"
        f"bs{row_int(row=row, key='request_prompt_batch_size')}.jsonl"
    )


def write_metric_rows(*, rows: list[dict[str, object]], output_dir: Path) -> None:
    """Write benchmark metrics as JSONL and compact CSV."""

    jsonl_path = output_dir / "metrics.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    csv_path = output_dir / "metrics_summary.csv"
    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def row_int(*, row: dict[str, object], key: str) -> int:
    """Read an integer from a benchmark row."""

    value = row[key]
    assert isinstance(value, int), f"{key} must be int"
    return value


def row_doc_ids(*, row: dict[str, object]) -> list[int]:
    """Read document ids from a benchmark row."""

    value = row["doc_ids"]
    assert isinstance(value, list), "doc_ids must be a list"
    return [int(doc_id) for doc_id in value]


def row_tree_search(*, row: dict[str, object]) -> dict[str, object]:
    """Read tree-search payload from a benchmark row."""

    value = row["tree_search"]
    assert isinstance(value, dict), "tree_search must be a dict"
    return {str(key): value for key, value in value.items()}


def run_grammar_temp_rows(
    *,
    rows: list[dict[str, object]],
    output_dir: Path,
    max_tokens: int,
) -> list[dict[str, object]]:
    """Run native grammar-temp vLLM benchmark rows."""

    vllm_mod = importlib.import_module("vllm")
    plugin_mod = importlib.import_module("vllm_experimental.vllm_plugin")
    job_progress_path(output_dir=output_dir).unlink(missing_ok=True)
    write_job_progress(output_dir=output_dir, event="llm_init_start")
    llm = vllm_mod.LLM(
        model=str(rows[0]["model_path"]),
        max_model_len=row_int(row=rows[0], key="max_model_len"),
        max_num_batched_tokens=row_int(row=rows[0], key="max_num_batched_tokens"),
        max_num_seqs=row_int(row=rows[0], key="max_num_seqs"),
        enable_prefix_caching=True,
        language_model_only=True,
        skip_mm_profiling=True,
        logits_processors=[plugin_mod.ThoughtGrammarLogitsProcessor],
    )
    write_job_progress(output_dir=output_dir, event="llm_init_complete")
    tokenizer = llm.get_tokenizer()
    write_job_progress(output_dir=output_dir, event="prompt_load_start")
    prompts = load_chat_prompts(row=rows[0], tokenizer=tokenizer)
    write_job_progress(
        output_dir=output_dir,
        event="prompt_load_complete",
        extra={"prompt_count": len(prompts)},
    )
    sample_outputs_path(output_dir=output_dir).unlink(missing_ok=True)
    mode = cast(BenchmarkMode, rows[0]["mode"])
    if native_scheduler_enabled(row=rows[0], mode=mode):
        if native_warmup_enabled():
            write_job_progress(output_dir=output_dir, event="native_warmup_start")
            warm_up_native_branch(
                llm=llm,
                vllm_mod=vllm_mod,
                tokenizer=tokenizer,
                prompt=prompts[0],
                row=rows[0],
                max_tokens=max_tokens,
            )
            write_job_progress(output_dir=output_dir, event="native_warmup_complete")
        else:
            write_job_progress(output_dir=output_dir, event="native_warmup_skipped")
    else:
        write_job_progress(output_dir=output_dir, event="grammar_warmup_start")
        warm_up_grammar(
            llm=llm,
            tokenizer=tokenizer,
            prompt=prompts[0],
            row=rows[0],
            max_tokens=min(max_tokens, 16),
        )
        write_job_progress(output_dir=output_dir, event="grammar_warmup_complete")
    if batch_shape_warmup_enabled():
        warmup_prompts = batch_shape_warmup_prompts(prompts=prompts, row=rows[0])
        warmup_tokens = batch_shape_warmup_token_count(max_tokens=max_tokens)
        write_job_progress(
            output_dir=output_dir,
            event="batch_shape_warmup_start",
            extra={
                "prompt_count": len(warmup_prompts),
                "max_tokens": warmup_tokens,
            },
        )
        warm_up_batch_shape(
            llm=llm,
            vllm_mod=vllm_mod,
            tokenizer=tokenizer,
            prompts=warmup_prompts,
            row=rows[0],
            max_tokens=warmup_tokens,
        )
        write_job_progress(output_dir=output_dir, event="batch_shape_warmup_complete")
    else:
        write_job_progress(output_dir=output_dir, event="batch_shape_warmup_skipped")
    metrics: list[dict[str, object]] = []
    for row in rows:
        metrics.append(
            run_one_row(
                llm=llm,
                tokenizer=tokenizer,
                prompts=prompts,
                row=row,
                output_dir=output_dir,
                max_tokens=max_tokens,
            )
        )
        write_metric_rows(rows=metrics, output_dir=output_dir)
    return metrics


def native_scheduler_enabled(*, row: dict[str, object], mode: BenchmarkMode) -> bool:
    """Return whether this row should use the patched native scheduler path."""

    tree_search = row_tree_search(row=row)
    return bool(tree_search.get("native_scheduler_kv_fork")) and branch_mode_enabled(
        mode=mode
    )


def native_frontier_enabled(*, row: dict[str, object], mode: BenchmarkMode) -> bool:
    """Return whether recursive branch-product frontier launches are needed."""

    if not native_scheduler_enabled(row=row, mode=mode):
        return False
    tree_search = row_tree_search(row=row)
    return tree_search_int(tree_search=tree_search, key="branch_fanout", default=1) > 1


def native_warmup_enabled() -> bool:
    """Return whether to run the unmeasured native branch warmup."""

    raw_value = os.environ.get("NATIVE_WARMUP_ENABLED", "1").strip().lower()
    return raw_value not in {"0", "false", "no", "off"}


def batch_shape_warmup_enabled() -> bool:
    """Return whether to run the unmeasured full-batch warmup."""

    raw_value = os.environ.get("BATCH_SHAPE_WARMUP_ENABLED", "1").strip().lower()
    return raw_value not in {"0", "false", "no", "off"}


def batch_shape_warmup_token_count(*, max_tokens: int) -> int:
    """Return the token cap for full-batch shape warmup."""

    configured = int(os.environ.get("BATCH_SHAPE_WARMUP_MAX_TOKENS", "4"))
    assert configured >= 1, "BATCH_SHAPE_WARMUP_MAX_TOKENS must be positive"
    return max(1, min(max_tokens, configured))


def batch_shape_warmup_prompts(
    *,
    prompts: list[str],
    row: dict[str, object],
) -> list[str]:
    """Return the first measured chunk shape for warmup."""

    batch_size = row_int(row=row, key="request_prompt_batch_size")
    request_count = row_int(row=row, key="prompt_concurrency")
    return cycled(prompts, count=min(batch_size, request_count))


def warm_up_grammar(
    *,
    llm: Any,
    tokenizer: Any,
    prompt: str,
    row: dict[str, object],
    max_tokens: int,
) -> None:
    """Run one unmeasured grammar request to JIT request-shape kernels."""

    vllm_mod = importlib.import_module("vllm")
    tree_search = row_tree_search(row=row)
    attach_token_payloads(tree_search=tree_search, tokenizer=tokenizer)
    sampling_params = vllm_mod.SamplingParams(
        max_tokens=max_tokens,
        temperature=1.0,
        extra_args={"vllm_experimental": tree_search},
    )
    llm.generate(prompts=[prompt], sampling_params=sampling_params, use_tqdm=False)


def warm_up_native_branch(
    *,
    llm: Any,
    vllm_mod: Any,
    tokenizer: Any,
    prompt: str,
    row: dict[str, object],
    max_tokens: int,
) -> None:
    """Run one unmeasured native branch fire to JIT branch-shape kernels."""

    tree_search = row_tree_search(row=row)
    attach_token_payloads(tree_search=tree_search, tokenizer=tokenizer)
    warm_tree_search = dict(tree_search)
    warm_tree_search["fire_rate"] = 1.0
    warm_event_path = Path(os.environ.get("TMPDIR", "/tmp")) / (
        f"vllm_exp_native_warmup_{os.getpid()}.jsonl"
    )
    warm_event_path.unlink(missing_ok=True)
    warm_tree_search["native_event_log_path"] = str(warm_event_path)
    sampling_params = vllm_mod.SamplingParams(
        max_tokens=min(max_tokens, 256),
        temperature=1.0,
        extra_args={"vllm_experimental": warm_tree_search},
    )
    llm.generate(prompts=[prompt], sampling_params=sampling_params, use_tqdm=False)
    warm_event_path.unlink(missing_ok=True)


def warm_up_batch_shape(
    *,
    llm: Any,
    vllm_mod: Any,
    tokenizer: Any,
    prompts: list[str],
    row: dict[str, object],
    max_tokens: int,
) -> None:
    """Run one unmeasured full-batch decode without branch fires."""

    tree_search = row_tree_search(row=row)
    attach_token_payloads(tree_search=tree_search, tokenizer=tokenizer)
    warm_tree_search = dict(tree_search)
    warm_tree_search["fire_rate"] = 0.0
    sampling_params = vllm_mod.SamplingParams(
        max_tokens=max_tokens,
        temperature=1.0,
        extra_args={"vllm_experimental": warm_tree_search},
    )
    llm.generate(prompts=prompts, sampling_params=sampling_params, use_tqdm=False)


def run_native_output_chunk(
    *,
    llm: Any,
    vllm_mod: Any,
    tokenizer: Any,
    row: dict[str, object],
    chunk: list[str],
    tree_search: dict[str, object],
    output_dir: Path,
    max_tokens: int,
    request_offset: int,
    frontier_path: Path,
) -> int:
    """Run and record one recursive native branch chunk."""

    leaves = run_native_frontier_chunk(
        llm=llm,
        vllm_mod=vllm_mod,
        tokenizer=tokenizer,
        prompts=chunk,
        tree_search=tree_search,
        max_tokens=max_tokens,
        request_offset=request_offset,
        trace_path=frontier_path,
        frontier_batch_size=native_frontier_batch_size(row=row),
        assistant_prefill=ASSISTANT_PREFILL,
    )
    prompts_by_index = {
        request_offset + local_index: prompt for local_index, prompt in enumerate(chunk)
    }
    write_native_leaf_outputs(
        path=sample_outputs_path(output_dir=output_dir),
        row=row,
        leaves=leaves,
        prompts_by_request_index=prompts_by_index,
    )
    return leaf_token_count(leaves=leaves)


def run_regular_output_chunk(
    *,
    llm: Any,
    sampling_params: Any,
    row: dict[str, object],
    chunk: list[str],
    output_dir: Path,
    request_offset: int,
) -> int:
    """Run and record one ordinary vLLM chunk."""

    outputs = llm.generate(
        prompts=chunk,
        sampling_params=sampling_params,
        use_tqdm=False,
    )
    write_native_sample_outputs(
        path=sample_outputs_path(output_dir=output_dir),
        row=row,
        prompts=chunk,
        outputs=outputs,
        request_offset=request_offset,
    )
    return output_token_count(outputs=outputs)


def row_result_payload(
    *,
    row: dict[str, object],
    tree_search: dict[str, object],
    wall: float,
    request_count: int,
    generated_tokens: int,
    latencies: list[float],
    native_metrics: dict[str, object],
    memory_high: int,
    use_native_frontier: bool,
) -> dict[str, object]:
    """Return one benchmark metrics row."""

    return {
        **{key: value for key, value in row.items() if key != "tree_search"},
        "wall_time_s": wall,
        "request_throughput_rps": request_count / wall,
        "generated_tokens": generated_tokens,
        "generated_tokens_s": generated_tokens / wall,
        "branch_tokens_s": 0.0,
        "prefill_tokens_avoided": native_metrics["prefill_tokens_avoided"],
        "p50_latency_s": median(latencies),
        "p90_latency_s": p90(latencies),
        "boundary_fire_count": native_metrics["boundary_fire_count"],
        "branch_count": native_metrics["branch_count"],
        "selected_candidate_ids": native_metrics["selected_candidate_ids"],
        "returned_candidate_ids": native_metrics["returned_candidate_ids"],
        "returned_branch_counts": native_metrics["returned_branch_counts"],
        "branch_depth_used_events": native_metrics["branch_depth_used"],
        "branch_depth_limit_events": native_metrics["branch_depth_limits"],
        "top_k_candidate_ids": native_metrics["top_k_candidate_ids"],
        "unique_candidate_counts": native_metrics["unique_candidate_counts"],
        "candidate_bounds": native_metrics["candidate_bounds"],
        "kv_blocks_allocated": native_metrics["kv_blocks_allocated"],
        "kv_blocks_copied": native_metrics["kv_blocks_copied"],
        "kv_blocks_freed": native_metrics["kv_blocks_freed"],
        "async_tokens_discarded": native_metrics["async_tokens_discarded"],
        "branch_pool_queued_count": native_metrics["branch_pool_queued_count"],
        "branch_pool_admitted_count": native_metrics["branch_pool_admitted_count"],
        "branch_pool_blocked_count": native_metrics["branch_pool_blocked_count"],
        "max_live_branch_pools": native_metrics["max_live_branch_pools"],
        "max_queued_branch_pools": native_metrics["max_queued_branch_pools"],
        "min_branch_free_blocks": native_metrics["min_branch_free_blocks"],
        "min_branch_seq_slots": native_metrics["min_branch_seq_slots"],
        "diversity_vector_source": native_metrics["diversity_vector_source"],
        "hidden_vector_child_count": native_metrics["hidden_vector_child_count"],
        "pool_hidden_pairwise_diversity": native_metrics[
            "pool_hidden_pairwise_diversity"
        ],
        "selected_hidden_diversity": native_metrics["selected_hidden_diversity"],
        "gpu_memory_high_watermark_mib": memory_high,
        "failure_reason": "",
        "branch_depth": tree_search_int(
            tree_search=tree_search,
            key="branch_depth",
            default=4,
        ),
        "native_frontier_batch_size": (
            native_frontier_batch_size(row=row) if use_native_frontier else 0
        ),
    }


def build_row_runtime(
    *,
    tokenizer: Any,
    prompts: list[str],
    row: dict[str, object],
    output_dir: Path,
    max_tokens: int,
) -> RowRuntime:
    """Resolve vLLM params, files, and progress logger for one row."""

    vllm_mod = importlib.import_module("vllm")
    request_count = row_int(row=row, key="prompt_concurrency")
    batch_size = row_int(row=row, key="request_prompt_batch_size")
    row_prompts = cycled(prompts, count=request_count)
    tree_search = row_tree_search(row=row)
    attach_token_payloads(tree_search=tree_search, tokenizer=tokenizer)
    event_path = native_event_path(output_dir=output_dir, row=row)
    frontier_path = native_frontier_path(output_dir=output_dir, row=row)
    mode = cast(BenchmarkMode, row["mode"])
    use_native_frontier = native_frontier_enabled(row=row, mode=mode)
    if bool(tree_search.get("native_scheduler_kv_fork", False)):
        event_path.unlink(missing_ok=True)
        frontier_path.unlink(missing_ok=True)
        tree_search["native_event_log_path"] = str(event_path)
    sampling_params = vllm_mod.SamplingParams(
        max_tokens=max_tokens,
        temperature=1.0,
        extra_args={"vllm_experimental": tree_search},
    )
    return RowRuntime(
        vllm_mod=vllm_mod,
        sampling_params=sampling_params,
        row_prompts=row_prompts,
        tree_search=tree_search,
        event_path=event_path,
        frontier_path=frontier_path,
        progress=build_progress_logger(
            row=row,
            output_dir=output_dir,
            event_path=event_path,
            frontier_path=frontier_path,
            request_count=request_count,
            batch_size=batch_size,
        ),
        request_count=request_count,
        batch_size=batch_size,
        use_native_frontier=use_native_frontier,
    )


def build_progress_logger(
    *,
    row: dict[str, object],
    output_dir: Path,
    event_path: Path,
    frontier_path: Path,
    request_count: int,
    batch_size: int,
) -> ProgressLogger:
    """Create the progress logger for one benchmark row."""

    return ProgressLogger(
        context=ProgressContext(
            mode=str(row["mode"]),
            prompt_concurrency=request_count,
            request_prompt_batch_size=batch_size,
            request_count=request_count,
            progress_path=progress_path(output_dir=output_dir, row=row),
            sample_path=sample_outputs_path(output_dir=output_dir),
            native_event_path=event_path,
            frontier_trace_path=frontier_path,
        ),
        memory_sampler=gpu_memory_used_mib,
        interval_s=float(os.environ.get("PROGRESS_INTERVAL_S", "30")),
    )


def run_row_chunk(
    *,
    llm: Any,
    tokenizer: Any,
    row: dict[str, object],
    output_dir: Path,
    max_tokens: int,
    runtime: RowRuntime,
    request_offset: int,
    memory_high: int,
) -> ChunkResult:
    """Run one prompt chunk and write its progress checkpoint."""

    chunk = runtime.row_prompts[request_offset : request_offset + runtime.batch_size]
    chunk_index = request_offset // runtime.batch_size
    chunk_start = time.perf_counter()
    runtime.progress.chunk_started(
        chunk_index=chunk_index,
        request_offset=request_offset,
        request_count=len(chunk),
    )
    if runtime.use_native_frontier:
        token_count = run_native_output_chunk(
            llm=llm,
            vllm_mod=runtime.vllm_mod,
            tokenizer=tokenizer,
            row=row,
            chunk=chunk,
            tree_search=runtime.tree_search,
            output_dir=output_dir,
            max_tokens=max_tokens,
            request_offset=request_offset,
            frontier_path=runtime.frontier_path,
        )
    else:
        token_count = run_regular_output_chunk(
            llm=llm,
            sampling_params=runtime.sampling_params,
            row=row,
            chunk=chunk,
            output_dir=output_dir,
            request_offset=request_offset,
        )
    latency_s = time.perf_counter() - chunk_start
    memory_high = max(memory_high, gpu_memory_used_mib())
    runtime.progress.chunk_completed(
        request_count=len(chunk),
        chunk_tokens=token_count,
        chunk_latency_s=latency_s,
        memory_high_mib=memory_high,
    )
    return ChunkResult(
        token_count=token_count,
        latency_s=latency_s,
        memory_high_mib=memory_high,
    )


def run_one_row(
    *,
    llm: Any,
    tokenizer: Any,
    prompts: list[str],
    row: dict[str, object],
    output_dir: Path,
    max_tokens: int,
) -> dict[str, object]:
    """Run one load row and return timing metrics."""

    runtime = build_row_runtime(
        tokenizer=tokenizer,
        prompts=prompts,
        row=row,
        output_dir=output_dir,
        max_tokens=max_tokens,
    )
    row_start = time.perf_counter()
    latencies: list[float] = []
    generated_tokens = 0
    memory_high = gpu_memory_used_mib()
    runtime.progress.start()
    try:
        for index in range(0, len(runtime.row_prompts), runtime.batch_size):
            chunk_result = run_row_chunk(
                llm=llm,
                tokenizer=tokenizer,
                row=row,
                output_dir=output_dir,
                max_tokens=max_tokens,
                runtime=runtime,
                request_offset=index,
                memory_high=memory_high,
            )
            generated_tokens += chunk_result.token_count
            latencies.append(chunk_result.latency_s)
            memory_high = chunk_result.memory_high_mib
    except BaseException as exc:
        runtime.progress.finish(
            event="row_failed",
            reason=f"{type(exc).__name__}: {exc}",
        )
        raise
    wall = time.perf_counter() - row_start
    native_metrics = read_native_event_metrics(path=runtime.event_path)
    result = row_result_payload(
        row=row,
        tree_search=runtime.tree_search,
        wall=wall,
        request_count=runtime.request_count,
        generated_tokens=generated_tokens,
        latencies=latencies,
        native_metrics=native_metrics,
        memory_high=memory_high,
        use_native_frontier=runtime.use_native_frontier,
    )
    runtime.progress.finish(event="row_complete", reason="")
    return result


def main(argv: list[str] | None = None) -> int:
    """Generate manifests and fail closed for unimplemented native branch modes."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument(
        "--mode",
        choices=[
            "grammar_temp",
            "eps_on_policy_diverse",
            "eps_off_policy_verbalized",
        ],
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_SCRATCH_ROOT)
    parser.add_argument("--max-tokens", type=int, default=2048)
    args = parser.parse_args(argv)
    mode: BenchmarkMode = args.mode
    output_dir = args.output_dir / args.run_name / mode
    config = build_config(run_name=args.run_name, mode=mode)
    rows = config.sweep_rows()
    write_rows(rows=rows, output_dir=output_dir)
    write_manifest(
        output_dir=output_dir,
        run_name=args.run_name,
        mode=mode,
        rows=rows,
    )
    metrics = run_grammar_temp_rows(
        rows=rows,
        output_dir=output_dir,
        max_tokens=args.max_tokens,
    )
    print(
        json.dumps(
            {"status": "complete", "rows": len(rows), "metrics": len(metrics)},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
