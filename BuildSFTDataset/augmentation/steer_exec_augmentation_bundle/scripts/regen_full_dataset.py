from __future__ import annotations

import argparse
import copy
import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import request as urllib_request
from urllib import error as urllib_error

import tiktoken
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class RegenTask:
    """One row that still needs continuation.

    Args:
        index: Zero-based row index inside `needs_regen`.
        row_id: Stable augmented row identifier.
        user_prompt: Original user turn.
        assistant_prefill: Saved assistant-side prefix for continuation.
        row: Full dataset row to update after generation.

    Example:
        >>> RegenTask(
        ...     index=0,
        ...     row_id="row-1",
        ...     user_prompt="Solve x+1=2",
        ...     assistant_prefill="<think>\\n<exec>We have",
        ...     row={"id": "row-1", "messages": []},
        ... ).index
        0
    """

    index: int
    row_id: str
    user_prompt: str
    assistant_prefill: str
    row: dict[str, Any]


@dataclass(frozen=True)
class RegenResult:
    """One regeneration outcome.

    Args:
        task: Input task metadata.
        finish_reason: vLLM finish reason or local error label.
        stop_reason: vLLM stop reason when available.
        assistant_content: Full regenerated assistant message.
        completed_row: Updated dataset row ready for merge.
        error: Optional error text.
        elapsed_seconds: End-to-end request latency.
    """

    task: RegenTask
    finish_reason: str
    stop_reason: int | str | None
    assistant_content: str | None
    completed_row: dict[str, Any] | None
    error: str | None
    elapsed_seconds: float
    round_count: int

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-friendly representation of the result."""
        payload = {
            "index": self.task.index,
            "row_id": self.task.row_id,
            "finish_reason": self.finish_reason,
            "stop_reason": self.stop_reason,
            "error": self.error,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "round_count": self.round_count,
        }
        if self.assistant_content is not None:
            payload["assistant_tail"] = self.assistant_content[-2000:]
        return payload


def normalize_vllm_base_url(*, base_url: str) -> str:
    """Normalize a raw server address into an absolute `/v1` base URL."""
    normalized_url = base_url.strip().rstrip("/")
    assert normalized_url, "base_url must not be empty"
    if "://" not in normalized_url:
        normalized_url = f"http://{normalized_url}"
    if normalized_url.endswith("/v1"):
        return normalized_url
    return f"{normalized_url}/v1"


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for full regen jobs."""
    parser = argparse.ArgumentParser(
        description="Regenerate all needs-regen rows with one-shot vLLM completions."
    )
    default_run_dir = ROOT / "out" / "run_20260422_400_c50_t256"
    parser.add_argument("--run-dir", default=str(default_run_dir))
    parser.add_argument("--all-path", default=None)
    parser.add_argument("--merge-ready-path", default=None)
    parser.add_argument("--needs-regen-path", default=None)
    parser.add_argument("--base-url", default="http://a0010:18000/v1")
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--max-tokens", type=int, default=12288)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1001)
    parser.add_argument("--max-concurrency", type=int, default=10)
    parser.add_argument("--sample-limit", type=int, default=None)
    parser.add_argument("--row-ids-file", default=None)
    parser.add_argument("--fallback-rounds", type=int, default=0)
    parser.add_argument("--fallback-chunk-tokens", type=int, default=1024)
    parser.add_argument(
        "--output-prefix", default="augmented_400_final_merge_ready_t1p0"
    )
    return parser


def load_jsonl(*, path: Path) -> list[dict[str, Any]]:
    """Load a JSONL file into memory.

    Args:
        path: Source JSONL path.

    Returns:
        Parsed row dictionaries.
    """

    return [json.loads(line) for line in path.open(encoding="utf-8")]


def write_jsonl(*, path: Path, rows: list[dict[str, Any]]) -> None:
    """Write rows to JSONL.

    Args:
        path: Destination JSONL path.
        rows: JSON-serializable row objects.

    Returns:
        None.
    """

    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def extract_user_prompt(*, row: dict[str, Any]) -> str:
    """Return the user message content for one dataset row."""
    for message in row["messages"]:
        if message.get("role") == "user":
            return str(message["content"])
    raise AssertionError(f"No user message found for row {row.get('id')}")


def split_assistant_content(*, content: str) -> tuple[str, str, str]:
    """Split assistant content into prefix, think body, and suffix.

    Args:
        content: Assistant message that contains one `<think>` block.

    Returns:
        Tuple of prefix, inner think text, and suffix.

    Example:
        >>> split_assistant_content(content="<think>\\na\\n</think>b")[1]
        'a'
    """

    start_marker = "<think>"
    end_marker = "</think>"
    start_index = content.index(start_marker)
    end_index = content.index(end_marker, start_index + len(start_marker))
    prefix = content[:start_index]
    think_text = content[start_index + len(start_marker) : end_index]
    suffix = content[end_index + len(end_marker) :]
    think_text = think_text.lstrip("\n").rstrip("\n")
    return prefix, think_text, suffix


def compute_think_token_count(
    *, encoding: tiktoken.Encoding, assistant_content: str
) -> int:
    """Count tokens in the inner think block.

    Args:
        encoding: Tokenizer encoding used by the augmentation pipeline.
        assistant_content: Full assistant message content.

    Returns:
        Token count for the inner `<think>` block.
    """

    _, think_text, _ = split_assistant_content(content=assistant_content)
    return len(encoding.encode(think_text))


def fetch_model_payload(*, base_url: str) -> dict[str, Any]:
    """Fetch the first advertised model payload from vLLM."""
    models_url = f"{normalize_vllm_base_url(base_url=base_url)}/models"
    with urllib_request.urlopen(models_url, timeout=20) as response:
        payload = json.loads(response.read().decode("utf-8"))
    models = payload.get("data", [])
    assert isinstance(models, list) and models, "No models returned by vLLM server"
    model_payload = models[0]
    assert isinstance(model_payload, dict), "Invalid model payload"
    return model_payload


def request_completion(
    *,
    base_url: str,
    model_id: str,
    prompt: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    seed: int,
) -> tuple[str, str, int | str | None]:
    """Run one `/v1/completions` request against vLLM.

    Args:
        base_url: OpenAI-compatible vLLM base URL.
        model_id: Served model identifier.
        prompt: Fully rendered prompt text.
        temperature: Sampling temperature.
        top_p: Nucleus sampling value.
        max_tokens: One-shot completion cap.
        seed: Deterministic seed.

    Returns:
        Generated text, finish reason, and optional stop reason.
    """

    payload = {
        "model": model_id,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "n": 1,
        "seed": seed,
        "stop": ["<|im_end|>"],
    }
    request_obj = urllib_request.Request(
        url=f"{normalize_vllm_base_url(base_url=base_url)}/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib_request.urlopen(request_obj, timeout=1800) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
    except urllib_error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"vLLM HTTP {exc.code}: {detail}") from exc

    choice = response_payload["choices"][0]
    return str(choice["text"]), str(choice["finish_reason"]), choice.get("stop_reason")


def has_closed_think(*, assistant_content: str) -> bool:
    """Return whether the assistant content contains a closed think block."""
    return "</think>" in assistant_content


def continue_until_think_closed(
    *,
    task: RegenTask,
    base_prompt: str,
    assistant_content: str,
    base_url: str,
    model_id: str,
    temperature: float,
    top_p: float,
    max_rounds: int,
    chunk_tokens: int,
    seed: int,
) -> tuple[str, str, int | str | None, int]:
    """Continue generation in smaller chunks until `</think>` appears.

    Args:
        task: Row-specific regeneration task.
        base_prompt: Tokenizer-rendered base prompt with open assistant turn.
        assistant_content: Current assistant text after the one-shot request.
        base_url: OpenAI-compatible vLLM base URL.
        model_id: Served model identifier.
        temperature: Sampling temperature.
        top_p: Nucleus sampling value.
        max_rounds: Maximum fallback continuation calls.
        chunk_tokens: Tokens per fallback continuation call.
        seed: Deterministic base seed.

    Returns:
        Updated assistant content, final finish reason, stop reason, and rounds used.
    """

    finish_reason = "open_think"
    stop_reason: int | str | None = None
    for round_index in range(max_rounds):
        prompt = attach_assistant_prefix(
            base_prompt=base_prompt,
            assistant_prefix=assistant_content,
        )
        text, finish_reason, stop_reason = request_completion(
            base_url=base_url,
            model_id=model_id,
            prompt=prompt,
            temperature=temperature,
            top_p=top_p,
            max_tokens=chunk_tokens,
            seed=seed + task.index * 100 + round_index,
        )
        assistant_content += text
        if has_closed_think(assistant_content=assistant_content):
            return assistant_content, finish_reason, stop_reason, round_index + 1
    raise ValueError("assistant content missing </think> after fallback continuation")


def build_base_prompt(*, tokenizer: Any, user_prompt: str) -> str:
    """Render the tokenizer chat template with an open assistant turn.

    Args:
        tokenizer: Hugging Face tokenizer with a chat template.
        user_prompt: Raw user prompt text.

    Returns:
        Chat-templated prompt text ending at assistant generation start.

    Example:
        >>> bool(build_base_prompt(tokenizer=tokenizer, user_prompt="Hi"))  # doctest: +SKIP
        True
    """

    return tokenizer.apply_chat_template(
        conversation=[{"role": "user", "content": user_prompt}],
        tokenize=False,
        add_generation_prompt=True,
    ).rstrip()


def attach_assistant_prefix(*, base_prompt: str, assistant_prefix: str) -> str:
    """Append the saved assistant prefix to the templated prompt."""
    if base_prompt.endswith("<think>") and assistant_prefix.startswith("<think>"):
        return base_prompt + assistant_prefix[len("<think>") :]
    return base_prompt + assistant_prefix


def build_task(*, index: int, row: dict[str, Any]) -> RegenTask:
    """Build one regeneration task from a needs-regen row."""
    regen_seed = row.get("regen_seed")
    assert isinstance(regen_seed, dict), f"Missing regen_seed for row {row.get('id')}"
    return RegenTask(
        index=index,
        row_id=str(row["id"]),
        user_prompt=extract_user_prompt(row=row),
        assistant_prefill=str(regen_seed["assistant_prefill"]),
        row=row,
    )


def filter_rows_by_id(
    *, rows: list[dict[str, Any]], row_ids_path: str | Path | None
) -> list[dict[str, Any]]:
    """Restrict rows to the IDs listed in a text file.

    Args:
        rows: Candidate dataset rows.
        row_ids_path: Optional newline-delimited row-id file.

    Returns:
        Filtered rows in their original order.
    """

    if row_ids_path is None:
        return rows
    selected_ids = {
        line.strip()
        for line in Path(row_ids_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    return [row for row in rows if str(row["id"]) in selected_ids]


def replace_last_assistant_message(
    *, row: dict[str, Any], assistant_content: str
) -> dict[str, Any]:
    """Return a row copy with the final assistant content replaced."""
    updated_row = copy.deepcopy(row)
    for index in range(len(updated_row["messages"]) - 1, -1, -1):
        if updated_row["messages"][index].get("role") == "assistant":
            updated_row["messages"][index]["content"] = assistant_content
            return updated_row
    raise AssertionError(f"No assistant message found for row {row.get('id')}")


def finalize_row(
    *,
    row: dict[str, Any],
    assistant_content: str,
    encoding: tiktoken.Encoding,
    model_id: str,
    base_url: str,
    temperature: float,
    max_tokens: int,
) -> dict[str, Any]:
    """Update a regenerated row for final merge readiness.

    Args:
        row: Source needs-regen row.
        assistant_content: Fully regenerated assistant content.
        encoding: Think-token counting encoding.
        model_id: Served model identifier.
        base_url: vLLM base URL used for regeneration.
        temperature: Sampling temperature.
        max_tokens: One-shot completion cap.

    Returns:
        Updated merge-ready dataset row.
    """

    updated_row = replace_last_assistant_message(
        row=row,
        assistant_content=assistant_content,
    )
    updated_row.pop("regen_seed", None)
    updated_row["think_token_count"] = compute_think_token_count(
        encoding=encoding,
        assistant_content=assistant_content,
    )
    augmentation_meta = copy.deepcopy(updated_row.get("augmentation_meta", {}))
    augmentation_meta["status"] = "merge_ready"
    augmentation_meta["regen"] = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "vllm_base_url": normalize_vllm_base_url(base_url=base_url),
        "vllm_model": model_id,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stop_tokens": ["<|im_end|>"],
    }
    updated_row["augmentation_meta"] = augmentation_meta
    return updated_row


def run_task(
    *,
    task: RegenTask,
    tokenizer: Any,
    encoding: tiktoken.Encoding,
    model_id: str,
    base_url: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    seed: int,
    fallback_rounds: int,
    fallback_chunk_tokens: int,
) -> RegenResult:
    """Regenerate one row with a single vLLM completion request.

    Args:
        task: Row-specific regeneration task.
        tokenizer: Checkpoint tokenizer for chat templating.
        encoding: Token counting encoding.
        model_id: Served model identifier.
        base_url: vLLM base URL.
        temperature: Sampling temperature.
        top_p: Nucleus sampling value.
        max_tokens: One-shot completion cap.
        seed: Deterministic base seed.
        fallback_rounds: Max fallback continuation calls after an open think.
        fallback_chunk_tokens: Tokens per fallback continuation call.

    Returns:
        Structured regeneration result.
    """

    started_at = time.perf_counter()
    base_prompt = build_base_prompt(tokenizer=tokenizer, user_prompt=task.user_prompt)
    rendered_prompt = attach_assistant_prefix(
        base_prompt=base_prompt,
        assistant_prefix=task.assistant_prefill,
    )
    try:
        text, finish_reason, stop_reason = request_completion(
            base_url=base_url,
            model_id=model_id,
            prompt=rendered_prompt,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            seed=seed + task.index,
        )
        assistant_content = task.assistant_prefill + text
        round_count = 1
        if not has_closed_think(assistant_content=assistant_content):
            if fallback_rounds <= 0:
                raise ValueError("assistant content missing </think>")
            (
                assistant_content,
                finish_reason,
                stop_reason,
                fallback_used,
            ) = continue_until_think_closed(
                task=task,
                base_prompt=base_prompt,
                assistant_content=assistant_content,
                base_url=base_url,
                model_id=model_id,
                temperature=temperature,
                top_p=top_p,
                max_rounds=fallback_rounds,
                chunk_tokens=fallback_chunk_tokens,
                seed=seed,
            )
            round_count += fallback_used
        completed_row = finalize_row(
            row=task.row,
            assistant_content=assistant_content,
            encoding=encoding,
            model_id=model_id,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return RegenResult(
            task=task,
            finish_reason=finish_reason,
            stop_reason=stop_reason,
            assistant_content=assistant_content,
            completed_row=completed_row,
            error=None,
            elapsed_seconds=time.perf_counter() - started_at,
            round_count=round_count,
        )
    except Exception as exc:  # noqa: BLE001
        return RegenResult(
            task=task,
            finish_reason="error",
            stop_reason=None,
            assistant_content=None,
            completed_row=None,
            error=f"{type(exc).__name__}: {exc}",
            elapsed_seconds=time.perf_counter() - started_at,
            round_count=0,
        )


def merge_final_rows(
    *, all_rows: list[dict[str, Any]], regenerated_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Replace needs-regen rows inside the full 400-row set."""
    regenerated_by_id = {str(row["id"]): row for row in regenerated_rows}
    merged_rows: list[dict[str, Any]] = []
    for row in all_rows:
        merged_rows.append(regenerated_by_id.get(str(row["id"]), row))
    return merged_rows


def validate_final_rows(*, rows: list[dict[str, Any]]) -> None:
    """Assert that final rows are merge-ready."""
    assert rows, "No rows to validate"
    assert all(
        "regen_seed" not in row for row in rows
    ), "regen_seed leaked into final rows"
    statuses = {row.get("augmentation_meta", {}).get("status") for row in rows}
    assert statuses == {"merge_ready"}, f"Unexpected statuses: {statuses}"


def build_summary(
    *,
    results: list[RegenResult],
    final_row_count: int | None,
    wrote_final_dataset: bool,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Build a compact JSON summary for the full regen run."""
    finish_counts = Counter(result.finish_reason for result in results)
    failed_ids = [result.task.row_id for result in results if result.error is not None]
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "row_count": final_row_count,
        "wrote_final_dataset": wrote_final_dataset,
        "regenerated_count": len(results),
        "success_count": len(results) - len(failed_ids),
        "failure_count": len(failed_ids),
        "failed_ids": failed_ids,
        "finish_counts": dict(finish_counts),
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "fallback_rounds": args.fallback_rounds,
        "fallback_chunk_tokens": args.fallback_chunk_tokens,
        "max_concurrency": args.max_concurrency,
        "base_url": normalize_vllm_base_url(base_url=args.base_url),
        "output_prefix": args.output_prefix,
    }


def main() -> None:
    """Run full one-shot regen and write merge-ready artifacts."""
    args = build_arg_parser().parse_args()
    run_dir = Path(args.run_dir)
    all_path = Path(args.all_path or run_dir / "augmented_400_all.jsonl")
    merge_ready_path = Path(
        args.merge_ready_path or run_dir / "augmented_400_merge_ready.jsonl"
    )
    needs_regen_path = Path(
        args.needs_regen_path or run_dir / "augmented_400_needs_regen.jsonl"
    )
    all_rows = load_jsonl(path=all_path)
    merge_ready_rows = load_jsonl(path=merge_ready_path)
    needs_regen_rows = load_jsonl(path=needs_regen_path)
    if args.sample_limit is not None:
        needs_regen_rows = needs_regen_rows[: args.sample_limit]
    needs_regen_rows = filter_rows_by_id(
        rows=needs_regen_rows,
        row_ids_path=args.row_ids_file,
    )

    model_payload = fetch_model_payload(base_url=args.base_url)
    model_id = str(args.model_id or model_payload["id"])
    tokenizer_path = str(args.tokenizer_path or model_payload["root"])
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    encoding = tiktoken.get_encoding("cl100k_base")
    tasks = [
        build_task(index=index, row=row) for index, row in enumerate(needs_regen_rows)
    ]

    print(
        f"Starting full regen for {len(tasks)} rows "
        f"with max_concurrency={args.max_concurrency}, max_tokens={args.max_tokens}, "
        f"temperature={args.temperature}."
    )
    started_at = time.perf_counter()
    results: list[RegenResult] = []
    with ThreadPoolExecutor(max_workers=args.max_concurrency) as executor:
        future_map = {
            executor.submit(
                run_task,
                task=task,
                tokenizer=tokenizer,
                encoding=encoding,
                model_id=model_id,
                base_url=args.base_url,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
                seed=args.seed,
                fallback_rounds=args.fallback_rounds,
                fallback_chunk_tokens=args.fallback_chunk_tokens,
            ): task
            for task in tasks
        }
        for done_count, future in enumerate(as_completed(future_map), start=1):
            result = future.result()
            results.append(result)
            status = "ok" if result.error is None else "error"
            line = (
                f"[{done_count}/{len(tasks)}] {status} finish={result.finish_reason} "
                f"rounds={result.round_count} time={result.elapsed_seconds:.1f}s "
                f"row_id={result.task.row_id}"
            )
            if result.error is not None:
                line += f" error={result.error[:160]}"
            print(line)

    results.sort(key=lambda item: item.task.index)
    regenerated_rows = [
        result.completed_row for result in results if result.completed_row is not None
    ]
    failed_results = [result for result in results if result.error is not None]

    output_prefix = args.output_prefix
    regen_rows_path = run_dir / f"{output_prefix}.regen_rows.jsonl"
    final_path = run_dir / f"{output_prefix}.jsonl"
    results_path = run_dir / f"{output_prefix}.results.json"
    summary_path = run_dir / f"{output_prefix}.summary.json"
    write_jsonl(path=regen_rows_path, rows=regenerated_rows)
    results_path.write_text(
        json.dumps([result.to_json() for result in results], indent=2),
        encoding="utf-8",
    )
    final_rows: list[dict[str, Any]] | None = None
    wrote_final_dataset = False
    is_full_regen = len(merge_ready_rows) + len(regenerated_rows) == len(all_rows)
    if not failed_results and is_full_regen:
        final_rows = merge_final_rows(
            all_rows=all_rows, regenerated_rows=regenerated_rows
        )
        validate_final_rows(rows=final_rows)
        assert len(final_rows) == len(all_rows)
        write_jsonl(path=final_path, rows=final_rows)
        wrote_final_dataset = True
    summary_path.write_text(
        json.dumps(
            build_summary(
                results=results,
                final_row_count=len(final_rows) if final_rows is not None else None,
                wrote_final_dataset=wrote_final_dataset,
                args=args,
            ),
            indent=2,
        ),
        encoding="utf-8",
    )

    elapsed = time.perf_counter() - started_at
    print(f"Wrote regenerated rows: {regen_rows_path}")
    if final_rows is not None:
        print(f"Wrote final merge-ready dataset: {final_path}")
    print(f"Wrote per-row results: {results_path}")
    print(f"Wrote summary: {summary_path}")
    print(f"Completed in {elapsed:.1f}s.")
    if failed_results:
        raise AssertionError(f"{len(failed_results)} rows failed regeneration")


if __name__ == "__main__":
    main()
