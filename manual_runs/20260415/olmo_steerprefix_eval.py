"""Run one OLMo AIME avg@32 eval with an injected `<steer>` assistant prefix.

Example:
    uv run --project Analysis python manual_runs/20260415/olmo_steerprefix_eval.py \
        --task-name aime25 \
        --model-path /fs/scratch/.../checkpoint-448 \
        --source-samples /users/.../samples_aime25_....jsonl \
        --base-url http://127.0.0.1:8000/v1 \
        --output-json /users/.../result.json \
        --output-samples /users/.../samples.jsonl
"""

from __future__ import annotations

import asyncio
import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib import request as urllib_request

from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "Eval"))

from eval_runner.aime_avgk import compute_avg_at_k, extract_target_answer  # noqa: E402


@dataclass(frozen=True)
class EvalDoc:
    """One AIME document with its standalone user prompt.

    Args:
        doc_id: Original eval doc id.
        doc: Source document payload.
        user_prompt: User-facing prompt text before chat templating.

    Returns:
        Dataclass describing one eval prompt row.
    """

    doc_id: int
    doc: dict[str, Any]
    user_prompt: str

    def to_sample_row(self, *, responses: list[str]) -> dict[str, Any]:
        """Build one saved sample row from generated responses.

        Args:
            responses: Generated response texts for this document.

        Returns:
            JSON-serializable sample payload with `avg_at_32`.
        """

        target = extract_target_answer(doc=self.doc)
        avg_at_32 = compute_avg_at_k(responses=responses, target=target)
        return {
            "doc_id": self.doc_id,
            "doc": self.doc,
            "target": target,
            "arguments": {
                "user_prompt": self.user_prompt,
                "assistant_prefix": "<think><steer>",
            },
            "resps": [responses],
            "filtered_resps": [responses],
            "filter": "avg@32",
            "metrics": ["avg_at_32"],
            "avg_at_32": avg_at_32,
        }


@dataclass(frozen=True)
class PreparedEvalDoc:
    """One eval row after prompt rendering and budget calculation.

    Args:
        index: Stable position from the source sample log.
        eval_doc: Source eval document metadata.
        prompt: Raw prompt ending in `<think><steer>`.
        completion_budget: Completion-token budget for this prompt.

    Returns:
        Dataclass used to launch one completion request.
    """

    index: int
    eval_doc: EvalDoc
    prompt: str
    completion_budget: int


@dataclass(frozen=True)
class EvalDocResult:
    """One completed doc result with stable ordering metadata.

    Args:
        index: Stable position from the source sample log.
        sample_row: JSON-serializable sample payload.

    Returns:
        Dataclass used to reassemble outputs in source order.
    """

    index: int
    sample_row: dict[str, Any]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for one steer-prefixed eval run.

    Returns:
        Parsed namespace with source paths, model metadata, and output paths.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--source-samples", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-samples", required=True)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--num-rollouts", type=int, default=32)
    parser.add_argument("--doc-concurrency", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def build_user_prompt(*, doc: dict[str, Any]) -> str:
    """Build the AIME standalone question prompt from one source document.

    Args:
        doc: AIME document payload containing `problem` or `Problem`.

    Returns:
        Prompt text in `Question: ...\\nAnswer:` format.
    """

    problem = doc.get("problem", doc.get("Problem"))
    assert isinstance(problem, str), f"Missing problem text in doc: {doc}"
    return f"Question: {problem}\nAnswer:"


def load_docs(*, source_samples_path: Path) -> list[EvalDoc]:
    """Load ordered eval documents from a prior standalone sample log.

    Args:
        source_samples_path: Existing sample JSONL containing doc order and payloads.

    Returns:
        Ordered evaluation documents for rerun.
    """

    docs: list[EvalDoc] = []
    for line in source_samples_path.open(encoding="utf-8"):
        row = json.loads(line)
        doc = row["doc"]
        docs.append(
            EvalDoc(
                doc_id=int(row["doc_id"]),
                doc=doc,
                user_prompt=build_user_prompt(doc=doc),
            )
        )
    return docs


def build_prefixed_prompt(*, tokenizer: Any, user_prompt: str) -> str:
    """Render one chat prompt and inject `<steer>` after the template `<think>`.

    Args:
        tokenizer: HF tokenizer with a chat template.
        user_prompt: User-side question prompt text.

    Returns:
        Raw completion prompt ending in `<think><steer>`.
    """

    prompt = tokenizer.apply_chat_template(
        conversation=[{"role": "user", "content": user_prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    assert isinstance(prompt, str), f"Unexpected prompt type: {type(prompt)}"
    stripped_prompt = prompt.rstrip()
    assert stripped_prompt.endswith("<think>"), stripped_prompt[-200:]
    return f"{stripped_prompt}<steer>"


def compute_completion_budget(
    *,
    tokenizer: Any,
    prompt: str,
    requested_max_tokens: int,
    max_model_len: int,
) -> int:
    """Compute a legal completion budget for one raw prompt.

    Args:
        tokenizer: HF tokenizer used for the served model.
        prompt: Fully rendered completion prompt.
        requested_max_tokens: Requested completion cap.
        max_model_len: Total prompt-plus-generation token budget.

    Returns:
        Completion token budget after subtracting prompt length.

    Example:
        >>> budget = compute_completion_budget(
        ...     tokenizer=tokenizer,
        ...     prompt="<|im_start|>assistant\\n<think><steer>",
        ...     requested_max_tokens=32768,
        ...     max_model_len=32768,
        ... )
        >>> assert budget >= 1
    """

    prompt_tokens = tokenizer.encode(text=prompt, add_special_tokens=False)
    available_tokens = max_model_len - len(prompt_tokens)
    assert available_tokens > 0, (
        f"Prompt uses {len(prompt_tokens)} tokens, exceeding "
        f"max_model_len={max_model_len}"
    )
    return min(requested_max_tokens, available_tokens)


def request_rollouts(
    *,
    base_url: str,
    model_path: str,
    prompt: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    num_rollouts: int,
    seed: int,
) -> list[str]:
    """Generate `n` raw completions from one prefixed prompt via vLLM.

    Args:
        base_url: OpenAI-compatible vLLM base URL ending in `/v1`.
        model_path: Served model identifier.
        prompt: Raw completion prompt text.
        temperature: Sampling temperature.
        top_p: Nucleus sampling value.
        max_tokens: Max generated tokens per rollout.
        num_rollouts: Number of sampled completions.
        seed: Request seed.

    Returns:
        Generated response texts in server order.
    """

    payload = {
        "model": model_path,
        "prompt": prompt,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "n": num_rollouts,
        "seed": seed,
    }
    request = urllib_request.Request(
        url=f"{base_url.rstrip('/')}/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib_request.urlopen(request, timeout=7200) as response:
        body = json.load(response)
    return [str(choice["text"]) for choice in body["choices"]]


def prepare_eval_docs(
    *,
    docs: list[EvalDoc],
    tokenizer: Any,
    requested_max_tokens: int,
    max_model_len: int,
) -> list[PreparedEvalDoc]:
    """Render prompts and per-doc budgets before launching async requests.

    Args:
        docs: Ordered source documents for the rerun.
        tokenizer: HF tokenizer used to render prompts and count tokens.
        requested_max_tokens: Requested completion cap.
        max_model_len: Total prompt-plus-generation token budget.

    Returns:
        Prepared requests ready for concurrent generation.
    """

    prepared_docs: list[PreparedEvalDoc] = []
    for index, eval_doc in enumerate(docs):
        prompt = build_prefixed_prompt(tokenizer=tokenizer, user_prompt=eval_doc.user_prompt)
        completion_budget = compute_completion_budget(
            tokenizer=tokenizer,
            prompt=prompt,
            requested_max_tokens=requested_max_tokens,
            max_model_len=max_model_len,
        )
        prepared_docs.append(
            PreparedEvalDoc(
                index=index,
                eval_doc=eval_doc,
                prompt=prompt,
                completion_budget=completion_budget,
            )
        )
    return prepared_docs


async def evaluate_prepared_doc(
    *,
    prepared_doc: PreparedEvalDoc,
    base_url: str,
    model_path: str,
    temperature: float,
    top_p: float,
    num_rollouts: int,
    seed: int,
    semaphore: asyncio.Semaphore,
) -> EvalDocResult:
    """Run one prepared doc request under a shared concurrency semaphore.

    Args:
        prepared_doc: Prompt and completion budget for one doc.
        base_url: OpenAI-compatible vLLM base URL ending in `/v1`.
        model_path: Served model identifier.
        temperature: Sampling temperature.
        top_p: Nucleus sampling value.
        num_rollouts: Number of completions to sample.
        seed: Request seed.
        semaphore: Shared concurrency limiter.

    Returns:
        Completed per-doc sample row with original ordering index.
    """

    async with semaphore:
        responses = await asyncio.to_thread(
            request_rollouts,
            base_url=base_url,
            model_path=model_path,
            prompt=prepared_doc.prompt,
            temperature=temperature,
            top_p=top_p,
            max_tokens=prepared_doc.completion_budget,
            num_rollouts=num_rollouts,
            seed=seed,
        )
    sample_row = prepared_doc.eval_doc.to_sample_row(responses=responses)
    print(
        json.dumps(
            {
                "doc_id": prepared_doc.eval_doc.doc_id,
                "avg_at_32": sample_row["avg_at_32"],
                "num_rollouts": len(responses),
            }
        ),
        flush=True,
    )
    return EvalDocResult(index=prepared_doc.index, sample_row=sample_row)


async def run_eval(
    *,
    prepared_docs: list[PreparedEvalDoc],
    base_url: str,
    model_path: str,
    temperature: float,
    top_p: float,
    num_rollouts: int,
    seed: int,
    doc_concurrency: int,
    on_result: Callable[[EvalDocResult], None] | None = None,
) -> list[EvalDocResult]:
    """Run the full eval with bounded doc-level concurrency.

    Args:
        prepared_docs: Ordered prompt requests for the rerun.
        base_url: OpenAI-compatible vLLM base URL ending in `/v1`.
        model_path: Served model identifier.
        temperature: Sampling temperature.
        top_p: Nucleus sampling value.
        num_rollouts: Number of completions to sample per doc.
        seed: Request seed.
        doc_concurrency: Max docs allowed in flight at once.
        on_result: Optional callback invoked after each completed doc.

    Returns:
        Completed results sorted back into source order.

    Example:
        >>> results = asyncio.run(
        ...     run_eval(
        ...         prepared_docs=prepared_docs,
        ...         base_url="http://127.0.0.1:8040/v1",
        ...         model_path="checkpoint-448",
        ...         temperature=0.6,
        ...         top_p=0.95,
        ...         num_rollouts=32,
        ...         seed=1234,
        ...         doc_concurrency=2,
        ...     )
        ... )
        >>> assert len(results) == len(prepared_docs)
    """

    semaphore = asyncio.Semaphore(doc_concurrency)
    tasks = [
        asyncio.create_task(
            evaluate_prepared_doc(
                prepared_doc=prepared_doc,
                base_url=base_url,
                model_path=model_path,
                temperature=temperature,
                top_p=top_p,
                num_rollouts=num_rollouts,
                seed=seed,
                semaphore=semaphore,
            )
        )
        for prepared_doc in prepared_docs
    ]
    completed_results: dict[int, EvalDocResult] = {}
    for task in asyncio.as_completed(tasks):
        result = await task
        completed_results[result.index] = result
        if on_result is not None:
            on_result(result)
    return [
        completed_results[index]
        for index in range(len(prepared_docs))
        if index in completed_results
    ]


def build_aggregate_payload(
    *,
    args: argparse.Namespace,
    sample_rows: list[dict[str, Any]],
    expected_docs: int,
) -> dict[str, Any]:
    """Build the saved aggregate JSON payload for one rerun.

    Args:
        args: Parsed CLI arguments for the run.
        sample_rows: Completed per-doc sample rows in source order.
        expected_docs: Total docs expected for the run.

    Returns:
        Aggregate JSON payload containing `avg_at_32`.
    """

    doc_scores = [float(row["avg_at_32"]) for row in sample_rows]
    completed_docs = len(sample_rows)
    return {
        "results": {args.task_name: {"avg_at_32": sum(doc_scores) / len(doc_scores)}},
        "metadata": {
            "model_path": args.model_path,
            "source_samples": args.source_samples,
            "assistant_prefix": "<think><steer>",
            "base_url": args.base_url,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "max_model_len": args.max_model_len,
            "num_rollouts": args.num_rollouts,
            "doc_concurrency": args.doc_concurrency,
            "completed_docs": completed_docs,
            "expected_docs": expected_docs,
            "is_complete": completed_docs == expected_docs,
            "seed": args.seed,
        },
    }


def ordered_sample_rows(
    *, completed_results_by_index: dict[int, EvalDocResult]
) -> list[dict[str, Any]]:
    """Return completed sample rows sorted back into source order.

    Args:
        completed_results_by_index: Finished per-doc results keyed by source index.

    Returns:
        Completed sample rows in source order.
    """

    return [
        completed_results_by_index[index].sample_row
        for index in sorted(completed_results_by_index)
    ]


def persist_progress(
    *,
    output_json_path: Path,
    output_samples_path: Path,
    args: argparse.Namespace,
    completed_results_by_index: dict[int, EvalDocResult],
    expected_docs: int,
) -> None:
    """Persist aggregate and samples for the completed-doc prefix.

    Args:
        output_json_path: Aggregate JSON destination.
        output_samples_path: Sample JSONL destination.
        args: Parsed CLI arguments.
        completed_results_by_index: Finished per-doc results keyed by source index.
        expected_docs: Total docs expected for the run.

    Returns:
        None.
    """

    sample_rows = ordered_sample_rows(
        completed_results_by_index=completed_results_by_index
    )
    aggregate = build_aggregate_payload(
        args=args,
        sample_rows=sample_rows,
        expected_docs=expected_docs,
    )
    output_json_path.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    write_jsonl(path=output_samples_path, rows=sample_rows)


def write_jsonl(*, path: Path, rows: list[dict[str, Any]]) -> None:
    """Write newline-delimited JSON rows to disk.

    Args:
        path: Destination JSONL path.
        rows: Sample rows to persist.

    Returns:
        None.
    """

    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row))
            handle.write("\n")


def main() -> None:
    """Run the steer-prefixed OLMo avg@32 eval and persist aggregate outputs."""

    args = parse_args()
    output_json_path = Path(args.output_json)
    output_samples_path = Path(args.output_samples)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_samples_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=args.model_path,
        trust_remote_code=True,
        fix_mistral_regex=True,
    )
    docs = load_docs(source_samples_path=Path(args.source_samples))
    prepared_docs = prepare_eval_docs(
        docs=docs,
        tokenizer=tokenizer,
        requested_max_tokens=args.max_tokens,
        max_model_len=args.max_model_len,
    )
    completed_results_by_index: dict[int, EvalDocResult] = {}

    def handle_result(result: EvalDocResult) -> None:
        """Persist one newly completed doc to disk."""

        completed_results_by_index[result.index] = result
        persist_progress(
            output_json_path=output_json_path,
            output_samples_path=output_samples_path,
            args=args,
            completed_results_by_index=completed_results_by_index,
            expected_docs=len(prepared_docs),
        )

    completed_results = asyncio.run(
        run_eval(
            prepared_docs=prepared_docs,
            base_url=args.base_url,
            model_path=args.model_path,
            temperature=args.temperature,
            top_p=args.top_p,
            num_rollouts=args.num_rollouts,
            seed=args.seed,
            doc_concurrency=args.doc_concurrency,
            on_result=handle_result,
        )
    )
    sample_rows = [result.sample_row for result in completed_results]
    aggregate = build_aggregate_payload(
        args=args,
        sample_rows=sample_rows,
        expected_docs=len(prepared_docs),
    )
    output_json_path.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    write_jsonl(path=output_samples_path, rows=sample_rows)


if __name__ == "__main__":
    main()
