#!/usr/bin/env python3
"""Generate a cleaned SFT warm-up dataset with the branching engine."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, cast

from datasets import Dataset

ANALYSIS_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = ANALYSIS_ROOT.parent
ANALYSIS_ROOT_TEXT = str(ANALYSIS_ROOT)
if ANALYSIS_ROOT_TEXT not in sys.path:
    sys.path.insert(0, ANALYSIS_ROOT_TEXT)

from branching_eval.artifact_store import ArtifactStore, build_run_manifest_payload
from branching_eval.branch_executor import BranchExecutor
from branching_eval.config_types import BranchingConfig, DecodingConfig, ExperimentSpec
from branching_eval.doc_progress import DocProgressTracker
from branching_eval.event_types import EventContext, utc_now_iso
from branching_eval.run_matrix import (
    leaf_scored_event_payload,
    read_git_commit_hash,
    selector_mode_for_spec,
)
from branching_eval.tree_types import LeafRollout
from branching_eval.vllm_runtime import wait_for_existing_server
from sft_warmup_cleaning import (
    WarmupResultRow,
    append_jsonl,
    clean_completion,
    read_jsonl,
    write_cleaned_outputs,
    write_json,
)
from vllm_client import VllmClient, VllmRequestError

DEFAULT_INPUT_PARQUET = REPO_ROOT / "BuildRLDataset/output/train.parquet"
DEFAULT_INPUT_JSONL = (
    REPO_ROOT
    / "BuildSFTDataset/output_transform_async_16384/transformed_subset_analysis/"
    "merged_with_output_transformed_output_aug390_t1p0_pruned_top10_max_exec_"
    "truncated_seed42_plus_non_sequitur500_1to2_v2.jsonl"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/fs/scratch/PAA0201/ollieproudman/DecomposedReasoning/"
    "Analysis/branching_eval/sft_warmup_off_policy"
)
DEFAULT_SYSTEM_PROMPT = (
    "Solve the task. Put your reasoning in one <think>...</think> "
    "block made of alternating non-empty <steer>...</steer> and "
    "<exec>...</exec> blocks, starting with <steer>. Use <steer> blocks "
    "to guide thinking, make executive decisions, choose subproblems, slow "
    "down, enumerate, verify, or backtrack. Examples: Guide thinking: "
    '"Try applying ___." Make decisions: "Name the dog \'___\'." '
    'Choose subproblems: "Consider a<=3." Slow down: '
    '"Use a more precise method." Enumerate: "List 5 options and choose '
    'one." Verify: "Double Check that calculation." Backtrack: '
    '"Abandon this approach." Use <exec> blocks to precisely carry out the '
    "chosen guidance with concrete work and deductions. After </think>, give "
    "the final answer clearly and concisely."
)
DEFAULT_MODEL = "qwen35-4b-5611097-step300"
DEFAULT_BASE_URL = "http://a0111:18001/v1"


@dataclass(frozen=True)
class WarmupPromptRecord:
    """One source prompt row used for warm-up generation."""

    source_index: int
    custom_id: str
    user_prompt: str
    conversation_messages: list[dict[str, str]]
    ground_truth: str
    source_payload: dict[str, Any]

    def rendered_prompt(
        self, *, system_prompt: str, initial_assistant_prefix: str
    ) -> str:
        """Render the completion endpoint prompt with an assistant prefix."""

        messages = self.prompt_messages(system_prompt=system_prompt)
        rendered_messages = [
            f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
            for message in messages
        ]
        return (
            "".join(rendered_messages)
            + f"<|im_start|>assistant\n{initial_assistant_prefix}"
        )

    def prompt_messages(self, *, system_prompt: str) -> list[dict[str, str]]:
        """Return the prompt messages stored in the cleaned dataset."""

        return [
            {"role": "system", "content": system_prompt},
            *self.conversation_messages,
        ]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-parquet", type=Path, default=DEFAULT_INPUT_PARQUET)
    parser.add_argument("--input-jsonl", type=Path, default=None)
    parser.add_argument("--source-indices-file", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max-concurrent-docs", type=int, default=16)
    parser.add_argument("--epsilon-greedy-prob", type=float, default=0.02)
    parser.add_argument("--off-policy-min-candidates", type=int, default=3)
    parser.add_argument("--off-policy-max-candidates", type=int, default=10)
    parser.add_argument("--max-gen-toks", type=int, default=16384)
    parser.add_argument("--max-model-len", type=int, default=17408)
    parser.add_argument("--decode-chunk-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-logprobs", type=int, default=5)
    parser.add_argument("--startup-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--force-initial-steer-open", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run the dataset generation workflow."""

    args = parse_args()
    run_dir = resolve_run_dir(output_root=args.output_root, run_dir=args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    records = load_prompt_records(
        input_parquet=args.input_parquet,
        input_jsonl=args.input_jsonl,
    )
    selected_records = slice_records(
        records=records, offset=args.offset, limit=args.limit
    )
    selected_records = filter_source_indices(
        records=selected_records,
        source_indices_file=args.source_indices_file,
    )
    asyncio.run(run_generation(args=args, run_dir=run_dir, records=selected_records))


def resolve_run_dir(*, output_root: Path, run_dir: Path | None) -> Path:
    """Return the artifact directory for this generation run."""

    if run_dir is not None:
        return run_dir
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return output_root / f"sft_warmup_ckpt300_offpolicy_p02_nobranch_{timestamp}"


def load_prompt_records(
    *, input_parquet: Path, input_jsonl: Path | None
) -> list[WarmupPromptRecord]:
    """Load prompt rows from the selected SFT/RL dataset artifact."""

    if input_jsonl is not None:
        return load_prompt_records_from_jsonl(input_jsonl=input_jsonl)
    dataset = Dataset.from_parquet(str(input_parquet))
    records: list[WarmupPromptRecord] = []
    for source_index, row in enumerate(cast(Iterable[dict[str, Any]], dataset)):
        records.append(record_from_parquet_row(row=row, source_index=source_index))
    return records


def load_prompt_records_from_jsonl(*, input_jsonl: Path) -> list[WarmupPromptRecord]:
    """Load prompt rows from the final filtered SFT JSONL artifact."""

    records: list[WarmupPromptRecord] = []
    with input_jsonl.open("r", encoding="utf-8") as handle:
        for source_index, line in enumerate(handle):
            stripped = line.strip()
            if not stripped:
                continue
            row = cast(dict[str, Any], json.loads(stripped))
            records.append(record_from_messages_row(row=row, source_index=source_index))
    return records


def record_from_parquet_row(
    *, row: dict[str, Any], source_index: int
) -> WarmupPromptRecord:
    """Convert one RL parquet row into the prompt record used by the engine."""

    prompt_messages = row.get("prompt")
    assert isinstance(
        prompt_messages, list
    ), "dataset row prompt must be a message list"
    user_messages = [
        message
        for message in prompt_messages
        if isinstance(message, dict) and message.get("role") == "user"
    ]
    assert len(user_messages) == 1, "warm-up rows must have exactly one user prompt"
    user_content = user_messages[0].get("content")
    assert isinstance(user_content, str) and user_content.strip(), "empty user prompt"
    custom_id = str(row.get("custom_id", f"row_{source_index}"))
    ground_truth = str(row.get("ground_truth", ""))
    return WarmupPromptRecord(
        source_index=source_index,
        custom_id=custom_id,
        user_prompt=user_content,
        conversation_messages=[{"role": "user", "content": user_content}],
        ground_truth=ground_truth,
        source_payload=dict(row),
    )


def record_from_messages_row(
    *, row: dict[str, Any], source_index: int
) -> WarmupPromptRecord:
    """Convert one final SFT JSONL row into the runtime prompt record."""

    messages = row.get("messages")
    assert isinstance(messages, list), "SFT row messages must be a list"
    prompt_messages = source_prompt_messages(messages=messages)
    user_messages = [
        message for message in prompt_messages if message["role"] == "user"
    ]
    assert user_messages, "SFT rows must include at least one user prompt"
    user_content = user_messages[-1]["content"]
    return WarmupPromptRecord(
        source_index=source_index,
        custom_id=str(row.get("id", f"row_{source_index}")),
        user_prompt=conversation_preview(messages=prompt_messages),
        conversation_messages=prompt_messages,
        ground_truth="",
        source_payload=dict(row),
    )


def source_prompt_messages(*, messages: list[Any]) -> list[dict[str, str]]:
    """Return source messages before the SFT target assistant completion."""

    end_index = len(messages)
    if (
        messages
        and isinstance(messages[-1], dict)
        and messages[-1].get("role") == "assistant"
    ):
        end_index -= 1
    prompt_messages: list[dict[str, str]] = []
    for message in messages[:end_index]:
        assert isinstance(message, dict), "message row must be an object"
        role = str(message.get("role", "")).strip()
        content = message.get("content")
        assert role in {"system", "user", "assistant"}, f"unsupported role: {role}"
        assert isinstance(content, str) and content.strip(), "empty message content"
        if role == "system":
            continue
        prompt_messages.append({"role": role, "content": content})
    assert prompt_messages, "SFT row has no prompt messages"
    assert prompt_messages[-1]["role"] == "user", "SFT prompt must end with user turn"
    return prompt_messages


def conversation_preview(*, messages: list[dict[str, str]]) -> str:
    """Return a compact text representation of the prompt conversation."""

    return "\n\n".join(
        f"{message['role']}:\n{message['content']}" for message in messages
    )


def slice_records(
    *, records: list[WarmupPromptRecord], offset: int, limit: int | None
) -> list[WarmupPromptRecord]:
    """Apply offset and limit to source records."""

    assert offset >= 0, "offset must be non-negative"
    end = None if limit is None else offset + max(0, limit)
    return records[offset:end]


def filter_source_indices(
    *, records: list[WarmupPromptRecord], source_indices_file: Path | None
) -> list[WarmupPromptRecord]:
    """Restrict prompt records to explicit source indices when requested."""

    if source_indices_file is None:
        return records
    source_indices = read_source_indices(path=source_indices_file)
    by_index = {record.source_index: record for record in records}
    missing = sorted(source_indices.difference(by_index))
    assert not missing, f"source indices not found: {missing[:10]}"
    return [by_index[source_index] for source_index in sorted(source_indices)]


def read_source_indices(*, path: Path) -> set[int]:
    """Read one source index per line, or JSONL rows with source_index."""

    source_indices: set[int] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("{"):
                row = cast(dict[str, Any], json.loads(stripped))
                source_indices.add(int(row["source_index"]))
            else:
                source_indices.add(int(stripped))
    assert source_indices, f"empty source index file: {path}"
    return source_indices


def initial_assistant_prefix(*, args: argparse.Namespace) -> str:
    """Return the assistant text prefilled before generation."""

    if args.force_initial_steer_open:
        return "<think>\n<steer>"
    return "<think>\n"


def render_record_prompt(
    *, record: WarmupPromptRecord, args: argparse.Namespace
) -> str:
    """Render the exact prompt sent to vLLM for tokenization."""

    return record.rendered_prompt(
        system_prompt=args.system_prompt,
        initial_assistant_prefix=initial_assistant_prefix(args=args),
    )


async def run_generation(
    *, args: argparse.Namespace, run_dir: Path, records: list[WarmupPromptRecord]
) -> None:
    """Generate, log, and clean all requested prompt records."""

    wait_for_existing_server(
        base_url=args.base_url,
        timeout_seconds=args.startup_timeout_seconds,
        poll_interval_seconds=2.0,
    )
    client = VllmClient(base_url=args.base_url, timeout_seconds=None)
    store = ArtifactStore(run_dir=run_dir, run_id=run_dir.name)
    write_run_setup(args=args, run_dir=run_dir, records=records, store=store)
    write_lock = asyncio.Lock()
    semaphore = asyncio.Semaphore(args.max_concurrent_docs)
    raw_path = run_dir / "raw_generations.jsonl"
    completed_ids = read_completed_source_indices(raw_path=raw_path)
    pending = [record for record in records if record.source_index not in completed_ids]
    try:
        tasks = [
            asyncio.create_task(
                run_one_record(
                    args=args,
                    record=record,
                    client=client,
                    store=store,
                    run_dir=run_dir,
                    raw_path=raw_path,
                    write_lock=write_lock,
                    semaphore=semaphore,
                )
            )
            for record in pending
        ]
        for task in asyncio.as_completed(tasks):
            row = await task
            print(
                f"completed source_index={row.source_index} "
                f"valid={int(row.structure_valid)} len={row.length_tokens_total}",
                flush=True,
            )
        cleaning_summary = write_cleaned_outputs(run_dir=run_dir)
        store.append_event(
            context=run_context(run_id=store.run_id),
            event_type="run_finished",
            payload=cleaning_summary,
        )
    finally:
        await client.close_async()
        store.close()


def write_run_setup(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    records: list[WarmupPromptRecord],
    store: ArtifactStore,
) -> None:
    """Write manifest/config artifacts and append the run-start event."""

    spec = experiment_spec()
    config_snapshot = {
        "script": Path(__file__).name,
        "input_parquet": str(args.input_parquet),
        "input_jsonl": str(args.input_jsonl) if args.input_jsonl is not None else None,
        "base_url": args.base_url,
        "model": args.model,
        "system_prompt": args.system_prompt,
        "initial_assistant_prefix": initial_assistant_prefix(args=args),
        "epsilon_greedy_prob": args.epsilon_greedy_prob,
        "off_policy_min_candidates": args.off_policy_min_candidates,
        "off_policy_max_candidates": args.off_policy_max_candidates,
        "branch_fanout": 1,
        "rollouts_per_prompt": 1,
        "max_gen_toks": args.max_gen_toks,
        "max_model_len": args.max_model_len,
        "decode_chunk_tokens": args.decode_chunk_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_logprobs": args.top_logprobs,
    }
    write_json(path=run_dir / "generation_config.json", payload=config_snapshot)
    store.write_config_snapshot(payload=config_snapshot)
    store.write_run_manifest(
        payload=build_run_manifest_payload(
            run_id=store.run_id,
            task_name=spec.task_name,
            model_id=spec.model_id,
            selector_mode=selector_mode_for_spec(spec=spec),
            config_snapshot=config_snapshot,
            git_commit=read_git_commit_hash(),
        )
    )
    store.append_event(
        context=run_context(run_id=store.run_id),
        event_type="run_started",
        payload={
            "doc_count_total": len(records),
            "mode": spec.mode,
            "dataset_source": str(args.input_jsonl or args.input_parquet),
            "source_indices_file": (
                str(args.source_indices_file)
                if args.source_indices_file is not None
                else None
            ),
            "rollouts_per_prompt": 1,
        },
    )


async def run_one_record(
    *,
    args: argparse.Namespace,
    record: WarmupPromptRecord,
    client: VllmClient,
    store: ArtifactStore,
    run_dir: Path,
    raw_path: Path,
    write_lock: asyncio.Lock,
    semaphore: asyncio.Semaphore,
) -> WarmupResultRow:
    """Generate one source prompt under the shared concurrency limit."""

    async with semaphore:
        context = doc_context(run_id=store.run_id, doc_id=record.source_index)
        rendered_prompt = render_record_prompt(record=record, args=args)
        prompt_token_ids = await client.tokenize_async(
            model=args.model,
            text=rendered_prompt,
            add_special_tokens=False,
        )
        store.append_event(
            context=context,
            event_type="doc_started",
            payload={"prompt_char_count": len(record.user_prompt)},
        )
        store.append_event(
            context=context,
            event_type="prompt_logged",
            payload=prompt_payload(record=record, args=args),
        )
        tracker = progress_tracker(store=store, record=record)
        store.write_doc_progress(
            snapshot=tracker.snapshot(last_update_timestamp=utc_now_iso())
        )
        try:
            leaf = await generate_leaf(
                args=args,
                record=record,
                client=client,
                store=store,
                context=context,
                initial_prompt_token_ids=prompt_token_ids,
                tracker=tracker,
            )
            row = result_row(record=record, leaf=leaf, system_prompt=args.system_prompt)
            finished_payload = doc_finished_payload(leaf=leaf, row=row)
        except VllmRequestError as exc:
            if not is_context_length_error(error=exc):
                raise
            row = failed_result_row(
                record=record,
                system_prompt=args.system_prompt,
                cleaning_reason="context_length_request_rejected",
            )
            finished_payload = doc_failed_payload(row=row, error_message=str(exc))
        tracker.mark_complete()
        store.write_doc_progress(
            snapshot=tracker.snapshot(last_update_timestamp=utc_now_iso())
        )
        store.append_event(
            context=context,
            event_type="doc_finished",
            payload=finished_payload,
        )
        async with write_lock:
            append_jsonl(path=raw_path, payload=row.raw_payload())
            write_json(
                path=run_dir / "latest_progress.json", payload=progress_payload(row=row)
            )
        return row


async def generate_leaf(
    *,
    args: argparse.Namespace,
    record: WarmupPromptRecord,
    client: VllmClient,
    store: ArtifactStore,
    context: EventContext,
    initial_prompt_token_ids: tuple[int, ...],
    tracker: DocProgressTracker,
) -> LeafRollout:
    """Run one epsilon-greedy off-policy no-branch rollout."""

    executor = BranchExecutor(
        client=client,
        cluster_client=None,
        prompt_text=record.user_prompt,
        model_name=args.model,
        cluster_model_name=None,
        decoding=decoding_config(args=args),
        branching=branching_config(args=args),
        artifact_store=store,
        requested_selectors=("embed_diverse_topk_random",),
        active_selector="embed_diverse_topk_random",
        seed=record.source_index + 1234,
        trigger_steer_enabled=True,
        env_paths=(),
        enable_request_priorities=True,
        branch_task_semaphore=None,
        allow_true_branching=True,
        close_runtime_clients_on_finish=False,
        initial_prompt_token_ids=initial_prompt_token_ids,
        on_leaf_completed=lambda leaf: score_leaf(
            leaf=leaf,
            store=store,
            context=context,
            tracker=tracker,
        ),
    )
    executor.set_event_context(
        doc_id=record.source_index,
        doc_attempt=0,
        task_name=task_name(),
        model_id=model_id(),
        selector_mode=selector_mode(),
    )
    tree = await executor.run_epsilon_greedy_rollouts_async(rollout_count=1)
    assert len(tree.leaves) == 1, "no-branch warm-up generation expects one leaf"
    return tree.leaves[0]


def decoding_config(*, args: argparse.Namespace) -> DecodingConfig:
    """Build decoding config for warm-up generation."""

    return DecodingConfig(
        temperature=args.temperature,
        initial_assistant_prefix=initial_assistant_prefix(args=args),
        top_p=args.top_p,
        max_gen_toks=args.max_gen_toks,
        max_model_len=args.max_model_len,
        top_logprobs=args.top_logprobs,
        decode_chunk_tokens=args.decode_chunk_tokens,
    )


def branching_config(*, args: argparse.Namespace) -> BranchingConfig:
    """Build no-branch verbalized off-policy config."""

    return BranchingConfig(
        branch_prob=0.0,
        max_branch_points_per_rollout=1_000_000,
        max_concurrent_branches=args.max_concurrent_docs,
        num_candidates=args.off_policy_max_candidates,
        branch_fanout=1,
        max_steer_tokens=30,
        epsilon_greedy_prob=args.epsilon_greedy_prob,
        off_policy_min_candidates=args.off_policy_min_candidates,
        off_policy_max_candidates=args.off_policy_max_candidates,
        verbalized_off_policy_enabled=True,
    )


def score_leaf(
    *,
    leaf: LeafRollout,
    store: ArtifactStore,
    context: EventContext,
    tracker: DocProgressTracker,
) -> LeafRollout:
    """Attach structure metrics and emit the leaf-scored event."""

    cleaned = clean_completion(completion=leaf.text)
    scored = replace(
        leaf,
        verification=int(cleaned.is_valid),
        task_metrics={
            "structure_valid": int(cleaned.is_valid),
            "steer_count": cleaned.steer_count,
            "exec_count": cleaned.exec_count,
        },
    )
    store.append_event(
        context=context,
        event_type="leaf_scored",
        payload=leaf_scored_event_payload(leaf=scored),
    )
    tracker.record_leaf(leaf=scored)
    store.write_doc_progress(
        snapshot=tracker.snapshot(last_update_timestamp=utc_now_iso())
    )
    return scored


def result_row(
    *, record: WarmupPromptRecord, leaf: LeafRollout, system_prompt: str
) -> WarmupResultRow:
    """Build one persisted result row."""

    cleaned = clean_completion(completion=leaf.text)
    return WarmupResultRow(
        source_index=record.source_index,
        custom_id=record.custom_id,
        ground_truth=record.ground_truth,
        prompt=record.prompt_messages(system_prompt=system_prompt),
        completion=leaf.text,
        stop_reason=leaf.stop_reason,
        length_tokens_total=leaf.length_tokens_total,
        structure_valid=cleaned.is_valid,
        cleaning_reason=cleaned.reason,
        final_answer=cleaned.final_answer,
        steer_count=cleaned.steer_count,
        exec_count=cleaned.exec_count,
    )


def failed_result_row(
    *, record: WarmupPromptRecord, system_prompt: str, cleaning_reason: str
) -> WarmupResultRow:
    """Build one persisted rejected result for a recoverable generation failure."""

    return WarmupResultRow(
        source_index=record.source_index,
        custom_id=record.custom_id,
        ground_truth=record.ground_truth,
        prompt=record.prompt_messages(system_prompt=system_prompt),
        completion="",
        stop_reason=cleaning_reason,
        length_tokens_total=0,
        structure_valid=False,
        cleaning_reason=cleaning_reason,
        final_answer="",
        steer_count=0,
        exec_count=0,
    )


def is_context_length_error(*, error: VllmRequestError) -> bool:
    """Return whether vLLM rejected only this request for context length."""

    message = str(error).lower()
    return "maximum context length" in message and "input_tokens" in message


def experiment_spec() -> ExperimentSpec:
    """Return the concrete experiment metadata for this generation job."""

    return ExperimentSpec(
        task_name=task_name(),
        model_id=model_id(),
        mode="epsilon_greedy_off_policy",
        selector="embed_diverse_topk_random",
        seed=1234,
        baseline_rollouts=1,
        trigger_steer=True,
    )


def progress_tracker(
    *, store: ArtifactStore, record: WarmupPromptRecord
) -> DocProgressTracker:
    """Create one document progress tracker."""

    return DocProgressTracker.create(
        run_id=store.run_id,
        doc_id=record.source_index,
        doc_attempt=0,
        task_name=task_name(),
        model_id=model_id(),
        selector_mode=selector_mode(),
        rollout_mode="epsilon_greedy_off_policy",
        answer_extractor=lambda response_text: clean_completion(
            completion=response_text
        ).final_answer,
    )


def prompt_payload(
    *, record: WarmupPromptRecord, args: argparse.Namespace
) -> dict[str, Any]:
    """Build graph-visible prompt payload."""

    return {
        "node_id": "node_root",
        "prompt_text": record.user_prompt,
        "system_prompt": args.system_prompt,
        "rendered_prompt": render_record_prompt(record=record, args=args),
        "initial_assistant_prefix": initial_assistant_prefix(args=args),
        "prompt_char_count": len(record.user_prompt),
        "golden_answer": record.ground_truth,
        "golden_answer_source": "ground_truth",
        "text_preview": record.ground_truth,
        "custom_id": record.custom_id,
    }


def doc_finished_payload(*, leaf: LeafRollout, row: WarmupResultRow) -> dict[str, Any]:
    """Build doc-finished event payload."""

    return {
        "status": "completed",
        "mode": "epsilon_greedy_off_policy",
        "leaf_count": 1,
        "leaf_lengths": [leaf.length_tokens_total],
        "doc_metrics": {
            "structure_valid": int(row.structure_valid),
            "steer_count": row.steer_count,
            "exec_count": row.exec_count,
        },
    }


def doc_failed_payload(*, row: WarmupResultRow, error_message: str) -> dict[str, Any]:
    """Build doc-finished metadata for one recoverable rejected generation."""

    return {
        "status": "failed",
        "mode": "epsilon_greedy_off_policy",
        "leaf_count": 0,
        "leaf_lengths": [],
        "doc_metrics": {
            "structure_valid": 0,
            "steer_count": 0,
            "exec_count": 0,
        },
        "failure": {
            "reason": row.cleaning_reason,
            "error_message": error_message,
        },
    }


def progress_payload(*, row: WarmupResultRow) -> dict[str, Any]:
    """Build latest-progress sidecar payload."""

    return {
        "last_source_index": row.source_index,
        "last_custom_id": row.custom_id,
        "last_structure_valid": row.structure_valid,
        "last_update_timestamp": utc_now_iso(),
    }


def run_context(*, run_id: str) -> EventContext:
    """Build a run-scoped event context."""

    return EventContext(
        run_id=run_id,
        doc_id=None,
        doc_attempt=None,
        task_name=task_name(),
        model_id=model_id(),
        selector_mode=selector_mode(),
    )


def doc_context(*, run_id: str, doc_id: int) -> EventContext:
    """Build a document-scoped event context."""

    return EventContext(
        run_id=run_id,
        doc_id=doc_id,
        doc_attempt=0,
        task_name=task_name(),
        model_id=model_id(),
        selector_mode=selector_mode(),
    )


def task_name() -> str:
    """Return the stable task label."""

    return "sft_warmup"


def model_id() -> str:
    """Return the stable model label."""

    return "qwen35_4b_5611097_step300"


def selector_mode() -> str:
    """Return the selector label used by the viewer."""

    return "verbalized_off_policy"


def read_completed_source_indices(*, raw_path: Path) -> set[int]:
    """Read already completed source indices from raw generation JSONL."""

    if not raw_path.exists():
        return set()
    return {int(row["source_index"]) for row in read_jsonl(path=raw_path)}


if __name__ == "__main__":
    main()
