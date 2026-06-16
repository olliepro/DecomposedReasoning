"""NoveltyBench generation runner backed by the branching engine."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from branching_eval.artifact_store import ArtifactStore, build_run_manifest_payload
from branching_eval.branch_executor import BranchExecutor
from branching_eval.cli import parse_doc_ids
from branching_eval.config_types import (
    BranchingEvalConfig,
    ExperimentSpec,
    ModelSpec,
)
from branching_eval.novelty_bench_types import (
    NoveltyBenchConfig,
    NoveltyBenchPrompt,
    NoveltyPromptResult,
    clean_generation_text,
    is_clean_generation_complete,
    load_config_bundle,
    load_novelty_bench_prompts,
    read_completed_prompt_ids,
)
from branching_eval.run_matrix import (
    build_config_snapshot,
    build_clustering_runtime_client,
    close_runtime_clients,
    default_env_paths,
    expand_experiments,
    read_git_commit_hash,
    runtime_branching_for_spec,
    seed_for_doc,
    selector_mode_for_spec,
    selector_modes_for_executor,
)
from branching_eval.selector_types import SelectorMode
from branching_eval.tree_types import LeafRollout
from branching_eval.vllm_runtime import managed_vllm_server
from io_utils import append_jsonl, write_json
from vllm_client import VllmClient, VllmRequestError


@dataclass(frozen=True)
class CleanGenerationCandidate:
    """One usable generation collected from a prompt attempt.

    Args:
        attempt_index: Zero-based prompt attempt that produced the generation.
        leaf_index: Leaf index within the attempt.
        raw_generation: Raw model output before cleaning.
        generation: Cleaned final answer.

    Returns:
        Candidate that can be mixed with later retries for one valid row.
    """

    attempt_index: int
    leaf_index: int
    raw_generation: str
    generation: str


class IncompleteNoveltyGenerationError(RuntimeError):
    """Raised when a NoveltyBench prompt produces unusable final answers.

    Args:
        message: Explanation of which generation indices were incomplete.
        complete_candidates: Usable generations from the failed attempt.
        incomplete_indices: Leaf indices rejected as incomplete.

    Returns:
        Retryable generation-quality exception.
    """

    def __init__(
        self,
        message: str,
        *,
        complete_candidates: tuple[CleanGenerationCandidate, ...] = (),
        incomplete_indices: tuple[int, ...] = (),
    ) -> None:
        super().__init__(message)
        self.complete_candidates = complete_candidates
        self.incomplete_indices = incomplete_indices


def parse_args() -> argparse.Namespace:
    """Parse NoveltyBench generation CLI arguments.

    Args:
        None.

    Returns:
        Parsed argument namespace.
    """

    parser = argparse.ArgumentParser(
        description="Generate NoveltyBench outputs with branching_eval runtime."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--doc-id", type=int, action="append", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument(
        "--selector",
        type=str,
        choices=(
            "cluster_across",
            "embed_diverse_topk_random",
            "within_cluster",
            "random",
        ),
        default=None,
    )
    parser.add_argument(
        "--dataset-split",
        type=str,
        choices=("curated", "wildchat"),
        default=None,
    )
    parser.add_argument("--num-generations", type=int, default=None)
    parser.add_argument(
        "--output-text-mode",
        choices=("raw", "after_think", "strip_internal_tags"),
        default=None,
    )
    return parser.parse_args()


def run_novelty_bench_matrix(
    *,
    config: BranchingEvalConfig,
    novelty_config: NoveltyBenchConfig,
    limit: int | None,
    doc_ids: tuple[int, ...] | None,
    seed_override: int | None,
    selector_override: SelectorMode | None,
    model_override: str | None,
) -> list[Path]:
    """Run the selected model/mode matrix and write NoveltyBench artifacts.

    Args:
        config: Branching-eval runtime configuration.
        novelty_config: NoveltyBench generation settings.
        limit: Optional prompt limit.
        doc_ids: Optional explicit prompt ids by sequential doc index.
        seed_override: Optional seed override for experiment expansion.
        selector_override: Optional selector override for branching runs.
        model_override: Optional model id override.

    Returns:
        Run directories containing `generations.jsonl`.
    """

    prompts = load_novelty_bench_prompts(
        novelty_config=novelty_config,
        limit=limit,
        doc_ids=doc_ids,
    )
    experiments = expand_experiments(
        config=config,
        seed_override=seed_override,
        selector_override=selector_override,
        model_override=model_override,
    )
    assert experiments, "No experiments selected for the provided overrides"
    run_dirs: list[Path] = []
    for model_index, model_spec in enumerate(config.models):
        run_dirs.extend(
            run_model_experiments(
                config=config,
                novelty_config=novelty_config,
                prompts=prompts,
                experiments=experiments,
                model_spec=model_spec,
                model_index=model_index,
            )
        )
    return run_dirs


def run_model_experiments(
    *,
    config: BranchingEvalConfig,
    novelty_config: NoveltyBenchConfig,
    prompts: list[NoveltyBenchPrompt],
    experiments: list[ExperimentSpec],
    model_spec: ModelSpec,
    model_index: int,
) -> list[Path]:
    """Run all selected experiments for one model server.

    Args:
        config: Branching-eval runtime config.
        novelty_config: NoveltyBench generation config.
        prompts: Loaded prompt records.
        experiments: Expanded experiment matrix.
        model_spec: Model spec for this server.
        model_index: Position used for port allocation.

    Returns:
        Run directories created for this model.
    """

    model_experiments = [
        spec for spec in experiments if spec.model_id == model_spec.model_id
    ]
    if not model_experiments:
        return []
    port = config.serve.base_port + model_index
    serve_log_dir = config.artifacts.output_root / "serve_logs"
    run_dirs: list[Path] = []
    with managed_vllm_server(
        model_spec=model_spec,
        serve_config=config.serve,
        port=port,
        log_dir=serve_log_dir,
    ) as running_server:
        client = VllmClient(
            base_url=running_server.base_url,
            timeout_seconds=config.serve.request_timeout_seconds,
        )
        cluster_client, cluster_model_name = build_clustering_runtime_client(
            model_spec=model_spec,
            serve_config=config.serve,
        )
        try:
            for spec in model_experiments:
                run_dirs.append(
                    run_one_novelty_experiment(
                        config=config,
                        novelty_config=novelty_config,
                        model_spec=model_spec,
                        spec=spec,
                        client=client,
                        cluster_client=cluster_client,
                        model_name_for_generation=(
                            running_server.model_name_for_generation
                        ),
                        cluster_model_name_for_generation=cluster_model_name,
                        prompts=prompts,
                    )
                )
        finally:
            close_runtime_clients(client=client, cluster_client=cluster_client)
    return run_dirs


def run_one_novelty_experiment(
    *,
    config: BranchingEvalConfig,
    novelty_config: NoveltyBenchConfig,
    model_spec: ModelSpec,
    spec: ExperimentSpec,
    client: VllmClient,
    cluster_client: VllmClient | None,
    model_name_for_generation: str,
    cluster_model_name_for_generation: str | None,
    prompts: list[NoveltyBenchPrompt],
) -> Path:
    """Run one concrete NoveltyBench experiment.

    Args:
        config: Branching-eval config.
        novelty_config: NoveltyBench config.
        model_spec: Model being evaluated.
        spec: Concrete matrix experiment.
        client: vLLM client.
        cluster_client: Optional clustering vLLM client.
        model_name_for_generation: Request-time generation model name.
        cluster_model_name_for_generation: Optional clustering model name.
        prompts: Prompt records to generate.

    Returns:
        Run directory with official and sidecar artifacts.
    """

    run_dir = build_novelty_run_dir(
        output_root=config.artifacts.output_root,
        novelty_config=novelty_config,
        spec=spec,
    )
    store = ArtifactStore(run_dir=run_dir, run_id=run_dir.name)
    try:
        initialize_novelty_run(
            store=store,
            config=config,
            novelty_config=novelty_config,
            model_spec=model_spec,
            spec=spec,
        )
        asyncio.run(
            run_prompt_set_async(
                config=config,
                novelty_config=novelty_config,
                spec=spec,
                client=client,
                cluster_client=cluster_client,
                model_name_for_generation=model_name_for_generation,
                cluster_model_name_for_generation=cluster_model_name_for_generation,
                prompts=prompts,
                store=store,
            )
        )
    finally:
        store.close()
    return run_dir


async def run_prompt_set_async(
    *,
    config: BranchingEvalConfig,
    novelty_config: NoveltyBenchConfig,
    spec: ExperimentSpec,
    client: VllmClient,
    cluster_client: VllmClient | None,
    model_name_for_generation: str,
    cluster_model_name_for_generation: str | None,
    prompts: list[NoveltyBenchPrompt],
    store: ArtifactStore,
) -> None:
    """Generate all prompts for one run with bounded concurrency.

    Args:
        config: Branching-eval config.
        novelty_config: NoveltyBench config.
        spec: Concrete experiment spec.
        client: vLLM client.
        cluster_client: Optional clustering client.
        model_name_for_generation: Request-time generation model.
        cluster_model_name_for_generation: Optional clustering model.
        prompts: Prompt records to execute.
        store: Artifact store for this run.

    Returns:
        None.
    """

    generations_path = store.run_dir / "generations.jsonl"
    metadata_path = store.run_dir / "generation_metadata.jsonl"
    completed_ids = read_completed_prompt_ids(
        generations_path=generations_path,
        num_generations=novelty_config.num_generations,
    )
    scheduled_prompts = [
        prompt for prompt in prompts if prompt.benchmark_id not in completed_ids
    ]
    await run_scheduled_prompts_async(
        config=config,
        novelty_config=novelty_config,
        spec=spec,
        client=client,
        cluster_client=cluster_client,
        model_name_for_generation=model_name_for_generation,
        cluster_model_name_for_generation=cluster_model_name_for_generation,
        prompts=scheduled_prompts,
        store=store,
        generations_path=generations_path,
        metadata_path=metadata_path,
    )
    write_json(
        path=store.run_dir / "summary.json",
        payload={
            "prompt_count_total": len(prompts),
            "prompt_count_previously_completed": len(completed_ids),
            "prompt_count_scheduled": len(scheduled_prompts),
            "num_generations": novelty_config.num_generations,
            "generations_path": str(generations_path),
        },
    )


async def run_scheduled_prompts_async(
    *,
    config: BranchingEvalConfig,
    novelty_config: NoveltyBenchConfig,
    spec: ExperimentSpec,
    client: VllmClient,
    cluster_client: VllmClient | None,
    model_name_for_generation: str,
    cluster_model_name_for_generation: str | None,
    prompts: list[NoveltyBenchPrompt],
    store: ArtifactStore,
    generations_path: Path,
    metadata_path: Path,
) -> None:
    """Run scheduled prompts and persist rows as each prompt completes.

    Args:
        config: Branching-eval config.
        novelty_config: NoveltyBench config.
        spec: Concrete experiment spec.
        client: vLLM client.
        cluster_client: Optional clustering client.
        model_name_for_generation: Request-time generation model.
        cluster_model_name_for_generation: Optional clustering model.
        prompts: Prompt records to execute.
        store: Artifact store.
        generations_path: Official output path.
        metadata_path: Metadata output path.

    Returns:
        None.
    """

    semaphore = asyncio.Semaphore(novelty_config.max_concurrent_prompts)
    branch_task_semaphore = asyncio.Semaphore(config.branching.max_concurrent_branches)
    write_lock = asyncio.Lock()
    tasks = [
        asyncio.create_task(
            run_prompt_with_semaphore(
                semaphore=semaphore,
                write_lock=write_lock,
                config=config,
                novelty_config=novelty_config,
                spec=spec,
                client=client,
                cluster_client=cluster_client,
                model_name_for_generation=model_name_for_generation,
                cluster_model_name_for_generation=cluster_model_name_for_generation,
                prompt=prompt,
                store=store,
                branch_task_semaphore=branch_task_semaphore,
                generations_path=generations_path,
                metadata_path=metadata_path,
            )
        )
        for prompt in prompts
    ]
    await asyncio.gather(*tasks)


async def run_prompt_with_semaphore(
    *,
    semaphore: asyncio.Semaphore,
    write_lock: asyncio.Lock,
    config: BranchingEvalConfig,
    novelty_config: NoveltyBenchConfig,
    spec: ExperimentSpec,
    client: VllmClient,
    cluster_client: VllmClient | None,
    model_name_for_generation: str,
    cluster_model_name_for_generation: str | None,
    prompt: NoveltyBenchPrompt,
    store: ArtifactStore,
    branch_task_semaphore: asyncio.Semaphore,
    generations_path: Path,
    metadata_path: Path,
) -> None:
    """Run one prompt and append official/metadata JSONL rows.

    Args:
        semaphore: Prompt concurrency limiter.
        write_lock: JSONL append lock.
        config: Branching-eval config.
        novelty_config: NoveltyBench config.
        spec: Concrete experiment spec.
        client: vLLM client.
        cluster_client: Optional clustering client.
        model_name_for_generation: Request-time generation model.
        cluster_model_name_for_generation: Optional clustering model.
        prompt: Prompt to execute.
        store: Artifact store for event logging.
        branch_task_semaphore: Shared branch task limiter.
        generations_path: Official output path.
        metadata_path: Sidecar metadata path.

    Returns:
        None.
    """

    async with semaphore:
        result = await run_one_prompt_with_retries_async(
            config=config,
            novelty_config=novelty_config,
            spec=spec,
            client=client,
            cluster_client=cluster_client,
            model_name_for_generation=model_name_for_generation,
            cluster_model_name_for_generation=cluster_model_name_for_generation,
            prompt=prompt,
            store=store,
            branch_task_semaphore=branch_task_semaphore,
        )
    async with write_lock:
        append_jsonl(
            path=generations_path,
            payload=result.official_row(model_name=spec.model_id),
        )
        append_jsonl(path=metadata_path, payload=result.metadata_row())


async def run_one_prompt_with_retries_async(
    *,
    config: BranchingEvalConfig,
    novelty_config: NoveltyBenchConfig,
    spec: ExperimentSpec,
    client: VllmClient,
    cluster_client: VllmClient | None,
    model_name_for_generation: str,
    cluster_model_name_for_generation: str | None,
    prompt: NoveltyBenchPrompt,
    store: ArtifactStore,
    branch_task_semaphore: asyncio.Semaphore,
) -> NoveltyPromptResult:
    """Run one prompt, retrying whole-prompt vLLM request failures.

    Args:
        config: Branching-eval config.
        novelty_config: NoveltyBench retry and output settings.
        spec: Concrete experiment spec.
        client: vLLM client.
        cluster_client: Optional clustering client.
        model_name_for_generation: Request-time generation model.
        cluster_model_name_for_generation: Optional clustering model.
        prompt: Prompt to execute.
        store: Artifact store for retry events.
        branch_task_semaphore: Shared branch-task limiter.

    Returns:
        Completed prompt result.

    Example:
        `prompt_max_attempts=3` retries a transient server disconnect twice.
    """

    collected_candidates: list[CleanGenerationCandidate] = []
    for attempt_index in range(novelty_config.prompt_max_attempts):
        try:
            return await run_one_prompt_async(
                config=config,
                novelty_config=novelty_config,
                spec=spec,
                client=client,
                cluster_client=cluster_client,
                model_name_for_generation=model_name_for_generation,
                cluster_model_name_for_generation=cluster_model_name_for_generation,
                prompt=prompt,
                store=store,
                branch_task_semaphore=branch_task_semaphore,
                attempt_index=attempt_index,
            )
        except IncompleteNoveltyGenerationError as request_error:
            added_count = append_complete_generation_candidates(
                collected_candidates=collected_candidates,
                attempt_candidates=request_error.complete_candidates,
                target_count=novelty_config.num_generations,
            )
            store.append_event(
                context=novelty_event_context(
                    store=store,
                    spec=spec,
                    prompt=prompt,
                    attempt_index=attempt_index,
                ),
                event_type="doc_partial_generations_accepted",
                payload={
                    "attempt_index": attempt_index,
                    "added_generation_count": added_count,
                    "complete_generation_count": len(collected_candidates),
                    "target_generation_count": novelty_config.num_generations,
                    "incomplete_indices": list(request_error.incomplete_indices),
                },
            )
            if len(collected_candidates) >= novelty_config.num_generations:
                result = build_prompt_result_from_candidates(
                    prompt=prompt,
                    candidates=tuple(collected_candidates),
                    num_generations=novelty_config.num_generations,
                )
                append_aggregated_doc_finished_event(
                    store=store,
                    spec=spec,
                    prompt=prompt,
                    result=result,
                    attempt_index=attempt_index,
                    source_candidates=tuple(collected_candidates),
                )
                return result
            if attempt_index + 1 >= novelty_config.prompt_max_attempts:
                raise IncompleteNoveltyGenerationError(
                    (
                        f"{spec.mode} produced {len(collected_candidates)} complete "
                        f"generations for doc_id={prompt.doc_id}; expected "
                        f"{novelty_config.num_generations}"
                    ),
                    complete_candidates=tuple(collected_candidates),
                    incomplete_indices=request_error.incomplete_indices,
                ) from request_error
            delay_seconds = prompt_retry_delay_seconds(
                novelty_config=novelty_config,
                attempt_index=attempt_index,
            )
            append_doc_retry_event(
                store=store,
                spec=spec,
                prompt=prompt,
                attempt_index=attempt_index,
                delay_seconds=delay_seconds,
                request_error=request_error,
                collected_count=len(collected_candidates),
                target_count=novelty_config.num_generations,
            )
            await asyncio.sleep(delay_seconds)
        except VllmRequestError as request_error:
            if attempt_index + 1 >= novelty_config.prompt_max_attempts:
                raise
            delay_seconds = prompt_retry_delay_seconds(
                novelty_config=novelty_config,
                attempt_index=attempt_index,
            )
            append_doc_retry_event(
                store=store,
                spec=spec,
                prompt=prompt,
                attempt_index=attempt_index,
                delay_seconds=delay_seconds,
                request_error=request_error,
                collected_count=len(collected_candidates),
                target_count=novelty_config.num_generations,
            )
            await asyncio.sleep(delay_seconds)
    raise AssertionError("prompt retry loop exhausted unexpectedly")


def append_complete_generation_candidates(
    *,
    collected_candidates: list[CleanGenerationCandidate],
    attempt_candidates: tuple[CleanGenerationCandidate, ...],
    target_count: int,
) -> int:
    """Append usable retry candidates until the target count is reached.

    Args:
        collected_candidates: Mutable cross-attempt candidate list.
        attempt_candidates: Complete generations from one failed attempt.
        target_count: Required number of generations for the final row.

    Returns:
        Number of candidates accepted from this attempt.
    """

    remaining_count = target_count - len(collected_candidates)
    if remaining_count <= 0:
        return 0
    accepted = attempt_candidates[:remaining_count]
    collected_candidates.extend(accepted)
    return len(accepted)


def append_doc_retry_event(
    *,
    store: ArtifactStore,
    spec: ExperimentSpec,
    prompt: NoveltyBenchPrompt,
    attempt_index: int,
    delay_seconds: float,
    request_error: BaseException,
    collected_count: int,
    target_count: int,
) -> None:
    """Log a prompt retry with partial-generation progress.

    Args:
        store: Artifact store for run events.
        spec: Concrete experiment spec.
        prompt: Prompt being retried.
        attempt_index: Failed attempt index.
        delay_seconds: Backoff before the next attempt.
        request_error: Retryable exception.
        collected_count: Complete generations collected so far.
        target_count: Required final generation count.

    Returns:
        None.
    """

    store.append_event(
        context=novelty_event_context(
            store=store,
            spec=spec,
            prompt=prompt,
            attempt_index=attempt_index,
        ),
        event_type="doc_retry",
        payload={
            "attempt_index": attempt_index,
            "next_attempt_index": attempt_index + 1,
            "delay_seconds": delay_seconds,
            "error_type": type(request_error).__name__,
            "error": str(request_error),
            "complete_generation_count": collected_count,
            "target_generation_count": target_count,
        },
    )


def prompt_retry_delay_seconds(
    *, novelty_config: NoveltyBenchConfig, attempt_index: int
) -> float:
    """Return bounded backoff for whole-prompt retry attempts.

    Args:
        novelty_config: NoveltyBench retry settings.
        attempt_index: Zero-based failed attempt index.

    Returns:
        Delay in seconds before retrying the prompt.
    """

    assert attempt_index >= 0, "attempt_index must be non-negative"
    delay_seconds = novelty_config.prompt_retry_base_delay_seconds * (2**attempt_index)
    return min(delay_seconds, 30.0)


async def run_one_prompt_async(
    *,
    config: BranchingEvalConfig,
    novelty_config: NoveltyBenchConfig,
    spec: ExperimentSpec,
    client: VllmClient,
    cluster_client: VllmClient | None,
    model_name_for_generation: str,
    cluster_model_name_for_generation: str | None,
    prompt: NoveltyBenchPrompt,
    store: ArtifactStore,
    branch_task_semaphore: asyncio.Semaphore,
    attempt_index: int,
) -> NoveltyPromptResult:
    """Generate one NoveltyBench prompt for a concrete experiment.

    Args:
        config: Branching-eval config.
        novelty_config: NoveltyBench config.
        spec: Concrete experiment spec.
        client: vLLM client.
        cluster_client: Optional clustering client.
        model_name_for_generation: Request-time generation model.
        cluster_model_name_for_generation: Optional clustering model.
        prompt: Prompt to execute.
        store: Artifact store for this run.
        branch_task_semaphore: Shared branch-task limiter.
        attempt_index: Zero-based whole-prompt generation attempt.

    Returns:
        Cleaned generations and raw outputs.
    """

    context = novelty_event_context(
        store=store,
        spec=spec,
        prompt=prompt,
        attempt_index=attempt_index,
    )
    store.append_event(
        context=context,
        event_type="doc_started",
        payload={
            "benchmark_id": prompt.benchmark_id,
            "prompt_char_count": len(prompt.prompt_text),
            "num_generations": novelty_config.num_generations,
        },
    )
    executor = build_prompt_executor(
        config=config,
        novelty_config=novelty_config,
        spec=spec,
        client=client,
        cluster_client=cluster_client,
        model_name_for_generation=model_name_for_generation,
        cluster_model_name_for_generation=cluster_model_name_for_generation,
        prompt=prompt,
        store=store,
        branch_task_semaphore=branch_task_semaphore,
        attempt_index=attempt_index,
    )
    leaves = await run_executor_for_spec(
        executor=executor,
        spec=spec,
        prompt=prompt,
        novelty_config=novelty_config,
        attempt_index=attempt_index,
    )
    result = build_prompt_result(
        prompt=prompt,
        leaves=leaves,
        novelty_config=novelty_config,
        spec=spec,
        attempt_index=attempt_index,
    )
    store.append_event(
        context=context,
        event_type="doc_finished",
        payload={
            "status": "completed",
            "mode": spec.mode,
            "benchmark_id": prompt.benchmark_id,
            "generation_count": len(result.generations),
            "raw_generation_char_counts": [
                len(text) for text in result.raw_generations
            ],
            "generation_char_counts": [len(text) for text in result.generations],
        },
    )
    return result


def build_prompt_result(
    *,
    prompt: NoveltyBenchPrompt,
    leaves: list[LeafRollout],
    novelty_config: NoveltyBenchConfig,
    spec: ExperimentSpec,
    attempt_index: int = 0,
) -> NoveltyPromptResult:
    """Clean and validate leaf outputs for one NoveltyBench row.

    Args:
        prompt: Prompt metadata.
        leaves: Completed engine leaves.
        novelty_config: NoveltyBench config.
        spec: Concrete experiment spec.
        attempt_index: Zero-based prompt attempt that produced the leaves.

    Returns:
        Official and sidecar generation payload.
    """

    candidates, incomplete_indices = collect_generation_candidates(
        leaves=leaves,
        novelty_config=novelty_config,
        attempt_index=attempt_index,
    )
    if incomplete_indices:
        raise IncompleteNoveltyGenerationError(
            f"{spec.mode} produced incomplete generations for doc_id="
            f"{prompt.doc_id}: indices={incomplete_indices}",
            complete_candidates=candidates,
            incomplete_indices=incomplete_indices,
        )
    assert len(candidates) == novelty_config.num_generations, (
        f"{spec.mode} produced {len(candidates)} generations for "
        f"doc_id={prompt.doc_id}; expected {novelty_config.num_generations}. "
        "For true branching, increase branch_fanout/max_branch_points or use "
        "epsilon_greedy/structured_baseline for exact-k prompt sets."
    )
    return build_prompt_result_from_candidates(
        prompt=prompt,
        candidates=candidates,
        num_generations=novelty_config.num_generations,
    )


def collect_generation_candidates(
    *,
    leaves: list[LeafRollout],
    novelty_config: NoveltyBenchConfig,
    attempt_index: int,
) -> tuple[tuple[CleanGenerationCandidate, ...], tuple[int, ...]]:
    """Return complete candidates and incomplete leaf indices.

    Args:
        leaves: Completed engine leaves.
        novelty_config: NoveltyBench config with cleaning mode.
        attempt_index: Prompt attempt that produced the leaves.

    Returns:
        Tuple of usable candidates and rejected leaf indices.
    """

    candidates: list[CleanGenerationCandidate] = []
    incomplete_indices: list[int] = []
    for leaf_index, leaf in enumerate(leaves):
        generation = clean_generation_text(
            text=leaf.text,
            mode=novelty_config.output_text_mode,
        )
        if not is_clean_generation_complete(generation=generation):
            incomplete_indices.append(leaf_index)
            continue
        candidates.append(
            CleanGenerationCandidate(
                attempt_index=attempt_index,
                leaf_index=leaf_index,
                raw_generation=leaf.text,
                generation=generation,
            )
        )
    return tuple(candidates), tuple(incomplete_indices)


def build_prompt_result_from_candidates(
    *,
    prompt: NoveltyBenchPrompt,
    candidates: tuple[CleanGenerationCandidate, ...],
    num_generations: int,
) -> NoveltyPromptResult:
    """Build a final prompt result from complete candidates.

    Args:
        prompt: Prompt metadata.
        candidates: Complete generation candidates in output order.
        num_generations: Required number of generations.

    Returns:
        Official and metadata prompt result.
    """

    selected = candidates[:num_generations]
    assert len(selected) == num_generations, "not enough candidates for prompt result"
    return NoveltyPromptResult(
        prompt=prompt,
        generations=tuple(candidate.generation for candidate in selected),
        raw_generations=tuple(candidate.raw_generation for candidate in selected),
    )


def append_aggregated_doc_finished_event(
    *,
    store: ArtifactStore,
    spec: ExperimentSpec,
    prompt: NoveltyBenchPrompt,
    result: NoveltyPromptResult,
    attempt_index: int,
    source_candidates: tuple[CleanGenerationCandidate, ...],
) -> None:
    """Log completion of a prompt assembled from retry attempts.

    Args:
        store: Artifact store for run events.
        spec: Concrete experiment spec.
        prompt: Prompt metadata.
        result: Aggregated prompt result.
        attempt_index: Attempt index that completed the aggregate.
        source_candidates: Candidates used to assemble the result.

    Returns:
        None.
    """

    store.append_event(
        context=novelty_event_context(
            store=store,
            spec=spec,
            prompt=prompt,
            attempt_index=attempt_index,
        ),
        event_type="doc_finished",
        payload={
            "status": "completed_from_partial_retries",
            "mode": spec.mode,
            "benchmark_id": prompt.benchmark_id,
            "generation_count": len(result.generations),
            "source_attempt_indices": [
                candidate.attempt_index for candidate in source_candidates
            ],
            "source_leaf_indices": [
                candidate.leaf_index for candidate in source_candidates
            ],
            "raw_generation_char_counts": [
                len(text) for text in result.raw_generations
            ],
            "generation_char_counts": [len(text) for text in result.generations],
        },
    )


def build_prompt_executor(
    *,
    config: BranchingEvalConfig,
    novelty_config: NoveltyBenchConfig,
    spec: ExperimentSpec,
    client: VllmClient,
    cluster_client: VllmClient | None,
    model_name_for_generation: str,
    cluster_model_name_for_generation: str | None,
    prompt: NoveltyBenchPrompt,
    store: ArtifactStore,
    branch_task_semaphore: asyncio.Semaphore,
    attempt_index: int,
) -> BranchExecutor:
    """Construct a branch executor for one NoveltyBench prompt.

    Args:
        config: Branching-eval config.
        novelty_config: NoveltyBench config.
        spec: Concrete experiment spec.
        client: vLLM client.
        cluster_client: Optional clustering client.
        model_name_for_generation: Request-time generation model.
        cluster_model_name_for_generation: Optional clustering model.
        prompt: Prompt to execute.
        store: Artifact store.
        branch_task_semaphore: Shared branch-task limiter.
        attempt_index: Zero-based whole-prompt generation attempt.

    Returns:
        Configured `BranchExecutor`.
    """

    _ = novelty_config
    active_selector: SelectorMode = spec.selector or "random"
    executor = BranchExecutor(
        client=client,
        cluster_client=cluster_client,
        prompt_text=prompt.prompt_text,
        model_name=model_name_for_generation,
        cluster_model_name=cluster_model_name_for_generation,
        decoding=config.decoding,
        branching=runtime_branching_for_spec(spec=spec, branching=config.branching),
        artifact_store=store,
        requested_selectors=selector_modes_for_executor(
            spec=spec,
            configured_selectors=config.run_matrix.selectors,
            active_selector=active_selector,
        ),
        active_selector=active_selector,
        seed=seed_for_doc(
            base_seed=spec.seed + attempt_index * 1_000_003,
            doc_id=prompt.doc_id,
        ),
        trigger_steer_enabled=spec.trigger_steer,
        env_paths=default_env_paths(),
        enable_request_priorities=config.serve.scheduling_policy == "priority",
        branch_task_semaphore=branch_task_semaphore,
        allow_true_branching=spec.mode != "epsilon_greedy",
        close_runtime_clients_on_finish=False,
    )
    executor.set_event_context(
        doc_id=prompt.doc_id,
        doc_attempt=attempt_index,
        task_name=spec.task_name,
        model_id=spec.model_id,
        selector_mode=selector_mode_for_spec(spec=spec),
    )
    return executor


async def run_executor_for_spec(
    *,
    executor: BranchExecutor,
    spec: ExperimentSpec,
    prompt: NoveltyBenchPrompt,
    novelty_config: NoveltyBenchConfig,
    attempt_index: int,
) -> list[LeafRollout]:
    """Run the proper executor method for one matrix mode.

    Args:
        executor: Prepared branch executor.
        spec: Concrete experiment spec.
        prompt: Prompt metadata.
        novelty_config: NoveltyBench config.
        attempt_index: Zero-based whole-prompt generation attempt.

    Returns:
        Leaf rollouts to export as generations.
    """

    if spec.mode == "baseline":
        return list(
            await executor.run_standard_rollouts_async(
                rollout_count=novelty_config.num_generations
            )
        )
    if spec.mode == "structured_baseline":
        tree = await executor.run_structured_rollouts_async(
            rollout_count=novelty_config.num_generations
        )
        return list(tree.leaves)
    if spec.mode == "epsilon_greedy":
        tree = await executor.run_epsilon_greedy_rollouts_async(
            rollout_count=novelty_config.num_generations
        )
        return list(tree.leaves)
    if spec.mode == "branching":
        tree = await executor.run_branching_rollouts_async(
            doc_id=prompt.doc_id,
            doc_attempt=attempt_index,
            task_name=spec.task_name,
            model_id=spec.model_id,
            leaf_budget=novelty_config.num_generations,
        )
        return list(tree.leaves)
    raise AssertionError(f"Unsupported experiment mode: {spec.mode}")


def initialize_novelty_run(
    *,
    store: ArtifactStore,
    config: BranchingEvalConfig,
    novelty_config: NoveltyBenchConfig,
    model_spec: ModelSpec,
    spec: ExperimentSpec,
) -> None:
    """Write run-level config and manifest artifacts.

    Args:
        store: Artifact store for this run.
        config: Branching-eval config.
        novelty_config: NoveltyBench config.
        model_spec: Model specification.
        spec: Concrete experiment spec.

    Returns:
        None.
    """

    config_snapshot = build_config_snapshot(
        config=config,
        spec=spec,
        model_spec=model_spec,
    )
    config_snapshot["novelty_bench"] = asdict(novelty_config)
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


def build_novelty_run_dir(
    *,
    output_root: Path,
    novelty_config: NoveltyBenchConfig,
    spec: ExperimentSpec,
) -> Path:
    """Build one NoveltyBench run directory path.

    Args:
        output_root: Root artifact directory.
        novelty_config: NoveltyBench settings.
        spec: Concrete experiment spec.

    Returns:
        Absolute run directory path.
    """

    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    selector_label = selector_mode_for_spec(spec=spec)
    name = (
        f"novelty_{novelty_config.dataset_split}_{spec.model_id}_{spec.mode}"
        f"_{selector_label}_seed{spec.seed}_{timestamp}"
    )
    return (output_root / name).resolve()


def novelty_event_context(
    *,
    store: ArtifactStore,
    spec: ExperimentSpec,
    prompt: NoveltyBenchPrompt,
    attempt_index: int,
) -> Any:
    """Build event context for one NoveltyBench prompt.

    Args:
        store: Artifact store.
        spec: Concrete experiment spec.
        prompt: Prompt metadata.
        attempt_index: Zero-based whole-prompt generation attempt.

    Returns:
        Event context dataclass.
    """

    from branching_eval.event_types import EventContext

    return EventContext(
        run_id=store.run_id,
        doc_id=prompt.doc_id,
        doc_attempt=attempt_index,
        task_name=spec.task_name,
        model_id=spec.model_id,
        selector_mode=selector_mode_for_spec(spec=spec),
    )


def main() -> None:
    """Run the NoveltyBench generation CLI.

    Args:
        None.

    Returns:
        None.
    """

    args = parse_args()
    config, novelty_config = load_config_bundle(
        config_path=args.config,
        dataset_split_override=args.dataset_split,
        num_generations_override=args.num_generations,
        output_text_mode_override=args.output_text_mode,
    )
    selector_override: SelectorMode | None = None
    if args.selector is not None:
        selector_override = args.selector
    run_dirs = run_novelty_bench_matrix(
        config=config,
        novelty_config=novelty_config,
        limit=args.limit,
        doc_ids=parse_doc_ids(raw_doc_ids=args.doc_id),
        seed_override=args.seed,
        selector_override=selector_override,
        model_override=args.model,
    )
    for run_dir in run_dirs:
        print(str(run_dir))


if __name__ == "__main__":
    main()
