"""Tests for NoveltyBench generation adapter helpers."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pytest

from branching_eval.artifact_store import ArtifactStore
from branching_eval.config_types import (
    ArtifactConfig,
    BranchingConfig,
    BranchingEvalConfig,
    DecodingConfig,
    ExperimentSpec,
    ModelSpec,
    RunMatrixConfig,
    ServeConfig,
    TaskConfig,
)
from branching_eval.novelty_bench import (
    CleanGenerationCandidate,
    IncompleteNoveltyGenerationError,
    build_prompt_result,
    run_novelty_bench_matrix,
    run_one_prompt_with_retries_async,
)
from branching_eval.novelty_bench_types import (
    NoveltyBenchConfig,
    NoveltyBenchPrompt,
    NoveltyPromptResult,
    clean_generation_text,
    parse_novelty_config,
)
from branching_eval.tree_types import LeafRollout
from io_utils import read_jsonl
from vllm_client import VllmRequestError


def test_clean_generation_text_after_think() -> None:
    """Generation cleaning should expose only the answer after `</think>`."""

    cleaned = clean_generation_text(
        text="<think><steer>x</steer></think>\n\nA crisp answer.",
        mode="after_think",
    )

    assert cleaned == "A crisp answer."


def test_clean_generation_text_after_empty_think() -> None:
    """Generation cleaning should not return reasoning as a final answer."""

    cleaned = clean_generation_text(text="<think>unfinished", mode="after_think")

    assert cleaned == ""


def test_clean_generation_text_after_think_without_answer() -> None:
    """Closed reasoning without final text should clean to an empty answer."""

    cleaned = clean_generation_text(text="<think>hidden</think>", mode="after_think")

    assert cleaned == ""


def test_clean_generation_text_strip_internal_tags() -> None:
    """Tag stripping should remove project-internal reasoning tags."""

    cleaned = clean_generation_text(
        text="<think><steer>plan</steer><exec>do it</exec></think>",
        mode="strip_internal_tags",
    )

    assert cleaned == "plando it"


def test_build_prompt_result_rejects_incomplete_generation() -> None:
    """Prompt result building should reject empty final answers."""

    with pytest.raises(IncompleteNoveltyGenerationError):
        build_prompt_result(
            prompt=NoveltyBenchPrompt(
                doc_id=4,
                benchmark_id="prompt-4",
                prompt_text="Name a dog.",
            ),
            leaves=[
                LeafRollout(
                    leaf_id="leaf_0",
                    node_id="node_root",
                    text="<think>still thinking",
                    token_ids=(),
                    tokens=(),
                    verification=0,
                    length_tokens_total=0,
                    length_tokens_exec=None,
                    stop_reason="length",
                    task_metrics={},
                )
            ],
            novelty_config=NoveltyBenchConfig(num_generations=1),
            spec=ExperimentSpec(
                task_name="novelty_bench",
                model_id="fake",
                mode="baseline",
                selector=None,
                seed=7,
                baseline_rollouts=1,
                trigger_steer=False,
            ),
        )


def test_parse_novelty_config_defaults() -> None:
    """Novelty config parser should preserve official default generation count."""

    config = parse_novelty_config(payload={})

    assert config.dataset_split == "curated"
    assert config.num_generations == 10
    assert config.prompt_max_attempts == 3


def test_parse_novelty_config_retry_settings() -> None:
    """Novelty config parser should accept whole-prompt retry settings."""

    config = parse_novelty_config(
        payload={
            "novelty_bench": {
                "prompt_max_attempts": 5,
                "prompt_retry_base_delay_seconds": 0.5,
            }
        }
    )

    assert config.prompt_max_attempts == 5
    assert config.prompt_retry_base_delay_seconds == 0.5


def test_run_one_prompt_retries_vllm_request_error(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Prompt-level retry should recover after exhausted request-level retries."""

    attempts = 0

    async def fake_run_one_prompt_async(**kwargs: object) -> NoveltyPromptResult:
        nonlocal attempts
        attempts += 1
        prompt = kwargs["prompt"]
        assert isinstance(prompt, NoveltyBenchPrompt)
        if attempts == 1:
            raise VllmRequestError("Server disconnected")
        return NoveltyPromptResult(
            prompt=prompt,
            generations=("answer",),
            raw_generations=("<think>x</think> answer",),
        )

    async def fake_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(
        "branching_eval.novelty_bench.run_one_prompt_async",
        fake_run_one_prompt_async,
    )
    monkeypatch.setattr("branching_eval.novelty_bench.asyncio.sleep", fake_sleep)
    store = ArtifactStore(run_dir=tmp_path / "run")
    client = ClosingFakeRuntimeClient(
        base_url="http://127.0.0.1:1/v1",
        timeout_seconds=1.0,
    )
    try:
        result = asyncio.run(
            run_one_prompt_with_retries_async(
                config=minimal_branching_config(output_root=tmp_path),
                novelty_config=NoveltyBenchConfig(
                    num_generations=1,
                    prompt_max_attempts=2,
                    prompt_retry_base_delay_seconds=0.0,
                ),
                spec=ExperimentSpec(
                    task_name="novelty_bench",
                    model_id="fake",
                    mode="baseline",
                    selector=None,
                    seed=7,
                    baseline_rollouts=1,
                    trigger_steer=False,
                ),
                client=cast(Any, client),
                cluster_client=None,
                model_name_for_generation="fake",
                cluster_model_name_for_generation=None,
                prompt=NoveltyBenchPrompt(
                    doc_id=0,
                    benchmark_id="prompt-0",
                    prompt_text="Hello",
                ),
                store=store,
                branch_task_semaphore=asyncio.Semaphore(1),
            )
        )
    finally:
        store.close()

    assert attempts == 2
    assert client.close_count == 0
    assert result.generations == ("answer",)


def test_run_one_prompt_retries_aggregates_partial_generations(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Retry handling should assemble a complete row from partial attempts."""

    attempts = 0

    async def fake_run_one_prompt_async(**kwargs: object) -> NoveltyPromptResult:
        nonlocal attempts
        prompt = kwargs["prompt"]
        assert isinstance(prompt, NoveltyBenchPrompt)
        raw_attempt_index = kwargs["attempt_index"]
        assert isinstance(raw_attempt_index, int)
        attempt_index = raw_attempt_index
        attempts += 1
        candidate = CleanGenerationCandidate(
            attempt_index=attempt_index,
            leaf_index=attempt_index,
            raw_generation=f"<think>x</think> answer {attempt_index}",
            generation=f"answer {attempt_index}",
        )
        raise IncompleteNoveltyGenerationError(
            f"baseline produced incomplete generations for doc_id={prompt.doc_id}",
            complete_candidates=(candidate,),
            incomplete_indices=(1,),
        )

    async def fake_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(
        "branching_eval.novelty_bench.run_one_prompt_async",
        fake_run_one_prompt_async,
    )
    monkeypatch.setattr("branching_eval.novelty_bench.asyncio.sleep", fake_sleep)
    store = ArtifactStore(run_dir=tmp_path / "run")
    client = ClosingFakeRuntimeClient(
        base_url="http://127.0.0.1:1/v1",
        timeout_seconds=1.0,
    )
    try:
        result = asyncio.run(
            run_one_prompt_with_retries_async(
                config=minimal_branching_config(output_root=tmp_path),
                novelty_config=NoveltyBenchConfig(
                    num_generations=2,
                    prompt_max_attempts=3,
                    prompt_retry_base_delay_seconds=0.0,
                ),
                spec=ExperimentSpec(
                    task_name="novelty_bench",
                    model_id="fake",
                    mode="baseline",
                    selector=None,
                    seed=7,
                    baseline_rollouts=2,
                    trigger_steer=False,
                ),
                client=cast(Any, client),
                cluster_client=None,
                model_name_for_generation="fake",
                cluster_model_name_for_generation=None,
                prompt=NoveltyBenchPrompt(
                    doc_id=0,
                    benchmark_id="prompt-0",
                    prompt_text="Hello",
                ),
                store=store,
                branch_task_semaphore=asyncio.Semaphore(1),
            )
        )
    finally:
        store.close()

    assert attempts == 2
    assert result.generations == ("answer 0", "answer 1")
    events = ArtifactStore(run_dir=tmp_path / "run").read_event_rows()
    assert [event["event_type"] for event in events] == [
        "doc_partial_generations_accepted",
        "doc_retry",
        "doc_partial_generations_accepted",
        "doc_finished",
    ]


def test_run_novelty_matrix_writes_generations(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Smoke run should write official-compatible `generations.jsonl`."""

    monkeypatch.setattr(
        "branching_eval.novelty_bench.load_novelty_bench_prompts",
        fake_prompt_loader,
    )
    monkeypatch.setattr(
        "branching_eval.novelty_bench.managed_vllm_server",
        fake_managed_vllm_server,
    )
    monkeypatch.setattr("branching_eval.novelty_bench.VllmClient", FakeRuntimeClient)
    monkeypatch.setattr(
        "branching_eval.novelty_bench.BranchExecutor",
        FakeBranchExecutor,
    )

    config = minimal_branching_config(output_root=tmp_path)
    novelty_config = NoveltyBenchConfig(num_generations=2)

    run_dirs = run_novelty_bench_matrix(
        config=config,
        novelty_config=novelty_config,
        limit=None,
        doc_ids=None,
        seed_override=None,
        selector_override=None,
        model_override=None,
    )

    assert len(run_dirs) == 1
    rows = read_jsonl(path=run_dirs[0] / "generations.jsonl")
    assert rows == [
        {
            "id": "prompt-0",
            "prompt": "Write two surprising product names.",
            "model": "fake",
            "generations": ["answer 0", "answer 1"],
        }
    ]


def minimal_branching_config(*, output_root: Path) -> BranchingEvalConfig:
    """Return a minimal NoveltyBench-compatible branching config."""

    return BranchingEvalConfig(
        tasks=TaskConfig(task_names=("novelty_bench",)),
        models=(ModelSpec(model_id="fake", checkpoint_or_repo="fake-model"),),
        serve=ServeConfig(tensor_parallel_size=1, kv_offloading_size_gb=0.0),
        decoding=DecodingConfig(max_gen_toks=8, top_logprobs=0),
        branching=BranchingConfig(
            num_candidates=2,
            branch_fanout=2,
            max_branch_points_per_rollout=1,
        ),
        artifacts=ArtifactConfig(output_root=output_root),
        run_matrix=RunMatrixConfig(
            include_baselines=True,
            include_branching=False,
            selectors=("random",),
            seed_values=(7,),
        ),
    )


def fake_prompt_loader(
    *,
    novelty_config: NoveltyBenchConfig,
    limit: int | None,
    doc_ids: tuple[int, ...] | None,
) -> list[NoveltyBenchPrompt]:
    """Return a tiny prompt set for smoke tests."""

    _ = novelty_config, limit, doc_ids
    return [
        NoveltyBenchPrompt(
            doc_id=0,
            benchmark_id="prompt-0",
            prompt_text="Write two surprising product names.",
        )
    ]


@dataclass(frozen=True)
class FakeRunningServer:
    """Minimal running-server payload for runtime smoke tests."""

    base_url: str
    model_name_for_generation: str


@contextmanager
def fake_managed_vllm_server(**_: object):
    """Yield one fake server object."""

    yield FakeRunningServer(
        base_url="http://127.0.0.1:8123/v1",
        model_name_for_generation="fake-model",
    )


class FakeRuntimeClient:
    """Fake vLLM client constructor used by the smoke runner."""

    def __init__(self, *, base_url: str, timeout_seconds: float | None = None) -> None:
        self.base_url = base_url
        self.timeout_seconds = timeout_seconds


class ClosingFakeRuntimeClient(FakeRuntimeClient):
    """Fake runtime client that records async close calls."""

    def __init__(self, *, base_url: str, timeout_seconds: float | None = None) -> None:
        super().__init__(base_url=base_url, timeout_seconds=timeout_seconds)
        self.close_count = 0

    async def close_async(self) -> None:
        """Record one session close request."""

        self.close_count += 1


class FakeBranchExecutor:
    """Fake branch executor returning deterministic leaves."""

    def __init__(self, **kwargs: object) -> None:
        self.prompt_text = str(kwargs["prompt_text"])

    def set_event_context(self, **_: object) -> None:
        """Accept event context setup."""

    async def run_standard_rollouts_async(
        self, *, rollout_count: int
    ) -> list[LeafRollout]:
        """Return deterministic baseline leaves."""

        await asyncio.sleep(0)
        return [
            LeafRollout(
                leaf_id=f"leaf_{index}",
                node_id="node_root",
                text=f"<think>hidden</think> answer {index}",
                token_ids=(),
                tokens=(),
                verification=0,
                length_tokens_total=0,
                length_tokens_exec=None,
                stop_reason="stop",
                task_metrics={},
            )
            for index in range(rollout_count)
        ]
