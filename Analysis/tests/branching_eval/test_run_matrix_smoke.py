"""Integration smoke test for matrix orchestration with fake runtime components."""

from __future__ import annotations

import asyncio
import json
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

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
from branching_eval.lm_eval_adapter import DocRecord
from branching_eval.run_matrix import (
    requested_selectors_for_spec,
    runtime_branching_for_spec,
    selector_modes_for_executor,
    run_experiment_matrix,
    seed_for_doc,
)
from branching_eval.selector_types import SelectionOutcome
from branching_eval.tree_types import LeafRollout
from vllm_client import GenerationChoice, ParsedToken


class FakeRuntimeClient:
    """Tiny fake vLLM client used for smoke coverage.

    Args:
        base_url: Ignored OpenAI-compatible base URL.

    Returns:
        Fake client with deterministic outputs for baseline/branching calls.
    """

    def __init__(self, *, base_url: str) -> None:
        self.base_url = base_url

    def completions(
        self,
        *,
        model: str,
        prompt: str | None,
        prompt_token_ids: tuple[int, ...] | None,
        temperature: float,
        top_p: float,
        max_tokens: int,
        n: int,
        seed: int,
        stop: tuple[str, ...] | None,
        top_logprobs: int,
        priority: int | None = None,
        repetition_penalty: float | None = None,
    ) -> tuple[GenerationChoice, ...]:
        _ = (
            model,
            prompt,
            prompt_token_ids,
            temperature,
            top_p,
            max_tokens,
            seed,
            stop,
            top_logprobs,
            priority,
            repetition_penalty,
        )
        if n > 1:
            return tuple(
                _choice(index=index, text=f"cand_{index}") for index in range(n)
            )
        return (_choice(index=0, text=" final 42"),)

    async def completions_async(
        self,
        *,
        model: str,
        prompt: str | None,
        prompt_token_ids: tuple[int, ...] | None,
        temperature: float,
        top_p: float,
        max_tokens: int,
        n: int,
        seed: int,
        stop: tuple[str, ...] | None,
        top_logprobs: int,
        priority: int | None = None,
        repetition_penalty: float | None = None,
    ) -> tuple[GenerationChoice, ...]:
        """Async completions shim matching runtime client surface."""

        return self.completions(
            model=model,
            prompt=prompt,
            prompt_token_ids=prompt_token_ids,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            n=n,
            seed=seed,
            stop=stop,
            top_logprobs=top_logprobs,
            priority=priority,
            repetition_penalty=repetition_penalty,
        )

    async def tokenize_async(
        self, *, model: str, text: str, add_special_tokens: bool
    ) -> tuple[int, ...]:
        """Async tokenize shim with deterministic ASCII ids."""

        _ = model, add_special_tokens
        return tuple(ord(char) for char in text)

    async def detokenize_async(self, *, model: str, token_ids: tuple[int, ...]) -> str:
        """Async detokenize shim inverse to `tokenize_async`."""

        _ = model
        return "".join(chr(token_id) for token_id in token_ids)


class FakeLmEvalAdapter:
    """Minimal lm_eval adapter for matrix-level smoke tests."""

    def __init__(self, *, task_name: str) -> None:
        self.task_name = task_name

    def docs(self, *, limit: int | None) -> list[DocRecord]:
        rows = [
            DocRecord(
                doc_id=0,
                doc_payload={"Answer": "42"},
                prompt_text="Solve this toy item.",
            )
        ]
        return rows if limit is None else rows[:limit]

    def score_response(
        self, *, doc: dict[str, Any], response_text: str
    ) -> dict[str, Any]:
        _ = doc
        return {"acc": float("42" in response_text)}

    def verification(self, *, doc: dict[str, Any], response_text: str) -> int:
        _ = doc
        return int("42" in response_text)

    def aggregate_doc_metrics(
        self, *, rollout_metrics: list[dict[str, Any]]
    ) -> dict[str, Any]:
        values = [float(item["acc"]) for item in rollout_metrics]
        return {"acc": mean(values) if values else 0.0}

    def aggregate_task_metrics(
        self, *, per_doc_metrics: list[dict[str, Any]]
    ) -> dict[str, Any]:
        values = [float(item.get("acc", 0.0)) for item in per_doc_metrics]
        return {"acc": mean(values) if values else 0.0}


class FakeManyDocLmEvalAdapter(FakeLmEvalAdapter):
    """Adapter variant that yields many docs for concurrency-limit tests."""

    def docs(self, *, limit: int | None) -> list[DocRecord]:
        rows = [
            DocRecord(
                doc_id=doc_id,
                doc_payload={"Answer": "42"},
                prompt_text=f"Solve toy item {doc_id}.",
            )
            for doc_id in range(12)
        ]
        return rows if limit is None else rows[:limit]


@dataclass(frozen=True)
class FakeRunningServer:
    """Context payload compatible with `run_matrix` runtime expectations."""

    base_url: str
    model_name_for_generation: str


@contextmanager
def fake_managed_vllm_server(**_: object):
    """Yield one fake running server object."""

    yield FakeRunningServer(
        base_url="http://127.0.0.1:8123/v1",
        model_name_for_generation="fake-model",
    )


def test_run_matrix_smoke_writes_expected_artifacts(
    tmp_path: Path, monkeypatch
) -> None:
    """Matrix smoke run should write manifests and diagnostics for baseline+branching."""

    config = BranchingEvalConfig(
        tasks=TaskConfig(task_names=("aime24",)),
        models=(
            ModelSpec(
                model_id="non_sft",
                checkpoint_or_repo="fake/checkpoint",
                trigger_steer_default=False,
                trigger_entropy_default=True,
            ),
        ),
        serve=ServeConfig(),
        decoding=DecodingConfig(
            temperature=0.6, top_p=0.95, max_gen_toks=4, top_logprobs=5
        ),
        branching=BranchingConfig(
            branch_prob=1.0,
            max_branch_points_per_rollout=2,
            num_candidates=4,
            branch_fanout=2,
            max_clusters=2,
            candidate_span_tokens=2,
            max_steer_tokens=2,
            entropy_threshold=-1.0,
            entropy_profile_name="aime24_default",
        ),
        artifacts=ArtifactConfig(output_root=tmp_path / "runs"),
        run_matrix=RunMatrixConfig(
            include_baselines=True,
            baseline_rollouts=2,
            include_branching=True,
            selectors=("random",),
            seed_values=(77,),
            default_limit=1,
        ),
    )
    config.validate()

    monkeypatch.setattr("branching_eval.run_matrix.LmEvalAdapter", FakeLmEvalAdapter)
    monkeypatch.setattr("branching_eval.run_matrix.VllmClient", FakeRuntimeClient)
    monkeypatch.setattr(
        "branching_eval.run_matrix.managed_vllm_server",
        fake_managed_vllm_server,
    )

    run_dirs = run_experiment_matrix(
        config=config,
        limit=1,
        seed_override=None,
        selector_override=None,
        model_override=None,
    )
    assert len(run_dirs) == 2
    for run_dir in run_dirs:
        assert (run_dir / "run_manifest.json").exists()
        assert (run_dir / "doc_diagnostics.jsonl").exists()
        assert (run_dir / "lm_eval_aggregates.json").exists()
        assert (run_dir / "variance_diagnostics.json").exists()
        assert (run_dir / "length_diagnostics.json").exists()
        payload = json.loads(
            (run_dir / "run_manifest.json").read_text(encoding="utf-8")
        )
        assert payload["task_name"] == "aime24"
        assert payload["model_id"] == "non_sft"


def test_run_matrix_smoke_writes_structured_baseline_artifacts(
    tmp_path: Path, monkeypatch
) -> None:
    """Structured baselines should write distinct manifests and diagnostics."""

    config = BranchingEvalConfig(
        tasks=TaskConfig(task_names=("aime24",)),
        models=(
            ModelSpec(
                model_id="non_sft",
                checkpoint_or_repo="fake/checkpoint",
                trigger_steer_default=True,
                trigger_entropy_default=False,
            ),
        ),
        serve=ServeConfig(),
        decoding=DecodingConfig(
            temperature=0.6, top_p=0.95, max_gen_toks=4, top_logprobs=5
        ),
        branching=BranchingConfig(
            branch_prob=0.0,
            max_branch_points_per_rollout=2,
            num_candidates=4,
            branch_fanout=2,
            max_clusters=2,
            candidate_span_tokens=2,
            max_steer_tokens=2,
            entropy_threshold=-1.0,
            entropy_profile_name="aime24_default",
        ),
        artifacts=ArtifactConfig(output_root=tmp_path / "runs"),
        run_matrix=RunMatrixConfig(
            include_baselines=False,
            include_structured_baselines=True,
            baseline_rollouts=2,
            include_branching=False,
            selectors=("random",),
            seed_values=(77,),
            default_limit=1,
        ),
    )
    config.validate()
    monkeypatch.setattr("branching_eval.run_matrix.LmEvalAdapter", FakeLmEvalAdapter)
    monkeypatch.setattr("branching_eval.run_matrix.VllmClient", FakeRuntimeClient)
    monkeypatch.setattr(
        "branching_eval.run_matrix.managed_vllm_server",
        fake_managed_vllm_server,
    )

    run_dirs = run_experiment_matrix(
        config=config,
        limit=1,
        seed_override=None,
        selector_override=None,
        model_override=None,
    )
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    assert manifest["selector_mode"] == "structured_baseline"
    assert (run_dir / "doc_diagnostics.jsonl").exists()
    events = [
        json.loads(line)
        for line in (run_dir / "tree_events.jsonl").read_text().splitlines()
    ]
    assert any(
        row["event_type"] == "run_started"
        and row["payload"]["mode"] == "structured_baseline"
        for row in events
    )
    assert sum(1 for row in events if row["event_type"] == "leaf_scored") == 2


def test_run_matrix_smoke_writes_epsilon_greedy_artifacts(
    tmp_path: Path, monkeypatch
) -> None:
    """Epsilon-greedy runs should write distinct mode and selector artifacts."""

    config = BranchingEvalConfig(
        tasks=TaskConfig(task_names=("aime24",)),
        models=(
            ModelSpec(
                model_id="non_sft",
                checkpoint_or_repo="fake/checkpoint",
                trigger_steer_default=False,
                trigger_entropy_default=True,
            ),
        ),
        serve=ServeConfig(),
        decoding=DecodingConfig(
            temperature=0.6, top_p=0.95, max_gen_toks=4, top_logprobs=5
        ),
        branching=BranchingConfig(
            branch_prob=0.0,
            epsilon_greedy_prob=1.0,
            max_branch_points_per_rollout=2,
            num_candidates=4,
            branch_fanout=4,
            max_clusters=2,
            candidate_span_tokens=2,
            max_steer_tokens=2,
            entropy_threshold=-1.0,
            entropy_profile_name="aime24_default",
        ),
        artifacts=ArtifactConfig(output_root=tmp_path / "runs"),
        run_matrix=RunMatrixConfig(
            include_baselines=False,
            baseline_rollouts=2,
            include_branching=False,
            include_epsilon_greedy=True,
            selectors=("random",),
            seed_values=(77,),
            default_limit=1,
        ),
    )
    config.validate()

    def fake_select_all_modes(**_: object) -> tuple[SelectionOutcome, ...]:
        return (
            SelectionOutcome(
                selector_mode="embed_diverse_topk_random",
                selected_candidate_ids=(0,),
            ),
        )

    async def fake_select_all_modes_async(
        **_: object,
    ) -> tuple[SelectionOutcome, ...]:
        return fake_select_all_modes()

    monkeypatch.setattr("branching_eval.run_matrix.LmEvalAdapter", FakeLmEvalAdapter)
    monkeypatch.setattr("branching_eval.run_matrix.VllmClient", FakeRuntimeClient)
    monkeypatch.setattr(
        "branching_eval.run_matrix.managed_vllm_server",
        fake_managed_vllm_server,
    )
    monkeypatch.setattr(
        "branching_eval.branch_executor.select_candidates_all_modes",
        fake_select_all_modes,
    )
    monkeypatch.setattr(
        "branching_eval.branch_executor.select_candidates_all_modes_async",
        fake_select_all_modes_async,
    )

    run_dirs = run_experiment_matrix(
        config=config,
        limit=1,
        seed_override=None,
        selector_override=None,
        model_override=None,
    )

    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    assert "epsilon_greedy" in run_dir.name
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    assert manifest["selector_mode"] == "embed_diverse_topk_random"
    events = [
        json.loads(line)
        for line in (run_dir / "tree_events.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert any(
        row["event_type"] == "run_started"
        and row["payload"]["mode"] == "epsilon_greedy"
        for row in events
    )
    assert any(
        row["event_type"] == "selector_applied"
        and row["payload"]["active_selector_mode"] == "embed_diverse_topk_random"
        for row in events
    )
    assert sum(1 for row in events if row["event_type"] == "leaf_scored") == 2


def test_requested_selectors_for_spec_includes_active_selector() -> None:
    """Requested selectors should include active selector for one-off overrides."""

    selectors = requested_selectors_for_spec(
        configured_selectors=("random",),
        active_selector="cluster_across",
    )
    assert selectors == ("random", "cluster_across")


def test_selector_modes_for_executor_uses_active_only_for_epsilon_mode() -> None:
    """Epsilon-greedy runs should request only the active selector."""

    spec = ExperimentSpec(
        task_name="aime24",
        model_id="non_sft",
        mode="epsilon_greedy",
        selector="embed_diverse_topk_random",
        seed=77,
        baseline_rollouts=2,
        trigger_steer=False,
        trigger_entropy=True,
    )

    selectors = selector_modes_for_executor(
        spec=spec,
        configured_selectors=("cluster_across", "random"),
        active_selector="embed_diverse_topk_random",
    )

    assert selectors == ("embed_diverse_topk_random",)


def test_runtime_branching_for_spec_sets_single_path_epsilon_behavior() -> None:
    """Epsilon-greedy runtime config should keep one-child trigger behavior."""

    spec = ExperimentSpec(
        task_name="aime24",
        model_id="non_sft",
        mode="epsilon_greedy",
        selector="embed_diverse_topk_random",
        seed=77,
        baseline_rollouts=2,
        trigger_steer=False,
        trigger_entropy=True,
    )
    branching = BranchingConfig(
        branch_prob=0.0, branch_fanout=4, epsilon_greedy_prob=0.2
    )

    runtime_branching = runtime_branching_for_spec(spec=spec, branching=branching)

    assert runtime_branching.branch_prob == 0.2
    assert runtime_branching.branch_fanout == 1
    assert runtime_branching.max_branch_points_per_rollout > 4


def test_branching_scores_leaves_before_rollout_finished(
    tmp_path: Path, monkeypatch
) -> None:
    """Branching runs should emit `leaf_scored` before `rollout_finished`."""

    config = BranchingEvalConfig(
        tasks=TaskConfig(task_names=("aime24",)),
        models=(
            ModelSpec(
                model_id="non_sft",
                checkpoint_or_repo="fake/checkpoint",
                trigger_steer_default=False,
                trigger_entropy_default=True,
            ),
        ),
        serve=ServeConfig(),
        decoding=DecodingConfig(
            temperature=0.6, top_p=0.95, max_gen_toks=4, top_logprobs=5
        ),
        branching=BranchingConfig(
            branch_prob=1.0,
            max_branch_points_per_rollout=2,
            num_candidates=4,
            branch_fanout=2,
            max_clusters=2,
            candidate_span_tokens=2,
            max_steer_tokens=2,
            entropy_threshold=-1.0,
            entropy_profile_name="aime24_default",
        ),
        artifacts=ArtifactConfig(output_root=tmp_path / "runs"),
        run_matrix=RunMatrixConfig(
            include_baselines=False,
            baseline_rollouts=2,
            include_branching=True,
            selectors=("random",),
            seed_values=(77,),
            default_limit=1,
        ),
    )
    config.validate()
    monkeypatch.setattr("branching_eval.run_matrix.LmEvalAdapter", FakeLmEvalAdapter)
    monkeypatch.setattr("branching_eval.run_matrix.VllmClient", FakeRuntimeClient)
    monkeypatch.setattr(
        "branching_eval.run_matrix.managed_vllm_server",
        fake_managed_vllm_server,
    )

    run_dirs = run_experiment_matrix(
        config=config,
        limit=1,
        seed_override=None,
        selector_override=None,
        model_override=None,
    )
    assert len(run_dirs) == 1
    events = [
        json.loads(line)
        for line in (run_dirs[0] / "tree_events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    leaf_scored_indices = [
        int(row["event_index"])
        for row in events
        if row.get("event_type") == "leaf_scored"
    ]
    rollout_finished_indices = [
        int(row["event_index"])
        for row in events
        if row.get("event_type") == "rollout_finished"
    ]
    assert leaf_scored_indices
    assert rollout_finished_indices
    assert max(leaf_scored_indices) < min(rollout_finished_indices)


def _choice(*, index: int, text: str) -> GenerationChoice:
    """Build one deterministic completion choice row."""

    parsed_token = ParsedToken(
        token=text,
        logprob=-0.1,
        top_entries=(("a", -0.1), ("b", -0.2)),
    )
    return GenerationChoice(
        index=index,
        text=text,
        finish_reason="length",
        stop_reason=None,
        tokens=(parsed_token,),
        prompt_token_ids=(1, 2),
        token_ids=(index + 10,),
    )


def test_run_matrix_caps_doc_async_concurrency_to_five(
    tmp_path: Path, monkeypatch
) -> None:
    """Doc-level rollout orchestration should never exceed five in-flight docs."""

    config = BranchingEvalConfig(
        tasks=TaskConfig(task_names=("aime24",)),
        models=(
            ModelSpec(
                model_id="non_sft",
                checkpoint_or_repo="fake/checkpoint",
                trigger_steer_default=False,
                trigger_entropy_default=True,
            ),
        ),
        serve=ServeConfig(),
        decoding=DecodingConfig(
            temperature=0.6, top_p=0.95, max_gen_toks=4, top_logprobs=5
        ),
        branching=BranchingConfig(
            branch_prob=1.0,
            max_branch_points_per_rollout=2,
            num_candidates=4,
            branch_fanout=2,
            max_clusters=2,
            candidate_span_tokens=2,
            max_steer_tokens=2,
            entropy_threshold=-1.0,
            entropy_profile_name="aime24_default",
        ),
        artifacts=ArtifactConfig(output_root=tmp_path / "runs"),
        run_matrix=RunMatrixConfig(
            include_baselines=False,
            baseline_rollouts=2,
            include_branching=True,
            selectors=("random",),
            seed_values=(77,),
            default_limit=12,
            max_concurrent_docs=5,
        ),
    )
    config.validate()
    concurrency = {"inflight": 0, "max_inflight": 0}

    async def fake_run_doc_rollouts_async(**kwargs):
        doc_id = int(kwargs["doc_id"])
        concurrency["inflight"] += 1
        concurrency["max_inflight"] = max(
            concurrency["max_inflight"], concurrency["inflight"]
        )
        await asyncio.sleep(0.02)
        concurrency["inflight"] -= 1
        leaf = LeafRollout(
            leaf_id=f"leaf_{doc_id}",
            node_id="node_root",
            text=" final 42",
            token_ids=(42,),
            tokens=(),
            verification=1,
            length_tokens_total=1,
            length_tokens_exec=1,
            stop_reason="length",
            task_metrics={"acc": 1.0},
        )
        return [leaf], None

    monkeypatch.setattr(
        "branching_eval.run_matrix.LmEvalAdapter", FakeManyDocLmEvalAdapter
    )
    monkeypatch.setattr("branching_eval.run_matrix.VllmClient", FakeRuntimeClient)
    monkeypatch.setattr(
        "branching_eval.run_matrix.managed_vllm_server",
        fake_managed_vllm_server,
    )
    monkeypatch.setattr(
        "branching_eval.run_matrix.run_doc_rollouts_async",
        fake_run_doc_rollouts_async,
    )

    run_dirs = run_experiment_matrix(
        config=config,
        limit=12,
        seed_override=None,
        selector_override=None,
        model_override=None,
    )
    assert len(run_dirs) == 1
    assert concurrency["max_inflight"] == 5


def test_seed_for_doc_is_deterministic_and_doc_specific() -> None:
    """Per-doc seeds should be stable and differ across doc ids."""

    seed_doc_0 = seed_for_doc(base_seed=1234, doc_id=0)
    seed_doc_1 = seed_for_doc(base_seed=1234, doc_id=1)
    seed_doc_0_repeat = seed_for_doc(base_seed=1234, doc_id=0)

    assert seed_doc_0 == seed_doc_0_repeat
    assert seed_doc_0 != seed_doc_1
