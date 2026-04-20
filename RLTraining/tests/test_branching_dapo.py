"""Unit tests for repo-local branching DAPO helpers."""

# pyright: reportMissingImports=false

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from branching_dapo.agent_loop_manager import BranchingAgentLoopManager
from branching_dapo.advantage import compute_branch_interpolated_grpo
from branching_dapo.config_types import BranchAdvantageIndex, BranchingRolloutSettings
from branching_dapo.rollout_utils import LeafBatchRecord, PromptGroup, build_prompt_groups
from branching_dapo.reward_fn import compute_score_branching_dapo
from branching_dapo.trainer import BranchingRayPPOTrainer
from verl.protocol import DataProto


class FakeBranchExecutor:
    """Minimal async executor stub for rollout-manager tests.

    Args:
        tree: Opaque branch-tree object returned by the executor.
        metrics: Optional numeric metrics emitted by the executor.

    Returns:
        Executor stub that records rollout arguments.
    """

    def __init__(self, *, tree: object, metrics: dict[str, float] | None = None) -> None:
        self.tree = tree
        self.calls: list[dict[str, object]] = []
        self._metrics = metrics or {}

    async def run_branching_rollouts_async(
        self,
        *,
        doc_id: int,
        doc_attempt: int,
        task_name: str,
        model_id: str,
        leaf_budget: int | None = None,
    ) -> object:
        """Record rollout arguments and return the configured tree object.

        Args:
            doc_id: Prompt-group document id.
            doc_attempt: Prompt attempt index.
            task_name: Task name label.
            model_id: Model id label.
            leaf_budget: Requested runtime leaf cap.

        Returns:
            Configured opaque tree object.
        """

        self.calls.append(
            {
                "doc_id": doc_id,
                "doc_attempt": doc_attempt,
                "task_name": task_name,
                "model_id": model_id,
                "leaf_budget": leaf_budget,
            }
        )
        return self.tree

    def metrics(self) -> dict[str, float]:
        """Return executor metrics for rollout-manager aggregation tests.

        Args:
            None.

        Returns:
            Numeric executor metrics.
        """

        return self._metrics


def make_index(
    *,
    leaf_id: str,
    leaf_node_id: str,
    path_node_ids: tuple[str, ...],
) -> str:
    """Build one serialized branch index for tests.

    Args:
        leaf_id: Leaf id for the branch index.
        leaf_node_id: Final node id for the leaf.
        path_node_ids: Ordered path node ids.

    Returns:
        Serialized branch index JSON string.
    """

    return BranchAdvantageIndex(
        prompt_uid="prompt-1",
        branch_tree_id="tree-1",
        leaf_id=leaf_id,
        leaf_node_id=leaf_node_id,
        path_node_ids=path_node_ids,
        parent_branch_id=path_node_ids[-2] if len(path_node_ids) > 1 else None,
        branch_depth=max(len(path_node_ids) - 1, 0),
        selected_cluster_id="cluster-a",
        cluster_name="cluster-a",
        selector_mode="cluster_across",
        candidate_pool_key="pool-1",
    ).to_json()


def make_leaf_record(
    *,
    prompt_uid: str,
    leaf_id: str,
    leaf_node_id: str,
    path_node_ids: tuple[str, ...],
    selector_mode: str,
) -> LeafBatchRecord:
    """Build one rollout leaf record for manager tests.

    Args:
        prompt_uid: Original prompt uid.
        leaf_id: Leaf identifier.
        leaf_node_id: Final node id for the leaf.
        path_node_ids: Ordered path node ids from root to leaf.
        selector_mode: Selector mode stored in branch metadata.

    Returns:
        One realized leaf batch record.
    """

    branch_index = BranchAdvantageIndex(
        prompt_uid=prompt_uid,
        branch_tree_id="tree-1",
        leaf_id=leaf_id,
        leaf_node_id=leaf_node_id,
        path_node_ids=path_node_ids,
        parent_branch_id=path_node_ids[-2] if len(path_node_ids) > 1 else None,
        branch_depth=max(len(path_node_ids) - 1, 0),
        selected_cluster_id="cluster-a" if selector_mode == "cluster_across" else None,
        cluster_name="cluster-a" if selector_mode == "cluster_across" else None,
        selector_mode=selector_mode,
        candidate_pool_key="pool-1" if selector_mode == "cluster_across" else None,
    )
    return LeafBatchRecord(
        prompt_ids=[1, 2, 3],
        response_ids=[4, 5],
        response_logprobs=None,
        reward_scores={"branch_metadata": {"leaf_id": leaf_id}},
        branch_index=branch_index,
    )


def build_source_batch() -> DataProto:
    """Build a small source batch for ragged trainer reconstruction tests.

    Args:
        None.

    Returns:
        Source prompt batch with two prompt rows.
    """

    batch = TensorDict(
        {
            "prompts": torch.tensor([[11, 12], [21, 22]], dtype=torch.int64),
            "attention_mask": torch.tensor([[1, 1], [1, 1]], dtype=torch.int64),
            "position_ids": torch.tensor([[0, 1], [0, 1]], dtype=torch.int64),
        },
        batch_size=[2],
    )
    return DataProto(
        batch=batch,
        non_tensor_batch={
            "uid": np.array(["u1", "u2"], dtype=object),
            "raw_prompt": np.array(
                [
                    [{"role": "user", "content": "prompt-1"}],
                    [{"role": "user", "content": "prompt-2"}],
                ],
                dtype=object,
            ),
            "data_source": np.array(["math", "math"], dtype=object),
            "extra_info": np.array(
                [
                    {"source_row_id": "row-1"},
                    {"source_row_id": "row-2"},
                ],
                dtype=object,
            ),
            "reward_model": np.array(
                [
                    {"ground_truth": ["4"]},
                    {"ground_truth": ["5"]},
                ],
                dtype=object,
            ),
        },
        meta_info={"source_name": "train"},
    )


def build_generation_batch() -> DataProto:
    """Build a ragged realized-leaf batch for trainer reconstruction tests.

    Args:
        None.

    Returns:
        Generation batch with three realized leaves from two source prompts.
    """

    batch = TensorDict(
        {
            "responses": torch.tensor([[31, 32], [33, 34], [41, 42]], dtype=torch.int64),
            "response_mask": torch.tensor([[1, 1], [1, 1], [1, 1]], dtype=torch.int64),
        },
        batch_size=[3],
    )
    return DataProto(
        batch=batch,
        non_tensor_batch={
            "uid": np.array(["u1", "u1", "u2"], dtype=object),
            "branch_uid": np.array(["branch-a", "branch-b", "branch-c"], dtype=object),
            "reward_scores": np.array(
                [
                    {"branch_metadata": {"leaf_id": "leaf-a"}},
                    {"branch_metadata": {"leaf_id": "leaf-b"}},
                    {"branch_metadata": {"leaf_id": "leaf-c"}},
                ],
                dtype=object,
            ),
        },
        meta_info={"timing": {"branching/generate_sequences/mean": 1.5}},
    )


def test_branching_rollout_settings_parse_entropy_threshold() -> None:
    """Rollout settings should preserve explicit entropy-trigger overrides."""

    config = SimpleNamespace(
        actor_rollout_ref=SimpleNamespace(
            rollout=SimpleNamespace(
                custom={
                    "branching_dapo": {
                        "trigger_entropy_enabled": True,
                        "entropy_threshold": 0.0,
                        "entropy_profile_name": "smoke",
                    }
                }
            )
        )
    )
    settings = BranchingRolloutSettings.from_config(config=config)
    assert settings.trigger_entropy_enabled is True
    assert settings.entropy_threshold == 0.0
    assert settings.entropy_profile_name == "smoke"


def test_branching_rollout_settings_build_artifact_paths(tmp_path: Path) -> None:
    """Rollout settings should build filesystem-safe run and batch artifact paths."""

    settings = BranchingRolloutSettings(cache_root=tmp_path)
    run_dir = settings.artifact_run_dir(run_name="full scale/run")
    batch_dir = settings.artifact_batch_dir(
        run_name="full scale/run",
        batch_name="batch 0 step 1",
    )

    assert run_dir == tmp_path / "artifacts" / "full_scale_run"
    assert batch_dir == run_dir / "batch_0_step_1"


def test_compute_score_branching_dapo_emits_branch_uid_payload() -> None:
    """Reward function should emit serialized branch metadata for the estimator."""
    result = compute_score_branching_dapo(
        data_source="AceReason-Math",
        solution_str="<think>Reason carefully.</think>\n\nFinal answer: \\boxed{5}",
        ground_truth=["5"],
        extra_info={
            "source_family": "AceReason-Math",
            "source_row_id": "row-1",
            "rollout_reward_scores": {
                "branch_metadata": {
                    "prompt_uid": "prompt-1",
                    "branch_tree_id": "tree-1",
                    "leaf_id": "leaf-1",
                    "leaf_node_id": "node_a1",
                    "path_node_ids": ["node_root", "node_a", "node_a1"],
                    "parent_branch_id": "node_a",
                    "branch_depth": 2,
                    "selected_cluster_id": "cluster-a",
                    "cluster_name": "cluster-a",
                    "selector_mode": "cluster_across",
                    "candidate_pool_key": "pool-1",
                }
            },
        },
    )

    parsed_index = BranchAdvantageIndex.from_json(str(result["branch_uid"]))
    assert parsed_index.prompt_uid == "prompt-1"
    assert result["source_family"] == "AceReason-Math"
    assert result["scorer_name"] == "math_verify"
    assert result["score"] == 1.1
    assert result["acc"] is True
    assert result["structure_reward"] == 0.1
    assert result["answer_reward"] == 1.0
    assert result["format_valid"] is True
    assert result["boxed_present"] is True


def test_compute_score_branching_dapo_accepts_boxed_numeric_normalization() -> None:
    """Reward function should normalize boxed numeric answers with separators."""

    result = compute_score_branching_dapo(
        data_source="rlvr_orz_math_57k_collected",
        solution_str="<think>Check the arithmetic.</think>\n\nThe final answer is \\boxed{30000}.",
        ground_truth=["30,000"],
        extra_info={"source_family": "rlvr_orz_math_57k_collected", "rollout_reward_scores": {}},
    )

    assert result["score"] == 1.1
    assert result["acc"] is True
    assert result["structure_reward"] == 0.1
    assert result["answer_reward"] == 1.0
    assert result["pred"] == "30000"
    assert result["reward_parse_mode"] == "math"


def test_compute_score_branching_dapo_uses_string_fallback_for_text_answers() -> None:
    """Reward function should support targeted string matching for non-math answers."""

    result = compute_score_branching_dapo(
        data_source="omega-combined-no-boxed",
        solution_str="<think>Evaluate the case split.</think>\n\nAnswer: \\boxed{No}",
        ground_truth=[r"\text{No}"],
        extra_info={"source_family": "omega-combined-no-boxed", "rollout_reward_scores": {}},
    )

    assert result["score"] == 1.1
    assert result["acc"] is True
    assert result["structure_reward"] == 0.1
    assert result["answer_reward"] == 1.0
    assert result["pred"] == "No"
    assert result["reward_parse_mode"] == "string"


def test_compute_score_branching_dapo_uses_string_fallback_for_interval_answers() -> None:
    """Reward function should match interval-style answers when math parsing fails."""

    result = compute_score_branching_dapo(
        data_source="AceReason-Math",
        solution_str="<think>Reduce to the feasible interval.</think>\n\nThe answer is \\boxed{(0,1)}.",
        ground_truth=["(0,1)"],
        extra_info={"source_family": "AceReason-Math", "rollout_reward_scores": {}},
    )

    assert result["score"] == 1.1
    assert result["acc"] is True
    assert result["structure_reward"] == 0.1
    assert result["answer_reward"] == 1.0
    assert result["pred"] == "(0,1)"
    assert result["reward_parse_mode"] == "string"


def test_compute_score_branching_dapo_accepts_interleaved_steer_exec_pairs() -> None:
    """Reward function should accept steer/exec-only think blocks with boxed output."""

    solution_str = (
        "<think>\n"
        "<steer>Plan factorization</steer>\n"
        "<exec>Factor the polynomial into linear terms.</exec>\n\n"
        "<steer>Check the roots</steer>\n"
        "<exec>Verify each root satisfies the original equation.</exec>\n"
        "</think>\n\n"
        "Therefore the answer is \\boxed{5}."
    )
    result = compute_score_branching_dapo(
        data_source="AceReason-Math",
        solution_str=solution_str,
        ground_truth=["5"],
        extra_info={"source_family": "AceReason-Math", "rollout_reward_scores": {}},
    )

    assert result["score"] == 1.1
    assert result["acc"] is True
    assert result["structure_reward"] == 0.1
    assert result["answer_reward"] == 1.0
    assert result["steer_exec_present"] is True
    assert result["steer_exec_pair_count"] == 2
    assert result["structure_issues"] == []


def test_compute_score_branching_dapo_awards_structure_only_reward() -> None:
    """Reward function should award only the structure bonus for wrong boxed answers."""

    result = compute_score_branching_dapo(
        data_source="AceReason-Math",
        solution_str="<think>Reason carefully.</think>\n\nFinal answer: \\boxed{7}",
        ground_truth=["5"],
        extra_info={"source_family": "AceReason-Math", "rollout_reward_scores": {}},
    )

    assert result["score"] == 0.1
    assert result["acc"] is False
    assert result["answer_acc"] is False
    assert result["structure_reward"] == 0.1
    assert result["answer_reward"] == 0.0
    assert result["format_valid"] is True


def test_compute_score_branching_dapo_rejects_multiple_think_blocks() -> None:
    """Reward function should reject responses with more than one think block."""

    result = compute_score_branching_dapo(
        data_source="AceReason-Math",
        solution_str="<think>first</think>\n<think>second</think>\n\n\\boxed{5}",
        ground_truth=["5"],
        extra_info={"source_family": "AceReason-Math", "rollout_reward_scores": {}},
    )

    assert result["score"] == 1.0
    assert result["acc"] is True
    assert result["answer_acc"] is True
    assert result["structure_reward"] == 0.0
    assert result["answer_reward"] == 1.0
    assert result["format_valid"] is False
    assert "expected_single_complete_think_block" in cast(list[str], result["structure_issues"])


def test_compute_score_branching_dapo_returns_zero_for_wrong_and_invalid_response() -> None:
    """Reward function should return zero when structure and answer are both wrong."""

    result = compute_score_branching_dapo(
        data_source="AceReason-Math",
        solution_str="Final answer: 7",
        ground_truth=["5"],
        extra_info={"source_family": "AceReason-Math", "rollout_reward_scores": {}},
    )

    assert result["score"] == 0.0
    assert result["acc"] is False
    assert result["answer_acc"] is False
    assert result["structure_reward"] == 0.0
    assert result["answer_reward"] == 0.0
    assert result["format_valid"] is False


def test_compute_score_branching_dapo_rejects_non_whitespace_between_steer_exec_pairs() -> None:
    """Reward function should reject think text with residual prose outside steer/exec pairs."""

    solution_str = (
        "<think>\n"
        "<steer>Draft plan</steer>\n"
        "<exec>Compute the determinant.</exec>\n"
        "Residual text\n"
        "<steer>Finish check</steer>\n"
        "<exec>Verify the sign.</exec>\n"
        "</think>\n\n"
        "\\boxed{5}"
    )
    result = compute_score_branching_dapo(
        data_source="AceReason-Math",
        solution_str=solution_str,
        ground_truth=["5"],
        extra_info={"source_family": "AceReason-Math", "rollout_reward_scores": {}},
    )

    assert result["score"] == 1.0
    assert result["acc"] is True
    assert result["structure_reward"] == 0.0
    assert result["answer_reward"] == 1.0
    assert result["format_valid"] is False
    assert "non_whitespace_outside_steer_exec" in cast(list[str], result["structure_issues"])


def test_compute_score_branching_dapo_rejects_unequal_steer_exec_counts() -> None:
    """Reward function should reject think blocks with unequal steer and exec counts."""

    solution_str = (
        "<think>\n"
        "<steer>Draft plan</steer>\n"
        "<exec>Compute the determinant.</exec>\n\n"
        "<steer>Finish check</steer>\n"
        "</think>\n\n"
        "\\boxed{5}"
    )
    result = compute_score_branching_dapo(
        data_source="AceReason-Math",
        solution_str=solution_str,
        ground_truth=["5"],
        extra_info={"source_family": "AceReason-Math", "rollout_reward_scores": {}},
    )

    assert result["score"] == 1.0
    assert result["acc"] is True
    assert result["structure_reward"] == 0.0
    assert result["answer_reward"] == 1.0
    assert result["format_valid"] is False
    assert "unequal_steer_exec_block_counts" in cast(list[str], result["structure_issues"])


def test_compute_score_branching_dapo_rejects_missing_boxed_output() -> None:
    """Reward function should require a boxed answer outside the think block."""

    result = compute_score_branching_dapo(
        data_source="AceReason-Math",
        solution_str="<think>Reason carefully.</think>\n\nFinal answer: 5",
        ground_truth=["5"],
        extra_info={"source_family": "AceReason-Math", "rollout_reward_scores": {}},
    )

    assert result["score"] == 1.0
    assert result["acc"] is True
    assert result["answer_acc"] is True
    assert result["structure_reward"] == 0.0
    assert result["answer_reward"] == 1.0
    assert result["boxed_present"] is False
    assert "missing_boxed_answer" in cast(list[str], result["structure_issues"])


def test_compute_score_branching_dapo_rejects_steer_exec_outside_think() -> None:
    """Reward function should reject steer/exec tags that appear outside think text."""

    solution_str = (
        "<think>Reason carefully.</think>\n\n"
        "<steer>Plan outside think</steer>\n"
        "<exec>Do not allow this.</exec>\n\n"
        "\\boxed{5}"
    )
    result = compute_score_branching_dapo(
        data_source="AceReason-Math",
        solution_str=solution_str,
        ground_truth=["5"],
        extra_info={"source_family": "AceReason-Math", "rollout_reward_scores": {}},
    )

    assert result["score"] == 1.0
    assert result["acc"] is True
    assert result["structure_reward"] == 0.0
    assert result["answer_reward"] == 1.0
    assert result["format_valid"] is False
    assert "steer_exec_outside_think_block" in cast(list[str], result["structure_issues"])


def test_compute_branch_interpolated_grpo_uses_recursive_values() -> None:
    """Recursive intra-branch values should contribute to the interpolated advantage."""
    token_level_rewards = torch.tensor(
        [
            [0.0, 1.0],
            [0.0, 3.0],
            [0.0, 5.0],
            [0.0, 7.0],
        ]
    )
    response_mask = torch.ones_like(token_level_rewards)
    index = np.array(
        [
            make_index(leaf_id="leaf-a1", leaf_node_id="node_a1", path_node_ids=("node_root", "node_a", "node_a1")),
            make_index(leaf_id="leaf-a2", leaf_node_id="node_a2", path_node_ids=("node_root", "node_a", "node_a2")),
            make_index(leaf_id="leaf-b1", leaf_node_id="node_b1", path_node_ids=("node_root", "node_b", "node_b1")),
            make_index(leaf_id="leaf-b2", leaf_node_id="node_b2", path_node_ids=("node_root", "node_b", "node_b2")),
        ],
        dtype=object,
    )

    advantages, returns = compute_branch_interpolated_grpo(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
        config={
            "branching_alpha": 0.5,
            "branching_epsilon": 1e-6,
            "branching_intra_norm_by_std": True,
            "norm_adv_by_std_in_grpo": True,
        },
    )

    scalar_advantages = advantages[:, 0]
    assert torch.allclose(advantages, returns)
    assert scalar_advantages[0] < scalar_advantages[1]
    assert scalar_advantages[1] < scalar_advantages[3]


def test_compute_branch_interpolated_grpo_falls_back_to_zero_without_branching() -> None:
    """No-branch singleton groups should produce zero recursive/inter advantages."""
    token_level_rewards = torch.tensor([[0.0, 2.0]])
    response_mask = torch.ones_like(token_level_rewards)
    index = np.array(
        [
            BranchAdvantageIndex(
                prompt_uid="prompt-1",
                branch_tree_id="tree-1",
                leaf_id="leaf-1",
                leaf_node_id="node_root",
                path_node_ids=("node_root",),
                parent_branch_id=None,
                branch_depth=0,
                selected_cluster_id=None,
                cluster_name=None,
                selector_mode="random",
                candidate_pool_key=None,
            ).to_json()
        ],
        dtype=object,
    )

    advantages, _ = compute_branch_interpolated_grpo(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
        config={"branching_alpha": 0.5},
    )

    assert torch.allclose(advantages, torch.zeros_like(advantages))


def test_build_prompt_groups_keeps_repeated_uid_groups_together() -> None:
    """Repeated prompt rows should be grouped into fixed-size prompt groups."""
    non_tensor_batch = {
        "uid": np.array(["u1", "u1", "u2", "u2"], dtype=object),
        "raw_prompt": np.array(
            [
                [{"role": "user", "content": "p1"}],
                [{"role": "user", "content": "p1"}],
                [{"role": "user", "content": "p2"}],
                [{"role": "user", "content": "p2"}],
            ],
            dtype=object,
        ),
        "data_source": np.array(["math", "math", "math", "math"], dtype=object),
        "extra_info": np.array([{}, {}, {}, {}], dtype=object),
        "reward_model": np.array(
            [
                {"ground_truth": ["1"]},
                {"ground_truth": ["1"]},
                {"ground_truth": ["2"]},
                {"ground_truth": ["2"]},
            ],
            dtype=object,
        ),
    }

    prompt_groups = build_prompt_groups(
        non_tensor_batch=non_tensor_batch,
        expected_group_size=2,
    )

    assert [group.prompt_uid for group in prompt_groups] == ["u1", "u2"]
    assert prompt_groups[0].group_size == 2


def test_generate_prompt_group_returns_all_realized_leaves() -> None:
    """Branching manager should not cap realized leaves to the nominal repeat count."""
    manager = cast(Any, object.__new__(BranchingAgentLoopManager))
    manager.doc_counter = 0
    manager.settings = SimpleNamespace(leaf_limit=lambda: 16)

    def fake_tokenize_prompt(*, raw_prompt: list[dict[str, str]]) -> list[int]:
        _ = raw_prompt
        return [1, 2, 3]

    tree = object()
    executor = FakeBranchExecutor(
        tree=tree,
        metrics={"branching/branch_point_count": 0.0},
    )
    manager._tokenize_prompt = fake_tokenize_prompt

    def fake_build_executor(*, prompt_text: str, doc_id: int, artifact_store: object) -> FakeBranchExecutor:
        _ = prompt_text, doc_id, artifact_store
        return executor

    def fake_build_branch_records(
        *, prompt_group: PromptGroup, prompt_ids: list[int], tree: object
    ) -> list[LeafBatchRecord]:
        _ = prompt_ids, tree
        return [
            make_leaf_record(
                prompt_uid=prompt_group.prompt_uid,
                leaf_id="leaf-1",
                leaf_node_id="node_root",
                path_node_ids=("node_root",),
                selector_mode="random",
            )
        ]

    manager._build_executor = fake_build_executor
    manager._build_branch_records = fake_build_branch_records
    prompt_group = PromptGroup(
        prompt_uid="u1",
        raw_prompt=[{"role": "user", "content": "2+2?"}],
        data_source="math",
        extra_info={"source_row_id": "row-1"},
        reward_model={"ground_truth": ["4"]},
        group_size=1,
    )

    generation = asyncio.run(
        manager._generate_prompt_group(prompt_group=prompt_group, artifact_store=object())
    )

    assert len(generation.records) == 1
    assert executor.calls == [
        {
            "doc_id": 0,
            "doc_attempt": 0,
            "task_name": "branching_dapo_train",
            "model_id": "branching_dapo",
            "leaf_budget": None,
        }
    ]
    assert generation.metrics["branching/max_possible_leaf_count"] == 16.0
    assert generation.metrics["branching/realized_leaf_count"] == 1.0
    assert generation.metrics["branching/unrealized_leaf_count"] == 15.0
    assert generation.metrics["branching/realization_rate"] == 1.0 / 16.0


def test_branch_task_semaphore_rebinds_when_event_loop_changes() -> None:
    """Branching manager should recreate shared semaphores across asyncio.run calls."""
    manager = cast(Any, object.__new__(BranchingAgentLoopManager))
    manager.settings = SimpleNamespace(max_concurrent_branches=3)
    manager.branch_task_semaphore = None
    manager.branch_task_semaphore_loop_id = None

    async def get_twice() -> tuple[asyncio.Semaphore, asyncio.Semaphore]:
        first = manager._ensure_branch_task_semaphore()
        second = manager._ensure_branch_task_semaphore()
        return first, second

    first_a, first_b = asyncio.run(get_twice())
    second_a, _ = asyncio.run(get_twice())

    assert first_a is first_b
    assert second_a is not first_a
    assert manager.branch_task_semaphore is second_a


def test_restore_generation_batch_aligns_source_fields_for_ragged_generation() -> None:
    """Trainer should rebuild prompt-side fields for realized ragged leaves by uid."""
    source_batch = build_source_batch()
    generation_batch = build_generation_batch()
    source_non_tensor_batch = BranchingRayPPOTrainer._snapshot_non_tensor_batch(batch=source_batch)

    restored_batch = BranchingRayPPOTrainer._restore_generation_batch(
        source_batch=source_batch,
        source_non_tensor_batch=source_non_tensor_batch,
        generation_batch=generation_batch,
    )

    assert len(restored_batch) == 3
    assert torch.equal(
        restored_batch.batch["prompts"],
        torch.tensor([[11, 12], [11, 12], [21, 22]], dtype=torch.int64),
    )
    assert list(restored_batch.non_tensor_batch["uid"]) == ["u1", "u1", "u2"]
    assert list(restored_batch.non_tensor_batch["data_source"]) == ["math", "math", "math"]
    assert list(restored_batch.non_tensor_batch["branch_uid"]) == [
        "branch-a",
        "branch-b",
        "branch-c",
    ]
    assert restored_batch.non_tensor_batch["extra_info"][2]["source_row_id"] == "row-2"
    assert restored_batch.meta_info["source_name"] == "train"
    assert restored_batch.meta_info["timing"]["branching/generate_sequences/mean"] == 1.5


def test_pad_for_actor_dp_pads_and_unpads_ragged_batch() -> None:
    """Trainer should pad ragged batches to actor DP width and recover original size."""

    trainer = cast(BranchingRayPPOTrainer, object.__new__(BranchingRayPPOTrainer))
    trainer.actor_rollout_wg = object()
    cast(Any, trainer)._get_dp_size = lambda worker_group, role: 4
    cast(Any, trainer)._fit_debug = lambda **_kwargs: None
    metrics: dict[str, float | int | object] = {}

    padded_batch, pad_size = trainer._pad_for_actor_dp(
        batch=build_generation_batch(),
        metrics=metrics,
        stage_name="update_actor",
    )

    assert pad_size == 1
    assert len(padded_batch) == 4
    assert metrics["trainer/update_actor_pad_size"] == 1
    restored_batch = BranchingRayPPOTrainer._unpad_output_batch(batch=padded_batch, pad_size=pad_size)
    assert len(restored_batch) == 3


def test_hybrid_fsdp_dispatch_info_for_fsdp2_world4() -> None:
    """Hybrid-sharded FSDP should dispatch one batch per shard group and collect once per group."""

    dispatch_info = BranchingRayPPOTrainer._hybrid_fsdp_dispatch_info(world_size=4, fsdp_size=2)

    assert dispatch_info == ([0, 0, 1, 1], [True, False, True, False])


def test_fit_debug_uses_init_step_label_before_global_steps_exist(capsys: pytest.CaptureFixture[str]) -> None:
    """Debug markers should still print during init before global step state exists."""

    trainer = cast(BranchingRayPPOTrainer, object.__new__(BranchingRayPPOTrainer))
    trainer.config = SimpleNamespace(trainer=SimpleNamespace(experiment_name="debug_fit_test"))

    trainer._fit_debug(message="dispatch_override")

    captured = capsys.readouterr()
    assert captured.out.strip() == "[fit-debug step=init] dispatch_override"
