"""Unit tests for repo-local branching DAPO helpers."""

# pyright: reportMissingImports=false

from __future__ import annotations

import asyncio
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch
from tensordict import TensorDict

import branching_dapo.agent_loop_manager as agent_loop_manager_module
from branching_dapo.agent_loop_manager import BranchingAgentLoopManager
from branching_dapo.advantage import (
    compute_branch_interpolated_grpo,
    compute_branch_segment_advantages,
)
from branching_dapo.config_types import BranchAdvantageIndex, BranchingRolloutSettings
from branching_dapo.rollout_utils import (
    LeafBatchRecord,
    PromptGroup,
    build_prompt_groups,
    extract_prompt_text,
    resolve_leaf_branch_metadata,
)
from branching_dapo.trainer import BranchingRayPPOTrainer
from branching_eval.selector_types import SelectionOutcome, SelectorParams
from branching_eval.tree_types import (
    BranchPointRecord,
    BranchTree,
    LeafRollout,
    TreeEdge,
    TreeNode,
)
from branching_eval.event_db import EventDatabase
from verl.protocol import DataProto


class FakeBranchExecutor:
    """Minimal async executor stub for rollout-manager tests.

    Args:
        tree: Opaque branch-tree object returned by the executor.
        metrics: Optional numeric metrics emitted by the executor.

    Returns:
        Executor stub that records rollout arguments.
    """

    def __init__(
        self, *, tree: object, metrics: dict[str, float] | None = None
    ) -> None:
        self.tree = tree
        self.calls: list[dict[str, object]] = []
        self._metrics = metrics or {}

    def set_event_context(
        self,
        *,
        doc_id: int,
        doc_attempt: int,
        task_name: str,
        model_id: str,
        selector_mode: str,
    ) -> None:
        """Record eval-style event context setup for baseline modes."""

        self.calls.append(
            {
                "method": "set_event_context",
                "doc_id": doc_id,
                "doc_attempt": doc_attempt,
                "task_name": task_name,
                "model_id": model_id,
                "selector_mode": selector_mode,
            }
        )

    async def run_standard_rollouts_async(
        self, *, rollout_count: int
    ) -> tuple[LeafRollout, ...]:
        """Return standard leaves while recording the requested count."""

        assert isinstance(self.tree, BranchTree)
        tree = cast(BranchTree, self.tree)
        self.calls.append(
            {"method": "run_standard_rollouts_async", "rollout_count": rollout_count}
        )
        return tuple(tree.leaves)

    async def run_structured_rollouts_async(self, *, rollout_count: int) -> BranchTree:
        """Return the configured tree while recording structured rollout count."""

        assert isinstance(self.tree, BranchTree)
        self.calls.append(
            {"method": "run_structured_rollouts_async", "rollout_count": rollout_count}
        )
        return self.tree

    async def run_epsilon_greedy_rollouts_async(
        self, *, rollout_count: int
    ) -> BranchTree:
        """Return the configured tree while recording epsilon rollout count."""

        assert isinstance(self.tree, BranchTree)
        self.calls.append(
            {
                "method": "run_epsilon_greedy_rollouts_async",
                "rollout_count": rollout_count,
            }
        )
        return self.tree

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
                "method": "run_branching_rollouts_async",
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
    branch_token_offsets: tuple[int, ...] | None = None,
) -> str:
    """Build one serialized branch index for tests.

    Args:
        leaf_id: Leaf id for the branch index.
        leaf_node_id: Final node id for the leaf.
        path_node_ids: Ordered path node ids.
        branch_token_offsets: Token offsets where path branches affect the response.

    Returns:
        Serialized branch index JSON string.
    """

    return BranchAdvantageIndex(
        prompt_uid="prompt-1",
        branch_tree_id="tree-1",
        leaf_id=leaf_id,
        leaf_node_id=leaf_node_id,
        path_node_ids=path_node_ids,
        branch_token_offsets=(
            branch_token_offsets
            if branch_token_offsets is not None
            else tuple(0 for _ in path_node_ids[1:])
        ),
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
        branch_token_offsets=tuple(0 for _ in path_node_ids[1:]),
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


def make_leaf_rollout(*, leaf_id: str) -> LeafRollout:
    """Build one minimal leaf rollout for rollout-count tests."""

    return LeafRollout(
        leaf_id=leaf_id,
        node_id=f"node-{leaf_id}",
        text="\\boxed{1}",
        token_ids=(1,),
        tokens=(),
        verification=0,
        length_tokens_total=1,
        length_tokens_exec=None,
        stop_reason="done",
        task_metrics={},
    )


def make_tree_with_leaf_count(*, leaf_count: int) -> BranchTree:
    """Build a minimal branch tree with a requested leaf count."""

    return BranchTree(
        doc_id=0,
        doc_attempt=0,
        run_id="run-1",
        task_name="branching_dapo_train",
        model_id="branching_dapo",
        selector_mode="random",
        root_prompt="prompt",
        nodes={},
        edges=[],
        branch_points=[],
        leaves=[
            make_leaf_rollout(leaf_id=f"leaf-{index}") for index in range(leaf_count)
        ],
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
            "responses": torch.tensor(
                [[31, 32], [33, 34], [41, 42]], dtype=torch.int64
            ),
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


def build_row_count_batch(*, row_count: int) -> DataProto:
    """Build a minimal `DataProto` with the requested row count."""

    return DataProto(
        batch=TensorDict(
            {
                "responses": torch.arange(row_count, dtype=torch.int64).unsqueeze(1),
                "response_mask": torch.ones((row_count, 1), dtype=torch.int64),
            },
            batch_size=[row_count],
        ),
        non_tensor_batch={
            "uid": np.asarray([f"u{i}" for i in range(row_count)], dtype=object)
        },
    )


def build_trainer_for_padding(
    *,
    dp_size: int,
    ppo_mini_batch_size: int,
    rollout_n: int,
) -> BranchingRayPPOTrainer:
    """Build a trainer shell with the config fields used by padding helpers."""

    trainer = cast(BranchingRayPPOTrainer, object.__new__(BranchingRayPPOTrainer))
    trainer.actor_rollout_wg = object()
    trainer.config = SimpleNamespace(
        actor_rollout_ref=SimpleNamespace(
            actor=SimpleNamespace(ppo_mini_batch_size=ppo_mini_batch_size),
            rollout=SimpleNamespace(n=rollout_n),
        )
    )
    cast(Any, trainer)._get_dp_size = lambda worker_group, role: dp_size
    cast(Any, trainer)._fit_debug = lambda **_kwargs: None
    return trainer


def test_apply_actor_update_mask_uses_sqrt_token_ratio_gradient_scale() -> None:
    """Steer-only masking should match subset token-mean gradient variance."""

    trainer = cast(BranchingRayPPOTrainer, object.__new__(BranchingRayPPOTrainer))
    trainer.tokenizer = object()
    trainer.config = SimpleNamespace(
        actor_rollout_ref=SimpleNamespace(
            rollout=SimpleNamespace(
                custom={"branching_dapo": {"update_mode": "steer_only"}}
            )
        )
    )
    original_advantages = torch.arange(8, dtype=torch.float32).unsqueeze(0)
    batch = DataProto(
        batch=TensorDict(
            {
                "responses": torch.arange(8, dtype=torch.int64).unsqueeze(0),
                "response_mask": torch.ones((1, 8), dtype=torch.int64),
                "advantages": original_advantages.clone(),
            },
            batch_size=[1],
        ),
        non_tensor_batch={
            "reward_scores": np.asarray(
                [{"steer_phase_token_spans": [[2, 4], [6, 7]]}],
                dtype=object,
            )
        },
    )
    metrics: dict[str, float | int | object] = {}

    state = trainer._apply_actor_update_mask(batch=batch, metrics=metrics)

    assert state is not None
    assert batch.batch["response_mask"].tolist() == [[0, 0, 1, 1, 0, 0, 1, 0]]
    assert metrics["actor/update_mask/selected_tokens"] == 3.0
    assert metrics["actor/update_mask/response_tokens"] == 8.0
    expected_scale = math.sqrt(3 / 8)
    assert metrics["actor/update_mask/loss_scale"] == expected_scale
    assert metrics["actor/update_mask/gradient_scale"] == expected_scale
    assert torch.equal(batch.batch["advantages"], original_advantages * expected_scale)

    batch.batch["response_mask"] = state.response_mask
    if state.advantages is not None:
        batch.batch["advantages"] = state.advantages
    assert torch.equal(
        batch.batch["response_mask"], torch.ones((1, 8), dtype=torch.int64)
    )
    assert torch.equal(batch.batch["advantages"], original_advantages)


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


def test_branching_rollout_settings_parse_epsilon_greedy_prob() -> None:
    """Rollout settings should expose inline epsilon-greedy exploration probability."""

    config = SimpleNamespace(
        actor_rollout_ref=SimpleNamespace(
            rollout=SimpleNamespace(
                custom={"branching_dapo": {"epsilon_greedy_prob": 0.2}}
            )
        )
    )

    settings = BranchingRolloutSettings.from_config(config=config)

    assert settings.epsilon_greedy_prob == 0.2
    assert settings.selector_mode == "cluster_across"


def test_branching_rollout_settings_parse_rollout_mode() -> None:
    """Rollout settings should expose the eval-aligned rollout mode."""

    config = SimpleNamespace(
        actor_rollout_ref=SimpleNamespace(
            rollout=SimpleNamespace(
                custom={"branching_dapo": {"rollout_mode": "structured_baseline"}}
            )
        )
    )

    settings = BranchingRolloutSettings.from_config(config)

    assert settings.rollout_mode == "structured_baseline"
    assert settings.validated_rollout_mode() == "structured_baseline"
    assert settings.selector_label_for_records() == "structured_baseline"


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


def test_compute_branch_interpolated_grpo_uses_recursive_values() -> None:
    """Recursive intra-branch values should contribute to the interpolated advantage."""
    token_level_rewards = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 3.0],
            [0.0, 0.0, 0.0, 0.0, 5.0],
            [0.0, 0.0, 0.0, 0.0, 7.0],
        ]
    )
    response_mask = torch.ones_like(token_level_rewards)
    index = np.array(
        [
            make_index(
                leaf_id="leaf-a1",
                leaf_node_id="node_a1",
                path_node_ids=("node_root", "node_a", "node_a1"),
                branch_token_offsets=(1, 3),
            ),
            make_index(
                leaf_id="leaf-a2",
                leaf_node_id="node_a2",
                path_node_ids=("node_root", "node_a", "node_a2"),
                branch_token_offsets=(1, 3),
            ),
            make_index(
                leaf_id="leaf-b1",
                leaf_node_id="node_b1",
                path_node_ids=("node_root", "node_b", "node_b1"),
                branch_token_offsets=(1, 3),
            ),
            make_index(
                leaf_id="leaf-b2",
                leaf_node_id="node_b2",
                path_node_ids=("node_root", "node_b", "node_b2"),
                branch_token_offsets=(1, 3),
            ),
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

    assert torch.allclose(advantages, returns)
    assert torch.any(advantages[:, 0] != 0.0)
    assert torch.any(advantages[:, 1] != 0.0)
    assert not torch.allclose(advantages[:, 1], advantages[:, 3])
    assert advantages[0, 3] < advantages[1, 3]
    assert advantages[1, 3] < advantages[3, 3]


def test_compute_branch_interpolated_grpo_applies_inter_to_all_tokens() -> None:
    """Prompt-level GRPO advantage should apply across the full response."""

    token_level_rewards = torch.tensor(
        [
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 3.0],
        ]
    )
    index = np.array(
        [
            make_index(
                leaf_id="leaf-a",
                leaf_node_id="node_a",
                path_node_ids=("node_root", "node_mid", "node_a"),
                branch_token_offsets=(1, 3),
            ),
            make_index(
                leaf_id="leaf-b",
                leaf_node_id="node_b",
                path_node_ids=("node_root", "node_mid", "node_b"),
                branch_token_offsets=(1, 3),
            ),
        ],
        dtype=object,
    )

    advantages, _ = compute_branch_interpolated_grpo(
        token_level_rewards=token_level_rewards,
        response_mask=torch.ones_like(token_level_rewards, dtype=torch.long),
        index=index,
        config={
            "branching_alpha": 0.0,
            "branching_epsilon": 1e-6,
            "branching_intra_norm_by_std": True,
            "norm_adv_by_std_in_grpo": True,
        },
    )

    assert advantages[0].tolist() == pytest.approx([-1.0, -1.0, -1.0, -1.0])
    assert advantages[1].tolist() == pytest.approx([1.0, 1.0, 1.0, 1.0])


def test_compute_branch_interpolated_grpo_does_not_push_intra_downstream() -> None:
    """A branch-local delta should apply only until the next branch point."""

    token_level_rewards = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 3.0],
            [0.0, 0.0, 0.0, 0.0, 5.0],
            [0.0, 0.0, 0.0, 0.0, 7.0],
        ]
    )
    index = np.array(
        [
            make_index(
                leaf_id="leaf-a1",
                leaf_node_id="node_a1",
                path_node_ids=("node_root", "node_a", "node_a1"),
                branch_token_offsets=(1, 3),
            ),
            make_index(
                leaf_id="leaf-a2",
                leaf_node_id="node_a2",
                path_node_ids=("node_root", "node_a", "node_a2"),
                branch_token_offsets=(1, 3),
            ),
            make_index(
                leaf_id="leaf-b1",
                leaf_node_id="node_b1",
                path_node_ids=("node_root", "node_b", "node_b1"),
                branch_token_offsets=(1, 3),
            ),
            make_index(
                leaf_id="leaf-b2",
                leaf_node_id="node_b2",
                path_node_ids=("node_root", "node_b", "node_b2"),
                branch_token_offsets=(1, 3),
            ),
        ],
        dtype=object,
    )

    advantages, _ = compute_branch_interpolated_grpo(
        token_level_rewards=token_level_rewards,
        response_mask=torch.ones_like(token_level_rewards, dtype=torch.long),
        index=index,
        config={
            "branching_alpha": 1.0,
            "branching_epsilon": 1e-6,
            "branching_intra_norm_by_std": False,
            "norm_adv_by_std_in_grpo": False,
        },
    )

    assert advantages[0].tolist() == pytest.approx([0.0, -4.0, -4.0, -2.0, -2.0])


def test_compute_branch_segment_advantages_aggregates_shared_segments() -> None:
    """Segment labels should average final token advantages over reused spans."""

    advantages = torch.tensor(
        [
            [10.0, 1.0, 3.0, 5.0, 7.0],
            [20.0, 9.0, 11.0, 13.0, 15.0],
        ]
    )
    response_mask = torch.ones_like(advantages, dtype=torch.long)
    index = np.array(
        [
            make_index(
                leaf_id="leaf-a1",
                leaf_node_id="node_a1",
                path_node_ids=("node_root", "node_a", "node_a1"),
                branch_token_offsets=(1, 3),
            ),
            make_index(
                leaf_id="leaf-a2",
                leaf_node_id="node_a2",
                path_node_ids=("node_root", "node_a", "node_a2"),
                branch_token_offsets=(1, 3),
            ),
        ],
        dtype=object,
    )

    segments = compute_branch_segment_advantages(
        advantages=advantages,
        response_mask=response_mask,
        index=index,
    )
    rows = {(row.parent_node_id, row.child_node_id): row for row in segments}

    shared = rows[("node_root", "node_a")]
    assert shared.mean_combined_advantage == pytest.approx(6.0)
    assert shared.token_count == 4
    assert shared.leaf_count == 2
    assert shared.token_start == 1
    assert shared.token_end == 3
    assert rows[("node_a", "node_a1")].mean_combined_advantage == pytest.approx(6.0)
    assert rows[("node_a", "node_a2")].mean_combined_advantage == pytest.approx(14.0)


def test_trainer_persists_branch_segment_advantages(tmp_path: Path) -> None:
    """Trainer should write segment advantage rows into the rollout SQLite DB."""

    db_path = tmp_path / "tree_events.sqlite"
    EventDatabase(path=db_path)
    trainer = cast(BranchingRayPPOTrainer, object.__new__(BranchingRayPPOTrainer))
    branch_uid = make_index(
        leaf_id="leaf-a",
        leaf_node_id="node_a",
        path_node_ids=("node_root", "node_a"),
        branch_token_offsets=(1,),
    )
    batch = DataProto(
        batch=TensorDict(
            {
                "advantages": torch.tensor([[0.0, 2.0, 4.0]], dtype=torch.float32),
                "response_mask": torch.tensor([[1, 1, 1]], dtype=torch.int64),
            },
            batch_size=[1],
        ),
        non_tensor_batch={
            "branch_uid": np.asarray([branch_uid], dtype=object),
            "tree_events_db_path": np.asarray([str(db_path)], dtype=object),
            "reward_scores": np.asarray(
                [
                    {
                        "branch_metadata": json.loads(branch_uid),
                        "event_context": {
                            "doc_id": 0,
                            "doc_attempt": 0,
                            "task_name": "branching_dapo_train",
                            "model_id": "branching_dapo",
                            "selector_mode": "structured_baseline",
                        },
                    }
                ],
                dtype=object,
            ),
        },
    )

    trainer._persist_branch_segment_advantages(batch=batch)

    rows = EventDatabase(path=db_path).read_node_advantage_rows_for_attempt(
        doc_id=0,
        doc_attempt=0,
        task_name="branching_dapo_train",
        model_id="branching_dapo",
        selector_mode="structured_baseline",
    )
    assert len(rows) == 1
    assert rows[0]["child_node_id"] == "node_a"
    assert rows[0]["mean_combined_advantage"] == pytest.approx(3.0)
    assert rows[0]["token_count"] == 2
    assert rows[0]["leaf_count"] == 1


def test_compute_branch_interpolated_grpo_falls_back_to_zero_without_branching() -> (
    None
):
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
                branch_token_offsets=(),
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


def test_repeat_generation_batch_preserves_sixteen_sample_uid_groups() -> None:
    """Trainer repeat should give GRPO sixteen same-uid leaves per prompt."""

    repeated_batch = BranchingRayPPOTrainer._repeat_generation_batch(
        generation_batch=build_source_batch(),
        repeat_times=16,
    )
    prompt_groups = build_prompt_groups(
        non_tensor_batch=repeated_batch.non_tensor_batch,
        expected_group_size=16,
    )

    assert len(repeated_batch) == 32
    assert [group.prompt_uid for group in prompt_groups] == ["u1", "u2"]
    assert [group.group_size for group in prompt_groups] == [16, 16]


def test_rollout_manager_expected_prompt_group_size_uses_rollout_n() -> None:
    """Rollout manager grouping should follow configured rollout.n."""

    manager = cast(Any, object.__new__(BranchingAgentLoopManager))
    manager.rollout_config = SimpleNamespace(n=16)

    assert manager._expected_prompt_group_size() == 16


@pytest.mark.parametrize(
    ("rollout_mode", "expected_call"),
    [
        ("no_branching", "run_standard_rollouts_async"),
        ("structured_baseline", "run_structured_rollouts_async"),
        ("epsilon_greedy", "run_epsilon_greedy_rollouts_async"),
    ],
)
def test_non_branch_modes_request_sixteen_leaves_per_prompt(
    rollout_mode: str, expected_call: str
) -> None:
    """Non-branch comparison modes should request 16 independent leaves."""

    manager = cast(Any, object.__new__(BranchingAgentLoopManager))
    manager.settings = BranchingRolloutSettings(rollout_mode=rollout_mode)
    manager.artifact_run_scope = SimpleNamespace(run_id="run-1")
    executor = FakeBranchExecutor(tree=make_tree_with_leaf_count(leaf_count=16))

    tree = asyncio.run(
        manager._run_rollout_tree(
            executor=executor,
            prompt_text="2+2?",
            doc_id=0,
            rollout_count=16,
        )
    )

    assert len(tree.leaves) == 16
    assert {"method": expected_call, "rollout_count": 16} in executor.calls


def test_branching_mode_binary_four_points_has_sixteen_leaf_capacity() -> None:
    """Branching target shape should be 2**4 leaves per prompt."""

    manager = cast(Any, object.__new__(BranchingAgentLoopManager))
    manager.settings = BranchingRolloutSettings(
        rollout_mode="branching",
        branch_fanout=2,
        max_branch_points_per_rollout=4,
    )

    assert manager._max_possible_leaf_count(rollout_count=1) == 16


def test_generate_prompt_group_returns_all_realized_leaves() -> None:
    """Branching manager should not cap realized leaves to the nominal repeat count."""

    class FakeArtifactStore:
        """Minimal prompt-event store for prompt-group generation tests."""

        run_id = "run-1"

        def __init__(self) -> None:
            self.events: list[dict[str, object]] = []

        def append_event(self, **kwargs: object) -> None:
            self.events.append(kwargs)

    manager = cast(Any, object.__new__(BranchingAgentLoopManager))
    manager.doc_counter = 0
    manager.settings = BranchingRolloutSettings(
        selector_mode="random",
        branch_fanout=4,
        max_branch_points_per_rollout=2,
        epsilon_greedy_prob=0.2,
    )

    def fake_tokenize_prompt(*, raw_prompt: list[dict[str, str]]) -> list[int]:
        _ = raw_prompt
        return [1, 2, 3]

    tree = object()
    executor = FakeBranchExecutor(
        tree=tree,
        metrics={"branching/branch_point_count": 0.0},
    )
    manager._tokenize_prompt = fake_tokenize_prompt
    manager._initial_assistant_prefix = lambda *, prompt_ids: ""

    def fake_build_executor(
        *,
        prompt_text: str,
        prompt_token_ids: list[int] | None,
        initial_assistant_prefix: str,
        doc_id: int,
        artifact_store: object,
    ) -> FakeBranchExecutor:
        _ = (
            prompt_text,
            prompt_token_ids,
            initial_assistant_prefix,
            doc_id,
            artifact_store,
        )
        return executor

    def fake_build_branch_records(
        *,
        prompt_group: PromptGroup,
        prompt_ids: list[int],
        initial_assistant_prefix: str,
        tree: object,
    ) -> list[LeafBatchRecord]:
        _ = prompt_ids, initial_assistant_prefix, tree
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
    artifact_store = FakeArtifactStore()

    generation = asyncio.run(
        manager._generate_prompt_group(
            prompt_group=prompt_group, artifact_store=artifact_store
        )
    )

    assert len(artifact_store.events) == 1
    assert len(generation.records) == 1
    assert executor.calls == [
        {
            "method": "run_branching_rollouts_async",
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
    assert generation.metrics["branching/epsilon_greedy_prob"] == 0.2
    assert generation.metrics["branching/epsilon_greedy_enabled"] == 1.0


def test_extract_prompt_text_accepts_system_plus_one_user_message() -> None:
    """Runtime extraction should keep the problem text separate from system policy."""

    prompt_text = extract_prompt_text(
        raw_prompt=[
            {"role": "system", "content": "Answer in boxed form."},
            {"role": "user", "content": "2+2?"},
        ]
    )

    assert prompt_text == "2+2?"


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
    source_non_tensor_batch = BranchingRayPPOTrainer._snapshot_non_tensor_batch(
        batch=source_batch
    )

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
    assert list(restored_batch.non_tensor_batch["data_source"]) == [
        "math",
        "math",
        "math",
    ]
    assert list(restored_batch.non_tensor_batch["branch_uid"]) == [
        "branch-a",
        "branch-b",
        "branch-c",
    ]
    assert restored_batch.non_tensor_batch["extra_info"][2]["source_row_id"] == "row-2"
    assert restored_batch.meta_info["source_name"] == "train"
    assert (
        restored_batch.meta_info["timing"]["branching/generate_sequences/mean"] == 1.5
    )


def test_pad_for_actor_dp_pads_and_unpads_ragged_batch() -> None:
    """Trainer should pad ragged batches to the actor update batch divisor."""

    trainer = build_trainer_for_padding(
        dp_size=4,
        ppo_mini_batch_size=1,
        rollout_n=4,
    )
    metrics: dict[str, float | int | object] = {}

    padded_batch, pad_size = trainer._pad_for_actor_dp(
        batch=build_generation_batch(),
        metrics=metrics,
        stage_name="update_actor",
    )

    assert pad_size == 1
    assert len(padded_batch) == 4
    assert metrics["trainer/update_actor_pad_size"] == 1
    assert metrics["trainer/update_actor_pad_divisor"] == 4
    restored_batch = BranchingRayPPOTrainer._unpad_output_batch(
        batch=padded_batch, pad_size=pad_size
    )
    assert len(restored_batch) == 3


def test_pad_for_actor_dp_matches_verl_update_actor_minibatch() -> None:
    """Trainer should pad failed 8x32 branching leaf counts to verl's divisor."""

    trainer = build_trainer_for_padding(
        dp_size=4,
        ppo_mini_batch_size=8,
        rollout_n=32,
    )
    metrics: dict[str, float | int | object] = {}

    padded_batch, pad_size = trainer._pad_for_actor_dp(
        batch=build_row_count_batch(row_count=224),
        metrics=metrics,
        stage_name="update_actor",
    )

    assert pad_size == 32
    assert len(padded_batch) == 256
    assert metrics["trainer/update_actor_pad_divisor"] == 256
    restored_batch = BranchingRayPPOTrainer._unpad_output_batch(
        batch=padded_batch, pad_size=pad_size
    )
    assert len(restored_batch) == 224


def test_reward_component_metrics_reduce_reward_extras() -> None:
    """Trainer should expose reward components as scalar logger metrics."""

    metrics = BranchingRayPPOTrainer._reward_component_metrics(
        reward_extra_infos_dict={
            "structure_reward": np.array([0.1, 0.0, 0.1], dtype=object),
            "answer_reward": np.array([1.0, 0.0, "bad", None], dtype=object),
            "format_valid": np.array([True, False], dtype=object),
        }
    )

    assert metrics == {
        "reward/structure_reward/mean": pytest.approx(0.2 / 3.0),
        "reward/structure_reward/max": 0.1,
        "reward/structure_reward/min": 0.0,
        "reward/answer_reward/mean": 0.5,
        "reward/answer_reward/max": 1.0,
        "reward/answer_reward/min": 0.0,
    }


def test_answer_diversity_metrics_reduce_reward_extras() -> None:
    """Trainer should expose answer diversity and all-zero prompt counts."""

    branch_uids = np.array(
        [
            json.dumps({"prompt_uid": "prompt-a"}),
            json.dumps({"prompt_uid": "prompt-a"}),
            json.dumps({"prompt_uid": "prompt-b"}),
            json.dumps({"prompt_uid": "prompt-b"}),
            json.dumps({"prompt_uid": "prompt-c"}),
            json.dumps({"prompt_uid": "prompt-c"}),
        ],
        dtype=object,
    )
    metrics = BranchingRayPPOTrainer._answer_diversity_metrics(
        reward_extra_infos_dict={
            "pred": np.array(["42", " 42 ", "7\n", "", None, "7"], dtype=object),
            "answer_reward": np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=object),
            "branch_uid": branch_uids,
        }
    )

    assert metrics == {
        "answer/given_answer_count": 4.0,
        "answer/unique_given_answer_count": 2.0,
        "answer/unique_given_answer_ratio": 0.5,
        "answer/prompt_count": 3.0,
        "answer/problem_pass_rate_mean": pytest.approx(1.0 / 6.0),
        "answer/problem_pass_rate_max": 0.5,
        "answer/problem_pass_rate_min": 0.0,
        "answer/prompts_zero_answer_reward_all_rollouts_count": 2.0,
        "answer/prompts_zero_answer_reward_all_rollouts_ratio": pytest.approx(
            2.0 / 3.0
        ),
    }


def test_problem_pass_rate_metrics_average_problem_rates() -> None:
    """Problem pass rate should be prompt-weighted, not leaf-weighted."""

    branch_uids = np.array(
        [
            json.dumps({"prompt_uid": "prompt-a"}),
            json.dumps({"prompt_uid": "prompt-a"}),
            json.dumps({"prompt_uid": "prompt-a"}),
            json.dumps({"prompt_uid": "prompt-a"}),
            json.dumps({"prompt_uid": "prompt-b"}),
        ],
        dtype=object,
    )
    metrics = BranchingRayPPOTrainer._answer_diversity_metrics(
        reward_extra_infos_dict={
            "answer_reward": np.array([1.0, 0.0, 0.0, 0.0, 1.0], dtype=object),
            "branch_uid": branch_uids,
        }
    )

    assert metrics["answer/problem_pass_rate_mean"] == pytest.approx(0.625)
    assert metrics["answer/problem_pass_rate_max"] == 1.0
    assert metrics["answer/problem_pass_rate_min"] == 0.25


def test_block_structure_metrics_reduce_reward_extras() -> None:
    """Trainer should expose weighted steer/exec block metrics."""

    metrics = BranchingRayPPOTrainer._block_structure_metrics(
        reward_extra_infos_dict={
            "exec_block_count": np.array([2, 1, 0], dtype=object),
            "exec_block_word_count": np.array([10, 5, 0], dtype=object),
            "steer_block_count": np.array([2, 2, 1], dtype=object),
            "steer_block_word_count": np.array([8, 6, 4], dtype=object),
        }
    )

    assert metrics == {
        "blocks/exec/num_blocks_total": 3.0,
        "blocks/exec/num_blocks_per_leaf": 1.0,
        "blocks/exec/words_total": 15.0,
        "blocks/exec/words_per_leaf": 5.0,
        "blocks/exec/avg_words_per_block": 5.0,
        "blocks/steer/num_blocks_total": 5.0,
        "blocks/steer/num_blocks_per_leaf": pytest.approx(5.0 / 3.0),
        "blocks/steer/words_total": 18.0,
        "blocks/steer/words_per_leaf": 6.0,
        "blocks/steer/avg_words_per_block": pytest.approx(18.0 / 5.0),
    }


def test_repetition_metrics_reduce_reward_extras() -> None:
    """Trainer should expose repeat forced-close counts and ratios."""

    metrics = BranchingRayPPOTrainer._repetition_metrics(
        reward_extra_infos_dict={
            "repeat_forced_think_close": np.array([True, False, True, True]),
            "repeat_block_kind": np.array(
                ["exec", None, "steer", "steer"], dtype=object
            ),
            "repeat_block_count": np.array([3, 0, 4, 5], dtype=object),
            "repeat_last_similarity_ratio": np.array([0.91, None, 0.96, 0.98]),
        }
    )

    assert metrics == {
        "repetition/leaf_count": 4.0,
        "repetition/forced_close_count": 3.0,
        "repetition/forced_close_ratio": 0.75,
        "repetition/exec/forced_close_count": 1.0,
        "repetition/exec/forced_close_ratio": 0.25,
        "repetition/exec/repeated_blocks_mean": 3.0,
        "repetition/exec/repeated_blocks_max": 3.0,
        "repetition/exec/last_similarity_mean": 0.91,
        "repetition/exec/last_similarity_max": 0.91,
        "repetition/steer/forced_close_count": 2.0,
        "repetition/steer/forced_close_ratio": 0.5,
        "repetition/steer/repeated_blocks_mean": 4.5,
        "repetition/steer/repeated_blocks_max": 5.0,
        "repetition/steer/last_similarity_mean": 0.97,
        "repetition/steer/last_similarity_max": 0.98,
    }


def test_hybrid_fsdp_dispatch_info_for_fsdp2_world4() -> None:
    """Hybrid-sharded FSDP should dispatch one batch per shard group and collect once per group."""

    dispatch_info = BranchingRayPPOTrainer._hybrid_fsdp_dispatch_info(
        world_size=4, fsdp_size=2
    )

    assert dispatch_info == ([0, 0, 1, 1], [True, False, True, False])


def test_fit_debug_uses_init_step_label_before_global_steps_exist(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Debug markers should still print during init before global step state exists."""

    trainer = cast(BranchingRayPPOTrainer, object.__new__(BranchingRayPPOTrainer))
    trainer.config = SimpleNamespace(
        trainer=SimpleNamespace(experiment_name="debug_fit_test")
    )

    trainer._fit_debug(message="dispatch_override")

    captured = capsys.readouterr()
    assert captured.out.strip() == "[fit-debug step=init] dispatch_override"
