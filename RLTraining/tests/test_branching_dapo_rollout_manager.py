"""Unit tests for branching DAPO rollout-manager integration helpers."""

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

import branching_dapo.agent_loop_manager as agent_loop_manager_module
from branching_dapo.agent_loop_manager import (
    BranchingAgentLoopManager,
    QWEN_THINK_ASSISTANT_PREFIX,
)
from branching_dapo.config_types import BranchingRolloutSettings
from branching_dapo.rollout_utils import (
    LeafBatchRecord,
    PromptGroup,
    resolve_leaf_branch_metadata,
)
from branching_eval.event_db import EventDatabase
from branching_eval.event_types import EventContext
from branching_eval.selector_types import SelectionOutcome, SelectorParams
from branching_eval.artifact_store import ArtifactStore
from branching_eval.tree_types import (
    BranchPointRecord,
    BranchTree,
    LeafRollout,
    TreeEdge,
    TreeNode,
)
from verl.protocol import DataProto


def test_branching_rollout_settings_parses_steer_sampling() -> None:
    """Hydra custom rollout config should preserve steer-specific sampling."""

    config = SimpleNamespace(
        actor_rollout_ref=SimpleNamespace(
            rollout=SimpleNamespace(
                custom={
                    "branching_dapo": {
                        "steer_temperature": 1.1,
                        "steer_top_p": 0.77,
                        "steer_repetition_penalty": 1.0,
                    }
                }
            )
        )
    )

    settings = BranchingRolloutSettings.from_config(config=config)

    assert settings.steer_temperature == 1.1
    assert settings.steer_top_p == 0.77
    assert settings.steer_repetition_penalty == 1.0


def test_initial_assistant_prefix_detects_qwen_think_prefill() -> None:
    """Qwen chat-template think prefill should be mirrored into executor state."""

    class FakeTokenizer:
        """Tokenizer stub returning the Qwen think prefill ids."""

        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            assert text == QWEN_THINK_ASSISTANT_PREFIX
            assert add_special_tokens is False
            return [248068, 198]

    manager = cast(Any, object.__new__(BranchingAgentLoopManager))
    manager.tokenizer = FakeTokenizer()

    assert (
        manager._initial_assistant_prefix(prompt_ids=[1, 2, 248068, 198])
        == QWEN_THINK_ASSISTANT_PREFIX
    )
    assert manager._initial_assistant_prefix(prompt_ids=[1, 2, 3]) == ""


def test_tokenize_prompt_enables_qwen_thinking_template() -> None:
    """Qwen chat templates must leave the assistant at an open think block."""

    class FakeTokenizer:
        """Tokenizer stub recording chat-template kwargs."""

        def __init__(self) -> None:
            self.enable_thinking: bool | None = None

        def apply_chat_template(
            self,
            raw_prompt: list[dict[str, str]],
            *,
            add_generation_prompt: bool,
            enable_thinking: bool,
            tokenize: bool,
        ) -> list[int]:
            _ = raw_prompt
            assert add_generation_prompt is True
            assert tokenize is True
            self.enable_thinking = enable_thinking
            return [1, 2, 248068, 198]

    tokenizer = FakeTokenizer()
    manager = cast(Any, object.__new__(BranchingAgentLoopManager))
    manager.tokenizer = tokenizer
    manager.rollout_config = SimpleNamespace(prompt_length=1024)

    prompt_ids = manager._tokenize_prompt(
        raw_prompt=[{"role": "user", "content": "Question?"}]
    )

    assert tokenizer.enable_thinking is True
    assert prompt_ids == [1, 2, 248068, 198]


def test_tokenize_prompt_left_truncates_to_rollout_prompt_length() -> None:
    """The rollout manager should not emit prompts longer than VERL accepts."""

    class FakeTokenizer:
        """Tokenizer stub returning an over-budget prompt."""

        def apply_chat_template(
            self,
            raw_prompt: list[dict[str, str]],
            *,
            add_generation_prompt: bool,
            enable_thinking: bool,
            tokenize: bool,
        ) -> list[int]:
            _ = raw_prompt, add_generation_prompt, enable_thinking, tokenize
            return list(range(10))

    manager = cast(Any, object.__new__(BranchingAgentLoopManager))
    manager.tokenizer = FakeTokenizer()
    manager.rollout_config = SimpleNamespace(prompt_length=4)

    prompt_ids = manager._tokenize_prompt(
        raw_prompt=[{"role": "user", "content": "Question?"}]
    )

    assert prompt_ids == [6, 7, 8, 9]


def test_rl_prompt_logged_event_shows_ground_truth_first(tmp_path) -> None:
    """RL artifacts should expose prompt and gold answer as the first graph event."""

    store = ArtifactStore(run_dir=tmp_path / "run")
    prompt_group = PromptGroup(
        prompt_uid="prompt-1",
        raw_prompt=[{"role": "user", "content": "Question?"}],
        data_source="math",
        extra_info={},
        reward_model={"ground_truth": ["5"]},
        group_size=1,
    )
    manager = cast(Any, object.__new__(BranchingAgentLoopManager))

    manager._append_prompt_logged_event(
        artifact_store=store,
        prompt_group=prompt_group,
        doc_id=0,
        prompt_text="Question?",
        selector_mode="cluster_across",
    )
    store.append_event(
        context=EventContext(
            run_id=store.run_id,
            doc_id=0,
            doc_attempt=0,
            task_name="branching_dapo_train",
            model_id="branching_dapo",
            selector_mode="cluster_across",
        ),
        event_type="node_created",
        payload={
            "node_id": "node_root",
            "parent_node_id": None,
            "branch_points_used": 0,
        },
    )
    store.flush_events()

    rows = store.read_event_rows()
    assert rows[0]["event_type"] == "prompt_logged"
    assert rows[0]["payload"]["prompt_text"] == "Question?"
    assert rows[0]["payload"]["golden_answer"] == "5"
    db = EventDatabase(path=tmp_path / "run" / "tree_events.sqlite")
    graph_events = db.read_node_event_rows_for_attempt(
        doc_id=0,
        doc_attempt=0,
        task_name="branching_dapo_train",
        model_id="branching_dapo",
        selector_mode="cluster_across",
    )
    assert graph_events[0]["event_type"] == "prompt_logged"
    assert graph_events[0]["text_preview"] == "5"
    detail_events = db.read_node_detail_rows(
        node_id="node_root",
        doc_id=0,
        doc_attempt=0,
        task_name="branching_dapo_train",
        model_id="branching_dapo",
        selector_mode="cluster_across",
    )
    assert detail_events[0]["prompt_text"] == "Question?"
    assert detail_events[0]["golden_answer"] == "5"


def test_resolve_leaf_branch_metadata_uses_candidate_pool_id() -> None:
    """Current Analysis branch trees expose candidate_pool_id at branch points."""
    leaf = LeafRollout(
        leaf_id="leaf-1",
        node_id="node-a",
        text="answer",
        token_ids=(1, 2),
        tokens=(),
        verification=1,
        length_tokens_total=2,
        length_tokens_exec=None,
        stop_reason="done",
        task_metrics={},
    )
    tree = BranchTree(
        doc_id=7,
        task_name="math",
        model_id="qwen35",
        selector_mode="cluster_across",
        root_prompt="prompt",
        doc_attempt=0,
        run_id="run-1",
        nodes={
            "node-root": TreeNode(
                node_id="node-root",
                parent_node_id=None,
                prompt_text="prompt",
                assistant_prefix="",
                prompt_token_ids=(1, 2, 3, 4),
                branch_points_used=0,
            ),
            "node-a": TreeNode(
                node_id="node-a",
                parent_node_id="node-root",
                prompt_text="prompt",
                assistant_prefix="choice",
                prompt_token_ids=(1, 2, 3, 4, 5, 6),
                branch_points_used=1,
            ),
        },
        edges=[
            TreeEdge(
                edge_id="edge-1",
                parent_node_id="node-root",
                child_node_id="node-a",
                candidate_pool_id="pool-1",
                candidate_id=2,
                selector_mode="cluster_across",
            )
        ],
        branch_points=[
            BranchPointRecord(
                branch_point_id="bp-1",
                node_id="node-root",
                trigger_type="steer_boundary",
                candidate_pool_id="pool-1",
                selections=(
                    SelectionOutcome(
                        selector_mode="cluster_across",
                        selected_candidate_ids=(2,),
                        cluster_by_candidate_id={2: "cluster-b"},
                    ),
                ),
            )
        ],
        leaves=[leaf],
    )

    branch_index = resolve_leaf_branch_metadata(
        tree=tree,
        leaf=leaf,
        prompt_uid="prompt-1",
        selector_mode="cluster_across",
        prompt_token_count=3,
    )

    assert branch_index.candidate_pool_key == "pool-1"
    assert branch_index.parent_branch_id == "node-root"
    assert branch_index.selected_cluster_id == "cluster-b"
    assert branch_index.branch_token_offsets == (1,)


def test_build_branch_records_preserves_logical_response_text() -> None:
    """Reward metadata should carry the executor's canonical leaf response."""

    leaf = LeafRollout(
        leaf_id="leaf-1",
        node_id="node-root",
        text="<think>\n<steer>Plan.</steer><exec>Compute.</exec></think>\n\n\\boxed{5}",
        token_ids=(10, 11),
        tokens=(),
        verification=1,
        length_tokens_total=2,
        length_tokens_exec=None,
        stop_reason="done",
        task_metrics={},
    )
    tree = BranchTree(
        doc_id=1,
        task_name="math",
        model_id="qwen35",
        selector_mode="random",
        root_prompt="prompt",
        doc_attempt=0,
        run_id="run-1",
        nodes={
            "node-root": TreeNode(
                node_id="node-root",
                parent_node_id=None,
                prompt_text="prompt",
                assistant_prefix="",
                prompt_token_ids=(),
                branch_points_used=0,
            )
        },
        edges=[],
        branch_points=[],
        leaves=[leaf],
    )
    manager = cast(Any, object.__new__(BranchingAgentLoopManager))
    manager.settings = BranchingRolloutSettings(selector_mode="random")
    prompt_group = PromptGroup(
        prompt_uid="prompt-1",
        raw_prompt=[{"role": "user", "content": "Question?"}],
        data_source="math",
        extra_info={},
        reward_model={"ground_truth": ["5"]},
        group_size=1,
    )

    records = manager._build_branch_records(
        prompt_group=prompt_group,
        prompt_ids=[1, 2, 3],
        initial_assistant_prefix=QWEN_THINK_ASSISTANT_PREFIX,
        tree=tree,
    )

    assert len(records) == 1
    reward_scores = records[0].reward_scores
    assert reward_scores["initial_assistant_prefix"] == QWEN_THINK_ASSISTANT_PREFIX
    assert reward_scores["logical_response_text"] == leaf.text
    assert reward_scores["event_context"] == {
        "doc_id": 1,
        "doc_attempt": 0,
        "task_name": "math",
        "model_id": "qwen35",
        "selector_mode": "random",
    }
    assert reward_scores["leaf_runtime"] == {
        "leaf_id": "leaf-1",
        "node_id": "node-root",
        "length_tokens_total": 2,
        "length_tokens_exec": None,
        "stop_reason": "done",
        "text": leaf.text,
        "text_preview": leaf.text,
        "steer_phase_token_spans": [],
    }


def test_attach_reward_outputs_logs_leaf_scored_event(tmp_path) -> None:
    """Reward-loop output should become graph-visible leaf score events."""

    class FakeRemoteMethod:
        """Minimal async Ray-style remote method."""

        def remote(self, data: object) -> object:
            _ = data
            return self._result()

        async def _result(self) -> dict[str, object]:
            return {
                "reward_score": 1.1,
                "reward_extra_info": {
                    "acc": True,
                    "answer_acc": True,
                    "format_valid": True,
                    "boxed_answer": "5",
                },
            }

    class FakeRewardWorker:
        """Minimal reward-loop worker handle."""

        compute_score = FakeRemoteMethod()

    reward_scores: dict[str, object] = {
        "branch_metadata": {
            "leaf_id": "leaf-1",
            "leaf_node_id": "node_root",
        },
        "event_context": {
            "doc_id": 7,
            "doc_attempt": 0,
            "task_name": "branching_dapo_train",
            "model_id": "branching_dapo",
            "selector_mode": "cluster_across",
        },
        "leaf_runtime": {
            "leaf_id": "leaf-1",
            "node_id": "node_root",
            "length_tokens_total": 3,
            "length_tokens_exec": 1,
            "stop_reason": "think_end",
            "text_preview": "preview",
        },
    }
    data = DataProto(
        batch=TensorDict(
            {
                "prompts": torch.tensor([[1, 2]], dtype=torch.long),
                "responses": torch.tensor([[3, 4, 5]], dtype=torch.long),
                "attention_mask": torch.tensor([[1, 1, 1, 1, 1]], dtype=torch.long),
            },
            batch_size=[1],
        ),
        non_tensor_batch={
            "reward_scores": np.array([reward_scores], dtype=object),
        },
        meta_info={},
    )
    manager = cast(Any, object.__new__(BranchingAgentLoopManager))
    manager.reward_loop_worker_handles = [FakeRewardWorker()]
    artifact_store = ArtifactStore(run_dir=tmp_path / "run")

    updated = asyncio.run(
        manager._attach_reward_outputs(
            data=data,
            artifact_store=artifact_store,
        )
    )

    rows = [
        row
        for row in artifact_store.read_event_rows()
        if row["event_type"] == "leaf_scored"
    ]
    assert len(rows) == 1
    payload = rows[0]["payload"]
    assert payload["leaf_id"] == "leaf-1"
    assert payload["node_id"] == "node_root"
    assert payload["verification"] == 1
    assert payload["task_metrics"]["score"] == 1.1
    assert payload["task_metrics"]["format_valid"] is True
    assert updated.batch["rm_scores"][0, 2].item() == pytest.approx(1.1)


def test_attach_reward_outputs_timeout_returns_zero_reward(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stuck reward future should not hang the rollout manager."""

    class HangingRemoteMethod:
        """Minimal async Ray-style remote method that never finishes in time."""

        def remote(self, data: object) -> object:
            _ = data
            return self._result()

        async def _result(self) -> dict[str, object]:
            await asyncio.sleep(10)
            return {"reward_score": 1.0, "reward_extra_info": {"acc": True}}

    class FakeRewardWorker:
        """Minimal reward-loop worker handle."""

        compute_score = HangingRemoteMethod()

    monkeypatch.setattr(
        agent_loop_manager_module,
        "REWARD_SCORE_TIMEOUT_SECONDS",
        0.01,
    )
    reward_scores: dict[str, object] = {
        "event_context": {
            "doc_id": 7,
            "doc_attempt": 0,
            "task_name": "branching_dapo_train",
            "model_id": "branching_dapo",
            "selector_mode": "cluster_across",
        },
        "leaf_runtime": {
            "leaf_id": "leaf-1",
            "node_id": "node_root",
            "length_tokens_total": 3,
            "length_tokens_exec": 1,
            "stop_reason": "think_end",
            "text_preview": "preview",
        },
    }
    data = DataProto(
        batch=TensorDict(
            {
                "prompts": torch.tensor([[1, 2]], dtype=torch.long),
                "responses": torch.tensor([[3, 4, 5]], dtype=torch.long),
                "attention_mask": torch.tensor([[1, 1, 1, 1, 1]], dtype=torch.long),
            },
            batch_size=[1],
        ),
        non_tensor_batch={
            "reward_scores": np.array([reward_scores], dtype=object),
        },
        meta_info={},
    )
    manager = cast(Any, object.__new__(BranchingAgentLoopManager))
    manager.reward_loop_worker_handles = [FakeRewardWorker()]
    artifact_store = ArtifactStore(run_dir=tmp_path / "run")

    updated = asyncio.run(
        manager._attach_reward_outputs(
            data=data,
            artifact_store=artifact_store,
        )
    )

    rows = [
        row
        for row in artifact_store.read_event_rows()
        if row["event_type"] == "leaf_scored"
    ]
    assert len(rows) == 1
    payload = rows[0]["payload"]
    assert payload["verification"] == 0
    assert payload["task_metrics"]["score"] == 0.0
    assert payload["task_metrics"]["reward_error"] == "timeout"
    assert payload["task_metrics"]["reward_timeout"] is True
    assert updated.batch["rm_scores"][0, 2].item() == pytest.approx(0.0)
    assert updated.non_tensor_batch["reward_timeout"][0] is True


def test_build_non_tensor_batch_embeds_reward_scores_in_extra_info(
    tmp_path: Path,
) -> None:
    """Async reward-loop managers receive rollout metadata through extra_info."""

    leaf = LeafRollout(
        leaf_id="leaf-1",
        node_id="node-root",
        text="<think>\n<steer>Plan.</steer><exec>Compute.</exec></think>\n\n\\boxed{5}",
        token_ids=(10, 11),
        tokens=(),
        verification=1,
        length_tokens_total=2,
        length_tokens_exec=None,
        stop_reason="done",
        task_metrics={},
    )
    branch_index = resolve_leaf_branch_metadata(
        tree=BranchTree(
            doc_id=1,
            task_name="math",
            model_id="qwen35",
            selector_mode="random",
            root_prompt="prompt",
            doc_attempt=0,
            run_id="run-1",
            nodes={},
            edges=[],
            branch_points=[],
            leaves=[leaf],
        ),
        leaf=leaf,
        prompt_uid="prompt-1",
        selector_mode="random",
        prompt_token_count=0,
    )
    reward_scores: dict[str, object] = {
        "initial_assistant_prefix": QWEN_THINK_ASSISTANT_PREFIX,
        "logical_response_text": leaf.text,
    }
    output = LeafBatchRecord(
        prompt_ids=[1],
        response_ids=[2],
        response_logprobs=None,
        reward_scores=reward_scores,
        branch_index=branch_index,
    )
    prompt_group = PromptGroup(
        prompt_uid="prompt-1",
        raw_prompt=[{"role": "user", "content": "Question?"}],
        data_source="math",
        extra_info={"source_row_id": "row-1"},
        reward_model={"ground_truth": ["5"]},
        group_size=1,
    )
    manager = cast(Any, object.__new__(BranchingAgentLoopManager))

    non_tensor_batch = manager._build_non_tensor_batch(
        outputs=[output],
        prompt_group_by_uid={"prompt-1": prompt_group},
        tree_events_db_path=tmp_path / "run" / "tree_events.sqlite",
    )

    assert non_tensor_batch["reward_scores"][0] == reward_scores
    assert non_tensor_batch["extra_info"][0]["source_row_id"] == "row-1"
    assert non_tensor_batch["extra_info"][0]["rollout_reward_scores"] == reward_scores
    assert non_tensor_batch["tree_events_db_path"][0].endswith("tree_events.sqlite")


def test_build_executor_passes_compatible_epsilon_greedy_prob(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Branching executor config should keep branching and inline epsilon-greedy compatible."""

    captured_kwargs: dict[str, Any] = {}

    class FakeVllmClient:
        """Minimal vLLM client capturing the selected server URL."""

        def __init__(self, *, base_url: str) -> None:
            self.base_url = base_url

    class FakeInstrumentedBranchExecutor:
        """Minimal executor capturing constructor keyword arguments."""

        def __init__(self, **kwargs: Any) -> None:
            captured_kwargs.update(kwargs)

    monkeypatch.setattr(agent_loop_manager_module, "VllmClient", FakeVllmClient)
    monkeypatch.setattr(
        agent_loop_manager_module,
        "InstrumentedBranchExecutor",
        FakeInstrumentedBranchExecutor,
    )
    manager = cast(Any, object.__new__(BranchingAgentLoopManager))
    manager.server_addresses = ["http://server-0"]
    manager.served_model_name = "qwen35"
    manager.rollout_config = SimpleNamespace(
        temperature=0.7,
        top_p=0.95,
        presence_penalty=1.5,
        repetition_penalty=1.0,
        response_length=1024,
        max_model_len=1536,
    )
    manager.settings = BranchingRolloutSettings(
        selector_mode="cluster_across",
        branch_prob=0.3,
        epsilon_greedy_prob=0.2,
        steer_temperature=1.1,
        steer_top_p=0.77,
        steer_repetition_penalty=1.0,
        top_logprobs=0,
    )
    manager._ensure_branch_task_semaphore = lambda: None

    _ = manager._build_executor(
        prompt_text="Solve.",
        prompt_token_ids=[101, 102],
        initial_assistant_prefix=QWEN_THINK_ASSISTANT_PREFIX,
        doc_id=0,
        artifact_store=object(),
    )

    branching = captured_kwargs["branching"]
    decoding = captured_kwargs["decoding"]
    assert captured_kwargs["active_selector"] == "cluster_across"
    assert captured_kwargs["requested_selectors"] == ("cluster_across",)
    assert captured_kwargs["allow_true_branching"] is True
    assert captured_kwargs["initial_prompt_token_ids"] == (101, 102)
    assert decoding.temperature == 0.7
    assert decoding.steer_temperature == 1.1
    assert decoding.top_p == 0.95
    assert decoding.steer_top_p == 0.77
    assert decoding.presence_penalty == 1.5
    assert decoding.repetition_penalty == 1.0
    assert decoding.max_model_len == 1536
    assert decoding.top_logprobs == 0
    assert decoding.initial_assistant_prefix == QWEN_THINK_ASSISTANT_PREFIX
    assert branching.branch_prob == 0.3
    assert branching.epsilon_greedy_prob == 0.2
    assert branching.steer_repetition_penalty == 1.0


def test_build_executor_uses_eval_style_epsilon_greedy_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Epsilon-greedy mode should disable true branching and use fanout-one config."""

    captured_kwargs: dict[str, Any] = {}

    class FakeVllmClient:
        """Minimal vLLM client capturing the selected server URL."""

        def __init__(self, *, base_url: str) -> None:
            self.base_url = base_url

    class FakeInstrumentedBranchExecutor:
        """Minimal executor capturing constructor keyword arguments."""

        def __init__(self, **kwargs: Any) -> None:
            captured_kwargs.update(kwargs)

    monkeypatch.setattr(agent_loop_manager_module, "VllmClient", FakeVllmClient)
    monkeypatch.setattr(
        agent_loop_manager_module,
        "InstrumentedBranchExecutor",
        FakeInstrumentedBranchExecutor,
    )
    manager = cast(Any, object.__new__(BranchingAgentLoopManager))
    manager.server_addresses = ["http://server-0"]
    manager.served_model_name = "qwen35"
    manager.rollout_config = SimpleNamespace(
        temperature=0.7,
        top_p=0.95,
        response_length=1024,
    )
    manager.settings = BranchingRolloutSettings(
        rollout_mode="epsilon_greedy",
        selector_mode="embed_diverse_topk_random",
        branch_prob=0.3,
        branch_fanout=4,
        epsilon_greedy_prob=0.2,
    )
    manager._ensure_branch_task_semaphore = lambda: None

    _ = manager._build_executor(
        prompt_text="Solve.",
        doc_id=0,
        artifact_store=object(),
    )

    branching = captured_kwargs["branching"]
    assert captured_kwargs["active_selector"] == "embed_diverse_topk_random"
    assert captured_kwargs["requested_selectors"] == ("embed_diverse_topk_random",)
    assert captured_kwargs["allow_true_branching"] is False
    assert branching.branch_prob == 0.2
    assert branching.branch_fanout == 1


def test_pad_response_ids_clamps_to_rollout_response_length() -> None:
    """Packed rollout tensors should honor the configured response length cap."""

    class FakeTokenizer:
        """Minimal tokenizer pad surface used by the rollout manager."""

        padding_side = "right"

        def pad(
            self,
            inputs: dict[str, list[list[int]]],
            *,
            padding: str,
            max_length: int,
            return_tensors: str,
            return_attention_mask: bool,
        ) -> dict[str, torch.Tensor]:
            assert padding == "max_length"
            assert return_tensors == "pt"
            assert return_attention_mask is True
            input_ids = inputs["input_ids"][0]
            padded_ids = input_ids + [0] * (max_length - len(input_ids))
            attention_mask = [1] * len(input_ids) + [0] * (max_length - len(input_ids))
            return {
                "input_ids": torch.tensor([padded_ids], dtype=torch.int64),
                "attention_mask": torch.tensor([attention_mask], dtype=torch.int64),
            }

    manager = cast(Any, object.__new__(BranchingAgentLoopManager))
    manager.rollout_config = SimpleNamespace(response_length=3)
    manager.tokenizer = FakeTokenizer()

    response_ids, attention_mask, response_mask, logprobs = manager._pad_response_ids(
        response_ids=[11, 12, 13, 14, 15],
        response_logprobs=[-0.1, -0.2, -0.3, -0.4, -0.5],
    )

    assert torch.equal(response_ids, torch.tensor([[11, 12, 13]], dtype=torch.int64))
    assert torch.equal(attention_mask, torch.tensor([[1, 1, 1]], dtype=torch.int64))
    assert torch.equal(response_mask, torch.tensor([[1, 1, 1]], dtype=torch.int64))
    assert logprobs is not None
    assert torch.equal(logprobs, torch.tensor([[-0.1, -0.2, -0.3]]))


def test_pad_response_ids_handles_empty_response() -> None:
    """A zero-token leaf should still pack into a valid padded response tensor."""

    class FakeTokenizer:
        """Minimal tokenizer pad surface that requires batched inputs."""

        padding_side = "right"

        def pad(
            self,
            inputs: dict[str, list[list[int]]],
            *,
            padding: str,
            max_length: int,
            return_tensors: str,
            return_attention_mask: bool,
        ) -> dict[str, torch.Tensor]:
            assert padding == "max_length"
            assert return_tensors == "pt"
            assert return_attention_mask is True
            assert inputs["input_ids"] == [[]]
            return {
                "input_ids": torch.tensor([[0, 0, 0]], dtype=torch.int64),
                "attention_mask": torch.tensor([[0, 0, 0]], dtype=torch.int64),
            }

    manager = cast(Any, object.__new__(BranchingAgentLoopManager))
    manager.rollout_config = SimpleNamespace(response_length=3)
    manager.tokenizer = FakeTokenizer()

    response_ids, attention_mask, response_mask, logprobs = manager._pad_response_ids(
        response_ids=[],
        response_logprobs=[],
    )

    assert torch.equal(response_ids, torch.tensor([[0, 0, 0]], dtype=torch.int64))
    assert torch.equal(attention_mask, torch.tensor([[0, 0, 0]], dtype=torch.int64))
    assert torch.equal(response_mask, torch.tensor([[0, 0, 0]], dtype=torch.int64))
    assert logprobs is not None
    assert torch.equal(logprobs, torch.tensor([[0.0, 0.0, 0.0]]))


def test_instrumented_executor_forwards_inline_epsilon_selector_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Instrumentation should not reject inline epsilon selector overrides."""

    captured: dict[str, object] = {}

    async def fake_base_resolve(
        self: object,
        *,
        pool: object,
        selector_params: SelectorParams | None = None,
        selector_modes: tuple[str, ...] | None = None,
    ) -> tuple[SelectionOutcome, ...]:
        _ = self
        captured["pool"] = pool
        captured["selector_params"] = selector_params
        captured["selector_modes"] = selector_modes
        return (
            SelectionOutcome(
                selector_mode="embed_diverse_topk_random",
                selected_candidate_ids=(0,),
                shortlist_candidate_ids=(0,),
            ),
        )

    monkeypatch.setattr(
        agent_loop_manager_module.BranchExecutor,
        "_resolve_selection_outcomes_async",
        fake_base_resolve,
    )
    executor = cast(
        Any,
        object.__new__(agent_loop_manager_module.InstrumentedBranchExecutor),
    )
    executor.active_selector = "embed_diverse_topk_random"
    executor.cluster_counts = []
    selector_params = SelectorParams(branch_fanout=1, max_clusters=4)

    outcomes = asyncio.run(
        executor._resolve_selection_outcomes_async(
            pool="pool",
            selector_params=selector_params,
            selector_modes=("embed_diverse_topk_random",),
        )
    )

    assert outcomes[0].selected_candidate_ids == (0,)
    assert captured["pool"] == "pool"
    assert captured["selector_params"] == selector_params
    assert captured["selector_modes"] == ("embed_diverse_topk_random",)
    assert executor.cluster_counts == [0.0]
