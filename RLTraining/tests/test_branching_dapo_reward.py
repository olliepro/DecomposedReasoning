"""Unit tests for branching DAPO reward scoring."""

from __future__ import annotations

from typing import cast

from branching_dapo.config_types import BranchAdvantageIndex
from branching_dapo.reward_fn import compute_score_branching_dapo

VALID_THINK_BLOCK = "<think><steer>Plan.</steer><exec>Compute.</exec></think>"


def test_compute_score_branching_dapo_emits_branch_uid_payload() -> None:
    """Reward function should emit serialized branch metadata for the estimator."""
    result = compute_score_branching_dapo(
        data_source="AceReason-Math",
        solution_str=f"{VALID_THINK_BLOCK}\n\nFinal answer: \\boxed{{5}}",
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
                    "branch_token_offsets": [2, 8],
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


def test_compute_score_branching_dapo_emits_repeat_metadata() -> None:
    """Reward extras should preserve runtime repeat-loop metadata for logging."""

    result = compute_score_branching_dapo(
        data_source="AceReason-Math",
        solution_str=f"{VALID_THINK_BLOCK}\n\nFinal answer: \\boxed{{5}}",
        ground_truth=["5"],
        extra_info={
            "source_family": "AceReason-Math",
            "rollout_reward_scores": {
                "branch_metadata": {
                    "prompt_uid": "prompt-1",
                    "branch_tree_id": "tree-1",
                    "leaf_id": "leaf-1",
                    "leaf_node_id": "node_a1",
                    "path_node_ids": ["node_root", "node_a1"],
                    "branch_token_offsets": [4],
                    "parent_branch_id": "node_root",
                    "branch_depth": 1,
                    "selected_cluster_id": None,
                    "cluster_name": None,
                    "selector_mode": "structured_baseline",
                    "candidate_pool_key": None,
                },
                "repeat_stop_reason": "repeated_steer_block_loop",
                "repeat_block_kind": "steer",
                "repeat_block_count": 4,
                "repeat_last_similarity_ratio": 0.96,
            },
        },
    )

    assert result["repeat_forced_think_close"] is True
    assert result["repeat_exec_forced_close"] is False
    assert result["repeat_steer_forced_close"] is True
    assert result["repeat_stop_reason"] == "repeated_steer_block_loop"
    assert result["repeat_block_kind"] == "steer"
    assert result["repeat_block_count"] == 4
    assert result["repeat_last_similarity_ratio"] == 0.96


def test_compute_score_branching_dapo_accepts_boxed_numeric_normalization() -> None:
    """Reward function should normalize boxed numeric answers with separators."""

    result = compute_score_branching_dapo(
        data_source="rlvr_orz_math_57k_collected",
        solution_str=f"{VALID_THINK_BLOCK}\n\nThe final answer is \\boxed{{30000}}.",
        ground_truth=["30,000"],
        extra_info={
            "source_family": "rlvr_orz_math_57k_collected",
            "rollout_reward_scores": {},
        },
    )

    assert result["score"] == 1.1
    assert result["acc"] is True
    assert result["structure_reward"] == 0.1
    assert result["answer_reward"] == 1.0
    assert result["pred"] == "30000"
    assert result["reward_parse_mode"] == "math"


def test_compute_score_branching_dapo_restores_prefilled_think_prefix() -> None:
    """Reward should score the logical assistant text when think is prefilled."""

    result = compute_score_branching_dapo(
        data_source="AceReason-Math",
        solution_str="<steer>Plan.</steer><exec>Compute.</exec></think>\n\n\\boxed{5}",
        ground_truth=["5"],
        extra_info={
            "source_family": "AceReason-Math",
            "rollout_reward_scores": {"initial_assistant_prefix": "<think>\n"},
        },
    )

    assert result["score"] == 1.1
    assert result["acc"] is True
    assert result["format_valid"] is True
    assert result["initial_assistant_prefix"] == "<think>\n"


def test_compute_score_branching_dapo_prefers_logical_response_text() -> None:
    """Reward should score the executor's canonical leaf text when provided."""

    logical_response_text = f"{VALID_THINK_BLOCK}\n\n\\boxed{{5}}"
    result = compute_score_branching_dapo(
        data_source="AceReason-Math",
        solution_str="decoded response ids missed the prompt prefill",
        ground_truth=["5"],
        extra_info={
            "source_family": "AceReason-Math",
            "rollout_reward_scores": {
                "initial_assistant_prefix": "<think>\n",
                "logical_response_text": logical_response_text,
            },
        },
    )

    assert result["score"] == 1.1
    assert result["acc"] is True
    assert result["format_valid"] is True
    assert result["used_logical_response_text"] is True
    assert result["initial_assistant_prefix"] == "<think>\n"


def test_compute_score_branching_dapo_uses_string_fallback_for_text_answers() -> None:
    """Reward function should support targeted string matching for non-math answers."""

    result = compute_score_branching_dapo(
        data_source="omega-combined-no-boxed",
        solution_str=f"{VALID_THINK_BLOCK}\n\nAnswer: \\boxed{{No}}",
        ground_truth=[r"\text{No}"],
        extra_info={
            "source_family": "omega-combined-no-boxed",
            "rollout_reward_scores": {},
        },
    )

    assert result["score"] == 1.1
    assert result["acc"] is True
    assert result["structure_reward"] == 0.1
    assert result["answer_reward"] == 1.0
    assert result["pred"] == "No"
    assert result["reward_parse_mode"] == "string"


def test_compute_score_branching_dapo_uses_string_fallback_for_interval_answers() -> (
    None
):
    """Reward function should match interval-style answers when math parsing fails."""

    result = compute_score_branching_dapo(
        data_source="AceReason-Math",
        solution_str=f"{VALID_THINK_BLOCK}\n\nThe answer is \\boxed{{(0,1)}}.",
        ground_truth=["(0,1)"],
        extra_info={"source_family": "AceReason-Math", "rollout_reward_scores": {}},
    )

    assert result["score"] == 1.1
    assert result["acc"] is True
    assert result["structure_reward"] == 0.1
    assert result["answer_reward"] == 1.0
    assert result["pred"] == "(0,1)"
    assert result["reward_parse_mode"] == "string"


def test_compute_score_branching_dapo_string_fallback_accepts_dfrac_alias() -> None:
    """String fallback should not penalize common LaTeX fraction aliases."""

    result = compute_score_branching_dapo(
        data_source="MathSub-30K",
        solution_str=(
            f"{VALID_THINK_BLOCK}\n\n" r"The answer is \boxed{\dfrac{7\pi}{24}}."
        ),
        ground_truth=[r"\frac{7\pi}{24}"],
        extra_info={"source_family": "MathSub-30K", "rollout_reward_scores": {}},
    )

    assert result["score"] == 1.1
    assert result["acc"] is True
    assert result["answer_acc"] is True
    assert result["raw_answer_acc"] is True
    assert result["pred"] == r"\dfrac{7\pi}{24}"
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
    assert result["exec_block_count"] == 2
    assert result["exec_block_word_count"] == 13
    assert result["steer_block_count"] == 2
    assert result["steer_block_word_count"] == 5
    assert result["structure_issues"] == []


def test_compute_score_branching_dapo_rejects_missing_steer_exec_by_default() -> None:
    """Strict reward mode should require steer/exec pairs inside think text."""

    result = compute_score_branching_dapo(
        data_source="AceReason-Math",
        solution_str="<think>Reason carefully.</think>\n\n\\boxed{5}",
        ground_truth=["5"],
        extra_info={"source_family": "AceReason-Math", "rollout_reward_scores": {}},
    )

    assert result["score"] == 1.0
    assert result["acc"] is True
    assert result["raw_answer_acc"] is True
    assert result["structure_reward"] == 0.0
    assert result["answer_reward"] == 1.0
    assert result["require_steer_exec"] is True
    assert "missing_steer_exec_blocks" in cast(list[str], result["structure_issues"])


def test_compute_score_branching_dapo_can_disable_steer_exec_requirement() -> None:
    """Reward kwargs should support steer/exec-free structure ablations."""

    result = compute_score_branching_dapo(
        data_source="AceReason-Math",
        solution_str="<think>Reason carefully.</think>\n\n\\boxed{5}",
        ground_truth=["5"],
        extra_info={"source_family": "AceReason-Math", "rollout_reward_scores": {}},
        require_steer_exec=False,
    )

    assert result["score"] == 1.1
    assert result["acc"] is True
    assert result["require_steer_exec"] is False
    assert result["structure_issues"] == []


def test_compute_score_branching_dapo_rejects_empty_think_block() -> None:
    """Reward function should reject empty thinking even when steer/exec is ablated."""

    result = compute_score_branching_dapo(
        data_source="AceReason-Math",
        solution_str="<think> </think>\n\n\\boxed{5}",
        ground_truth=["5"],
        extra_info={"source_family": "AceReason-Math", "rollout_reward_scores": {}},
        require_steer_exec=False,
    )

    assert result["score"] == 1.0
    assert result["acc"] is True
    assert result["answer_acc"] is True
    assert result["structure_reward"] == 0.0
    assert result["answer_reward"] == 1.0
    assert result["format_valid"] is False
    assert "empty_think_block" in cast(list[str], result["structure_issues"])


def test_compute_score_branching_dapo_rejects_empty_steer_exec_block() -> None:
    """Reward function should require non-empty steer and exec content."""

    result = compute_score_branching_dapo(
        data_source="AceReason-Math",
        solution_str=(
            "<think><steer></steer><exec>Compute.</exec></think>\n\n" "\\boxed{5}"
        ),
        ground_truth=["5"],
        extra_info={"source_family": "AceReason-Math", "rollout_reward_scores": {}},
    )

    assert result["score"] == 1.0
    assert result["acc"] is True
    assert result["answer_acc"] is True
    assert result["structure_reward"] == 0.0
    assert result["answer_reward"] == 1.0
    assert result["format_valid"] is False
    assert "empty_steer_exec_block" in cast(list[str], result["structure_issues"])


def test_compute_score_branching_dapo_rejects_answer_before_think() -> None:
    """Reward function should only extract answers that appear after thinking."""

    result = compute_score_branching_dapo(
        data_source="AceReason-Math",
        solution_str=f"\\boxed{{5}}\n{VALID_THINK_BLOCK}",
        ground_truth=["5"],
        extra_info={"source_family": "AceReason-Math", "rollout_reward_scores": {}},
    )

    assert result["score"] == 0.0
    assert result["boxed_present"] is False
    assert "non_whitespace_before_think_block" in cast(
        list[str], result["structure_issues"]
    )
    assert "missing_boxed_answer" in cast(list[str], result["structure_issues"])


def test_compute_score_branching_dapo_uses_final_boxed_answer() -> None:
    """Reward function should score the final boxed answer after thinking."""

    result = compute_score_branching_dapo(
        data_source="AceReason-Math",
        solution_str=f"{VALID_THINK_BLOCK}\n\nFirst \\boxed{{4}}, then \\boxed{{5}}.",
        ground_truth=["5"],
        extra_info={"source_family": "AceReason-Math", "rollout_reward_scores": {}},
    )

    assert result["score"] == 1.1
    assert result["format_valid"] is True
    assert result["boxed_present"] is True
    assert result["boxed_answer"] == "\\boxed{5}"
    assert result["pred"] == "5"
    assert "multiple_boxed_answers" not in cast(list[str], result["structure_issues"])


def test_compute_score_branching_dapo_awards_structure_only_reward() -> None:
    """Reward function should award only the structure bonus for wrong boxed answers."""

    result = compute_score_branching_dapo(
        data_source="AceReason-Math",
        solution_str=f"{VALID_THINK_BLOCK}\n\nFinal answer: \\boxed{{7}}",
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
    assert result["raw_answer_acc"] is True
    assert result["structure_reward"] == 0.0
    assert result["answer_reward"] == 1.0
    assert result["format_valid"] is False
    assert "expected_single_complete_think_block" in cast(
        list[str], result["structure_issues"]
    )


def test_compute_score_branching_dapo_returns_zero_for_wrong_and_invalid_response() -> (
    None
):
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


def test_compute_score_branching_dapo_rejects_non_whitespace_between_steer_exec_pairs() -> (
    None
):
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
    assert result["answer_acc"] is True
    assert result["structure_reward"] == 0.0
    assert result["answer_reward"] == 1.0
    assert result["format_valid"] is False
    assert "non_whitespace_outside_steer_exec" in cast(
        list[str], result["structure_issues"]
    )


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
    assert result["answer_acc"] is True
    assert result["structure_reward"] == 0.0
    assert result["answer_reward"] == 1.0
    assert result["format_valid"] is False
    assert "unequal_steer_exec_block_counts" in cast(
        list[str], result["structure_issues"]
    )


def test_compute_score_branching_dapo_rejects_missing_boxed_output() -> None:
    """Reward function should require a boxed answer outside the think block."""

    result = compute_score_branching_dapo(
        data_source="AceReason-Math",
        solution_str=f"{VALID_THINK_BLOCK}\n\nFinal answer: 5",
        ground_truth=["5"],
        extra_info={"source_family": "AceReason-Math", "rollout_reward_scores": {}},
    )

    assert result["score"] == 0.0
    assert result["acc"] is False
    assert result["answer_acc"] is False
    assert result["raw_answer_acc"] is False
    assert result["structure_reward"] == 0.0
    assert result["answer_reward"] == 0.0
    assert result["pred"] is None
    assert result["boxed_present"] is False
    assert "missing_boxed_answer" in cast(list[str], result["structure_issues"])


def test_compute_score_branching_dapo_rejects_steer_exec_outside_think() -> None:
    """Reward function should reject steer/exec tags that appear outside think text."""

    solution_str = (
        f"{VALID_THINK_BLOCK}\n\n"
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
    assert result["answer_acc"] is True
    assert result["structure_reward"] == 0.0
    assert result["answer_reward"] == 1.0
    assert result["format_valid"] is False
    assert "steer_exec_outside_think_block" in cast(
        list[str], result["structure_issues"]
    )
