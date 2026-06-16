"""Tests for actor update token masking."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from branching_dapo.config_types import BranchAdvantageIndex, BranchingRolloutSettings
from branching_dapo.rollout_utils import build_reward_scores
from branching_dapo.update_masking import (
    build_steer_only_response_mask,
    steer_content_spans,
    validate_update_mode,
)


class CharTokenizer:
    """Minimal tokenizer with one token per character."""

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        """Decode integer character ids."""

        del skip_special_tokens
        del clean_up_tokenization_spaces
        return "".join(chr(token_id) for token_id in token_ids)

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
        return_offsets_mapping: bool = False,
    ) -> dict[str, object]:
        """Encode text with character offsets."""

        del add_special_tokens
        assert return_offsets_mapping
        return {
            "input_ids": [ord(character) for character in text],
            "offset_mapping": [
                (character_index, character_index + 1)
                for character_index in range(len(text))
            ],
        }


def token_ids_for_text(text: str) -> list[int]:
    """Return one integer token id per character."""

    return [ord(character) for character in text]


def test_steer_content_spans_select_only_block_content() -> None:
    """Span extraction should exclude steer tags."""

    text = "<think><steer>Plan A</steer><exec>Work</exec></think>"
    spans = steer_content_spans(text=text)

    assert [text[start:end] for start, end in spans] == ["Plan A"]


def test_build_steer_only_response_mask_selects_steer_text_tokens() -> None:
    """Text fallback should preserve padding and select only steer content."""

    text = "<think><steer>Plan</steer><exec>Work</exec></think>"
    token_ids = token_ids_for_text(text)
    responses = torch.tensor([token_ids + [0, 0]], dtype=torch.long)
    response_mask = torch.tensor([[1] * len(token_ids) + [0, 0]], dtype=torch.long)

    steer_mask, stats = build_steer_only_response_mask(
        responses=responses,
        response_mask=response_mask,
        tokenizer=CharTokenizer(),
    )

    selected_text = "".join(
        chr(int(responses[0, index].item()))
        for index in torch.nonzero(steer_mask[0], as_tuple=False).flatten()
    )
    assert selected_text == "Plan"
    assert stats.selected_token_count == 4
    assert stats.response_token_count == len(token_ids)
    assert steer_mask[0, -2:].tolist() == [0, 0]


def test_build_steer_only_response_mask_prefers_tracked_phase_spans() -> None:
    """Tracked spans should include tags generated during steer-phase requests."""

    text = "<think><exec>Work</exec>\n<steer>Plan</steer></think>"
    token_ids = token_ids_for_text(text)
    responses = torch.tensor([token_ids], dtype=torch.long)
    response_mask = torch.ones_like(responses)
    span_start = text.index("<steer>")
    span_end = len(text)

    steer_mask, stats = build_steer_only_response_mask(
        responses=responses,
        response_mask=response_mask,
        tokenizer=CharTokenizer(),
        steer_phase_token_spans=[[[span_start, span_end]]],
    )

    selected_text = "".join(
        chr(int(responses[0, index].item()))
        for index in torch.nonzero(steer_mask[0], as_tuple=False).flatten()
    )
    assert selected_text == "<steer>Plan</steer></think>"
    assert stats.selected_token_count == len("<steer>Plan</steer></think>")


def test_build_steer_only_response_mask_rejects_missing_steer_content() -> None:
    """Steer-only mode should fail instead of silently producing no update."""

    text = "<think><exec>Work</exec></think>"
    token_ids = token_ids_for_text(text)
    responses = torch.tensor([token_ids], dtype=torch.long)
    response_mask = torch.ones_like(responses)

    with pytest.raises(AssertionError, match="no complete <steer>"):
        build_steer_only_response_mask(
            responses=responses,
            response_mask=response_mask,
            tokenizer=CharTokenizer(),
        )


def test_branching_rollout_settings_parse_update_mode() -> None:
    """Hydra custom rollout config should preserve actor update mode."""

    config = SimpleNamespace(
        actor_rollout_ref=SimpleNamespace(
            rollout=SimpleNamespace(
                custom={"branching_dapo": {"update_mode": "steer_only"}}
            )
        )
    )

    settings = BranchingRolloutSettings.from_config(config=config)

    assert settings.update_mode == "steer_only"
    assert settings.validated_update_mode() == "steer_only"


def test_build_reward_scores_tracks_steer_phase_spans() -> None:
    """Rollout reward metadata should carry steer-phase spans into training."""

    branch_index = make_branch_index()

    reward_scores = build_reward_scores(
        branch_index=branch_index,
        steer_phase_token_spans=((3, 9),),
    )

    assert reward_scores["steer_phase_token_spans"] == [[3, 9]]


def test_validate_update_mode_rejects_unknown_mode() -> None:
    """Unknown update modes should fail before training starts."""

    with pytest.raises(AssertionError, match="Unsupported RL update mode"):
        validate_update_mode(mode="exec_only")


def make_branch_index() -> BranchAdvantageIndex:
    """Build minimal branch metadata for reward-score tests."""

    return BranchAdvantageIndex(
        prompt_uid="prompt-1",
        branch_tree_id="tree-1",
        leaf_id="leaf-1",
        leaf_node_id="node-1",
        path_node_ids=("node-root", "node-1"),
        branch_token_offsets=(3,),
        parent_branch_id="node-root",
        branch_depth=1,
        selected_cluster_id=None,
        cluster_name=None,
        selector_mode="random",
        candidate_pool_key=None,
    )
