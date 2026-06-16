"""Tests for legacy steer-rollout text helpers."""

from __future__ import annotations

from branching_eval.legacy_steer_rollout import (
    is_chat_eos_stop_reason,
    is_natural_finish_reason,
)


def test_chat_eos_stop_reason_marks_model_finished() -> None:
    """Qwen chat EOS stop markers should count as natural model completion."""

    assert is_chat_eos_stop_reason(stop_reason="<|im_end|>")
    assert is_natural_finish_reason(
        finish_reason="stop",
        stop_reason="<|im_end|>",
    )


def test_structural_stop_reason_is_not_chat_eos() -> None:
    """Structural steer stops should remain continuation boundaries."""

    assert not is_chat_eos_stop_reason(stop_reason="</think>")
    assert not is_natural_finish_reason(
        finish_reason="stop",
        stop_reason="</think>",
    )


def test_plain_slash_s_stop_reason_is_not_chat_eos() -> None:
    """Qwen3.5 tokenizes slash-s as ordinary text, not model EOS."""

    assert not is_chat_eos_stop_reason(stop_reason="</s>")
    assert not is_natural_finish_reason(
        finish_reason="stop",
        stop_reason="</s>",
    )
