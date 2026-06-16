"""Tests for steer-boundary suffix validation semantics."""

from __future__ import annotations

import pytest

from branching_eval.branch_decode_utils import (
    assert_no_text_after_first_steer_close,
    has_boxed_answer_after_first_think_close,
    is_steer_stop_reason,
    steer_candidate_has_decision_tag,
    text_through_first_steer_terminal,
)


def test_steer_boundary_allows_whitespace_suffix() -> None:
    """Whitespace-only suffix after `</steer>` should be accepted."""

    assert_no_text_after_first_steer_close(text="abc</steer>\n")
    assert_no_text_after_first_steer_close(text="abc</steer>\n\n\t")


def test_steer_boundary_allows_terminal_think_close_suffix() -> None:
    """A steer block may be followed immediately by terminal `</think>`."""

    assert_no_text_after_first_steer_close(text="abc</steer></think>")
    assert_no_text_after_first_steer_close(text="abc</steer>\n</think>\n")


def test_steer_boundary_rejects_non_whitespace_suffix() -> None:
    """Non-whitespace suffix after `</steer>` should hard-fail."""

    with pytest.raises(AssertionError, match="unexpected text after first </steer>"):
        assert_no_text_after_first_steer_close(text="abc</steer>tail")
    with pytest.raises(AssertionError, match="unexpected text after first </steer>"):
        assert_no_text_after_first_steer_close(text="abc</steer></think>tail")


def test_steer_boundary_trim_drops_non_whitespace_suffix() -> None:
    """Candidate-pool cleanup should trim generated text after terminal tags."""

    assert (
        text_through_first_steer_terminal(text="abc</steer>\nanswer") == "abc</steer>"
    )
    assert (
        text_through_first_steer_terminal(text="abc</think>\nanswer") == "abc</think>"
    )


def test_steer_boundary_trim_preserves_whitespace_suffix() -> None:
    """Whitespace-only terminal suffix should be left intact."""

    assert text_through_first_steer_terminal(text="abc</steer>\n") == "abc</steer>\n"
    assert (
        text_through_first_steer_terminal(text="abc</steer></think>")
        == "abc</steer></think>"
    )


def test_post_think_boxed_answer_requires_complete_box() -> None:
    """Post-think answer detection should require a complete boxed expression."""

    assert not has_boxed_answer_after_first_think_close(
        text="\\boxed{7}</think>\nAnswer 7"
    )
    assert not has_boxed_answer_after_first_think_close(text="</think>\nAnswer 7")
    assert not has_boxed_answer_after_first_think_close(text="</think>\n\\boxed{7")
    assert has_boxed_answer_after_first_think_close(
        text="</think>\n\\boxed{\\frac{1}{2}}"
    )


def test_steer_stop_reason_requires_exec_stop_text() -> None:
    """Only explicit exec stop strings should trigger steer-boundary handling."""

    assert is_steer_stop_reason(stop_reason="</exec")
    assert not is_steer_stop_reason(stop_reason="length")
    assert not is_steer_stop_reason(stop_reason=123)


def test_steer_candidate_rejects_think_close_inside_new_steer() -> None:
    """A terminal close inside a new steer tag is malformed."""

    assert not steer_candidate_has_decision_tag(
        prefix="prefix</exec>\n",
        candidate_text="<steer></think>",
    )


def test_steer_candidate_allows_stop_excluded_new_steer() -> None:
    """Stop-excluded steer candidates may omit their closing tag."""

    assert steer_candidate_has_decision_tag(
        prefix="prefix</exec>\n",
        candidate_text="<steer>Conclude k=4 failure",
    )


def test_steer_candidate_rejects_control_tag_inside_open_steer() -> None:
    """Existing open-steer candidates should not contain nested control tags."""

    assert not steer_candidate_has_decision_tag(
        prefix="prefix</exec>\n<steer>",
        candidate_text="<exec></exec>",
    )
