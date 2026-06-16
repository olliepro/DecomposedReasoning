"""Unit tests for custom AIME answer extraction."""

from __future__ import annotations

from eval_runner.aime_avgk import extract_candidate_answer, normalize_answer_text


def test_extract_candidate_answer_prefers_boxed_over_answer_marker() -> None:
    """Boxed answers are the highest-priority explicit final answer format."""

    response = "Answer: 123. A later correction is \\boxed{456}. Also $789$."

    assert extract_candidate_answer(response=response) == "456"


def test_extract_candidate_answer_prefers_marker_over_later_dollar_math() -> None:
    """An explicit answer marker should beat later loose dollar-delimited math."""

    response = "Scratch says $999$. Answer: 123. A stale display says $456$."

    assert extract_candidate_answer(response=response) == "123"


def test_extract_candidate_answer_keeps_dollar_fallback_without_marker() -> None:
    """Dollar-delimited answers remain the fallback when no explicit marker exists."""

    response = "The derivation ends with $456$."

    assert extract_candidate_answer(response=response) == "456"


def test_extract_candidate_answer_supports_marker_wrapped_in_dollars() -> None:
    """Answer-marker parsing should still accept a dollar-wrapped numeric answer."""

    response = "The answer is $033$.<|im_end|>"

    assert normalize_answer_text(value=extract_candidate_answer(response=response)) == "33"
