"""Tests for probability and entropy helpers."""

from __future__ import annotations

from token_metrics import (
    approximate_entropy,
    canonicalize_top_logprobs,
    probability_from_logprob,
)


def test_probability_from_logprob() -> None:
    """Probability conversion should map logprob zero to one."""
    assert probability_from_logprob(logprob=0.0) == 1.0


def test_canonicalize_top_logprobs_dict_and_list_shapes() -> None:
    """Top-logprob parser should support dict and list payloads."""
    dict_shape = canonicalize_top_logprobs(raw_top_logprobs={"a": -0.1, "b": -1.0})
    list_shape = canonicalize_top_logprobs(
        raw_top_logprobs=[
            {"token": "a", "logprob": -0.1},
            {"token": "b", "logprob": -1.0},
        ]
    )
    assert len(dict_shape) == 2
    assert len(list_shape) == 2


def test_approximate_entropy_outputs_selected_probability_and_alternatives() -> None:
    """Entropy helper should return positive entropy and sorted alternatives."""
    probability, entropy_value, alternatives = approximate_entropy(
        selected_token="x",
        selected_logprob=-0.2,
        top_entries=[("x", -0.2), ("y", -1.2), ("z", -2.2)],
    )
    assert probability > 0.0
    assert entropy_value > 0.0
    assert alternatives[0].token == "x"
