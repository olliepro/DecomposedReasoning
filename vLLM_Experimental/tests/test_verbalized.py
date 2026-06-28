"""Tests for verbalized off-policy helpers."""

from __future__ import annotations

from vllm_experimental.verbalized import (
    continue_exec_prefill,
    continue_steer,
    enumeration_prompt,
    parse_verbalized_options,
    sample_option_numbers,
)


def test_canonical_verbalized_strings() -> None:
    assert (
        enumeration_prompt(candidate_count=5)
        == "Enumerate 5 distinct options for the immediate next decision/step"
    )
    assert continue_steer(option_number=2) == "Proceed with option 2"
    assert continue_exec_prefill(option_number=2) == "Let's do option 2:"


def test_parse_verbalized_options() -> None:
    options = parse_verbalized_options(
        text="1. Try substitution\n2) Bound the expression\n2. Duplicate\n"
    )
    assert [option.option_number for option in options] == [1, 2]
    assert options[1].text == "Bound the expression"


def test_sample_option_numbers_is_deterministic() -> None:
    assert sample_option_numbers(available=(1, 2, 3, 4), count=2, seed=7) == (
        2,
        4,
    )
