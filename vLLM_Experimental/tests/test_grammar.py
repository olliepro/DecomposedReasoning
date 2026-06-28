"""Tests for token-level grammar helpers."""

from __future__ import annotations

import pytest

from vllm_experimental.grammar import (
    GrammarState,
    GrammarTokenIds,
    GrammarTracker,
    suffix_matches,
    temperature_for_state,
)


def test_grammar_tracks_valid_control_flow() -> None:
    tokens = GrammarTokenIds(
        think_open=1,
        think_close=2,
        steer_open=3,
        steer_close=4,
        exec_open=5,
        exec_close=6,
        newline=7,
        eos=8,
    )
    tracker = GrammarTracker(tokens=tokens, max_steer_tokens=2, max_exec_tokens=3)
    state = tracker.observe_many([1, 7, 3, 10, 4, 7, 5, 11, 12, 6, 7, 2, 99, 8])
    assert state == GrammarState.DONE


def test_grammar_allows_newline_between_think_and_steer() -> None:
    tokens = GrammarTokenIds(
        think_open=1,
        think_close=2,
        steer_open=3,
        steer_close=4,
        exec_open=5,
        exec_close=6,
        newline=8,
    )
    tracker = GrammarTracker(tokens=tokens)
    state = tracker.observe_many([1, 8, 3])
    assert state == GrammarState.IN_STEER


def test_grammar_requires_newline_between_think_and_steer() -> None:
    tokens = GrammarTokenIds(
        think_open=1,
        think_close=2,
        steer_open=3,
        steer_close=4,
        exec_open=5,
        exec_close=6,
        newline=8,
    )
    tracker = GrammarTracker(tokens=tokens)
    with pytest.raises(AssertionError, match="expected newline"):
        tracker.observe_many([1, 3])


def test_grammar_allows_think_close_after_think_newline() -> None:
    tokens = GrammarTokenIds(
        think_open=1,
        think_close=2,
        steer_open=3,
        steer_close=4,
        exec_open=5,
        exec_close=6,
        newline=8,
    )
    tracker = GrammarTracker(tokens=tokens)
    state = tracker.observe_many([1, 8, 2])
    assert state == GrammarState.AFTER_THINK_CLOSE


def test_grammar_allows_eos_only_after_final_answer_content() -> None:
    tokens = GrammarTokenIds(
        think_open=1,
        think_close=2,
        steer_open=3,
        steer_close=4,
        exec_open=5,
        exec_close=6,
        newline=8,
        eos=9,
    )

    steer_tracker = GrammarTracker(tokens=tokens)
    with pytest.raises(AssertionError, match="eos is only allowed"):
        steer_tracker.observe_many([1, 8, 3, 9])

    exec_tracker = GrammarTracker(tokens=tokens)
    with pytest.raises(AssertionError, match="eos is only allowed"):
        exec_tracker.observe_many([1, 8, 3, 10, 4, 8, 5, 9])

    immediate_final_tracker = GrammarTracker(tokens=tokens)
    with pytest.raises(AssertionError, match="at least one final answer token"):
        immediate_final_tracker.observe_many([1, 8, 2, 9])

    final_tracker = GrammarTracker(tokens=tokens)
    state = final_tracker.observe_many([1, 8, 2, 99, 9])
    assert state == GrammarState.DONE


def test_grammar_requires_newline_between_steer_and_exec() -> None:
    tokens = GrammarTokenIds(
        think_open=1,
        think_close=2,
        steer_open=3,
        steer_close=4,
        exec_open=5,
        exec_close=6,
        newline=8,
    )
    tracker = GrammarTracker(tokens=tokens)
    with pytest.raises(AssertionError, match="expected newline after </steer>"):
        tracker.observe_many([1, 8, 3, 10, 4, 5])


def test_grammar_requires_newline_after_exec_close() -> None:
    tokens = GrammarTokenIds(
        think_open=1,
        think_close=2,
        steer_open=3,
        steer_close=4,
        exec_open=5,
        exec_close=6,
        newline=8,
    )
    tracker = GrammarTracker(tokens=tokens)
    with pytest.raises(AssertionError, match="expected newline after </exec>"):
        tracker.observe_many([1, 8, 3, 10, 4, 8, 5, 11, 6, 3])


def test_grammar_rejects_invalid_next_tag() -> None:
    tracker = GrammarTracker(
        tokens=GrammarTokenIds(
            think_open=1,
            think_close=2,
            steer_open=3,
            steer_close=4,
            exec_open=5,
            exec_close=6,
        )
    )
    with pytest.raises(AssertionError, match="expected <think>"):
        tracker.observe(token_id=3)


def test_strict_limit_rejection_can_be_disabled_for_prefix_replay() -> None:
    tokens = GrammarTokenIds(
        think_open=1,
        think_close=2,
        steer_open=3,
        steer_close=4,
        exec_open=5,
        exec_close=6,
    )
    strict_tracker = GrammarTracker(tokens=tokens, max_steer_tokens=2)
    with pytest.raises(AssertionError, match="steer block exceeded token limit"):
        strict_tracker.observe_many([1, 3, 10, 11, 12])

    replay_tracker = GrammarTracker(tokens=tokens, max_steer_tokens=2)
    state = replay_tracker.observe_many([1, 3, 10, 11, 12], strict_limits=False)
    assert state == GrammarState.IN_STEER
    assert replay_tracker.steer_token_count == 3


def test_temperature_by_state() -> None:
    assert (
        temperature_for_state(
            state=GrammarState.IN_STEER,
            steer_temperature=1.0,
            exec_temperature=0.7,
            post_think_temperature=0.6,
        )
        == 1.0
    )
    assert (
        temperature_for_state(
            state=GrammarState.AFTER_EXEC_CLOSE,
            steer_temperature=1.0,
            exec_temperature=0.7,
            post_think_temperature=0.6,
        )
        == 1.0
    )
    assert (
        temperature_for_state(
            state=GrammarState.IN_EXEC,
            steer_temperature=1.0,
            exec_temperature=0.7,
            post_think_temperature=0.6,
        )
        == 0.7
    )
    assert (
        temperature_for_state(
            state=GrammarState.AFTER_THINK_CLOSE,
            steer_temperature=1.0,
            exec_temperature=0.7,
            post_think_temperature=0.6,
        )
        == 0.6
    )


def test_suffix_matches() -> None:
    assert suffix_matches(token_ids=[1, 2, 3, 4], suffix=(3, 4))
    assert not suffix_matches(token_ids=[1, 2, 3], suffix=(3, 4))
