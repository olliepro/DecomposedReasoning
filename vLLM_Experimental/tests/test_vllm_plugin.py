"""Tests for vLLM grammar plugin helpers."""

from __future__ import annotations

from vllm_experimental.grammar import GrammarState, GrammarTokenIds, GrammarTracker
from vllm_experimental.vllm_plugin import (
    IncrementalGrammarReplay,
    choice_token_ids,
    empty_block_close_token,
    forced_close_token,
    forced_open_token,
    forced_script_token,
    invalid_control_token_ids,
    replay_tracker,
)


def test_forced_close_token_at_steer_limit() -> None:
    """Processor should force `</steer>` once steer reaches its token cap."""

    tokens = GrammarTokenIds(
        think_open=1,
        think_close=2,
        steer_open=3,
        steer_close=4,
        exec_open=5,
        exec_close=6,
    )
    tracker = GrammarTracker(tokens=tokens, max_steer_tokens=2)
    tracker.observe_many([1, 3, 100, 101])
    assert forced_close_token(tracker=tracker) == tokens.steer_close


def test_forced_close_token_at_exec_limit() -> None:
    """Processor should force `</exec>` once exec reaches its token cap."""

    tokens = GrammarTokenIds(
        think_open=1,
        think_close=2,
        steer_open=3,
        steer_close=4,
        exec_open=5,
        exec_close=6,
    )
    tracker = GrammarTracker(tokens=tokens, max_exec_tokens=2)
    tracker.observe_many([1, 3, 100, 4, 5, 101, 102])
    assert forced_close_token(tracker=tracker) == tokens.exec_close


def test_replay_tracker_forces_close_after_over_cap_prefix() -> None:
    """Replayed native prefixes can exceed a block cap before masking."""

    tokens = GrammarTokenIds(
        think_open=1,
        think_close=2,
        steer_open=3,
        steer_close=4,
        exec_open=5,
        exec_close=6,
    )
    tracker = replay_tracker(
        payload={
            "prefix_output_token_ids": [1, 3, 100, 101, 102],
            "max_steer_tokens": 2,
        },
        output_ids=[],
        tokens=tokens,
    )
    assert tracker.steer_token_count == 3
    assert forced_close_token(tracker=tracker) == tokens.steer_close


def test_replay_tracker_uses_prefix_output_for_native_child_state() -> None:
    """Native hidden children replay parent output while keeping suffix new-only."""

    tokens = GrammarTokenIds(
        think_open=1,
        think_close=2,
        steer_open=3,
        steer_close=4,
        exec_open=5,
        exec_close=6,
        newline=7,
    )
    tracker = replay_tracker(
        payload={
            "prefix_output_token_ids": [1, 7, 3, 100, 4, 7, 5, 101, 6, 7],
            "forced_output_start_index": 0,
            "forced_output_token_ids": [3],
            "max_steer_tokens": 30,
            "max_exec_tokens": 512,
        },
        output_ids=[],
        tokens=tokens,
    )
    assert tracker.state == GrammarState.AFTER_EXEC_CLOSE
    assert tracker.exec_close_newline_seen
    assert (
        forced_script_token(
            payload={
                "forced_output_start_index": 0,
                "forced_output_token_ids": [tokens.steer_open],
            },
            output_ids=[],
            tokens=tokens,
        )
        == tokens.steer_open
    )


def test_incremental_replay_advances_only_new_output_tokens() -> None:
    tokens = GrammarTokenIds(
        think_open=1,
        think_close=2,
        steer_open=3,
        steer_close=4,
        exec_open=5,
        exec_close=6,
        newline=7,
    )
    payload = {"prefix_output_token_ids": [1, 7, 3], "max_steer_tokens": 30}
    replay = IncrementalGrammarReplay(payload=payload, tokens=tokens)

    tracker = replay.sync(payload=payload, output_ids=[100])
    assert tracker.state == GrammarState.IN_STEER
    assert tracker.steer_token_count == 1
    assert replay.observed_len == 1

    same_tracker = replay.sync(payload=payload, output_ids=[100])
    assert same_tracker is tracker
    assert same_tracker.steer_token_count == 1

    next_tracker = replay.sync(payload=payload, output_ids=[100, 101])
    assert next_tracker is tracker
    assert next_tracker.steer_token_count == 2
    assert replay.observed_len == 2


def test_incremental_replay_resets_when_payload_changes() -> None:
    tokens = GrammarTokenIds(
        think_open=1,
        think_close=2,
        steer_open=3,
        steer_close=4,
        exec_open=5,
        exec_close=6,
        newline=7,
    )
    payload = {"prefix_output_token_ids": [1, 7, 3], "max_steer_tokens": 30}
    replay = IncrementalGrammarReplay(payload=payload, tokens=tokens)
    replay.sync(payload=payload, output_ids=[100, 101])

    changed_payload = {
        "prefix_output_token_ids": [1, 7, 3],
        "forced_output_start_index": 2,
        "forced_output_token_ids": [102],
        "max_steer_tokens": 30,
    }
    tracker = replay.sync(payload=changed_payload, output_ids=[100, 101])
    assert tracker.state == GrammarState.IN_STEER
    assert tracker.steer_token_count == 2
    assert replay.payload_identity == id(changed_payload)


def test_forced_open_token_requires_think_newline_prefix() -> None:
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
    assert forced_open_token(tracker=tracker) == tokens.think_open
    tracker.observe(token_id=tokens.think_open)
    assert forced_open_token(tracker=tracker) == tokens.newline
    assert tokens.newline is not None
    tracker.observe(token_id=tokens.newline)
    assert forced_open_token(tracker=tracker) is None
    assert choice_token_ids(tracker=tracker) == {
        tokens.steer_open,
        tokens.think_close,
    }


def test_forced_open_token_requires_newlines_between_blocks() -> None:
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
    assert tokens.newline is not None
    tracker.observe_many(
        [
            tokens.think_open,
            tokens.newline,
            tokens.steer_open,
            100,
            tokens.steer_close,
        ]
    )
    assert forced_open_token(tracker=tracker) == tokens.newline
    tracker.observe(token_id=tokens.newline)
    assert forced_open_token(tracker=tracker) == tokens.exec_open

    tracker.observe_many([tokens.exec_open, 101, tokens.exec_close])
    assert forced_open_token(tracker=tracker) == tokens.newline
    tracker.observe(token_id=tokens.newline)
    assert forced_open_token(tracker=tracker) is None
    assert choice_token_ids(tracker=tracker) == {
        tokens.steer_open,
        tokens.think_close,
    }


def test_forced_script_token_uses_output_start_index() -> None:
    tokens = GrammarTokenIds(
        think_open=1,
        think_close=2,
        steer_open=3,
        steer_close=4,
        exec_open=5,
        exec_close=6,
    )
    payload = {
        "forced_output_start_index": 3,
        "forced_output_token_ids": [101, 102],
    }
    assert (
        forced_script_token(payload=payload, output_ids=[1, 2, 6], tokens=tokens) == 101
    )
    assert (
        forced_script_token(payload=payload, output_ids=[1, 2, 6, 101], tokens=tokens)
        == 102
    )
    assert (
        forced_script_token(
            payload=payload, output_ids=[1, 2, 6, 101, 102], tokens=tokens
        )
        is None
    )


def test_forced_script_token_activates_followup_after_exec_close() -> None:
    tokens = GrammarTokenIds(
        think_open=1,
        think_close=2,
        steer_open=3,
        steer_close=4,
        exec_open=5,
        exec_close=6,
    )
    payload = {
        "forced_output_start_index": 0,
        "forced_output_token_ids": [3, 88, 4, 5],
        "forced_after_exec_close_token_ids": [3, 77, 4, 5, 66],
        "forced_after_exec_close_trigger_after": 4,
    }
    assert (
        forced_script_token(payload=payload, output_ids=[3, 88, 4, 5], tokens=tokens)
        is None
    )
    assert (
        forced_script_token(
            payload=payload,
            output_ids=[3, 88, 4, 5, 9, 9, 6],
            tokens=tokens,
        )
        == 3
    )
    assert (
        forced_script_token(
            payload=payload,
            output_ids=[3, 88, 4, 5, 9, 9, 6, 3],
            tokens=tokens,
        )
        == 77
    )


def test_empty_block_close_token_blocks_empty_steer_and_exec() -> None:
    """Processor should suppress close tokens for empty steer/exec blocks."""

    tokens = GrammarTokenIds(
        think_open=1,
        think_close=2,
        steer_open=3,
        steer_close=4,
        exec_open=5,
        exec_close=6,
    )
    steer_tracker = GrammarTracker(tokens=tokens)
    steer_tracker.observe_many([1, 3])
    assert empty_block_close_token(tracker=steer_tracker) == tokens.steer_close

    exec_tracker = GrammarTracker(tokens=tokens)
    exec_tracker.observe_many([1, 3, 100, 4, 5])
    assert empty_block_close_token(tracker=exec_tracker) == tokens.exec_close


def test_empty_block_close_token_allows_newline_as_non_tag_content() -> None:
    """A newline inside a block counts as a non-tag content token."""

    tokens = GrammarTokenIds(
        think_open=1,
        think_close=2,
        steer_open=3,
        steer_close=4,
        exec_open=5,
        exec_close=6,
        newline=7,
    )
    steer_tracker = GrammarTracker(tokens=tokens)
    steer_tracker.observe_many([1, 7, 3, 7])
    assert empty_block_close_token(tracker=steer_tracker) is None

    exec_tracker = GrammarTracker(tokens=tokens)
    exec_tracker.observe_many([1, 7, 3, 100, 4, 7, 5, 7])
    assert empty_block_close_token(tracker=exec_tracker) is None


def test_empty_block_close_token_allows_content_after_layout() -> None:
    """A block can close after at least one non-layout token."""

    tokens = GrammarTokenIds(
        think_open=1,
        think_close=2,
        steer_open=3,
        steer_close=4,
        exec_open=5,
        exec_close=6,
        newline=7,
    )
    steer_tracker = GrammarTracker(tokens=tokens)
    steer_tracker.observe_many([1, 7, 3, 7, 100])
    assert empty_block_close_token(tracker=steer_tracker) is None

    exec_tracker = GrammarTracker(tokens=tokens)
    exec_tracker.observe_many([1, 7, 3, 100, 4, 7, 5, 7, 101])
    assert empty_block_close_token(tracker=exec_tracker) is None


def test_invalid_control_tokens_are_masked_inside_blocks() -> None:
    """Other control tags cannot appear as steer/exec content."""

    tokens = GrammarTokenIds(
        think_open=1,
        think_close=2,
        steer_open=3,
        steer_close=4,
        exec_open=5,
        exec_close=6,
    )
    steer_tracker = GrammarTracker(tokens=tokens)
    steer_tracker.observe_many([1, 3, 100])
    assert invalid_control_token_ids(tracker=steer_tracker) == {1, 2, 3, 5, 6}

    exec_tracker = GrammarTracker(tokens=tokens)
    exec_tracker.observe_many([1, 3, 100, 4, 5, 101])
    assert invalid_control_token_ids(tracker=exec_tracker) == {1, 2, 3, 4, 5}

    final_tracker = GrammarTracker(tokens=tokens)
    final_tracker.observe_many([1, 3, 100, 4, 5, 101, 6, 2])
    assert invalid_control_token_ids(tracker=final_tracker) == {1, 2, 3, 4, 5, 6}


def test_eos_is_masked_until_after_final_answer_content() -> None:
    tokens = GrammarTokenIds(
        think_open=1,
        think_close=2,
        steer_open=3,
        steer_close=4,
        exec_open=5,
        exec_close=6,
        eos=7,
    )

    steer_tracker = GrammarTracker(tokens=tokens)
    steer_tracker.observe_many([1, 3, 100])
    assert tokens.eos in invalid_control_token_ids(tracker=steer_tracker)

    exec_tracker = GrammarTracker(tokens=tokens)
    exec_tracker.observe_many([1, 3, 100, 4, 5, 101])
    assert tokens.eos in invalid_control_token_ids(tracker=exec_tracker)

    final_tracker = GrammarTracker(tokens=tokens)
    final_tracker.observe_many([1, 3, 100, 4, 5, 101, 6, 2])
    assert tokens.eos in invalid_control_token_ids(tracker=final_tracker)

    final_tracker.observe(99)
    assert tokens.eos not in invalid_control_token_ids(tracker=final_tracker)
