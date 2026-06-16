"""Tests for streaming repetition parser edge cases."""

from __future__ import annotations

from branching_eval.branch_decode_utils import (
    initialize_exec_repetition_state,
    initialize_steer_repetition_state,
    normalize_steer_block_text,
    update_exec_repetition_state,
    update_steer_repetition_state,
)


def test_exec_repetition_tracks_split_open_and_close_tags() -> None:
    """Exec repeat detection should survive tag fragments split across chunks."""

    state = initialize_exec_repetition_state(text="")
    similarity_ratio: float | None = None
    for chunk in ("<ex", "ec>Repeat block</ex", "ec>"):
        state, similarity_ratio = update_exec_repetition_state(
            state=state,
            chunk_text=chunk,
        )
    assert similarity_ratio is None
    assert state.previous_exec_block == "repeat block"
    assert state.repeated_exec_blocks == 1

    for chunk in ("<exec>Repeat block</ex", "ec>"):
        state, similarity_ratio = update_exec_repetition_state(
            state=state,
            chunk_text=chunk,
        )

    assert similarity_ratio == 1.0
    assert state.previous_exec_block == "repeat block"
    assert state.repeated_exec_blocks == 2


def test_exec_repetition_excludes_split_close_tag_from_block_text() -> None:
    """A split close tag should not be buffered as exec-block content."""

    state = initialize_exec_repetition_state(text="<exec>")
    state, similarity_ratio = update_exec_repetition_state(
        state=state,
        chunk_text="Repeated calculation</ex",
    )
    assert similarity_ratio is None
    assert state.in_exec_block
    assert state.pending_exec_text == "Repeated calculation"

    state, similarity_ratio = update_exec_repetition_state(
        state=state,
        chunk_text="ec>",
    )

    assert similarity_ratio is None
    assert not state.in_exec_block
    assert state.previous_exec_block == "repeated calculation"


def test_steer_repetition_tracks_split_open_tag() -> None:
    """Steer repeat detection should also carry partial open tags."""

    state = initialize_steer_repetition_state(text="")
    similarity_ratio: float | None = None
    for chunk in ("<s", "teer>Final check</steer>"):
        state, similarity_ratio = update_steer_repetition_state(
            state=state,
            chunk_text=chunk,
        )
    assert similarity_ratio is None
    assert state.previous_exec_block == "final check"
    assert state.repeated_exec_blocks == 1

    for chunk in ("<st", "eer>Final check</steer>"):
        state, similarity_ratio = update_steer_repetition_state(
            state=state,
            chunk_text=chunk,
        )

    assert similarity_ratio == 1.0
    assert state.previous_exec_block == "final check"
    assert state.repeated_exec_blocks == 2


def test_steer_repetition_normalizes_edge_control_fragments() -> None:
    """Steer repeat comparison should ignore edge tags and partial tags."""

    assert (
        normalize_steer_block_text(text="<steer>Conclude k=4 failure</st")
        == "conclude k=4 failure"
    )
    state = initialize_steer_repetition_state(
        text="<steer>Conclude k=4 failure</steer>",
        similarity_threshold=0.92,
    )
    state, similarity_ratio = update_steer_repetition_state(
        state=state,
        chunk_text="<steer><steer>Conclude k=4 failure</st</steer>",
        similarity_threshold=0.92,
    )

    assert similarity_ratio == 1.0
    assert state.previous_exec_block == "conclude k=4 failure"
    assert state.repeated_exec_blocks == 2
