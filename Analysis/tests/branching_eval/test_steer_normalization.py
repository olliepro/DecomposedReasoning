"""Tests for legacy-parity steer boundary normalization helpers."""

from __future__ import annotations

from branching_eval.steer_normalization import (
    explicit_exec_stop_completion_suffix,
    forced_boundary_suffix,
    has_valid_steer_entry_boundary,
    exec_choice_boundary_suffix,
    inferred_exec_to_steer_separator,
    is_initial_steer_decision_boundary,
    is_inside_open_exec,
    is_steer_decision_boundary,
    normalize_steer_exec_chunk_text,
    normalize_steer_boundary_text,
    selected_candidate_normalization_suffix,
    steer_candidate_stop_markers,
)


def test_detects_open_exec_state() -> None:
    """Open `<exec>` blocks should be detected correctly."""

    assert is_inside_open_exec(text="<exec>abc")
    assert not is_inside_open_exec(text="<exec>abc</exec>")


def test_uses_single_exec_to_steer_separator() -> None:
    """Exec-to-steer insertion should use one newline."""

    text = "x</exec>\n<steer>y</steer></exec>\t<steer>"
    assert inferred_exec_to_steer_separator(text=text) == "\n"


def test_normalize_keeps_valid_exec_boundary() -> None:
    """Already-valid boundaries should remain unchanged."""

    text = "prefix</exec>\n<steer>"
    assert normalize_steer_boundary_text(text=text) == text


def test_normalize_closes_open_exec_without_forcing_steer() -> None:
    """Open exec contexts should stop at exec boundary when no steer prefix exists."""

    assert normalize_steer_boundary_text(text="<exec>work") == "<exec>work\n</exec>\n"
    assert normalize_steer_boundary_text(text="<exec>work\n") == "<exec>work\n</exec>\n"


def test_normalize_completes_partial_steer_prefix_inside_exec() -> None:
    """A 2+ char `<steer>` prefix should preserve the old steer boundary."""

    for partial in ("<s", "<st", "<ste", "<stee", "<steer", "<steer>"):
        normalized_text = normalize_steer_boundary_text(text=f"<exec>work{partial}")
        assert normalized_text.endswith("</steer></exec>\n<steer>")
        assert has_valid_steer_entry_boundary(text=normalized_text)


def test_normalize_closes_open_exec_before_steer() -> None:
    """Open exec contexts should be closed before steer branching."""

    text = "<exec>work<steer"
    normalized_text = normalize_steer_boundary_text(text=text)
    assert normalized_text.endswith("</steer></exec>\n<steer>")
    assert has_valid_steer_entry_boundary(text=normalized_text)


def test_normalize_repairs_invalid_non_exec_steer_entry() -> None:
    """Non-exec prefixes ending in `<steer>` should be repaired to valid entry."""

    normalized_text = normalize_steer_boundary_text(text="plain text")
    assert normalized_text.endswith("</steer><exec></exec>\n<steer>")
    assert has_valid_steer_entry_boundary(text=normalized_text)


def test_forced_boundary_suffix_respects_exec_context() -> None:
    """Forced boundary suffix should close exec blocks when needed."""

    assert forced_boundary_suffix(text="<exec>abc") == "\n</exec>\n"
    assert forced_boundary_suffix(text="<exec>abc</exec>") == "\n"
    assert forced_boundary_suffix(text="<exec>abc</exec>\n") == ""
    assert forced_boundary_suffix(text="<exec>abc</exec>\n\n") == ""
    assert forced_boundary_suffix(text="<exec>abc\n") == "</exec>\n"
    assert forced_boundary_suffix(text="<exec>abc<s").endswith(
        "teer></steer></exec>\n<steer>"
    )
    assert forced_boundary_suffix(text="plain") == "<steer>"


def test_exec_choice_boundary_suffix_is_append_only() -> None:
    """Exec boundary helper should only return missing newlines."""

    assert exec_choice_boundary_suffix(text="x</exec>") == "\n"
    assert exec_choice_boundary_suffix(text="x</exec>\n") == ""
    assert exec_choice_boundary_suffix(text="x</exec>\n\n") == ""


def test_steer_decision_boundary_detection_is_narrow() -> None:
    """Only the initial think-open and post-exec points are steer choices."""

    assert is_initial_steer_decision_boundary(text="<think>\n")
    assert not is_initial_steer_decision_boundary(text="<think>\nWork")
    assert is_steer_decision_boundary(text="<think>\n")
    assert is_steer_decision_boundary(text="<exec>Work</exec>\n")
    assert not is_steer_decision_boundary(text="<exec>Work")
    assert not is_steer_decision_boundary(text="<think>\n<steer>")


def test_explicit_exec_stop_completion_uses_minimal_suffix() -> None:
    """Explicit stop completion should not duplicate partial close tags."""

    assert explicit_exec_stop_completion_suffix(text="x") == "</exec>"
    for prefix_length in range(1, len("</exec>")):
        partial_close = "</exec>"[:prefix_length]
        expected_suffix = "</exec>"[prefix_length:]
        assert (
            explicit_exec_stop_completion_suffix(text=f"x{partial_close}")
            == expected_suffix
        )
    assert explicit_exec_stop_completion_suffix(text="x</exec>") == ""


def test_steer_candidate_stop_markers_allow_terminal_after_exec_boundary() -> None:
    """Candidate generation may stop at think close only outside an open steer tag."""

    assert steer_candidate_stop_markers(text="x</exec>\n") == (
        "</steer",
        "</think>",
    )
    assert steer_candidate_stop_markers(text="x</exec>\n<steer>") == ("</steer",)


def test_selected_candidate_normalization_adds_exec_open() -> None:
    """Selected steer candidate should normalize to close-steer then open-exec."""

    suffix, mode = selected_candidate_normalization_suffix(text="candidate</steer>")
    assert suffix == "\n<exec>"
    assert mode == "already_closed"


def test_selected_candidate_normalization_repairs_partial_close_first() -> None:
    """Partial close tags should be completed before appending `<exec>` open."""

    suffix, mode = selected_candidate_normalization_suffix(text="candidate</st")
    assert suffix == "eer>\n<exec>"
    assert mode == "partial_completed"


def test_selected_candidate_normalization_repairs_missing_close_bracket() -> None:
    """`</steer` stop text should normalize to a fully closed steer block."""

    suffix, mode = selected_candidate_normalization_suffix(text="candidate</steer")
    assert suffix == ">\n<exec>"
    assert mode == "partial_completed"


def test_selected_candidate_normalization_preserves_terminal_think_close() -> None:
    """Terminal `</think>` candidates should not receive steer/exec suffixes."""

    suffix, mode = selected_candidate_normalization_suffix(text="</think>")
    assert suffix == ""
    assert mode == "think_closed"


def test_decode_chunk_normalization_canonicalizes_exec_boundary() -> None:
    """Decode chunk normalization should canonicalize steer/exec separators."""

    text = "a</steer>\n\n<exec>\nwork\n</exec>\n<steer>"
    normalized = normalize_steer_exec_chunk_text(text=text)
    assert normalized == "a</steer>\n<exec>work\n</exec>\n<steer>"


def test_decode_chunk_normalization_repairs_partial_trailing_tag() -> None:
    """Decode chunk normalization should complete trailing partial steer tags."""

    text = "prefix</exec>\n<steer"
    normalized = normalize_steer_exec_chunk_text(text=text)
    assert normalized.endswith("<steer>")
