"""Tests for structural tag scanning behavior."""

from __future__ import annotations

from tag_scanner import (
    choose_next_event,
    find_branch_event,
    find_think_end_event,
    trim_after_first_stop,
)


def test_detect_branch_and_think_events() -> None:
    """Scanner should find both branch and think-end events."""
    text = "<think>\n<steer>x</steer>\n<exec>a</exec>\n</think>"
    branch_event = find_branch_event(
        text=text, start_index=0, boundary_pattern=r"<steer\b[^>]*>"
    )
    think_event = find_think_end_event(text=text, start_index=0)
    earliest = choose_next_event(branch_event=branch_event, think_end_event=think_event)
    assert branch_event is not None
    assert think_event is not None
    assert earliest is not None
    assert earliest.event_type == "branch"


def test_detect_branch_event_with_execute_tags() -> None:
    """Branch detection should trigger at first steer-open regardless exec flavor."""

    text = "<execute>a</execute>\n<steer>x</steer></think>"
    branch_event = find_branch_event(
        text=text,
        start_index=0,
        boundary_pattern=r"<steer\b[^>]*>",
    )
    assert branch_event is not None
    assert branch_event.event_type == "branch"


def test_stop_sequence_boundary_trim_behavior() -> None:
    """Stop-trimming should keep content before earliest marker."""
    text = "abc</steer>def</think>"
    trimmed = trim_after_first_stop(text=text, stop_markers=("</think>", "</steer>"))
    assert trimmed == "abc"
