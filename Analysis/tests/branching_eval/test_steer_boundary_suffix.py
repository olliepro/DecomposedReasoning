"""Tests for steer-boundary suffix validation semantics."""

from __future__ import annotations

import pytest

from branching_eval.branch_decode_utils import assert_no_text_after_first_steer_close


def test_steer_boundary_allows_whitespace_suffix() -> None:
    """Whitespace-only suffix after `</steer>` should be accepted."""

    assert_no_text_after_first_steer_close(text="abc</steer>\n")
    assert_no_text_after_first_steer_close(text="abc</steer>\n\n\t")


def test_steer_boundary_rejects_non_whitespace_suffix() -> None:
    """Non-whitespace suffix after `</steer>` should hard-fail."""

    with pytest.raises(AssertionError, match="unexpected text after first </steer>"):
        assert_no_text_after_first_steer_close(text="abc</steer>tail")
