"""Tests for raw IM prompt templating behavior."""

from __future__ import annotations

from chat_templating import build_raw_im_prompt


def test_im_start_im_end_raw_template_builder() -> None:
    """Raw IM template helper should stitch user and assistant prefixes."""
    prompt = build_raw_im_prompt(prompt="2+2?", assistant_prefix="<think>")
    assert prompt.startswith("<|im_start|>user\n2+2?<|im_end|>")
    assert prompt.endswith("<|im_start|>assistant\n<think>")
