"""Prompt templating helpers for vLLM completions requests."""

from __future__ import annotations


def build_raw_im_prompt(*, prompt: str, assistant_prefix: str = "") -> str:
    """Build raw `<|im_start|>` prompt for completions mode.

    Args:
        prompt: User prompt text.
        assistant_prefix: Existing assistant text continuation prefix.

    Returns:
        Prompt string for `/v1/completions`.

    Example:
        >>> build_raw_im_prompt(prompt="2+2?", assistant_prefix="<think>")
        '<|im_start|>user\n2+2?<|im_end|>\n<|im_start|>assistant\n<think>'
    """
    parts = [
        "<|im_start|>user\n",
        prompt,
        "<|im_end|>\n",
        "<|im_start|>assistant\n",
        assistant_prefix,
    ]
    return "".join(parts)
