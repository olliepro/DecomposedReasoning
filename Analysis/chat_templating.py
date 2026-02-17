"""Prompt templating helpers for vLLM chat and completions requests."""

from __future__ import annotations

from typing import Any

from analysis_types import ApiMode, TemplateConfig


def build_user_messages(
    *, prompt: str, content_format: str = "string"
) -> list[dict[str, Any]]:
    """Build one-user-turn message list.

    Args:
        prompt: User prompt text.
        content_format: `string` or `openai` content schema.

    Returns:
        List with one user message.

    Example:
        >>> build_user_messages(prompt="hi")
        [{'role': 'user', 'content': 'hi'}]
    """
    content = format_chat_content(content=prompt, content_format=content_format)
    return [{"role": "user", "content": content}]


def format_chat_content(*, content: str, content_format: str) -> Any:
    """Format message content for selected chat API content schema.

    Args:
        content: Message text.
        content_format: `string` or `openai` style content objects.

    Returns:
        Content payload for one chat message.
    """
    assert content_format in {"string", "openai"}, "unsupported content_format"
    if content_format == "string":
        return content
    return [{"type": "text", "text": content}]


def build_chat_messages(
    *,
    prompt: str,
    assistant_prefix: str,
    content_format: str,
) -> list[dict[str, Any]]:
    """Build chat messages with optional assistant prefix continuation.

    Args:
        prompt: User prompt text.
        assistant_prefix: Existing assistant text used as continuation prefix.
        content_format: `string` or `openai` content schema.

    Returns:
        Chat message list.
    """
    messages = build_user_messages(prompt=prompt, content_format=content_format)
    if not assistant_prefix:
        return messages
    assistant_content = format_chat_content(
        content=assistant_prefix,
        content_format=content_format,
    )
    messages.append({"role": "assistant", "content": assistant_content})
    return messages


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


def build_chat_template_fields(*, template_config: TemplateConfig) -> dict[str, Any]:
    """Build optional chat-template request fields for chat completions.

    Args:
        template_config: Template settings.

    Returns:
        Mapping of chat-template-related request fields.
    """
    fields: dict[str, Any] = {
        "add_generation_prompt": template_config.add_generation_prompt,
        "continue_final_message": template_config.continue_final_message,
    }
    if template_config.chat_template is not None:
        fields["chat_template"] = template_config.chat_template
    if template_config.chat_template_kwargs:
        fields["chat_template_kwargs"] = dict(template_config.chat_template_kwargs)
    return fields


def resolve_mode_after_error(
    *,
    preferred_mode: ApiMode,
    allow_fallback: bool,
    had_chat_template_error: bool,
) -> ApiMode:
    """Resolve active request mode after chat template failures.

    Args:
        preferred_mode: Preferred request mode.
        allow_fallback: Enables fallback behavior.
        had_chat_template_error: Whether last error was chat-template related.

    Returns:
        Effective request mode.
    """
    if preferred_mode != "chat":
        return preferred_mode
    if not had_chat_template_error:
        return preferred_mode
    if not allow_fallback:
        return preferred_mode
    return "completions"
