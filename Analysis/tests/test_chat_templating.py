"""Tests for chat templating behavior and mode fallback decisions."""

from __future__ import annotations

from analysis_types import TemplateConfig
from chat_templating import (
    build_chat_messages,
    build_chat_template_fields,
    build_raw_im_prompt,
    format_chat_content,
    resolve_mode_after_error,
)


def test_chat_templating_add_generation_prompt_flag() -> None:
    """Chat payload fields should preserve add-generation-prompt values."""
    fields_true = build_chat_template_fields(
        template_config=TemplateConfig(
            add_generation_prompt=True, continue_final_message=False
        )
    )
    fields_false = build_chat_template_fields(
        template_config=TemplateConfig(
            add_generation_prompt=False, continue_final_message=False
        )
    )
    assert fields_true["add_generation_prompt"] is True
    assert fields_false["add_generation_prompt"] is False


def test_chat_templating_continue_final_message_flag() -> None:
    """Chat payload fields should preserve continue-final-message values."""
    fields = build_chat_template_fields(
        template_config=TemplateConfig(
            add_generation_prompt=False, continue_final_message=True
        )
    )
    assert fields["continue_final_message"] is True


def test_chat_templating_custom_template_kwargs_passthrough() -> None:
    """Custom chat template kwargs should pass through unchanged."""
    config = TemplateConfig(
        add_generation_prompt=False,
        continue_final_message=True,
        chat_template_kwargs={"foo": "bar", "temperature_tag": "t"},
    )
    fields = build_chat_template_fields(template_config=config)
    assert fields["chat_template_kwargs"] == {"foo": "bar", "temperature_tag": "t"}


def test_chat_mode_vs_completions_mode_equivalent_prefix() -> None:
    """Chat and completions representations should contain same assistant prefix."""
    prompt = "Find all a"
    assistant_prefix = "<think>"
    raw_prompt = build_raw_im_prompt(prompt=prompt, assistant_prefix=assistant_prefix)
    messages = build_chat_messages(
        prompt=prompt,
        assistant_prefix=assistant_prefix,
        content_format="string",
    )
    assert raw_prompt.endswith(assistant_prefix)
    assert messages[-1]["role"] == "assistant"
    assert messages[-1]["content"] == assistant_prefix


def test_im_start_im_end_raw_template_builder() -> None:
    """Raw prompt builder should emit canonical `<|im_start|>` format."""
    prompt = "hello"
    generated = build_raw_im_prompt(prompt=prompt, assistant_prefix="pref")
    assert generated.startswith(
        "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n"
    )
    assert generated.endswith("pref")


def test_chat_template_content_format_string_vs_openai() -> None:
    """Content formatter should support string and OpenAI object formats."""
    assert format_chat_content(content="hi", content_format="string") == "hi"
    assert format_chat_content(content="hi", content_format="openai") == [
        {"type": "text", "text": "hi"}
    ]


def test_missing_chat_template_fallback_strategy() -> None:
    """Chat template failures should fallback only when enabled."""
    fallback_mode = resolve_mode_after_error(
        preferred_mode="chat",
        allow_fallback=True,
        had_chat_template_error=True,
    )
    no_fallback_mode = resolve_mode_after_error(
        preferred_mode="chat",
        allow_fallback=False,
        had_chat_template_error=True,
    )
    assert fallback_mode == "completions"
    assert no_fallback_mode == "chat"
