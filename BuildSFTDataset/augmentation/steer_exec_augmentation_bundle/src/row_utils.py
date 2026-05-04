from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import tiktoken

from .trace_augmentor import TraceBlock


THINK_OPEN = "<think>"
STEER_OPEN = "<steer>"
STEER_CLOSE = "</steer>"


def capitalize_first_word(text: str) -> str:
    """Uppercase the first alphabetical character in the first word.

    Args:
        text: Steer text to normalize.

    Returns:
        Text with the first word capitalized when possible.
    """

    chars = list(text)
    seen_non_whitespace = False
    for index, char in enumerate(chars):
        if not char.isspace():
            seen_non_whitespace = True
        if seen_non_whitespace and char.isalpha():
            chars[index] = char.upper()
            return "".join(chars)
    return text


def normalize_steer_blocks(blocks: Sequence[TraceBlock]) -> list[TraceBlock]:
    """Return blocks with steer texts normalized to capitalized first words.

    Args:
        blocks: Interleaved trace blocks.

    Returns:
        Normalized blocks.
    """

    normalized_blocks: list[TraceBlock] = []
    for block in blocks:
        if block.type == "steer":
            normalized_blocks.append(
                TraceBlock(type=block.type, text=capitalize_first_word(block.text))
            )
            continue
        normalized_blocks.append(block)
    return normalized_blocks


def split_assistant_content(content: str) -> tuple[str, str, str]:
    """Split assistant content into prefix, think body, and suffix.

    Args:
        content: Assistant message content containing one `<think>` block.

    Returns:
        Tuple of prefix text, inner think text, and suffix text.
    """

    assert content.count("<think>") == 1, "Expected one opening <think> tag."
    assert content.count("</think>") == 1, "Expected one closing </think> tag."
    prefix, remainder = content.split("<think>", 1)
    think_text, suffix = remainder.split("</think>", 1)
    return prefix, think_text, suffix


def split_think_padding(think_text: str) -> tuple[str, str]:
    """Split raw `<think>` content into leading and trailing whitespace.

    Args:
        think_text: Raw inner text between `<think>` and `</think>`.

    Returns:
        Leading and trailing whitespace substrings.
    """

    start = 0
    end = len(think_text)
    while start < end and think_text[start].isspace():
        start += 1
    while end > start and think_text[end - 1].isspace():
        end -= 1
    return think_text[:start], think_text[end:]


def render_pair_section(*, steer_text: str, exec_text: str) -> str:
    """Render one `<steer>/<exec>` pair using dataset whitespace conventions.

    Args:
        steer_text: Steer text for the pair.
        exec_text: Exec text for the pair.

    Returns:
        Rendered pair section.
    """

    return f"{STEER_OPEN}{steer_text}{STEER_CLOSE}\n<exec>\n{exec_text}\n</exec>"


def render_dataset_trace(blocks: Sequence[TraceBlock]) -> str:
    """Render blocks with the whitespace pattern used in transformed rows.

    Args:
        blocks: Interleaved steer/exec blocks.

    Returns:
        Canonically formatted inner `<think>` content.
    """

    normalized_blocks = normalize_steer_blocks(blocks=blocks)
    assert len(normalized_blocks) % 2 == 0, "Expected an even block count."
    sections: list[str] = []
    for pair_start in range(0, len(normalized_blocks), 2):
        steer_block = normalized_blocks[pair_start]
        exec_block = normalized_blocks[pair_start + 1]
        sections.append(
            render_pair_section(
                steer_text=steer_block.text,
                exec_text=exec_block.text,
            )
        )
    return "\n\n".join(sections)


def compute_steer_mask_char_ranges(
    *,
    prefix: str,
    leading_think_whitespace: str,
    blocks: Sequence[TraceBlock],
    final_trace_block_indexes: Sequence[int],
) -> list[tuple[int, int]]:
    """Compute exact assistant-content char spans for inserted steer payloads.

    The returned spans start immediately after `<steer>` and end after
    `</steer>`, so the opening tag is preserved while the steer text and closing
    tag are maskable.

    Args:
        prefix: Assistant content prefix before `<think>`.
        leading_think_whitespace: Whitespace preserved after `<think>`.
        blocks: Final augmented trace blocks.
        final_trace_block_indexes: Final block indexes of inserted steer blocks.

    Returns:
        Inclusive-start, exclusive-end char spans in `messages[-1].content`.
    """

    normalized_blocks = normalize_steer_blocks(blocks=blocks)
    target_indexes = set(final_trace_block_indexes)
    think_offset = 0
    prefix_offset = len(prefix) + len(THINK_OPEN) + len(leading_think_whitespace)
    char_ranges: list[tuple[int, int]] = []
    for pair_start in range(0, len(normalized_blocks), 2):
        if pair_start > 0:
            think_offset += 2
        steer_block = normalized_blocks[pair_start]
        exec_block = normalized_blocks[pair_start + 1]
        if pair_start in target_indexes:
            start = prefix_offset + think_offset + len(STEER_OPEN)
            end = start + len(steer_block.text) + len(STEER_CLOSE)
            char_ranges.append((start, end))
        think_offset += len(
            render_pair_section(
                steer_text=steer_block.text,
                exec_text=exec_block.text,
            )
        )
    return char_ranges


def render_assistant_content(
    *,
    prefix: str,
    leading_think_whitespace: str,
    blocks: Sequence[TraceBlock],
    trailing_think_whitespace: str,
    suffix: str,
) -> str:
    """Render full assistant content while preserving prefix and suffix.

    Args:
        prefix: Text before `<think>`.
        leading_think_whitespace: Whitespace preserved after `<think>`.
        blocks: Interleaved steer/exec blocks.
        trailing_think_whitespace: Whitespace preserved before `</think>`.
        suffix: Text after `</think>`.

    Returns:
        Updated assistant content.
    """

    think_text = render_dataset_trace(blocks=blocks)
    return (
        f"{prefix}<think>{leading_think_whitespace}{think_text}"
        f"{trailing_think_whitespace}</think>{suffix}"
    )


def replace_last_assistant_content(
    *, messages: Sequence[Mapping[str, Any]], new_content: str
) -> list[dict[str, Any]]:
    """Replace the last assistant message content in a messages list.

    Args:
        messages: Conversation messages.
        new_content: Replacement assistant content.

    Returns:
        Updated messages list.
    """

    updated_messages = [dict(message) for message in messages]
    for index in range(len(updated_messages) - 1, -1, -1):
        if updated_messages[index].get("role") == "assistant":
            updated_messages[index]["content"] = new_content
            return updated_messages
    raise ValueError("Could not find an assistant message to replace.")


def compute_think_token_count(
    *, encoding: tiktoken.Encoding, assistant_content: str
) -> int:
    """Count tokens in the inner `<think>` block for one assistant message.

    Args:
        encoding: Tokenizer encoding.
        assistant_content: Assistant message containing a `<think>` block.

    Returns:
        Think token count.
    """

    _, think_text, _ = split_assistant_content(content=assistant_content)
    return len(encoding.encode(think_text))


def messages_to_prompt_text(messages: Sequence[Mapping[str, Any]]) -> str:
    """Serialize row messages into the full-row token-count format.

    Args:
        messages: Conversation messages.

    Returns:
        Concatenated `role: content` transcript.
    """

    parts: list[str] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, str):
            continue
        role = str(message.get("role", ""))
        parts.append(f"{role}: {content}")
    return "\n".join(parts)


def compute_complete_prompt_token_count(
    *, encoding: tiktoken.Encoding, messages: Sequence[Mapping[str, Any]]
) -> int:
    """Count full-row message tokens from concatenated `role: content` text.

    Args:
        encoding: Tokenizer encoding.
        messages: Conversation messages.

    Returns:
        Full-row token count.
    """

    return len(encoding.encode(messages_to_prompt_text(messages=messages)))
