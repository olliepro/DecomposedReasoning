"""Helpers for masking non-sequitur steer spans during assistant-only SFT."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence, cast
from uuid import uuid4

from transformers import PreTrainedTokenizerBase


@dataclass(frozen=True)
class CharSpan:
    """Half-open character span over a source string.

    Args:
        start: Inclusive starting character index.
        end: Exclusive ending character index.
    """

    start: int
    end: int

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> CharSpan:
        """Build one span from a serialized mapping.

        Args:
            payload: Serialized span record with `start` and `end`.

        Returns:
            Parsed `CharSpan`.
        """
        return cls(start=int(payload["start"]), end=int(payload["end"]))

    def overlaps(self, other: CharSpan) -> bool:
        """Return whether two half-open spans overlap.

        Args:
            other: Span to compare against.

        Returns:
            True when the spans share any covered character.
        """
        return self.end > other.start and self.start < other.end

    def shift(self, offset: int) -> CharSpan:
        """Translate the span by a fixed character offset.

        Args:
            offset: Signed offset added to both boundaries.

        Returns:
            Shifted `CharSpan`.
        """
        return CharSpan(start=self.start + offset, end=self.end + offset)

    def to_record(self) -> dict[str, int]:
        """Serialize the span for dataset rows.

        Returns:
            JSON-serializable span mapping.
        """
        return {"start": self.start, "end": self.end}


@dataclass(frozen=True)
class MaskTargets:
    """Masking metadata derived from inserted non-sequitur steer spans.

    Args:
        final_string_reference: Source string descriptor for the char spans.
        final_string_char_ranges: Message-local char spans to suppress.
        strategy: High-level masking strategy label.
        generated_steer_texts: Inserted steer texts recorded for auditing.
        final_trace_block_indexes: Final trace block indexes for inserted steers.
        final_trace_pair_indexes: Final trace pair indexes for inserted steers.
    """

    final_string_reference: str
    final_string_char_ranges: tuple[CharSpan, ...]
    strategy: str = "mask_inserted_steer_blocks"
    generated_steer_texts: tuple[str, ...] = ()
    final_trace_block_indexes: tuple[int, ...] = ()
    final_trace_pair_indexes: tuple[int, ...] = ()

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> MaskTargets:
        """Build masking metadata from a serialized mapping.

        Args:
            payload: Serialized `mask_targets` dictionary.

        Returns:
            Parsed `MaskTargets`.
        """
        char_ranges = tuple(
            CharSpan.from_mapping(payload=cast(Mapping[str, Any], item))
            for item in cast(
                Sequence[Mapping[str, Any]], payload["final_string_char_ranges"]
            )
        )
        return cls(
            final_string_reference=str(payload["final_string_reference"]),
            final_string_char_ranges=char_ranges,
            strategy=str(payload.get("strategy", "mask_inserted_steer_blocks")),
            generated_steer_texts=tuple(
                str(text)
                for text in cast(
                    Sequence[Any], payload.get("generated_steer_texts", ())
                )
            ),
            final_trace_block_indexes=tuple(
                int(index)
                for index in cast(
                    Sequence[Any], payload.get("final_trace_block_indexes", ())
                )
            ),
            final_trace_pair_indexes=tuple(
                int(index)
                for index in cast(
                    Sequence[Any], payload.get("final_trace_pair_indexes", ())
                )
            ),
        )

    def to_record(self) -> dict[str, object]:
        """Serialize masking metadata for intermediate dataset rows.

        Returns:
            JSON-serializable `mask_targets` payload.
        """
        return {
            "strategy": self.strategy,
            "final_string_reference": self.final_string_reference,
            "final_string_char_ranges": [
                span.to_record() for span in self.final_string_char_ranges
            ],
            "generated_steer_texts": list(self.generated_steer_texts),
            "final_trace_block_indexes": list(self.final_trace_block_indexes),
            "final_trace_pair_indexes": list(self.final_trace_pair_indexes),
        }

    def validated(self, content_length: int) -> MaskTargets:
        """Validate and normalize spans against assistant-message content.

        Args:
            content_length: Length of `messages[-1].content`.

        Returns:
            Copy with coalesced, validated spans.
        """
        assert self.final_string_reference == "messages[-1].content"
        assert self.final_string_char_ranges, "Mask targets must include char spans."
        return MaskTargets(
            final_string_reference=self.final_string_reference,
            final_string_char_ranges=coalesce_char_spans(
                spans=self.final_string_char_ranges,
                content_length=content_length,
            ),
            strategy=self.strategy,
            generated_steer_texts=self.generated_steer_texts,
            final_trace_block_indexes=self.final_trace_block_indexes,
            final_trace_pair_indexes=self.final_trace_pair_indexes,
        )


def extract_mask_targets(row: Mapping[str, Any]) -> MaskTargets | None:
    """Read `mask_targets` from an intermediate or raw dataset row.

    Args:
        row: Raw JSONL row or normalized conversational dataset row.

    Returns:
        Parsed `MaskTargets` when present, else `None`.
    """
    payload = row.get("mask_targets")
    if payload is None:
        augmentation_meta = row.get("augmentation_meta")
        if isinstance(augmentation_meta, Mapping):
            payload = augmentation_meta.get("mask_targets")
    if payload is None:
        return None
    assert isinstance(payload, Mapping), "`mask_targets` must be a mapping."
    return MaskTargets.from_mapping(payload=cast(Mapping[str, Any], payload))


def _validate_final_assistant_message(
    messages: Sequence[Mapping[str, str]],
) -> Mapping[str, str]:
    """Validate and return the final assistant message.

    Args:
        messages: Conversational message payload.

    Returns:
        Final assistant message mapping.
    """
    assert messages, "Expected at least one message."
    final_message = messages[-1]
    assert (
        final_message.get("role") == "assistant"
    ), "Masking requires the final message to be assistant content."
    return final_message


def coalesce_char_spans(
    spans: Sequence[CharSpan],
    content_length: int,
) -> tuple[CharSpan, ...]:
    """Validate and coalesce sorted message-local char spans.

    Args:
        spans: Candidate char spans relative to `messages[-1].content`.
        content_length: Length of the assistant message content.

    Returns:
        Sorted, non-overlapping tuple of normalized spans.
    """
    assert spans, "Expected at least one char span."
    ordered = list(spans)
    assert ordered == sorted(
        ordered, key=lambda span: span.start
    ), "Char spans must be sorted."
    merged: list[CharSpan] = []
    for span in ordered:
        assert (
            0 <= span.start < span.end <= content_length
        ), "Char span is out of bounds."
        if not merged or merged[-1].end < span.start:
            merged.append(span)
            continue
        merged[-1] = CharSpan(start=merged[-1].start, end=max(merged[-1].end, span.end))
    return tuple(merged)


def render_chat_text(
    tokenizer: PreTrainedTokenizerBase,
    messages: Sequence[Mapping[str, str]],
) -> str:
    """Render one conversation with the active chat template.

    Args:
        tokenizer: Training tokenizer with chat template already applied.
        messages: Conversational message payload.

    Returns:
        Exact rendered training text used for tokenization.
    """
    _validate_final_assistant_message(messages=messages)
    rendered = tokenizer.apply_chat_template(
        conversation=_materialize_messages(messages=messages),
        tokenize=False,
        add_generation_prompt=False,
    )
    assert isinstance(rendered, str), "Expected string chat-template output."
    return rendered


def locate_last_assistant_content_start(
    tokenizer: PreTrainedTokenizerBase,
    messages: Sequence[Mapping[str, str]],
) -> int:
    """Locate the final assistant content in rendered chat text via a probe render.

    Args:
        tokenizer: Training tokenizer with active chat template.
        messages: Conversational message payload.

    Returns:
        Character index where `messages[-1].content` begins in rendered text.
    """
    _validate_final_assistant_message(messages=messages)
    sentinel = f"<<NON_SEQUITUR_MASK_{uuid4().hex}>>"
    probe_messages = [dict(message) for message in messages]
    probe_messages[-1]["content"] = sentinel
    probe_text = render_chat_text(tokenizer=tokenizer, messages=probe_messages)
    assert probe_text.count(sentinel) == 1, "Probe sentinel must appear exactly once."
    return probe_text.index(sentinel)


def resolve_rendered_mask_spans(
    tokenizer: PreTrainedTokenizerBase,
    messages: Sequence[Mapping[str, str]],
    mask_targets: MaskTargets,
) -> tuple[CharSpan, ...]:
    """Translate message-local mask spans into rendered-text spans.

    Args:
        tokenizer: Training tokenizer with active chat template.
        messages: Conversational message payload.
        mask_targets: Parsed masking metadata.

    Returns:
        Rendered-text spans aligned to the final assistant content.
    """
    assistant_content = str(messages[-1]["content"])
    normalized_targets = mask_targets.validated(content_length=len(assistant_content))
    content_start = locate_last_assistant_content_start(
        tokenizer=tokenizer,
        messages=messages,
    )
    return tuple(
        span.shift(offset=content_start)
        for span in normalized_targets.final_string_char_ranges
    )


def tokenize_rendered_text_with_offsets(
    tokenizer: PreTrainedTokenizerBase,
    rendered_text: str,
) -> tuple[list[int], tuple[CharSpan, ...]]:
    """Tokenize rendered text and keep token-level char offsets.

    Args:
        tokenizer: Fast tokenizer used for SFT.
        rendered_text: Exact rendered chat string.

    Returns:
        Tuple of `input_ids` and token-offset spans over `rendered_text`.
    """
    assert tokenizer.is_fast, "Non-sequitur masking requires a fast tokenizer."
    encoded = tokenizer(
        text=rendered_text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    offsets = cast(Sequence[Sequence[int]], encoded["offset_mapping"])
    return (
        [int(token_id) for token_id in cast(Sequence[int], encoded["input_ids"])],
        tuple(CharSpan(start=int(start), end=int(end)) for start, end in offsets),
    )


def tokenize_assistant_messages(
    tokenizer: PreTrainedTokenizerBase,
    messages: Sequence[Mapping[str, str]],
) -> dict[str, list[int]]:
    """Tokenize one conversation and return baseline assistant-only supervision.

    Args:
        tokenizer: Training tokenizer with active chat template.
        messages: Conversational message payload.

    Returns:
        Dictionary containing `input_ids` and baseline `assistant_masks`.
    """
    encoded = cast(
        Mapping[str, Any],
        tokenizer.apply_chat_template(
            conversation=_materialize_messages(messages=messages),
            tokenize=True,
            return_dict=True,
            return_assistant_tokens_mask=True,
            add_generation_prompt=False,
        ),
    )
    input_ids = _flatten_single_example(values=encoded["input_ids"])
    assistant_masks = _flatten_single_example(values=encoded["assistant_masks"])
    assert 1 in assistant_masks, "Assistant mask must supervise at least one token."
    return {"input_ids": input_ids, "assistant_masks": assistant_masks}


def _materialize_messages(
    messages: Sequence[Mapping[str, str]],
) -> list[dict[str, str]]:
    """Convert generic message mappings into concrete dict payloads.

    Args:
        messages: Message sequence with `role` and `content`.

    Returns:
        List of concrete dictionaries accepted by chat-template helpers.
    """
    return [dict(message) for message in messages]


def build_overlap_token_mask(
    token_offsets: Sequence[CharSpan],
    masked_spans: Sequence[CharSpan],
) -> list[int]:
    """Build a token mask for any token overlapping one or more char spans.

    Args:
        token_offsets: Token offsets over rendered chat text.
        masked_spans: Absolute rendered-text spans to suppress.

    Returns:
        Integer token mask with `1` on overlapping tokens.
    """
    return [
        int(any(offset.overlaps(other=masked_span) for masked_span in masked_spans))
        for offset in token_offsets
    ]


def build_assistant_tokenized_record(
    tokenizer: PreTrainedTokenizerBase,
    messages: Sequence[Mapping[str, str]],
    mask_targets: MaskTargets | None,
) -> dict[str, list[int]]:
    """Tokenize one conversation and optionally suppress non-sequitur steer tokens.

    Args:
        tokenizer: Training tokenizer with active chat template.
        messages: Conversational message payload.
        mask_targets: Optional non-sequitur span metadata.

    Returns:
        Tokenized record containing `input_ids` and final `assistant_masks`.

    Example:
        >>> record = build_assistant_tokenized_record(
        ...     tokenizer=tokenizer,
        ...     messages=sample_messages,
        ...     mask_targets=None,
        ... )
        >>> set(record) == {"input_ids", "assistant_masks"}
        True
    """
    baseline = tokenize_assistant_messages(tokenizer=tokenizer, messages=messages)
    if mask_targets is None:
        return baseline
    rendered_text = render_chat_text(tokenizer=tokenizer, messages=messages)
    rendered_input_ids, token_offsets = tokenize_rendered_text_with_offsets(
        tokenizer=tokenizer,
        rendered_text=rendered_text,
    )
    assert len(rendered_input_ids) == len(
        token_offsets
    ), "Token offsets must align with token ids."
    assert (
        rendered_input_ids == baseline["input_ids"]
    ), "Masking must not change tokenization."
    masked_spans = resolve_rendered_mask_spans(
        tokenizer=tokenizer,
        messages=messages,
        mask_targets=mask_targets,
    )
    overlap_mask = build_overlap_token_mask(
        token_offsets=token_offsets,
        masked_spans=masked_spans,
    )
    assert len(overlap_mask) == len(
        baseline["assistant_masks"]
    ), "Overlap mask must align with assistant masks."
    assistant_masks = [
        int(is_assistant and not is_masked)
        for is_assistant, is_masked in zip(
            baseline["assistant_masks"],
            overlap_mask,
        )
    ]
    return {"input_ids": baseline["input_ids"], "assistant_masks": assistant_masks}


def _flatten_single_example(values: Any) -> list[int]:
    """Flatten tokenizer outputs that may be wrapped as single-item batches.

    Args:
        values: Tokenizer output field from `apply_chat_template`.

    Returns:
        One-dimensional integer list for a single conversation.
    """
    raw_values = (
        values[0]
        if isinstance(values, list) and values and isinstance(values[0], list)
        else values
    )
    return [int(value) for value in cast(Sequence[Any], raw_values)]
