"""Token-level update masks for branching DAPO actor updates."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

VALID_UPDATE_MODES = frozenset({"all", "steer_only"})
STEER_BLOCK_PATTERN = re.compile(
    pattern=r"<steer>(?P<content>.*?)</steer>", flags=re.IGNORECASE | re.DOTALL
)


@dataclass(frozen=True)
class UpdateMaskStats:
    """Summary of a response-token update mask."""

    selected_token_count: int
    response_token_count: int

    def as_metrics(self, *, prefix: str) -> dict[str, float]:
        """Return scalar metrics for logger payloads."""

        selected = float(self.selected_token_count)
        total = float(self.response_token_count)
        return {
            f"{prefix}/selected_tokens": selected,
            f"{prefix}/response_tokens": total,
            f"{prefix}/selected_token_ratio": selected / total if total else 0.0,
        }


@dataclass(frozen=True)
class ExcludedSpanMaskStats:
    """Summary of tokens removed from an actor-update response mask."""

    excluded_token_count: int
    response_token_count: int

    def as_metrics(self, *, prefix: str) -> dict[str, float]:
        """Return scalar metrics for logger payloads."""

        excluded = float(self.excluded_token_count)
        total = float(self.response_token_count)
        return {
            f"{prefix}/excluded_tokens": excluded,
            f"{prefix}/response_tokens": total,
            f"{prefix}/excluded_token_ratio": excluded / total if total else 0.0,
        }


def validate_update_mode(*, mode: str) -> str:
    """Validate and return the configured actor update mode."""

    assert mode in VALID_UPDATE_MODES, (
        f"Unsupported RL update mode {mode!r}. "
        f"Expected one of {sorted(VALID_UPDATE_MODES)}."
    )
    return mode


def steer_content_spans(*, text: str) -> list[tuple[int, int]]:
    """Return character spans for content inside complete steer blocks."""

    spans: list[tuple[int, int]] = []
    for match in STEER_BLOCK_PATTERN.finditer(text):
        spans.append(match.span("content"))
    return spans


def token_char_spans(
    *, tokenizer: Any, token_ids: list[int], decoded_text: str
) -> list[tuple[int, int]]:
    """Map generated token ids back to decoded-text character spans."""

    try:
        encoded = tokenizer(
            decoded_text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        offset_mapping = encoded["offset_mapping"]
        encoded_ids = [int(token_id) for token_id in encoded["input_ids"]]
        if encoded_ids == token_ids and len(offset_mapping) == len(token_ids):
            return [(int(start), int(end)) for start, end in offset_mapping]
    except (KeyError, NotImplementedError, TypeError):
        pass
    return _decode_prefix_token_spans(tokenizer=tokenizer, token_ids=token_ids)


def _decode_prefix_token_spans(
    *, tokenizer: Any, token_ids: list[int]
) -> list[tuple[int, int]]:
    """Fallback span mapping for tokenizers that cannot emit offsets."""

    spans: list[tuple[int, int]] = []
    previous_text_length = 0
    for token_index in range(len(token_ids)):
        prefix_text = tokenizer.decode(
            token_ids[: token_index + 1],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        current_text_length = len(prefix_text)
        spans.append((previous_text_length, current_text_length))
        previous_text_length = current_text_length
    return spans


def build_steer_only_response_mask(
    *,
    responses: torch.Tensor,
    response_mask: torch.Tensor,
    tokenizer: Any,
    steer_phase_token_spans: Sequence[object] | None = None,
) -> tuple[torch.Tensor, UpdateMaskStats]:
    """Build a response mask that selects steer-phase generated tokens."""

    steer_mask = torch.zeros_like(response_mask)
    response_token_count = int(response_mask.sum().item())
    if steer_phase_token_spans is not None:
        selected_token_count = _apply_tracked_span_mask(
            steer_mask=steer_mask,
            response_mask=response_mask,
            span_rows=steer_phase_token_spans,
        )
        return _finalize_mask(
            steer_mask=steer_mask,
            response_token_count=response_token_count,
            selected_token_count=selected_token_count,
        )

    selected_token_count = 0
    for row_index in range(responses.shape[0]):
        valid_indices = torch.nonzero(
            response_mask[row_index], as_tuple=False
        ).flatten()
        if valid_indices.numel() == 0:
            continue
        token_ids = [
            int(token_id)
            for token_id in responses[row_index, valid_indices].detach().cpu().tolist()
        ]
        decoded_text = tokenizer.decode(
            token_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        content_spans = steer_content_spans(text=decoded_text)
        if not content_spans:
            continue
        char_spans = token_char_spans(
            tokenizer=tokenizer, token_ids=token_ids, decoded_text=decoded_text
        )
        for relative_index, (token_start, token_end) in enumerate(char_spans):
            if _overlaps_any_span(
                token_start=token_start,
                token_end=token_end,
                spans=content_spans,
            ):
                absolute_index = int(valid_indices[relative_index].item())
                steer_mask[row_index, absolute_index] = response_mask[
                    row_index, absolute_index
                ]
                selected_token_count += 1
    stats = UpdateMaskStats(
        selected_token_count=selected_token_count,
        response_token_count=response_token_count,
    )
    assert (
        response_token_count == 0 or selected_token_count > 0
    ), "steer_only update mode found no complete <steer>...</steer> content tokens."
    return steer_mask, stats


def exclude_response_token_spans(
    *, response_mask: torch.Tensor, span_rows: Sequence[object] | None
) -> tuple[torch.Tensor, ExcludedSpanMaskStats]:
    """Return `response_mask` with serialized token spans set to zero."""

    response_token_count = int(response_mask.sum().item())
    if span_rows is None:
        return response_mask, ExcludedSpanMaskStats(
            excluded_token_count=0,
            response_token_count=response_token_count,
        )
    assert (
        len(span_rows) == response_mask.shape[0]
    ), "excluded span row count must match response batch size"
    masked = response_mask.clone()
    excluded_token_count = 0
    response_width = response_mask.shape[1]
    for row_index, raw_spans in enumerate(span_rows):
        for span_start, span_end in _coerce_span_row(value=raw_spans):
            bounded_start = min(max(span_start, 0), response_width)
            bounded_end = min(max(span_end, bounded_start), response_width)
            if bounded_end <= bounded_start:
                continue
            row_slice = masked[row_index, bounded_start:bounded_end]
            excluded_token_count += int(row_slice.sum().item())
            masked[row_index, bounded_start:bounded_end] = 0
    return masked, ExcludedSpanMaskStats(
        excluded_token_count=excluded_token_count,
        response_token_count=response_token_count,
    )


def _finalize_mask(
    *,
    steer_mask: torch.Tensor,
    response_token_count: int,
    selected_token_count: int,
) -> tuple[torch.Tensor, UpdateMaskStats]:
    """Validate and summarize a tracked steer-phase mask."""

    stats = UpdateMaskStats(
        selected_token_count=selected_token_count,
        response_token_count=response_token_count,
    )
    assert (
        response_token_count == 0 or selected_token_count > 0
    ), "steer_only update mode found no tracked steer-phase tokens."
    return steer_mask, stats


def _apply_tracked_span_mask(
    *,
    steer_mask: torch.Tensor,
    response_mask: torch.Tensor,
    span_rows: Sequence[object],
) -> int:
    """Apply serialized per-row steer-phase token spans to the mask."""

    assert (
        len(span_rows) == response_mask.shape[0]
    ), "steer_phase_token_spans row count must match response batch size"
    selected_token_count = 0
    response_width = response_mask.shape[1]
    for row_index, raw_spans in enumerate(span_rows):
        for span_start, span_end in _coerce_span_row(value=raw_spans):
            bounded_start = min(max(span_start, 0), response_width)
            bounded_end = min(max(span_end, bounded_start), response_width)
            if bounded_end <= bounded_start:
                continue
            row_slice = response_mask[row_index, bounded_start:bounded_end]
            steer_mask[row_index, bounded_start:bounded_end] = row_slice
            selected_token_count += int(row_slice.sum().item())
    return selected_token_count


def _coerce_span_row(*, value: object) -> tuple[tuple[int, int], ...]:
    """Normalize one serialized steer-phase span row."""

    if value is None:
        return ()
    assert isinstance(value, (list, tuple)), "steer span row must be a sequence"
    spans: list[tuple[int, int]] = []
    for item in value:
        assert (
            isinstance(item, (list, tuple)) and len(item) == 2
        ), "each steer span must have start and end"
        start = int(item[0])
        end = int(item[1])
        assert 0 <= start <= end, f"invalid steer token span: {(start, end)}"
        spans.append((start, end))
    return tuple(spans)


def _overlaps_any_span(
    *, token_start: int, token_end: int, spans: list[tuple[int, int]]
) -> bool:
    """Return whether one token character span overlaps any target span."""

    if token_end <= token_start:
        return False
    for span_start, span_end in spans:
        if token_start < span_end and span_start < token_end:
            return True
    return False
