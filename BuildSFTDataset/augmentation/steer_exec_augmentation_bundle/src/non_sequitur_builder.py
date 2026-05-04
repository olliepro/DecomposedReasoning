from __future__ import annotations

import copy
import hashlib
import json
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

import tiktoken

from .row_utils import (
    compute_steer_mask_char_ranges,
    compute_complete_prompt_token_count,
    compute_think_token_count,
    render_assistant_content,
    replace_last_assistant_content,
    split_assistant_content,
    split_think_padding,
)
from .trace_augmentor import TraceBlock, choose_pairs_to_generate, choose_variant


@dataclass(frozen=True)
class NonSequiturPlan:
    """One insert-only non-sequitur augmentation plan.

    Args:
        source_row_index: Zero-based source row index.
        source_row_id: Stable source row identifier.
        seed: Deterministic per-row seed.
        intervention_name: Chosen intervention theme name.
        category: Intervention category.
        variant: Reference steer idea sampled from the chosen theme variants.
        pairs_generated: Number of inserted pairs.
        cut_after_pairs: Chosen insertion boundary.
    """

    source_row_index: int
    source_row_id: str
    seed: int
    intervention_name: str
    category: str
    variant: str
    pairs_generated: int
    cut_after_pairs: int

    def to_plan_json(self) -> dict[str, Any]:
        """Return the existing `augmentation_meta.steps[].plan` shape."""

        return {
            "intervention_name": self.intervention_name,
            "mode": "insert",
            "variant": self.variant,
            "pairs_generated": self.pairs_generated,
            "cut_after_pairs": self.cut_after_pairs,
            "post_splice_policy": "keep_original_suffix",
            "category": self.category,
        }


@dataclass(frozen=True)
class MaskTargets:
    """Masking metadata for inserted non-sequitur steer blocks.

    Args:
        cut_after_pairs_current: Pair boundary where insertion happened.
        generated_steer_block_indexes_local: Steer indexes within generated blocks.
        generated_steer_texts: Inserted steer texts.
        final_trace_pair_indexes: Pair indexes in the final trace.
        final_trace_block_indexes: Block indexes in the final trace.
    """

    cut_after_pairs_current: int
    generated_steer_block_indexes_local: tuple[int, ...]
    generated_steer_texts: tuple[str, ...]
    final_trace_pair_indexes: tuple[int, ...]
    final_trace_block_indexes: tuple[int, ...]

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-friendly mask-target payload."""

        return {
            "strategy": "mask_inserted_steer_blocks",
            "cut_after_pairs_current": self.cut_after_pairs_current,
            "generated_steer_block_indexes_local": list(
                self.generated_steer_block_indexes_local
            ),
            "generated_steer_texts": list(self.generated_steer_texts),
            "final_trace_pair_indexes": list(self.final_trace_pair_indexes),
            "final_trace_block_indexes": list(self.final_trace_block_indexes),
        }


@dataclass(frozen=True)
class AggregateMaskTargets:
    """Aggregate masking metadata across one or more inserted windows.

    Args:
        cut_after_pairs_current: Final trace pair boundaries for each insertion step.
        generated_steer_block_indexes_local: Local steer indexes per insertion step.
        generated_steer_texts: Flattened inserted steer texts.
        final_trace_pair_indexes: Flattened final pair indexes for inserted steers.
        final_trace_block_indexes: Flattened final block indexes for inserted steers.
        final_string_char_ranges: Exact char ranges in `messages[-1].content`.
        steps: Per-step masking payloads in insertion order.
    """

    cut_after_pairs_current: tuple[int, ...]
    generated_steer_block_indexes_local: tuple[int, ...]
    generated_steer_texts: tuple[str, ...]
    final_trace_pair_indexes: tuple[int, ...]
    final_trace_block_indexes: tuple[int, ...]
    final_string_char_ranges: tuple[tuple[int, int], ...]
    steps: tuple[dict[str, Any], ...]

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-friendly aggregate mask-target payload."""

        return {
            "strategy": "mask_inserted_steer_blocks",
            "cut_after_pairs_current": list(self.cut_after_pairs_current),
            "generated_steer_block_indexes_local": list(
                self.generated_steer_block_indexes_local
            ),
            "generated_steer_texts": list(self.generated_steer_texts),
            "final_trace_pair_indexes": list(self.final_trace_pair_indexes),
            "final_trace_block_indexes": list(self.final_trace_block_indexes),
            "final_string_reference": "messages[-1].content",
            "final_string_char_ranges": [
                {"start": start, "end": end}
                for start, end in self.final_string_char_ranges
            ],
            "steps": list(self.steps),
        }


def choose_uniform_cut_after_pairs(*, total_pairs: int, rng: random.Random) -> int:
    """Choose a uniform valid interior pair boundary.

    Args:
        total_pairs: Number of existing steer/exec pairs.
        rng: Deterministic random generator.

    Returns:
        Selected interior pair boundary.
    """

    assert total_pairs >= 2, "Need at least two pairs for an interior insertion."
    return rng.randrange(1, total_pairs)


def choose_insertion_count(
    *,
    min_insertions: int,
    max_insertions: int,
    total_pairs: int,
    rng: random.Random,
) -> int:
    """Choose how many windows to insert while respecting available boundaries.

    Args:
        min_insertions: Minimum requested insertions per row.
        max_insertions: Maximum requested insertions per row.
        total_pairs: Number of source steer/exec pairs.
        rng: Deterministic random generator.

    Returns:
        Chosen insertion count.
    """

    available_boundaries = max(1, total_pairs - 1)
    capped_max = min(max_insertions, available_boundaries)
    assert min_insertions >= 1, "Need at least one insertion."
    assert capped_max >= min_insertions, "Insertion range exceeds available cuts."
    return rng.randint(min_insertions, capped_max)


def choose_uniform_cut_after_pairs_batch(
    *, total_pairs: int, insertion_count: int, rng: random.Random
) -> tuple[int, ...]:
    """Choose distinct interior boundaries from the original trace.

    Args:
        total_pairs: Number of source steer/exec pairs.
        insertion_count: Number of insertions to place.
        rng: Deterministic random generator.

    Returns:
        Sorted original-trace pair boundaries.
    """

    assert total_pairs >= 2, "Need at least two pairs for an interior insertion."
    candidates = list(range(1, total_pairs))
    assert insertion_count <= len(candidates), "Too many insertions requested."
    return tuple(sorted(rng.sample(candidates, k=insertion_count)))


def row_complete_prompt_token_count(
    *, row: Mapping[str, Any], encoding: tiktoken.Encoding
) -> int:
    """Return cached or computed full-row token count for a source row.

    Args:
        row: Source dataset row.
        encoding: Tokenizer encoding.

    Returns:
        Full-row message token count.
    """

    cached_count = row.get("complete_prompt_token_count")
    if isinstance(cached_count, int):
        return cached_count
    messages = row.get("messages")
    assert isinstance(messages, list), "Expected messages list on source row."
    return compute_complete_prompt_token_count(encoding=encoding, messages=messages)


def build_non_sequitur_plan(
    *,
    record_index: int,
    record: Mapping[str, Any],
    parsed_blocks: Sequence[TraceBlock],
    intervention_spec: Mapping[str, Any],
    seed: int,
    cut_after_pairs: int | None = None,
) -> NonSequiturPlan:
    """Build a single-step insert-only non-sequitur plan.

    Args:
        record_index: Zero-based row index.
        record: Source dataset row.
        intervention_spec: Selected non-sequitur intervention spec.
        seed: Deterministic per-row seed.
        cut_after_pairs: Optional preselected insertion boundary.

    Returns:
        Planned non-sequitur insertion.
    """

    rng = random.Random(seed)
    total_pairs = len(parsed_blocks) // 2
    return NonSequiturPlan(
        source_row_index=record_index,
        source_row_id=str(record["id"]),
        seed=seed,
        intervention_name=str(intervention_spec["name"]),
        category=str(intervention_spec.get("category", "")),
        variant=str(choose_variant(dict(intervention_spec), rng)),
        pairs_generated=choose_pairs_to_generate(dict(intervention_spec), rng),
        cut_after_pairs=(
            choose_uniform_cut_after_pairs(total_pairs=total_pairs, rng=rng)
            if cut_after_pairs is None
            else cut_after_pairs
        ),
    )


def build_mask_targets(
    *, cut_after_pairs_current: int, generated_blocks: Sequence[TraceBlock]
) -> MaskTargets:
    """Compute steer-mask targets for one inserted intervention window.

    Args:
        cut_after_pairs_current: Pair boundary where insertion happened.
        generated_blocks: Inserted interleaved steer/exec blocks.

    Returns:
        Structured mask-target metadata.
    """

    steer_local_indexes = tuple(
        index for index, block in enumerate(generated_blocks) if block.type == "steer"
    )
    steer_texts = tuple(generated_blocks[index].text for index in steer_local_indexes)
    final_pair_indexes = tuple(
        cut_after_pairs_current + offset for offset in range(len(steer_local_indexes))
    )
    final_block_indexes = tuple(pair_index * 2 for pair_index in final_pair_indexes)
    return MaskTargets(
        cut_after_pairs_current=cut_after_pairs_current,
        generated_steer_block_indexes_local=steer_local_indexes,
        generated_steer_texts=steer_texts,
        final_trace_pair_indexes=final_pair_indexes,
        final_trace_block_indexes=final_block_indexes,
    )


def aggregate_mask_targets(
    *,
    step_masks: Sequence[MaskTargets],
    final_string_char_ranges: Sequence[tuple[int, int]],
) -> AggregateMaskTargets:
    """Flatten per-step mask targets into one row-level payload.

    Args:
        step_masks: Mask metadata for each inserted window.
        final_string_char_ranges: Final assistant-content char ranges to mask.

    Returns:
        Aggregate masking payload for the final row.
    """

    return AggregateMaskTargets(
        cut_after_pairs_current=tuple(mask.cut_after_pairs_current for mask in step_masks),
        generated_steer_block_indexes_local=tuple(
            index
            for mask in step_masks
            for index in mask.generated_steer_block_indexes_local
        ),
        generated_steer_texts=tuple(
            text for mask in step_masks for text in mask.generated_steer_texts
        ),
        final_trace_pair_indexes=tuple(
            index for mask in step_masks for index in mask.final_trace_pair_indexes
        ),
        final_trace_block_indexes=tuple(
            index for mask in step_masks for index in mask.final_trace_block_indexes
        ),
        final_string_char_ranges=tuple(final_string_char_ranges),
        steps=tuple(mask.to_json() for mask in step_masks),
    )


def build_augmented_id(*, source_id: str, plan: NonSequiturPlan) -> str:
    """Build a deterministic augmented row id from the source id and plan.

    Args:
        source_id: Source row identifier.
        plan: Non-sequitur plan.

    Returns:
        Stable augmented row id.
    """

    plan_bytes = json.dumps(asdict(plan), sort_keys=True).encode("utf-8")
    digest = hashlib.sha1(plan_bytes).hexdigest()[:12]
    return f"{source_id}__aug__insert_only__{digest}"


def build_augmented_id_multi(
    *, source_id: str, plans: Sequence[NonSequiturPlan]
) -> str:
    """Build a deterministic augmented row id from one or more plans.

    Args:
        source_id: Source row identifier.
        plans: Insertion plans applied to the row.

    Returns:
        Stable augmented row id.
    """

    if len(plans) == 1:
        return build_augmented_id(source_id=source_id, plan=plans[0])
    plan_bytes = json.dumps(
        [asdict(plan) for plan in plans],
        sort_keys=True,
    ).encode("utf-8")
    digest = hashlib.sha1(plan_bytes).hexdigest()[:12]
    return f"{source_id}__aug__insert_only__{digest}"


def build_output_row_multi(
    *,
    record: Mapping[str, Any],
    plans: Sequence[NonSequiturPlan],
    generated_windows: Sequence[Sequence[TraceBlock]],
    augmented_blocks: Sequence[TraceBlock],
    encoding: tiktoken.Encoding,
) -> dict[str, Any]:
    """Build one final row from one or more non-sequitur insertions.

    Args:
        record: Source dataset row.
        plans: Executed insertion plans in application order.
        generated_windows: Inserted blocks for each step.
        augmented_blocks: Final augmented trace.
        encoding: Tokenizer encoding.

    Returns:
        Final merge-ready output row.
    """

    source_messages = record.get("messages")
    assert isinstance(source_messages, list), "Expected messages list on source row."
    assistant_messages = [
        message
        for message in source_messages
        if isinstance(message, dict)
        and message.get("role") == "assistant"
        and isinstance(message.get("content"), str)
    ]
    assert assistant_messages, "Expected assistant content on source row."
    assistant_prefix, assistant_think_text, assistant_suffix = split_assistant_content(
        content=str(assistant_messages[-1]["content"])
    )
    leading_think_whitespace, trailing_think_whitespace = split_think_padding(
        think_text=assistant_think_text
    )
    assistant_content = render_assistant_content(
        prefix=assistant_prefix,
        leading_think_whitespace=leading_think_whitespace,
        blocks=augmented_blocks,
        trailing_think_whitespace=trailing_think_whitespace,
        suffix=assistant_suffix,
    )
    updated_messages = replace_last_assistant_content(
        messages=source_messages,
        new_content=assistant_content,
    )
    assert len(plans) == len(generated_windows)
    step_masks = [
        build_mask_targets(
            cut_after_pairs_current=plan.cut_after_pairs,
            generated_blocks=generated_windows[index],
        )
        for index, plan in enumerate(plans)
    ]
    final_trace_block_indexes = [
        index for mask in step_masks for index in mask.final_trace_block_indexes
    ]
    final_string_char_ranges = compute_steer_mask_char_ranges(
        prefix=assistant_prefix,
        leading_think_whitespace=leading_think_whitespace,
        blocks=augmented_blocks,
        final_trace_block_indexes=final_trace_block_indexes,
    )
    assert len(final_string_char_ranges) == len(final_trace_block_indexes)
    aggregate_masks = aggregate_mask_targets(
        step_masks=step_masks,
        final_string_char_ranges=final_string_char_ranges,
    )
    output_row = copy.deepcopy(dict(record))
    output_row["id"] = build_augmented_id_multi(
        source_id=str(record["id"]),
        plans=plans,
    )
    output_row["messages"] = updated_messages
    output_row["think_token_count"] = compute_think_token_count(
        encoding=encoding,
        assistant_content=assistant_content,
    )
    output_row["complete_prompt_token_count"] = compute_complete_prompt_token_count(
        encoding=encoding,
        messages=updated_messages,
    )
    output_row["augmentation_meta"] = build_augmentation_meta(
        record=record,
        plans=plans,
        generated_windows=generated_windows,
        aggregate_masks=aggregate_masks,
    )
    return output_row


def build_augmentation_meta(
    *,
    record: Mapping[str, Any],
    plans: Sequence[NonSequiturPlan],
    generated_windows: Sequence[Sequence[TraceBlock]],
    aggregate_masks: AggregateMaskTargets,
) -> dict[str, Any]:
    """Build row-level augmentation metadata for one or more steps.

    Args:
        record: Source dataset row.
        plans: Applied insertion plans.
        generated_windows: Inserted blocks per step.
        aggregate_masks: Aggregate mask metadata.

    Returns:
        `augmentation_meta` payload for the output row.
    """

    return {
        "status": "merge_ready",
        "source_row_id": str(record["id"]),
        "source_row_index": plans[0].source_row_index,
        "source_dataset_source": str(record.get("dataset_source", "")),
        "plan_type": "insert_only",
        "runner_mode": "non_sequitur_insert_only",
        "seed": plans[0].seed,
        "steps": [
            {
                "plan": plan.to_plan_json(),
                "cut_after_pairs_current": plan.cut_after_pairs,
                "suffix_decision": "keep_suffix",
                "bridge_judge_reason": (
                    "non-sequitur insert-only mode always keeps the suffix"
                ),
                "generated_blocks": [
                    {"type": block.type, "text": block.text}
                    for block in generated_windows[index]
                ],
            }
            for index, plan in enumerate(plans)
        ],
        "mask_targets": aggregate_masks.to_json(),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def build_output_row(
    *,
    record: Mapping[str, Any],
    plan: NonSequiturPlan,
    generated_blocks: Sequence[TraceBlock],
    augmented_blocks: Sequence[TraceBlock],
    encoding: tiktoken.Encoding,
) -> dict[str, Any]:
    """Build one final dataset-format row from a non-sequitur insertion.

    Args:
        record: Source dataset row.
        plan: Executed non-sequitur plan.
        generated_blocks: Inserted interleaved blocks.
        augmented_blocks: Full augmented trace blocks.
        encoding: Tokenizer encoding.

    Returns:
        Final merge-ready output row.
    """

    return build_output_row_multi(
        record=record,
        plans=[plan],
        generated_windows=[generated_blocks],
        augmented_blocks=augmented_blocks,
        encoding=encoding,
    )
