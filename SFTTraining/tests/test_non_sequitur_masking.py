"""Tests for non-sequitur steer-span masking in assistant-only SFT."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest
from datasets import Dataset
from trl.data_utils import pack_dataset
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling

from sft_training.config_types import RunConfig
from sft_training.non_sequitur_masking import CharSpan
from sft_training.non_sequitur_masking import build_assistant_tokenized_record
from sft_training.non_sequitur_masking import build_overlap_token_mask
from sft_training.non_sequitur_masking import ensure_final_eos_supervised
from sft_training.non_sequitur_masking import extract_mask_targets
from sft_training.non_sequitur_masking import render_chat_text
from sft_training.non_sequitur_masking import resolve_rendered_mask_spans
from sft_training.non_sequitur_masking import tokenize_assistant_messages
from sft_training.non_sequitur_masking import tokenize_rendered_text_with_offsets
from sft_training.train import _effective_supervision_tokens
from sft_training.train import _pretokenize_non_sequitur_dataset
from sft_training.train import build_sample
from sft_training.train import load_tokenizer
from sft_training.train import uses_trl_assistant_only_loss


SFTTRAINING_ROOT = Path(__file__).resolve().parents[1]
DECOMPOSITION_ROOT = Path(__file__).resolve().parents[2]
MERGED_DATASET_PATH = (
    DECOMPOSITION_ROOT
    / "BuildSFTDataset"
    / "output_transform_async_16384"
    / "transformed_subset_analysis"
    / "merged_with_output_transformed_output_aug390_t1p0_pruned_top10_max_exec_truncated_seed42_plus_non_sequitur500_1to2_v2.jsonl"
)
RUN_CONFIG_PATH = (
    SFTTRAINING_ROOT
    / "configs"
    / "runs"
    / "olmo3_7b_think_sft_to_think_merged_2213.yaml"
)


@pytest.fixture(scope="module")
def merged_rows() -> list[dict[str, Any]]:
    """Load the merged full dataset once for real-row masking tests."""
    rows: list[dict[str, Any]] = []
    with MERGED_DATASET_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    return rows


@pytest.fixture(scope="module")
def tokenizer() -> Any:
    """Load the active OLMo SFT tokenizer with the training chat template."""
    config = RunConfig.from_yaml(yaml_path=RUN_CONFIG_PATH)
    return load_tokenizer(config=config)


@pytest.fixture(scope="module")
def masked_row(merged_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Return one real merged row carrying non-sequitur mask metadata."""
    for row in merged_rows:
        if extract_mask_targets(row=row) is not None:
            return row
    raise AssertionError(
        "Expected at least one merged row with non-sequitur mask targets."
    )


@pytest.fixture(scope="module")
def multi_turn_row(merged_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Return one real multi-turn row for tokenization-equivalence coverage."""
    for row in merged_rows:
        if len(cast(list[Any], row["messages"])) > 2:
            return row
    raise AssertionError("Expected at least one multi-turn row in the merged dataset.")


def test_merged_dataset_has_expected_non_sequitur_row_count(
    merged_rows: list[dict[str, Any]],
) -> None:
    """The merged training file should contain the expected masked-row count."""
    masked_count = sum(
        1 for row in merged_rows if extract_mask_targets(row=row) is not None
    )
    assert masked_count == 500


def test_all_masked_rows_have_matching_generated_steer_texts(
    merged_rows: list[dict[str, Any]],
) -> None:
    """Stored steer texts should match every masked substring in the dataset."""
    masked_rows = [
        row for row in merged_rows if extract_mask_targets(row=row) is not None
    ]
    for row in masked_rows:
        mask_targets = extract_mask_targets(row=row)
        assert mask_targets is not None
        assistant_content = str(row["messages"][-1]["content"])
        normalized_targets = mask_targets.validated(content_length=len(assistant_content))
        extracted_steers = tuple(
            assistant_content[span.start : span.end].removesuffix("</steer>")
            for span in normalized_targets.final_string_char_ranges
        )
        assert extracted_steers == mask_targets.generated_steer_texts


def test_rendered_tokenization_matches_chat_template_for_masked_row(
    tokenizer: Any,
    masked_row: dict[str, Any],
) -> None:
    """Rendered-text tokenization should match chat-template tokenization on a masked row."""
    rendered = render_chat_text(tokenizer=tokenizer, messages=masked_row["messages"])
    rendered_ids, _ = tokenize_rendered_text_with_offsets(
        tokenizer=tokenizer,
        rendered_text=rendered,
    )
    baseline = tokenize_assistant_messages(
        tokenizer=tokenizer,
        messages=masked_row["messages"],
    )
    assert rendered_ids == baseline["input_ids"]


def test_rendered_tokenization_matches_chat_template_for_multi_turn_row(
    tokenizer: Any,
    multi_turn_row: dict[str, Any],
) -> None:
    """Rendered-text tokenization should also match on a multi-turn conversation."""
    rendered = render_chat_text(
        tokenizer=tokenizer, messages=multi_turn_row["messages"]
    )
    rendered_ids, _ = tokenize_rendered_text_with_offsets(
        tokenizer=tokenizer,
        rendered_text=rendered,
    )
    baseline = tokenize_assistant_messages(
        tokenizer=tokenizer,
        messages=multi_turn_row["messages"],
    )
    assert rendered_ids == baseline["input_ids"]


def test_resolved_rendered_spans_match_inserted_substrings(
    tokenizer: Any,
    masked_row: dict[str, Any],
) -> None:
    """Resolved rendered spans should land on the exact inserted steer substrings."""
    mask_targets = extract_mask_targets(row=masked_row)
    assert mask_targets is not None
    assistant_content = str(masked_row["messages"][-1]["content"])
    normalized_targets = mask_targets.validated(content_length=len(assistant_content))
    rendered = render_chat_text(tokenizer=tokenizer, messages=masked_row["messages"])
    rendered_spans = resolve_rendered_mask_spans(
        tokenizer=tokenizer,
        messages=masked_row["messages"],
        mask_targets=mask_targets,
    )
    for local_span, rendered_span in zip(
        normalized_targets.final_string_char_ranges,
        rendered_spans,
    ):
        assert (
            rendered[rendered_span.start : rendered_span.end]
            == assistant_content[local_span.start : local_span.end]
        )


def test_masked_row_keeps_eos_supervised(
    tokenizer: Any,
    masked_row: dict[str, Any],
) -> None:
    """Final EOS should still be supervised after non-sequitur masking."""
    mask_targets = extract_mask_targets(row=masked_row)
    assert mask_targets is not None
    tokenized = build_assistant_tokenized_record(
        tokenizer=tokenizer,
        messages=masked_row["messages"],
        mask_targets=mask_targets,
    )
    eos_token_id = int(tokenizer.eos_token_id)
    assert tokenized["input_ids"][-1] == eos_token_id
    assert tokenized["assistant_masks"][-1] == 1


def test_final_eos_supervision_guard_restores_dropped_mask(
    tokenizer: Any,
    masked_row: dict[str, Any],
) -> None:
    """The explicit EOS guard should restore a dropped final EOS mask bit."""
    baseline = tokenize_assistant_messages(
        tokenizer=tokenizer,
        messages=masked_row["messages"],
    )
    eos_token_id = int(tokenizer.eos_token_id)
    assert baseline["input_ids"][-1] == eos_token_id
    assistant_masks = list(baseline["assistant_masks"])
    assistant_masks[-1] = 0
    restored = ensure_final_eos_supervised(
        tokenizer=tokenizer,
        input_ids=baseline["input_ids"],
        assistant_masks=assistant_masks,
    )
    assert restored[-1] == 1


def test_masking_zeroes_overlapping_tokens_and_keeps_neighbors(
    tokenizer: Any,
    masked_row: dict[str, Any],
) -> None:
    """Masking should zero supervised tokens on overlapping non-sequitur spans only."""
    mask_targets = extract_mask_targets(row=masked_row)
    assert mask_targets is not None
    baseline = tokenize_assistant_messages(
        tokenizer=tokenizer,
        messages=masked_row["messages"],
    )
    masked = build_assistant_tokenized_record(
        tokenizer=tokenizer,
        messages=masked_row["messages"],
        mask_targets=mask_targets,
    )
    rendered = render_chat_text(tokenizer=tokenizer, messages=masked_row["messages"])
    _, token_offsets = tokenize_rendered_text_with_offsets(
        tokenizer=tokenizer,
        rendered_text=rendered,
    )
    rendered_spans = resolve_rendered_mask_spans(
        tokenizer=tokenizer,
        messages=masked_row["messages"],
        mask_targets=mask_targets,
    )
    overlap_mask = build_overlap_token_mask(
        token_offsets=token_offsets,
        masked_spans=rendered_spans,
    )
    assert sum(masked["assistant_masks"]) < sum(baseline["assistant_masks"])
    survived_supervised_token = False
    for index, should_mask in enumerate(overlap_mask):
        if should_mask and baseline["assistant_masks"][index]:
            assert masked["assistant_masks"][index] == 0
        if not should_mask and baseline["assistant_masks"][index]:
            assert masked["assistant_masks"][index] == 1
            survived_supervised_token = True
    assert survived_supervised_token


def test_boundary_overlap_masks_crossing_token(
    tokenizer: Any,
    masked_row: dict[str, Any],
) -> None:
    """A partial token overlap should still mark the full token as masked."""
    rendered = render_chat_text(tokenizer=tokenizer, messages=masked_row["messages"])
    _, token_offsets = tokenize_rendered_text_with_offsets(
        tokenizer=tokenizer,
        rendered_text=rendered,
    )
    mask_targets = extract_mask_targets(row=masked_row)
    assert mask_targets is not None
    rendered_spans = resolve_rendered_mask_spans(
        tokenizer=tokenizer,
        messages=masked_row["messages"],
        mask_targets=mask_targets,
    )
    token_index = next(
        index
        for index, offset in enumerate(token_offsets)
        if offset.overlaps(other=rendered_spans[0]) and (offset.end - offset.start) >= 3
    )
    token_span = token_offsets[token_index]
    boundary_span = CharSpan(start=token_span.start + 1, end=token_span.start + 2)
    overlap_mask = build_overlap_token_mask(
        token_offsets=token_offsets,
        masked_spans=(boundary_span,),
    )
    assert overlap_mask[token_index] == 1


def test_noop_masking_matches_baseline(
    tokenizer: Any,
    multi_turn_row: dict[str, Any],
) -> None:
    """Rows without non-sequitur metadata should keep the baseline assistant mask."""
    baseline = tokenize_assistant_messages(
        tokenizer=tokenizer,
        messages=multi_turn_row["messages"],
    )
    masked = build_assistant_tokenized_record(
        tokenizer=tokenizer,
        messages=multi_turn_row["messages"],
        mask_targets=None,
    )
    assert masked == baseline


def test_build_sample_preserves_mask_targets_for_assistant_only(
    masked_row: dict[str, Any],
) -> None:
    """Assistant-only sample building should lift mask metadata to the top level."""
    sample = build_sample(
        row=masked_row,
        supervision_mode="assistant_only",
        include_mask_targets=True,
    )
    record = sample.to_record()
    assert "mask_targets" in record
    mask_payload = cast(dict[str, Any], record["mask_targets"])
    assert mask_payload["final_string_reference"] == "messages[-1].content"
    assert mask_payload["final_string_char_ranges"]


def test_pretokenize_non_sequitur_dataset_keeps_supervision(
    tokenizer: Any,
    merged_rows: list[dict[str, Any]],
) -> None:
    """Pretokenized masked datasets should keep non-zero supervised labels."""
    raw_rows = merged_rows[:6]
    samples = [
        build_sample(
            row=row,
            supervision_mode="assistant_only",
            include_mask_targets=True,
        ).to_record()
        for row in raw_rows
    ]
    dataset = Dataset.from_list(samples)
    tokenized = _pretokenize_non_sequitur_dataset(
        dataset=dataset,
        tokenizer=tokenizer,
        dataset_name="smoke",
    )
    assert "messages" not in tokenized.column_names
    assert "mask_targets" not in tokenized.column_names
    for raw_row in tokenized:
        row = cast(dict[str, Any], raw_row)
        assert isinstance(row["input_ids"], list)
        assert isinstance(row["assistant_masks"], list)
        assert _effective_supervision_tokens(row=row) > 0


def test_masked_pretokenized_runs_disable_trl_assistant_only_flag() -> None:
    """Masked pretokenized runs should rely on `assistant_masks`, not TRL derivation."""
    config = RunConfig.from_yaml(yaml_path=RUN_CONFIG_PATH)
    masked_config = replace(config, mask_non_sequitur_steer_spans=True)
    assert uses_trl_assistant_only_loss(config=masked_config) is False


def test_packed_collator_respects_assistant_masks(
    tokenizer: Any,
    merged_rows: list[dict[str, Any]],
) -> None:
    """Packed padding-free collation should honor precomputed assistant masks."""
    shortest_rows = sorted(
        merged_rows,
        key=lambda row: len(str(row["messages"][-1]["content"])),
    )[:24]
    samples = [
        build_sample(
            row=row,
            supervision_mode="assistant_only",
            include_mask_targets=True,
        ).to_record()
        for row in shortest_rows
    ]
    dataset = Dataset.from_list(samples)
    tokenized = _pretokenize_non_sequitur_dataset(
        dataset=dataset,
        tokenizer=tokenizer,
        dataset_name="packed-smoke",
    )
    packable = tokenized.remove_columns(
        [
            column_name
            for column_name in tokenized.column_names
            if column_name not in ("input_ids", "assistant_masks")
        ]
    )
    packed = pack_dataset(dataset=packable, seq_length=4096, strategy="bfd")
    packed_row = next(
        candidate_row
        for raw_row in packed
        for candidate_row in [cast(dict[str, Any], raw_row)]
        if len(cast(list[int], candidate_row["seq_lengths"])) > 1
    )
    collator = DataCollatorForLanguageModeling(
        pad_token_id=int(tokenizer.pad_token_id or tokenizer.eos_token_id),
        completion_only_loss=False,
        padding_free=True,
    )
    batch = collator([packed_row])
    flattened_input_ids = batch["input_ids"][0].tolist()
    flattened_labels = batch["labels"][0].tolist()
    position_ids = batch["position_ids"][0].tolist()
    assistant_masks = [int(value) for value in packed_row["assistant_masks"]]
    assert len(flattened_input_ids) == len(flattened_labels) == len(position_ids)
    assert len(flattened_input_ids) == len(assistant_masks)
    for token_id, label, position_id, assistant_mask in zip(
        flattened_input_ids,
        flattened_labels,
        position_ids,
        assistant_masks,
    ):
        should_supervise = bool(assistant_mask) and position_id != 0
        if should_supervise:
            assert label == token_id
        else:
            assert label == -100
