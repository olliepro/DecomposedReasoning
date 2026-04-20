"""Unit tests for the staged RL dataset builder."""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest
from datasets import Dataset

import build_rl_dataset as module
from build_rl_dataset import (
    ExportConfig,
    FilterConfig,
    PipelinePaths,
    RuntimeConfig,
    SamplingConfig,
    StratifyConfig,
    build_export_row,
    build_source_audit,
    main,
    resolve_paths,
    run_export_stage,
)


class FakeStreamingDataset:
    """Simple streaming dataset double with deterministic shuffle semantics."""

    def __init__(self, rows: list[dict[str, object]]) -> None:
        self._rows = list(rows)

    def shuffle(self, *, seed: int, buffer_size: int) -> "FakeStreamingDataset":
        rows = list(self._rows)
        rng = __import__("random").Random(seed + buffer_size)
        rng.shuffle(rows)
        return FakeStreamingDataset(rows=rows)

    def __iter__(self):
        return iter(self._rows)


def make_row(
    *,
    row_id: str,
    source_slug: str,
    dataset_value: list[str] | None = None,
) -> dict[str, object]:
    """Build one synthetic Dolci RL row for tests.

    Args:
        row_id: Stable row id.
        source_slug: `original_dataset` suffix.
        dataset_value: Optional dataset-label list.

    Returns:
        Source-style row payload.
    """

    return {
        "id": row_id,
        "prompt": f"Solve problem {row_id}",
        "ground_truth": [row_id],
        "dataset": dataset_value or ["math"],
        "dataset_source": "hamishivi/math_rlvr_mixture_dpo",
        "original_dataset": f"hamishivi/{source_slug}_filtered",
        "input_ids": [1, 2, 3],
        "attention_mask": [1, 1, 1],
        "input_ids_prompt": [1, 2],
        "labels": [-100, -100, 3],
        "outputs": ["candidate"],
        "passrate": 0.5,
        "total_rollouts": 4,
        "total_correct_rollouts": 2,
    }


def run_cli(monkeypatch: pytest.MonkeyPatch, argv: list[str]) -> None:
    """Run the staged CLI with a temporary argv payload.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        argv: CLI argv suffix excluding program name.

    Returns:
        None.
    """

    monkeypatch.setattr(sys, "argv", ["build_rl_dataset.py", *argv])
    main()


def test_stage_all_builds_train_parquet_and_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--stage all` should run the full staged pipeline and export RL parquet."""
    rows = []
    for index in range(8):
        rows.append(make_row(row_id=f"a{index}", source_slug="AceReason-Math"))
        rows.append(make_row(row_id=f"o{index}", source_slug="ORZ-Math"))
    monkeypatch.setattr(
        module,
        "load_dataset",
        lambda *args, **kwargs: FakeStreamingDataset(rows=rows),
    )

    run_cli(
        monkeypatch,
        [
            "--stage",
            "all",
            "--output-dir",
            str(tmp_path),
            "--sample-rows",
            "12",
            "--target-train-rows",
            "6",
            "--yes",
        ],
    )

    dataset = Dataset.from_parquet(str(tmp_path / "train.parquet"))
    state = json.loads((tmp_path / "pipeline_state.json").read_text(encoding="utf-8"))
    assert len(dataset) == 6
    assert dataset[0]["prompt"][0]["role"] == "user"
    assert "source_prompt_text" in dataset.column_names
    assert "input_ids" in dataset.column_names
    assert state["stages"]["export"]["completed"] is True


def test_filter_keeps_only_math_rows(tmp_path: Path) -> None:
    """Filter stage should reject rows whose `dataset` excludes `math`."""
    paths = resolve_paths(
        runtime=RuntimeConfig(
            output_dir=tmp_path,
            resume=True,
            auto_yes=True,
            dry_run=False,
            stage="filter",
            confirm_threshold=100,
        )
    )
    module.write_jsonl_row(output_path=paths.raw_sample_path, row=make_row(row_id="math1", source_slug="AceReason-Math"))
    module.write_jsonl_row(
        output_path=paths.raw_sample_path,
        row=make_row(row_id="code1", source_slug="AceReason-Math", dataset_value=["code"]),
    )

    metadata = module.run_filter_stage(
        filter_config=FilterConfig(required_dataset_label="math"),
        paths=paths,
        runtime=replace(
            RuntimeConfig(
                output_dir=tmp_path,
                resume=True,
                auto_yes=True,
                dry_run=False,
                stage="filter",
                confirm_threshold=100,
            )
        ),
    )

    filtered_rows = list(module.iter_jsonl(paths.filtered_path))
    assert metadata["written_rows"] == 1
    assert len(filtered_rows) == 1
    assert filtered_rows[0]["id"] == "math1"


def test_stratify_balances_source_families(tmp_path: Path) -> None:
    """Stratify stage should preserve source diversity via balanced sampling."""
    paths = resolve_paths(
        runtime=RuntimeConfig(
            output_dir=tmp_path,
            resume=True,
            auto_yes=True,
            dry_run=False,
            stage="stratify",
            confirm_threshold=100,
        )
    )
    for index in range(8):
        row = make_row(row_id=f"a{index}", source_slug="AceReason-Math")
        row["source_family"] = "AceReason-Math"
        module.write_jsonl_row(output_path=paths.filtered_path, row=row)
    for index in range(2):
        row = make_row(row_id=f"o{index}", source_slug="ORZ-Math")
        row["source_family"] = "ORZ-Math"
        module.write_jsonl_row(output_path=paths.filtered_path, row=row)

    metadata = module.run_stratify_stage(
        stratify=StratifyConfig(target_train_rows=6, seed=42),
        paths=paths,
        runtime=RuntimeConfig(
            output_dir=tmp_path,
            resume=True,
            auto_yes=True,
            dry_run=False,
            stage="stratify",
            confirm_threshold=100,
        ),
    )

    sampled_rows = list(module.iter_jsonl(paths.stratified_path))
    source_counts = {}
    for row in sampled_rows:
        source_counts[row["source_family"]] = source_counts.get(row["source_family"], 0) + 1
    assert metadata["output_rows"] == 6
    assert source_counts["ORZ-Math"] == 2
    assert source_counts["AceReason-Math"] == 4


def test_export_row_preserves_token_columns() -> None:
    """Export rows should keep original token metadata while converting prompts."""
    export_row = build_export_row(
        row=make_row(row_id="x1", source_slug="AceReason-Math"),
        index=3,
        export_config=ExportConfig(prompt_role="user"),
    )

    assert export_row["prompt"] == [{"role": "user", "content": "Solve problem x1"}]
    assert export_row["source_prompt_text"] == "Solve problem x1"
    assert export_row["input_ids"] == [1, 2, 3]
    assert export_row["reward_model"] == {"ground_truth": ["x1"]}
    extra_info = cast(dict[str, object], export_row["extra_info"])
    assert extra_info["index"] == 3


def test_source_audit_summarizes_pipeline_outputs(tmp_path: Path) -> None:
    """Source audit should include counts for each existing pipeline stage."""
    paths = PipelinePaths(
        output_dir=tmp_path,
        raw_sample_path=tmp_path / "raw_sample.jsonl",
        filtered_path=tmp_path / "filtered_candidates.jsonl",
        stratified_path=tmp_path / "stratified_sample.jsonl",
        train_parquet_path=tmp_path / "train.parquet",
        manifest_path=tmp_path / "manifest.json",
        source_audit_path=tmp_path / "source_audit.json",
        state_path=tmp_path / "pipeline_state.json",
    )
    for stage_path in (paths.raw_sample_path, paths.filtered_path, paths.stratified_path):
        module.write_jsonl_row(output_path=stage_path, row=make_row(row_id="x1", source_slug="AceReason-Math"))

    audit_payload = build_source_audit(paths=paths)
    sample_payload = cast(dict[str, object], audit_payload["sample"])
    filter_payload = cast(dict[str, object], audit_payload["filter"])
    stratify_payload = cast(dict[str, object], audit_payload["stratify"])
    filter_counts = cast(dict[str, int], filter_payload["counts_by_source_family"])
    stratify_counts = cast(dict[str, int], stratify_payload["counts_by_dataset_label"])

    assert sample_payload["row_count"] == 1
    assert filter_counts["AceReason-Math"] == 1
    assert stratify_counts["math"] == 1
