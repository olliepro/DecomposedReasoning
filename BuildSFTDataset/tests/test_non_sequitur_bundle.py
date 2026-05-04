from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import tiktoken

REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR))
sys.path.insert(
    0,
    str(REPO_DIR / "augmentation" / "steer_exec_augmentation_bundle"),
)

from augmentation.steer_exec_augmentation_bundle.src.non_sequitur_source import (  # noqa: E402
    prepare_source_rows,
)
from augmentation.steer_exec_augmentation_bundle.src.prompt_routing import (  # noqa: E402
    choose_prompt_path,
)
from augmentation.steer_exec_augmentation_bundle.src.trace_augmentor import (  # noqa: E402
    TokenCounter,
    extract_trace_text,
    parse_trace_text,
    validate_generated_window,
)


def render_trace_content(*, first_steer: str, first_exec: str) -> str:
    """Build a two-pair assistant content sample for tests.

    Args:
        first_steer: First steer text.
        first_exec: First exec text.

    Returns:
        Assistant content with two steer/exec pairs.
    """

    return (
        "<think>\n"
        f"<steer>{first_steer}</steer>\n"
        "<exec>\n"
        f"{first_exec}\n"
        "</exec>\n\n"
        "<steer>Continue the local check</steer>\n"
        "<exec>\n"
        "Keep the reasoning concrete and move forward.\n"
        "</exec>\n"
        "</think>"
    )


def build_source_rows() -> list[dict[str, object]]:
    """Return small source rows for CLI and filter tests.

    Returns:
        Source rows with one eligible row, one over-limit row, and one short row.
    """

    return [
        {
            "id": "eligible-row",
            "dataset_source": "test-source",
            "messages": [
                {"role": "user", "content": "Solve the example task."},
                {
                    "role": "assistant",
                    "content": render_trace_content(
                        first_steer="Plan the first step",
                        first_exec="Inspect the givens and note the target.",
                    ),
                },
            ],
        },
        {
            "id": "over-limit-row",
            "dataset_source": "test-source",
            "messages": [
                {"role": "user", "content": "token " * 15050},
                {
                    "role": "assistant",
                    "content": render_trace_content(
                        first_steer="State a toy plan",
                        first_exec="Write a tiny local check.",
                    ),
                },
            ],
        },
        {
            "id": "short-trace-row",
            "dataset_source": "test-source",
            "messages": [
                {"role": "user", "content": "Short trace sample."},
                {
                    "role": "assistant",
                    "content": (
                        "<think>\n"
                        "<steer>Only one pair</steer>\n"
                        "<exec>\n"
                        "This row should be filtered out.\n"
                        "</exec>\n"
                        "</think>"
                    ),
                },
            ],
        },
    ]


def write_jsonl(*, path: Path, rows: list[dict[str, object]]) -> None:
    """Write rows to JSONL.

    Args:
        path: Output JSONL path.
        rows: Rows to serialize.

    Returns:
        None.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def run_cli(*, source_path: Path, output_dir: Path) -> subprocess.CompletedProcess[str]:
    """Run the non-sequitur builder in mock mode.

    Args:
        source_path: Source JSONL for the run.
        output_dir: Temporary output directory.

    Returns:
        Completed subprocess result.
    """

    return subprocess.run(
        [
            sys.executable,
            str(
                REPO_DIR
                / "augmentation"
                / "steer_exec_augmentation_bundle"
                / "scripts"
                / "build_non_sequitur_dataset.py"
            ),
            "--source",
            str(source_path),
            "--output-dir",
            str(output_dir),
            "--target-rows",
            "1",
            "--seed",
            "11",
            "--max-concurrency",
            "1",
            "--mock-intervention",
            "--max-source-tokens",
            "14000",
        ],
        cwd=str(REPO_DIR),
        text=True,
        capture_output=True,
        check=False,
    )


def assert_catalog_and_prompt_routing() -> None:
    """Check the catalog count and prompt-template override behavior."""

    catalog_path = (
        REPO_DIR
        / "augmentation"
        / "steer_exec_augmentation_bundle"
        / "non-sequiturs.json"
    )
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    interventions = catalog["interventions"]
    assert len(interventions) == 25
    assert (
        catalog["sampling_defaults"]["first_new_steer_should_match_variant_exactly"]
        is False
    )
    prompts_dir = (
        REPO_DIR / "augmentation" / "steer_exec_augmentation_bundle" / "prompts"
    )
    override_path = choose_prompt_path(
        prompts_dir=prompts_dir,
        mode="insert",
        intervention_spec={"prompt_template": "non_sequitur_insert.md"},
    )
    default_path = choose_prompt_path(
        prompts_dir=prompts_dir,
        mode="insert",
        intervention_spec={},
    )
    assert override_path.name == "non_sequitur_insert.md"
    assert default_path.name == "insert.md"


def assert_relaxed_first_steer_validation() -> None:
    """Check that non-sequitur validation allows steer variation."""

    token_counter = TokenCounter(None)
    raw_window = {
        "blocks": [
            {"type": "steer", "text": "Try a side route"},
            {
                "type": "exec",
                "text": "Take a brief detour, carry out the local action, and keep it concrete.",
            },
        ]
    }
    _, strict_errors = validate_generated_window(
        obj=raw_window,
        requested_pairs=1,
        required_first_steer="Run a side calc",
        enforce_first_steer_exact=True,
        token_counter=token_counter,
        exec_token_limit=512,
    )
    _, relaxed_errors = validate_generated_window(
        obj=raw_window,
        requested_pairs=1,
        required_first_steer="Run a side calc",
        enforce_first_steer_exact=False,
        token_counter=token_counter,
        exec_token_limit=512,
    )
    assert strict_errors
    assert not relaxed_errors


def assert_source_filtering(rows: list[dict[str, object]]) -> None:
    """Check the strict `< 14000` source filter and trace eligibility.

    Args:
        rows: Synthetic source rows.

    Returns:
        None.
    """

    encoding = tiktoken.get_encoding("cl100k_base")
    prepared_rows, prep_summary = prepare_source_rows(
        rows=rows,
        encoding=encoding,
        max_source_tokens=14000,
    )
    assert prep_summary.total_rows == 3
    assert prep_summary.eligible_rows == 1
    assert prep_summary.skipped_over_token_limit == 1
    assert prep_summary.skipped_short_trace == 1
    assert prepared_rows[0].row["id"] == "eligible-row"
    assert prepared_rows[0].complete_prompt_token_count < 14000


def assert_cli_output(
    *, output_dir: Path, result: subprocess.CompletedProcess[str]
) -> None:
    """Validate one mock CLI run end-to-end.

    Args:
        output_dir: Output directory used by the run.
        result: Completed subprocess result.

    Returns:
        None.
    """

    assert result.returncode == 0, result.stderr
    assert "[non_sequitur]" in result.stdout
    output_path = output_dir / "augmented_1_all.jsonl"
    summary_path = output_dir / "augmented_1.summary.json"
    row = json.loads(output_path.read_text(encoding="utf-8").splitlines()[0])
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert row["augmentation_meta"]["plan_type"] == "insert_only"
    assert row["augmentation_meta"]["runner_mode"] == "non_sequitur_insert_only"
    assert row["augmentation_meta"]["status"] == "merge_ready"
    assert "regen_seed" not in row
    assert isinstance(row["complete_prompt_token_count"], int)
    assert row["augmentation_meta"]["source_row_id"] == "eligible-row"
    assert summary["source_prep"]["eligible_rows"] == 1
    assert summary["row_count"] == 1
    assert row["messages"][-1]["content"].startswith("<think>\n<steer>")
    assert "</exec>\n</think>" in row["messages"][-1]["content"]
    generated_blocks = row["augmentation_meta"]["steps"][0]["generated_blocks"]
    mask_targets = row["augmentation_meta"]["mask_targets"]
    expected_texts = [
        generated_blocks[index]["text"]
        for index in mask_targets["generated_steer_block_indexes_local"]
    ]
    assert expected_texts == mask_targets["generated_steer_texts"]
    parsed_blocks = parse_trace_text(extract_trace_text(row))
    final_block_indexes = mask_targets["final_trace_block_indexes"]
    generated_steer_texts = mask_targets["generated_steer_texts"]
    assert len(final_block_indexes) == len(generated_steer_texts)
    for block_index, steer_text in zip(final_block_indexes, generated_steer_texts):
        assert parsed_blocks[block_index].type == "steer"
        assert parsed_blocks[block_index].text == steer_text
    assistant_content = row["messages"][-1]["content"]
    char_ranges = mask_targets["final_string_char_ranges"]
    assert len(char_ranges) == len(generated_steer_texts)
    for char_range, steer_text in zip(char_ranges, generated_steer_texts):
        masked_text = assistant_content[char_range["start"] : char_range["end"]]
        assert not masked_text.startswith("<steer>")
        assert masked_text.endswith("</steer>")
        assert masked_text == f"{steer_text}</steer>"


def main() -> None:
    """Run non-sequitur catalog, filter, and CLI smoke tests."""

    assert_catalog_and_prompt_routing()
    assert_relaxed_first_steer_validation()
    rows = build_source_rows()
    assert_source_filtering(rows=rows)
    with tempfile.TemporaryDirectory(prefix="non_sequitur_bundle_") as tmpdir:
        tmp_path = Path(tmpdir)
        source_path = tmp_path / "source.jsonl"
        output_dir = tmp_path / "out"
        write_jsonl(path=source_path, rows=rows)
        result = run_cli(source_path=source_path, output_dir=output_dir)
        assert_cli_output(output_dir=output_dir, result=result)
    print("ok: non-sequitur bundle smoke test passed")


if __name__ == "__main__":
    main()
