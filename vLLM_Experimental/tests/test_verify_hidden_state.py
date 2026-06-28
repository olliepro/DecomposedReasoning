"""Tests for hidden-state diversity artifact verification."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vllm_experimental.verify_hidden_state import verify_run


def write_jsonl(*, path: Path, rows: list[dict[str, object]]) -> None:
    """Write compact JSONL test fixtures."""

    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_verify_run_accepts_hidden_state_evidence(tmp_path: Path) -> None:
    write_jsonl(
        path=tmp_path / "metrics.jsonl",
        rows=[
            {
                "diversity_vector_source": "model_hidden_state",
                "hidden_vector_child_count": 2,
                "pool_hidden_pairwise_diversity": 0.3,
            }
        ],
    )
    write_jsonl(
        path=tmp_path / "native_events_pc1_bs1.jsonl",
        rows=[
            {
                "diversity_vector_source": "model_hidden_state",
                "event": "branch_promote",
                "hidden_vector_child_count": 2,
                "pool_hidden_pairwise_diversity": 0.3,
                "selected_hidden_diversity": 0.4,
            }
        ],
    )

    result = verify_run(run_dir=tmp_path)

    assert result.metric_rows == 1
    assert result.promote_events == 1
    assert result.hidden_vector_child_count == 2
    assert result.max_pairwise_diversity == 0.3
    assert result.max_selected_diversity == 0.4


def test_verify_run_rejects_token_set_only(tmp_path: Path) -> None:
    write_jsonl(
        path=tmp_path / "metrics.jsonl",
        rows=[
            {
                "diversity_vector_source": "token_set",
                "hidden_vector_child_count": 0,
            }
        ],
    )

    with pytest.raises(AssertionError, match="model-hidden-state diversity"):
        verify_run(run_dir=tmp_path)
