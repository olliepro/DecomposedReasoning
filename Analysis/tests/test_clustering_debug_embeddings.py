"""Tests for clustering_debug_embeddings."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from clustering_debug_embeddings import (
    ClusterCount,
    ClusterGroup,
    PromptClusterItem,
    SuccessfulClusterAttempt,
    TextClusterSummary,
    extract_successful_cluster_attempts,
    normalized_embedding_matrix,
    parse_prompt_items,
    sample_diverse_text_summaries,
    summaries_for_attempt,
)


def test_parse_prompt_items_extracts_clean_text() -> None:
    prompt = "\n".join(
        [
            "header",
            "## Options to group:",
            "1: count=2 | text=Try factoring </steer>",
            "2: count=1 | text=Check parity",
        ]
    )

    items = parse_prompt_items(prompt=prompt)

    assert [item.item_id for item in items] == [1, 2]
    assert [item.source_count for item in items] == [2, 1]
    assert [item.text for item in items] == ["Try factoring", "Check parity"]


def test_extract_successful_cluster_attempts_filters_failures(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "clustering_debug.jsonl"
    prompt = "\n".join(
        [
            "header",
            "## Options to group:",
            "1: count=1 | text=Try factoring",
            "2: count=1 | text=Check parity",
        ]
    )
    failed_raw_text = (
        '{"groups":[{"name":"Factor","key":"factor"}],"assignments":{"1":"factor"}}'
    )
    successful_raw_text = json.dumps(
        {
            "groups": [
                {"name": "Factor", "key": "factor"},
                {"name": "Parity", "key": "parity"},
            ],
            "assignments": {"1": "factor", "2": "parity"},
        }
    )
    rows = [
        {
            "event": "attempt_raw_response",
            "attempt_number": 1,
            "item_count": 2,
            "model_id": "demo-model",
            "prompt": prompt,
            "raw_text": failed_raw_text,
        },
        {
            "event": "attempt_failure",
            "attempt_number": 1,
            "item_count": 2,
            "model_id": "demo-model",
            "error": "missing assignment",
        },
        {
            "event": "attempt_raw_response",
            "attempt_number": 2,
            "item_count": 2,
            "model_id": "demo-model",
            "prompt": prompt,
            "raw_text": successful_raw_text,
        },
        {
            "event": "attempt_success",
            "attempt_number": 2,
            "item_count": 2,
            "model_id": "demo-model",
            "assignment_rate": 1.0,
        },
    ]
    input_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    attempts = extract_successful_cluster_attempts(input_path=input_path)

    assert len(attempts) == 1
    assert attempts[0].attempt_number == 2
    assert attempts[0].assignments == {1: "factor", 2: "parity"}


def test_sample_diverse_text_summaries_uses_frequency_then_distance() -> None:
    summaries = (
        TextClusterSummary(
            text="anchor",
            occurrences=9,
            attempts=4,
            cluster_counts=(ClusterCount("anchor", "Anchor", 9),),
        ),
        TextClusterSummary(
            text="left",
            occurrences=3,
            attempts=3,
            cluster_counts=(ClusterCount("left", "Left", 3),),
        ),
        TextClusterSummary(
            text="right",
            occurrences=2,
            attempts=2,
            cluster_counts=(ClusterCount("right", "Right", 2),),
        ),
    )
    embeddings_by_text = {
        "anchor": [1.0, 0.0],
        "left": [0.0, 1.0],
        "right": [-1.0, 0.0],
    }

    sampled = sample_diverse_text_summaries(
        summaries=summaries,
        embeddings_by_text=embeddings_by_text,
        sample_count=3,
    )

    assert [summary.text for summary in sampled] == ["anchor", "right", "left"]


def test_summaries_for_attempt_stay_within_one_prompt() -> None:
    attempt = SuccessfulClusterAttempt(
        success_index=3,
        attempt_number=1,
        model_id="demo-model",
        items=(
            PromptClusterItem(item_id=1, source_count=7, text="Visualize polygon path"),
            PromptClusterItem(item_id=2, source_count=1, text="Final sanity check"),
        ),
        groups=(
            ClusterGroup(key="visualize_polygon_path", name="Visualize Polygon Path"),
            ClusterGroup(key="final_sanity_check", name="Final Sanity Check"),
        ),
        assignments={1: "visualize_polygon_path", 2: "final_sanity_check"},
    )

    summaries = summaries_for_attempt(attempt=attempt)

    assert [summary.text for summary in summaries] == [
        "Visualize polygon path",
        "Final sanity check",
    ]
    assert [summary.occurrences for summary in summaries] == [7, 1]
    assert summaries[0].cluster_counts[0].cluster_key == "visualize_polygon_path"


def test_normalized_embedding_matrix_normalizes_full_vectors() -> None:
    embeddings_by_text = {
        "alpha": [3.0, 4.0],
        "beta": [5.0, 12.0],
    }

    matrix = normalized_embedding_matrix(
        texts=["alpha", "beta"],
        embeddings_by_text=embeddings_by_text,
    )

    expected = np.asarray([[0.6, 0.8], [5.0 / 13.0, 12.0 / 13.0]], dtype=np.float32)
    assert matrix.shape == (2, 2)
    assert np.allclose(matrix, expected)
