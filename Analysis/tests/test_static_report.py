"""Tests for cluster-driven static report payload and HTML output."""

from __future__ import annotations

from static_report import (
    align_rows_to_target_text,
    build_report_payload,
    render_report_html,
)


def test_payload_dedupes_candidates_and_counts_occurrences() -> None:
    """Payload should dedupe cluster strings and preserve occurrence counts."""
    payload = build_report_payload(
        config={"model": "m"},
        steps=[{"step_index": 0, "selected_candidate_index": 1}],
        candidates=[
            {"step_index": 0, "candidate_index": 0, "text": "Try substitution</steer>"},
            {"step_index": 0, "candidate_index": 1, "text": "Try substitution</steer>"},
            {"step_index": 0, "candidate_index": 2, "text": "Factor first</steer>"},
        ],
        token_stats=[],
        final_text="done",
    )
    step = payload["step_views"][0]
    assert step["candidate_count"] == 3
    assert step["cluster_count"] >= 1
    first_cluster = step["clusters"][0]
    total_counts = sum(int(item["count"]) for item in first_cluster["items"])
    assert total_counts == int(first_cluster["count"])


def test_payload_includes_sorted_rollout_metric_distributions() -> None:
    """Payload should expose rollout probability and entropy samples for percentiles."""
    payload = build_report_payload(
        config={"model": "m"},
        steps=[],
        candidates=[],
        token_stats=[
            {
                "source": "rollout",
                "step_index": 0,
                "candidate_index": -1,
                "token_index": 1,
                "token": "a",
                "probability": 0.9,
                "entropy": 1.4,
                "alternatives": [],
            },
            {
                "source": "candidate",
                "step_index": 0,
                "candidate_index": 0,
                "token_index": 0,
                "token": "b",
                "probability": 0.3,
                "alternatives": [],
            },
            {
                "source": "rollout",
                "step_index": 0,
                "candidate_index": -1,
                "token_index": 0,
                "token": "c",
                "probability": 0.1,
                "entropy": 0.2,
                "alternatives": [],
            },
        ],
        final_text="",
    )
    assert payload["rollout_probabilities"] == [0.1, 0.9]
    assert payload["rollout_entropies"] == [0.2, 1.4]
    assert payload["trajectory_token_count"] == 2


def test_payload_removes_close_tag_tokens_and_extracts_execution_text() -> None:
    """Chosen execution text should come from `final_text` exec blocks by step."""
    payload = build_report_payload(
        config={"model": "m"},
        steps=[
            {
                "step_index": 0,
                "selected_candidate_index": 0,
                "selected_text": "x</steer>",
            }
        ],
        candidates=[{"step_index": 0, "candidate_index": 0, "text": "x</steer>"}],
        token_stats=[
            {
                "source": "candidate",
                "step_index": 0,
                "candidate_index": 0,
                "token_index": 0,
                "token": "x",
                "probability": 0.8,
                "alternatives": [],
            },
            {
                "source": "candidate",
                "step_index": 0,
                "candidate_index": 0,
                "token_index": 1,
                "token": "</steer>",
                "probability": 0.8,
                "alternatives": [],
            },
            {
                "source": "candidate",
                "step_index": 0,
                "candidate_index": 0,
                "token_index": 2,
                "token": "<exec>",
                "probability": 0.8,
                "alternatives": [],
            },
            {
                "source": "candidate",
                "step_index": 0,
                "candidate_index": 0,
                "token_index": 3,
                "token": "hello",
                "probability": 0.7,
                "alternatives": [],
            },
            {
                "source": "candidate",
                "step_index": 0,
                "candidate_index": 0,
                "token_index": 4,
                "token": "</exec>",
                "probability": 0.7,
                "alternatives": [],
            },
            {
                "source": "rollout",
                "step_index": 0,
                "candidate_index": -1,
                "token_index": 0,
                "token": " wrong-step-token",
                "probability": 0.6,
                "alternatives": [],
            },
            {
                "source": "rollout",
                "step_index": 1,
                "candidate_index": -1,
                "token_index": 0,
                "token": " right-step-token",
                "probability": 0.6,
                "alternatives": [],
            },
        ],
        final_text=(
            "<think>\n"
            "<steer>x</steer>\n"
            "<exec>\n"
            "from final\n"
            "</exec>\n"
            "</think>"
        ),
    )
    chosen = payload["step_views"][0]["chosen_entry"]
    assert [token["token"] for token in chosen["tokens"]] == ["x"]
    assert [token["token"] for token in chosen["full_tokens"]] == [
        "x",
        "</steer>",
        "<exec>",
        "hello",
        "</exec>",
    ]
    assert [token["token"] for token in chosen["rollout_tokens"]] == [
        " right-step-token"
    ]
    assert chosen["execution_text"] == "from final"


def test_payload_aligns_execution_text_by_matching_steer_text() -> None:
    """Execution extraction should align by steer text and avoid off-by-one shifts."""

    payload = build_report_payload(
        config={"model": "m"},
        steps=[
            {
                "step_index": 0,
                "selected_candidate_index": 0,
                "selected_text": "Target steer</steer>",
            }
        ],
        candidates=[
            {"step_index": 0, "candidate_index": 0, "text": "Target steer</steer>"}
        ],
        token_stats=[],
        final_text=(
            "<think>\n"
            "<steer>Warmup steer</steer>\n"
            "<exec>\n"
            "warmup exec\n"
            "</exec>\n"
            "<steer>Target steer</steer>\n"
            "<exec>\n"
            "target exec\n"
            "</exec>\n"
            "</think>"
        ),
    )
    chosen = payload["step_views"][0]["chosen_entry"]
    assert chosen["execution_text"] == "target exec"


def test_payload_extracts_execution_text_from_execute_tags() -> None:
    """Execution extraction should accept `<execute>...</execute>` blocks."""

    payload = build_report_payload(
        config={"model": "m"},
        steps=[
            {
                "step_index": 0,
                "selected_candidate_index": 0,
                "selected_text": "Target steer</steer>",
            }
        ],
        candidates=[
            {"step_index": 0, "candidate_index": 0, "text": "Target steer</steer>"}
        ],
        token_stats=[],
        final_text=(
            "<think>\n"
            "<steer>Target steer</steer>\n"
            "<execute>\n"
            "target execute block\n"
            "</execute>\n"
            "</think>"
        ),
    )
    chosen = payload["step_views"][0]["chosen_entry"]
    assert chosen["execution_text"] == "target execute block"


def test_payload_preserves_rollout_token_file_order_in_trajectory() -> None:
    """Trajectory token stream should preserve rollout row order across chunks."""

    payload = build_report_payload(
        config={"model": "m"},
        steps=[],
        candidates=[],
        token_stats=[
            {
                "source": "rollout",
                "step_index": 0,
                "candidate_index": -1,
                "token_index": 0,
                "token": "A",
                "probability": 0.8,
                "entropy": 0.3,
                "alternatives": [],
            },
            {
                "source": "rollout",
                "step_index": 0,
                "candidate_index": -1,
                "token_index": 1,
                "token": "B",
                "probability": 0.7,
                "entropy": 0.4,
                "alternatives": [],
            },
            {
                "source": "rollout",
                "step_index": 0,
                "candidate_index": -1,
                "token_index": 0,
                "token": "C",
                "probability": 0.6,
                "entropy": 0.5,
                "alternatives": [],
            },
            {
                "source": "rollout",
                "step_index": 0,
                "candidate_index": -1,
                "token_index": 1,
                "token": "D",
                "probability": 0.5,
                "entropy": 0.6,
                "alternatives": [],
            },
        ],
        final_text="ABCD",
    )
    assert [token["token"] for token in payload["trajectory_tokens"]] == [
        "A",
        "B",
        "C",
        "D",
    ]


def test_payload_trims_rollout_overflow_using_prefix_boundaries() -> None:
    """Trajectory token rows should drop rollout/candidate overflow past step bounds."""

    payload = build_report_payload(
        config={"model": "m"},
        steps=[
            {
                "step_index": 0,
                "prefix_char_end": 1,
                "selected_candidate_index": 0,
                "selected_text": "b",
            },
            {
                "step_index": 1,
                "prefix_char_end": 3,
                "selected_candidate_index": -1,
                "selected_text": "",
                "terminated": True,
                "termination_reason": "think_end",
            },
        ],
        candidates=[{"step_index": 0, "candidate_index": 0, "text": "b"}],
        token_stats=[
            {
                "source": "rollout",
                "step_index": 0,
                "candidate_index": -1,
                "token_index": 0,
                "token": "a",
                "probability": 0.8,
                "entropy": 0.3,
                "alternatives": [],
            },
            {
                "source": "rollout",
                "step_index": 0,
                "candidate_index": -1,
                "token_index": 1,
                "token": "X",
                "probability": 0.7,
                "entropy": 0.4,
                "alternatives": [],
            },
            {
                "source": "candidate",
                "step_index": 0,
                "candidate_index": 0,
                "token_index": 0,
                "token": "b",
                "probability": 0.9,
                "entropy": 0.2,
                "alternatives": [],
            },
            {
                "source": "candidate",
                "step_index": 0,
                "candidate_index": 0,
                "token_index": 1,
                "token": "Y",
                "probability": 0.6,
                "entropy": 0.5,
                "alternatives": [],
            },
            {
                "source": "rollout",
                "step_index": 1,
                "candidate_index": -1,
                "token_index": 0,
                "token": "c",
                "probability": 0.5,
                "entropy": 0.6,
                "alternatives": [],
            },
            {
                "source": "rollout",
                "step_index": 1,
                "candidate_index": -1,
                "token_index": 1,
                "token": "Z",
                "probability": 0.4,
                "entropy": 0.7,
                "alternatives": [],
            },
        ],
        final_text="abc",
    )
    assert [token["token"] for token in payload["trajectory_tokens"]] == [
        "a",
        "b",
        "c",
    ]


def test_align_rows_preserves_suffix_after_inserted_symbol() -> None:
    """Alignment should insert missing symbol while preserving source-token suffixes."""

    rows = [
        {"token": "f(5.25)"},
        {"token": " -"},
        {"token": "4.86"},
        {"token": ","},
    ]
    aligned = align_rows_to_target_text(
        rows=rows, target_text="f(5.25) ≈ -4.86,"
    )
    assert "".join(str(row.get("token", "")) for row in aligned) == "f(5.25) ≈ -4.86,"
    assert any(bool(row.get("synthetic")) for row in aligned)
    assert any(
        not bool(row.get("synthetic")) and str(row.get("token", "")) == "4.86"
        for row in aligned
    )


def test_align_rows_handles_source_deletion_without_synthetic_fallback() -> None:
    """Alignment should drop deleted source spans while keeping real-token metadata."""

    rows = [{"token": "abcXYZdef", "probability": 0.7}]
    aligned = align_rows_to_target_text(rows=rows, target_text="abcdef")
    assert "".join(str(row.get("token", "")) for row in aligned) == "abcdef"
    assert not any(bool(row.get("synthetic")) for row in aligned)
    assert all(float(row.get("probability", 0.0)) == 0.7 for row in aligned)


def test_payload_uses_selected_text_in_step_header() -> None:
    """Step payload should expose cleaned selected steer text for collapsible headers."""
    payload = build_report_payload(
        config={"model": "m"},
        steps=[
            {
                "step_index": 3,
                "selected_candidate_index": 9,
                "selected_text": "Plan next move</steer>",
            }
        ],
        candidates=[],
        token_stats=[],
        final_text="done",
    )
    assert payload["step_views"][0]["selected_text"] == "Plan next move"


def test_payload_strips_truncated_steer_suffix_text() -> None:
    """Payload should strip truncated trailing steer-close fragments."""
    raw_text = 'Try "A.L.G.O.R.I.T.H.M" pun</ste '
    payload = build_report_payload(
        config={"model": "m"},
        steps=[
            {
                "step_index": 0,
                "selected_candidate_index": 0,
                "selected_text": raw_text,
            }
        ],
        candidates=[{"step_index": 0, "candidate_index": 0, "text": raw_text}],
        token_stats=[],
        final_text="",
    )
    step = payload["step_views"][0]
    assert step["selected_text"] == 'Try "A.L.G.O.R.I.T.H.M" pun'
    assert step["chosen_entry"]["text"] == 'Try "A.L.G.O.R.I.T.H.M" pun'


def test_payload_removes_partial_close_tag_tokens() -> None:
    """Chosen token strip should drop trailing partial close-tag tokens."""
    payload = build_report_payload(
        config={"model": "m"},
        steps=[
            {
                "step_index": 0,
                "selected_candidate_index": 0,
                "selected_text": "x</ste",
            }
        ],
        candidates=[{"step_index": 0, "candidate_index": 0, "text": "x</ste"}],
        token_stats=[
            {
                "source": "candidate",
                "step_index": 0,
                "candidate_index": 0,
                "token_index": 0,
                "token": "x",
                "probability": 0.8,
                "alternatives": [],
            },
            {
                "source": "candidate",
                "step_index": 0,
                "candidate_index": 0,
                "token_index": 1,
                "token": "</",
                "probability": 0.8,
                "alternatives": [],
            },
            {
                "source": "candidate",
                "step_index": 0,
                "candidate_index": 0,
                "token_index": 2,
                "token": "ste",
                "probability": 0.8,
                "alternatives": [],
            },
        ],
        final_text="",
    )
    chosen = payload["step_views"][0]["chosen_entry"]
    assert [token["token"] for token in chosen["tokens"]] == ["x"]


def test_payload_omits_blank_terminal_step() -> None:
    """Payload should drop terminal rows that have no selected text/candidates."""
    payload = build_report_payload(
        config={"model": "m"},
        steps=[
            {
                "step_index": 0,
                "selected_candidate_index": 0,
                "selected_text": "A</steer>",
            },
            {
                "step_index": 1,
                "selected_candidate_index": -1,
                "selected_text": "",
                "total_candidates": 0,
                "unique_candidate_count": 0,
                "terminated": True,
                "termination_reason": "think_end",
            },
        ],
        candidates=[{"step_index": 0, "candidate_index": 0, "text": "A</steer>"}],
        token_stats=[],
        final_text="<think><steer>A</steer><exec>x</exec></think>",
    )
    step_indices = [int(view["step_index"]) for view in payload["step_views"]]
    assert step_indices == [0]


def test_payload_extracts_post_think_final_answer_text() -> None:
    """Payload should expose text that appears after `</think>`."""
    payload = build_report_payload(
        config={"model": "m"},
        steps=[],
        candidates=[],
        token_stats=[],
        final_text="<think>trace</think>\nFinal answer here.",
    )
    assert payload["final_answer_text"] == "Final answer here."


def test_render_html_contains_cluster_interaction_hooks() -> None:
    """HTML should include step timeline, tooltips, and cluster view labels."""
    payload = build_report_payload(
        config={"model": "m"},
        steps=[],
        candidates=[],
        token_stats=[],
        final_text="",
    )
    html = render_report_html(report_payload=payload)
    assert "id='timeline'" in html
    assert "id='final-answer'" in html
    assert "id='tooltip'" in html
    assert "Candidate Clusters" in html
    assert "<exec>" in html


def test_render_html_escapes_script_breakers() -> None:
    """Renderer should escape script-breaking payload content."""
    payload = build_report_payload(
        config={"model": "m"},
        steps=[],
        candidates=[],
        token_stats=[],
        final_text="</script><script>alert(1)</script>\u2028\u2029",
    )
    html = render_report_html(report_payload=payload)
    assert "\\u003c/script>\\u003cscript>alert(1)\\u003c/script>" in html
    assert "</script><script>" not in html
    assert "\u2028" not in html
    assert "\u2029" not in html
