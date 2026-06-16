#!/usr/bin/env python3
"""Generate events-only HTML visualization for one branching_eval run."""

from __future__ import annotations

import json
import shutil
import sys
import time
from dataclasses import dataclass
from html import escape
from pathlib import Path

ANALYSIS_ROOT = Path(__file__).resolve().parents[1]
if str(ANALYSIS_ROOT) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_ROOT))

from branching_eval.artifact_store import ArtifactStore
from branching_eval.event_db import EVENT_DB_FILENAME, event_source_signature
from branching_eval.event_types import EventEnvelope

try:
    from scripts.visualize_branching_interactive import (
        render_tree_workspace,
        tree_workspace_script,
    )
    from scripts.visualize_branching_replay import (
        AttemptKey,
        AttemptState,
        LeafView,
        RenderSummary,
        parse_args,
        read_events_lenient,
        replay_attempts,
        selected_attempts_by_doc,
    )
    from scripts.visualize_branching_payload import (
        node_detail_payloads_for_attempt,
        tree_payload_for_attempt,
    )
    from scripts.visualize_branching_web import wrap_page
except ModuleNotFoundError:
    from visualize_branching_interactive import (
        render_tree_workspace,
        tree_workspace_script,
    )
    from visualize_branching_replay import (
        AttemptKey,
        AttemptState,
        LeafView,
        RenderSummary,
        parse_args,
        read_events_lenient,
        replay_attempts,
        selected_attempts_by_doc,
    )
    from visualize_branching_payload import (
        node_detail_payloads_for_attempt,
        tree_payload_for_attempt,
    )
    from visualize_branching_web import wrap_page


@dataclass(frozen=True)
class ProgressAttemptView:
    """Lightweight per-doc summary row sourced from progress snapshots."""

    doc_id: int
    doc_attempt: int
    task_name: str
    model_id: str
    selector_mode: str
    rollout_mode: str
    status: str
    leaf_count: int
    passrate: float
    avg_token_length: float
    correct_count: int
    incorrect_count: int
    natural_count: int
    max_count: int
    repeating_count: int
    other_count: int
    unique_answer_count: int
    last_update_timestamp: str

    def slug(self) -> str:
        """Return filesystem-safe page slug for this attempt."""

        return (
            f"doc_{self.doc_id}_attempt_{self.doc_attempt}_"
            f"{self.task_name}_{self.model_id}_{self.selector_mode}"
        ).replace("/", "_")

    def label(self) -> str:
        """Return concise display label."""

        return f"doc {self.doc_id} · attempt {self.doc_attempt}"


def render_attempt_page(
    *,
    run_dir: Path,
    state: AttemptState,
    output_path: Path,
    graph_path: Path,
) -> None:
    """Render one per-attempt HTML page."""

    detail_base_url = f"data/{state.key.slug()}"
    tree_payload = tree_payload_for_attempt(
        state=state,
        detail_base_url=detail_base_url,
    )
    detail_payloads = node_detail_payloads_for_attempt(
        state=state,
        detail_base_url=detail_base_url,
    )
    write_json_atomic(path=graph_path, payload=tree_payload)
    write_node_detail_payloads(
        detail_payloads=detail_payloads,
        detail_dir=graph_path.parent / state.key.slug() / "nodes",
    )
    title = (
        f"{state.key.label()} · {state.key.task_name} · "
        f"{state.key.model_id} · {state.key.selector_mode}"
    )
    subtitle = f"{run_dir.name} · events-only replay"
    status_class = status_badge_class(status=state.status())
    metrics_html = render_metrics_panel(state=state)
    leaves_html = render_leaves_panel(state=state)
    tree_html = render_tree_workspace(graph_path=f"data/{graph_path.name}")
    body_html = f"""
<section class="panel">
  <h1>{escape(title)}</h1>
  <p class="muted" style="margin:0.45rem 0 0.9rem 0">
    Attempt replay is sourced from <code>tree_events.sqlite</code>.
  </p>
  <div class="pill-row">
    <span class="pill {status_class}">{escape(state.status())}</span>
    <span class="pill">events={state.event_count}</span>
    <span class="pill">nodes={len(state.nodes)}</span>
    <span class="pill">edges={len(state.edges)}</span>
    <span class="pill">leaves={state.leaf_count()}</span>
    <span class="pill">vllm={state.vllm_request_count}/{state.vllm_response_count}</span>
    <span class="pill">vllm_errors={state.vllm_error_count}</span>
    <span class="pill">last_event_index={state.last_event_index}</span>
  </div>
</section>
{metrics_html}
{tree_html}
{leaves_html}
<section class="panel">
  <a href="../index.html">Back to run index</a>
</section>
"""
    footer = f"Last event timestamp: {state.last_timestamp_utc or 'n/a'}"
    page = wrap_page(
        title=title,
        subtitle=subtitle,
        body_html=body_html,
        footer_text=footer,
        script=tree_workspace_script(),
    )
    write_text_atomic(path=output_path, text=page)


def render_metrics_panel(*, state: AttemptState) -> str:
    """Render status/diagnostic summary panel for one attempt."""

    diagnostics = state.finished_payload.get("diagnostics", {})
    doc_metrics = state.finished_payload.get("doc_metrics", {})
    if not isinstance(diagnostics, dict):
        diagnostics = {}
    if not isinstance(doc_metrics, dict):
        doc_metrics = {}
    metric_pills = "".join(
        f"<span class='pill'>{escape(str(key))}={escape(_format_metric(value))}</span>"
        for key, value in sorted(doc_metrics.items())
    )
    diag_pills = "".join(
        f"<span class='pill'>{escape(str(key))}={escape(_format_metric(value))}</span>"
        for key, value in sorted(diagnostics.items())
        if isinstance(value, (int, float, str))
    )
    resumed_text = state.resumed_reason if state.resumed_reason else "none"
    return f"""
<section class="panel">
  <h2>Doc Summary</h2>
  <p class="muted" style="margin:0.45rem 0">resumed_reason={escape(resumed_text)}</p>
  <div class="pill-row">{metric_pills or "<span class='pill'>no doc metrics</span>"}</div>
  <div class="pill-row" style="margin-top:0.55rem">
    {diag_pills or "<span class='pill'>no scalar diagnostics</span>"}
  </div>
</section>
"""


def render_leaves_panel(*, state: AttemptState) -> str:
    """Render scored leaf table panel."""

    sorted_leaves = sorted_leaves_for_table(state=state)
    heatmap_html = render_verification_stop_heatmap(leaves=sorted_leaves)
    rows = []
    for leaf in sorted_leaves:
        metrics_text = ", ".join(
            f"{key}={_format_metric(value)}"
            for key, value in sorted(leaf.task_metrics.items())
        )
        rows.append(
            "<tr>"
            f"<td><code>{escape(leaf.leaf_id)}</code></td>"
            f"<td><code>{escape(leaf.node_id)}</code></td>"
            f"<td>{'' if leaf.verification is None else leaf.verification}</td>"
            f"<td>{'' if leaf.length_tokens_total is None else leaf.length_tokens_total}</td>"
            f"<td>{escape(stop_reason_display_label(stop_reason=leaf.stop_reason))}</td>"
            f"<td>{escape(metrics_text)}</td>"
            "</tr>"
        )
    rows_html = (
        "".join(rows) if rows else "<tr><td colspan='6'>no leaf_scored events</td></tr>"
    )
    return f"""
<section class="panel">
  <h2>Scored Leaves</h2>
  {heatmap_html}
  <table>
    <thead>
      <tr><th>leaf_id</th><th>node</th><th>verify</th><th>length</th><th>stop</th><th>metrics</th></tr>
    </thead>
    <tbody>{rows_html}</tbody>
  </table>
</section>
"""


def sorted_leaves_for_table(*, state: AttemptState) -> list[LeafView]:
    """Return leaves sorted for scored table output.

    Args:
        state: Replayed attempt state containing scored leaves.

    Returns:
        Sorted leaves with verify-first ordering and `think_end` stop priority.

    Example:
        >>> from scripts.visualize_branching_replay import AttemptKey, AttemptState, LeafView
        >>> state = AttemptState(key=AttemptKey(0, 0, "t", "m", "s"))
        >>> state.leaves = {"a": LeafView("a", "n", "", 0, 3, "length", {}), "b": LeafView("b", "n", "", 1, 3, "think_end", {})}
        >>> [leaf.leaf_id for leaf in sorted_leaves_for_table(state=state)]
        ['b', 'a']
    """

    return sorted(state.leaves.values(), key=leaf_table_sort_key)


def leaf_table_sort_key(leaf: LeafView) -> tuple[int, int, tuple[int, str], str]:
    """Return ordering key for scored-leaf table rows."""

    verify_rank = verification_sort_rank(verification=leaf.verification)
    think_end_rank = stop_reason_think_end_rank(stop_reason=leaf.stop_reason)
    stop_key = stop_reason_sort_key(stop_reason=leaf.stop_reason)
    return (-verify_rank, -think_end_rank, stop_key, leaf.leaf_id)


def verification_sort_rank(*, verification: int | None) -> int:
    """Return verify rank where 1 (correct) > 0 (incorrect) > unknown."""

    if verification == 1:
        return 2
    if verification == 0:
        return 1
    return 0


def stop_reason_think_end_rank(*, stop_reason: str) -> int:
    """Return stop-reason rank with `think_end` prioritized."""

    return 1 if normalized_stop_reason(stop_reason=stop_reason) == "think_end" else 0


def stop_reason_sort_key(*, stop_reason: str) -> tuple[int, str]:
    """Return sortable stop-reason key with `think_end` first then lexical."""

    normalized = normalized_stop_reason(stop_reason=stop_reason)
    think_end_first = 0 if normalized == "think_end" else 1
    return (think_end_first, normalized)


def normalized_stop_reason(*, stop_reason: str) -> str:
    """Return normalized stop reason used for sorting/grouping."""

    return str(stop_reason).strip().lower()


def stop_reason_icon(*, stop_reason: str) -> str:
    """Return stop-reason emoji used in table/heatmap labels.

    Args:
        stop_reason: Raw stop reason from leaf payloads.

    Returns:
        Emoji string describing the stop type, or an empty string.

    Example:
        >>> stop_reason_icon(stop_reason="repeated_exec_block_loop")
        '🔁'
    """

    normalized = normalized_stop_reason(stop_reason=stop_reason)
    if not normalized:
        return ""
    if normalized == "missing_steer_or_think_close":
        return "⚠️"
    if normalized.startswith("repeated_") and normalized.endswith("_block_loop"):
        return "🔁"
    if "length" in normalized or normalized == "max_gen_toks_reached":
        return "🛑"
    return "🏁"


def stop_reason_display_label(*, stop_reason: str) -> str:
    """Return user-facing stop-reason label with emoji prefix.

    Args:
        stop_reason: Raw stop reason from leaf payloads.

    Returns:
        Formatted stop reason label, preserving raw text and adding an icon.
    """

    reason_text = str(stop_reason).strip()
    if not reason_text:
        return ""
    icon = stop_reason_icon(stop_reason=reason_text)
    return f"{icon} {reason_text}" if icon else reason_text


def render_verification_stop_heatmap(*, leaves: list[LeafView]) -> str:
    """Render verify-vs-stop count heatmap HTML."""

    rows_by_verify = [("correct", 1), ("incorrect", 0)]
    scored_leaves = [leaf for leaf in leaves if leaf.verification in {0, 1}]
    if not scored_leaves:
        return "<p class='muted' style='margin:0.45rem 0 0.8rem 0'>No correct/incorrect rows for heatmap.</p>"

    stop_reasons = sorted(
        {leaf.stop_reason for leaf in scored_leaves},
        key=lambda reason: stop_reason_sort_key(stop_reason=reason),
    )
    counts = {
        (label, stop_reason): 0
        for label, _ in rows_by_verify
        for stop_reason in stop_reasons
    }
    for leaf in scored_leaves:
        label = "correct" if leaf.verification == 1 else "incorrect"
        counts[(label, leaf.stop_reason)] += 1
    max_count = max(counts.values()) if counts else 0
    header_cells = "".join(
        f"<th>{escape(stop_reason_display_label(stop_reason=stop_reason))}</th>"
        for stop_reason in stop_reasons
    )
    body_rows = []
    for label, verify_value in rows_by_verify:
        row_cells = []
        for stop_reason in stop_reasons:
            count = counts[(label, stop_reason)]
            style = heatmap_cell_style(
                count=count,
                max_count=max_count,
                verification=verify_value,
            )
            row_cells.append(f"<td style='{style}'>{count}</td>")
        body_rows.append(f"<tr><th scope='row'>{label}</th>{''.join(row_cells)}</tr>")
    return f"""
  <h3 style="margin:0.28rem 0 0.55rem 0">Verification x Stop Reason</h3>
  <table style="margin-bottom:0.7rem">
    <thead><tr><th>verify \\ stop</th>{header_cells}</tr></thead>
    <tbody>{''.join(body_rows)}</tbody>
  </table>
"""


def heatmap_cell_style(*, count: int, max_count: int, verification: int) -> str:
    """Return inline CSS style string for one heatmap cell."""

    scale = (count / max_count) if max_count > 0 else 0.0
    alpha = 0.1 + (0.55 * scale)
    if verification == 1:
        color = f"rgba(96, 211, 148, {alpha:.3f})"
    else:
        color = f"rgba(239, 71, 111, {alpha:.3f})"
    return (
        f"background:{color};"
        "text-align:center;"
        "font-family:'IBM Plex Mono',monospace;"
    )


def status_badge_class(*, status: str) -> str:
    """Return CSS class for attempt status badge."""

    if status in {"complete", "completed"}:
        return "good"
    if status == "incomplete":
        return "warn"
    return "bad"


def verification_counts_text(*, state: AttemptState) -> str:
    """Return answer-correct/answer-incorrect counts for scored leaves.

    Args:
        state: Attempt replay state containing scored leaves.

    Returns:
        Text formatted as `{correct}/{incorrect}`.
    """

    answer_values = [
        value
        for leaf in state.leaves.values()
        if (value := bool_metric(value=leaf.task_metrics.get("raw_answer_acc")))
        is not None
    ]
    if answer_values:
        correct_count = sum(1 for value in answer_values if value)
        return f"{correct_count}/{len(answer_values) - correct_count}"
    correct_count = sum(1 for leaf in state.leaves.values() if leaf.verification == 1)
    incorrect_count = sum(1 for leaf in state.leaves.values() if leaf.verification == 0)
    return f"{correct_count}/{incorrect_count}"


def bool_metric(*, value: object) -> bool | None:
    """Parse a boolean-ish metric logged as bool, number, or text."""

    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "1.0", "yes", "y"}:
        return True
    if normalized in {"false", "0", "0.0", "no", "n"}:
        return False
    return None


def progress_verification_counts_text(*, progress: ProgressAttemptView) -> str:
    """Return fallback `correct/incorrect` text from one progress snapshot."""

    return f"{progress.correct_count}/{progress.incorrect_count}"


def progress_finish_reason_text(*, progress: ProgressAttemptView) -> str:
    """Return a readable finish-reason summary for one progress snapshot.

    Args:
        progress: Selected progress snapshot row for one doc attempt.

    Returns:
        Human-readable finish-reason summary for index display.

    Example:
        >>> progress_finish_reason_text(progress=ProgressAttemptView(
        ...     doc_id=0, doc_attempt=0, task_name="a", model_id="m",
        ...     selector_mode="s", rollout_mode="r", status="complete",
        ...     leaf_count=4, passrate=0.5, avg_token_length=12.0,
        ...     correct_count=2, incorrect_count=2,
        ...     natural_count=1, max_count=2, repeating_count=1,
        ...     other_count=0, unique_answer_count=3,
        ...     last_update_timestamp="2026-01-01T00:00:00+00:00",
        ... ))
        'mixed(natural=1, max=2, repeating=1)'
    """

    labeled_counts = [
        ("natural", progress.natural_count),
        ("max", progress.max_count),
        ("repeating", progress.repeating_count),
        ("other", progress.other_count),
    ]
    nonzero_counts = [
        (label, count) for label, count in labeled_counts if int(count) > 0
    ]
    if not nonzero_counts:
        return "none"
    if len(nonzero_counts) == 1 and nonzero_counts[0][1] == progress.leaf_count:
        return nonzero_counts[0][0]
    detail_text = ", ".join(f"{label}={count}" for label, count in nonzero_counts)
    return f"mixed({detail_text})"


def read_progress_attempts(*, run_dir: Path) -> list[ProgressAttemptView]:
    """Read compact per-doc progress snapshots for one run."""

    payloads = ArtifactStore(run_dir=run_dir).read_doc_progress_snapshots()
    attempts: list[ProgressAttemptView] = []
    for payload in payloads:
        attempts.append(
            ProgressAttemptView(
                doc_id=int(payload.get("doc_id", 0)),
                doc_attempt=int(payload.get("doc_attempt", 0)),
                task_name=str(payload.get("task_name", "")),
                model_id=str(payload.get("model_id", "")),
                selector_mode=str(payload.get("selector_mode", "")),
                rollout_mode=str(payload.get("rollout_mode", "")),
                status=str(payload.get("status", "incomplete")),
                leaf_count=int(payload.get("leaf_count", 0)),
                passrate=float(payload.get("passrate", 0.0)),
                avg_token_length=float(payload.get("avg_token_length", 0.0)),
                correct_count=int(payload.get("correct_count", 0)),
                incorrect_count=int(payload.get("incorrect_count", 0)),
                natural_count=int(payload.get("natural_count", 0)),
                max_count=int(payload.get("max_count", 0)),
                repeating_count=int(payload.get("repeating_count", 0)),
                other_count=int(payload.get("other_count", 0)),
                unique_answer_count=int(payload.get("unique_answer_count", 0)),
                last_update_timestamp=str(payload.get("last_update_timestamp", "")),
            )
        )
    return attempts


def selected_progress_attempts_by_doc(
    *, progress_attempts: list[ProgressAttemptView]
) -> list[ProgressAttemptView]:
    """Select one preferred progress row per doc.

    Completed attempts win over incomplete attempts. Within the same status
    bucket, later attempts win.
    """

    selected_by_doc: dict[int, ProgressAttemptView] = {}
    for progress in progress_attempts:
        existing = selected_by_doc.get(progress.doc_id)
        if existing is None:
            selected_by_doc[progress.doc_id] = progress
            continue
        progress_rank = (
            1 if progress.status in {"complete", "completed"} else 0,
            progress.doc_attempt,
            progress.last_update_timestamp,
        )
        existing_rank = (
            1 if existing.status in {"complete", "completed"} else 0,
            existing.doc_attempt,
            existing.last_update_timestamp,
        )
        if progress_rank > existing_rank:
            selected_by_doc[progress.doc_id] = progress
    return [selected_by_doc[doc_id] for doc_id in sorted(selected_by_doc)]


def render_index_page(
    *,
    run_dir: Path,
    output_dir: Path,
    events: list[EventEnvelope],
    states: dict[AttemptKey, AttemptState],
    selected_states: list[AttemptState],
    selected_progress_attempts: list[ProgressAttemptView],
) -> None:
    """Render run-level index page linking all replayed attempt pages."""

    source_path = str(run_dir / EVENT_DB_FILENAME)
    selected_rows = "".join(
        index_selected_row_html(progress=progress)
        for progress in selected_progress_attempts
    )
    if not selected_rows:
        selected_rows = "".join(
            index_selected_fallback_row_html(state=state) for state in selected_states
        )
    if not selected_rows:
        selected_rows = (
            "<tr><td colspan='9'>No completed or partial doc attempts found.</td></tr>"
        )
    all_rows = "".join(
        index_all_attempt_row_html(state=state)
        for state in sorted(
            states.values(),
            key=lambda row: (row.key.doc_id, row.key.doc_attempt, row.last_event_index),
        )
    )
    if not all_rows:
        all_rows = "<tr><td colspan='9'>No attempt events found.</td></tr>"
    body_html = f"""
<section class="panel">
  <h1>Run Replay: {escape(run_dir.name)}</h1>
  <p class="muted path-row">
    <span class="path-label">Source of truth:</span>
    <code class="path-code" title="{escape(source_path)}">{escape(source_path)}</code>
  </p>
  <div class="pill-row">
    <span class="pill">events={len(events)}</span>
    <span class="pill">attempts={len(states)}</span>
    <span class="pill">selected_docs={len(selected_progress_attempts) if selected_progress_attempts else len(selected_states)}</span>
  </div>
</section>
<section class="panel">
  <h2>Default Doc View (Latest Completed Attempt)</h2>
  <table>
    <thead>
      <tr><th>doc</th><th>attempt</th><th>status</th><th>mode</th><th>selector</th><th>passrate</th><th>avg_tok</th><th>correct/incorrect</th><th>unique_answers</th><th>view</th></tr>
    </thead>
    <tbody>{selected_rows}</tbody>
  </table>
</section>
<section class="panel">
  <h2>All Attempts</h2>
  <table>
    <thead>
      <tr><th>doc</th><th>attempt</th><th>status</th><th>task</th><th>model</th><th>selector</th><th>correct/incorrect</th><th>last_event</th><th>view</th></tr>
    </thead>
    <tbody>{all_rows}</tbody>
  </table>
</section>
"""
    page = wrap_page(
        title=f"Branching Replay · {run_dir.name}",
        subtitle="events-only replay",
        body_html=body_html,
        footer_text="Regenerated from tree_events.sqlite without tree snapshots.",
    )
    write_text_atomic(path=output_dir / "index.html", text=page)


def index_selected_row_html(*, progress: ProgressAttemptView) -> str:
    """Build one selected-doc row sourced from a progress snapshot."""

    rel_link = f"docs/{progress.slug()}.html"
    status = progress.status
    verification_counts = progress_verification_counts_text(progress=progress)
    return (
        "<tr>"
        f"<td>{progress.doc_id}</td>"
        f"<td>{progress.doc_attempt}</td>"
        f"<td><span class='pill {status_badge_class(status=status)}'>{escape(status)}</span></td>"
        f"<td><code>{escape(progress.rollout_mode)}</code></td>"
        f"<td><code>{escape(progress.selector_mode)}</code></td>"
        f"<td>{escape(_format_metric(progress.passrate))}</td>"
        f"<td>{escape(_format_metric(progress.avg_token_length))}</td>"
        f"<td>{verification_counts}</td>"
        f"<td>{progress.unique_answer_count}</td>"
        f"<td><a href='{escape(rel_link)}'>open</a></td>"
        "</tr>"
    )


def index_selected_fallback_row_html(*, state: AttemptState) -> str:
    """Build one selected-doc row from replay state when progress is unavailable."""

    rel_link = f"docs/{state.key.slug()}.html"
    status = state.status()
    verification_counts = verification_counts_text(state=state)
    return (
        "<tr>"
        f"<td>{state.key.doc_id}</td>"
        f"<td>{state.key.doc_attempt}</td>"
        f"<td><span class='pill {status_badge_class(status=status)}'>{escape(status)}</span></td>"
        f"<td><code>unknown</code></td>"
        f"<td><code>{escape(state.key.selector_mode)}</code></td>"
        f"<td>-</td>"
        f"<td>-</td>"
        f"<td>{verification_counts}</td>"
        f"<td>-</td>"
        f"<td><a href='{escape(rel_link)}'>open</a></td>"
        "</tr>"
    )


def index_all_attempt_row_html(*, state: AttemptState) -> str:
    """Build one all-attempt row for index page."""

    rel_link = f"docs/{state.key.slug()}.html"
    status = state.status()
    verification_counts = verification_counts_text(state=state)
    return (
        "<tr>"
        f"<td>{state.key.doc_id}</td>"
        f"<td>{state.key.doc_attempt}</td>"
        f"<td><span class='pill {status_badge_class(status=status)}'>{escape(status)}</span></td>"
        f"<td><code>{escape(state.key.task_name)}</code></td>"
        f"<td><code>{escape(state.key.model_id)}</code></td>"
        f"<td><code>{escape(state.key.selector_mode)}</code></td>"
        f"<td>{verification_counts}</td>"
        f"<td>{state.last_event_index}</td>"
        f"<td><a href='{escape(rel_link)}'>open</a></td>"
        "</tr>"
    )


def write_summary_json(
    *,
    run_dir: Path,
    output_dir: Path,
    events: list[EventEnvelope],
    states: dict[AttemptKey, AttemptState],
    selected_states: list[AttemptState],
    selected_progress_attempts: list[ProgressAttemptView],
) -> None:
    """Write machine-readable replay summary JSON."""

    payload = {
        "run_dir": str(run_dir),
        "event_count": len(events),
        "attempt_count": len(states),
        "selected_doc_count": (
            len(selected_progress_attempts)
            if selected_progress_attempts
            else len(selected_states)
        ),
        "selected_attempts": (
            [
                {
                    "doc_id": progress.doc_id,
                    "doc_attempt": progress.doc_attempt,
                    "task_name": progress.task_name,
                    "model_id": progress.model_id,
                    "selector_mode": progress.selector_mode,
                    "rollout_mode": progress.rollout_mode,
                    "status": progress.status,
                    "leaf_count": progress.leaf_count,
                    "passrate": progress.passrate,
                    "avg_token_length": progress.avg_token_length,
                    "correct_count": progress.correct_count,
                    "incorrect_count": progress.incorrect_count,
                    "natural_count": progress.natural_count,
                    "max_count": progress.max_count,
                    "repeating_count": progress.repeating_count,
                    "other_count": progress.other_count,
                    "unique_answer_count": progress.unique_answer_count,
                    "last_update_timestamp": progress.last_update_timestamp,
                }
                for progress in selected_progress_attempts
            ]
            if selected_progress_attempts
            else [
                {
                    "doc_id": state.key.doc_id,
                    "doc_attempt": state.key.doc_attempt,
                    "task_name": state.key.task_name,
                    "model_id": state.key.model_id,
                    "selector_mode": state.key.selector_mode,
                    "status": state.status(),
                    "leaf_count": state.leaf_count(),
                    "node_count": len(state.nodes),
                    "vllm_request_count": state.vllm_request_count,
                    "vllm_response_count": state.vllm_response_count,
                    "vllm_error_count": state.vllm_error_count,
                    "last_event_index": state.last_event_index,
                }
                for state in selected_states
            ]
        ),
    }
    write_text_atomic(
        path=output_dir / "summary.json",
        text=json.dumps(payload, indent=2) + "\n",
    )


def write_text_atomic(*, path: Path, text: str) -> None:
    """Atomically write text file by temp-write then replace."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(text, encoding="utf-8")
    temp_path.replace(path)


def write_json_atomic(*, path: Path, payload: dict[str, object]) -> None:
    """Atomically write one JSON payload file."""

    write_text_atomic(
        path=path,
        text=json.dumps(payload, indent=2) + "\n",
    )


def write_node_detail_payloads(
    *,
    detail_payloads: dict[str, dict[str, object]],
    detail_dir: Path,
) -> None:
    """Write per-node lazy inspector payload files."""

    if detail_dir.parent.exists():
        shutil.rmtree(detail_dir.parent)
    for node_id, payload in detail_payloads.items():
        write_json_atomic(
            path=detail_dir / f"{node_id}.json",
            payload=payload,
        )


def render_snapshot(*, run_dir: Path, output_dir: Path) -> RenderSummary:
    """Render one snapshot from current SQLite event content."""

    events = read_events_lenient(path=run_dir)
    states = replay_attempts(events=events)
    selected_states = selected_attempts_by_doc(states=states)
    progress_attempts = read_progress_attempts(run_dir=run_dir)
    selected_progress_attempts = selected_progress_attempts_by_doc(
        progress_attempts=progress_attempts
    )
    docs_dir = output_dir / "docs"
    graph_dir = docs_dir / "data"
    docs_dir.mkdir(parents=True, exist_ok=True)
    graph_dir.mkdir(parents=True, exist_ok=True)
    for state in states.values():
        render_attempt_page(
            run_dir=run_dir,
            state=state,
            output_path=docs_dir / f"{state.key.slug()}.html",
            graph_path=graph_dir / f"{state.key.slug()}.json",
        )
    render_index_page(
        run_dir=run_dir,
        output_dir=output_dir,
        events=events,
        states=states,
        selected_states=selected_states,
        selected_progress_attempts=selected_progress_attempts,
    )
    write_summary_json(
        run_dir=run_dir,
        output_dir=output_dir,
        events=events,
        states=states,
        selected_states=selected_states,
        selected_progress_attempts=selected_progress_attempts,
    )
    return RenderSummary(
        event_count=len(events),
        attempt_count=len(states),
        selected_doc_count=(
            len(selected_progress_attempts)
            if selected_progress_attempts
            else len(selected_states)
        ),
    )


def event_file_signature(*, path: Path) -> tuple[int, int]:
    """Return `(size_bytes, mtime_ns)` signature for polling reloads."""

    if not path.exists():
        return (0, 0)
    stat = path.stat()
    return (int(stat.st_size), int(stat.st_mtime_ns))


def _format_metric(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def main() -> None:
    """Run CLI entrypoint for event-only visualization rendering."""

    args = parse_args()
    run_dir = args.run_dir.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir else run_dir / "viz"
    poll_seconds = max(0.1, float(args.poll_seconds))
    last_signature: tuple[int, int] | None = None
    iteration = 0
    while True:
        signature = event_source_signature(run_dir=run_dir)
        if signature != last_signature:
            summary = render_snapshot(run_dir=run_dir, output_dir=output_dir)
            last_signature = signature
            print(
                "[viz] rendered "
                f"events={summary.event_count} "
                f"attempts={summary.attempt_count} "
                f"selected_docs={summary.selected_doc_count}"
            )
        if not args.follow:
            break
        iteration += 1
        if args.max_follow_iterations is not None and iteration >= int(
            args.max_follow_iterations
        ):
            break
        time.sleep(poll_seconds)


if __name__ == "__main__":
    main()
