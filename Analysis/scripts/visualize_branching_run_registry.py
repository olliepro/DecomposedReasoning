"""Run discovery and index HTML for the dynamic branching viewer."""

from __future__ import annotations

import hashlib
import os
import re
import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from html import escape
from pathlib import Path

from branching_eval.event_db import EVENT_DB_FILENAME
from scripts.visualize_branching_web import wrap_page


@dataclass(frozen=True)
class RunRegistryEntry:
    """One SQLite-backed run exposed by the dynamic viewer."""

    run_id: str
    run_dir: Path

    def url_path(self) -> str:
        """Return the route prefix for this run."""

        return f"/runs/{self.run_id}/"


@dataclass(frozen=True)
class RunRegistrySummary:
    """Compact row displayed on the multi-run landing page."""

    run_id: str
    run_name: str
    run_dir: Path
    event_count: int | None
    attempt_count: int | None
    selected_doc_count: int | None
    avg_problem_passrate: float | None
    problem_passrate_count: int

    def count_text(self, *, value: int | None) -> str:
        """Return a table-safe count display string."""

        return "n/a" if value is None else str(value)

    def passrate_text(self) -> str:
        """Return the average passrate display text."""

        if self.avg_problem_passrate is None:
            return "n/a"
        return f"{self.avg_problem_passrate:.4f} (n={self.problem_passrate_count})"


@dataclass(frozen=True)
class RunNameParts:
    """Parsed display fields for one generated run directory name."""

    task: str
    model: str
    size: str
    variant: str
    selector: str
    started_at: str

    def search_text(self, *, run_name: str) -> str:
        """Return lowercase text used by the client-side run filter."""

        return " ".join(
            [
                run_name,
                self.task,
                self.model,
                self.size,
                self.selector,
                self.variant,
                self.started_at,
            ]
        ).lower()


class RunRegistry:
    """Thread-safe registry of run directories served by one HTTP process."""

    def __init__(
        self,
        *,
        run_dirs: list[Path],
        run_roots: list[Path],
        run_root_prefixes: list[str] | None = None,
        latest_batch_only: bool = False,
    ) -> None:
        self._explicit_run_dirs = [path.resolve() for path in run_dirs]
        self._run_roots = [path.resolve() for path in run_roots]
        self._run_root_prefixes = tuple(run_root_prefixes or [])
        self._latest_batch_only = latest_batch_only
        self._lock = threading.Lock()
        self._entries: dict[str, RunRegistryEntry] = {}
        self._refreshed_at = 0.0
        self._refresh_after_seconds = 300.0
        self._refreshing = False

    def refresh(self) -> None:
        """Refresh discovered runs from roots and explicit run directories."""

        with self._lock:
            if self._refreshing:
                return
            self._refreshing = True
        try:
            self._replace_entries(run_dirs=self._current_run_dirs())
        finally:
            with self._lock:
                self._refreshing = False

    def warm(self) -> None:
        """Populate the initial registry before accepting browser requests."""

        self.refresh()

    def warm_async(self) -> None:
        """Populate the registry in the background after the server binds."""

        self._refresh_if_stale(force=False)

    def _replace_entries(self, *, run_dirs: list[Path]) -> None:
        entries = _entries_for_run_dirs(run_dirs=run_dirs)
        with self._lock:
            self._entries = {entry.run_id: entry for entry in entries}
            self._refreshed_at = time.monotonic()

    def entries(self) -> list[RunRegistryEntry]:
        """Return entries sorted by newest run directory mtime first."""

        self._refresh_if_stale(force=False)
        with self._lock:
            return sorted(
                self._entries.values(),
                key=lambda entry: entry.run_dir.stat().st_mtime,
                reverse=True,
            )

    def single_entry(self) -> RunRegistryEntry | None:
        """Return the only entry when this server is in single-run mode."""

        if self._run_roots or len(self._explicit_run_dirs) != 1:
            return None
        entries = self.entries()
        if len(entries) != 1:
            return None
        return entries[0]

    def entry_for_id(self, *, run_id: str) -> RunRegistryEntry | None:
        """Return an entry by route id."""

        self._refresh_if_stale(force=False)
        with self._lock:
            entry = self._entries.get(run_id)
            has_entries = bool(self._entries)
        if entry is not None or has_entries:
            return entry
        self.refresh()
        with self._lock:
            return self._entries.get(run_id)

    def _refresh_if_stale(self, *, force: bool) -> None:
        with self._lock:
            fresh = time.monotonic() - self._refreshed_at < self._refresh_after_seconds
            has_entries = bool(self._entries)
            already_refreshing = self._refreshing
        if not force and fresh:
            return
        if force:
            self.refresh()
            return
        if already_refreshing:
            return
        thread = threading.Thread(target=self.refresh, daemon=True)
        thread.start()

    def _current_run_dirs(self) -> list[Path]:
        run_dirs = list(self._explicit_run_dirs)
        for root in self._run_roots:
            run_dirs.extend(
                _discover_run_dirs(
                    run_root=root,
                    child_name_prefixes=self._run_root_prefixes,
                    latest_batch_only=self._latest_batch_only,
                )
            )
        return _dedupe_existing_run_dirs(run_dirs=run_dirs)


def split_run_route(
    *, registry: RunRegistry, path: str
) -> tuple[RunRegistryEntry, str] | tuple[None, None]:
    """Split a `/runs/<run_id>/...` path into run entry and inner route."""

    match = re.fullmatch(r"/runs/([^/]+)(/.*)?", path)
    if match is None:
        return None, None
    entry = registry.entry_for_id(run_id=match.group(1))
    if entry is None:
        return None, None
    inner_path = match.group(2) or "/"
    return entry, inner_path


def run_registry_page_html(
    *, entries: list[RunRegistryEntry], summaries: list[RunRegistrySummary]
) -> str:
    """Return the multi-run landing page HTML."""

    row_inputs = _summary_rows_with_duplicate_flags(summaries=summaries)
    duplicate_count = sum(1 for _, is_duplicate in row_inputs if is_duplicate)
    rows = "".join(
        _summary_row(summary=summary, is_duplicate=is_duplicate)
        for summary, is_duplicate in row_inputs
    )
    if not rows:
        rows = "<tr><td colspan='8'>No SQLite-backed runs found.</td></tr>"
    body_html = f"""
<style>
.run-filter {{ width:100%; max-width:42rem; margin-top:0.8rem; }}
.run-filter input {{
  width:100%;
  border:1px solid var(--line);
  border-radius:0.55rem;
  background:rgba(8, 12, 22, 0.62);
  color:var(--text);
  padding:0.55rem 0.65rem;
  font:inherit;
}}
	.run-name {{
	  display:block;
	  max-width:36rem;
	  overflow:hidden;
	  text-overflow:ellipsis;
	  white-space:nowrap;
	}}
	#run-table td[data-pivot] {{
	  vertical-align:top;
	  border-right:1px solid var(--line);
	}}
	</style>
<section class="panel">
  <h1>Branching Runs</h1>
  <p class="muted" style="margin:0.45rem 0 0.9rem 0">
    Select a SQLite-backed run.
  </p>
  <div class="pill-row">
    <span class="pill">sqlite_runs={len(entries)}</span>
    <span class="pill" id="run-visible-count">visible={len(entries)}</span>
    <span class="pill">hidden_duplicates={duplicate_count}</span>
  </div>
  <div class="run-filter">
    <input id="run-filter-input" type="search" placeholder="Filter: 2b structured aime25 hf_direct tree_cluster eps33">
  </div>
  <label class="muted" style="display:block;margin-top:0.65rem">
    <input id="show-duplicates-input" type="checkbox"> show older duplicate runs
  </label>
</section>
<section class="panel">
  <table id="run-table">
    <thead>
      <tr><th>model</th><th>size</th><th>task</th><th>selector</th><th>started</th><th>avg doc passrate</th><th>open</th><th>variant</th></tr>
    </thead>
    <tbody>{rows}</tbody>
  </table>
</section>
"""
    return wrap_page(
        title="Branching Runs",
        subtitle="multi-run SQLite viewer",
        body_html=body_html,
        footer_text="Each linked run is read directly from its tree_events.sqlite file.",
        script=_run_filter_script(),
        style_url="/assets/theme.css",
    )


def _summary_rows_with_duplicate_flags(
    *, summaries: list[RunRegistrySummary]
) -> list[tuple[RunRegistrySummary, bool]]:
    seen: set[tuple[str, str, str, str]] = set()
    rows: list[tuple[RunRegistrySummary, bool]] = []
    for summary in sorted(summaries, key=_summary_sort_key):
        parts = _parse_run_name(run_name=summary.run_name)
        key = (parts.model, parts.size, parts.task, parts.selector)
        is_duplicate = key in seen
        seen.add(key)
        rows.append((summary, is_duplicate))
    return rows


def _summary_sort_key(
    summary: RunRegistrySummary,
) -> tuple[str, float, str, str, float, float, str]:
    parts = _parse_run_name(run_name=summary.run_name)
    return (
        parts.model,
        _size_sort_value(size=parts.size),
        parts.task,
        parts.selector,
        _started_at_desc_value(run_name=summary.run_name),
        _step_desc_value(run_name=summary.run_name),
        summary.run_name,
    )


def _summary_row(*, summary: RunRegistrySummary, is_duplicate: bool) -> str:
    run_url = f"/runs/{summary.run_id}/"
    parts = _parse_run_name(run_name=summary.run_name)
    search_text = parts.search_text(run_name=summary.run_name)
    duplicate_attr = "1" if is_duplicate else "0"
    duplicate_badge = " <span class='pill warn'>older</span>" if is_duplicate else ""
    return (
        f"<tr data-search='{escape(search_text)}' data-duplicate='{duplicate_attr}' "
        f"data-model='{escape(parts.model)}' data-size='{escape(parts.size)}' "
        f"data-task='{escape(parts.task)}' data-selector='{escape(parts.selector)}'>"
        f"<td data-pivot='model'><code>{escape(parts.model)}</code></td>"
        f"<td data-pivot='size'><code>{escape(parts.size)}</code></td>"
        f"<td data-pivot='task'><code>{escape(parts.task)}</code></td>"
        f"<td data-pivot='selector'><code>{escape(parts.selector)}</code></td>"
        f"<td><code>{escape(parts.started_at)}</code></td>"
        f"<td>{escape(summary.passrate_text())}</td>"
        f"<td><a href='{escape(run_url)}'>open</a></td>"
        f"<td><code>{escape(parts.variant)}</code>{duplicate_badge}"
        f"<code class='muted run-name' title='{escape(summary.run_name)}'>{escape(summary.run_name)}</code></td>"
        "</tr>"
    )


def _run_filter_script() -> str:
    return """
const input = document.getElementById("run-filter-input");
const showDuplicates = document.getElementById("show-duplicates-input");
const counter = document.getElementById("run-visible-count");
const rows = Array.from(document.querySelectorAll("#run-table tbody tr[data-search]"));
const pivotColumns = ["model", "size", "task", "selector"];
function applyRunFilter() {
  const query = (input?.value || "").trim().toLowerCase();
  const includeDuplicates = Boolean(showDuplicates?.checked);
  let visible = 0;
  for (const row of rows) {
    const matchesQuery = query === "" || row.dataset.search.includes(query);
    const matchesDuplicate = includeDuplicates || row.dataset.duplicate !== "1";
    const keep = matchesQuery && matchesDuplicate;
    row.style.display = keep ? "" : "none";
    if (keep) visible += 1;
  }
  if (counter) counter.textContent = `visible=${visible}`;
  applyPivotRowspans();
}
function applyPivotRowspans() {
  resetPivotCells();
  const visibleRows = rows.filter((row) => row.style.display !== "none");
  for (let columnIndex = 0; columnIndex < pivotColumns.length; columnIndex += 1) {
    const column = pivotColumns[columnIndex];
    let anchor = null;
    let activeKey = "";
    let span = 0;
    for (const row of visibleRows) {
      const cell = row.querySelector(`[data-pivot="${column}"]`);
      const key = pivotColumns
        .slice(0, columnIndex + 1)
        .map((name) => row.dataset[name] || "")
        .join("\\u001f");
      if (anchor !== null && key === activeKey) {
        cell.style.display = "none";
        span += 1;
        anchor.rowSpan = span;
      } else {
        anchor = cell;
        activeKey = key;
        span = 1;
        cell.rowSpan = 1;
      }
    }
  }
}
function resetPivotCells() {
  for (const row of rows) {
    for (const cell of row.querySelectorAll("[data-pivot]")) {
      cell.style.display = "";
      cell.rowSpan = 1;
    }
  }
}
input?.addEventListener("input", applyRunFilter);
showDuplicates?.addEventListener("change", applyRunFilter);
applyRunFilter();
"""


def _parse_run_name(*, run_name: str) -> RunNameParts:
    task = _first_match(pattern=r"^(aime\d+)_", value=run_name, fallback="unknown")
    if task == "unknown" and re.search(r"(?:^|/)qwen35_", run_name):
        task = "rl"
    model = _first_match(
        pattern=r"(?:^|_)(qwen35)_", value=run_name, fallback="unknown"
    )
    size = _first_match(
        pattern=r"(?:^|_)qwen35_(0p8b|2b|4b)_",
        value=run_name,
        fallback="?",
    )
    raw_timestamp = _raw_started_at(run_name=run_name)
    return RunNameParts(
        task=task,
        model=model,
        size=size,
        variant=_variant_label(run_name=run_name),
        selector=_selector_label(run_name=run_name),
        started_at=_human_started_at(raw_timestamp=raw_timestamp),
    )


def _first_match(*, pattern: str, value: str, fallback: str) -> str:
    match = re.search(pattern, value)
    return fallback if match is None else match.group(1)


def _raw_started_at(*, run_name: str) -> str:
    return _first_match(
        pattern=r"_(\d{8}T\d{6}(?:\.\d+)?Z)(?:$|[^0-9A-Za-z])",
        value=run_name,
        fallback="",
    )


def _size_sort_value(*, size: str) -> float:
    if size == "?":
        return float("inf")
    return float(size.removesuffix("b").replace("p", "."))


def _started_at_desc_value(*, run_name: str) -> float:
    raw_timestamp = _raw_started_at(run_name=run_name)
    if not raw_timestamp:
        return float("inf")
    compact = raw_timestamp.removesuffix("Z")
    date_format = "%Y%m%dT%H%M%S.%f" if "." in compact else "%Y%m%dT%H%M%S"
    timestamp = datetime.strptime(compact, date_format).replace(tzinfo=timezone.utc)
    return -timestamp.timestamp()


def _step_desc_value(*, run_name: str) -> float:
    match = re.search(r"batch_(\d+)_step_(\d+)", run_name)
    if match is None:
        return 0.0
    batch_index = int(match.group(1))
    step_index = int(match.group(2))
    return -float(batch_index * 1_000_000 + step_index)


def _variant_label(*, run_name: str) -> str:
    labels = [
        ("hf4b_base", "base_hf"),
        ("base_hf", "base_hf"),
        ("hf_direct", "hf_direct"),
        ("sft_structured_baseline_rerun_one_nl", "sft_structured_rerun"),
        ("sft_structured_baseline", "sft_structured"),
        ("sft_eps33", "sft_eps33"),
        ("sft_tree_cluster", "sft_tree_cluster"),
        ("sft_tree_random", "sft_tree_random"),
        ("epsilon_greedy", "epsilon_greedy"),
        ("structured_baseline", "structured_baseline"),
        ("branching", "branching"),
    ]
    for marker, label in labels:
        if marker in run_name:
            return label
    return "unknown"


def _selector_label(*, run_name: str) -> str:
    labels = [
        ("cluster_across", "cluster_across"),
        ("topk_random", "topk_random"),
        ("branching_random", "branching_random"),
        ("epsilon_greedy", "epsilon_greedy"),
        ("structured_baseline", "structured_baseline"),
        ("branching", "branching"),
        ("baseline", "baseline"),
    ]
    for marker, label in labels:
        if marker in run_name:
            return label
    return "unknown"


def _human_started_at(*, raw_timestamp: str) -> str:
    if not raw_timestamp:
        return ""
    compact = raw_timestamp.split(".", maxsplit=1)[0].removesuffix("Z")
    timestamp = datetime.strptime(compact, "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
    return timestamp.strftime("%Y-%m-%d %H:%M UTC")


def _entries_for_run_dirs(*, run_dirs: list[Path]) -> list[RunRegistryEntry]:
    return [
        RunRegistryEntry(run_id=_run_id_for_path(run_dir=run_dir), run_dir=run_dir)
        for run_dir in run_dirs
    ]


def _discover_run_dirs(
    *,
    run_root: Path,
    child_name_prefixes: tuple[str, ...] = (),
    latest_batch_only: bool = False,
) -> list[Path]:
    if _has_readable_event_db(run_dir=run_root):
        return [run_root]
    run_dirs: list[Path] = []
    try:
        entries = list(os.scandir(run_root))
    except FileNotFoundError:
        return []
    entries = [
        entry
        for entry in entries
        if _matches_child_name_prefix(
            name=entry.name,
            child_name_prefixes=child_name_prefixes,
        )
    ]
    direct_run_dirs = [
        Path(entry.path)
        for entry in entries
        if entry.is_dir(follow_symlinks=False)
        and _has_readable_event_db(run_dir=Path(entry.path))
    ]
    batch_run_dirs = [
        run_dir for run_dir in direct_run_dirs if run_dir.name.startswith("batch_")
    ]
    if batch_run_dirs:
        if latest_batch_only:
            return [max(batch_run_dirs, key=lambda path: path.name)]
        return sorted(batch_run_dirs)
    for entry in entries:
        if not entry.is_dir(follow_symlinks=False):
            continue
        run_dir = Path(entry.path)
        if _has_readable_event_db(run_dir=run_dir):
            run_dirs.append(run_dir)
            continue
        if latest_batch_only:
            latest_child = _latest_child_run_dir(run_dir=run_dir)
            if latest_child is not None:
                run_dirs.append(latest_child)
            continue
        run_dirs.extend(_child_run_dirs(run_dir=run_dir))
    return sorted(run_dirs)


def _matches_child_name_prefix(
    *, name: str, child_name_prefixes: tuple[str, ...]
) -> bool:
    if not child_name_prefixes:
        return True
    return any(name.startswith(prefix) for prefix in child_name_prefixes)


def _child_run_dirs(*, run_dir: Path) -> list[Path]:
    """Return child directories with non-empty event DBs."""

    try:
        children = list(os.scandir(run_dir))
    except FileNotFoundError:
        return []
    candidates = [
        Path(child.path)
        for child in children
        if child.is_dir(follow_symlinks=False)
        and _has_readable_event_db(run_dir=Path(child.path))
    ]
    return sorted(candidates)


def _latest_child_run_dir(*, run_dir: Path) -> Path | None:
    """Return the newest batch child with a non-empty event DB."""

    try:
        children = sorted(
            os.scandir(run_dir),
            key=lambda child: child.name,
            reverse=True,
        )
    except FileNotFoundError:
        return None
    for child in children:
        if not child.is_dir(follow_symlinks=False):
            continue
        child_path = Path(child.path)
        if child_path.name.startswith("batch_") and _has_readable_event_db(
            run_dir=child_path
        ):
            return child_path
    return None


def _has_readable_event_db(*, run_dir: Path) -> bool:
    """Return whether a run dir has a non-empty SQLite event DB."""

    db_path = run_dir / EVENT_DB_FILENAME
    return db_path.is_file() and db_path.stat().st_size > 0


def _dedupe_existing_run_dirs(*, run_dirs: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    result: list[Path] = []
    for run_dir in run_dirs:
        if run_dir in seen or not _has_readable_event_db(run_dir=run_dir):
            continue
        seen.add(run_dir)
        result.append(run_dir)
    return result


def _run_id_for_path(*, run_dir: Path) -> str:
    digest = hashlib.sha1(str(run_dir).encode("utf-8")).hexdigest()[:8]
    return f"{_slugify(value=run_dir.name)}-{digest}"


def _slugify(*, value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")
    return slug or "run"


def _short_path(path: str) -> str:
    parts = Path(path).parts
    if len(parts) <= 5:
        return path
    return str(Path("...").joinpath(*parts[-4:]))


def summary_from_sqlite(*, entry: RunRegistryEntry) -> RunRegistrySummary:
    """Build one registry row using small aggregate SQLite queries."""

    db_path = entry.run_dir / EVENT_DB_FILENAME
    try:
        connection = sqlite3.connect(
            f"file:{db_path}?mode=ro",
            uri=True,
            timeout=0.2,
        )
        try:
            connection.execute("PRAGMA query_only=ON")
            event_count = _event_count(connection=connection)
            attempt_count = _attempt_count(connection=connection)
            selected_doc_count, avg_passrate = _selected_doc_passrate(
                connection=connection
            )
        finally:
            connection.close()
    except sqlite3.Error:
        event_count = 0
        attempt_count = 0
        selected_doc_count = 0
        avg_passrate = None
    return RunRegistrySummary(
        run_id=entry.run_id,
        run_name=_display_run_name(run_dir=entry.run_dir),
        run_dir=entry.run_dir,
        event_count=event_count,
        attempt_count=attempt_count,
        selected_doc_count=selected_doc_count,
        avg_problem_passrate=avg_passrate,
        problem_passrate_count=selected_doc_count,
    )


def summary_from_path(*, entry: RunRegistryEntry) -> RunRegistrySummary:
    """Build one run-picker row without opening the run database."""

    return RunRegistrySummary(
        run_id=entry.run_id,
        run_name=_display_run_name(run_dir=entry.run_dir),
        run_dir=entry.run_dir,
        event_count=None,
        attempt_count=None,
        selected_doc_count=None,
        avg_problem_passrate=None,
        problem_passrate_count=0,
    )


def _display_run_name(*, run_dir: Path) -> str:
    if re.fullmatch(r"batch_\d+_step_\d+", run_dir.name):
        return f"{run_dir.parent.name}/{run_dir.name}"
    return run_dir.name


def _event_count(*, connection: sqlite3.Connection) -> int:
    row = connection.execute(
        "SELECT COALESCE(MAX(event_index) + 1, 0) FROM event_log"
    ).fetchone()
    return int(row[0])


def _attempt_count(*, connection: sqlite3.Connection) -> int:
    row = connection.execute("SELECT COUNT(*) FROM doc_progress_typed").fetchone()
    progress_count = int(row[0])
    if progress_count > 0:
        return progress_count
    row = connection.execute("""
        SELECT COUNT(*)
        FROM (
            SELECT doc_id, doc_attempt, task_name, model_id, selector_mode
            FROM leaf_score
            GROUP BY doc_id, doc_attempt, task_name, model_id, selector_mode
        )
        """).fetchone()
    return int(row[0])


def _selected_doc_passrate(
    *, connection: sqlite3.Connection
) -> tuple[int, float | None]:
    leaf_count, leaf_passrate = _selected_doc_leaf_answer_passrate(
        connection=connection
    )
    if leaf_count > 0:
        return leaf_count, leaf_passrate
    return _selected_doc_progress_passrate(connection=connection)


def _selected_doc_progress_passrate(
    *, connection: sqlite3.Connection
) -> tuple[int, float | None]:
    row = connection.execute("""
        WITH ranked AS (
            SELECT
                doc_id,
                passrate,
                leaf_count,
                ROW_NUMBER() OVER (
                    PARTITION BY doc_id
                    ORDER BY
                        CASE WHEN status IN ('complete', 'completed') THEN 1 ELSE 0 END DESC,
                        doc_attempt DESC,
                        last_update_timestamp DESC
                ) AS row_rank
            FROM doc_progress_typed
            WHERE leaf_count > 0
        )
        SELECT COUNT(*), AVG(passrate)
        FROM ranked
        WHERE row_rank = 1
        """).fetchone()
    count = int(row[0])
    return count, None if row[1] is None else float(row[1])


def _selected_doc_leaf_answer_passrate(
    *, connection: sqlite3.Connection
) -> tuple[int, float | None]:
    row = connection.execute("""
        WITH leaf_attempts AS (
            SELECT
                leaf_score.doc_id, leaf_score.doc_attempt,
                leaf_score.task_name, leaf_score.model_id,
                leaf_score.selector_mode,
                COUNT(*) AS leaf_count,
                SUM(CASE
                    WHEN raw_answer.metric_value != 0 THEN 1
                    WHEN LOWER(raw_answer.metric_text) IN (
                        'true', '1', '1.0', 'yes', 'y'
                    ) THEN 1
                    ELSE 0
                END) AS answer_correct_count,
                COUNT(raw_answer.metric_text) AS answer_count,
                SUM(CASE WHEN leaf_score.verification = 1 THEN 1 ELSE 0 END)
                    AS reward_correct_count
            FROM leaf_score
            LEFT JOIN leaf_metric AS raw_answer
              ON raw_answer.leaf_event_index = leaf_score.event_index
             AND raw_answer.metric_name = 'raw_answer_acc'
            GROUP BY leaf_score.doc_id, leaf_score.doc_attempt,
                leaf_score.task_name, leaf_score.model_id,
                leaf_score.selector_mode
            HAVING leaf_count > 0
        ),
        ranked AS (
            SELECT
                leaf_attempts.*,
                ROW_NUMBER() OVER (
                    PARTITION BY leaf_attempts.doc_id
                    ORDER BY
                        leaf_attempts.doc_attempt DESC,
                        leaf_attempts.task_name,
                        leaf_attempts.model_id,
                        leaf_attempts.selector_mode
                ) AS row_rank
            FROM leaf_attempts
        ),
        selected AS (
            SELECT CASE
                WHEN answer_count > 0
                THEN answer_correct_count * 1.0 / answer_count
                ELSE reward_correct_count * 1.0 / leaf_count
            END AS passrate
            FROM ranked
            WHERE row_rank = 1
        )
        SELECT COUNT(*), AVG(passrate)
        FROM selected
        """).fetchone()
    count = int(row[0])
    return count, None if row[1] is None else float(row[1])
