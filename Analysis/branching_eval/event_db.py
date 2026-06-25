"""SQLite persistence for branching tree events and progress snapshots."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from branching_eval import event_db_sql, event_db_writer

EVENT_DB_FILENAME = "tree_events.sqlite"
NORMALIZED_METADATA_KEY = "normalized_event_index"
VISIBLE_RUNTIME_EVENT_TYPES = (
    "prompt_logged",
    "trigger_skipped_max_branch_points",
    "candidate_pool_resolved",
    "selector_applied",
    "selector_continued_inline",
    "verbalized_sampling_applied",
    "malformed_steer_decision",
    "repeat_forced_think_close",
    "leaf_completed",
    "leaf_scored",
    "doc_diagnostics_recorded",
)
GRAPH_EVENT_TYPES = VISIBLE_RUNTIME_EVENT_TYPES


class EventDatabase:
    """Small SQLite-backed store for canonical branching event rows."""

    def __init__(
        self,
        *,
        path: Path,
        initialize: bool = True,
    ) -> None:
        self.path = path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if initialize:
            self._initialize()

    def connect(self) -> sqlite3.Connection:
        """Open one configured SQLite connection."""

        connection = sqlite3.connect(
            str(self.path),
            timeout=30.0,
            isolation_level=None,
        )
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def ensure_fast_read_indexes(self) -> None:
        """Create lightweight indexes used by the live SQLite viewer."""

        connection = sqlite3.connect(str(self.path), timeout=1.0, isolation_level=None)
        connection.execute("PRAGMA busy_timeout=1000")
        with closing(connection):
            connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_node_event_attempt_event
                ON node_event(
                    doc_id, doc_attempt, task_name, model_id, selector_mode,
                    event_index
                )
                """)
            connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_generated_chunk_attempt_event
                ON generated_chunk(
                    doc_id, doc_attempt, task_name, model_id, selector_mode,
                    event_index
                )
                """)
            connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_node_event_graph_cover
                ON node_event(
                    doc_id, doc_attempt, task_name, model_id, selector_mode,
                    event_index, timestamp_utc, event_type, node_id, summary,
                    step_delta, token_delta, branch_point_id, leaf_id,
                    verification, stop_reason, length_tokens_total
                )
                """)
            connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_vllm_request_id
                ON vllm_request(
                    doc_id, doc_attempt, task_name, model_id, selector_mode,
                    request_id
                )
                """)
            connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_vllm_response_attempt_stream
                ON vllm_response(
                    doc_id, doc_attempt, task_name, model_id, selector_mode,
                    request_stream_id, event_index
                )
                """)

    def read_connect(self) -> sqlite3.Connection:
        """Open one read-only connection configured for low-latency viewer reads."""

        uri = f"file:{self.path.as_posix()}?mode=ro"
        connection = sqlite3.connect(uri, timeout=1.0, isolation_level=None, uri=True)
        connection.execute("PRAGMA query_only=ON")
        connection.execute("PRAGMA busy_timeout=1000")
        return connection

    def append_event_rows(
        self,
        *,
        connection: sqlite3.Connection,
        rows: Sequence[dict[str, Any]],
    ) -> None:
        """Append event rows inside one caller-managed transaction."""

        if not rows:
            return
        event_db_writer.append_normalized_rows(connection=connection, rows=rows)
        connection.executemany(
            """
            INSERT INTO event_log (
                event_index, event_version, timestamp_utc, run_id, doc_id,
                doc_attempt, task_name, model_id, selector_mode, event_type,
                payload_json, row_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [_event_insert_values(row=row) for row in rows],
        )
        event_index = max(int(row["event_index"]) for row in rows)
        _set_metadata_event_index(
            connection=connection, key=NORMALIZED_METADATA_KEY, event_index=event_index
        )

    def read_event_rows(self) -> list[dict[str, Any]]:
        """Return all event rows in event-index order."""

        with closing(self.connect()) as connection:
            cursor = connection.execute(
                "SELECT row_json FROM event_log ORDER BY event_index"
            )
            rows = [_loads_mapping(text=str(row[0])) for row in cursor.fetchall()]
            return _hydrate_compact_vllm_response_rows(connection=connection, rows=rows)

    def read_event_rows_for_attempt(
        self,
        *,
        doc_id: int,
        doc_attempt: int,
        task_name: str,
        model_id: str,
        selector_mode: str,
    ) -> list[dict[str, Any]]:
        """Return event rows for exactly one document attempt."""

        with closing(self.connect()) as connection:
            cursor = connection.execute(
                """
                SELECT row_json FROM event_log
                WHERE doc_id = ?
                  AND doc_attempt = ?
                  AND task_name = ?
                  AND model_id = ?
                  AND selector_mode = ?
                ORDER BY event_index
                """,
                (doc_id, doc_attempt, task_name, model_id, selector_mode),
            )
            rows = [_loads_mapping(text=str(row[0])) for row in cursor.fetchall()]
            return _hydrate_compact_vllm_response_rows(connection=connection, rows=rows)

    def read_attempt_key_rows(self) -> list[dict[str, Any]]:
        """Return distinct attempt keys and lightweight event counts."""

        with closing(self.read_connect()) as connection:
            _ensure_normalized_current(connection=connection)
            cursor = connection.execute("""
                SELECT
                    doc_id, doc_attempt, task_name, model_id, selector_mode,
                    COUNT(*) AS event_count,
                    MAX(event_index) AS last_event_index
                FROM event_log
                WHERE doc_id IS NOT NULL AND doc_attempt IS NOT NULL
                GROUP BY doc_id, doc_attempt, task_name, model_id, selector_mode
                ORDER BY doc_id, doc_attempt, task_name, model_id, selector_mode
                """)
            rows = cursor.fetchall()
        return [
            {
                "doc_id": int(row[0]),
                "doc_attempt": int(row[1]),
                "task_name": str(row[2]),
                "model_id": str(row[3]),
                "selector_mode": str(row[4]),
                "event_count": int(row[5]),
                "last_event_index": int(row[6]),
            }
            for row in rows
        ]

    def read_attempt_summary_rows(self) -> list[dict[str, Any]]:
        """Return attempt-level aggregate rows from indexed event metadata."""

        with closing(self.read_connect()) as connection:
            _ensure_normalized_current(connection=connection)
            cursor = connection.execute("""
                WITH attempt_summary AS (
                    SELECT
                        doc_id, doc_attempt, task_name, model_id, selector_mode,
                        COUNT(*) AS event_count,
                        MAX(event_index) AS last_event_index,
                        SUM(CASE WHEN event_type IN (
                            'doc_started', 'doc_resumed', 'rollout_started'
                        ) THEN 1 ELSE 0 END) AS started_count,
                        SUM(CASE WHEN event_type IN (
                            'doc_finished', 'rollout_finished'
                        ) THEN 1 ELSE 0 END) AS finished_count
                    FROM event_log
                    WHERE doc_id IS NOT NULL AND doc_attempt IS NOT NULL
                    GROUP BY doc_id, doc_attempt, task_name, model_id, selector_mode
                )
                SELECT
                    summary.doc_id, summary.doc_attempt, summary.task_name,
                    summary.model_id, summary.selector_mode,
                    summary.event_count, summary.last_event_index,
                    last_event.timestamp_utc,
                    summary.started_count, summary.finished_count
                FROM attempt_summary AS summary
                LEFT JOIN event_log AS last_event
                  ON last_event.event_index = summary.last_event_index
                ORDER BY
                    summary.doc_id, summary.doc_attempt, summary.task_name,
                    summary.model_id, summary.selector_mode
                """)
            rows = cursor.fetchall()
        return [
            {
                "doc_id": int(row[0]),
                "doc_attempt": int(row[1]),
                "task_name": str(row[2]),
                "model_id": str(row[3]),
                "selector_mode": str(row[4]),
                "event_count": int(row[5]),
                "last_event_index": int(row[6]),
                "last_timestamp_utc": str(row[7] or ""),
                "started_count": int(row[8] or 0),
                "finished_count": int(row[9] or 0),
            }
            for row in rows
        ]

    def read_attempt_summary_row(self, **key: Any) -> dict[str, Any] | None:
        """Return one attempt aggregate row from indexed event metadata."""

        with closing(self.read_connect()) as connection:
            cursor = connection.execute(
                event_db_sql.ATTEMPT_SUMMARY_ROW_SQL,
                _attempt_values(key=key),
            )
            row = cursor.fetchone()
        return None if row is None else _attempt_summary_row(row=row)

    def read_doc_diagnostics_row(self, **key: Any) -> dict[str, Any] | None:
        """Return the latest canonical per-doc diagnostics event for one attempt."""

        with closing(self.read_connect()) as connection:
            _ensure_normalized_current(connection=connection)
            cursor = connection.execute(
                """
                SELECT event_index, timestamp_utc, payload_json
                FROM event_log
                WHERE doc_id = ? AND doc_attempt = ?
                  AND task_name = ? AND model_id = ? AND selector_mode = ?
                  AND event_type = 'doc_diagnostics_recorded'
                ORDER BY event_index DESC
                LIMIT 1
                """,
                _attempt_values(key=key),
            )
            row = cursor.fetchone()
        if row is None:
            return None
        return _mapping_from_cursor(cursor=cursor, row=row)

    def read_leaf_score_rows(self) -> list[dict[str, Any]]:
        """Return only leaf-scored payloads plus indexed attempt keys."""

        with closing(self.read_connect()) as connection:
            _ensure_normalized_current(connection=connection)
            cursor = connection.execute("""
                SELECT
                    doc_id, doc_attempt, task_name, model_id, selector_mode,
                    payload_json
                FROM event_log
                WHERE event_type = 'leaf_scored'
                  AND doc_id IS NOT NULL
                  AND doc_attempt IS NOT NULL
                ORDER BY event_index
                """)
            rows = cursor.fetchall()
        return [
            {
                "doc_id": int(row[0]),
                "doc_attempt": int(row[1]),
                "task_name": str(row[2]),
                "model_id": str(row[3]),
                "selector_mode": str(row[4]),
                "payload": _loads_mapping(text=str(row[5])),
            }
            for row in rows
        ]

    def read_leaf_summary_rows(self) -> list[dict[str, Any]]:
        """Return attempt-level leaf aggregates from typed columns."""

        with closing(self.read_connect()) as connection:
            _ensure_normalized_current(connection=connection)
            cursor = connection.execute("""
                SELECT
                    leaf_score.doc_id, leaf_score.doc_attempt,
                    leaf_score.task_name, leaf_score.model_id,
                    leaf_score.selector_mode,
                    COUNT(*) AS leaf_count,
                    SUM(CASE WHEN verification = 1 THEN 1 ELSE 0 END),
                    SUM(CASE WHEN verification = 0 THEN 1 ELSE 0 END),
                    SUM(CASE
                        WHEN raw_answer.metric_value != 0 THEN 1
                        WHEN LOWER(raw_answer.metric_text) IN (
                            'true', '1', '1.0', 'yes', 'y'
                        ) THEN 1
                        ELSE 0
                    END),
                    COUNT(raw_answer.metric_text),
                    SUM(CASE WHEN stop_reason = 'think_end' THEN 1 ELSE 0 END),
                    SUM(CASE WHEN stop_reason = 'length' THEN 1 ELSE 0 END),
                    SUM(CASE WHEN stop_reason = 'repeated_exec_block_loop' THEN 1 ELSE 0 END),
                    SUM(CASE WHEN stop_reason NOT IN (
                        'think_end', 'length', 'repeated_exec_block_loop'
                    ) THEN 1 ELSE 0 END),
                    SUM(COALESCE(length_tokens_total, 0)),
                    SUM(CASE WHEN length_tokens_total IS NULL THEN 0 ELSE 1 END),
                    COUNT(DISTINCT text_preview)
                FROM leaf_score
                LEFT JOIN leaf_metric AS raw_answer
                  ON raw_answer.leaf_event_index = leaf_score.event_index
                 AND raw_answer.metric_name = 'raw_answer_acc'
                GROUP BY leaf_score.doc_id, leaf_score.doc_attempt,
                    leaf_score.task_name, leaf_score.model_id,
                    leaf_score.selector_mode
                """)
            rows = cursor.fetchall()
        return [_leaf_summary_row(row=row) for row in rows]

    def read_node_rows_for_attempt(self, **key: Any) -> list[dict[str, Any]]:
        """Return typed node rows for one attempt."""

        with closing(self.read_connect()) as connection:
            _ensure_normalized_current(connection=connection)
            values = _attempt_values(key=key)
            cursor = connection.execute(event_db_sql.NODE_ROWS_SQL, (*values, *values))
            return [_mapping_from_cursor(cursor=cursor, row=row) for row in cursor]

    def read_node_row_for_attempt(
        self, *, node_id: str, **key: Any
    ) -> dict[str, Any] | None:
        """Return one typed node row for one attempt."""

        with closing(self.read_connect()) as connection:
            _ensure_normalized_current(connection=connection)
            values = _attempt_values(key=key)
            cursor = connection.execute(
                event_db_sql.NODE_ROW_SQL,
                (*values, node_id, *values, node_id),
            )
            row = cursor.fetchone()
        return None if row is None else _mapping_from_cursor(cursor=cursor, row=row)

    def read_edge_rows_for_attempt(self, **key: Any) -> list[dict[str, Any]]:
        """Return typed selected-edge rows for one attempt."""

        with closing(self.read_connect()) as connection:
            _ensure_normalized_current(connection=connection)
            cursor = connection.execute(
                event_db_sql.EDGE_ROWS_SQL, _attempt_values(key=key)
            )
            return [_mapping_from_cursor(cursor=cursor, row=row) for row in cursor]

    def read_edge_path_rows_for_node(
        self, *, node_id: str, **key: Any
    ) -> list[dict[str, Any]]:
        """Return selected-edge rows from root to one node."""

        with closing(self.read_connect()) as connection:
            _ensure_normalized_current(connection=connection)
            values = _attempt_values(key=key)
            cursor = connection.execute(
                event_db_sql.EDGE_PATH_ROWS_FOR_NODE_SQL,
                (*values, node_id, *values),
            )
            return [_mapping_from_cursor(cursor=cursor, row=row) for row in cursor]

    def upsert_node_advantage_rows(self, *, rows: Sequence[dict[str, Any]]) -> None:
        """Upsert RL segment advantage rows keyed by incoming child node."""

        if not rows:
            return
        with closing(self.connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.executemany(
                event_db_sql.UPSERT_NODE_ADVANTAGE_SQL,
                [_node_advantage_values(row=row) for row in rows],
            )
            connection.commit()

    def read_node_advantage_rows_for_attempt(self, **key: Any) -> list[dict[str, Any]]:
        """Return typed RL segment advantage rows for one attempt."""

        with closing(self.read_connect()) as connection:
            if not _has_table(connection=connection, table_name="node_advantage"):
                return []
            cursor = connection.execute(
                event_db_sql.NODE_ADVANTAGE_ROWS_SQL, _attempt_values(key=key)
            )
            return [_mapping_from_cursor(cursor=cursor, row=row) for row in cursor]

    def read_node_event_rows_for_attempt(self, **key: Any) -> list[dict[str, Any]]:
        """Return graph-visible node events for one attempt."""

        with closing(self.read_connect()) as connection:
            _ensure_normalized_current(connection=connection)
            cursor = connection.execute(
                event_db_sql.NODE_EVENT_ROWS_SQL,
                _attempt_values(key=key),
            )
            rows = [_mapping_from_cursor(cursor=cursor, row=row) for row in cursor]
        return _visible_node_event_rows(rows=rows)

    def read_node_detail_rows(
        self, *, node_id: str, event_limit: int = 500, **key: Any
    ) -> list[dict[str, Any]]:
        """Return node-local event rows for one clicked node."""

        with closing(self.read_connect()) as connection:
            _ensure_normalized_current(connection=connection)
            prompt_columns, prompt_join = _prompt_context_sql_parts(
                connection=connection
            )
            sql = event_db_sql.NODE_DETAIL_ROWS_SQL.format(
                event_type_placeholders=_placeholders(
                    count=len(VISIBLE_RUNTIME_EVENT_TYPES)
                ),
                prompt_context_columns=prompt_columns,
                prompt_context_join=prompt_join,
            )
            cursor = connection.execute(
                sql,
                (
                    *_attempt_values(key=key),
                    node_id,
                    *VISIBLE_RUNTIME_EVENT_TYPES,
                    event_limit,
                ),
            )
            return [_mapping_from_cursor(cursor=cursor, row=row) for row in cursor]

    def read_node_event_row_by_index(
        self, *, event_index: int
    ) -> dict[str, Any] | None:
        """Return one runtime node-event row by event index."""

        with closing(self.read_connect()) as connection:
            _ensure_normalized_current(connection=connection)
            prompt_columns, prompt_join = _prompt_context_sql_parts(
                connection=connection
            )
            sql = event_db_sql.NODE_EVENT_ROW_BY_INDEX_SQL.format(
                prompt_context_columns=prompt_columns,
                prompt_context_join=prompt_join,
            )
            cursor = connection.execute(
                sql,
                (event_index,),
            )
            row = cursor.fetchone()
        return None if row is None else _mapping_from_cursor(cursor=cursor, row=row)

    def read_prompt_text_by_event_index(self, *, event_index: int) -> str | None:
        """Return the full logged prompt text for one prompt event."""

        with closing(self.read_connect()) as connection:
            _ensure_normalized_current(connection=connection)
            if not _has_table(connection=connection, table_name="prompt_context"):
                return None
            row = connection.execute(
                event_db_sql.PROMPT_TEXT_BY_EVENT_INDEX_SQL,
                (event_index,),
            ).fetchone()
        return None if row is None else str(row[0])

    def read_vllm_step_graph_rows_for_attempt(self, **key: Any) -> list[dict[str, Any]]:
        """Return graph-light decode vLLM rows for one attempt."""

        with closing(self.read_connect()) as connection:
            _ensure_normalized_current(connection=connection)
            cursor = connection.execute(
                event_db_sql.VLLM_STEP_GRAPH_ROWS_FOR_ATTEMPT_SQL,
                _attempt_values(key=key),
            )
            return [_mapping_from_cursor(cursor=cursor, row=row) for row in cursor]

    def read_vllm_step_rows_for_node(
        self, *, node_id: str, event_limit: int = 500, **key: Any
    ) -> list[dict[str, Any]]:
        """Return merged decode vLLM request/response rows for one node."""

        with closing(self.read_connect()) as connection:
            _ensure_normalized_current(connection=connection)
            cursor = connection.execute(
                event_db_sql.VLLM_STEP_ROWS_FOR_NODE_SQL,
                (*_attempt_values(key=key), f"decode:{node_id}", event_limit),
            )
            return [_mapping_from_cursor(cursor=cursor, row=row) for row in cursor]

    def read_vllm_step_row_by_event(self, *, event_index: int) -> dict[str, Any] | None:
        """Return one merged decode vLLM step row by event index."""

        with closing(self.read_connect()) as connection:
            _ensure_normalized_current(connection=connection)
            cursor = connection.execute(
                event_db_sql.VLLM_RESPONSE_STEP_ROW_BY_EVENT_SQL,
                (event_index,),
            )
            row = cursor.fetchone()
            if row is None:
                cursor = connection.execute(
                    event_db_sql.VLLM_PENDING_REQUEST_STEP_ROW_BY_EVENT_SQL,
                    (event_index,),
                )
                row = cursor.fetchone()
        return None if row is None else _mapping_from_cursor(cursor=cursor, row=row)

    def read_vllm_text_rows_for_node(
        self, *, node_id: str, **key: Any
    ) -> list[dict[str, Any]]:
        """Return generated first-choice text rows for one decode node."""

        with closing(self.read_connect()) as connection:
            _ensure_normalized_current(connection=connection)
            cursor = connection.execute(
                event_db_sql.VLLM_TEXT_ROWS_FOR_NODE_SQL,
                (*_attempt_values(key=key), f"decode:{node_id}"),
            )
            return [_mapping_from_cursor(cursor=cursor, row=row) for row in cursor]

    def read_vllm_token_trajectory_rows_for_node(
        self, *, node_id: str, **key: Any
    ) -> list[dict[str, Any]]:
        """Return first-choice generated token rows for one decode node."""

        return self.read_vllm_token_trajectory_rows_for_stream_id(
            request_stream_id=f"decode:{node_id}",
            **key,
        )

    def read_vllm_token_trajectory_rows_for_stream_id(
        self, *, request_stream_id: str, **key: Any
    ) -> list[dict[str, Any]]:
        """Return first-choice generated token rows for one request stream."""

        return self.read_vllm_token_trajectory_rows_for_stream_ids(
            request_stream_ids=[request_stream_id],
            **key,
        )

    def read_vllm_token_trajectory_rows_for_stream_ids(
        self, *, request_stream_ids: Sequence[str], **key: Any
    ) -> list[dict[str, Any]]:
        """Return first-choice generated token rows for request streams."""

        if not request_stream_ids:
            return []
        with closing(self.read_connect()) as connection:
            _ensure_normalized_current(connection=connection)
            placeholders = ",".join("?" for _ in request_stream_ids)
            cursor = connection.execute(
                event_db_sql.VLLM_TOKEN_TRAJECTORY_ROWS_FOR_STREAMS_SQL.format(
                    placeholders=placeholders
                ),
                (*_attempt_values(key=key), *tuple(request_stream_ids)),
            )
            return [_mapping_from_cursor(cursor=cursor, row=row) for row in cursor]

    def read_leaf_rows_for_attempt(self, **key: Any) -> list[dict[str, Any]]:
        """Return typed leaf rows for one attempt."""

        with closing(self.read_connect()) as connection:
            _ensure_normalized_current(connection=connection)
            cursor = connection.execute(
                event_db_sql.LEAF_ROWS_SQL, _attempt_values(key=key)
            )
            return [_mapping_from_cursor(cursor=cursor, row=row) for row in cursor]

    def read_baseline_leaf_rows_for_attempt(self, **key: Any) -> list[dict[str, Any]]:
        """Return baseline display leaf rows without scanning all leaf text."""

        with closing(self.read_connect()) as connection:
            _ensure_normalized_current(connection=connection)
            cursor = connection.execute(
                event_db_sql.BASELINE_LEAF_ROWS_SQL, _attempt_values(key=key)
            )
            return [_mapping_from_cursor(cursor=cursor, row=row) for row in cursor]

    def read_leaf_rows_for_node(
        self, *, node_id: str, **key: Any
    ) -> list[dict[str, Any]]:
        """Return typed scored leaves for one node."""

        with closing(self.read_connect()) as connection:
            _ensure_normalized_current(connection=connection)
            cursor = connection.execute(
                event_db_sql.LEAF_ROWS_FOR_NODE_SQL,
                (*_attempt_values(key=key), node_id),
            )
            return [_mapping_from_cursor(cursor=cursor, row=row) for row in cursor]

    def read_candidate_rows_for_events(
        self, *, event_indexes: Sequence[int]
    ) -> list[dict[str, Any]]:
        """Return candidate rows for candidate-pool event indexes."""

        if not event_indexes:
            return []
        with closing(self.read_connect()) as connection:
            placeholders = ",".join("?" for _ in event_indexes)
            cursor = connection.execute(
                event_db_sql.CANDIDATE_ROWS_FOR_EVENTS_SQL.format(
                    placeholders=placeholders
                ),
                tuple(int(index) for index in event_indexes),
            )
            return [_mapping_from_cursor(cursor=cursor, row=row) for row in cursor]

    def read_candidate_token_rows_for_events(
        self, *, event_indexes: Sequence[int]
    ) -> list[dict[str, Any]]:
        """Return token probability rows for candidate-pool event indexes."""

        if not event_indexes:
            return []
        with closing(self.read_connect()) as connection:
            placeholders = ",".join("?" for _ in event_indexes)
            cursor = connection.execute(
                event_db_sql.CANDIDATE_TOKENS_FOR_EVENTS_SQL.format(
                    placeholders=placeholders
                ),
                tuple(int(index) for index in event_indexes),
            )
            return [_mapping_from_cursor(cursor=cursor, row=row) for row in cursor]

    def read_selector_flag_rows_for_events(
        self, *, event_indexes: Sequence[int]
    ) -> list[dict[str, Any]]:
        """Return selector candidate flags for selector event indexes."""

        if not event_indexes:
            return []
        with closing(self.read_connect()) as connection:
            placeholders = ",".join("?" for _ in event_indexes)
            cursor = connection.execute(
                event_db_sql.SELECTOR_FLAGS_FOR_EVENTS_SQL.format(
                    placeholders=placeholders
                ),
                tuple(int(index) for index in event_indexes),
            )
            return [_mapping_from_cursor(cursor=cursor, row=row) for row in cursor]

    def read_selector_cluster_rows_for_events(
        self, *, event_indexes: Sequence[int]
    ) -> list[dict[str, Any]]:
        """Return selector candidate cluster assignments for selector event indexes."""

        if not event_indexes:
            return []
        self._ensure_selector_clusters_for_events(event_indexes=event_indexes)
        with closing(self.read_connect()) as connection:
            placeholders = ",".join("?" for _ in event_indexes)
            cursor = connection.execute(
                event_db_sql.SELECTOR_CLUSTERS_FOR_EVENTS_SQL.format(
                    placeholders=placeholders
                ),
                tuple(int(index) for index in event_indexes),
            )
            return [_mapping_from_cursor(cursor=cursor, row=row) for row in cursor]

    def _ensure_selector_clusters_for_events(
        self, *, event_indexes: Sequence[int]
    ) -> None:
        with closing(self.connect()) as connection:
            connection.executescript(event_db_sql.SCHEMA_SQL)
            placeholders = ",".join("?" for _ in event_indexes)
            connection.execute(
                f"""
                INSERT OR REPLACE INTO selector_candidate_cluster
                SELECT event_log.event_index,
                       cluster_assignments_by_mode.key,
                       COALESCE(
                           json_extract(assignment.value, '$.candidate_id'),
                           -1
                       ),
                       COALESCE(
                           json_extract(assignment.value, '$.cluster_name'),
                           ''
                       )
                FROM event_log,
                     json_each(
                         event_log.payload_json,
                         '$.cluster_assignments_by_mode'
                     ) AS cluster_assignments_by_mode,
                     json_each(cluster_assignments_by_mode.value) AS assignment
                WHERE event_log.event_index IN ({placeholders})
                  AND event_log.event_type IN (
                      'selector_applied', 'selector_continued_inline'
                  )
                  AND event_log.doc_id IS NOT NULL
                  AND event_log.doc_attempt IS NOT NULL
                """,
                tuple(int(index) for index in event_indexes),
            )

    def read_selector_pool_rows_for_events(
        self, *, event_indexes: Sequence[int]
    ) -> list[dict[str, Any]]:
        """Return candidate-pool rows joined to selector event indexes."""

        if not event_indexes:
            return []
        with closing(self.read_connect()) as connection:
            placeholders = ",".join("?" for _ in event_indexes)
            cursor = connection.execute(
                event_db_sql.SELECTOR_POOL_ROWS_FOR_EVENTS_SQL.format(
                    placeholders=placeholders
                ),
                tuple(int(index) for index in event_indexes),
            )
            return [_mapping_from_cursor(cursor=cursor, row=row) for row in cursor]

    def read_verbalized_sampling_decision_rows_for_events(
        self, *, event_indexes: Sequence[int]
    ) -> list[dict[str, Any]]:
        """Return verbalized sampling decision rows for event indexes."""

        if not event_indexes:
            return []
        with closing(self.read_connect()) as connection:
            if not _has_table(
                connection=connection, table_name="verbalized_sampling_decision"
            ):
                return []
            placeholders = ",".join("?" for _ in event_indexes)
            cursor = connection.execute(
                event_db_sql.VERBALIZED_SAMPLING_DECISION_ROWS_FOR_EVENTS_SQL.format(
                    placeholders=placeholders
                ),
                tuple(int(index) for index in event_indexes),
            )
            return [_mapping_from_cursor(cursor=cursor, row=row) for row in cursor]

    def read_verbalized_sampling_candidate_rows_for_events(
        self, *, event_indexes: Sequence[int]
    ) -> list[dict[str, Any]]:
        """Return parsed verbalized sampling candidates for event indexes."""

        if not event_indexes:
            return []
        with closing(self.read_connect()) as connection:
            if not _has_table(
                connection=connection, table_name="verbalized_sampling_candidate"
            ):
                return []
            placeholders = ",".join("?" for _ in event_indexes)
            cursor = connection.execute(
                event_db_sql.VERBALIZED_SAMPLING_CANDIDATE_ROWS_FOR_EVENTS_SQL.format(
                    placeholders=placeholders
                ),
                tuple(int(index) for index in event_indexes),
            )
            return [_mapping_from_cursor(cursor=cursor, row=row) for row in cursor]

    def read_vllm_choice_rows_for_events(
        self, *, event_indexes: Sequence[int]
    ) -> list[dict[str, Any]]:
        """Return compact choice rows for vLLM response event indexes."""

        if not event_indexes:
            return []
        with closing(self.read_connect()) as connection:
            placeholders = ",".join("?" for _ in event_indexes)
            cursor = connection.execute(
                event_db_sql.VLLM_CHOICES_FOR_EVENTS_SQL.format(
                    placeholders=placeholders
                ),
                tuple(int(index) for index in event_indexes),
            )
            return [_mapping_from_cursor(cursor=cursor, row=row) for row in cursor]

    def read_vllm_choice_token_rows_for_events(
        self, *, event_indexes: Sequence[int]
    ) -> list[dict[str, Any]]:
        """Return per-token vLLM choice rows for response event indexes."""

        if not event_indexes:
            return []
        with closing(self.read_connect()) as connection:
            _ensure_normalized_current(connection=connection)
            placeholders = ",".join("?" for _ in event_indexes)
            cursor = connection.execute(
                event_db_sql.VLLM_CHOICE_TOKENS_FOR_EVENTS_SQL.format(
                    placeholders=placeholders
                ),
                tuple(int(index) for index in event_indexes),
            )
            return [_mapping_from_cursor(cursor=cursor, row=row) for row in cursor]

    def read_generated_chunk_rows_for_events(
        self, *, event_indexes: Sequence[int]
    ) -> list[dict[str, Any]]:
        """Return generated text chunks for decode/steer event indexes."""

        if not event_indexes:
            return []
        with closing(self.read_connect()) as connection:
            _ensure_normalized_current(connection=connection)
            placeholders = ",".join("?" for _ in event_indexes)
            cursor = connection.execute(
                event_db_sql.GENERATED_CHUNKS_FOR_EVENTS_SQL.format(
                    placeholders=placeholders
                ),
                tuple(int(index) for index in event_indexes),
            )
            return [_mapping_from_cursor(cursor=cursor, row=row) for row in cursor]

    def read_generated_chunk_rows_for_node(
        self, *, node_id: str, **key: Any
    ) -> list[dict[str, Any]]:
        """Return normalized generated text chunks for one decode node."""

        with closing(self.read_connect()) as connection:
            _ensure_normalized_current(connection=connection)
            cursor = connection.execute(
                event_db_sql.GENERATED_CHUNKS_FOR_NODE_SQL,
                (*_attempt_values(key=key), node_id),
            )
            return [_mapping_from_cursor(cursor=cursor, row=row) for row in cursor]

    def event_count(self) -> int:
        """Return the number of persisted event rows."""

        with closing(self.read_connect()) as connection:
            row = connection.execute("SELECT COUNT(*) FROM event_log").fetchone()
        return int(row[0]) if row is not None else 0

    def last_event_index(self) -> int:
        """Return the max event index currently persisted, or -1."""

        with closing(self.read_connect()) as connection:
            row = connection.execute(
                "SELECT MAX(event_index) FROM event_log"
            ).fetchone()
        value = row[0] if row is not None else None
        return int(value) if isinstance(value, int) else -1

    def write_doc_progress(self, *, payload: dict[str, Any]) -> None:
        """Upsert the latest compact progress snapshot for one attempt."""

        with closing(self.connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            self.upsert_doc_progress(connection=connection, payload=payload)
            connection.commit()

    def upsert_doc_progress(
        self, *, connection: sqlite3.Connection, payload: dict[str, Any]
    ) -> None:
        """Upsert one doc-progress snapshot on a caller-owned connection."""

        run_id = str(payload["run_id"])
        doc_id = int(payload["doc_id"])
        doc_attempt = int(payload["doc_attempt"])
        timestamp = str(payload.get("last_update_timestamp", ""))
        payload_json = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        connection.execute(
            """
            INSERT INTO doc_progress (
                run_id, doc_id, doc_attempt, payload_json, last_update_timestamp
            ) VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(run_id, doc_id, doc_attempt) DO UPDATE SET
                payload_json = excluded.payload_json,
                last_update_timestamp = excluded.last_update_timestamp
            """,
            (run_id, doc_id, doc_attempt, payload_json, timestamp),
        )
        connection.execute(
            """
            INSERT INTO doc_progress_typed (
                run_id, doc_id, doc_attempt, task_name, model_id,
                selector_mode, rollout_mode, status, leaf_count, passrate,
                avg_token_length, correct_count, incorrect_count,
                natural_count, max_count, repeating_count, other_count,
                unique_answer_count, last_update_timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id, doc_id, doc_attempt) DO UPDATE SET
                task_name = excluded.task_name,
                model_id = excluded.model_id,
                selector_mode = excluded.selector_mode,
                rollout_mode = excluded.rollout_mode,
                status = excluded.status,
                leaf_count = excluded.leaf_count,
                passrate = excluded.passrate,
                avg_token_length = excluded.avg_token_length,
                correct_count = excluded.correct_count,
                incorrect_count = excluded.incorrect_count,
                natural_count = excluded.natural_count,
                max_count = excluded.max_count,
                repeating_count = excluded.repeating_count,
                other_count = excluded.other_count,
                unique_answer_count = excluded.unique_answer_count,
                last_update_timestamp = excluded.last_update_timestamp
            """,
            _doc_progress_values(payload=payload),
        )

    def read_doc_progress_rows(self) -> list[dict[str, Any]]:
        """Return latest compact progress snapshots from SQLite."""

        with closing(self.read_connect()) as connection:
            cursor = connection.execute("""
                SELECT
                    run_id, doc_id, doc_attempt, task_name, model_id,
                    selector_mode, rollout_mode, status, leaf_count, passrate,
                    avg_token_length, correct_count, incorrect_count,
                    natural_count, max_count, repeating_count, other_count,
                    unique_answer_count, last_update_timestamp
                FROM doc_progress_typed
                ORDER BY doc_id, doc_attempt, last_update_timestamp
                """)
            rows = cursor.fetchall()
        return [_doc_progress_row(row=row) for row in rows]

    def read_node_attempt_progress_rows(self) -> list[dict[str, Any]]:
        """Return progress snapshots from typed leaf rows and finish events."""

        with closing(self.read_connect()) as connection:
            rows = connection.execute("""
                WITH attempts AS (
                    SELECT DISTINCT doc_id, doc_attempt, task_name, model_id, selector_mode
                    FROM node_event
                    UNION
                    SELECT DISTINCT doc_id, doc_attempt, task_name, model_id, selector_mode
                    FROM node_created
                    UNION
                    SELECT DISTINCT doc_id, doc_attempt, task_name, model_id, selector_mode
                    FROM leaf_score
                ),
                leaf AS (
                    SELECT
                        doc_id, doc_attempt, task_name, model_id, selector_mode,
                        COUNT(*) AS leaf_count,
                        SUM(CASE WHEN verification = 1 THEN 1 ELSE 0 END) AS correct_count,
                        SUM(CASE WHEN verification = 0 THEN 1 ELSE 0 END) AS incorrect_count,
                        SUM(CASE WHEN stop_reason = 'think_end' THEN 1 ELSE 0 END) AS natural_count,
                        SUM(CASE WHEN stop_reason = 'length' THEN 1 ELSE 0 END) AS max_count,
                        SUM(CASE WHEN stop_reason = 'repeated_exec_block_loop' THEN 1 ELSE 0 END) AS repeating_count,
                        SUM(CASE WHEN stop_reason NOT IN (
                            'think_end', 'length', 'repeated_exec_block_loop'
                        ) THEN 1 ELSE 0 END) AS other_count,
                        SUM(COALESCE(length_tokens_total, 0)) AS length_token_total,
                        SUM(CASE WHEN length_tokens_total IS NULL THEN 0 ELSE 1 END) AS length_count,
                        COUNT(DISTINCT text_preview) AS unique_answer_count,
                        MAX(timestamp_utc) AS last_leaf_timestamp
                    FROM leaf_score
                    GROUP BY doc_id, doc_attempt, task_name, model_id, selector_mode
                ),
                finish AS (
                    SELECT
                        doc_id, doc_attempt, task_name, model_id, selector_mode,
                        COUNT(*) AS finished_count,
                        MAX(timestamp_utc) AS last_finish_timestamp
                    FROM event_log INDEXED BY idx_event_log_type
                    WHERE event_type IN ('doc_finished', 'rollout_finished')
                      AND doc_id IS NOT NULL
                    GROUP BY doc_id, doc_attempt, task_name, model_id, selector_mode
                )
                SELECT
                    attempts.doc_id, attempts.doc_attempt, attempts.task_name,
                    attempts.model_id, attempts.selector_mode,
                    COALESCE(leaf.leaf_count, 0), COALESCE(leaf.correct_count, 0),
                    COALESCE(leaf.incorrect_count, 0), COALESCE(leaf.natural_count, 0),
                    COALESCE(leaf.max_count, 0), COALESCE(leaf.repeating_count, 0),
                    COALESCE(leaf.other_count, 0), COALESCE(leaf.length_token_total, 0),
                    COALESCE(leaf.length_count, 0),
                    COALESCE(leaf.unique_answer_count, 0),
                    COALESCE(finish.finished_count, 0),
                    COALESCE(finish.last_finish_timestamp, leaf.last_leaf_timestamp, '')
                FROM attempts
                LEFT JOIN leaf USING (
                    doc_id, doc_attempt, task_name, model_id, selector_mode
                )
                LEFT JOIN finish USING (
                    doc_id, doc_attempt, task_name, model_id, selector_mode
                )
                ORDER BY attempts.doc_id, attempts.doc_attempt,
                    attempts.task_name, attempts.model_id, attempts.selector_mode
                """).fetchall()
        return [_node_attempt_progress_row(row=row) for row in rows]

    def read_attempt_count_row(self, **key: Any) -> dict[str, Any]:
        """Return count-only data for one attempt page shell."""

        with closing(self.read_connect()) as connection:
            values = _attempt_values(key=key)
            node = connection.execute(
                """
                SELECT COUNT(*), MAX(event_index), MAX(timestamp_utc)
                FROM node_created
                WHERE doc_id = ? AND doc_attempt = ? AND task_name = ?
                  AND model_id = ? AND selector_mode = ?
                """,
                values,
            ).fetchone()
            edge = connection.execute(
                """
                SELECT COUNT(*), MAX(event_index), MAX(timestamp_utc)
                FROM edge_selected
                WHERE doc_id = ? AND doc_attempt = ? AND task_name = ?
                  AND model_id = ? AND selector_mode = ?
                """,
                values,
            ).fetchone()
            leaf = connection.execute(
                """
                SELECT COUNT(*), MAX(event_index), MAX(timestamp_utc)
                FROM leaf_score
                WHERE doc_id = ? AND doc_attempt = ? AND task_name = ?
                  AND model_id = ? AND selector_mode = ?
                """,
                values,
            ).fetchone()
            lifecycle = connection.execute(
                """
                SELECT
                    COUNT(*), MAX(event_index), MAX(timestamp_utc),
                    SUM(CASE WHEN event_type IN (
                        'doc_started', 'doc_resumed', 'rollout_started'
                    ) THEN 1 ELSE 0 END),
                    SUM(CASE WHEN event_type IN (
                        'doc_finished', 'rollout_finished'
                    ) THEN 1 ELSE 0 END)
                FROM event_log INDEXED BY idx_event_log_type
                WHERE doc_id = ? AND doc_attempt = ? AND task_name = ?
                  AND model_id = ? AND selector_mode = ?
                  AND event_type IN (
                      'doc_started', 'doc_resumed', 'rollout_started',
                      'doc_finished', 'rollout_finished'
                  )
                """,
                values,
            ).fetchone()
        return _attempt_count_row(
            key=key,
            node=node,
            edge=edge,
            leaf=leaf,
            lifecycle=lifecycle,
        )

    def _initialize(self) -> None:
        with closing(self.connect()) as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=NORMAL")
            connection.executescript(event_db_sql.SCHEMA_SQL)
            _ensure_normalized_current(connection=connection)


def event_source_signature(*, run_dir: Path) -> tuple[int, int]:
    """Return a combined signature for DB/WAL event files."""

    db_path = run_dir / EVENT_DB_FILENAME
    paths = (db_path, db_path.with_name(db_path.name + "-wal"))
    size_total = 0
    mtime_max = 0
    for path in paths:
        if not path.exists():
            continue
        stat = path.stat()
        size_total += int(stat.st_size)
        mtime_max = max(mtime_max, int(stat.st_mtime_ns))
    return (size_total, mtime_max)


def resolve_event_db_path(*, run_dir: Path) -> Path:
    """Return the canonical event DB path for one run directory."""

    return run_dir / EVENT_DB_FILENAME


def _event_insert_values(*, row: dict[str, Any]) -> tuple[Any, ...]:
    compact_row = _compact_event_log_row(row=row)
    payload = compact_row.get("payload", {})
    assert isinstance(payload, dict), "event payload must be a mapping"
    return (
        int(compact_row["event_index"]),
        int(compact_row["event_version"]),
        str(compact_row["timestamp_utc"]),
        str(compact_row["run_id"]),
        compact_row.get("doc_id"),
        compact_row.get("doc_attempt"),
        str(compact_row["task_name"]),
        str(compact_row["model_id"]),
        str(compact_row["selector_mode"]),
        str(compact_row["event_type"]),
        json.dumps(payload, sort_keys=True, ensure_ascii=False),
        json.dumps(compact_row, sort_keys=True, ensure_ascii=False),
    )


def _compact_event_log_row(*, row: dict[str, Any]) -> dict[str, Any]:
    """Return a raw event-log row without duplicate vLLM token arrays."""

    if row.get("event_type") != "vllm_response":
        return row
    payload = row.get("payload", {})
    if not isinstance(payload, dict) or payload.get("status") != "ok":
        return row
    choices = payload.get("choices")
    if not isinstance(choices, list):
        return row
    compact_payload = dict(payload)
    compact_payload["choices"] = [
        _compact_event_log_choice(choice=choice) for choice in choices
    ]
    compact_payload["choices_token_rows_compacted"] = True
    compact_row = dict(row)
    compact_row["payload"] = compact_payload
    return compact_row


def _compact_event_log_choice(*, choice: Any) -> Any:
    """Return one vLLM choice without token-level duplicate fields."""

    if not isinstance(choice, dict):
        return choice
    compact_choice = dict(choice)
    compact_choice.pop("tokens", None)
    compact_choice.pop("token_ids", None)
    return compact_choice


def _loads_mapping(*, text: str) -> dict[str, Any]:
    value = json.loads(text)
    assert isinstance(value, dict), "stored JSON must decode to a mapping"
    return value


def _hydrate_compact_vllm_response_rows(
    *, connection: sqlite3.Connection, rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Reconstruct compacted vLLM response token arrays from typed tables."""

    event_indices = [
        int(row["event_index"]) for row in rows if _needs_vllm_token_hydration(row=row)
    ]
    if not event_indices:
        return rows
    choices_by_event = _read_vllm_choices(
        connection=connection, event_indices=event_indices
    )
    tokens_by_choice = _read_vllm_choice_tokens(
        connection=connection, event_indices=event_indices
    )
    return [
        _hydrated_event_row(
            row=row,
            choices_by_event=choices_by_event,
            tokens_by_choice=tokens_by_choice,
        )
        for row in rows
    ]


def _needs_vllm_token_hydration(*, row: dict[str, Any]) -> bool:
    """Return whether one raw event row has compacted vLLM choice tokens."""

    if row.get("event_type") != "vllm_response":
        return False
    payload = row.get("payload", {})
    if not isinstance(payload, dict):
        return False
    if payload.get("choices_token_rows_compacted") is True:
        return True
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return False
    return all(
        isinstance(choice, dict) and "tokens" not in choice for choice in choices
    )


def _read_vllm_choices(
    *, connection: sqlite3.Connection, event_indices: Sequence[int]
) -> dict[int, list[dict[str, Any]]]:
    """Read typed vLLM choices for compacted response events."""

    choices_by_event: dict[int, list[dict[str, Any]]] = {}
    for chunk in _chunked_ints(values=event_indices, size=900):
        placeholders = ",".join("?" for _ in chunk)
        cursor = connection.execute(
            f"""
            SELECT response_event_index, choice_index, text, text_preview,
                   finish_reason, stop_reason, output_token_count
            FROM vllm_choice
            WHERE response_event_index IN ({placeholders})
            ORDER BY response_event_index, choice_index
            """,
            tuple(chunk),
        )
        for row in cursor.fetchall():
            event_index = int(row[0])
            choices_by_event.setdefault(event_index, []).append(
                _hydrated_choice_base(row=row)
            )
    return choices_by_event


def _read_vllm_choice_tokens(
    *, connection: sqlite3.Connection, event_indices: Sequence[int]
) -> dict[tuple[int, int], list[dict[str, Any]]]:
    """Read typed vLLM choice tokens for compacted response events."""

    tokens_by_choice: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for chunk in _chunked_ints(values=event_indices, size=900):
        placeholders = ",".join("?" for _ in chunk)
        cursor = connection.execute(
            f"""
            SELECT response_event_index, choice_index, token_index, token_id,
                   token_text, selected_logprob
            FROM vllm_choice_token
            WHERE response_event_index IN ({placeholders})
            ORDER BY response_event_index, choice_index, token_index
            """,
            tuple(chunk),
        )
        for row in cursor.fetchall():
            key = (int(row[0]), int(row[1]))
            tokens_by_choice.setdefault(key, []).append(_hydrated_token(row=row))
    return tokens_by_choice


def _hydrated_event_row(
    *,
    row: dict[str, Any],
    choices_by_event: dict[int, list[dict[str, Any]]],
    tokens_by_choice: dict[tuple[int, int], list[dict[str, Any]]],
) -> dict[str, Any]:
    """Return one event row with compacted vLLM tokens restored."""

    event_index = int(row["event_index"])
    choices = choices_by_event.get(event_index)
    if choices is None:
        return row
    hydrated_choices = [
        _hydrated_choice(
            event_index=event_index,
            choice=choice,
            tokens_by_choice=tokens_by_choice,
        )
        for choice in choices
    ]
    payload = dict(row["payload"])
    payload["choices"] = hydrated_choices
    payload.pop("choices_token_rows_compacted", None)
    hydrated_row = dict(row)
    hydrated_row["payload"] = payload
    return hydrated_row


def _hydrated_choice(
    *,
    event_index: int,
    choice: dict[str, Any],
    tokens_by_choice: dict[tuple[int, int], list[dict[str, Any]]],
) -> dict[str, Any]:
    """Return one choice with token rows and token id vector restored."""

    choice_index = int(choice["index"])
    tokens = tokens_by_choice.get((event_index, choice_index), [])
    hydrated = dict(choice)
    hydrated["token_ids"] = [
        token["token_id"] for token in tokens if token.get("token_id") is not None
    ]
    hydrated["tokens"] = tokens
    return hydrated


def _hydrated_choice_base(*, row: tuple[Any, ...]) -> dict[str, Any]:
    """Return one choice mapping from a typed SQLite row."""

    choice = {
        "index": int(row[1]),
        "text_preview": str(row[3]),
        "finish_reason": str(row[4]),
        "stop_reason": str(row[5]),
    }
    output_token_count = row[6]
    if output_token_count is not None:
        choice["output_token_count"] = int(output_token_count)
    text = str(row[2])
    if text != choice["text_preview"]:
        choice["text"] = text
    return choice


def _hydrated_token(*, row: tuple[Any, ...]) -> dict[str, Any]:
    """Return one compact token mapping from a typed SQLite row."""

    token = {
        "token_index": int(row[2]),
        "token_text": str(row[4]),
    }
    if row[3] is not None:
        token["token_id"] = int(row[3])
    if row[5] is not None:
        token["selected_logprob"] = float(row[5])
    return token


def _chunked_ints(*, values: Sequence[int], size: int) -> list[tuple[int, ...]]:
    """Split integer values into SQLite parameter chunks."""

    assert size > 0
    return [
        tuple(values[index : index + size]) for index in range(0, len(values), size)
    ]


def _ensure_normalized_current(*, connection: sqlite3.Connection) -> None:
    _ = connection
    return


def _backfill_normalized_range(
    *, connection: sqlite3.Connection, start_index: int, end_index: int
) -> None:
    params = {"start": start_index, "end": end_index}
    for sql in event_db_sql.BACKFILL_SQL_STATEMENTS:
        connection.execute(sql, params)


def _metadata_int(*, connection: sqlite3.Connection, key: str) -> int:
    row = connection.execute(
        "SELECT value FROM metadata WHERE key = ?",
        (key,),
    ).fetchone()
    return int(row[0]) if row is not None and str(row[0]).strip() else -1


def _set_metadata_event_index(
    *, connection: sqlite3.Connection, key: str, event_index: int
) -> None:
    prior = _metadata_int(connection=connection, key=key)
    value = max(prior, event_index)
    connection.execute(
        """
        INSERT INTO metadata(key, value) VALUES (?, ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """,
        (key, str(value)),
    )


def _has_column(
    *, connection: sqlite3.Connection, table_name: str, column: str
) -> bool:
    rows = connection.execute(f"PRAGMA table_info({table_name})").fetchall()
    return any(str(row[1]) == column for row in rows)


def _has_table(*, connection: sqlite3.Connection, table_name: str) -> bool:
    row = connection.execute(
        """
        SELECT 1
        FROM sqlite_master
        WHERE type = 'table' AND name = ?
        LIMIT 1
        """,
        (table_name,),
    ).fetchone()
    return row is not None


def _prompt_context_sql_parts(*, connection: sqlite3.Connection) -> tuple[str, str]:
    if _has_table(connection=connection, table_name="prompt_context"):
        return (
            event_db_sql.PROMPT_CONTEXT_COLUMNS_SQL,
            event_db_sql.PROMPT_CONTEXT_JOIN_SQL,
        )
    return (event_db_sql.NULL_PROMPT_CONTEXT_COLUMNS_SQL, "")


def _nullable_int(value: Any) -> int | None:
    return int(value) if isinstance(value, int) else None


def _attempt_values(*, key: dict[str, Any]) -> tuple[Any, ...]:
    return (
        int(key["doc_id"]),
        int(key["doc_attempt"]),
        str(key["task_name"]),
        str(key["model_id"]),
        str(key["selector_mode"]),
    )


def _node_advantage_values(*, row: dict[str, Any]) -> tuple[Any, ...]:
    updated_at = str(row.get("updated_at") or datetime.now(tz=timezone.utc).isoformat())
    return (
        int(row["doc_id"]),
        int(row["doc_attempt"]),
        str(row["task_name"]),
        str(row["model_id"]),
        str(row["selector_mode"]),
        str(row["prompt_uid"]),
        str(row["branch_tree_id"]),
        str(row["parent_node_id"]),
        str(row["child_node_id"]),
        int(row["branch_depth"]),
        int(row["token_start"]),
        int(row["token_end"]),
        float(row["mean_combined_advantage"]),
        int(row["token_count"]),
        int(row["leaf_count"]),
        int(row["updated_at_event_index"]),
        updated_at,
    )


def _placeholders(*, count: int) -> str:
    assert count > 0, "placeholder count must be positive"
    return ",".join("?" for _ in range(count))


def _mapping_from_cursor(
    *, cursor: sqlite3.Cursor, row: sqlite3.Row | tuple[Any, ...]
) -> dict[str, Any]:
    columns = [description[0] for description in cursor.description]
    return {column: row[index] for index, column in enumerate(columns)}


def _visible_node_event_rows(*, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    visible_rows: list[dict[str, Any]] = []
    local_steps: dict[str, int] = {}
    for row in rows:
        if str(row.get("event_type") or "") not in VISIBLE_RUNTIME_EVENT_TYPES:
            continue
        node_id = str(row.get("node_id") or "")
        local_steps[node_id] = local_steps.get(node_id, 0) + 1
        row["local_step"] = local_steps[node_id]
        visible_rows.append(row)
    return visible_rows


def _node_attempt_progress_row(*, row: tuple[Any, ...]) -> dict[str, Any]:
    leaf_count = int(row[5] or 0)
    correct_count = int(row[6] or 0)
    length_count = int(row[13] or 0)
    length_total = int(row[12] or 0)
    finished_count = int(row[15] or 0)
    return {
        "doc_id": int(row[0]),
        "doc_attempt": int(row[1]),
        "task_name": str(row[2]),
        "model_id": str(row[3]),
        "selector_mode": str(row[4]),
        "rollout_mode": "",
        "status": "completed" if finished_count else "incomplete",
        "leaf_count": leaf_count,
        "passrate": correct_count / leaf_count if leaf_count else 0.0,
        "avg_token_length": length_total / length_count if length_count else 0.0,
        "correct_count": correct_count,
        "incorrect_count": int(row[7] or 0),
        "natural_count": int(row[8] or 0),
        "max_count": int(row[9] or 0),
        "repeating_count": int(row[10] or 0),
        "other_count": int(row[11] or 0),
        "unique_answer_count": int(row[14] or 0),
        "last_update_timestamp": str(row[16] or ""),
    }


def _attempt_count_row(
    *,
    key: dict[str, Any],
    node: tuple[Any, ...] | None,
    edge: tuple[Any, ...] | None,
    leaf: tuple[Any, ...] | None,
    lifecycle: tuple[Any, ...] | None,
) -> dict[str, Any]:
    node_count, node_index, node_timestamp = _count_max_row(row=node)
    edge_count, edge_index, edge_timestamp = _count_max_row(row=edge)
    leaf_count, leaf_index, leaf_timestamp = _count_max_row(row=leaf)
    lifecycle_count, lifecycle_index, lifecycle_timestamp = _count_max_row(
        row=lifecycle
    )
    started_count = int(lifecycle[3] or 0) if lifecycle is not None else 0
    finished_count = int(lifecycle[4] or 0) if lifecycle is not None else 0
    last_event_index = max(node_index, edge_index, leaf_index, lifecycle_index)
    return {
        **key,
        "event_count": node_count + edge_count + leaf_count + lifecycle_count,
        "node_count": node_count,
        "edge_count": edge_count,
        "leaf_count": leaf_count,
        "last_event_index": last_event_index,
        "last_timestamp_utc": max(
            node_timestamp, edge_timestamp, leaf_timestamp, lifecycle_timestamp
        ),
        "started_count": started_count or (1 if last_event_index >= 0 else 0),
        "finished_count": finished_count,
    }


def _count_max_row(*, row: tuple[Any, ...] | None) -> tuple[int, int, str]:
    if row is None:
        return (0, -1, "")
    return (int(row[0] or 0), int(row[1] or -1), str(row[2] or ""))


def _leaf_summary_row(*, row: tuple[Any, ...]) -> dict[str, Any]:
    length_count = int(row[15] or 0)
    length_total = int(row[14] or 0)
    return {
        "doc_id": int(row[0]),
        "doc_attempt": int(row[1]),
        "task_name": str(row[2]),
        "model_id": str(row[3]),
        "selector_mode": str(row[4]),
        "leaf_count": int(row[5] or 0),
        "correct_count": int(row[6] or 0),
        "incorrect_count": int(row[7] or 0),
        "answer_correct_count": int(row[8] or 0),
        "answer_count": int(row[9] or 0),
        "natural_count": int(row[10] or 0),
        "max_count": int(row[11] or 0),
        "repeating_count": int(row[12] or 0),
        "other_count": int(row[13] or 0),
        "avg_token_length": (length_total / length_count if length_count else 0.0),
        "unique_answer_count": int(row[16] or 0),
    }


def _attempt_summary_row(*, row: tuple[Any, ...]) -> dict[str, Any]:
    return {
        "doc_id": int(row[0]),
        "doc_attempt": int(row[1]),
        "task_name": str(row[2]),
        "model_id": str(row[3]),
        "selector_mode": str(row[4]),
        "event_count": int(row[5]),
        "last_event_index": int(row[6]),
        "last_timestamp_utc": str(row[7] or ""),
        "started_count": int(row[8] or 0),
        "finished_count": int(row[9] or 0),
    }


def _doc_progress_values(*, payload: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(payload["run_id"]),
        int(payload["doc_id"]),
        int(payload["doc_attempt"]),
        str(payload.get("task_name", "")),
        str(payload.get("model_id", "")),
        str(payload.get("selector_mode", "")),
        str(payload.get("rollout_mode", "")),
        str(payload.get("status", "incomplete")),
        int(payload.get("leaf_count", 0)),
        float(payload.get("passrate", 0.0)),
        float(payload.get("avg_token_length", 0.0)),
        int(payload.get("correct_count", 0)),
        int(payload.get("incorrect_count", 0)),
        int(payload.get("natural_count", 0)),
        int(payload.get("max_count", 0)),
        int(payload.get("repeating_count", 0)),
        int(payload.get("other_count", 0)),
        int(payload.get("unique_answer_count", 0)),
        str(payload.get("last_update_timestamp", "")),
    )


def _doc_progress_row(*, row: tuple[Any, ...]) -> dict[str, Any]:
    keys = (
        "run_id",
        "doc_id",
        "doc_attempt",
        "task_name",
        "model_id",
        "selector_mode",
        "rollout_mode",
        "status",
        "leaf_count",
        "passrate",
        "avg_token_length",
        "correct_count",
        "incorrect_count",
        "natural_count",
        "max_count",
        "repeating_count",
        "other_count",
        "unique_answer_count",
        "last_update_timestamp",
    )
    assert len(keys) == len(row)
    return dict(zip(keys, row))
