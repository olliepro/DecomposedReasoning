"""SQL constants for the branching event SQLite store."""

from __future__ import annotations

INSERT_NODE_SQL = """
INSERT OR IGNORE INTO node_created (
    event_index, doc_id, doc_attempt, task_name, model_id, selector_mode,
    timestamp_utc, node_id, parent_node_id, branch_points_used
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

INSERT_EDGE_SQL = """
INSERT OR IGNORE INTO edge_selected (
    event_index, doc_id, doc_attempt, task_name, model_id, selector_mode,
    timestamp_utc, parent_node_id, child_node_id, candidate_id, edge_selector_mode,
    candidate_text, candidate_token_count
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

INSERT_LEAF_SQL = """
INSERT OR IGNORE INTO leaf_score (
    event_index, doc_id, doc_attempt, task_name, model_id, selector_mode,
    timestamp_utc, leaf_id, node_id, verification, length_tokens_total,
    length_tokens_exec, stop_reason, text, text_preview
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

INSERT_LEAF_METRIC_SQL = """
INSERT OR REPLACE INTO leaf_metric (
    leaf_event_index, metric_name, metric_value, metric_text
) VALUES (?, ?, ?, ?)
"""

INSERT_NODE_EVENT_SQL = """
INSERT OR IGNORE INTO node_event (
    event_index, doc_id, doc_attempt, task_name, model_id, selector_mode,
    timestamp_utc, event_type, node_id, summary, step_delta, token_delta,
    branch_point_id, leaf_id, verification, stop_reason, length_tokens_total,
    request_id, request_stream_id, request_kind, status, latency_seconds,
    error_message, choice_count, output_token_count, text_preview, text
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

INSERT_PROMPT_CONTEXT_SQL = """
INSERT OR REPLACE INTO prompt_context (
    event_index, doc_id, doc_attempt, task_name, model_id, selector_mode,
    timestamp_utc, node_id, prompt_text, prompt_char_count, golden_answer,
    golden_answer_source
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

PROMPT_CONTEXT_COLUMNS_SQL = """
    prompt_context.prompt_text, prompt_context.prompt_char_count,
    prompt_context.golden_answer, prompt_context.golden_answer_source
"""

NULL_PROMPT_CONTEXT_COLUMNS_SQL = """
    NULL AS prompt_text, NULL AS prompt_char_count,
    NULL AS golden_answer, NULL AS golden_answer_source
"""

PROMPT_CONTEXT_JOIN_SQL = """
LEFT JOIN prompt_context
  ON prompt_context.event_index = node_event.event_index
"""

INSERT_CANDIDATE_POOL_SQL = """
INSERT OR IGNORE INTO candidate_pool (
    event_index, doc_id, doc_attempt, task_name, model_id, selector_mode,
    timestamp_utc, branch_point_id, candidate_pool_id, node_id, trigger_type,
    num_candidates
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

INSERT_CANDIDATE_SQL = """
INSERT OR REPLACE INTO candidate_pool_candidate (
    pool_event_index, candidate_id, text, text_preview, output_token_count,
    finish_reason, stop_reason
) VALUES (?, ?, ?, ?, ?, ?, ?)
"""

INSERT_CANDIDATE_TOKEN_SQL = """
INSERT OR REPLACE INTO candidate_pool_candidate_token (
    pool_event_index, candidate_id, token_index, token_id, token_text,
    selected_logprob, selected_probability
) VALUES (?, ?, ?, ?, ?, ?, ?)
"""

INSERT_SELECTOR_SQL = """
INSERT OR IGNORE INTO selector_event (
    event_index, doc_id, doc_attempt, task_name, model_id, selector_mode,
    timestamp_utc, selector_event_type, branch_point_id, node_id,
    active_selector_mode
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

INSERT_SELECTOR_FLAG_SQL = """
INSERT OR REPLACE INTO selector_candidate_flag (
    selector_event_index, mode_name, candidate_id, selected, shortlisted
) VALUES (?, ?, ?, ?, ?)
"""

INSERT_SELECTOR_CLUSTER_SQL = """
INSERT OR REPLACE INTO selector_candidate_cluster (
    selector_event_index, mode_name, candidate_id, cluster_name
) VALUES (?, ?, ?, ?)
"""

INSERT_VERBALIZED_SAMPLING_DECISION_SQL = """
INSERT OR REPLACE INTO verbalized_sampling_decision (
    event_index, doc_id, doc_attempt, task_name, model_id, selector_mode,
    timestamp_utc, branch_point_id, candidate_pool_id, node_id,
    candidate_count, branch_fanout, sampled_option_numbers,
    parse_status, enumeration_exec_text
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

INSERT_VERBALIZED_SAMPLING_CANDIDATE_SQL = """
INSERT OR REPLACE INTO verbalized_sampling_candidate (
    decision_event_index, option_number, candidate_rank, candidate_text, selected
) VALUES (?, ?, ?, ?, ?)
"""

INSERT_VLLM_REQUEST_SQL = """
INSERT OR IGNORE INTO vllm_request (
    event_index, doc_id, doc_attempt, task_name, model_id, selector_mode,
    timestamp_utc, request_id, request_stream_id, request_kind, prev_request_id,
    current_input_token_count, base_prefix_token_count, delta_token_count,
    assistant_prefix_char_count, assistant_prefix_tail
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

INSERT_VLLM_RESPONSE_SQL = """
INSERT OR IGNORE INTO vllm_response (
    event_index, doc_id, doc_attempt, task_name, model_id, selector_mode,
    timestamp_utc, request_id, request_stream_id, request_kind, status,
    latency_seconds, error_message, choice_count
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

INSERT_VLLM_CHOICE_SQL = """
INSERT OR IGNORE INTO vllm_choice (
    response_event_index, choice_index, text, text_preview, finish_reason,
    stop_reason, output_token_count
) VALUES (?, ?, ?, ?, ?, ?, ?)
"""

INSERT_VLLM_CHOICE_TOKEN_SQL = """
INSERT OR IGNORE INTO vllm_choice_token (
    response_event_index, choice_index, token_index, token_id, token_text,
    selected_logprob, selected_probability
) VALUES (?, ?, ?, ?, ?, ?, ?)
"""

INSERT_GENERATED_CHUNK_SQL = """
INSERT OR REPLACE INTO generated_chunk (
    event_index, doc_id, doc_attempt, task_name, model_id, selector_mode,
    timestamp_utc, event_type, node_id, chunk_text, token_count,
    generated_tokens_before_chunk, generated_tokens_after_chunk,
    chunk_was_normalized, chunk_token_ids_source, source
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

UPSERT_NODE_ADVANTAGE_SQL = """
INSERT INTO node_advantage (
    doc_id, doc_attempt, task_name, model_id, selector_mode, prompt_uid,
    branch_tree_id, parent_node_id, child_node_id, branch_depth, token_start,
    token_end, mean_combined_advantage, token_count, leaf_count,
    updated_at_event_index, updated_at
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(
    doc_id, doc_attempt, task_name, model_id, selector_mode, parent_node_id,
    child_node_id
) DO UPDATE SET
    prompt_uid = excluded.prompt_uid,
    branch_tree_id = excluded.branch_tree_id,
    branch_depth = excluded.branch_depth,
    token_start = excluded.token_start,
    token_end = excluded.token_end,
    mean_combined_advantage = excluded.mean_combined_advantage,
    token_count = excluded.token_count,
    leaf_count = excluded.leaf_count,
    updated_at_event_index = excluded.updated_at_event_index,
    updated_at = excluded.updated_at
"""

NODE_ROWS_SQL = """
SELECT
    nodes.node_id,
    nodes.parent_node_id,
    nodes.branch_points_used,
    COALESCE(edges.candidate_text, 'Root') AS candidate_preview,
    0 AS event_count,
    COALESCE(leaves.leaf_count, 0) AS leaf_count,
    COALESCE(edges.candidate_token_count, 0) AS token_total,
    nodes.event_index AS first_event_index,
    nodes.timestamp_utc AS first_timestamp_utc
FROM node_created AS nodes
LEFT JOIN edge_selected AS edges
  ON edges.child_node_id = nodes.node_id
 AND edges.doc_id = nodes.doc_id
 AND edges.doc_attempt = nodes.doc_attempt
 AND edges.task_name = nodes.task_name
 AND edges.model_id = nodes.model_id
 AND edges.selector_mode = nodes.selector_mode
LEFT JOIN (
    SELECT node_id, COUNT(*) AS leaf_count
    FROM leaf_score
    WHERE doc_id = ? AND doc_attempt = ? AND task_name = ?
      AND model_id = ? AND selector_mode = ?
    GROUP BY node_id
) AS leaves ON leaves.node_id = nodes.node_id
WHERE nodes.doc_id = ? AND nodes.doc_attempt = ? AND nodes.task_name = ?
  AND nodes.model_id = ? AND nodes.selector_mode = ?
ORDER BY nodes.event_index
"""

NODE_ROW_SQL = """
SELECT
    nodes.node_id,
    nodes.parent_node_id,
    nodes.branch_points_used,
    COALESCE(edges.candidate_text, 'Root') AS candidate_preview,
    0 AS event_count,
    COALESCE(leaves.leaf_count, 0) AS leaf_count,
    COALESCE(edges.candidate_token_count, 0) AS token_total,
    nodes.event_index AS first_event_index,
    nodes.timestamp_utc AS first_timestamp_utc
FROM node_created AS nodes
LEFT JOIN edge_selected AS edges
  ON edges.child_node_id = nodes.node_id
 AND edges.doc_id = nodes.doc_id
 AND edges.doc_attempt = nodes.doc_attempt
 AND edges.task_name = nodes.task_name
 AND edges.model_id = nodes.model_id
 AND edges.selector_mode = nodes.selector_mode
LEFT JOIN (
    SELECT node_id, COUNT(*) AS leaf_count
    FROM leaf_score
    WHERE doc_id = ? AND doc_attempt = ? AND task_name = ?
      AND model_id = ? AND selector_mode = ? AND node_id = ?
    GROUP BY node_id
) AS leaves ON leaves.node_id = nodes.node_id
WHERE nodes.doc_id = ? AND nodes.doc_attempt = ? AND nodes.task_name = ?
  AND nodes.model_id = ? AND nodes.selector_mode = ? AND nodes.node_id = ?
LIMIT 1
"""

EDGE_ROWS_SQL = """
SELECT parent_node_id, child_node_id, candidate_id,
       edge_selector_mode AS selector_mode, candidate_text, candidate_token_count
FROM edge_selected
WHERE doc_id = ? AND doc_attempt = ? AND task_name = ?
  AND model_id = ? AND selector_mode = ?
ORDER BY event_index
"""

NODE_ADVANTAGE_ROWS_SQL = """
SELECT
    doc_id, doc_attempt, task_name, model_id, selector_mode, prompt_uid,
    branch_tree_id, parent_node_id, child_node_id, branch_depth, token_start,
    token_end, mean_combined_advantage, token_count, leaf_count,
    updated_at_event_index, updated_at
FROM node_advantage
WHERE doc_id = ? AND doc_attempt = ? AND task_name = ?
  AND model_id = ? AND selector_mode = ?
ORDER BY branch_depth, child_node_id
"""

VERBALIZED_SAMPLING_DECISION_ROWS_FOR_EVENTS_SQL = """
SELECT
    event_index, branch_point_id, candidate_pool_id, node_id, candidate_count,
    branch_fanout, sampled_option_numbers, parse_status, enumeration_exec_text
FROM verbalized_sampling_decision
WHERE event_index IN ({placeholders})
ORDER BY event_index
"""

VERBALIZED_SAMPLING_CANDIDATE_ROWS_FOR_EVENTS_SQL = """
SELECT decision_event_index, option_number, candidate_rank, candidate_text, selected
FROM verbalized_sampling_candidate
WHERE decision_event_index IN ({placeholders})
ORDER BY decision_event_index, selected DESC, option_number
"""

EDGE_PATH_ROWS_FOR_NODE_SQL = """
WITH RECURSIVE path_rows(
    depth, event_index, parent_node_id, child_node_id, candidate_id,
    edge_selector_mode, candidate_text, candidate_token_count
) AS (
    SELECT
        0, event_index, parent_node_id, child_node_id, candidate_id,
        edge_selector_mode, candidate_text, candidate_token_count
    FROM edge_selected
    WHERE doc_id = ? AND doc_attempt = ? AND task_name = ?
      AND model_id = ? AND selector_mode = ? AND child_node_id = ?
    UNION ALL
    SELECT
        path_rows.depth + 1, edge_selected.event_index,
        edge_selected.parent_node_id, edge_selected.child_node_id,
        edge_selected.candidate_id, edge_selected.edge_selector_mode,
        edge_selected.candidate_text, edge_selected.candidate_token_count
    FROM edge_selected
    JOIN path_rows
      ON edge_selected.child_node_id = path_rows.parent_node_id
    WHERE edge_selected.doc_id = ? AND edge_selected.doc_attempt = ?
      AND edge_selected.task_name = ? AND edge_selected.model_id = ?
      AND edge_selected.selector_mode = ?
)
SELECT
    event_index, parent_node_id, child_node_id, candidate_id,
    edge_selector_mode AS selector_mode, candidate_text, candidate_token_count
FROM path_rows
ORDER BY depth DESC
"""

ATTEMPT_SUMMARY_ROW_SQL = """
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
    WHERE doc_id = ? AND doc_attempt = ? AND task_name = ?
      AND model_id = ? AND selector_mode = ?
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
"""

NODE_EVENT_ROWS_SQL = """
SELECT
    event_index, timestamp_utc, event_type, node_id, summary, step_delta,
    token_delta, branch_point_id, leaf_id, verification, stop_reason,
    length_tokens_total,
    CASE
      WHEN event_type = 'prompt_logged'
      THEN COALESCE((
        SELECT golden_answer
        FROM prompt_context
        WHERE prompt_context.event_index = node_event.event_index
      ), '')
      ELSE ''
    END AS text_preview,
    '' AS text,
    (
        SELECT metric_text FROM leaf_metric
        WHERE leaf_event_index = node_event.event_index
          AND metric_name = 'raw_answer_acc'
    ) AS raw_answer_acc,
    (
        SELECT metric_text FROM leaf_metric
        WHERE leaf_event_index = node_event.event_index
          AND metric_name = 'format_valid'
    ) AS format_valid,
    (
        SELECT metric_text FROM leaf_metric
        WHERE leaf_event_index = node_event.event_index
          AND metric_name = 'answer_acc'
    ) AS answer_acc,
    (
        SELECT metric_text FROM leaf_metric
        WHERE leaf_event_index = node_event.event_index
          AND metric_name = 'boxed_answer'
    ) AS boxed_answer,
    (
        SELECT metric_text FROM leaf_metric
        WHERE leaf_event_index = node_event.event_index
          AND metric_name = 'structure_issues'
    ) AS structure_issues
FROM node_event
WHERE doc_id = ? AND doc_attempt = ? AND task_name = ?
  AND model_id = ? AND selector_mode = ?
ORDER BY event_index
"""

NODE_DETAIL_ROWS_SQL = """
SELECT
    node_event.event_index, node_event.timestamp_utc, node_event.event_type,
    node_event.node_id, node_event.summary, node_event.step_delta,
    node_event.token_delta, node_event.branch_point_id, node_event.leaf_id,
    node_event.verification, node_event.stop_reason,
    node_event.length_tokens_total, node_event.request_id,
    node_event.request_stream_id, node_event.request_kind, node_event.status,
    node_event.latency_seconds, node_event.error_message,
    node_event.choice_count, node_event.output_token_count,
    node_event.text_preview, node_event.text,
    (
        SELECT metric_text FROM leaf_metric
        WHERE leaf_event_index = node_event.event_index
          AND metric_name = 'raw_answer_acc'
    ) AS raw_answer_acc,
    (
        SELECT metric_text FROM leaf_metric
        WHERE leaf_event_index = node_event.event_index
          AND metric_name = 'format_valid'
    ) AS format_valid,
    (
        SELECT metric_text FROM leaf_metric
        WHERE leaf_event_index = node_event.event_index
          AND metric_name = 'answer_acc'
    ) AS answer_acc,
    (
        SELECT metric_text FROM leaf_metric
        WHERE leaf_event_index = node_event.event_index
          AND metric_name = 'boxed_answer'
    ) AS boxed_answer,
    (
        SELECT metric_text FROM leaf_metric
        WHERE leaf_event_index = node_event.event_index
          AND metric_name = 'structure_issues'
    ) AS structure_issues,
    {prompt_context_columns}
FROM node_event
{prompt_context_join}
WHERE node_event.doc_id = ? AND node_event.doc_attempt = ?
  AND node_event.task_name = ? AND node_event.model_id = ?
  AND node_event.selector_mode = ? AND node_event.node_id = ?
  AND node_event.event_type IN ({event_type_placeholders})
ORDER BY node_event.event_index
LIMIT ?
"""

NODE_EVENT_ROW_BY_INDEX_SQL = """
SELECT
    node_event.event_index, node_event.doc_id, node_event.doc_attempt,
    node_event.task_name, node_event.model_id, node_event.selector_mode,
    node_event.timestamp_utc, node_event.event_type, node_event.node_id,
    node_event.summary, node_event.step_delta, node_event.token_delta,
    node_event.branch_point_id, node_event.leaf_id, node_event.verification,
    node_event.stop_reason, node_event.length_tokens_total,
    node_event.request_id, node_event.request_stream_id, node_event.request_kind,
    node_event.status, node_event.latency_seconds, node_event.error_message,
    node_event.choice_count, node_event.output_token_count,
    node_event.text_preview, node_event.text,
    (
        SELECT metric_text FROM leaf_metric
        WHERE leaf_event_index = node_event.event_index
          AND metric_name = 'raw_answer_acc'
    ) AS raw_answer_acc,
    (
        SELECT metric_text FROM leaf_metric
        WHERE leaf_event_index = node_event.event_index
          AND metric_name = 'format_valid'
    ) AS format_valid,
    (
        SELECT metric_text FROM leaf_metric
        WHERE leaf_event_index = node_event.event_index
          AND metric_name = 'answer_acc'
    ) AS answer_acc,
    (
        SELECT metric_text FROM leaf_metric
        WHERE leaf_event_index = node_event.event_index
          AND metric_name = 'boxed_answer'
    ) AS boxed_answer,
    (
        SELECT metric_text FROM leaf_metric
        WHERE leaf_event_index = node_event.event_index
          AND metric_name = 'structure_issues'
    ) AS structure_issues,
    {prompt_context_columns}
FROM node_event
{prompt_context_join}
WHERE node_event.event_index = ?
LIMIT 1
"""

PROMPT_TEXT_BY_EVENT_INDEX_SQL = """
SELECT prompt_text
FROM prompt_context
WHERE event_index = ?
LIMIT 1
"""

VLLM_STEP_GRAPH_ROWS_FOR_ATTEMPT_SQL = """
SELECT
    response.event_index AS event_index,
    response.timestamp_utc AS timestamp_utc,
    'vllm_step' AS event_type,
    substr(response.request_stream_id, 8) AS node_id,
    request.request_id,
    response.request_stream_id,
    response.request_kind,
    request.current_input_token_count,
    request.base_prefix_token_count,
    request.delta_token_count,
    request.assistant_prefix_char_count,
    '' AS assistant_prefix_tail,
    response.status,
    response.latency_seconds,
    response.error_message,
    COALESCE(response.choice_count, 0) AS choice_count,
    choice0.output_token_count AS output_token_count,
    1 AS step_delta,
    CASE
      WHEN response.status = 'ok'
      THEN COALESCE(choice0.output_token_count, 0)
      ELSE 0
    END AS token_delta,
    1 AS local_step,
    '' AS summary,
    '' AS branch_point_id,
    '' AS leaf_id,
    NULL AS verification,
    '' AS stop_reason,
    NULL AS length_tokens_total,
    '' AS text_preview
FROM vllm_response AS response
JOIN vllm_request AS request
  ON request.doc_id = response.doc_id
 AND request.doc_attempt = response.doc_attempt
 AND request.task_name = response.task_name
 AND request.model_id = response.model_id
 AND request.selector_mode = response.selector_mode
 AND request.request_id = response.request_id
LEFT JOIN vllm_choice AS choice0
  ON choice0.response_event_index = response.event_index
 AND choice0.choice_index = 0
WHERE response.doc_id = ? AND response.doc_attempt = ? AND response.task_name = ?
  AND response.model_id = ? AND response.selector_mode = ?
  AND response.request_stream_id >= 'decode:'
  AND response.request_stream_id < 'decode;'
ORDER BY response.event_index
"""

VLLM_STEP_ROWS_FOR_NODE_SQL = """
SELECT
    COALESCE(response.event_index, request.event_index) AS event_index,
    COALESCE(response.timestamp_utc, request.timestamp_utc) AS timestamp_utc,
    'vllm_step' AS event_type,
    substr(request.request_stream_id, 8) AS node_id,
    request.request_id,
    request.request_stream_id,
    request.request_kind,
    request.current_input_token_count,
    request.base_prefix_token_count,
    request.delta_token_count,
    request.assistant_prefix_char_count,
    request.assistant_prefix_tail,
    COALESCE(response.status, 'pending') AS status,
    response.latency_seconds,
    COALESCE(response.error_message, '') AS error_message,
    COALESCE(response.choice_count, 0) AS choice_count,
    choice0.output_token_count AS output_token_count,
    1 AS step_delta,
    CASE
      WHEN COALESCE(response.status, 'pending') = 'ok'
      THEN COALESCE(choice0.output_token_count, 0)
      ELSE 0
    END AS token_delta,
    ROW_NUMBER() OVER (
        PARTITION BY substr(request.request_stream_id, 8)
        ORDER BY COALESCE(response.event_index, request.event_index)
    ) AS local_step,
    '' AS summary,
    '' AS branch_point_id,
    '' AS leaf_id,
    NULL AS verification,
    '' AS stop_reason,
    NULL AS length_tokens_total,
    '' AS text_preview
FROM vllm_request AS request
LEFT JOIN vllm_response AS response
  ON response.doc_id = request.doc_id
 AND response.doc_attempt = request.doc_attempt
 AND response.task_name = request.task_name
 AND response.model_id = request.model_id
 AND response.selector_mode = request.selector_mode
 AND response.request_id = request.request_id
LEFT JOIN vllm_choice AS choice0
  ON choice0.response_event_index = response.event_index
 AND choice0.choice_index = 0
WHERE request.doc_id = ? AND request.doc_attempt = ? AND request.task_name = ?
  AND request.model_id = ? AND request.selector_mode = ?
  AND request.request_stream_id = ?
ORDER BY event_index
LIMIT ?
"""

VLLM_RESPONSE_STEP_ROW_BY_EVENT_SQL = """
SELECT
    response.event_index AS event_index,
    response.timestamp_utc AS timestamp_utc,
    'vllm_step' AS event_type,
    substr(request.request_stream_id, 8) AS node_id,
    request.request_id,
    request.request_stream_id,
    request.request_kind,
    request.current_input_token_count,
    request.base_prefix_token_count,
    request.delta_token_count,
    request.assistant_prefix_char_count,
    request.assistant_prefix_tail,
    COALESCE(response.status, 'pending') AS status,
    response.latency_seconds,
    COALESCE(response.error_message, '') AS error_message,
    COALESCE(response.choice_count, 0) AS choice_count,
    choice0.output_token_count AS output_token_count,
    1 AS step_delta,
    CASE
      WHEN COALESCE(response.status, 'pending') = 'ok'
      THEN COALESCE(choice0.output_token_count, 0)
      ELSE 0
    END AS token_delta,
    1 AS local_step,
    '' AS summary,
    '' AS branch_point_id,
    '' AS leaf_id,
    NULL AS verification,
    '' AS stop_reason,
    NULL AS length_tokens_total,
    '' AS text_preview
FROM vllm_request AS request
JOIN vllm_response AS response
  ON response.doc_id = request.doc_id
 AND response.doc_attempt = request.doc_attempt
 AND response.task_name = request.task_name
 AND response.model_id = request.model_id
 AND response.selector_mode = request.selector_mode
 AND response.request_id = request.request_id
LEFT JOIN vllm_choice AS choice0
  ON choice0.response_event_index = response.event_index
 AND choice0.choice_index = 0
WHERE response.event_index = ?
LIMIT 1
"""

VLLM_PENDING_REQUEST_STEP_ROW_BY_EVENT_SQL = """
SELECT
    request.event_index AS event_index,
    request.timestamp_utc AS timestamp_utc,
    'vllm_step' AS event_type,
    substr(request.request_stream_id, 8) AS node_id,
    request.request_id,
    request.request_stream_id,
    request.request_kind,
    request.current_input_token_count,
    request.base_prefix_token_count,
    request.delta_token_count,
    request.assistant_prefix_char_count,
    request.assistant_prefix_tail,
    'pending' AS status,
    NULL AS latency_seconds,
    '' AS error_message,
    0 AS choice_count,
    NULL AS output_token_count,
    1 AS step_delta,
    0 AS token_delta,
    1 AS local_step,
    '' AS summary,
    '' AS branch_point_id,
    '' AS leaf_id,
    NULL AS verification,
    '' AS stop_reason,
    NULL AS length_tokens_total,
    '' AS text_preview
FROM vllm_request AS request
LEFT JOIN vllm_response AS response
  ON response.doc_id = request.doc_id
 AND response.doc_attempt = request.doc_attempt
 AND response.task_name = request.task_name
 AND response.model_id = request.model_id
 AND response.selector_mode = request.selector_mode
 AND response.request_id = request.request_id
WHERE request.event_index = ?
  AND response.event_index IS NULL
LIMIT 1
"""

VLLM_TEXT_ROWS_FOR_NODE_SQL = """
SELECT
    response.event_index,
    choice0.text,
    COALESCE(choice0.output_token_count, 0) AS output_token_count
FROM vllm_request AS request
JOIN vllm_response AS response
  ON response.doc_id = request.doc_id
 AND response.doc_attempt = request.doc_attempt
 AND response.task_name = request.task_name
 AND response.model_id = request.model_id
 AND response.selector_mode = request.selector_mode
 AND response.request_id = request.request_id
JOIN vllm_choice AS choice0
  ON choice0.response_event_index = response.event_index
 AND choice0.choice_index = 0
WHERE request.doc_id = ? AND request.doc_attempt = ? AND request.task_name = ?
  AND request.model_id = ? AND request.selector_mode = ?
  AND request.request_stream_id = ?
  AND response.status = 'ok'
ORDER BY response.event_index
"""

VLLM_TOKEN_TRAJECTORY_ROWS_FOR_NODE_SQL = """
SELECT
    response.event_index AS response_event_index,
    request.event_index AS request_event_index,
    request.request_id,
    request.request_kind,
    request.assistant_prefix_char_count,
    request.assistant_prefix_tail,
    token.token_index AS step_token_index,
    token.token_id,
    token.token_text,
    token.selected_logprob,
    token.selected_probability
FROM vllm_request AS request INDEXED BY idx_vllm_request_stream
CROSS JOIN vllm_response AS response INDEXED BY idx_vllm_response_id
CROSS JOIN vllm_choice_token AS token INDEXED BY idx_vllm_choice_token_event
WHERE request.doc_id = ? AND request.doc_attempt = ? AND request.task_name = ?
  AND request.model_id = ? AND request.selector_mode = ?
  AND request.request_stream_id = ?
  AND response.doc_id = request.doc_id
  AND response.doc_attempt = request.doc_attempt
  AND response.task_name = request.task_name
  AND response.model_id = request.model_id
  AND response.selector_mode = request.selector_mode
  AND response.request_id = request.request_id
  AND response.status = 'ok'
  AND token.response_event_index = response.event_index
  AND token.choice_index = 0
ORDER BY request.event_index, response.event_index, token.token_index
"""

VLLM_TOKEN_TRAJECTORY_ROWS_FOR_STREAMS_SQL = """
SELECT
    response.event_index AS response_event_index,
    request.event_index AS request_event_index,
    request.request_id,
    request.request_kind,
    request.assistant_prefix_char_count,
    request.assistant_prefix_tail,
    token.token_index AS step_token_index,
    token.token_id,
    token.token_text,
    token.selected_logprob,
    token.selected_probability
FROM vllm_request AS request INDEXED BY idx_vllm_request_stream
CROSS JOIN vllm_response AS response INDEXED BY idx_vllm_response_id
CROSS JOIN vllm_choice_token AS token INDEXED BY idx_vllm_choice_token_event
WHERE request.doc_id = ? AND request.doc_attempt = ? AND request.task_name = ?
  AND request.model_id = ? AND request.selector_mode = ?
  AND request.request_stream_id IN ({placeholders})
  AND response.doc_id = request.doc_id
  AND response.doc_attempt = request.doc_attempt
  AND response.task_name = request.task_name
  AND response.model_id = request.model_id
  AND response.selector_mode = request.selector_mode
  AND response.request_id = request.request_id
  AND response.status = 'ok'
  AND token.response_event_index = response.event_index
  AND token.choice_index = 0
ORDER BY request.event_index, response.event_index, token.token_index
"""

LEAF_ROWS_SQL = """
SELECT
    leaf_score.leaf_id, leaf_score.node_id, leaf_score.verification,
    leaf_score.length_tokens_total, leaf_score.stop_reason, leaf_score.text,
    leaf_score.text_preview, leaf_score.event_index,
    (
        SELECT metric_text FROM leaf_metric
        WHERE leaf_event_index = leaf_score.event_index
          AND metric_name = 'raw_answer_acc'
    ) AS raw_answer_acc,
    (
        SELECT metric_text FROM leaf_metric
        WHERE leaf_event_index = leaf_score.event_index
          AND metric_name = 'format_valid'
    ) AS format_valid,
    (
        SELECT metric_text FROM leaf_metric
        WHERE leaf_event_index = leaf_score.event_index
          AND metric_name = 'answer_acc'
    ) AS answer_acc,
    (
        SELECT metric_text FROM leaf_metric
        WHERE leaf_event_index = leaf_score.event_index
          AND metric_name = 'boxed_answer'
    ) AS boxed_answer,
    (
        SELECT metric_text FROM leaf_metric
        WHERE leaf_event_index = leaf_score.event_index
          AND metric_name = 'structure_issues'
    ) AS structure_issues
FROM leaf_score
WHERE leaf_score.doc_id = ? AND leaf_score.doc_attempt = ?
  AND leaf_score.task_name = ? AND leaf_score.model_id = ?
  AND leaf_score.selector_mode = ?
ORDER BY leaf_score.event_index
"""

BASELINE_LEAF_ROWS_SQL = """
SELECT
    leaf_score.leaf_id, leaf_score.node_id, leaf_score.verification,
    leaf_score.length_tokens_total, leaf_score.stop_reason, leaf_score.text,
    leaf_score.text_preview, leaf_score.event_index,
    (
        SELECT metric_text FROM leaf_metric
        WHERE leaf_event_index = leaf_score.event_index
          AND metric_name = 'raw_answer_acc'
    ) AS raw_answer_acc,
    (
        SELECT metric_text FROM leaf_metric
        WHERE leaf_event_index = leaf_score.event_index
          AND metric_name = 'format_valid'
    ) AS format_valid,
    (
        SELECT metric_text FROM leaf_metric
        WHERE leaf_event_index = leaf_score.event_index
          AND metric_name = 'answer_acc'
    ) AS answer_acc,
    (
        SELECT metric_text FROM leaf_metric
        WHERE leaf_event_index = leaf_score.event_index
          AND metric_name = 'boxed_answer'
    ) AS boxed_answer,
    (
        SELECT metric_text FROM leaf_metric
        WHERE leaf_event_index = leaf_score.event_index
          AND metric_name = 'structure_issues'
    ) AS structure_issues
FROM leaf_score
WHERE leaf_score.doc_id = ? AND leaf_score.doc_attempt = ?
  AND leaf_score.task_name = ? AND leaf_score.model_id = ?
  AND leaf_score.selector_mode = ?
  AND leaf_score.leaf_id LIKE 'leaf_baseline_%'
ORDER BY leaf_score.event_index
"""

LEAF_ROWS_FOR_NODE_SQL = """
SELECT
    leaf_score.leaf_id, leaf_score.node_id, leaf_score.verification,
    leaf_score.length_tokens_total, leaf_score.stop_reason, leaf_score.text,
    leaf_score.text_preview, leaf_score.event_index,
    (
        SELECT metric_text FROM leaf_metric
        WHERE leaf_event_index = leaf_score.event_index
          AND metric_name = 'raw_answer_acc'
    ) AS raw_answer_acc,
    (
        SELECT metric_text FROM leaf_metric
        WHERE leaf_event_index = leaf_score.event_index
          AND metric_name = 'format_valid'
    ) AS format_valid,
    (
        SELECT metric_text FROM leaf_metric
        WHERE leaf_event_index = leaf_score.event_index
          AND metric_name = 'answer_acc'
    ) AS answer_acc,
    (
        SELECT metric_text FROM leaf_metric
        WHERE leaf_event_index = leaf_score.event_index
          AND metric_name = 'boxed_answer'
    ) AS boxed_answer,
    (
        SELECT metric_text FROM leaf_metric
        WHERE leaf_event_index = leaf_score.event_index
          AND metric_name = 'structure_issues'
    ) AS structure_issues
FROM leaf_score
WHERE leaf_score.doc_id = ? AND leaf_score.doc_attempt = ?
  AND leaf_score.task_name = ? AND leaf_score.model_id = ?
  AND leaf_score.selector_mode = ? AND leaf_score.node_id = ?
ORDER BY leaf_score.event_index
"""

CANDIDATE_ROWS_FOR_EVENTS_SQL = """
SELECT pool_event_index, candidate_id, text, text_preview, output_token_count,
       finish_reason, stop_reason
FROM candidate_pool_candidate
WHERE pool_event_index IN ({placeholders})
ORDER BY pool_event_index, candidate_id
"""

CANDIDATE_TOKENS_FOR_EVENTS_SQL = """
SELECT pool_event_index, candidate_id, token_index, token_id, token_text,
       selected_logprob, selected_probability
FROM candidate_pool_candidate_token
WHERE pool_event_index IN ({placeholders})
ORDER BY pool_event_index, candidate_id, token_index
"""

SELECTOR_FLAGS_FOR_EVENTS_SQL = """
SELECT selector_event_index, mode_name, candidate_id, selected, shortlisted
FROM selector_candidate_flag
WHERE selector_event_index IN ({placeholders})
ORDER BY selector_event_index, mode_name, candidate_id
"""

SELECTOR_CLUSTERS_FOR_EVENTS_SQL = """
SELECT selector_event_index, mode_name, candidate_id, cluster_name
FROM selector_candidate_cluster
WHERE selector_event_index IN ({placeholders})
ORDER BY selector_event_index, mode_name, cluster_name, candidate_id
"""

SELECTOR_POOL_ROWS_FOR_EVENTS_SQL = """
SELECT
    selector_event.event_index AS selector_event_index,
    selector_event.selector_event_type,
    selector_event.active_selector_mode,
    candidate_pool.event_index AS pool_event_index,
    candidate_pool.doc_id,
    candidate_pool.doc_attempt,
    candidate_pool.task_name,
    candidate_pool.model_id,
    candidate_pool.selector_mode,
    candidate_pool.timestamp_utc,
    candidate_pool.branch_point_id,
    candidate_pool.candidate_pool_id,
    candidate_pool.node_id,
    candidate_pool.trigger_type,
    candidate_pool.num_candidates
FROM selector_event
JOIN candidate_pool
  ON candidate_pool.doc_id = selector_event.doc_id
 AND candidate_pool.doc_attempt = selector_event.doc_attempt
 AND candidate_pool.task_name = selector_event.task_name
 AND candidate_pool.model_id = selector_event.model_id
 AND candidate_pool.selector_mode = selector_event.selector_mode
 AND candidate_pool.branch_point_id = selector_event.branch_point_id
WHERE selector_event.event_index IN ({placeholders})
ORDER BY selector_event.event_index, candidate_pool.event_index
"""

VLLM_CHOICES_FOR_EVENTS_SQL = """
SELECT response_event_index, choice_index, text, text_preview, finish_reason,
       stop_reason, output_token_count
FROM vllm_choice
WHERE response_event_index IN ({placeholders})
ORDER BY response_event_index, choice_index
"""

VLLM_CHOICE_TOKENS_FOR_EVENTS_SQL = """
SELECT response_event_index, choice_index, token_index, token_id, token_text,
       selected_logprob, selected_probability
FROM vllm_choice_token
WHERE response_event_index IN ({placeholders})
ORDER BY response_event_index, choice_index, token_index
"""

GENERATED_CHUNKS_FOR_EVENTS_SQL = """
SELECT event_index, timestamp_utc, event_type, node_id, chunk_text, token_count,
       generated_tokens_before_chunk, generated_tokens_after_chunk,
       chunk_was_normalized, chunk_token_ids_source, source
FROM generated_chunk
WHERE event_index IN ({placeholders})
ORDER BY event_index
"""

GENERATED_CHUNKS_FOR_NODE_SQL = """
SELECT event_index, timestamp_utc, event_type, node_id, chunk_text, token_count,
       generated_tokens_before_chunk, generated_tokens_after_chunk,
       chunk_was_normalized, chunk_token_ids_source, source
FROM generated_chunk
WHERE doc_id = ? AND doc_attempt = ? AND task_name = ?
  AND model_id = ? AND selector_mode = ? AND node_id = ?
ORDER BY event_index
"""

BACKFILL_SQL_STATEMENTS = (
    """
    INSERT OR IGNORE INTO node_created
    SELECT event_index, doc_id, doc_attempt, task_name, model_id, selector_mode,
           timestamp_utc,
           COALESCE(json_extract(payload_json, '$.node_id'), ''),
           json_extract(payload_json, '$.parent_node_id'),
           COALESCE(json_extract(payload_json, '$.branch_points_used'), 0)
    FROM event_log
    WHERE event_index > :start AND event_index <= :end
      AND event_type = 'node_created'
      AND doc_id IS NOT NULL AND doc_attempt IS NOT NULL
    """,
    """
    INSERT OR IGNORE INTO edge_selected
    SELECT event_index, doc_id, doc_attempt, task_name, model_id, selector_mode,
           timestamp_utc,
           COALESCE(json_extract(payload_json, '$.parent_node_id'), ''),
           COALESCE(json_extract(payload_json, '$.child_node_id'), ''),
           json_extract(payload_json, '$.candidate_id'),
           COALESCE(json_extract(payload_json, '$.selector_mode'), ''),
           COALESCE(json_extract(payload_json, '$.candidate_text_normalized'), ''),
           COALESCE(json_array_length(json_extract(payload_json, '$.candidate_token_ids_normalized')), 0)
    FROM event_log
    WHERE event_index > :start AND event_index <= :end
      AND event_type = 'edge_selected'
      AND doc_id IS NOT NULL AND doc_attempt IS NOT NULL
    """,
    """
    INSERT OR IGNORE INTO leaf_score
    SELECT event_index, doc_id, doc_attempt, task_name, model_id, selector_mode,
           timestamp_utc,
           COALESCE(json_extract(payload_json, '$.leaf_id'), ''),
           COALESCE(json_extract(payload_json, '$.node_id'), ''),
           json_extract(payload_json, '$.verification'),
           json_extract(payload_json, '$.length_tokens_total'),
           json_extract(payload_json, '$.length_tokens_exec'),
           COALESCE(json_extract(payload_json, '$.stop_reason'), ''),
           COALESCE(json_extract(payload_json, '$.text'), json_extract(payload_json, '$.text_preview'), ''),
           COALESCE(json_extract(payload_json, '$.text_preview'), json_extract(payload_json, '$.text'), '')
    FROM event_log
    WHERE event_index > :start AND event_index <= :end
      AND event_type = 'leaf_scored'
      AND doc_id IS NOT NULL AND doc_attempt IS NOT NULL
    """,
    """
    INSERT OR REPLACE INTO leaf_metric
    SELECT event_log.event_index, metric.key,
           CASE WHEN json_type(metric.value) IN ('integer', 'real') THEN metric.value END,
           CAST(metric.value AS TEXT)
    FROM event_log, json_each(event_log.payload_json, '$.task_metrics') AS metric
    WHERE event_log.event_index > :start AND event_log.event_index <= :end
      AND event_log.event_type = 'leaf_scored'
      AND event_log.doc_id IS NOT NULL AND event_log.doc_attempt IS NOT NULL
    """,
    """
    INSERT OR IGNORE INTO candidate_pool
    SELECT event_index, doc_id, doc_attempt, task_name, model_id, selector_mode,
           timestamp_utc,
           COALESCE(json_extract(payload_json, '$.branch_point_id'), ''),
           COALESCE(json_extract(payload_json, '$.candidate_pool_id'), ''),
           COALESCE(json_extract(payload_json, '$.node_id'), ''),
           COALESCE(json_extract(payload_json, '$.trigger_type'), ''),
           COALESCE(json_extract(payload_json, '$.num_candidates'), 0)
    FROM event_log
    WHERE event_index > :start AND event_index <= :end
      AND event_type = 'candidate_pool_resolved'
      AND doc_id IS NOT NULL AND doc_attempt IS NOT NULL
    """,
    """
    INSERT OR REPLACE INTO candidate_pool_candidate
    SELECT event_log.event_index,
           COALESCE(json_extract(candidate.value, '$.candidate_id'), -1),
           COALESCE(json_extract(candidate.value, '$.text'), json_extract(candidate.value, '$.text_preview'), ''),
           COALESCE(json_extract(candidate.value, '$.text_preview'), json_extract(candidate.value, '$.text'), ''),
           json_extract(candidate.value, '$.output_token_count'),
           COALESCE(json_extract(candidate.value, '$.finish_reason'), ''),
           COALESCE(json_extract(candidate.value, '$.stop_reason'), '')
    FROM event_log, json_each(event_log.payload_json, '$.candidates') AS candidate
    WHERE event_log.event_index > :start AND event_log.event_index <= :end
      AND event_log.event_type = 'candidate_pool_resolved'
      AND event_log.doc_id IS NOT NULL AND event_log.doc_attempt IS NOT NULL
    """,
    """
    INSERT OR IGNORE INTO selector_event
    SELECT event_index, doc_id, doc_attempt, task_name, model_id, selector_mode,
           timestamp_utc, event_type,
           COALESCE(json_extract(payload_json, '$.branch_point_id'), ''),
           COALESCE(json_extract(payload_json, '$.node_id'), ''),
           COALESCE(json_extract(payload_json, '$.active_selector_mode'), '')
    FROM event_log
    WHERE event_index > :start AND event_index <= :end
      AND event_type IN ('selector_applied', 'selector_continued_inline')
      AND doc_id IS NOT NULL AND doc_attempt IS NOT NULL
    """,
    """
    INSERT OR REPLACE INTO prompt_context
    SELECT event_index, doc_id, doc_attempt, task_name, model_id, selector_mode,
           timestamp_utc,
           COALESCE(json_extract(payload_json, '$.node_id'), 'node_root'),
           COALESCE(json_extract(payload_json, '$.prompt_text'), ''),
           json_extract(payload_json, '$.prompt_char_count'),
           COALESCE(json_extract(payload_json, '$.golden_answer'), ''),
           COALESCE(json_extract(payload_json, '$.golden_answer_source'), '')
    FROM event_log
    WHERE event_index > :start AND event_index <= :end
      AND event_type = 'prompt_logged'
      AND doc_id IS NOT NULL AND doc_attempt IS NOT NULL
    """,
    """
    INSERT OR REPLACE INTO selector_candidate_flag
    SELECT event_log.event_index, 'active', selected.value, 1, 0
    FROM event_log, json_each(event_log.payload_json, '$.selected_candidate_ids') AS selected
    WHERE event_log.event_index > :start AND event_log.event_index <= :end
      AND event_log.event_type = 'selector_applied'
      AND event_log.doc_id IS NOT NULL AND event_log.doc_attempt IS NOT NULL
    """,
    """
    INSERT OR REPLACE INTO selector_candidate_flag
    SELECT event_log.event_index, 'active',
           COALESCE(json_extract(event_log.payload_json, '$.selected_candidate_id'), -1),
           1, 0
    FROM event_log
    WHERE event_log.event_index > :start AND event_log.event_index <= :end
      AND event_log.event_type = 'selector_continued_inline'
      AND event_log.doc_id IS NOT NULL AND event_log.doc_attempt IS NOT NULL
    """,
    """
    INSERT OR REPLACE INTO selector_candidate_flag
    SELECT event_log.event_index, selected_by_mode.key, selected.value, 1, 0
    FROM event_log,
         json_each(event_log.payload_json, '$.selected_by_mode') AS selected_by_mode,
         json_each(selected_by_mode.value) AS selected
    WHERE event_log.event_index > :start AND event_log.event_index <= :end
      AND event_log.event_type IN ('selector_applied', 'selector_continued_inline')
      AND event_log.doc_id IS NOT NULL AND event_log.doc_attempt IS NOT NULL
    """,
    """
    INSERT OR REPLACE INTO selector_candidate_flag
    SELECT event_log.event_index, shortlist_by_mode.key, shortlisted.value, 0, 1
    FROM event_log,
         json_each(event_log.payload_json, '$.shortlist_by_mode') AS shortlist_by_mode,
         json_each(shortlist_by_mode.value) AS shortlisted
    WHERE event_log.event_index > :start AND event_log.event_index <= :end
      AND event_log.event_type IN ('selector_applied', 'selector_continued_inline')
      AND event_log.doc_id IS NOT NULL AND event_log.doc_attempt IS NOT NULL
    """,
    """
    INSERT OR REPLACE INTO selector_candidate_cluster
    SELECT event_log.event_index,
           cluster_assignments_by_mode.key,
           COALESCE(json_extract(assignment.value, '$.candidate_id'), -1),
           COALESCE(json_extract(assignment.value, '$.cluster_name'), '')
    FROM event_log,
         json_each(event_log.payload_json, '$.cluster_assignments_by_mode') AS cluster_assignments_by_mode,
         json_each(cluster_assignments_by_mode.value) AS assignment
    WHERE event_log.event_index > :start AND event_log.event_index <= :end
      AND event_log.event_type IN ('selector_applied', 'selector_continued_inline')
      AND event_log.doc_id IS NOT NULL AND event_log.doc_attempt IS NOT NULL
    """,
    """
    INSERT OR IGNORE INTO vllm_request
    SELECT event_index, doc_id, doc_attempt, task_name, model_id, selector_mode,
           timestamp_utc,
           COALESCE(json_extract(payload_json, '$.request_id'), ''),
           COALESCE(json_extract(payload_json, '$.request_stream_id'), ''),
           COALESCE(json_extract(payload_json, '$.request_kind'), ''),
           json_extract(payload_json, '$.prev_request_id'),
           json_extract(payload_json, '$.current_input_token_count'),
           json_extract(payload_json, '$.base_prefix_token_count'),
           json_extract(payload_json, '$.delta_token_count'),
           json_extract(payload_json, '$.assistant_prefix_char_count'),
           COALESCE(json_extract(payload_json, '$.assistant_prefix_tail'), '')
    FROM event_log
    WHERE event_index > :start AND event_index <= :end
      AND event_type = 'vllm_request'
      AND doc_id IS NOT NULL AND doc_attempt IS NOT NULL
    """,
    """
    INSERT OR IGNORE INTO vllm_response
    SELECT event_index, doc_id, doc_attempt, task_name, model_id, selector_mode,
           timestamp_utc,
           COALESCE(json_extract(payload_json, '$.request_id'), ''),
           COALESCE(json_extract(payload_json, '$.request_stream_id'), ''),
           COALESCE(json_extract(payload_json, '$.request_kind'), ''),
           COALESCE(json_extract(payload_json, '$.status'), ''),
           json_extract(payload_json, '$.latency_seconds'),
           COALESCE(json_extract(payload_json, '$.error_message'), ''),
           COALESCE(json_extract(payload_json, '$.choice_count'), 0)
    FROM event_log
    WHERE event_index > :start AND event_index <= :end
      AND event_type = 'vllm_response'
      AND doc_id IS NOT NULL AND doc_attempt IS NOT NULL
    """,
    """
    INSERT OR REPLACE INTO vllm_choice
    SELECT event_log.event_index,
           COALESCE(json_extract(choice.value, '$.index'), 0),
           COALESCE(json_extract(choice.value, '$.text'), json_extract(choice.value, '$.text_preview'), ''),
           COALESCE(json_extract(choice.value, '$.text_preview'), json_extract(choice.value, '$.text'), ''),
           COALESCE(json_extract(choice.value, '$.finish_reason'), ''),
           COALESCE(json_extract(choice.value, '$.stop_reason'), ''),
           json_extract(choice.value, '$.output_token_count')
    FROM event_log, json_each(event_log.payload_json, '$.choices') AS choice
    WHERE event_log.event_index > :start AND event_log.event_index <= :end
      AND event_log.event_type = 'vllm_response'
      AND event_log.doc_id IS NOT NULL AND event_log.doc_attempt IS NOT NULL
    """,
    """
    INSERT OR IGNORE INTO node_event
    SELECT event_index, doc_id, doc_attempt, task_name, model_id, selector_mode,
           timestamp_utc,
           event_type,
           COALESCE(json_extract(payload_json, '$.node_id'), ''),
           CASE
             WHEN event_type = 'prompt_logged' THEN
               CASE
                 WHEN COALESCE(json_extract(payload_json, '$.golden_answer'), '') = ''
                 THEN 'input prompt'
                 ELSE 'input prompt · gold ' || COALESCE(json_extract(payload_json, '$.golden_answer'), '')
               END
             WHEN event_type = 'trigger_fired' THEN 'trigger ' || COALESCE(json_extract(payload_json, '$.trigger_type'), '')
             WHEN event_type = 'trigger_skipped_max_branch_points' THEN 'trigger skipped max branch points'
             WHEN event_type = 'candidate_pool_resolved' THEN 'candidate pool n=' || COALESCE(json_extract(payload_json, '$.num_candidates'), 0)
             WHEN event_type = 'selector_applied' THEN 'selector kept ' || COALESCE(json_array_length(json_extract(payload_json, '$.selected_candidate_ids')), 0)
             WHEN event_type = 'selector_continued_inline' THEN 'selector continued inline ' || COALESCE(json_extract(payload_json, '$.selected_candidate_id'), '')
             WHEN event_type = 'verbalized_sampling_applied' THEN 'verbalized sampling selected ' || COALESCE(json_extract(payload_json, '$.sampled_option_numbers[0]'), '')
             WHEN event_type = 'leaf_completed' THEN 'leaf completed ' || COALESCE(json_extract(payload_json, '$.leaf_id'), '')
             WHEN event_type = 'leaf_scored' THEN 'leaf scored verify=' || COALESCE(json_extract(payload_json, '$.verification'), '')
             ELSE event_type
           END,
           1,
           0,
           COALESCE(json_extract(payload_json, '$.branch_point_id'), ''),
           COALESCE(json_extract(payload_json, '$.leaf_id'), ''),
           json_extract(payload_json, '$.verification'),
           COALESCE(json_extract(payload_json, '$.stop_reason'), ''),
           json_extract(payload_json, '$.length_tokens_total'),
           '',
           '',
           '',
           '',
           NULL,
           '',
           NULL,
           NULL,
           COALESCE(json_extract(payload_json, '$.text_preview'), ''),
           CASE
             WHEN event_type IN ('leaf_completed', 'leaf_scored') THEN
               COALESCE(json_extract(payload_json, '$.text'), json_extract(payload_json, '$.text_preview'), '')
             WHEN event_type = 'malformed_steer_decision' THEN
               COALESCE(json_extract(payload_json, '$.candidate_text'), json_extract(payload_json, '$.assistant_prefix_tail'), '')
             WHEN event_type = 'verbalized_sampling_applied' THEN
               COALESCE(json_extract(payload_json, '$.enumeration_exec_text'), '')
             ELSE ''
           END
    FROM event_log
    WHERE event_index > :start AND event_index <= :end
      AND event_type IN (
          'prompt_logged',
          'trigger_fired', 'trigger_skipped_max_branch_points',
          'candidate_pool_resolved', 'selector_applied',
          'selector_continued_inline', 'verbalized_sampling_applied',
          'leaf_completed', 'leaf_scored'
      )
      AND doc_id IS NOT NULL AND doc_attempt IS NOT NULL
    """,
)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS event_log (
    event_index INTEGER PRIMARY KEY,
    event_version INTEGER NOT NULL,
    timestamp_utc TEXT NOT NULL,
    run_id TEXT NOT NULL,
    doc_id INTEGER,
    doc_attempt INTEGER,
    task_name TEXT NOT NULL,
    model_id TEXT NOT NULL,
    selector_mode TEXT NOT NULL,
    event_type TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    row_json TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_event_log_doc
ON event_log(doc_id, doc_attempt, event_index);

CREATE INDEX IF NOT EXISTS idx_event_log_attempt_full
ON event_log(doc_id, doc_attempt, task_name, model_id, selector_mode, event_index);

CREATE INDEX IF NOT EXISTS idx_event_log_type
ON event_log(event_type, event_index);

CREATE INDEX IF NOT EXISTS idx_event_log_selector
ON event_log(task_name, model_id, selector_mode, event_index);

CREATE TABLE IF NOT EXISTS doc_progress (
    run_id TEXT NOT NULL,
    doc_id INTEGER NOT NULL,
    doc_attempt INTEGER NOT NULL,
    payload_json TEXT NOT NULL,
    last_update_timestamp TEXT NOT NULL,
    PRIMARY KEY(run_id, doc_id, doc_attempt)
);

CREATE TABLE IF NOT EXISTS metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS node_created (
    event_index INTEGER PRIMARY KEY,
    doc_id INTEGER NOT NULL,
    doc_attempt INTEGER NOT NULL,
    task_name TEXT NOT NULL,
    model_id TEXT NOT NULL,
    selector_mode TEXT NOT NULL,
    timestamp_utc TEXT NOT NULL,
    node_id TEXT NOT NULL,
    parent_node_id TEXT,
    branch_points_used INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_node_created_attempt
ON node_created(doc_id, doc_attempt, task_name, model_id, selector_mode, node_id);

CREATE TABLE IF NOT EXISTS edge_selected (
    event_index INTEGER PRIMARY KEY,
    doc_id INTEGER NOT NULL,
    doc_attempt INTEGER NOT NULL,
    task_name TEXT NOT NULL,
    model_id TEXT NOT NULL,
    selector_mode TEXT NOT NULL,
    timestamp_utc TEXT NOT NULL,
    parent_node_id TEXT NOT NULL,
    child_node_id TEXT NOT NULL,
    candidate_id INTEGER,
    edge_selector_mode TEXT NOT NULL,
    candidate_text TEXT NOT NULL,
    candidate_token_count INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_edge_selected_attempt
ON edge_selected(doc_id, doc_attempt, task_name, model_id, selector_mode, child_node_id);

CREATE TABLE IF NOT EXISTS leaf_score (
    event_index INTEGER PRIMARY KEY,
    doc_id INTEGER NOT NULL,
    doc_attempt INTEGER NOT NULL,
    task_name TEXT NOT NULL,
    model_id TEXT NOT NULL,
    selector_mode TEXT NOT NULL,
    timestamp_utc TEXT NOT NULL,
    leaf_id TEXT NOT NULL,
    node_id TEXT NOT NULL,
    verification INTEGER,
    length_tokens_total INTEGER,
    length_tokens_exec INTEGER,
    stop_reason TEXT NOT NULL,
    text TEXT NOT NULL,
    text_preview TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_leaf_score_attempt
ON leaf_score(doc_id, doc_attempt, task_name, model_id, selector_mode, node_id);

CREATE TABLE IF NOT EXISTS leaf_metric (
    leaf_event_index INTEGER NOT NULL,
    metric_name TEXT NOT NULL,
    metric_value REAL,
    metric_text TEXT NOT NULL,
    PRIMARY KEY(leaf_event_index, metric_name)
);

CREATE TABLE IF NOT EXISTS node_event (
    event_index INTEGER PRIMARY KEY,
    doc_id INTEGER NOT NULL,
    doc_attempt INTEGER NOT NULL,
    task_name TEXT NOT NULL,
    model_id TEXT NOT NULL,
    selector_mode TEXT NOT NULL,
    timestamp_utc TEXT NOT NULL,
    event_type TEXT NOT NULL,
    node_id TEXT NOT NULL,
    summary TEXT NOT NULL,
    step_delta INTEGER NOT NULL,
    token_delta INTEGER NOT NULL,
    branch_point_id TEXT NOT NULL,
    leaf_id TEXT NOT NULL,
    verification INTEGER,
    stop_reason TEXT NOT NULL,
    length_tokens_total INTEGER,
    request_id TEXT NOT NULL,
    request_stream_id TEXT NOT NULL,
    request_kind TEXT NOT NULL,
    status TEXT NOT NULL,
    latency_seconds REAL,
    error_message TEXT NOT NULL,
    choice_count INTEGER,
    output_token_count INTEGER,
    text_preview TEXT NOT NULL,
    text TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_node_event_attempt_node
ON node_event(doc_id, doc_attempt, task_name, model_id, selector_mode, node_id, event_index);

CREATE INDEX IF NOT EXISTS idx_node_event_attempt_type
ON node_event(doc_id, doc_attempt, task_name, model_id, selector_mode, event_type, event_index);

CREATE INDEX IF NOT EXISTS idx_node_event_attempt_event
ON node_event(doc_id, doc_attempt, task_name, model_id, selector_mode, event_index);

CREATE INDEX IF NOT EXISTS idx_node_event_graph_cover
ON node_event(
    doc_id, doc_attempt, task_name, model_id, selector_mode, event_index,
    timestamp_utc, event_type, node_id, summary, step_delta, token_delta,
    branch_point_id, leaf_id, verification, stop_reason, length_tokens_total
);

CREATE TABLE IF NOT EXISTS prompt_context (
    event_index INTEGER PRIMARY KEY,
    doc_id INTEGER NOT NULL,
    doc_attempt INTEGER NOT NULL,
    task_name TEXT NOT NULL,
    model_id TEXT NOT NULL,
    selector_mode TEXT NOT NULL,
    timestamp_utc TEXT NOT NULL,
    node_id TEXT NOT NULL,
    prompt_text TEXT NOT NULL,
    prompt_char_count INTEGER,
    golden_answer TEXT NOT NULL,
    golden_answer_source TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_prompt_context_attempt
ON prompt_context(doc_id, doc_attempt, task_name, model_id, selector_mode, event_index);

CREATE TABLE IF NOT EXISTS candidate_pool (
    event_index INTEGER PRIMARY KEY,
    doc_id INTEGER NOT NULL,
    doc_attempt INTEGER NOT NULL,
    task_name TEXT NOT NULL,
    model_id TEXT NOT NULL,
    selector_mode TEXT NOT NULL,
    timestamp_utc TEXT NOT NULL,
    branch_point_id TEXT NOT NULL,
    candidate_pool_id TEXT NOT NULL,
    node_id TEXT NOT NULL,
    trigger_type TEXT NOT NULL,
    num_candidates INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_candidate_pool_branch
ON candidate_pool(doc_id, doc_attempt, task_name, model_id, selector_mode, branch_point_id);

CREATE TABLE IF NOT EXISTS candidate_pool_candidate (
    pool_event_index INTEGER NOT NULL,
    candidate_id INTEGER NOT NULL,
    text TEXT NOT NULL,
    text_preview TEXT NOT NULL,
    output_token_count INTEGER,
    finish_reason TEXT NOT NULL,
    stop_reason TEXT NOT NULL,
    PRIMARY KEY(pool_event_index, candidate_id)
);

CREATE TABLE IF NOT EXISTS candidate_pool_candidate_token (
    pool_event_index INTEGER NOT NULL,
    candidate_id INTEGER NOT NULL,
    token_index INTEGER NOT NULL,
    token_id INTEGER,
    token_text TEXT NOT NULL,
    selected_logprob REAL,
    selected_probability REAL,
    PRIMARY KEY(pool_event_index, candidate_id, token_index)
);

CREATE INDEX IF NOT EXISTS idx_candidate_pool_candidate_token_event
ON candidate_pool_candidate_token(pool_event_index, candidate_id, token_index);

CREATE TABLE IF NOT EXISTS selector_event (
    event_index INTEGER PRIMARY KEY,
    doc_id INTEGER NOT NULL,
    doc_attempt INTEGER NOT NULL,
    task_name TEXT NOT NULL,
    model_id TEXT NOT NULL,
    selector_mode TEXT NOT NULL,
    timestamp_utc TEXT NOT NULL,
    selector_event_type TEXT NOT NULL,
    branch_point_id TEXT NOT NULL,
    node_id TEXT NOT NULL,
    active_selector_mode TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_selector_event_branch
ON selector_event(doc_id, doc_attempt, task_name, model_id, selector_mode, branch_point_id);

CREATE TABLE IF NOT EXISTS selector_candidate_flag (
    selector_event_index INTEGER NOT NULL,
    mode_name TEXT NOT NULL,
    candidate_id INTEGER NOT NULL,
    selected INTEGER NOT NULL,
    shortlisted INTEGER NOT NULL,
    PRIMARY KEY(selector_event_index, mode_name, candidate_id)
);

CREATE TABLE IF NOT EXISTS selector_candidate_cluster (
    selector_event_index INTEGER NOT NULL,
    mode_name TEXT NOT NULL,
    candidate_id INTEGER NOT NULL,
    cluster_name TEXT NOT NULL,
    PRIMARY KEY(selector_event_index, mode_name, candidate_id)
);

CREATE INDEX IF NOT EXISTS idx_selector_candidate_cluster_event
ON selector_candidate_cluster(selector_event_index, mode_name, cluster_name);

CREATE TABLE IF NOT EXISTS verbalized_sampling_decision (
    event_index INTEGER PRIMARY KEY,
    doc_id INTEGER NOT NULL,
    doc_attempt INTEGER NOT NULL,
    task_name TEXT NOT NULL,
    model_id TEXT NOT NULL,
    selector_mode TEXT NOT NULL,
    timestamp_utc TEXT NOT NULL,
    branch_point_id TEXT NOT NULL,
    candidate_pool_id TEXT NOT NULL,
    node_id TEXT NOT NULL,
    candidate_count INTEGER NOT NULL,
    branch_fanout INTEGER NOT NULL,
    sampled_option_numbers TEXT NOT NULL,
    parse_status TEXT NOT NULL,
    enumeration_exec_text TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_verbalized_sampling_decision_attempt
ON verbalized_sampling_decision(
    doc_id, doc_attempt, task_name, model_id, selector_mode, event_index
);

CREATE TABLE IF NOT EXISTS verbalized_sampling_candidate (
    decision_event_index INTEGER NOT NULL,
    option_number INTEGER NOT NULL,
    candidate_rank INTEGER NOT NULL,
    candidate_text TEXT NOT NULL,
    selected INTEGER NOT NULL,
    PRIMARY KEY(decision_event_index, option_number)
);

CREATE INDEX IF NOT EXISTS idx_verbalized_sampling_candidate_event
ON verbalized_sampling_candidate(decision_event_index, selected, option_number);

CREATE TABLE IF NOT EXISTS vllm_request (
    event_index INTEGER PRIMARY KEY,
    doc_id INTEGER NOT NULL,
    doc_attempt INTEGER NOT NULL,
    task_name TEXT NOT NULL,
    model_id TEXT NOT NULL,
    selector_mode TEXT NOT NULL,
    timestamp_utc TEXT NOT NULL,
    request_id TEXT NOT NULL,
    request_stream_id TEXT NOT NULL,
    request_kind TEXT NOT NULL,
    prev_request_id TEXT,
    current_input_token_count INTEGER,
    base_prefix_token_count INTEGER,
    delta_token_count INTEGER,
    assistant_prefix_char_count INTEGER,
    assistant_prefix_tail TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_vllm_request_id
ON vllm_request(doc_id, doc_attempt, task_name, model_id, selector_mode, request_id);

CREATE INDEX IF NOT EXISTS idx_vllm_request_stream
ON vllm_request(
    doc_id, doc_attempt, task_name, model_id, selector_mode,
    request_stream_id, event_index
);

CREATE TABLE IF NOT EXISTS vllm_response (
    event_index INTEGER PRIMARY KEY,
    doc_id INTEGER NOT NULL,
    doc_attempt INTEGER NOT NULL,
    task_name TEXT NOT NULL,
    model_id TEXT NOT NULL,
    selector_mode TEXT NOT NULL,
    timestamp_utc TEXT NOT NULL,
    request_id TEXT NOT NULL,
    request_stream_id TEXT NOT NULL,
    request_kind TEXT NOT NULL,
    status TEXT NOT NULL,
    latency_seconds REAL,
    error_message TEXT NOT NULL,
    choice_count INTEGER
);

CREATE INDEX IF NOT EXISTS idx_vllm_response_id
ON vllm_response(doc_id, doc_attempt, task_name, model_id, selector_mode, request_id);

CREATE INDEX IF NOT EXISTS idx_vllm_response_attempt_stream
ON vllm_response(
    doc_id, doc_attempt, task_name, model_id, selector_mode,
    request_stream_id, event_index
);

CREATE TABLE IF NOT EXISTS vllm_choice (
    response_event_index INTEGER NOT NULL,
    choice_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    text_preview TEXT NOT NULL,
    finish_reason TEXT NOT NULL,
    stop_reason TEXT NOT NULL,
    output_token_count INTEGER,
    PRIMARY KEY(response_event_index, choice_index)
);

CREATE TABLE IF NOT EXISTS vllm_choice_token (
    response_event_index INTEGER NOT NULL,
    choice_index INTEGER NOT NULL,
    token_index INTEGER NOT NULL,
    token_id INTEGER,
    token_text TEXT NOT NULL,
    selected_logprob REAL,
    selected_probability REAL,
    PRIMARY KEY(response_event_index, choice_index, token_index)
);

CREATE INDEX IF NOT EXISTS idx_vllm_choice_token_event
ON vllm_choice_token(response_event_index, choice_index, token_index);

CREATE TABLE IF NOT EXISTS generated_chunk (
    event_index INTEGER PRIMARY KEY,
    doc_id INTEGER NOT NULL,
    doc_attempt INTEGER NOT NULL,
    task_name TEXT NOT NULL,
    model_id TEXT NOT NULL,
    selector_mode TEXT NOT NULL,
    timestamp_utc TEXT NOT NULL,
    event_type TEXT NOT NULL,
    node_id TEXT NOT NULL,
    chunk_text TEXT NOT NULL,
    token_count INTEGER NOT NULL,
    generated_tokens_before_chunk INTEGER,
    generated_tokens_after_chunk INTEGER,
    chunk_was_normalized INTEGER NOT NULL,
    chunk_token_ids_source TEXT NOT NULL,
    source TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_generated_chunk_node
ON generated_chunk(
    doc_id, doc_attempt, task_name, model_id, selector_mode, node_id, event_index
);

CREATE INDEX IF NOT EXISTS idx_generated_chunk_attempt_event
ON generated_chunk(
    doc_id, doc_attempt, task_name, model_id, selector_mode, event_index
);

CREATE TABLE IF NOT EXISTS node_advantage (
    doc_id INTEGER NOT NULL,
    doc_attempt INTEGER NOT NULL,
    task_name TEXT NOT NULL,
    model_id TEXT NOT NULL,
    selector_mode TEXT NOT NULL,
    prompt_uid TEXT NOT NULL,
    branch_tree_id TEXT NOT NULL,
    parent_node_id TEXT NOT NULL,
    child_node_id TEXT NOT NULL,
    branch_depth INTEGER NOT NULL,
    token_start INTEGER NOT NULL,
    token_end INTEGER NOT NULL,
    mean_combined_advantage REAL NOT NULL,
    token_count INTEGER NOT NULL,
    leaf_count INTEGER NOT NULL,
    updated_at_event_index INTEGER NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY(
        doc_id, doc_attempt, task_name, model_id, selector_mode,
        parent_node_id, child_node_id
    )
);

CREATE INDEX IF NOT EXISTS idx_node_advantage_attempt_child
ON node_advantage(
    doc_id, doc_attempt, task_name, model_id, selector_mode, child_node_id
);

CREATE TABLE IF NOT EXISTS doc_progress_typed (
    run_id TEXT NOT NULL,
    doc_id INTEGER NOT NULL,
    doc_attempt INTEGER NOT NULL,
    task_name TEXT NOT NULL,
    model_id TEXT NOT NULL,
    selector_mode TEXT NOT NULL,
    rollout_mode TEXT NOT NULL,
    status TEXT NOT NULL,
    leaf_count INTEGER NOT NULL,
    passrate REAL NOT NULL,
    avg_token_length REAL NOT NULL,
    correct_count INTEGER NOT NULL,
    incorrect_count INTEGER NOT NULL,
    natural_count INTEGER NOT NULL,
    max_count INTEGER NOT NULL,
    repeating_count INTEGER NOT NULL,
    other_count INTEGER NOT NULL,
    unique_answer_count INTEGER NOT NULL,
    last_update_timestamp TEXT NOT NULL,
    PRIMARY KEY(run_id, doc_id, doc_attempt)
);
"""
