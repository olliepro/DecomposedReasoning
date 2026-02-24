"""Event-log replay helpers for partial doc resume in branching runs."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

from branching_eval.branch_decode_utils import (
    append_prompt_token_ids,
    length_tokens_exec,
)
from branching_eval.event_types import EventEnvelope
from branching_eval.runtime_types import PathState
from branching_eval.tree_types import (
    BranchTree,
    LeafRollout,
    TokenTrace,
    TreeEdge,
    TreeNode,
)


@dataclass(frozen=True)
class BranchResumeState:
    """Replayed branching state used to continue one partial doc attempt.

    Args:
        tree: Reconstructed branch tree.
        frontier: Active decode frontier nodes that still need expansion.
        pre_scored_leaves: Leaf rows already scored before interruption.
        request_stream_state: Last observed request-stream state per stream id.
        rollout_finished_seen: Whether this attempt already emitted rollout_finished.

    Returns:
        Resume payload used by runtime to continue from logs.

    Example:
        >>> state = BranchResumeState(
        ...     tree=BranchTree(
        ...         doc_id=0,
        ...         task_name='aime24',
        ...         model_id='fake',
        ...         selector_mode='random',
        ...         root_prompt='p',
        ...     ),
        ...     frontier=(),
        ...     pre_scored_leaves={},
        ...     request_stream_state={},
        ...     rollout_finished_seen=False,
        ... )
        >>> state.rollout_finished_seen
        False
    """

    tree: BranchTree
    frontier: tuple[PathState, ...]
    pre_scored_leaves: dict[str, LeafRollout]
    request_stream_state: dict[str, tuple[str, tuple[int, ...], tuple[int, ...]]]
    rollout_finished_seen: bool


@dataclass
class _NodeState:
    """Mutable replay state for one node while rebuilding resume frontier."""

    node_id: str
    parent_node_id: str | None
    branch_points_used: int
    assistant_prefix: str
    prompt_token_ids: tuple[int, ...] | None
    token_ids: list[int]
    token_traces: list[TokenTrace]
    terminal: bool


@dataclass(frozen=True)
class _DecodeStep:
    """One decode request/response step queued until matching decode_chunk arrives."""

    input_token_ids: tuple[int, ...]
    output_token_ids: tuple[int, ...]
    token_rows: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class _RequestState:
    """Per-stream request reconstruction state for delta-chain replay."""

    request_id: str
    input_token_ids: tuple[int, ...]
    output_token_ids: tuple[int, ...]


def replay_branching_resume_state(
    *,
    events: list[EventEnvelope],
    prompt_text: str,
    run_id: str,
    doc_id: int,
    doc_attempt: int,
    task_name: str,
    model_id: str,
    selector_mode: str,
) -> BranchResumeState:
    """Replay one doc attempt into resumable branching runtime state.

    Args:
        events: Full run events from `tree_events.jsonl`.
        prompt_text: Prompt text for this document.
        run_id: Stable run id.
        doc_id: Document id.
        doc_attempt: Attempt index.
        task_name: Task label.
        model_id: Model label.
        selector_mode: Selector label.

    Returns:
        Reconstructed tree/frontier/pre-scored-leaf payload for continuation.

    Example:
        >>> replay_branching_resume_state(
        ...     events=[],
        ...     prompt_text='p',
        ...     run_id='run_x',
        ...     doc_id=0,
        ...     doc_attempt=0,
        ...     task_name='aime24',
        ...     model_id='m',
        ...     selector_mode='random',
        ... ).frontier[0].node_id
        'node_root'
    """

    attempt_events = sorted(
        [
            event
            for event in events
            if event.doc_id == doc_id and event.doc_attempt == doc_attempt
        ],
        key=lambda row: row.event_index,
    )
    tree = _initial_tree(
        prompt_text=prompt_text,
        run_id=run_id,
        doc_id=doc_id,
        doc_attempt=doc_attempt,
        task_name=task_name,
        model_id=model_id,
        selector_mode=selector_mode,
    )
    node_state = {"node_root": _root_node_state()}
    leaves_by_id: dict[str, LeafRollout] = {}
    pre_scored_leaves: dict[str, LeafRollout] = {}
    decode_steps = _init_decode_steps()
    request_inputs: dict[str, tuple[str, tuple[int, ...]]] = {}
    request_stream_state: dict[str, _RequestState] = {}
    rollout_finished_seen = False
    for event in attempt_events:
        if event.event_type == "node_created":
            _apply_node_created(event=event, tree=tree, node_state=node_state)
            continue
        if event.event_type == "edge_selected":
            _apply_edge_selected(event=event, tree=tree, node_state=node_state)
            continue
        if event.event_type == "vllm_request":
            _apply_vllm_request(
                event=event,
                request_inputs=request_inputs,
                request_stream_state=request_stream_state,
            )
            continue
        if event.event_type == "vllm_response":
            _apply_vllm_response(
                event=event,
                request_inputs=request_inputs,
                request_stream_state=request_stream_state,
                decode_steps=decode_steps,
            )
            continue
        if event.event_type == "decode_chunk":
            _apply_decode_chunk(
                event=event,
                node_state=node_state,
                tree=tree,
                decode_steps=decode_steps,
            )
            continue
        if event.event_type == "leaf_completed":
            _apply_leaf_completed(
                event=event,
                node_state=node_state,
                tree=tree,
                leaves_by_id=leaves_by_id,
            )
            continue
        if event.event_type == "leaf_scored":
            _apply_leaf_scored(
                event=event,
                node_state=node_state,
                leaves_by_id=leaves_by_id,
                pre_scored_leaves=pre_scored_leaves,
            )
            continue
        if event.event_type == "rollout_finished":
            rollout_finished_seen = True
    tree.leaves = [leaves_by_id[leaf_id] for leaf_id in sorted(leaves_by_id)]
    frontier = _build_frontier(tree=tree, node_state=node_state)
    return BranchResumeState(
        tree=tree,
        frontier=frontier,
        pre_scored_leaves=pre_scored_leaves,
        request_stream_state=serialized_request_stream_state(
            request_stream_state=request_stream_state
        ),
        rollout_finished_seen=rollout_finished_seen,
    )


def _initial_tree(
    *,
    prompt_text: str,
    run_id: str,
    doc_id: int,
    doc_attempt: int,
    task_name: str,
    model_id: str,
    selector_mode: str,
) -> BranchTree:
    """Return replay seed tree with root node present."""

    tree = BranchTree(
        doc_id=doc_id,
        doc_attempt=doc_attempt,
        run_id=run_id,
        task_name=task_name,
        model_id=model_id,
        selector_mode=selector_mode,
        root_prompt=prompt_text,
    )
    tree.add_node(
        node=TreeNode(
            node_id="node_root",
            parent_node_id=None,
            prompt_text=prompt_text,
            assistant_prefix="",
            prompt_token_ids=None,
            branch_points_used=0,
        )
    )
    return tree


def _root_node_state() -> _NodeState:
    """Return mutable runtime state for root node."""

    return _NodeState(
        node_id="node_root",
        parent_node_id=None,
        branch_points_used=0,
        assistant_prefix="",
        prompt_token_ids=None,
        token_ids=[],
        token_traces=[],
        terminal=False,
    )


def _init_decode_steps() -> dict[str, deque[_DecodeStep]]:
    """Return per-node decode queue container."""

    return {}


def _ensure_node_state(
    *,
    node_id: str,
    parent_node_id: str | None,
    branch_points_used: int,
    node_state: dict[str, _NodeState],
) -> _NodeState:
    """Return mutable node state, creating when missing."""

    existing = node_state.get(node_id)
    if existing is not None:
        return existing
    created = _NodeState(
        node_id=node_id,
        parent_node_id=parent_node_id,
        branch_points_used=branch_points_used,
        assistant_prefix="",
        prompt_token_ids=None,
        token_ids=[],
        token_traces=[],
        terminal=False,
    )
    node_state[node_id] = created
    return created


def _apply_node_created(
    *,
    event: EventEnvelope,
    tree: BranchTree,
    node_state: dict[str, _NodeState],
) -> None:
    """Apply node_created event to replayed tree and node-state maps."""

    payload = event.payload
    node_id = str(payload.get("node_id", ""))
    if not node_id:
        return
    parent_raw = payload.get("parent_node_id")
    parent_node_id = str(parent_raw) if parent_raw is not None else None
    branch_points_used = int(payload.get("branch_points_used", 0))
    state = _ensure_node_state(
        node_id=node_id,
        parent_node_id=parent_node_id,
        branch_points_used=branch_points_used,
        node_state=node_state,
    )
    state.parent_node_id = parent_node_id
    state.branch_points_used = branch_points_used
    tree.add_node(
        node=TreeNode(
            node_id=node_id,
            parent_node_id=parent_node_id,
            prompt_text=tree.root_prompt,
            assistant_prefix=state.assistant_prefix,
            prompt_token_ids=state.prompt_token_ids,
            branch_points_used=state.branch_points_used,
        )
    )


def _apply_edge_selected(
    *,
    event: EventEnvelope,
    tree: BranchTree,
    node_state: dict[str, _NodeState],
) -> None:
    """Apply edge_selected event and reconstruct child-node prefix/token state."""

    payload = event.payload
    parent_id = str(payload.get("parent_node_id", ""))
    child_id = str(payload.get("child_node_id", ""))
    if not parent_id or not child_id:
        return
    parent_state = _ensure_node_state(
        node_id=parent_id,
        parent_node_id=None,
        branch_points_used=0,
        node_state=node_state,
    )
    candidate_text = str(payload.get("candidate_text_normalized", ""))
    candidate_ids = _ints(payload.get("candidate_token_ids_normalized"))
    child_existing = node_state.get(child_id)
    child_branch_points = (
        child_existing.branch_points_used
        if child_existing is not None
        else parent_state.branch_points_used + 1
    )
    child_state = _ensure_node_state(
        node_id=child_id,
        parent_node_id=parent_id,
        branch_points_used=child_branch_points,
        node_state=node_state,
    )
    child_state.parent_node_id = parent_id
    child_state.branch_points_used = child_branch_points
    child_state.assistant_prefix = parent_state.assistant_prefix + candidate_text
    child_state.prompt_token_ids = append_prompt_token_ids(
        prompt_token_ids=parent_state.prompt_token_ids,
        continuation_token_ids=tuple(candidate_ids),
    )
    child_state.token_ids = list(parent_state.token_ids) + candidate_ids
    child_state.token_traces = list(parent_state.token_traces)
    _update_tree_node(tree=tree, state=child_state)
    _append_tree_edge(tree=tree, payload=payload)


def _apply_vllm_request(
    *,
    event: EventEnvelope,
    request_inputs: dict[str, tuple[str, tuple[int, ...]]],
    request_stream_state: dict[str, _RequestState],
) -> None:
    """Reconstruct current request input token ids from stream delta-chain."""

    payload = event.payload
    request_id = str(payload.get("request_id", ""))
    stream_id = str(payload.get("request_stream_id", ""))
    if not request_id or not stream_id:
        return
    delta_ids = tuple(_ints(payload.get("delta_input_token_ids")))
    previous = request_stream_state.get(stream_id)
    if previous is None:
        current_input_ids = delta_ids
    else:
        current_input_ids = (
            previous.input_token_ids + previous.output_token_ids + delta_ids
        )
    request_inputs[request_id] = (stream_id, current_input_ids)


def _apply_vllm_response(
    *,
    event: EventEnvelope,
    request_inputs: dict[str, tuple[str, tuple[int, ...]]],
    request_stream_state: dict[str, _RequestState],
    decode_steps: dict[str, deque[_DecodeStep]],
) -> None:
    """Attach response metadata to request stream replay and decode-step queues."""

    payload = event.payload
    request_id = str(payload.get("request_id", ""))
    status = str(payload.get("status", ""))
    request_context = request_inputs.get(request_id)
    if request_context is None or status != "ok":
        return
    stream_id, input_ids = request_context
    output_ids, token_rows = _response_choice_tokens(payload=payload)
    request_stream_state[stream_id] = _RequestState(
        request_id=request_id,
        input_token_ids=input_ids,
        output_token_ids=output_ids,
    )
    node_id = _decode_node_id(stream_id=stream_id)
    if node_id is None:
        return
    decode_steps.setdefault(node_id, deque()).append(
        _DecodeStep(
            input_token_ids=input_ids,
            output_token_ids=output_ids,
            token_rows=token_rows,
        )
    )


def _apply_decode_chunk(
    *,
    event: EventEnvelope,
    node_state: dict[str, _NodeState],
    tree: BranchTree,
    decode_steps: dict[str, deque[_DecodeStep]],
) -> None:
    """Apply decode_chunk state delta and consume queued response token stats."""

    payload = event.payload
    node_id = str(payload.get("node_id", ""))
    if not node_id:
        return
    state = _ensure_node_state(
        node_id=node_id,
        parent_node_id=None,
        branch_points_used=0,
        node_state=node_state,
    )
    state.assistant_prefix += str(payload.get("chunk_text", ""))
    consumed_tokens = max(
        0,
        int(payload.get("generated_tokens_after_chunk", 0))
        - int(payload.get("generated_tokens_before_chunk", 0)),
    )
    step = _pop_decode_step(decode_steps=decode_steps, node_id=node_id)
    if step is not None:
        _apply_decode_step_tokens(
            state=state,
            payload=payload,
            step=step,
            consumed_tokens=consumed_tokens,
        )
    else:
        fallback_ids = _ints(payload.get("chunk_token_ids"))
        state.token_ids.extend(fallback_ids)
        if state.prompt_token_ids is not None:
            state.prompt_token_ids = state.prompt_token_ids + tuple(fallback_ids)
    _update_tree_node(tree=tree, state=state)


def _pop_decode_step(
    *,
    decode_steps: dict[str, deque[_DecodeStep]],
    node_id: str,
) -> _DecodeStep | None:
    """Pop one queued decode step for a node, if available."""

    queue = decode_steps.get(node_id)
    if queue is None or not queue:
        return None
    return queue.popleft()


def _apply_decode_step_tokens(
    *,
    state: _NodeState,
    payload: dict[str, Any],
    step: _DecodeStep,
    consumed_tokens: int,
) -> None:
    """Apply one decoded token step to node token/prompt/runtime fields."""

    consumed_ids = list(step.output_token_ids[:consumed_tokens])
    state.token_ids.extend(consumed_ids)
    for token_index, token_row in enumerate(step.token_rows[:consumed_tokens]):
        state.token_traces.append(
            _token_trace_from_row(token_index=token_index, row=token_row)
        )
    source = str(payload.get("chunk_token_ids_source", ""))
    if source == "prompt_token_delta":
        prompt_delta = tuple(_ints(payload.get("chunk_token_ids")))
        state.prompt_token_ids = step.input_token_ids + prompt_delta
        return
    state.prompt_token_ids = step.input_token_ids + tuple(consumed_ids)


def _apply_leaf_completed(
    *,
    event: EventEnvelope,
    node_state: dict[str, _NodeState],
    tree: BranchTree,
    leaves_by_id: dict[str, LeafRollout],
) -> None:
    """Apply leaf_completed event and capture resumable unscored leaf payload."""

    payload = event.payload
    leaf_id = str(payload.get("leaf_id", ""))
    node_id = str(payload.get("node_id", ""))
    if not leaf_id or not node_id:
        return
    state = _ensure_node_state(
        node_id=node_id,
        parent_node_id=None,
        branch_points_used=0,
        node_state=node_state,
    )
    state.terminal = True
    leaves_by_id[leaf_id] = LeafRollout(
        leaf_id=leaf_id,
        node_id=node_id,
        text=state.assistant_prefix,
        token_ids=tuple(state.token_ids),
        tokens=tuple(state.token_traces),
        verification=0,
        length_tokens_total=int(
            payload.get("length_tokens_total", len(state.token_ids))
        ),
        length_tokens_exec=(
            int(payload["length_tokens_exec"])
            if payload.get("length_tokens_exec") is not None
            else length_tokens_exec(text=state.assistant_prefix)
        ),
        stop_reason=str(payload.get("stop_reason", "")),
        task_metrics={},
    )
    _update_tree_node(tree=tree, state=state)


def _apply_leaf_scored(
    *,
    event: EventEnvelope,
    node_state: dict[str, _NodeState],
    leaves_by_id: dict[str, LeafRollout],
    pre_scored_leaves: dict[str, LeafRollout],
) -> None:
    """Apply leaf_scored event and track already-scored leaves for resume."""

    payload = event.payload
    leaf_id = str(payload.get("leaf_id", ""))
    node_id = str(payload.get("node_id", ""))
    if not leaf_id or not node_id:
        return
    task_metrics = payload.get("task_metrics", {})
    metrics_mapping = task_metrics if isinstance(task_metrics, dict) else {}
    token_ids = tuple(_ints(payload.get("token_ids")))
    token_rows = payload.get("tokens", [])
    token_traces = _token_traces_from_payload(token_rows=token_rows)
    scored = LeafRollout(
        leaf_id=leaf_id,
        node_id=node_id,
        text=str(payload.get("text", "")),
        token_ids=token_ids,
        tokens=token_traces,
        verification=int(payload.get("verification", 0)),
        length_tokens_total=int(payload.get("length_tokens_total", len(token_ids))),
        length_tokens_exec=(
            int(payload["length_tokens_exec"])
            if payload.get("length_tokens_exec") is not None
            else length_tokens_exec(text=str(payload.get("text", "")))
        ),
        stop_reason=str(payload.get("stop_reason", "")),
        task_metrics=metrics_mapping,
    )
    leaves_by_id[leaf_id] = scored
    pre_scored_leaves[leaf_id] = scored
    state = _ensure_node_state(
        node_id=node_id,
        parent_node_id=None,
        branch_points_used=0,
        node_state=node_state,
    )
    state.terminal = True


def _build_frontier(
    *,
    tree: BranchTree,
    node_state: dict[str, _NodeState],
) -> tuple[PathState, ...]:
    """Build resumable frontier from replayed node states and selected edges."""

    expanded = {edge.parent_node_id for edge in tree.edges}
    rows: list[PathState] = []
    for node_id in sorted(node_state):
        state = node_state[node_id]
        if state.terminal or node_id in expanded:
            continue
        rows.append(
            PathState(
                node_id=node_id,
                assistant_prefix=state.assistant_prefix,
                prompt_token_ids=state.prompt_token_ids,
                token_ids=tuple(state.token_ids),
                token_traces=tuple(state.token_traces),
                branch_points_used=state.branch_points_used,
            )
        )
    if rows:
        return tuple(rows)
    if tree.leaves:
        return ()
    root_state = node_state.get("node_root")
    assert root_state is not None, "resume replay must preserve root state"
    if root_state.terminal:
        return ()
    return (
        PathState(
            node_id="node_root",
            assistant_prefix=root_state.assistant_prefix,
            prompt_token_ids=root_state.prompt_token_ids,
            token_ids=tuple(root_state.token_ids),
            token_traces=tuple(root_state.token_traces),
            branch_points_used=root_state.branch_points_used,
        ),
    )


def _response_choice_tokens(
    *,
    payload: dict[str, Any],
) -> tuple[tuple[int, ...], tuple[dict[str, Any], ...]]:
    """Return first-choice output token ids and token rows from response payload."""

    choices = payload.get("choices", [])
    if not isinstance(choices, list) or not choices:
        return (), ()
    first = choices[0]
    if not isinstance(first, dict):
        return (), ()
    token_ids = tuple(_ints(first.get("token_ids")))
    raw_rows = first.get("tokens", [])
    if not isinstance(raw_rows, list):
        return token_ids, ()
    row_dicts = tuple(row for row in raw_rows if isinstance(row, dict))
    return token_ids, row_dicts


def _token_trace_from_row(*, token_index: int, row: dict[str, Any]) -> TokenTrace:
    """Parse one token trace row from serialized event payload mapping."""

    token_id = row.get("token_id")
    return TokenTrace(
        token_index=token_index,
        token_id=int(token_id) if token_id is not None else None,
        token_text=str(row.get("token_text", "")),
        logprob=float(row.get("selected_logprob", 0.0)),
        probability=float(row.get("selected_probability", 0.0)),
        entropy=float(row.get("selected_entropy", 0.0)),
    )


def _token_traces_from_payload(*, token_rows: Any) -> tuple[TokenTrace, ...]:
    """Parse token-trace tuple from serialized leaf payload mapping."""

    if not isinstance(token_rows, list):
        return ()
    parsed = []
    for index, row in enumerate(token_rows):
        if not isinstance(row, dict):
            continue
        parsed.append(_token_trace_from_row(token_index=index, row=row))
    return tuple(parsed)


def _update_tree_node(*, tree: BranchTree, state: _NodeState) -> None:
    """Upsert tree-node snapshot with latest replayed node state fields."""

    existing = tree.nodes.get(state.node_id)
    prompt_text = tree.root_prompt if existing is None else existing.prompt_text
    tree.add_node(
        node=TreeNode(
            node_id=state.node_id,
            parent_node_id=state.parent_node_id,
            prompt_text=prompt_text,
            assistant_prefix=state.assistant_prefix,
            prompt_token_ids=state.prompt_token_ids,
            branch_points_used=state.branch_points_used,
        )
    )


def _append_tree_edge(*, tree: BranchTree, payload: dict[str, Any]) -> None:
    """Append replayed edge row when not already present in tree state."""

    edge_id = str(payload.get("edge_id", ""))
    parent_id = str(payload.get("parent_node_id", ""))
    child_id = str(payload.get("child_node_id", ""))
    if not edge_id or not parent_id or not child_id:
        return
    if any(existing.edge_id == edge_id for existing in tree.edges):
        return
    tree.edges.append(
        TreeEdge(
            edge_id=edge_id,
            parent_node_id=parent_id,
            child_node_id=child_id,
            candidate_pool_id=str(payload.get("candidate_pool_id", "")),
            candidate_id=int(payload.get("candidate_id", -1)),
            selector_mode=str(payload.get("selector_mode", "")),
        )
    )


def _decode_node_id(*, stream_id: str) -> str | None:
    """Decode node id from decode-stream id, else return None."""

    if not stream_id.startswith("decode:"):
        return None
    node_id = stream_id.split(":", maxsplit=1)[1]
    return node_id if node_id else None


def serialized_request_stream_state(
    *, request_stream_state: dict[str, _RequestState]
) -> dict[str, tuple[str, tuple[int, ...], tuple[int, ...]]]:
    """Return request-stream state mapping ready for executor restoration."""

    return {
        stream_id: (state.request_id, state.input_token_ids, state.output_token_ids)
        for stream_id, state in request_stream_state.items()
    }


def _ints(value: Any) -> list[int]:
    """Return integer list parsed from unknown JSON payload value."""

    if not isinstance(value, list):
        return []
    return [int(item) for item in value]
