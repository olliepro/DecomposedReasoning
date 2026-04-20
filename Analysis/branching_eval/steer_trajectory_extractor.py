"""Extract one successful steer trajectory from canonical tree-event logs."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

STEER_REQUEST_KIND = "steer_single_candidate"
TREE_EVENTS_FILENAME = "tree_events.jsonl"
JsonDict = dict[str, Any]


@dataclass(frozen=True)
class NodeKey:
    """Stable identity for one node within a document attempt.

    Args:
        doc_id: Document id attached to the event row.
        doc_attempt: Attempt index attached to the event row.
        node_id: Node identifier within the rollout tree.

    Returns:
        Dataclass used to connect node ancestry and steer rows.
    """

    doc_id: int
    doc_attempt: int
    node_id: str

    def to_json(self) -> JsonDict:
        """Return JSON-friendly mapping for this node key."""

        return {
            "doc_id": self.doc_id,
            "doc_attempt": self.doc_attempt,
            "node_id": self.node_id,
        }


@dataclass(frozen=True)
class SuccessfulLeaf:
    """One successful terminal leaf used to select a trajectory.

    Args:
        leaf_id: Leaf identifier from the success event.
        terminal_node: Terminal node for the successful leaf.
        source_event_type: Event type that recorded the success.
        event_index: Global event index of the success row.

    Returns:
        Dataclass describing the chosen successful leaf.
    """

    leaf_id: str
    terminal_node: NodeKey
    source_event_type: str
    event_index: int

    def sort_key(self) -> tuple[int, int]:
        """Return deterministic ordering for candidate selection."""

        return (
            success_source_rank(event_type=self.source_event_type),
            self.event_index,
        )

    def to_json(self) -> JsonDict:
        """Return JSON-friendly mapping for this successful leaf."""

        payload = asdict(self)
        payload["terminal_node"] = self.terminal_node.to_json()
        return payload


@dataclass(frozen=True)
class SteerResponseSummary:
    """Compact view of one response-side steer generation.

    Args:
        event_index: Event index of the raw response row.
        timestamp_utc: UTC timestamp of the raw response row.
        node: Node tied to this steer generation.
        request_id: vLLM request id.
        steer_text: Generated steer text from the first choice.

    Returns:
        Dataclass used for readable summaries.
    """

    event_index: int
    timestamp_utc: str
    node: NodeKey
    request_id: str
    steer_text: str

    def to_json(self) -> JsonDict:
        """Return JSON-friendly mapping for this steer response."""

        return {
            "event_index": self.event_index,
            "timestamp_utc": self.timestamp_utc,
            "node": self.node.to_json(),
            "request_id": self.request_id,
            "steer_text": self.steer_text,
        }


@dataclass(frozen=True)
class TrajectoryExtraction:
    """Extracted steer rows for one successful trajectory.

    Args:
        tree_events_path: Input event file path.
        selected_leaf: Successful leaf that anchors the trajectory.
        path_nodes: Node ancestry from root to terminal success node.
        steer_rows: Raw request/response rows on the successful path.
        success_count: Total successful leaves found in the file.
        response_summaries: Compact steer response summaries on the path.

    Returns:
        Dataclass describing the extracted successful trajectory.

    Example:
        >>> extraction = TrajectoryExtraction(
        ...     tree_events_path=Path("/tmp/tree_events.jsonl"),
        ...     selected_leaf=SuccessfulLeaf(
        ...         leaf_id="leaf_1",
        ...         terminal_node=NodeKey(doc_id=0, doc_attempt=0, node_id="node_a"),
        ...         source_event_type="leaf_scored",
        ...         event_index=7,
        ...     ),
        ...     path_nodes=(NodeKey(doc_id=0, doc_attempt=0, node_id="node_a"),),
        ...     steer_rows=(),
        ...     success_count=1,
        ...     response_summaries=(),
        ... )
        >>> extraction.selected_leaf.leaf_id
        'leaf_1'
    """

    tree_events_path: Path
    selected_leaf: SuccessfulLeaf
    path_nodes: tuple[NodeKey, ...]
    steer_rows: tuple[JsonDict, ...]
    success_count: int
    response_summaries: tuple[SteerResponseSummary, ...]

    def summary_payload(self) -> JsonDict:
        """Return JSON-serializable summary metadata for one extraction."""

        return {
            "tree_events_path": str(self.tree_events_path),
            "selected_leaf": self.selected_leaf.to_json(),
            "success_count": self.success_count,
            "path_nodes": [node.to_json() for node in self.path_nodes],
            "steer_row_count": len(self.steer_rows),
            "steer_response_count": len(self.response_summaries),
            "response_summaries": [
                summary.to_json() for summary in self.response_summaries
            ],
        }


@dataclass
class FileScanState:
    """Mutable scan state while walking one tree-event file.

    Args:
        parent_by_node: Parent linkage for every created node.
        steer_rows_by_node: Raw steer request/response rows grouped by node.
        selected_leaf: Earliest successful leaf seen so far.
        success_count: Number of successful leaves seen so far.

    Returns:
        Mutable scan state for extraction.
    """

    parent_by_node: dict[NodeKey, str | None] = field(default_factory=dict)
    steer_rows_by_node: dict[NodeKey, list[JsonDict]] = field(default_factory=dict)
    selected_leaf: SuccessfulLeaf | None = None
    success_count: int = 0
    successful_leaf_keys: set[tuple[int, int, str]] = field(default_factory=set)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for successful steer extraction."""

    parser = argparse.ArgumentParser(
        description=(
            "Extract raw steer_single_candidate rows for one successful "
            "trajectory from tree_events.jsonl files."
        )
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        required=True,
        help="Tree-events file or directory containing tree_events.jsonl files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where extracted artifacts will be written.",
    )
    return parser.parse_args()


def iter_tree_event_paths(*, input_path: Path) -> tuple[Path, ...]:
    """Return ordered tree-event files beneath the given input path.

    Args:
        input_path: Either one tree-events file or a directory tree.

    Returns:
        Sorted tree-event paths to process.

    Example:
        >>> iter_tree_event_paths(input_path=Path('/tmp/missing'))
        ()
    """

    if input_path.is_file():
        assert (
            input_path.name == TREE_EVENTS_FILENAME
        ), "input file must be tree_events.jsonl"
        return (input_path,)
    assert input_path.is_dir(), f"input path does not exist: {input_path}"
    return tuple(sorted(input_path.rglob(TREE_EVENTS_FILENAME)))


def success_source_rank(*, event_type: str) -> int:
    """Return priority rank for success-bearing event types."""

    if event_type == "leaf_scored":
        return 0
    if event_type == "leaf_completed":
        return 1
    raise AssertionError(f"unsupported success event type: {event_type}")


def parse_json_row(*, line: str) -> JsonDict:
    """Parse one canonical JSONL row into a mapping."""

    row = json.loads(line)
    assert isinstance(row, dict), "tree-event rows must decode to mappings"
    return row


def iter_json_rows(*, path: Path) -> Iterable[JsonDict]:
    """Yield parsed rows from one tree-events file.

    Args:
        path: Canonical tree-events file path.

    Returns:
        Iterator of parsed JSON row mappings.
    """

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line_text = line.strip()
            if not line_text:
                continue
            yield parse_json_row(line=line_text)


def node_key_from_row(*, row: JsonDict, node_id: str) -> NodeKey:
    """Build a node key from one event row and node id."""

    doc_id = row.get("doc_id")
    doc_attempt = row.get("doc_attempt")
    assert isinstance(doc_id, int), "doc_id must be an int for node-scoped rows"
    assert isinstance(
        doc_attempt, int
    ), "doc_attempt must be an int for node-scoped rows"
    return NodeKey(doc_id=doc_id, doc_attempt=doc_attempt, node_id=node_id)


def request_node_id(*, payload: JsonDict) -> str | None:
    """Return node id attached to one steer request or response payload."""

    stream_id = payload.get("request_stream_id")
    if not isinstance(stream_id, str) or not stream_id.startswith("decode:"):
        return None
    node_id = stream_id.split(":", maxsplit=1)[1]
    return node_id or None


def maybe_select_success(
    *, current_leaf: SuccessfulLeaf | None, row: JsonDict
) -> SuccessfulLeaf | None:
    """Return updated selected success for the given row."""

    event_type = row.get("event_type")
    if event_type not in {"leaf_completed", "leaf_scored"}:
        return current_leaf
    payload = row.get("payload")
    if not isinstance(payload, dict) or payload.get("verification") != 1:
        return current_leaf
    leaf_id = payload.get("leaf_id")
    node_id = payload.get("node_id")
    if not isinstance(leaf_id, str) or not isinstance(node_id, str):
        return current_leaf
    candidate = SuccessfulLeaf(
        leaf_id=leaf_id,
        terminal_node=node_key_from_row(row=row, node_id=node_id),
        source_event_type=event_type,
        event_index=int(row["event_index"]),
    )
    if current_leaf is None or candidate.sort_key() < current_leaf.sort_key():
        return candidate
    return current_leaf


def apply_scan_row(*, state: FileScanState, row: JsonDict) -> None:
    """Update file scan state from one raw tree-event row."""

    event_type = row.get("event_type")
    payload = row.get("payload", {})
    assert isinstance(payload, dict), "event payload must be a mapping"
    state.selected_leaf = maybe_select_success(
        current_leaf=state.selected_leaf, row=row
    )
    if (
        event_type in {"leaf_completed", "leaf_scored"}
        and payload.get("verification") == 1
    ):
        record_success_leaf(state=state, row=row, payload=payload)
    if event_type == "node_created":
        node_id = payload.get("node_id")
        if isinstance(node_id, str) and node_id:
            state.parent_by_node[node_key_from_row(row=row, node_id=node_id)] = (
                str(payload["parent_node_id"])
                if payload.get("parent_node_id") is not None
                else None
            )
        return
    if event_type not in {"vllm_request", "vllm_response"}:
        return
    if payload.get("request_kind") != STEER_REQUEST_KIND:
        return
    node_id = request_node_id(payload=payload)
    if node_id is None:
        return
    node_key = node_key_from_row(row=row, node_id=node_id)
    state.steer_rows_by_node.setdefault(node_key, []).append(row)


def record_success_leaf(
    *, state: FileScanState, row: JsonDict, payload: JsonDict
) -> None:
    """Record one unique successful leaf from a success-bearing event row."""

    leaf_id = payload.get("leaf_id")
    doc_id = row.get("doc_id")
    doc_attempt = row.get("doc_attempt")
    if not isinstance(leaf_id, str):
        return
    if not isinstance(doc_id, int) or not isinstance(doc_attempt, int):
        return
    leaf_key = (doc_id, doc_attempt, leaf_id)
    if leaf_key in state.successful_leaf_keys:
        return
    state.successful_leaf_keys.add(leaf_key)
    state.success_count += 1


def scan_tree_events_file(*, path: Path) -> FileScanState:
    """Scan one tree-events file for successful leaves and steer rows.

    Args:
        path: Canonical tree-events file path.

    Returns:
        Scan state containing ancestry links and steer rows.
    """

    state = FileScanState()
    for row in iter_json_rows(path=path):
        apply_scan_row(state=state, row=row)
    return state


def build_success_path(
    *, parent_by_node: dict[NodeKey, str | None], selected_leaf: SuccessfulLeaf
) -> tuple[NodeKey, ...]:
    """Reconstruct the root-to-leaf node path for a selected success.

    Args:
        parent_by_node: Parent linkage keyed by node.
        selected_leaf: Successful leaf anchoring the trajectory.

    Returns:
        Ordered node path from root to terminal node.

    Example:
        >>> root = NodeKey(doc_id=0, doc_attempt=0, node_id='root')
        >>> child = NodeKey(doc_id=0, doc_attempt=0, node_id='child')
        >>> success = SuccessfulLeaf(
        ...     leaf_id='leaf',
        ...     terminal_node=child,
        ...     source_event_type='leaf_scored',
        ...     event_index=4,
        ... )
        >>> build_success_path(
        ...     parent_by_node={root: None, child: 'root'},
        ...     selected_leaf=success,
        ... )
        (NodeKey(doc_id=0, doc_attempt=0, node_id='root'), NodeKey(doc_id=0, doc_attempt=0, node_id='child'))
    """

    path_nodes: list[NodeKey] = []
    visited: set[NodeKey] = set()
    current_node = selected_leaf.terminal_node
    while True:
        assert current_node not in visited, "node ancestry loop detected"
        assert (
            current_node in parent_by_node
        ), f"missing node_created row for {current_node.node_id}"
        visited.add(current_node)
        path_nodes.append(current_node)
        parent_node_id = parent_by_node[current_node]
        if parent_node_id is None:
            break
        current_node = NodeKey(
            doc_id=current_node.doc_id,
            doc_attempt=current_node.doc_attempt,
            node_id=parent_node_id,
        )
    path_nodes.reverse()
    return tuple(path_nodes)


def gather_steer_rows(
    *,
    path_nodes: tuple[NodeKey, ...],
    steer_rows_by_node: dict[NodeKey, list[JsonDict]],
) -> tuple[JsonDict, ...]:
    """Collect raw steer request/response rows for the selected path."""

    selected_rows: list[JsonDict] = []
    for node in path_nodes:
        selected_rows.extend(steer_rows_by_node.get(node, ()))
    selected_rows.sort(key=lambda row: int(row["event_index"]))
    return tuple(selected_rows)


def summarize_response_rows(
    *, steer_rows: tuple[JsonDict, ...]
) -> tuple[SteerResponseSummary, ...]:
    """Return compact summaries for response-side steer rows."""

    summaries: list[SteerResponseSummary] = []
    for row in steer_rows:
        if row.get("event_type") != "vllm_response":
            continue
        payload = row.get("payload", {})
        assert isinstance(payload, dict), "response payload must be a mapping"
        choices = payload.get("choices", [])
        first_choice = choices[0] if isinstance(choices, list) and choices else {}
        if not isinstance(first_choice, dict):
            first_choice = {}
        node_id = request_node_id(payload=payload)
        assert node_id is not None, "steer response rows must carry a decode node id"
        summaries.append(
            SteerResponseSummary(
                event_index=int(row["event_index"]),
                timestamp_utc=str(row["timestamp_utc"]),
                node=node_key_from_row(row=row, node_id=node_id),
                request_id=str(payload.get("request_id", "")),
                steer_text=str(first_choice.get("text", "")),
            )
        )
    return tuple(summaries)


def extract_successful_trajectory(*, path: Path) -> TrajectoryExtraction | None:
    """Extract one successful steer trajectory from one tree-events file.

    Args:
        path: Canonical tree-events file path.

    Returns:
        Extracted trajectory or `None` when no successful leaf exists.

    Example:
        >>> extract_successful_trajectory(path=Path('/tmp/missing-tree-events.jsonl'))
        Traceback (most recent call last):
        ...
        FileNotFoundError: ...
    """

    scan_state = scan_tree_events_file(path=path)
    if scan_state.selected_leaf is None:
        return None
    path_nodes = build_success_path(
        parent_by_node=scan_state.parent_by_node,
        selected_leaf=scan_state.selected_leaf,
    )
    steer_rows = gather_steer_rows(
        path_nodes=path_nodes,
        steer_rows_by_node=scan_state.steer_rows_by_node,
    )
    return TrajectoryExtraction(
        tree_events_path=path,
        selected_leaf=scan_state.selected_leaf,
        path_nodes=path_nodes,
        steer_rows=steer_rows,
        success_count=scan_state.success_count,
        response_summaries=summarize_response_rows(steer_rows=steer_rows),
    )


def ensure_directory(*, path: Path) -> None:
    """Create the target directory if it does not already exist."""

    path.mkdir(parents=True, exist_ok=True)


def write_jsonl(*, path: Path, rows: tuple[JsonDict, ...]) -> None:
    """Write JSONL rows to disk with stable UTF-8 formatting."""

    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def write_json(*, path: Path, payload: JsonDict) -> None:
    """Write one JSON document to disk with indentation."""

    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def output_subdir(*, input_root: Path, tree_events_path: Path) -> Path:
    """Return mirrored output subdirectory for one tree-events file."""

    if input_root.is_file():
        return Path(".")
    return tree_events_path.parent.relative_to(input_root)


def write_extraction_artifacts(
    *,
    extraction: TrajectoryExtraction,
    input_root: Path,
    output_root: Path,
) -> JsonDict:
    """Write extraction JSONL and summary artifacts for one file."""

    relative_dir = output_subdir(
        input_root=input_root, tree_events_path=extraction.tree_events_path
    )
    target_dir = output_root / relative_dir
    ensure_directory(path=target_dir)
    jsonl_path = target_dir / "successful_steer_single_candidate.jsonl"
    summary_path = target_dir / "successful_steer_single_candidate.summary.json"
    write_jsonl(path=jsonl_path, rows=extraction.steer_rows)
    write_json(path=summary_path, payload=extraction.summary_payload())
    return {
        "tree_events_path": str(extraction.tree_events_path),
        "jsonl_path": str(jsonl_path),
        "summary_path": str(summary_path),
        "steer_row_count": len(extraction.steer_rows),
        "success_count": extraction.success_count,
        "selected_leaf": extraction.selected_leaf.to_json(),
    }


def default_output_dir(*, input_path: Path) -> Path:
    """Return default output directory for extraction artifacts."""

    base_dir = input_path if input_path.is_dir() else input_path.parent
    return base_dir / "successful_steer_trajectory_extracts"


def run_cli(*, input_path: Path, output_dir: Path) -> JsonDict:
    """Extract successful steer trajectories and write artifacts.

    Args:
        input_path: Tree-events file or directory tree.
        output_dir: Directory where extraction artifacts will be written.

    Returns:
        Index payload describing all processed files.
    """

    tree_event_paths = iter_tree_event_paths(input_path=input_path)
    ensure_directory(path=output_dir)
    written: list[JsonDict] = []
    skipped: list[JsonDict] = []
    for tree_events_path in tree_event_paths:
        extraction = extract_successful_trajectory(path=tree_events_path)
        if extraction is None:
            skipped.append(
                {
                    "tree_events_path": str(tree_events_path),
                    "reason": "no successful leaf",
                }
            )
            continue
        record = write_extraction_artifacts(
            extraction=extraction,
            input_root=input_path,
            output_root=output_dir,
        )
        if not extraction.steer_rows:
            record["note"] = (
                "selected success path contained no steer_single_candidate rows"
            )
        written.append(record)
    index_payload = {
        "input_path": str(input_path),
        "written": written,
        "skipped": skipped,
    }
    write_json(path=output_dir / "index.json", payload=index_payload)
    return index_payload


def main() -> None:
    """Run the successful steer trajectory extraction CLI."""

    args = parse_args()
    output_dir = args.output_dir or default_output_dir(input_path=args.input_path)
    run_cli(input_path=args.input_path, output_dir=output_dir)


if __name__ == "__main__":
    main()
