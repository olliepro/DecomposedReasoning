"""Branching rollout execution with trigger policies and selector strategies."""

from __future__ import annotations
import asyncio
import hashlib
import random
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Awaitable, Callable, cast
from candidate_clustering import ClusteringCache
from branching_eval.artifact_store import ArtifactStore
from branching_eval.branch_decode_utils import (
    assert_no_text_after_first_steer_close,
    append_prompt_token_ids,
    candidate_from_choice,
    candidate_text_by_id,
    consume_choice_tokens,
    initialize_exec_repetition_state,
    initialize_steer_repetition_state,
    is_explicit_steer_stop,
    rollout_stop_markers,
    selected_ids_for_mode,
    update_exec_repetition_state,
    update_steer_repetition_state,
    updated_prompt_token_ids,
)
from branching_eval.config_types import BranchingConfig, DecodingConfig
from branching_eval.legacy_steer_rollout import contains_think_close_or_partial
from branching_eval.runtime_types import DecodeOutcome, PathState
from branching_eval.selector_runtime import (
    EmbeddingCache,
    build_candidate_pool_cache_key,
    parse_selection_outcomes_from_cache,
    resolve_gemini_api_key,
    select_candidates_all_modes,
)
from branching_eval.selector_types import SelectionOutcome, SelectorMode, SelectorParams
from branching_eval.steer_normalization import (
    normalize_steer_boundary_text,
    selected_candidate_normalization_suffix,
)
from branching_eval.event_types import EventContext
from branching_eval.steer_decode_flow import (
    continue_with_single_steer_candidate_async,
    resolve_steer_length_outcome_async,
    resolve_think_close_outcome_async,
    should_branch_at_trigger,
)
from branching_eval.tree_runtime_utils import (
    leaf_event_payload,
    leaf_from_choice,
    leaf_from_outcome,
    node_event_payload,
)
from branching_eval.tree_types import (
    BranchPointRecord,
    BranchTree,
    CandidatePoolRecord,
    CandidateRecord,
    LeafRollout,
    TokenTrace,
    TreeEdge,
    TreeNode,
)
from chat_templating import build_raw_im_prompt
from token_metrics import approximate_entropy
from vllm_client import GenerationChoice, ParsedToken, VllmClient, VllmRequestError

MAX_LOGPROB_ALTERNATIVES = 4
EXEC_REPEAT_SIMILARITY_THRESHOLD = 0.85
EXEC_REPEAT_SIMILARITY_LOOKBACK_WINDOW = 3
EXEC_REPEAT_TERMINATION_BLOCK_COUNT = 3
EXEC_REPEAT_STOP_REASON = "repeated_exec_block_loop"
STEER_REPEAT_SIMILARITY_THRESHOLD = 0.85
STEER_REPEAT_SIMILARITY_LOOKBACK_WINDOW = 3
STEER_REPEAT_TERMINATION_BLOCK_COUNT = 4
STEER_REPEAT_STOP_REASON = "repeated_steer_block_loop"
REPEAT_TERMINATION_MIN_GENERATED_TOKENS = 10_000
STEER_REPETITION_REQUEST_KINDS = frozenset(
    {"candidate_pool_steer_boundary", "steer_single_candidate"}
)
NODE_CHILD_ID_PATTERN = re.compile(
    r"^node_(?P<parent_node_id>.+)_(?P<child_offset>\d+)_(?P<candidate_id>\d+)$"
)


@dataclass(frozen=True)
class _RequestStreamState:
    """Previous request state used for token-prefix invariant validation.

    Args:
        request_id: Previous request id in this stream.
        input_token_ids: Previous request input token ids.
        output_token_ids: Previous request output token ids.

    Returns:
        Dataclass used for per-stream prefix assertions.
    """

    request_id: str
    input_token_ids: tuple[int, ...]
    output_token_ids: tuple[int, ...]


@dataclass(frozen=True)
class _RequestPriority:
    """Resolved priority metadata for one vLLM generation request.

    Args:
        value: Integer priority value passed to vLLM (`lower` means earlier).
        branch_number: Dotted branch number used for visualization (`1.1.2`, etc.).

    Returns:
        Dataclass containing request-priority metadata.
    """

    value: int
    branch_number: str


@dataclass(frozen=True)
class _ScheduledDecode:
    """One queued decode state with branching controls.

    Args:
        state: Decode path state to process.
        branching_enabled: Whether trigger branching is enabled for this decode.
        steer_normalization_enabled: Optional steer normalization override.

    Returns:
        Dataclass used by the streaming scheduler queue.
    """

    state: PathState
    branching_enabled: bool = True
    steer_normalization_enabled: bool | None = None


@dataclass(frozen=True)
class _ScheduledExpansion:
    """One queued triggered expansion task.

    Args:
        state: Parent decode state that triggered branching.
        outcome: Trigger decode outcome used to build branch point metadata.
        doc_id: Document id used for candidate-pool resolution.

    Returns:
        Dataclass used by async expansion queue.
    """

    state: PathState
    outcome: DecodeOutcome
    doc_id: int


class BranchExecutor:
    """Runs baseline and branching rollouts for one prompt/model pair.

    Args:
        client: vLLM client.
        prompt_text: Prompt text used for generation.
        model_name: Request-time model name.
        decoding: Decoding settings.
        branching: Branching policy settings.
        artifact_store: Artifact store used for cache reuse.
        requested_selectors: Selector modes used for fairness-shared pools.
        active_selector: Selector mode for current branching expansion.
        on_leaf_completed: Optional callback invoked on each completed leaf.
        seed: RNG seed.
        env_paths: Dotenv paths for Gemini API resolution.
        cluster_cache_path: Cache path used for Gemini cluster prompts.
        embedding_cache_path: Cache path used for Gemini embeddings.
        enable_request_priorities: Enables per-request vLLM priority metadata.

    Returns:
        Branching execution helper.
    """

    def __init__(
        self,
        *,
        client: VllmClient,
        prompt_text: str,
        model_name: str,
        decoding: DecodingConfig,
        branching: BranchingConfig,
        artifact_store: ArtifactStore,
        requested_selectors: tuple[SelectorMode, ...],
        active_selector: SelectorMode,
        seed: int,
        trigger_steer_enabled: bool,
        trigger_entropy_enabled: bool,
        env_paths: tuple[Path, ...],
        cluster_cache_path: Path,
        embedding_cache_path: Path,
        on_leaf_completed: Callable[[LeafRollout], LeafRollout] | None = None,
        enable_request_priorities: bool = False,
    ) -> None:
        self.client = client
        self.prompt_text = prompt_text
        self.model_name = model_name
        self.decoding = decoding
        self.branching = branching
        self.artifact_store = artifact_store
        self.requested_selectors: tuple[SelectorMode, ...] = requested_selectors
        self.active_selector: SelectorMode = active_selector
        self.on_leaf_completed = on_leaf_completed
        self.seed = seed
        self.enable_request_priorities = enable_request_priorities
        self.trigger_steer_enabled = trigger_steer_enabled
        self.trigger_entropy_enabled = trigger_entropy_enabled
        self.random = random.Random(seed)
        self.request_counter = 0
        self.selector_params = SelectorParams(
            branch_fanout=branching.branch_fanout,
            max_clusters=branching.max_clusters,
        )
        self.gemini_api_key = resolve_gemini_api_key(env_paths=env_paths)
        self.cluster_cache = ClusteringCache.from_path(path=cluster_cache_path)
        self.embedding_cache = EmbeddingCache(
            cache_path=embedding_cache_path,
            model_name="gemini-embedding-001",
        )
        self.max_async_inflight_requests = 1000
        self._request_semaphore: asyncio.Semaphore | None = None
        self._request_semaphore_loop_id: int | None = None
        self._event_context: EventContext | None = None
        self._request_stream_state: dict[str, _RequestStreamState] = {}
        self._request_event_counter = 0

    def run_standard_rollouts(self, *, rollout_count: int) -> list[LeafRollout]:
        """Run non-branching baseline rollouts.

        Args:
            rollout_count: Number of baseline rollouts (`N`).

        Returns:
            Baseline leaf rollout rows.
        """

        return asyncio.run(
            self.run_standard_rollouts_async(rollout_count=rollout_count)
        )

    def set_event_context(
        self,
        *,
        doc_id: int,
        doc_attempt: int,
        task_name: str,
        model_id: str,
        selector_mode: str,
    ) -> None:
        """Set event attribution labels for subsequent runtime requests/events.

        Args:
            doc_id: Document id for event attribution.
            doc_attempt: Attempt index for this document.
            task_name: Task name label.
            model_id: Model id label.
            selector_mode: Selector mode label.

        Returns:
            None.
        """

        self._event_context = EventContext(
            run_id=self.artifact_store.run_id,
            doc_id=doc_id,
            doc_attempt=doc_attempt,
            task_name=task_name,
            model_id=model_id,
            selector_mode=selector_mode,
        )
        self._request_stream_state.clear()

    def restore_request_stream_state(
        self,
        *,
        request_stream_state: dict[str, tuple[str, tuple[int, ...], tuple[int, ...]]],
    ) -> None:
        """Restore per-stream prefix state after replaying prior request logs.

        Args:
            request_stream_state: Mapping from stream id to
                `(request_id, input_token_ids, output_token_ids)` tuples.

        Returns:
            None.
        """

        self._request_stream_state = {
            stream_id: _RequestStreamState(
                request_id=request_id,
                input_token_ids=tuple(input_token_ids),
                output_token_ids=tuple(output_token_ids),
            )
            for stream_id, (
                request_id,
                input_token_ids,
                output_token_ids,
            ) in request_stream_state.items()
        }

    async def run_standard_rollouts_async(
        self, *, rollout_count: int
    ) -> list[LeafRollout]:
        """Run non-branching baseline rollouts via async completions endpoint."""

        context = self._require_event_context()
        choices = await self._generate_many_async(
            assistant_prefix="",
            prompt_token_ids=None,
            max_tokens=self.decoding.max_gen_toks,
            n=rollout_count,
            stop=None,
            request_kind="baseline_rollout_pool",
            request_stream_id=(
                f"baseline:{context.doc_id}:{context.doc_attempt}:{rollout_count}"
            ),
            enforce_prefix_chain=False,
        )
        return [
            leaf_from_choice(choice=choice, index=index)
            for index, choice in enumerate(choices)
        ]

    def run_branching_rollouts(
        self,
        *,
        doc_id: int,
        doc_attempt: int = 0,
        task_name: str,
        model_id: str,
    ) -> BranchTree:
        """Run branching rollout expansion for one document.

        Args:
            doc_id: Document id.
            task_name: Task name.
            model_id: Model id label.

        Returns:
            Branch tree with candidate pools, branch points, and leaves.
        """

        return asyncio.run(
            self.run_branching_rollouts_async(
                doc_id=doc_id,
                doc_attempt=doc_attempt,
                task_name=task_name,
                model_id=model_id,
            )
        )

    async def run_branching_rollouts_async(
        self,
        *,
        doc_id: int,
        doc_attempt: int = 0,
        task_name: str,
        model_id: str,
    ) -> BranchTree:
        """Run branching rollout expansion with streaming async scheduling."""

        self.set_event_context(
            doc_id=doc_id,
            doc_attempt=doc_attempt,
            task_name=task_name,
            model_id=model_id,
            selector_mode=self.active_selector,
        )
        tree, frontier, leaf_limit = self._initialize_tree(
            doc_id=doc_id,
            doc_attempt=doc_attempt,
            task_name=task_name,
            model_id=model_id,
        )
        await self._decode_frontier_streaming_async(
            tree=tree,
            frontier=frontier,
            doc_id=doc_id,
            leaf_limit=leaf_limit,
        )
        self._append_tree_event(
            tree=tree,
            event_type="rollout_finished",
            payload={
                "leaf_count": len(tree.leaves),
                "branch_point_count": len(tree.branch_points),
                "node_count": len(tree.nodes),
                "edge_count": len(tree.edges),
            },
        )
        return tree

    def run_branching_rollouts_from_frontier(
        self,
        *,
        doc_id: int,
        doc_attempt: int,
        task_name: str,
        model_id: str,
        tree: BranchTree,
        frontier: list[PathState],
    ) -> BranchTree:
        """Resume branching rollout expansion from replayed in-memory state.

        Args:
            doc_id: Document id.
            doc_attempt: Attempt index.
            task_name: Task name label.
            model_id: Model id label.
            tree: Replayed tree state from canonical event log.
            frontier: Replayed decode frontier to continue.

        Returns:
            Updated branch tree with resumed expansion appended.
        """

        return asyncio.run(
            self.run_branching_rollouts_from_frontier_async(
                doc_id=doc_id,
                doc_attempt=doc_attempt,
                task_name=task_name,
                model_id=model_id,
                tree=tree,
                frontier=frontier,
            )
        )

    async def run_branching_rollouts_from_frontier_async(
        self,
        *,
        doc_id: int,
        doc_attempt: int,
        task_name: str,
        model_id: str,
        tree: BranchTree,
        frontier: list[PathState],
    ) -> BranchTree:
        """Resume branching rollout expansion from replayed in-memory state.

        Args:
            doc_id: Document id.
            doc_attempt: Attempt index.
            task_name: Task name label.
            model_id: Model id label.
            tree: Replayed tree state from canonical event log.
            frontier: Replayed decode frontier to continue.

        Returns:
            Updated branch tree with resumed expansion appended.
        """

        self.set_event_context(
            doc_id=doc_id,
            doc_attempt=doc_attempt,
            task_name=task_name,
            model_id=model_id,
            selector_mode=self.active_selector,
        )
        tree.doc_id = doc_id
        tree.doc_attempt = doc_attempt
        tree.run_id = self.artifact_store.run_id
        tree.task_name = task_name
        tree.model_id = model_id
        tree.selector_mode = self.active_selector
        assert tree.nodes, "resumed tree must include at least one node"
        leaf_limit = (
            self.branching.branch_fanout**self.branching.max_branch_points_per_rollout
        )
        if frontier and len(tree.leaves) < leaf_limit:
            await self._decode_frontier_streaming_async(
                tree=tree,
                frontier=frontier,
                doc_id=doc_id,
                leaf_limit=leaf_limit,
            )
        self._append_tree_event(
            tree=tree,
            event_type="rollout_finished",
            payload={
                "leaf_count": len(tree.leaves),
                "branch_point_count": len(tree.branch_points),
                "node_count": len(tree.nodes),
                "edge_count": len(tree.edges),
                "resumed_from_logs": True,
                "frontier_count_start": len(frontier),
            },
        )
        return tree

    async def _decode_frontier_streaming_async(
        self,
        *,
        tree: BranchTree,
        frontier: list[PathState],
        doc_id: int,
        leaf_limit: int,
    ) -> None:
        """Decode frontier states as they complete and enqueue children immediately.

        Args:
            tree: Active branch tree.
            frontier: Initial decode states.
            doc_id: Document id for candidate pool resolution.
            leaf_limit: Maximum leaf count for rollout.

        Returns:
            None.
        """

        pending_decode: set[asyncio.Task[tuple[_ScheduledDecode, DecodeOutcome]]] = {
            self._schedule_decode_task(
                tree=tree,
                scheduled=_ScheduledDecode(state=state),
            )
            for state in frontier
        }
        pending_expansion: set[asyncio.Task[list[_ScheduledDecode]]] = set()
        while (pending_decode or pending_expansion) and len(tree.leaves) < leaf_limit:
            waiting: set[asyncio.Task[Any]] = set(pending_decode)
            waiting.update(pending_expansion)
            done, _ = await asyncio.wait(waiting, return_when=asyncio.FIRST_COMPLETED)
            for completed in done:
                if completed in pending_decode:
                    pending_decode.remove(
                        cast(
                            asyncio.Task[tuple[_ScheduledDecode, DecodeOutcome]],
                            completed,
                        )
                    )
                    scheduled, outcome = completed.result()
                    (
                        next_scheduled,
                        scheduled_expansion,
                    ) = self._handle_state_outcome(
                        tree=tree,
                        scheduled=scheduled,
                        outcome=outcome,
                        doc_id=doc_id,
                        leaf_limit=leaf_limit,
                    )
                    if len(tree.leaves) >= leaf_limit:
                        break
                    for child in next_scheduled:
                        pending_decode.add(
                            self._schedule_decode_task(
                                tree=tree,
                                scheduled=child,
                            )
                        )
                    if scheduled_expansion is not None:
                        pending_expansion.add(
                            self._schedule_expansion_task(
                                tree=tree,
                                scheduled=scheduled_expansion,
                            )
                        )
                    continue
                pending_expansion.remove(
                    cast(asyncio.Task[list[_ScheduledDecode]], completed)
                )
                next_scheduled = completed.result()
                if len(tree.leaves) >= leaf_limit:
                    break
                for child in next_scheduled:
                    pending_decode.add(
                        self._schedule_decode_task(
                            tree=tree,
                            scheduled=child,
                        )
                    )
            if len(tree.leaves) >= leaf_limit:
                break
        await self._cancel_pending_scheduler_tasks(
            pending_decode=pending_decode,
            pending_expansion=pending_expansion,
        )

    def _schedule_decode_task(
        self, *, tree: BranchTree, scheduled: _ScheduledDecode
    ) -> asyncio.Task[tuple[_ScheduledDecode, DecodeOutcome]]:
        """Create one async decode task for the streaming scheduler queue.

        Args:
            tree: Active branch tree.
            scheduled: Decode work item including branching controls.

        Returns:
            Async task returning `(scheduled, outcome)` when finished.
        """

        return asyncio.create_task(
            self._decode_state_outcome_async(tree=tree, scheduled=scheduled)
        )

    async def _decode_state_outcome_async(
        self, *, tree: BranchTree, scheduled: _ScheduledDecode
    ) -> tuple[_ScheduledDecode, DecodeOutcome]:
        """Decode one scheduled state and return `(scheduled, outcome)` pair."""

        if (
            scheduled.branching_enabled
            and scheduled.steer_normalization_enabled is None
        ):
            outcome = await self._decode_until_event_async(
                tree=tree,
                state=scheduled.state,
            )
        else:
            outcome = await self._decode_until_event_async(
                tree=tree,
                state=scheduled.state,
                branching_enabled=scheduled.branching_enabled,
                steer_normalization_enabled=scheduled.steer_normalization_enabled,
            )
        return scheduled, outcome

    def _handle_state_outcome(
        self,
        *,
        tree: BranchTree,
        scheduled: _ScheduledDecode,
        outcome: DecodeOutcome,
        doc_id: int,
        leaf_limit: int,
    ) -> tuple[list[_ScheduledDecode], _ScheduledExpansion | None]:
        """Persist one decode outcome and return scheduler work queues.

        Args:
            tree: Active branch tree.
            scheduled: Decoded state with branching controls.
            outcome: Decode outcome.
            doc_id: Document id for candidate pool resolution.
            leaf_limit: Maximum leaf count for rollout.

        Returns:
            Tuple `(next_decode, next_expansion)`.
        """

        state = scheduled.state
        if len(tree.leaves) >= leaf_limit:
            return [], None
        if outcome.event_type == "terminated":
            self._append_leaf_if_room(
                tree=tree,
                leaf=leaf_from_outcome(outcome=outcome, state=state),
                leaf_limit=leaf_limit,
            )
            return [], None
        if (
            scheduled.branching_enabled
            and state.branch_points_used >= self.branching.max_branch_points_per_rollout
        ):
            self._append_tree_event(
                tree=tree,
                event_type="trigger_skipped_max_branch_points",
                payload={
                    "node_id": state.node_id,
                    "trigger_type": str(outcome.trigger_type),
                    "entropy_value": outcome.entropy_value,
                },
            )
            resumed_state = self._state_from_outcome(state=state, outcome=outcome)
            return (
                [
                    _ScheduledDecode(
                        state=resumed_state,
                        branching_enabled=False,
                        steer_normalization_enabled=True,
                    )
                ],
                None,
            )
        self._append_tree_event(
            tree=tree,
            event_type="trigger_fired",
            payload={
                "node_id": state.node_id,
                "trigger_type": str(outcome.trigger_type),
                "entropy_value": outcome.entropy_value,
                "generated_tokens": outcome.generated_tokens,
            },
        )
        if len(tree.leaves) >= leaf_limit:
            return [], None
        return [], _ScheduledExpansion(
            state=state,
            outcome=outcome,
            doc_id=doc_id,
        )

    def _schedule_expansion_task(
        self,
        *,
        tree: BranchTree,
        scheduled: _ScheduledExpansion,
    ) -> asyncio.Task[list[_ScheduledDecode]]:
        """Create one async expansion task for triggered branch handling."""

        return asyncio.create_task(
            self._expand_scheduled_state_async(
                tree=tree,
                scheduled=scheduled,
            )
        )

    async def _expand_scheduled_state_async(
        self,
        *,
        tree: BranchTree,
        scheduled: _ScheduledExpansion,
    ) -> list[_ScheduledDecode]:
        """Expand one triggered decode outcome and return child decode work."""

        child_states = await self._expand_branchable_states_async(
            tree=tree,
            branchable=[(scheduled.state, scheduled.outcome)],
            doc_id=scheduled.doc_id,
        )
        return [_ScheduledDecode(state=child_state) for child_state in child_states]

    async def _cancel_pending_scheduler_tasks(
        self,
        *,
        pending_decode: set[asyncio.Task[tuple[_ScheduledDecode, DecodeOutcome]]],
        pending_expansion: set[asyncio.Task[list[_ScheduledDecode]]],
    ) -> None:
        """Cancel and drain all pending scheduler tasks."""

        pending: set[asyncio.Task[Any]] = set(pending_decode)
        pending.update(pending_expansion)
        if not pending:
            return
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)

    def _initialize_tree(
        self,
        *,
        doc_id: int,
        doc_attempt: int,
        task_name: str,
        model_id: str,
    ) -> tuple[BranchTree, list[PathState], int]:
        """Create tree/root node/frontier for one document rollout."""

        tree = BranchTree(
            doc_id=doc_id,
            doc_attempt=doc_attempt,
            run_id=self.artifact_store.run_id,
            task_name=task_name,
            model_id=model_id,
            selector_mode=self.active_selector,
            root_prompt=self.prompt_text,
        )
        root_node = TreeNode(
            node_id="node_root",
            parent_node_id=None,
            prompt_text=self.prompt_text,
            assistant_prefix="",
            prompt_token_ids=None,
            branch_points_used=0,
        )
        tree.add_node(node=root_node)
        frontier = [
            PathState(
                node_id=root_node.node_id,
                assistant_prefix="",
                prompt_token_ids=None,
                token_ids=(),
                token_traces=(),
                branch_points_used=0,
            )
        ]
        leaf_limit = (
            self.branching.branch_fanout**self.branching.max_branch_points_per_rollout
        )
        self._append_tree_event(
            tree=tree,
            event_type="rollout_started",
            payload={"root_node_id": root_node.node_id, "leaf_limit": leaf_limit},
        )
        self._append_tree_event(
            tree=tree,
            event_type="node_created",
            payload=node_event_payload(node=root_node),
        )
        return tree, frontier, leaf_limit

    async def _decode_frontier_batch_async(
        self,
        *,
        tree: BranchTree,
        frontier: list[PathState],
        doc_id: int,
        leaf_limit: int,
    ) -> list[PathState]:
        """Decode one frontier wave concurrently and expand triggered branches."""

        outcomes = await asyncio.gather(
            *[
                self._decode_until_event_async(tree=tree, state=state)
                for state in frontier
            ]
        )
        branchable: list[tuple[PathState, DecodeOutcome]] = []
        maxed: list[tuple[PathState, DecodeOutcome]] = []
        for state, outcome in zip(frontier, outcomes):
            if outcome.event_type == "terminated":
                self._append_leaf_if_room(
                    tree=tree,
                    leaf=leaf_from_outcome(outcome=outcome, state=state),
                    leaf_limit=leaf_limit,
                )
                continue
            if state.branch_points_used >= self.branching.max_branch_points_per_rollout:
                self._append_tree_event(
                    tree=tree,
                    event_type="trigger_skipped_max_branch_points",
                    payload={
                        "node_id": state.node_id,
                        "trigger_type": str(outcome.trigger_type),
                        "entropy_value": outcome.entropy_value,
                    },
                )
                maxed.append((state, outcome))
                continue
            self._append_tree_event(
                tree=tree,
                event_type="trigger_fired",
                payload={
                    "node_id": state.node_id,
                    "trigger_type": str(outcome.trigger_type),
                    "entropy_value": outcome.entropy_value,
                    "generated_tokens": outcome.generated_tokens,
                },
            )
            branchable.append((state, outcome))
        if len(tree.leaves) >= leaf_limit:
            return []
        if maxed:
            await self._resume_maxed_states_async(
                tree=tree,
                maxed=maxed,
                leaf_limit=leaf_limit,
            )
        if len(tree.leaves) >= leaf_limit or not branchable:
            return []
        return await self._expand_branchable_states_async(
            tree=tree,
            branchable=branchable,
            doc_id=doc_id,
        )

    def _append_leaf_if_room(
        self,
        *,
        tree: BranchTree,
        leaf: LeafRollout,
        leaf_limit: int,
    ) -> None:
        """Append a completed leaf only when rollout leaf budget allows it."""

        if len(tree.leaves) >= leaf_limit:
            return
        tree.leaves.append(leaf)
        self._append_tree_event(
            tree=tree,
            event_type="leaf_completed",
            payload=leaf_event_payload(leaf=leaf),
        )
        tree.leaves[-1] = self._apply_leaf_completion_hook(leaf=leaf)

    def _apply_leaf_completion_hook(self, *, leaf: LeafRollout) -> LeafRollout:
        """Apply optional leaf completion callback and keep leaf identity stable.

        Args:
            leaf: Completed leaf before optional score/verification post-processing.

        Returns:
            Leaf row after callback transformation.
        """

        if self.on_leaf_completed is None:
            return leaf
        updated_leaf = self.on_leaf_completed(leaf)
        assert (
            updated_leaf.leaf_id == leaf.leaf_id
        ), "leaf completion hook must preserve leaf_id"
        assert (
            updated_leaf.node_id == leaf.node_id
        ), "leaf completion hook must preserve node_id"
        return updated_leaf

    async def _resume_maxed_states_async(
        self,
        *,
        tree: BranchTree,
        maxed: list[tuple[PathState, DecodeOutcome]],
        leaf_limit: int,
    ) -> None:
        """Resume maxed states without branching and append final leaves.

        Steer-mode stop/normalization remains enabled so resumed paths preserve
        `<steer>/<exec>` boundary behavior while forbidding additional branches.
        """

        resumed_outcomes = await asyncio.gather(
            *[
                self._decode_until_event_async(
                    tree=tree,
                    state=self._state_from_outcome(state=state, outcome=outcome),
                    branching_enabled=False,
                    steer_normalization_enabled=True,
                )
                for state, outcome in maxed
            ]
        )
        for (state, _), resumed in zip(maxed, resumed_outcomes):
            self._append_leaf_if_room(
                tree=tree,
                leaf=leaf_from_outcome(outcome=resumed, state=state),
                leaf_limit=leaf_limit,
            )

    async def _expand_branchable_states_async(
        self,
        *,
        tree: BranchTree,
        branchable: list[tuple[PathState, DecodeOutcome]],
        doc_id: int,
    ) -> list[PathState]:
        """Resolve pools/selectors for triggered states and return next frontier."""

        pool_results = await asyncio.gather(
            *[
                self._resolve_candidate_pool_async(
                    doc_id=doc_id,
                    state=state,
                    trigger_type=str(outcome.trigger_type),
                    entropy_value=outcome.entropy_value,
                    assistant_prefix=outcome.assistant_prefix,
                    prompt_token_ids=outcome.prompt_token_ids,
                )
                for state, outcome in branchable
            ]
        )
        next_frontier: list[PathState] = []
        for (state, outcome), (pool, loaded_from_cache) in zip(
            branchable, pool_results
        ):
            expanded = await self._expand_one_triggered_state_async(
                tree=tree,
                state=state,
                outcome=outcome,
                pool=pool,
                loaded_from_cache=loaded_from_cache,
            )
            next_frontier.extend(expanded)
        return next_frontier

    async def _expand_one_triggered_state_async(
        self,
        *,
        tree: BranchTree,
        state: PathState,
        outcome: DecodeOutcome,
        pool: CandidatePoolRecord,
        loaded_from_cache: bool,
    ) -> list[PathState]:
        """Apply selection and expand children for one triggered branch point."""

        self._append_tree_event(
            tree=tree,
            event_type="candidate_pool_resolved",
            payload={
                "branch_point_id": pool.branch_point_id,
                "candidate_pool_id": pool.candidate_pool_id,
                "node_id": state.node_id,
                "trigger_type": pool.trigger_type,
                "loaded_from_cache": loaded_from_cache,
                "num_candidates": len(pool.candidates),
                "candidates": [
                    self._serialize_candidate_for_pool_event(candidate=candidate)
                    for candidate in pool.candidates
                ],
            },
        )
        selections = self._resolve_selection_outcomes(pool=pool)
        branch_point = BranchPointRecord(
            branch_point_id=pool.branch_point_id,
            node_id=state.node_id,
            trigger_type=str(outcome.trigger_type),
            entropy_value=outcome.entropy_value,
            candidate_pool_key=pool.cache_key,
            candidate_pool_id=pool.candidate_pool_id,
            selections=selections,
        )
        tree.branch_points.append(branch_point)
        tree.candidate_pools.append(pool)
        selected_ids = selected_ids_for_mode(
            selections=selections,
            selector_mode=self.active_selector,
        )
        selected_ids = self._selected_ids_for_branch(
            pool=pool, selected_ids=selected_ids
        )
        self._append_tree_event(
            tree=tree,
            event_type="selector_applied",
            payload={
                "branch_point_id": branch_point.branch_point_id,
                "node_id": state.node_id,
                "active_selector_mode": self.active_selector,
                "selected_candidate_ids": list(selected_ids),
                "selected_candidates": [
                    {
                        "candidate_id": candidate_id,
                        "text": candidate_text_by_id(
                            pool=pool, candidate_id=candidate_id
                        ),
                    }
                    for candidate_id in selected_ids
                ],
                "selected_by_mode": {
                    selection.selector_mode: list(selection.selected_candidate_ids)
                    for selection in selections
                },
                "cluster_assignments_by_mode": {
                    selection.selector_mode: self._cluster_assignment_rows(
                        selection=selection
                    )
                    for selection in selections
                    if selection.cluster_by_candidate_id is not None
                },
                "cluster_groups_by_mode": {
                    selection.selector_mode: self._cluster_group_rows(
                        selection=selection
                    )
                    for selection in selections
                    if selection.cluster_by_candidate_id is not None
                },
            },
        )
        return await self._expand_children_async(
            tree=tree,
            parent_state=state,
            outcome=outcome,
            pool=pool,
            selected_ids=selected_ids,
        )

    def _selected_ids_for_branch(
        self,
        *,
        pool: CandidatePoolRecord,
        selected_ids: tuple[int, ...],
    ) -> tuple[int, ...]:
        """Return selected ids sanitized for branch expansion.

        Args:
            pool: Candidate pool for this branch point.
            selected_ids: Selector-produced candidate ids.

        Returns:
            Candidate ids used for expansion.
        """

        if pool.trigger_type != "steer_boundary":
            return selected_ids
        return self._sanitize_steer_selected_ids(
            pool=pool,
            selected_ids=selected_ids,
        )

    def _sanitize_steer_selected_ids(
        self,
        *,
        pool: CandidatePoolRecord,
        selected_ids: tuple[int, ...],
    ) -> tuple[int, ...]:
        """Deduplicate steer selections and avoid replacement by text/id.

        Args:
            pool: Steer-trigger candidate pool.
            selected_ids: Selector-produced candidate ids.

        Returns:
            Up to `branch_fanout` unique steer candidate ids.
        """

        candidates_by_id = {
            candidate.candidate_id: candidate for candidate in pool.candidates
        }
        deduped: list[int] = []
        seen_ids: set[int] = set()
        seen_texts: set[str] = set()
        for candidate_id in selected_ids:
            candidate = candidates_by_id.get(candidate_id)
            if candidate is None:
                continue
            if candidate_id in seen_ids:
                continue
            if candidate.text in seen_texts:
                continue
            deduped.append(candidate_id)
            seen_ids.add(candidate_id)
            seen_texts.add(candidate.text)
            if len(deduped) >= self.branching.branch_fanout:
                break
        if len(deduped) >= self.branching.branch_fanout:
            return tuple(deduped)
        for candidate in pool.candidates:
            if candidate.candidate_id in seen_ids:
                continue
            if candidate.text in seen_texts:
                continue
            deduped.append(candidate.candidate_id)
            seen_ids.add(candidate.candidate_id)
            seen_texts.add(candidate.text)
            if len(deduped) >= self.branching.branch_fanout:
                break
        return tuple(deduped)

    async def _expand_children_async(
        self,
        *,
        tree: BranchTree,
        parent_state: PathState,
        outcome: DecodeOutcome,
        pool: CandidatePoolRecord,
        selected_ids: tuple[int, ...],
    ) -> list[PathState]:
        """Expand selected child nodes and return next frontier states."""

        children: list[PathState] = []
        candidates_by_id = {
            candidate.candidate_id: candidate for candidate in pool.candidates
        }
        for child_offset, candidate_id in enumerate(selected_ids):
            candidate = candidates_by_id.get(candidate_id)
            if candidate is None:
                continue
            candidate_text, candidate_token_ids = (
                await self._normalized_child_candidate_async(
                    trigger_type=pool.trigger_type,
                    candidate=candidate,
                )
            )
            child_node_id = f"node_{parent_state.node_id}_{child_offset}_{candidate_id}"
            child_prefix = outcome.assistant_prefix + candidate_text
            child_token_ids = tuple(outcome.token_ids) + tuple(candidate_token_ids)
            child_token_traces = tuple(outcome.token_traces) + tuple(candidate.tokens)
            child_state = PathState(
                node_id=child_node_id,
                assistant_prefix=child_prefix,
                prompt_token_ids=append_prompt_token_ids(
                    prompt_token_ids=outcome.prompt_token_ids,
                    continuation_token_ids=candidate_token_ids,
                ),
                token_ids=child_token_ids,
                token_traces=child_token_traces,
                branch_points_used=parent_state.branch_points_used + 1,
            )
            tree.add_node(
                node=TreeNode(
                    node_id=child_node_id,
                    parent_node_id=parent_state.node_id,
                    prompt_text=self.prompt_text,
                    assistant_prefix=child_prefix,
                    prompt_token_ids=child_state.prompt_token_ids,
                    branch_points_used=child_state.branch_points_used,
                )
            )
            edge = TreeEdge(
                edge_id=f"edge_{parent_state.node_id}_{child_node_id}",
                parent_node_id=parent_state.node_id,
                child_node_id=child_node_id,
                candidate_pool_id=pool.candidate_pool_id,
                candidate_id=candidate.candidate_id,
                selector_mode=self.active_selector,
            )
            tree.edges.append(edge)
            self._append_tree_event(
                tree=tree,
                event_type="node_created",
                payload={
                    "node_id": child_node_id,
                    "parent_node_id": parent_state.node_id,
                    "branch_points_used": child_state.branch_points_used,
                },
            )
            self._append_tree_event(
                tree=tree,
                event_type="edge_selected",
                payload={
                    "edge_id": edge.edge_id,
                    "parent_node_id": edge.parent_node_id,
                    "child_node_id": edge.child_node_id,
                    "candidate_pool_id": edge.candidate_pool_id,
                    "candidate_id": edge.candidate_id,
                    "selector_mode": edge.selector_mode,
                    "candidate_text_normalized": candidate_text,
                    "candidate_token_ids_normalized": list(candidate_token_ids),
                },
            )
            children.append(child_state)
        return children

    def _decode_until_event(
        self,
        *,
        tree: BranchTree,
        state: PathState,
        branching_enabled: bool = True,
        steer_normalization_enabled: bool | None = None,
    ) -> DecodeOutcome:
        """Compatibility wrapper that executes the async decode workflow."""

        return asyncio.run(
            self._decode_until_event_async(
                tree=tree,
                state=state,
                branching_enabled=branching_enabled,
                steer_normalization_enabled=steer_normalization_enabled,
            )
        )

    async def _decode_until_event_async(
        self,
        *,
        tree: BranchTree,
        state: PathState,
        branching_enabled: bool = True,
        steer_normalization_enabled: bool | None = None,
    ) -> DecodeOutcome:
        """Async decode loop until trigger or termination for one path state.

        Args:
            tree: Active branch tree.
            state: Decode path state.
            branching_enabled: Whether branching trigger returns are allowed.
            steer_normalization_enabled: Optional steer boundary normalization toggle.
                Defaults to `branching_enabled` for backward-compatible behavior.

        Returns:
            Decode outcome for one path.
        """

        assistant_prefix = state.assistant_prefix
        prompt_token_ids = state.prompt_token_ids
        token_ids = list(state.token_ids)
        token_traces = list(state.token_traces)
        generated_tokens = len(token_ids)
        steer_mode_enabled = (
            steer_normalization_enabled
            if steer_normalization_enabled is not None
            else branching_enabled
        )
        trigger_steer_enabled = self.trigger_steer_enabled and steer_mode_enabled
        trigger_entropy_enabled = self.trigger_entropy_enabled and branching_enabled
        entropy_threshold = (
            self._resolved_entropy_threshold() if trigger_entropy_enabled else 0.0
        )
        branch_prob = self.branching.branch_prob if branching_enabled else 0.0
        rollout_stop = rollout_stop_markers(steer_enabled=trigger_steer_enabled)
        exec_repetition_state = initialize_exec_repetition_state(
            text=assistant_prefix,
            similarity_threshold=EXEC_REPEAT_SIMILARITY_THRESHOLD,
            similarity_lookback_window=EXEC_REPEAT_SIMILARITY_LOOKBACK_WINDOW,
        )
        steer_repetition_state = initialize_steer_repetition_state(
            text=assistant_prefix,
            similarity_threshold=STEER_REPEAT_SIMILARITY_THRESHOLD,
            similarity_lookback_window=STEER_REPEAT_SIMILARITY_LOOKBACK_WINDOW,
        )

        def update_exec_repetition(*, delta_text: str) -> float | None:
            """Update exec repetition tracking from one appended text delta."""

            nonlocal exec_repetition_state
            if not trigger_steer_enabled:
                return None
            (
                exec_repetition_state,
                similarity_ratio,
            ) = update_exec_repetition_state(
                state=exec_repetition_state,
                chunk_text=delta_text,
                similarity_threshold=EXEC_REPEAT_SIMILARITY_THRESHOLD,
                similarity_lookback_window=EXEC_REPEAT_SIMILARITY_LOOKBACK_WINDOW,
            )
            return similarity_ratio

        def update_steer_repetition(*, delta_text: str) -> float | None:
            """Update steer repetition tracking from one appended text delta."""

            nonlocal steer_repetition_state
            if not trigger_steer_enabled:
                return None
            (
                steer_repetition_state,
                similarity_ratio,
            ) = update_steer_repetition_state(
                state=steer_repetition_state,
                chunk_text=delta_text,
                similarity_threshold=STEER_REPEAT_SIMILARITY_THRESHOLD,
                similarity_lookback_window=STEER_REPEAT_SIMILARITY_LOOKBACK_WINDOW,
            )
            return similarity_ratio

        def repeated_block_termination_outcome(
            *,
            last_exec_similarity_ratio: float | None,
            last_steer_similarity_ratio: float | None,
        ) -> DecodeOutcome | None:
            """Return repeat-loop termination outcome when threshold is reached."""

            if (
                trigger_steer_enabled
                and generated_tokens >= REPEAT_TERMINATION_MIN_GENERATED_TOKENS
                and exec_repetition_state.repeated_exec_blocks
                >= EXEC_REPEAT_TERMINATION_BLOCK_COUNT
            ):
                self._append_exec_repeat_terminated_event(
                    tree=tree,
                    node_id=state.node_id,
                    repeated_exec_blocks=exec_repetition_state.repeated_exec_blocks,
                    last_similarity_ratio=last_exec_similarity_ratio,
                    previous_exec_block=exec_repetition_state.previous_exec_block,
                )
                return DecodeOutcome(
                    event_type="terminated",
                    trigger_type=None,
                    entropy_value=None,
                    assistant_prefix=assistant_prefix,
                    prompt_token_ids=prompt_token_ids,
                    token_ids=tuple(token_ids),
                    token_traces=tuple(token_traces),
                    generated_tokens=generated_tokens,
                    stop_reason=EXEC_REPEAT_STOP_REASON,
                )
            if (
                trigger_steer_enabled
                and generated_tokens >= REPEAT_TERMINATION_MIN_GENERATED_TOKENS
                and steer_repetition_state.repeated_exec_blocks
                >= STEER_REPEAT_TERMINATION_BLOCK_COUNT
            ):
                self._append_steer_repeat_terminated_event(
                    tree=tree,
                    node_id=state.node_id,
                    repeated_steer_blocks=steer_repetition_state.repeated_exec_blocks,
                    last_similarity_ratio=last_steer_similarity_ratio,
                    previous_steer_block=steer_repetition_state.previous_exec_block,
                )
                return DecodeOutcome(
                    event_type="terminated",
                    trigger_type=None,
                    entropy_value=None,
                    assistant_prefix=assistant_prefix,
                    prompt_token_ids=prompt_token_ids,
                    token_ids=tuple(token_ids),
                    token_traces=tuple(token_traces),
                    generated_tokens=generated_tokens,
                    stop_reason=STEER_REPEAT_STOP_REASON,
                )
            return None

        while generated_tokens < self.decoding.max_gen_toks:
            chunk_prefix_before = assistant_prefix
            chunk_prompt_token_ids_before = prompt_token_ids
            chunk_token_ids_before = tuple(token_ids)
            generated_before_chunk = generated_tokens
            chunk_tokens = min(
                self.decoding.decode_chunk_tokens,
                self.decoding.max_gen_toks - generated_tokens,
            )
            choice = await self._generate_choice_async(
                assistant_prefix=assistant_prefix,
                prompt_token_ids=prompt_token_ids,
                max_tokens=chunk_tokens,
                stop=rollout_stop,
                n=1,
                request_kind="decode_chunk",
                request_stream_id=f"decode:{state.node_id}",
                enforce_prefix_chain=True,
            )
            raw_chunk_text = str(choice.text)
            raw_chunk_token_ids = tuple(choice.token_ids or ())

            def append_decode_event(
                *,
                prefix_after: str,
                prompt_token_ids_after: tuple[int, ...] | None,
                token_ids_after: tuple[int, ...],
                generated_after: int,
            ) -> None:
                self._append_decode_chunk_event(
                    tree=tree,
                    node_id=state.node_id,
                    raw_chunk_text=raw_chunk_text,
                    raw_chunk_token_ids=raw_chunk_token_ids,
                    finish_reason=str(choice.finish_reason),
                    generated_tokens_before_chunk=generated_before_chunk,
                    generated_tokens_after_chunk=generated_after,
                    branching_enabled=branching_enabled,
                    prefix_before=chunk_prefix_before,
                    prefix_after=prefix_after,
                    prompt_token_ids_before=chunk_prompt_token_ids_before,
                    prompt_token_ids_after=prompt_token_ids_after,
                    token_ids_before=chunk_token_ids_before,
                    token_ids_after=token_ids_after,
                )

            if not choice.tokens:
                append_decode_event(
                    prefix_after=assistant_prefix,
                    prompt_token_ids_after=prompt_token_ids,
                    token_ids_after=tuple(token_ids),
                    generated_after=generated_tokens,
                )
                return DecodeOutcome(
                    event_type="terminated",
                    trigger_type=None,
                    entropy_value=None,
                    assistant_prefix=assistant_prefix,
                    prompt_token_ids=prompt_token_ids,
                    token_ids=tuple(token_ids),
                    token_traces=tuple(token_traces),
                    generated_tokens=generated_tokens,
                    stop_reason="empty_generation",
                )
            explicit_steer_stop = is_explicit_steer_stop(
                choice=choice,
                steer_enabled=trigger_steer_enabled,
            )
            chunk_outcome = consume_choice_tokens(
                choice=choice,
                assistant_prefix=assistant_prefix,
                token_ids=token_ids,
                token_traces=token_traces,
                generated_tokens=generated_tokens,
                trigger_steer=(
                    trigger_steer_enabled
                    and branching_enabled
                    and not explicit_steer_stop
                ),
                trigger_entropy=trigger_entropy_enabled,
                entropy_threshold=entropy_threshold,
                branch_prob=branch_prob,
                rng=self.random,
            )
            assistant_prefix = chunk_outcome.assistant_prefix
            token_ids = list(chunk_outcome.token_ids)
            token_traces = list(chunk_outcome.token_traces)
            generated_tokens = chunk_outcome.generated_tokens
            consumed_tokens = generated_tokens - generated_before_chunk
            prompt_token_ids = updated_prompt_token_ids(
                current_prompt_token_ids=prompt_token_ids,
                choice=choice,
                consumed_tokens=consumed_tokens,
            )
            decode_chunk_text, _ = self._state_delta(
                prefix_before=chunk_prefix_before,
                prefix_after=assistant_prefix,
                token_ids_before=chunk_token_ids_before,
                token_ids_after=tuple(token_ids),
            )
            last_exec_similarity_ratio = update_exec_repetition(
                delta_text=decode_chunk_text
            )
            last_steer_similarity_ratio = update_steer_repetition(
                delta_text=decode_chunk_text
            )
            repeated_outcome = repeated_block_termination_outcome(
                last_exec_similarity_ratio=last_exec_similarity_ratio,
                last_steer_similarity_ratio=last_steer_similarity_ratio,
            )
            if repeated_outcome is not None:
                append_decode_event(
                    prefix_after=repeated_outcome.assistant_prefix,
                    prompt_token_ids_after=repeated_outcome.prompt_token_ids,
                    token_ids_after=repeated_outcome.token_ids,
                    generated_after=repeated_outcome.generated_tokens,
                )
                return repeated_outcome
            if chunk_outcome.event_type == "trigger":
                if chunk_outcome.trigger_type != "steer_boundary":
                    trigger_outcome = replace(
                        chunk_outcome,
                        prompt_token_ids=prompt_token_ids,
                    )
                    append_decode_event(
                        prefix_after=trigger_outcome.assistant_prefix,
                        prompt_token_ids_after=trigger_outcome.prompt_token_ids,
                        token_ids_after=trigger_outcome.token_ids,
                        generated_after=trigger_outcome.generated_tokens,
                    )
                    return trigger_outcome
                normalized_prefix, normalized_prompt_ids = (
                    await self._normalize_steer_prefix_prompt_ids_async(
                        assistant_prefix=assistant_prefix,
                        prompt_token_ids=prompt_token_ids,
                    )
                )
                trigger_outcome = replace(
                    chunk_outcome,
                    assistant_prefix=normalized_prefix,
                    prompt_token_ids=normalized_prompt_ids,
                    stop_reason="",
                )
                append_decode_event(
                    prefix_after=trigger_outcome.assistant_prefix,
                    prompt_token_ids_after=trigger_outcome.prompt_token_ids,
                    token_ids_after=trigger_outcome.token_ids,
                    generated_after=trigger_outcome.generated_tokens,
                )
                return trigger_outcome
            if trigger_steer_enabled and contains_think_close_or_partial(
                text=str(choice.text)
            ):
                think_close_outcome = await resolve_think_close_outcome_async(
                    executor=self,
                    choice=choice,
                    assistant_prefix=assistant_prefix,
                    prompt_token_ids=prompt_token_ids,
                    token_ids=tuple(token_ids),
                    token_traces=tuple(token_traces),
                    generated_tokens=generated_tokens,
                    request_stream_id=f"decode:{state.node_id}",
                )
                append_decode_event(
                    prefix_after=think_close_outcome.assistant_prefix,
                    prompt_token_ids_after=think_close_outcome.prompt_token_ids,
                    token_ids_after=think_close_outcome.token_ids,
                    generated_after=think_close_outcome.generated_tokens,
                )
                return think_close_outcome
            if explicit_steer_stop:
                normalized_prefix, normalized_prompt_ids = (
                    await self._normalize_steer_prefix_prompt_ids_async(
                        assistant_prefix=assistant_prefix,
                        prompt_token_ids=prompt_token_ids,
                    )
                )
                explicit_trigger = replace(
                    chunk_outcome,
                    event_type="trigger",
                    trigger_type="steer_boundary",
                    entropy_value=None,
                    assistant_prefix=normalized_prefix,
                    prompt_token_ids=normalized_prompt_ids,
                    stop_reason="",
                )
                explicit_trigger_text, _ = self._state_delta(
                    prefix_before=assistant_prefix,
                    prefix_after=explicit_trigger.assistant_prefix,
                    token_ids_before=tuple(token_ids),
                    token_ids_after=explicit_trigger.token_ids,
                )
                _ = update_exec_repetition(delta_text=explicit_trigger_text)
                _ = update_steer_repetition(delta_text=explicit_trigger_text)
                if branching_enabled and should_branch_at_trigger(executor=self):
                    append_decode_event(
                        prefix_after=explicit_trigger.assistant_prefix,
                        prompt_token_ids_after=explicit_trigger.prompt_token_ids,
                        token_ids_after=explicit_trigger.token_ids,
                        generated_after=explicit_trigger.generated_tokens,
                    )
                    return explicit_trigger
                continued = await continue_with_single_steer_candidate_async(
                    executor=self,
                    assistant_prefix=explicit_trigger.assistant_prefix,
                    prompt_token_ids=explicit_trigger.prompt_token_ids,
                    token_ids=explicit_trigger.token_ids,
                    token_traces=explicit_trigger.token_traces,
                    generated_tokens=explicit_trigger.generated_tokens,
                    request_stream_id=f"decode:{state.node_id}",
                )
                if continued.event_type == "terminated":
                    append_decode_event(
                        prefix_after=continued.assistant_prefix,
                        prompt_token_ids_after=continued.prompt_token_ids,
                        token_ids_after=continued.token_ids,
                        generated_after=continued.generated_tokens,
                    )
                    return continued
                append_decode_event(
                    prefix_after=explicit_trigger.assistant_prefix,
                    prompt_token_ids_after=explicit_trigger.prompt_token_ids,
                    token_ids_after=explicit_trigger.token_ids,
                    generated_after=explicit_trigger.generated_tokens,
                )
                self._append_steer_block_event(
                    tree=tree,
                    node_id=state.node_id,
                    source="explicit_stop_nonbranch",
                    base_prefix=explicit_trigger.assistant_prefix,
                    base_token_ids=explicit_trigger.token_ids,
                    generated_tokens_before_chunk=explicit_trigger.generated_tokens,
                    continued=continued,
                )
                steer_chunk_text, _ = self._state_delta(
                    prefix_before=explicit_trigger.assistant_prefix,
                    prefix_after=continued.assistant_prefix,
                    token_ids_before=explicit_trigger.token_ids,
                    token_ids_after=continued.token_ids,
                )
                last_exec_similarity_ratio = update_exec_repetition(
                    delta_text=steer_chunk_text
                )
                last_steer_similarity_ratio = update_steer_repetition(
                    delta_text=steer_chunk_text
                )
                repeated_outcome = repeated_block_termination_outcome(
                    last_exec_similarity_ratio=last_exec_similarity_ratio,
                    last_steer_similarity_ratio=last_steer_similarity_ratio,
                )
                if repeated_outcome is not None:
                    return repeated_outcome
                assistant_prefix = continued.assistant_prefix
                prompt_token_ids = continued.prompt_token_ids
                token_ids = list(continued.token_ids)
                token_traces = list(continued.token_traces)
                generated_tokens = continued.generated_tokens
                continue
            if trigger_steer_enabled and str(choice.finish_reason) == "length":
                steer_length_outcome = await resolve_steer_length_outcome_async(
                    executor=self,
                    choice=choice,
                    assistant_prefix=assistant_prefix,
                    prompt_token_ids=prompt_token_ids,
                    token_ids=tuple(token_ids),
                    token_traces=tuple(token_traces),
                    generated_tokens=generated_tokens,
                    request_stream_id=f"decode:{state.node_id}",
                )
                if steer_length_outcome.event_type != "trigger":
                    append_decode_event(
                        prefix_after=steer_length_outcome.assistant_prefix,
                        prompt_token_ids_after=steer_length_outcome.prompt_token_ids,
                        token_ids_after=steer_length_outcome.token_ids,
                        generated_after=steer_length_outcome.generated_tokens,
                    )
                    return steer_length_outcome
                if branching_enabled and should_branch_at_trigger(executor=self):
                    append_decode_event(
                        prefix_after=steer_length_outcome.assistant_prefix,
                        prompt_token_ids_after=steer_length_outcome.prompt_token_ids,
                        token_ids_after=steer_length_outcome.token_ids,
                        generated_after=steer_length_outcome.generated_tokens,
                    )
                    return steer_length_outcome
                steer_length_trigger_text, _ = self._state_delta(
                    prefix_before=assistant_prefix,
                    prefix_after=steer_length_outcome.assistant_prefix,
                    token_ids_before=tuple(token_ids),
                    token_ids_after=steer_length_outcome.token_ids,
                )
                _ = update_exec_repetition(delta_text=steer_length_trigger_text)
                _ = update_steer_repetition(delta_text=steer_length_trigger_text)
                continued = await continue_with_single_steer_candidate_async(
                    executor=self,
                    assistant_prefix=steer_length_outcome.assistant_prefix,
                    prompt_token_ids=steer_length_outcome.prompt_token_ids,
                    token_ids=steer_length_outcome.token_ids,
                    token_traces=steer_length_outcome.token_traces,
                    generated_tokens=steer_length_outcome.generated_tokens,
                    request_stream_id=f"decode:{state.node_id}",
                )
                if continued.event_type == "terminated":
                    append_decode_event(
                        prefix_after=steer_length_outcome.assistant_prefix,
                        prompt_token_ids_after=steer_length_outcome.prompt_token_ids,
                        token_ids_after=steer_length_outcome.token_ids,
                        generated_after=steer_length_outcome.generated_tokens,
                    )
                    return continued
                append_decode_event(
                    prefix_after=steer_length_outcome.assistant_prefix,
                    prompt_token_ids_after=steer_length_outcome.prompt_token_ids,
                    token_ids_after=steer_length_outcome.token_ids,
                    generated_after=steer_length_outcome.generated_tokens,
                )
                self._append_steer_block_event(
                    tree=tree,
                    node_id=state.node_id,
                    source="length_boundary_nonbranch",
                    base_prefix=steer_length_outcome.assistant_prefix,
                    base_token_ids=steer_length_outcome.token_ids,
                    generated_tokens_before_chunk=steer_length_outcome.generated_tokens,
                    continued=continued,
                )
                steer_chunk_text, _ = self._state_delta(
                    prefix_before=steer_length_outcome.assistant_prefix,
                    prefix_after=continued.assistant_prefix,
                    token_ids_before=steer_length_outcome.token_ids,
                    token_ids_after=continued.token_ids,
                )
                last_exec_similarity_ratio = update_exec_repetition(
                    delta_text=steer_chunk_text
                )
                last_steer_similarity_ratio = update_steer_repetition(
                    delta_text=steer_chunk_text
                )
                repeated_outcome = repeated_block_termination_outcome(
                    last_exec_similarity_ratio=last_exec_similarity_ratio,
                    last_steer_similarity_ratio=last_steer_similarity_ratio,
                )
                if repeated_outcome is not None:
                    return repeated_outcome
                assistant_prefix = continued.assistant_prefix
                prompt_token_ids = continued.prompt_token_ids
                token_ids = list(continued.token_ids)
                token_traces = list(continued.token_traces)
                generated_tokens = continued.generated_tokens
                continue
            if choice.finish_reason in {"stop", "eos"}:
                finished_outcome = replace(
                    chunk_outcome,
                    prompt_token_ids=prompt_token_ids,
                    stop_reason="model_finished",
                )
                append_decode_event(
                    prefix_after=finished_outcome.assistant_prefix,
                    prompt_token_ids_after=finished_outcome.prompt_token_ids,
                    token_ids_after=finished_outcome.token_ids,
                    generated_after=finished_outcome.generated_tokens,
                )
                return finished_outcome
            append_decode_event(
                prefix_after=assistant_prefix,
                prompt_token_ids_after=prompt_token_ids,
                token_ids_after=tuple(token_ids),
                generated_after=generated_tokens,
            )
        return DecodeOutcome(
            event_type="terminated",
            trigger_type=None,
            entropy_value=None,
            assistant_prefix=assistant_prefix,
            prompt_token_ids=prompt_token_ids,
            token_ids=tuple(token_ids),
            token_traces=tuple(token_traces),
            generated_tokens=generated_tokens,
            stop_reason="max_gen_toks_reached",
        )

    @staticmethod
    def _state_from_outcome(*, state: PathState, outcome: DecodeOutcome) -> PathState:
        return PathState(
            node_id=state.node_id,
            assistant_prefix=outcome.assistant_prefix,
            prompt_token_ids=outcome.prompt_token_ids,
            token_ids=outcome.token_ids,
            token_traces=outcome.token_traces,
            branch_points_used=state.branch_points_used,
        )

    def _resolve_candidate_pool(
        self,
        *,
        doc_id: int,
        state: PathState,
        trigger_type: str,
        entropy_value: float | None,
        assistant_prefix: str,
        prompt_token_ids: tuple[int, ...] | None,
    ) -> tuple[CandidatePoolRecord, bool]:
        """Compatibility wrapper that executes async pool resolution."""

        return asyncio.run(
            self._resolve_candidate_pool_async(
                doc_id=doc_id,
                state=state,
                trigger_type=trigger_type,
                entropy_value=entropy_value,
                assistant_prefix=assistant_prefix,
                prompt_token_ids=prompt_token_ids,
            )
        )

    async def _resolve_candidate_pool_async(
        self,
        *,
        doc_id: int,
        state: PathState,
        trigger_type: str,
        entropy_value: float | None,
        assistant_prefix: str,
        prompt_token_ids: tuple[int, ...] | None,
    ) -> tuple[CandidatePoolRecord, bool]:
        """Resolve candidate pool asynchronously with cache reuse."""

        cache_key = build_candidate_pool_cache_key(
            doc_id=doc_id,
            node_id=state.node_id,
            trigger_type=trigger_type,
            seed=self.seed,
            model_name=self.model_name,
            decoding=self.decoding,
            branching=self.branching,
        )
        cached_pool_id = self.artifact_store.load_candidate_pool_id(cache_key=cache_key)
        if cached_pool_id is not None:
            cached = self.artifact_store.load_candidate_pool(
                candidate_pool_id=cached_pool_id
            )
            if cached is not None:
                return cached, True
        pool = await self._generate_candidate_pool_async(
            cache_key=cache_key,
            state=state,
            trigger_type=trigger_type,
            entropy_value=entropy_value,
            assistant_prefix=assistant_prefix,
            prompt_token_ids=prompt_token_ids,
        )
        self.artifact_store.persist_candidate_pool(cache_key=cache_key, pool=pool)
        return pool, False

    def _generate_candidate_pool(
        self,
        *,
        cache_key: str,
        state: PathState,
        trigger_type: str,
        entropy_value: float | None,
        assistant_prefix: str,
        prompt_token_ids: tuple[int, ...] | None,
    ) -> CandidatePoolRecord:
        """Compatibility wrapper that executes async candidate generation."""

        return asyncio.run(
            self._generate_candidate_pool_async(
                cache_key=cache_key,
                state=state,
                trigger_type=trigger_type,
                entropy_value=entropy_value,
                assistant_prefix=assistant_prefix,
                prompt_token_ids=prompt_token_ids,
            )
        )

    async def _generate_candidate_pool_async(
        self,
        *,
        cache_key: str,
        state: PathState,
        trigger_type: str,
        entropy_value: float | None,
        assistant_prefix: str,
        prompt_token_ids: tuple[int, ...] | None,
    ) -> CandidatePoolRecord:
        """Generate one candidate pool via async completions calls."""

        if trigger_type == "steer_boundary":
            canonical_prefix, canonical_prompt_ids = (
                await self._normalize_steer_prefix_prompt_ids_async(
                    assistant_prefix=assistant_prefix,
                    prompt_token_ids=prompt_token_ids,
                )
            )
            choices = await self._generate_many_async(
                assistant_prefix=canonical_prefix,
                prompt_token_ids=canonical_prompt_ids,
                max_tokens=self.branching.max_steer_tokens,
                n=self.branching.num_candidates,
                stop=("</steer",),
                temperature=1.0,
                request_kind="candidate_pool_steer_boundary",
                request_stream_id=(
                    f"candidate_pool:{state.node_id}:{trigger_type}:{cache_key}"
                ),
                enforce_prefix_chain=False,
            )
        else:
            choices = await self._generate_many_async(
                assistant_prefix=assistant_prefix,
                prompt_token_ids=prompt_token_ids,
                max_tokens=self.branching.candidate_span_tokens,
                n=self.branching.num_candidates,
                stop=None,
                temperature=1.0,
                request_kind="candidate_pool_high_entropy",
                request_stream_id=(
                    f"candidate_pool:{state.node_id}:{trigger_type}:{cache_key}"
                ),
                enforce_prefix_chain=False,
            )
        pool_id = f"pool_{hashlib.sha256(cache_key.encode('utf-8')).hexdigest()[:16]}"
        candidates = tuple(
            candidate_from_choice(
                candidate_id=index,
                choice=choice,
                enforce_steer_stop_boundary=trigger_type == "steer_boundary",
            )
            for index, choice in enumerate(choices)
        )
        aligned_candidates = await asyncio.gather(
            *[
                self._candidate_with_aligned_text_async(candidate=candidate)
                for candidate in candidates
            ]
        )
        candidates = tuple(aligned_candidates)
        if trigger_type == "steer_boundary":
            for candidate in candidates:
                assert_no_text_after_first_steer_close(text=candidate.text)
        return CandidatePoolRecord(
            candidate_pool_id=pool_id,
            cache_key=cache_key,
            branch_point_id=f"bp_{state.node_id}_{pool_id}",
            node_id=state.node_id,
            trigger_type=trigger_type,
            entropy_value=entropy_value,
            candidates=candidates,
        )

    def _resolve_selection_outcomes(
        self, *, pool: CandidatePoolRecord
    ) -> tuple[SelectionOutcome, ...]:
        cached = self.artifact_store.load_selection_cache(
            candidate_pool_id=pool.candidate_pool_id
        )
        if cached is not None:
            return parse_selection_outcomes_from_cache(cached=cached)
        outcomes = select_candidates_all_modes(
            pool=pool,
            selector_params=self.selector_params,
            selector_modes=self.requested_selectors,
            rng=self.random,
            cluster_cache=self.cluster_cache,
            embedding_cache=self.embedding_cache,
            gemini_api_key=self.gemini_api_key,
        )
        self.artifact_store.persist_selection_cache(
            candidate_pool_id=pool.candidate_pool_id,
            selections=outcomes,
        )
        return outcomes

    def _normalize_steer_prefix_prompt_ids(
        self,
        *,
        assistant_prefix: str,
        prompt_token_ids: tuple[int, ...] | None,
    ) -> tuple[str, tuple[int, ...] | None]:
        normalized_prefix = normalize_steer_boundary_text(text=assistant_prefix)
        if prompt_token_ids is None:
            return normalized_prefix, None
        if not normalized_prefix.startswith(assistant_prefix):
            return normalized_prefix, None
        injected_suffix = normalized_prefix[len(assistant_prefix) :]
        if not injected_suffix:
            return normalized_prefix, prompt_token_ids
        suffix_token_ids = self.client.tokenize(
            model=self.model_name,
            text=injected_suffix,
            add_special_tokens=False,
        )
        self._assert_text_token_alignment(
            text=injected_suffix,
            token_ids=tuple(suffix_token_ids),
            context="normalize_steer_prefix_suffix",
        )
        return normalized_prefix, tuple(prompt_token_ids) + tuple(suffix_token_ids)

    async def _normalize_steer_prefix_prompt_ids_async(
        self,
        *,
        assistant_prefix: str,
        prompt_token_ids: tuple[int, ...] | None,
    ) -> tuple[str, tuple[int, ...] | None]:
        """Async steer-prefix normalization using async tokenizer endpoints."""

        normalized_prefix = normalize_steer_boundary_text(text=assistant_prefix)
        if prompt_token_ids is None:
            return normalized_prefix, None
        if not normalized_prefix.startswith(assistant_prefix):
            return normalized_prefix, None
        injected_suffix = normalized_prefix[len(assistant_prefix) :]
        if not injected_suffix:
            return normalized_prefix, prompt_token_ids
        suffix_token_ids = await self._tokenize_text_async(text=injected_suffix)
        await self._assert_text_token_alignment_async(
            text=injected_suffix,
            token_ids=tuple(suffix_token_ids),
            context="normalize_steer_prefix_suffix",
        )
        return normalized_prefix, tuple(prompt_token_ids) + tuple(suffix_token_ids)

    def _normalized_child_candidate(
        self, *, trigger_type: str, candidate: CandidateRecord
    ) -> tuple[str, tuple[int, ...]]:
        aligned_candidate = self._candidate_with_aligned_text(candidate=candidate)
        if trigger_type != "steer_boundary":
            return aligned_candidate.text, aligned_candidate.token_ids
        assert_no_text_after_first_steer_close(text=aligned_candidate.text)
        injected_suffix, _ = selected_candidate_normalization_suffix(
            text=aligned_candidate.text
        )
        if not injected_suffix:
            self._assert_text_token_alignment(
                text=aligned_candidate.text,
                token_ids=tuple(aligned_candidate.token_ids),
                context="normalized_child_candidate",
            )
            return aligned_candidate.text, aligned_candidate.token_ids
        suffix_token_ids = self.client.tokenize(
            model=self.model_name,
            text=injected_suffix,
            add_special_tokens=False,
        )
        normalized_text = aligned_candidate.text + injected_suffix
        normalized_token_ids = tuple(aligned_candidate.token_ids) + tuple(
            suffix_token_ids
        )
        self._assert_text_token_alignment(
            text=normalized_text,
            token_ids=normalized_token_ids,
            context="normalized_child_candidate",
        )
        return (normalized_text, normalized_token_ids)

    async def _normalized_child_candidate_async(
        self, *, trigger_type: str, candidate: CandidateRecord
    ) -> tuple[str, tuple[int, ...]]:
        """Async child-candidate normalization with async tokenizer validation."""

        aligned_candidate = await self._candidate_with_aligned_text_async(
            candidate=candidate
        )
        if trigger_type != "steer_boundary":
            return aligned_candidate.text, aligned_candidate.token_ids
        assert_no_text_after_first_steer_close(text=aligned_candidate.text)
        injected_suffix, _ = selected_candidate_normalization_suffix(
            text=aligned_candidate.text
        )
        if not injected_suffix:
            await self._assert_text_token_alignment_async(
                text=aligned_candidate.text,
                token_ids=tuple(aligned_candidate.token_ids),
                context="normalized_child_candidate",
            )
            return aligned_candidate.text, aligned_candidate.token_ids
        suffix_token_ids = await self._tokenize_text_async(text=injected_suffix)
        normalized_text = aligned_candidate.text + injected_suffix
        normalized_token_ids = tuple(aligned_candidate.token_ids) + tuple(
            suffix_token_ids
        )
        await self._assert_text_token_alignment_async(
            text=normalized_text,
            token_ids=normalized_token_ids,
            context="normalized_child_candidate",
        )
        return (normalized_text, normalized_token_ids)

    def _candidate_with_aligned_text(
        self, *, candidate: CandidateRecord
    ) -> CandidateRecord:
        """Return candidate with text aligned to detokenized token ids when available.

        Args:
            candidate: Candidate record.

        Returns:
            Candidate whose text matches detokenized token ids.
        """

        if not candidate.token_ids:
            return candidate
        detokenize = getattr(self.client, "detokenize", None)
        if detokenize is None:
            return candidate
        decoded_text = str(
            detokenize(model=self.model_name, token_ids=candidate.token_ids)
        )
        if decoded_text == candidate.text:
            return candidate
        return replace(candidate, text=decoded_text)

    async def _candidate_with_aligned_text_async(
        self, *, candidate: CandidateRecord
    ) -> CandidateRecord:
        """Async candidate text alignment by detokenizing candidate token IDs."""

        if not candidate.token_ids:
            return candidate
        decoded_text = await self._detokenize_ids_async(token_ids=candidate.token_ids)
        if decoded_text is None:
            return candidate
        if decoded_text == candidate.text:
            return candidate
        return replace(candidate, text=decoded_text)

    def _assert_text_token_alignment(
        self,
        *,
        text: str,
        token_ids: tuple[int, ...],
        context: str,
    ) -> None:
        """Assert that detokenizing `token_ids` reproduces `text` exactly.

        Args:
            text: Expected text representation.
            token_ids: Token IDs expected to decode to `text`.
            context: Debug label used in assertion messages.

        Returns:
            None.
        """

        if not token_ids:
            assert text == "", f"{context}: empty token_ids for non-empty text"
            return
        detokenize = getattr(self.client, "detokenize", None)
        if detokenize is None:
            return
        decoded_text = str(detokenize(model=self.model_name, token_ids=token_ids))
        assert decoded_text == text, (
            f"{context}: detokenized text mismatch; "
            f"decoded={decoded_text!r} expected={text!r}"
        )

    async def _assert_text_token_alignment_async(
        self,
        *,
        text: str,
        token_ids: tuple[int, ...],
        context: str,
    ) -> None:
        """Async assertion that detokenizing `token_ids` reproduces `text`."""

        if not token_ids:
            assert text == "", f"{context}: empty token_ids for non-empty text"
            return
        decoded_text = await self._detokenize_ids_async(token_ids=token_ids)
        if decoded_text is None:
            return
        assert decoded_text == text, (
            f"{context}: detokenized text mismatch; "
            f"decoded={decoded_text!r} expected={text!r}"
        )

    def _resolved_entropy_threshold(self) -> float:
        if self.branching.entropy_threshold is None:
            raise RuntimeError(
                "entropy_threshold must be resolved before branching execution"
            )
        return self.branching.entropy_threshold

    def _append_tree_event(
        self,
        *,
        tree: BranchTree,
        event_type: str,
        payload: dict[str, object],
    ) -> None:
        self.artifact_store.append_tree_event(
            tree=tree,
            event_type=event_type,
            payload=payload,
        )

    def _require_event_context(self) -> EventContext:
        if self._event_context is None:
            self._event_context = EventContext(
                run_id=self.artifact_store.run_id,
                doc_id=-1,
                doc_attempt=0,
                task_name="unknown",
                model_id=self.model_name,
                selector_mode=self.active_selector,
            )
        return self._event_context

    def _append_steer_block_event(
        self,
        *,
        tree: BranchTree,
        node_id: str,
        source: str,
        base_prefix: str,
        base_token_ids: tuple[int, ...],
        generated_tokens_before_chunk: int,
        continued: DecodeOutcome,
    ) -> None:
        """Append one steer-block generation event for non-branch continuation.

        Args:
            tree: Active tree.
            node_id: Node id being decoded.
            source: Continuation source label.
            base_prefix: Assistant prefix before steer-block generation.
            base_token_ids: Token ids before steer-block generation.
            generated_tokens_before_chunk: Generated-token count before steer block.
            continued: Decode outcome after steer-block generation.

        Returns:
            None.
        """

        if continued.assistant_prefix.startswith(base_prefix):
            chunk_text = continued.assistant_prefix[len(base_prefix) :]
        else:
            chunk_text = continued.assistant_prefix
        if len(continued.token_ids) >= len(base_token_ids):
            chunk_token_ids = list(continued.token_ids[len(base_token_ids) :])
        else:
            chunk_token_ids = list(continued.token_ids)
        self._append_tree_event(
            tree=tree,
            event_type="steer_block_generated",
            payload={
                "node_id": node_id,
                "source": source,
                "chunk_text": chunk_text,
                "chunk_token_ids": chunk_token_ids,
                "generated_tokens_before_chunk": generated_tokens_before_chunk,
                "generated_tokens_after_chunk": continued.generated_tokens,
                "branching_enabled": True,
            },
        )

    def _append_exec_repeat_terminated_event(
        self,
        *,
        tree: BranchTree,
        node_id: str,
        repeated_exec_blocks: int,
        last_similarity_ratio: float | None,
        previous_exec_block: str | None,
    ) -> None:
        """Append one branch-termination event for repeated exec-block loops.

        Args:
            tree: Active tree.
            node_id: Node id being terminated.
            repeated_exec_blocks: Consecutive near-duplicate block count.
            last_similarity_ratio: Last computed fuzzy similarity ratio.
            previous_exec_block: Most recent normalized exec block text.

        Returns:
            None.
        """

        preview_text = (
            previous_exec_block[:240] if previous_exec_block is not None else ""
        )
        self._append_tree_event(
            tree=tree,
            event_type="exec_repeat_terminated",
            payload={
                "node_id": node_id,
                "similarity_threshold": EXEC_REPEAT_SIMILARITY_THRESHOLD,
                "similarity_lookback_window": (EXEC_REPEAT_SIMILARITY_LOOKBACK_WINDOW),
                "termination_block_count": EXEC_REPEAT_TERMINATION_BLOCK_COUNT,
                "repeated_exec_blocks": repeated_exec_blocks,
                "last_similarity_ratio": last_similarity_ratio,
                "normalized_exec_block_preview": preview_text,
            },
        )

    def _append_steer_repeat_terminated_event(
        self,
        *,
        tree: BranchTree,
        node_id: str,
        repeated_steer_blocks: int,
        last_similarity_ratio: float | None,
        previous_steer_block: str | None,
    ) -> None:
        """Append one branch-termination event for repeated steer-block loops.

        Args:
            tree: Active tree.
            node_id: Node id being terminated.
            repeated_steer_blocks: Consecutive near-duplicate steer-block count.
            last_similarity_ratio: Last computed fuzzy similarity ratio.
            previous_steer_block: Most recent normalized steer block text.

        Returns:
            None.
        """

        preview_text = (
            previous_steer_block[:240] if previous_steer_block is not None else ""
        )
        self._append_tree_event(
            tree=tree,
            event_type="steer_repeat_terminated",
            payload={
                "node_id": node_id,
                "similarity_threshold": STEER_REPEAT_SIMILARITY_THRESHOLD,
                "similarity_lookback_window": (STEER_REPEAT_SIMILARITY_LOOKBACK_WINDOW),
                "termination_block_count": STEER_REPEAT_TERMINATION_BLOCK_COUNT,
                "repeated_steer_blocks": repeated_steer_blocks,
                "last_similarity_ratio": last_similarity_ratio,
                "normalized_steer_block_preview": preview_text,
            },
        )

    def _next_seed(self) -> int:
        self.request_counter += 1
        return 10_000_000 + self.request_counter

    @staticmethod
    def _state_delta(
        *,
        prefix_before: str,
        prefix_after: str,
        token_ids_before: tuple[int, ...],
        token_ids_after: tuple[int, ...],
    ) -> tuple[str, list[int]]:
        """Return text/token delta between two decode states.

        Args:
            prefix_before: Prefix before decode step.
            prefix_after: Prefix after decode step.
            token_ids_before: Token ids before decode step.
            token_ids_after: Token ids after decode step.

        Returns:
            Tuple of `(chunk_text, chunk_token_ids)` appended during step.
        """

        if prefix_after.startswith(prefix_before):
            chunk_text = prefix_after[len(prefix_before) :]
        else:
            chunk_text = prefix_after
        if len(token_ids_after) >= len(token_ids_before):
            chunk_token_ids = list(token_ids_after[len(token_ids_before) :])
        else:
            chunk_token_ids = list(token_ids_after)
        return (chunk_text, chunk_token_ids)

    def _append_decode_chunk_event(
        self,
        *,
        tree: BranchTree,
        node_id: str,
        raw_chunk_text: str,
        raw_chunk_token_ids: tuple[int, ...],
        finish_reason: str,
        generated_tokens_before_chunk: int,
        generated_tokens_after_chunk: int,
        branching_enabled: bool,
        prefix_before: str,
        prefix_after: str,
        prompt_token_ids_before: tuple[int, ...] | None,
        prompt_token_ids_after: tuple[int, ...] | None,
        token_ids_before: tuple[int, ...],
        token_ids_after: tuple[int, ...],
    ) -> None:
        """Append one decode-chunk event from true state transitions.

        Args:
            tree: Active branch tree.
            node_id: Node id for the decode step.
            raw_chunk_text: Raw chunk text from vLLM for this request.
            raw_chunk_token_ids: Raw token ids from vLLM for this request.
            finish_reason: vLLM finish reason for this request.
            generated_tokens_before_chunk: Generated token count before request.
            generated_tokens_after_chunk: Generated token count after request.
            branching_enabled: Whether branching was enabled for this request.
            prefix_before: Prefix before request.
            prefix_after: Prefix after executor processing.
            prompt_token_ids_before: Prompt token ids before request.
            prompt_token_ids_after: Prompt token ids after executor processing.
            token_ids_before: Token ids before request.
            token_ids_after: Token ids after executor processing.

        Returns:
            None.
        """

        chunk_text, generated_chunk_token_ids = self._state_delta(
            prefix_before=prefix_before,
            prefix_after=prefix_after,
            token_ids_before=token_ids_before,
            token_ids_after=token_ids_after,
        )
        if (
            prompt_token_ids_before is not None
            and prompt_token_ids_after is not None
            and len(prompt_token_ids_after) >= len(prompt_token_ids_before)
        ):
            chunk_token_ids = list(
                prompt_token_ids_after[len(prompt_token_ids_before) :]
            )
            token_source = "prompt_token_delta"
        else:
            chunk_token_ids = generated_chunk_token_ids
            token_source = "generated_token_delta"
        chunk_was_normalized = chunk_text != raw_chunk_text or tuple(
            chunk_token_ids
        ) != tuple(raw_chunk_token_ids)
        self._append_tree_event(
            tree=tree,
            event_type="decode_chunk",
            payload={
                "node_id": node_id,
                "chunk_text": chunk_text,
                "chunk_was_normalized": chunk_was_normalized,
                "chunk_token_ids": chunk_token_ids,
                "chunk_token_ids_source": token_source,
                "finish_reason": finish_reason,
                "generated_tokens_before_chunk": generated_tokens_before_chunk,
                "generated_tokens_after_chunk": generated_tokens_after_chunk,
                "branching_enabled": branching_enabled,
            },
        )

    def _generate_choice(
        self,
        *,
        assistant_prefix: str,
        prompt_token_ids: tuple[int, ...] | None,
        max_tokens: int,
        stop: tuple[str, ...] | None,
        n: int,
        temperature: float | None = None,
        request_kind: str = "decode_chunk",
        request_stream_id: str | None = None,
        enforce_prefix_chain: bool = True,
    ) -> GenerationChoice:
        """Compatibility wrapper that executes async single-choice generation."""

        return asyncio.run(
            self._generate_choice_async(
                assistant_prefix=assistant_prefix,
                prompt_token_ids=prompt_token_ids,
                max_tokens=max_tokens,
                stop=stop,
                n=n,
                temperature=temperature,
                request_kind=request_kind,
                request_stream_id=request_stream_id,
                enforce_prefix_chain=enforce_prefix_chain,
            )
        )

    async def _generate_choice_async(
        self,
        *,
        assistant_prefix: str,
        prompt_token_ids: tuple[int, ...] | None,
        max_tokens: int,
        stop: tuple[str, ...] | None,
        n: int,
        temperature: float | None = None,
        request_kind: str = "decode_chunk",
        request_stream_id: str | None = None,
        enforce_prefix_chain: bool = True,
    ) -> GenerationChoice:
        """Async one-choice generation helper."""

        choices = await self._generate_many_async(
            assistant_prefix=assistant_prefix,
            prompt_token_ids=prompt_token_ids,
            max_tokens=max_tokens,
            n=n,
            stop=stop,
            temperature=temperature,
            request_kind=request_kind,
            request_stream_id=request_stream_id,
            enforce_prefix_chain=enforce_prefix_chain,
        )
        return choices[0]

    def _generate_many(
        self,
        *,
        assistant_prefix: str,
        prompt_token_ids: tuple[int, ...] | None,
        max_tokens: int,
        n: int,
        stop: tuple[str, ...] | None,
        temperature: float | None = None,
        request_kind: str = "candidate_pool",
        request_stream_id: str | None = None,
        enforce_prefix_chain: bool = False,
    ) -> tuple[GenerationChoice, ...]:
        """Compatibility wrapper that executes async multi-choice generation."""

        return asyncio.run(
            self._generate_many_async(
                assistant_prefix=assistant_prefix,
                prompt_token_ids=prompt_token_ids,
                max_tokens=max_tokens,
                n=n,
                stop=stop,
                temperature=temperature,
                request_kind=request_kind,
                request_stream_id=request_stream_id,
                enforce_prefix_chain=enforce_prefix_chain,
            )
        )

    async def _generate_many_async(
        self,
        *,
        assistant_prefix: str,
        prompt_token_ids: tuple[int, ...] | None,
        max_tokens: int,
        n: int,
        stop: tuple[str, ...] | None,
        temperature: float | None = None,
        request_kind: str = "candidate_pool",
        request_stream_id: str | None = None,
        enforce_prefix_chain: bool = False,
    ) -> tuple[GenerationChoice, ...]:
        """Async generation helper that always uses async completions endpoint."""

        completions_async = getattr(self.client, "completions_async", None)
        assert completions_async is not None, "client must provide completions_async"
        prompt_text = build_raw_im_prompt(
            prompt=self.prompt_text,
            assistant_prefix=assistant_prefix,
        )
        sample_temperature = (
            self.decoding.temperature if temperature is None else float(temperature)
        )
        repetition_penalty = self._request_repetition_penalty(request_kind=request_kind)
        generation_seed = self._next_seed()
        stream_id = (
            request_stream_id
            if request_stream_id is not None
            else f"{request_kind}:oneshot:{generation_seed}"
        )
        request_priority = self._resolve_request_priority(request_stream_id=stream_id)
        prefix_chain_enabled = bool(
            enforce_prefix_chain and request_stream_id is not None and n == 1
        )
        use_token_prompt = prompt_token_ids is not None
        cached_support = getattr(self.client, "supports_prompt_token_ids", None)
        if use_token_prompt and cached_support is not False:
            try:
                choices = await self._request_completions_with_limit(
                    completions_async=completions_async,
                    model=self.model_name,
                    prompt=None,
                    prompt_token_ids=prompt_token_ids,
                    temperature=sample_temperature,
                    top_p=self.decoding.top_p,
                    max_tokens=max_tokens,
                    n=n,
                    seed=generation_seed,
                    stop=stop,
                    top_logprobs=self.decoding.top_logprobs,
                    assistant_prefix=assistant_prefix,
                    request_kind=request_kind,
                    request_stream_id=stream_id,
                    prefix_chain_enabled=prefix_chain_enabled,
                    request_priority=request_priority,
                    repetition_penalty=repetition_penalty,
                )
                setattr(self.client, "supports_prompt_token_ids", True)
                return choices
            except VllmRequestError:
                setattr(self.client, "supports_prompt_token_ids", False)
        return await self._request_completions_with_limit(
            completions_async=completions_async,
            model=self.model_name,
            prompt=prompt_text,
            prompt_token_ids=None,
            temperature=sample_temperature,
            top_p=self.decoding.top_p,
            max_tokens=max_tokens,
            n=n,
            seed=generation_seed,
            stop=stop,
            top_logprobs=self.decoding.top_logprobs,
            assistant_prefix=assistant_prefix,
            request_kind=request_kind,
            request_stream_id=stream_id,
            prefix_chain_enabled=prefix_chain_enabled,
            request_priority=request_priority,
            repetition_penalty=repetition_penalty,
        )

    async def _request_completions_with_limit(
        self,
        *,
        completions_async: Callable[..., Awaitable[tuple[GenerationChoice, ...]]],
        model: str,
        prompt: str | None,
        prompt_token_ids: tuple[int, ...] | None,
        temperature: float,
        top_p: float,
        max_tokens: int,
        n: int,
        seed: int,
        stop: tuple[str, ...] | None,
        top_logprobs: int,
        assistant_prefix: str,
        request_kind: str,
        request_stream_id: str,
        prefix_chain_enabled: bool,
        request_priority: _RequestPriority | None = None,
        repetition_penalty: float | None = None,
    ) -> tuple[GenerationChoice, ...]:
        """Dispatch one async completions request under global inflight limit."""

        semaphore = self._ensure_request_semaphore()
        assert callable(completions_async), "completions_async must be callable"
        request_id = self._next_request_id()
        input_token_ids = await self._resolve_input_token_ids_async(
            prompt=prompt,
            prompt_token_ids=prompt_token_ids,
        )
        base_prefix_ids, delta_ids, prev_request_id = self._resolve_prefix_base_delta(
            request_stream_id=request_stream_id,
            current_input_token_ids=input_token_ids,
            prefix_chain_enabled=prefix_chain_enabled,
        )
        self._append_vllm_request_event(
            request_id=request_id,
            request_stream_id=request_stream_id,
            prev_request_id=prev_request_id,
            request_kind=request_kind,
            assistant_prefix=assistant_prefix,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            n=n,
            seed=seed,
            stop=stop,
            top_logprobs=top_logprobs,
            current_input_token_ids=input_token_ids,
            base_prefix_token_ids=base_prefix_ids,
            delta_input_token_ids=delta_ids,
            request_priority_value=(
                request_priority.value if request_priority is not None else None
            ),
            request_branch_number=(
                request_priority.branch_number if request_priority is not None else None
            ),
            repetition_penalty=repetition_penalty,
        )
        start_time = asyncio.get_running_loop().time()
        async with semaphore:
            try:
                choices = await completions_async(
                    model=model,
                    prompt=prompt,
                    prompt_token_ids=prompt_token_ids,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    n=n,
                    seed=seed,
                    stop=stop,
                    top_logprobs=top_logprobs,
                    priority=(
                        request_priority.value if request_priority is not None else None
                    ),
                    repetition_penalty=repetition_penalty,
                )
            except Exception as exc:
                latency_seconds = asyncio.get_running_loop().time() - start_time
                self._append_vllm_response_error_event(
                    request_id=request_id,
                    request_stream_id=request_stream_id,
                    request_kind=request_kind,
                    error_message=str(exc),
                    latency_seconds=latency_seconds,
                )
                raise
        latency_seconds = asyncio.get_running_loop().time() - start_time
        self._append_vllm_response_event(
            request_id=request_id,
            request_stream_id=request_stream_id,
            request_kind=request_kind,
            latency_seconds=latency_seconds,
            choices=choices,
        )
        if prefix_chain_enabled and choices:
            output_token_ids = tuple(choices[0].token_ids or ())
            self._request_stream_state[request_stream_id] = _RequestStreamState(
                request_id=request_id,
                input_token_ids=input_token_ids,
                output_token_ids=output_token_ids,
            )
        return choices

    async def _resolve_input_token_ids_async(
        self,
        *,
        prompt: str | None,
        prompt_token_ids: tuple[int, ...] | None,
    ) -> tuple[int, ...]:
        if prompt_token_ids is not None:
            return tuple(prompt_token_ids)
        assert (
            prompt is not None
        ), "prompt text is required when prompt_token_ids is None"
        return await self._tokenize_text_async(text=prompt)

    def _next_request_id(self) -> str:
        self._request_event_counter += 1
        return f"req_{self._request_event_counter:08d}"

    def _resolve_request_priority(
        self, *, request_stream_id: str
    ) -> _RequestPriority | None:
        """Resolve optional request priority from branch-number ordering."""

        if not self.enable_request_priorities:
            return None
        node_id = self._node_id_from_request_stream_id(
            request_stream_id=request_stream_id
        )
        if node_id is None:
            return None
        branch_path = self._branch_path_for_node_id(node_id=node_id)
        if branch_path is None:
            return None
        branch_number = ".".join(str(value) for value in branch_path)
        priority_value = self._priority_value_for_branch_path(branch_path=branch_path)
        return _RequestPriority(value=priority_value, branch_number=branch_number)

    def _request_repetition_penalty(self, *, request_kind: str) -> float | None:
        """Resolve optional repetition penalty for steer-token generation.

        Args:
            request_kind: Request kind label for the outbound vLLM call.

        Returns:
            Configured steer repetition penalty for steer-token request kinds,
            otherwise `None`.
        """

        if request_kind in STEER_REPETITION_REQUEST_KINDS:
            return self.branching.steer_repetition_penalty
        return None

    def _priority_value_for_branch_path(self, *, branch_path: tuple[int, ...]) -> int:
        """Encode one dotted branch path as sortable integer priority."""

        assert branch_path, "branch path must contain at least one segment"
        max_component = max(branch_path)
        radix = max(self.branching.branch_fanout + 1, max_component + 1, 2)
        max_depth = max(
            self.branching.max_branch_points_per_rollout + 1,
            len(branch_path),
        )
        padded = branch_path + (0,) * (max_depth - len(branch_path))
        priority_value = 0
        for component in padded:
            assert component >= 0, "branch path values must be non-negative"
            priority_value = (priority_value * radix) + component
        return priority_value

    def _branch_path_for_node_id(self, *, node_id: str) -> tuple[int, ...] | None:
        """Derive dotted branch path from runtime node id format."""

        if node_id == "node_root":
            return (1,)
        branch_suffix: list[int] = []
        current_node_id = node_id
        visited: set[str] = set()
        while current_node_id != "node_root":
            assert current_node_id not in visited, "node id lineage must be acyclic"
            visited.add(current_node_id)
            parsed = self._parent_node_and_child_offset(node_id=current_node_id)
            if parsed is None:
                return None
            parent_node_id, child_offset = parsed
            branch_suffix.append(child_offset + 1)
            current_node_id = parent_node_id
        branch_suffix.reverse()
        return (1, *branch_suffix)

    @staticmethod
    def _node_id_from_request_stream_id(*, request_stream_id: str) -> str | None:
        """Extract node id from decode/candidate_pool request stream id."""

        if request_stream_id.startswith("decode:"):
            node_id = request_stream_id.split(":", maxsplit=1)[1]
            return node_id if node_id else None
        if request_stream_id.startswith("candidate_pool:"):
            parts = request_stream_id.split(":", maxsplit=3)
            if len(parts) < 2:
                return None
            node_id = parts[1]
            return node_id if node_id else None
        return None

    @staticmethod
    def _parent_node_and_child_offset(*, node_id: str) -> tuple[str, int] | None:
        """Parse parent node id and child offset from canonical child node id."""

        matched = NODE_CHILD_ID_PATTERN.match(node_id)
        if matched is None:
            return None
        return (
            matched.group("parent_node_id"),
            int(matched.group("child_offset")),
        )

    def _resolve_prefix_base_delta(
        self,
        *,
        request_stream_id: str,
        current_input_token_ids: tuple[int, ...],
        prefix_chain_enabled: bool,
    ) -> tuple[tuple[int, ...], tuple[int, ...], str | None]:
        if not prefix_chain_enabled:
            return (), current_input_token_ids, None
        previous_state = self._request_stream_state.get(request_stream_id)
        if previous_state is None:
            return (), current_input_token_ids, None
        base = previous_state.input_token_ids + previous_state.output_token_ids
        assert current_input_token_ids[: len(base)] == base, (
            f"request stream prefix mismatch for {request_stream_id}: "
            f"expected base token prefix length {len(base)}, "
            f"got input length {len(current_input_token_ids)}"
        )
        delta = current_input_token_ids[len(base) :]
        return base, delta, previous_state.request_id

    def _append_vllm_request_event(
        self,
        *,
        request_id: str,
        request_stream_id: str,
        prev_request_id: str | None,
        request_kind: str,
        assistant_prefix: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        n: int,
        seed: int,
        stop: tuple[str, ...] | None,
        top_logprobs: int,
        current_input_token_ids: tuple[int, ...],
        base_prefix_token_ids: tuple[int, ...],
        delta_input_token_ids: tuple[int, ...],
        request_priority_value: int | None = None,
        request_branch_number: str | None = None,
        repetition_penalty: float | None = None,
    ) -> None:
        context = self._require_event_context()
        self.artifact_store.append_event(
            context=context,
            event_type="vllm_request",
            payload={
                "request_id": request_id,
                "request_stream_id": request_stream_id,
                "prev_request_id": prev_request_id,
                "request_kind": request_kind,
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
                "n": n,
                "seed": seed,
                "stop": list(stop) if stop is not None else None,
                "top_logprobs": top_logprobs,
                "request_priority": request_priority_value,
                "request_branch_number": request_branch_number,
                "repetition_penalty": repetition_penalty,
                "current_input_token_count": len(current_input_token_ids),
                "base_prefix_token_count": len(base_prefix_token_ids),
                "delta_token_count": len(delta_input_token_ids),
                "delta_input_token_ids": list(delta_input_token_ids),
                "assistant_prefix_tail": assistant_prefix[-200:],
            },
        )

    def _append_vllm_response_event(
        self,
        *,
        request_id: str,
        request_stream_id: str,
        request_kind: str,
        latency_seconds: float,
        choices: tuple[GenerationChoice, ...],
    ) -> None:
        context = self._require_event_context()
        self.artifact_store.append_event(
            context=context,
            event_type="vllm_response",
            payload={
                "request_id": request_id,
                "request_stream_id": request_stream_id,
                "request_kind": request_kind,
                "status": "ok",
                "latency_seconds": latency_seconds,
                "choice_count": len(choices),
                "choices": [
                    self._serialize_choice_for_vllm_event(choice=choice)
                    for choice in choices
                ],
            },
        )

    def _append_vllm_response_error_event(
        self,
        *,
        request_id: str,
        request_stream_id: str,
        request_kind: str,
        error_message: str,
        latency_seconds: float,
    ) -> None:
        context = self._require_event_context()
        self.artifact_store.append_event(
            context=context,
            event_type="vllm_response",
            payload={
                "request_id": request_id,
                "request_stream_id": request_stream_id,
                "request_kind": request_kind,
                "status": "error",
                "error_message": error_message,
                "latency_seconds": latency_seconds,
                "choices": [],
            },
        )

    def _serialize_choice_for_vllm_event(
        self, *, choice: GenerationChoice
    ) -> dict[str, Any]:
        token_ids = tuple(choice.token_ids or ())
        token_rows = []
        for token_index, parsed_token in enumerate(choice.tokens):
            token_id = token_ids[token_index] if token_index < len(token_ids) else None
            probability, entropy, _ = approximate_entropy(
                selected_token=parsed_token.token,
                selected_logprob=parsed_token.logprob,
                top_entries=parsed_token.top_entries,
            )
            token_rows.append(
                {
                    "token_index": token_index,
                    "token_id": token_id,
                    "token_text": parsed_token.token,
                    "selected_logprob": parsed_token.logprob,
                    "selected_probability": probability,
                    "selected_entropy": entropy,
                    "top_logprob_alternatives": self._top_logprob_alternatives(
                        parsed_token=parsed_token
                    ),
                }
            )
        return {
            "index": choice.index,
            "text": choice.text,
            "token_ids": list(token_ids),
            "finish_reason": choice.finish_reason,
            "stop_reason": choice.stop_reason,
            "output_token_count": len(token_ids),
            "tokens": token_rows,
        }

    @staticmethod
    def _serialize_candidate_for_pool_event(
        *, candidate: CandidateRecord
    ) -> dict[str, Any]:
        """Serialize candidate-pool row for rich runtime event inspection.

        Args:
            candidate: Candidate continuation generated at a branch point.

        Returns:
            JSON-ready candidate payload with token-level probability stats.
        """

        return {
            "candidate_id": candidate.candidate_id,
            "text": candidate.text,
            "token_ids": list(candidate.token_ids),
            "output_token_count": len(candidate.token_ids),
            "finish_reason": candidate.finish_reason,
            "stop_reason": candidate.stop_reason,
            "tokens": [
                {
                    "token_index": token.token_index,
                    "token_id": token.token_id,
                    "token_text": token.token_text,
                    "selected_logprob": token.logprob,
                    "selected_probability": token.probability,
                    "selected_entropy": token.entropy,
                    "top_logprob_alternatives": [],
                }
                for token in candidate.tokens
            ],
        }

    @staticmethod
    def _top_logprob_alternatives(*, parsed_token: ParsedToken) -> list[dict[str, Any]]:
        """Return at most four alternate top-logprob rows.

        Args:
            parsed_token: Parsed token row containing selected token and top entries.

        Returns:
            Up to four alternate token/logprob rows excluding the selected token.
        """

        alternatives: list[dict[str, Any]] = []
        for alt_token, alt_logprob in parsed_token.top_entries:
            if alt_token == parsed_token.token:
                continue
            alternatives.append({"token_text": alt_token, "logprob": alt_logprob})
            if len(alternatives) >= MAX_LOGPROB_ALTERNATIVES:
                break
        return alternatives

    @staticmethod
    def _cluster_assignment_rows(
        *, selection: SelectionOutcome
    ) -> list[dict[str, Any]]:
        """Serialize candidate-level cluster assignment rows for one selector.

        Args:
            selection: Selection outcome containing optional cluster assignments.

        Returns:
            Sorted assignment rows with `candidate_id` and `cluster_name`.
        """

        if selection.cluster_by_candidate_id is None:
            return []
        rows: list[dict[str, Any]] = []
        for candidate_id in sorted(selection.cluster_by_candidate_id):
            cluster_name = str(selection.cluster_by_candidate_id[candidate_id])
            rows.append(
                {"candidate_id": int(candidate_id), "cluster_name": cluster_name}
            )
        return rows

    @staticmethod
    def _cluster_group_rows(*, selection: SelectionOutcome) -> list[dict[str, Any]]:
        """Serialize grouped cluster rows for one selector outcome.

        Args:
            selection: Selection outcome containing optional cluster assignments.

        Returns:
            Cluster summary rows with candidate ids and selected ids per cluster.
        """

        assignment = selection.cluster_by_candidate_id
        if assignment is None:
            return []
        grouped: dict[str, list[int]] = {}
        for candidate_id, cluster_name in assignment.items():
            grouped.setdefault(str(cluster_name), []).append(int(candidate_id))
        selected_set = set(selection.selected_candidate_ids)
        rows: list[dict[str, Any]] = []
        for cluster_name in sorted(grouped):
            candidate_ids = sorted(grouped[cluster_name])
            selected_ids = [
                candidate_id
                for candidate_id in candidate_ids
                if candidate_id in selected_set
            ]
            rows.append(
                {
                    "cluster_name": cluster_name,
                    "candidate_ids": candidate_ids,
                    "candidate_count": len(candidate_ids),
                    "selected_candidate_ids": selected_ids,
                    "selected_candidate_count": len(selected_ids),
                }
            )
        return rows

    async def _tokenize_text_async(self, *, text: str) -> tuple[int, ...]:
        """Tokenize text with async tokenizer endpoint under request semaphore."""

        tokenize_async = getattr(self.client, "tokenize_async", None)
        assert tokenize_async is not None, "client must provide tokenize_async"
        semaphore = self._ensure_request_semaphore()
        async with semaphore:
            token_ids = await tokenize_async(
                model=self.model_name,
                text=text,
                add_special_tokens=False,
            )
        return tuple(token_ids)

    async def _detokenize_ids_async(self, *, token_ids: tuple[int, ...]) -> str | None:
        """Detokenize token IDs with async endpoint under request semaphore."""

        if not token_ids:
            return ""
        detokenize_async = getattr(self.client, "detokenize_async", None)
        if detokenize_async is None:
            return None
        semaphore = self._ensure_request_semaphore()
        async with semaphore:
            decoded_text = await detokenize_async(
                model=self.model_name,
                token_ids=token_ids,
            )
        return str(decoded_text)

    def _ensure_request_semaphore(self) -> asyncio.Semaphore:
        """Return shared async request semaphore for this executor instance."""

        running_loop_id = id(asyncio.get_running_loop())
        if (
            self._request_semaphore is None
            or self._request_semaphore_loop_id != running_loop_id
        ):
            self._request_semaphore = asyncio.Semaphore(
                self.max_async_inflight_requests
            )
            self._request_semaphore_loop_id = running_loop_id
        return self._request_semaphore
