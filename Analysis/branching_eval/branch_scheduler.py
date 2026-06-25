"""Streaming frontier scheduler for branching rollouts."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Protocol, cast

from branching_eval.runtime_types import DecodeOutcome, PathState
from branching_eval.tree_types import BranchTree


@dataclass(frozen=True)
class ScheduledDecode:
    """One queued decode state with branching controls."""

    state: PathState
    branching_enabled: bool = True
    steer_normalization_enabled: bool | None = None
    inline_epsilon_enabled: bool | None = None


@dataclass(frozen=True)
class ScheduledExpansion:
    """One queued branch expansion for a completed trigger outcome."""

    state: PathState
    outcome: DecodeOutcome
    doc_id: int


class StreamingSchedulerExecutor(Protocol):
    """Executor callbacks required by the streaming frontier scheduler."""

    def _schedule_decode_task(
        self, *, tree: BranchTree, scheduled: ScheduledDecode
    ) -> asyncio.Task[tuple[ScheduledDecode, DecodeOutcome]]: ...

    async def _handle_state_outcome(
        self,
        *,
        tree: BranchTree,
        scheduled: ScheduledDecode,
        outcome: DecodeOutcome,
        doc_id: int,
        leaf_limit: int,
    ) -> tuple[list[ScheduledDecode], ScheduledExpansion | None]: ...

    def _schedule_expansion_task(
        self, *, tree: BranchTree, scheduled: ScheduledExpansion
    ) -> asyncio.Task[list[ScheduledDecode]]: ...

    async def _cancel_pending_scheduler_tasks(
        self,
        *,
        pending_decode: set[asyncio.Task[tuple[ScheduledDecode, DecodeOutcome]]],
        pending_expansion: set[asyncio.Task[list[ScheduledDecode]]],
    ) -> None: ...


async def decode_frontier_streaming_async(
    *,
    executor: StreamingSchedulerExecutor,
    tree: BranchTree,
    frontier: list[PathState],
    doc_id: int,
    leaf_limit: int,
    initial_scheduled: list[ScheduledDecode] | None = None,
) -> None:
    """Decode frontier states as they complete and enqueue children immediately."""

    scheduled_frontier = (
        [ScheduledDecode(state=state) for state in frontier]
        if initial_scheduled is None
        else initial_scheduled
    )
    pending_decode: set[asyncio.Task[tuple[ScheduledDecode, DecodeOutcome]]] = {
        executor._schedule_decode_task(tree=tree, scheduled=scheduled)
        for scheduled in scheduled_frontier
    }
    pending_expansion: set[asyncio.Task[list[ScheduledDecode]]] = set()
    try:
        while (pending_decode or pending_expansion) and len(tree.leaves) < leaf_limit:
            waiting: set[asyncio.Task[Any]] = set(pending_decode)
            waiting.update(pending_expansion)
            done, _ = await asyncio.wait(waiting, return_when=asyncio.FIRST_COMPLETED)
            for completed in done:
                if completed in pending_decode:
                    pending_decode.remove(
                        cast(
                            asyncio.Task[tuple[ScheduledDecode, DecodeOutcome]],
                            completed,
                        )
                    )
                    scheduled, outcome = completed.result()
                    next_scheduled, scheduled_expansion = (
                        await executor._handle_state_outcome(
                            tree=tree,
                            scheduled=scheduled,
                            outcome=outcome,
                            doc_id=doc_id,
                            leaf_limit=leaf_limit,
                        )
                    )
                    if len(tree.leaves) >= leaf_limit:
                        break
                    for child in next_scheduled:
                        pending_decode.add(
                            executor._schedule_decode_task(tree=tree, scheduled=child)
                        )
                    if scheduled_expansion is not None:
                        pending_expansion.add(
                            executor._schedule_expansion_task(
                                tree=tree,
                                scheduled=scheduled_expansion,
                            )
                        )
                    continue
                pending_expansion.remove(
                    cast(asyncio.Task[list[ScheduledDecode]], completed)
                )
                next_scheduled = completed.result()
                if len(tree.leaves) >= leaf_limit:
                    break
                for child in next_scheduled:
                    pending_decode.add(
                        executor._schedule_decode_task(tree=tree, scheduled=child)
                    )
            if len(tree.leaves) >= leaf_limit:
                break
    finally:
        await executor._cancel_pending_scheduler_tasks(
            pending_decode=pending_decode,
            pending_expansion=pending_expansion,
        )
