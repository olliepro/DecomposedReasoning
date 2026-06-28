"""Tests for fake KV block accounting."""

from __future__ import annotations

from vllm_experimental.kv_accounting import (
    FakeBlockPool,
    fork_blocks_for_branch,
    free_loser_blocks,
)


def test_fork_aligned_boundary_shares_full_blocks() -> None:
    pool = FakeBlockPool(refcounts={10: 1, 11: 1})
    child_blocks, copies = fork_blocks_for_branch(
        parent_blocks=(10, 11),
        computed_tokens=32,
        block_size=16,
        pool=pool,
    )
    assert child_blocks == (10, 11)
    assert copies == ()
    assert pool.refcounts == {10: 2, 11: 2}


def test_fork_partial_tail_copies_tail_block() -> None:
    pool = FakeBlockPool(refcounts={10: 1, 11: 1})
    child_blocks, copies = fork_blocks_for_branch(
        parent_blocks=(10, 11),
        computed_tokens=20,
        block_size=16,
        pool=pool,
    )
    assert child_blocks == (10, 1000)
    assert [(copy.src_block_id, copy.dst_block_id) for copy in copies] == [(11, 1000)]
    assert pool.refcounts == {10: 2, 11: 1, 1000: 1}


def test_fork_partial_tail_can_recompute_tail() -> None:
    pool = FakeBlockPool(refcounts={10: 1, 11: 1})
    child_blocks, copies = fork_blocks_for_branch(
        parent_blocks=(10, 11),
        computed_tokens=20,
        block_size=16,
        pool=pool,
        copy_partial_tail=False,
    )
    assert child_blocks == (10,)
    assert copies == ()
    assert pool.refcounts == {10: 2, 11: 1}


def test_free_loser_blocks() -> None:
    pool = FakeBlockPool(refcounts={10: 2, 1000: 1})
    free_loser_blocks(block_ids=(10, 1000), pool=pool)
    assert pool.refcounts == {10: 1}
    assert pool.freed_blocks == [1000]
