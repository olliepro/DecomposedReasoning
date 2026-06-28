"""Fake KV fork/promote accounting used before GPU integration tests."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KVBlockCopy:
    """One device-side block copy command."""

    src_block_id: int
    dst_block_id: int


@dataclass
class FakeBlockPool:
    """Small block-pool model for unit tests."""

    next_block_id: int = 1000
    refcounts: dict[int, int] | None = None
    freed_blocks: list[int] | None = None

    def __post_init__(self) -> None:
        if self.refcounts is None:
            self.refcounts = {}
        if self.freed_blocks is None:
            self.freed_blocks = []

    def touch(self, block_id: int) -> None:
        """Increment one block refcount."""

        assert self.refcounts is not None
        self.refcounts[block_id] = self.refcounts.get(block_id, 0) + 1

    def allocate(self) -> int:
        """Allocate one new block id."""

        block_id = self.next_block_id
        self.next_block_id += 1
        self.touch(block_id=block_id)
        return block_id

    def free(self, block_id: int) -> None:
        """Decrement or record final free for one block."""

        assert self.refcounts is not None
        assert self.freed_blocks is not None
        current = self.refcounts.get(block_id, 0)
        assert current > 0, f"block {block_id} was freed too often"
        if current == 1:
            del self.refcounts[block_id]
            self.freed_blocks.append(block_id)
            return
        self.refcounts[block_id] = current - 1


def fork_blocks_for_branch(
    *,
    parent_blocks: tuple[int, ...],
    computed_tokens: int,
    block_size: int,
    pool: FakeBlockPool,
    copy_partial_tail: bool = True,
) -> tuple[tuple[int, ...], tuple[KVBlockCopy, ...]]:
    """Share full blocks and optionally copy the partial tail block."""

    assert block_size >= 1, "block_size must be positive"
    full_block_count = computed_tokens // block_size
    tail_tokens = computed_tokens % block_size
    assert full_block_count <= len(parent_blocks), "parent block list is too short"

    child_blocks: list[int] = []
    for block_id in parent_blocks[:full_block_count]:
        pool.touch(block_id=block_id)
        child_blocks.append(block_id)

    copies: list[KVBlockCopy] = []
    if tail_tokens and copy_partial_tail:
        assert full_block_count < len(parent_blocks), "missing parent tail block"
        copied_tail = pool.allocate()
        child_blocks.append(copied_tail)
        copies.append(
            KVBlockCopy(
                src_block_id=parent_blocks[full_block_count],
                dst_block_id=copied_tail,
            )
        )
    return tuple(child_blocks), tuple(copies)


def free_loser_blocks(*, block_ids: tuple[int, ...], pool: FakeBlockPool) -> None:
    """Free all blocks owned by a losing branch."""

    for block_id in block_ids:
        pool.free(block_id=block_id)
