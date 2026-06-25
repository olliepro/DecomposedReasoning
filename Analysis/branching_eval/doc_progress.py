"""Lightweight per-document progress snapshots for branching runs."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Callable

from branching_eval.tree_types import LeafRollout

AnswerExtractor = Callable[[str], str]
TERMINATION_BUCKETS = ("natural", "max", "repeating", "other")


def normalize_stop_reason_bucket(*, stop_reason: str) -> str:
    """Map one raw stop reason into a stable progress bucket.

    Args:
        stop_reason: Raw leaf stop reason.

    Returns:
        Bucket label in `natural|max|repeating|other`.

    Example:
        >>> normalize_stop_reason_bucket(stop_reason="think_end")
        'natural'
    """

    normalized = str(stop_reason).strip().lower()
    if normalized in {"think_end", "model_finished", "stop", "eos"}:
        return "natural"
    if normalized == "max_gen_toks_reached" or "length" in normalized:
        return "max"
    if normalized.startswith("repeated_") and normalized.endswith("_block_loop"):
        return "repeating"
    return "other"


@dataclass(frozen=True)
class DocProgressSnapshot:
    """Compact progress payload for one document attempt.

    Args:
        run_id: Stable run identifier.
        doc_id: Document identifier.
        doc_attempt: Attempt index within the run.
        task_name: Task label.
        model_id: Model label.
        selector_mode: Selector label used for this attempt.
        rollout_mode: Rollout mode label such as `baseline` or `branching`.
        status: Attempt status in `incomplete|complete`.
        leaf_count: Number of scored leaves observed so far.
        passrate: Fraction of scored leaves with verification `1`.
        avg_token_length: Mean `length_tokens_total` across scored leaves.
        correct_count: Number of verified-correct leaves.
        incorrect_count: Number of verified-incorrect leaves.
        natural_count: Count of naturally terminated leaves.
        max_count: Count of length-limited leaves.
        repeating_count: Count of repeat-loop terminated leaves.
        other_count: Count of uncategorized stop reasons.
        unique_answer_count: Count of distinct normalized extracted answers.
        last_update_timestamp: ISO timestamp of the latest snapshot write.

    Returns:
        Immutable per-doc progress snapshot.
    """

    run_id: str
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

    def filename(self) -> str:
        """Return filesystem-safe filename for this attempt snapshot."""

        return f"doc_{self.doc_id}_attempt_{self.doc_attempt}.json"

    def to_payload(self) -> dict[str, object]:
        """Return JSON-ready payload for persistence."""

        return {
            "run_id": self.run_id,
            "doc_id": self.doc_id,
            "doc_attempt": self.doc_attempt,
            "task_name": self.task_name,
            "model_id": self.model_id,
            "selector_mode": self.selector_mode,
            "rollout_mode": self.rollout_mode,
            "status": self.status,
            "leaf_count": self.leaf_count,
            "passrate": self.passrate,
            "avg_token_length": self.avg_token_length,
            "correct_count": self.correct_count,
            "incorrect_count": self.incorrect_count,
            "natural_count": self.natural_count,
            "max_count": self.max_count,
            "repeating_count": self.repeating_count,
            "other_count": self.other_count,
            "unique_answer_count": self.unique_answer_count,
            "last_update_timestamp": self.last_update_timestamp,
        }


@dataclass
class DocProgressTracker:
    """Mutable tracker that emits compact per-doc snapshots.

    Args:
        run_id: Stable run identifier.
        doc_id: Document identifier.
        doc_attempt: Attempt index within the run.
        task_name: Task label.
        model_id: Model label.
        selector_mode: Selector label used for this attempt.
        rollout_mode: Rollout mode label.
        answer_extractor: Canonical answer extractor for unique-answer counting.
        status: Current document status.
        leaves_by_id: Latest scored leaf rows keyed by `leaf_id`.

    Returns:
        Mutable tracker used while one doc attempt is executing.
    """

    run_id: str
    doc_id: int
    doc_attempt: int
    task_name: str
    model_id: str
    selector_mode: str
    rollout_mode: str
    answer_extractor: AnswerExtractor
    status: str = "incomplete"
    leaves_by_id: dict[str, LeafRollout] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        run_id: str,
        doc_id: int,
        doc_attempt: int,
        task_name: str,
        model_id: str,
        selector_mode: str,
        rollout_mode: str,
        answer_extractor: AnswerExtractor,
        existing_leaves: tuple[LeafRollout, ...] = (),
    ) -> DocProgressTracker:
        """Build a tracker seeded with already-scored leaves.

        Args:
            run_id: Stable run identifier.
            doc_id: Document identifier.
            doc_attempt: Attempt index within the run.
            task_name: Task label.
            model_id: Model label.
            selector_mode: Selector label used for this attempt.
            rollout_mode: Rollout mode label.
            answer_extractor: Canonical answer extractor for unique-answer counting.
            existing_leaves: Already-scored leaves replayed during resume.

        Returns:
            Seeded progress tracker.
        """

        leaves_by_id = {leaf.leaf_id: leaf for leaf in existing_leaves}
        return cls(
            run_id=run_id,
            doc_id=doc_id,
            doc_attempt=doc_attempt,
            task_name=task_name,
            model_id=model_id,
            selector_mode=selector_mode,
            rollout_mode=rollout_mode,
            answer_extractor=answer_extractor,
            leaves_by_id=leaves_by_id,
        )

    def record_leaf(self, *, leaf: LeafRollout) -> None:
        """Record or replace one scored leaf in the tracker."""

        self.leaves_by_id[leaf.leaf_id] = leaf

    def mark_complete(self) -> None:
        """Mark this document attempt as complete."""

        self.status = "complete"

    def snapshot(self, *, last_update_timestamp: str) -> DocProgressSnapshot:
        """Return a compact immutable snapshot for the current tracker state.

        Args:
            last_update_timestamp: ISO timestamp for this snapshot.

        Returns:
            Immutable progress snapshot.
        """

        leaves = tuple(self.leaves_by_id.values())
        leaf_count = len(leaves)
        correct_count = sum(1 for leaf in leaves if leaf.verification == 1)
        incorrect_count = sum(1 for leaf in leaves if leaf.verification == 0)
        passrate = (correct_count / leaf_count) if leaf_count else 0.0
        avg_token_length = (
            sum(float(leaf.length_tokens_total) for leaf in leaves) / leaf_count
            if leaf_count
            else 0.0
        )
        bucket_counts = Counter(
            normalize_stop_reason_bucket(stop_reason=leaf.stop_reason)
            for leaf in leaves
        )
        unique_answers = {
            extracted
            for extracted in (
                self.answer_extractor(leaf.text).strip() for leaf in leaves
            )
            if extracted
        }
        return DocProgressSnapshot(
            run_id=self.run_id,
            doc_id=self.doc_id,
            doc_attempt=self.doc_attempt,
            task_name=self.task_name,
            model_id=self.model_id,
            selector_mode=self.selector_mode,
            rollout_mode=self.rollout_mode,
            status=self.status,
            leaf_count=leaf_count,
            passrate=passrate,
            avg_token_length=avg_token_length,
            correct_count=correct_count,
            incorrect_count=incorrect_count,
            natural_count=bucket_counts["natural"],
            max_count=bucket_counts["max"],
            repeating_count=bucket_counts["repeating"],
            other_count=bucket_counts["other"],
            unique_answer_count=len(unique_answers),
            last_update_timestamp=last_update_timestamp,
        )
