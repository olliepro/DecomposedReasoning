"""Progress-row helpers for the dynamic branching viewer."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProgressAttemptView:
    """Lightweight per-doc summary row sourced from progress snapshots."""

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

    def slug(self) -> str:
        """Return filesystem-safe page slug for this progress row."""

        return (
            f"doc_{self.doc_id}_attempt_{self.doc_attempt}_"
            f"{self.task_name}_{self.model_id}_{self.selector_mode}"
        ).replace("/", "_")

    def label(self) -> str:
        """Return concise display label."""

        return f"doc {self.doc_id} · attempt {self.doc_attempt}"


def selected_progress_attempts_by_doc(
    *, progress_attempts: list[ProgressAttemptView]
) -> list[ProgressAttemptView]:
    """Select one preferred progress row per doc."""

    selected_by_doc: dict[int, ProgressAttemptView] = {}
    for progress in progress_attempts:
        existing = selected_by_doc.get(progress.doc_id)
        if existing is None:
            selected_by_doc[progress.doc_id] = progress
            continue
        progress_rank = (
            1 if progress.status in {"complete", "completed"} else 0,
            progress.doc_attempt,
            progress.last_update_timestamp,
        )
        existing_rank = (
            1 if existing.status in {"complete", "completed"} else 0,
            existing.doc_attempt,
            existing.last_update_timestamp,
        )
        if progress_rank > existing_rank:
            selected_by_doc[progress.doc_id] = progress
    return [selected_by_doc[doc_id] for doc_id in sorted(selected_by_doc)]
