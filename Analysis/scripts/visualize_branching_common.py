"""Shared types and text helpers for the dynamic branching viewer."""

from __future__ import annotations

from dataclasses import dataclass

from candidate_clustering import strip_steer_suffix


@dataclass(frozen=True)
class AttemptKey:
    """Identity key for one document attempt stream."""

    doc_id: int
    doc_attempt: int
    task_name: str
    model_id: str
    selector_mode: str

    def slug(self) -> str:
        """Return filesystem-safe slug for this attempt key."""

        return (
            f"doc_{self.doc_id}_attempt_{self.doc_attempt}_"
            f"{self.task_name}_{self.model_id}_{self.selector_mode}"
        ).replace("/", "_")

    def label(self) -> str:
        """Return concise display label."""

        return f"doc {self.doc_id} · attempt {self.doc_attempt}"


def clean_candidate_preview(*, text: str, max_chars: int) -> str:
    """Collapse multi-line text into a compact one-line preview."""

    cleaned_text = strip_steer_suffix(text=str(text))
    collapsed = " ".join(cleaned_text.replace("\t", " ").split())
    if not collapsed:
        return "(empty)"
    if len(collapsed) <= max_chars:
        return collapsed
    return f"{collapsed[: max_chars - 3]}..."
