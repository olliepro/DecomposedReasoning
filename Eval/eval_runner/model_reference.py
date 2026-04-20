"""Helpers for distinguishing local checkpoint paths from HF repo ids."""

from __future__ import annotations

from pathlib import Path
from typing import Union

ModelReference = Union[str, Path]


def resolve_local_model_path(model_reference: ModelReference) -> Path | None:
    """Resolve an existing local model path when the reference is filesystem-backed.

    Args:
        model_reference: Local checkpoint path or remote model id.

    Returns:
        Resolved local path when it exists, otherwise `None`.
    """

    candidate_path = Path(str(model_reference)).expanduser()
    if not candidate_path.exists():
        return None
    return candidate_path.resolve()


def resolve_pretrained_arg(model_reference: ModelReference) -> str:
    """Build the `pretrained` argument for `lm-eval` model loading.

    Args:
        model_reference: Local checkpoint path or remote model id.

    Returns:
        Absolute local checkpoint path when present, else the original repo id.
    """

    local_path = resolve_local_model_path(model_reference=model_reference)
    if local_path is not None:
        return str(local_path)
    return str(model_reference)


def model_reference_name(model_reference: ModelReference) -> str:
    """Return a display-friendly terminal name for a model reference.

    Args:
        model_reference: Local checkpoint path or remote model id.

    Returns:
        Final path component or repo-name suffix.
    """

    return Path(str(model_reference)).name
