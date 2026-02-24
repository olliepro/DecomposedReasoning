"""Bridge helpers that reuse AIME scoring utilities from `Eval/eval_runner`."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any


def verify_aime_response(*, doc: dict[str, Any], response_text: str) -> int:
    """Score one AIME response as a binary verification label.

    Args:
        doc: AIME task document payload.
        response_text: Model response text.

    Returns:
        `1` when response matches target answer, else `0`.

    Example:
        >>> verify_aime_response(doc={"Answer": "42"}, response_text="42")
        1
    """

    module = _load_aime_module()
    target = module.extract_target_answer(doc)
    candidate = module.extract_candidate_answer(response_text)
    return int(module.is_correct_answer(candidate=candidate, target=target))


def _load_aime_module() -> Any:
    """Load `Eval/eval_runner/aime_avgk.py` as importable module.

    Args:
        None.

    Returns:
        Imported module object.
    """

    eval_dir = _resolve_eval_dir()
    eval_dir_text = str(eval_dir)
    if eval_dir_text not in sys.path:
        sys.path.insert(0, eval_dir_text)
    return importlib.import_module("eval_runner.aime_avgk")


def _resolve_eval_dir() -> Path:
    current = Path(__file__).resolve()
    repo_root = current.parent.parent.parent
    eval_dir = repo_root / "Eval"
    assert eval_dir.exists(), f"Eval directory not found: {eval_dir}"
    return eval_dir
