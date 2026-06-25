"""Lightweight compatibility helpers for repo-local branching extensions."""

from __future__ import annotations

import importlib.util
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

from branching_dapo.bootstrap import project_root


def register_adv_est(
    name_or_enum: str | Any,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Return the `verl` advantage-registration decorator or a test fallback.

    Args:
        name_or_enum: Advantage-estimator registry key.

    Returns:
        Decorator that registers the estimator in `verl` when available.
    """

    try:
        from verl.trainer.ppo.core_algos import (
            register_adv_est as verl_register_adv_est,
        )

        return verl_register_adv_est(name_or_enum)
    except ModuleNotFoundError:
        return _identity_register(name=name_or_enum)


def _identity_register(
    name: str | Any,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Return a no-op decorator for unit tests without full `verl` deps.

    Args:
        name: Unused registry key retained for call-shape compatibility.

    Returns:
        Decorator that returns the original function unchanged.
    """

    def decorator(function: Callable[..., Any]) -> Callable[..., Any]:
        _ = name
        return function

    return decorator


@lru_cache(maxsize=1)
def load_math_dapo_module() -> ModuleType:
    """Load the `math_dapo` reward scorer with or without full `verl` imports.

    Args:
        None.

    Returns:
        Imported module object exposing `compute_score`.
    """

    try:
        from verl.utils.reward_score import math_dapo

        return math_dapo
    except ModuleNotFoundError:
        module_path = _math_dapo_path()
        spec = importlib.util.spec_from_file_location(
            "branching_dapo_math_dapo", module_path
        )
        assert (
            spec is not None and spec.loader is not None
        ), f"Unable to load math_dapo module from {module_path}"
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


def _math_dapo_path() -> Path:
    """Return the vendored `verl` math-dapo module path.

    Args:
        None.

    Returns:
        Absolute path to `math_dapo.py`.
    """

    return (
        project_root()
        / "RLTraining"
        / "verl"
        / "verl"
        / "utils"
        / "reward_score"
        / "math_dapo.py"
    )
