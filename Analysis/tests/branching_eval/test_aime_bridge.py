"""Tests for AIME bridge utility parity with Eval scoring helpers."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

from branching_eval.aime_bridge import verify_aime_response


def test_aime_bridge_matches_eval_utils_for_boxed_answer() -> None:
    """Bridge verification should match Eval utility behavior."""

    eval_dir = Path(__file__).resolve().parents[3] / "Eval"
    eval_dir_text = str(eval_dir)
    if eval_dir_text not in sys.path:
        sys.path.insert(0, eval_dir_text)
    aime_avgk = importlib.import_module("eval_runner.aime_avgk")

    doc = {"Answer": "42"}
    response = "The answer is \\boxed{42}."
    expected = aime_avgk.is_correct_answer(
        candidate=aime_avgk.extract_candidate_answer(response),
        target=aime_avgk.extract_target_answer(doc),
    )
    actual = verify_aime_response(doc=doc, response_text=response)
    assert actual == expected == 1
