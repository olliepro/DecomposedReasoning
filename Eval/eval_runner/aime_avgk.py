"""Custom AIME avg@k metric utilities."""

from __future__ import annotations

from typing import Any, Callable

from lm_eval.tasks.aime import utils as aime_utils


def extract_target_answer(doc: dict[str, Any]) -> str:
    """Extract the reference answer string from one AIME document.

    Args:
        doc: AIME task document payload.

    Returns:
        Canonical target answer string.
    """
    answer_key = next((key for key in doc if key.lower() == "answer"), None)
    assert answer_key is not None, "AIME doc is missing an answer field."
    return str(doc[answer_key])


def _extract_dollar_wrapped_answer(response: str) -> str:
    """Extract answer text delimited by `$...$` from one response.

    Args:
        response: Model-generated response text.

    Returns:
        Extracted candidate answer from dollar delimiters when present.
    """
    indices = [position for position, char in enumerate(response) if char == "$"]
    if len(indices) <= 1:
        return response
    return response[indices[0] + 1 : indices[-1]]


def extract_candidate_answer(response: str) -> str:
    """Extract candidate answer text from one model response.

    Args:
        response: Model-generated response text.

    Returns:
        Parsed candidate answer string.
    """
    candidate = _extract_dollar_wrapped_answer(response=response)
    boxed_answer = aime_utils.last_boxed_only_string(response)
    if boxed_answer is None:
        return candidate
    try:
        boxed_content = aime_utils.remove_boxed(boxed_answer)
    except (AssertionError, IndexError):
        return candidate
    return candidate if boxed_content is None else str(boxed_content)


def is_correct_answer(candidate: str, target: str) -> int:
    """Score one candidate answer against one target answer.

    Args:
        candidate: Parsed candidate answer string.
        target: Target answer string.

    Returns:
        `1` for a correct answer and `0` otherwise.
    """
    return int(aime_utils.is_equiv(candidate, target))


def compute_avg_at_k(responses: list[str], target: str) -> float:
    """Compute per-question mean correctness over `k` sampled responses.

    Args:
        responses: Candidate responses after `take_first_k` filtering.
        target: Ground-truth answer string.

    Returns:
        Mean correctness score in `[0.0, 1.0]`.
    """
    assert responses, "avg@k requires at least one response."
    scores = [
        is_correct_answer(
            candidate=extract_candidate_answer(response=response),
            target=target,
        )
        for response in responses
    ]
    return float(sum(scores) / len(scores))


def _extract_responses(results: list[Any]) -> list[str]:
    """Normalize `lm-eval` result payload into a list of response strings.

    Args:
        results: Raw response payload passed into task `process_results`.

    Returns:
        List of response strings for scoring.
    """
    assert results, "Expected at least one task result entry."
    first_result = results[0]
    if isinstance(first_result, list):
        return [str(response) for response in first_result]
    return [str(first_result)]


def build_aime_avgk_process_results(
    metric_name: str,
) -> Callable[[dict[str, Any], list[Any]], dict[str, float]]:
    """Build an AIME `process_results` function that outputs a custom avg@k metric.

    Args:
        metric_name: Output metric key, typically `avg_at_<k>`.

    Returns:
        Callable that `lm-eval` uses as task-local `process_results`.

    Example:
        >>> process_results = build_aime_avgk_process_results(metric_name="avg_at_32")
        >>> sorted(process_results(doc={"Answer": "42"}, results=[["42"]]).keys())
        ['avg_at_32']
    """
    assert metric_name.startswith("avg_at_"), "Expected metric name format: avg_at_<k>."

    def process_results(doc: dict[str, Any], results: list[Any]) -> dict[str, float]:
        responses = _extract_responses(results=results)
        target = extract_target_answer(doc=doc)
        score = compute_avg_at_k(responses=responses, target=target)
        return {metric_name: score}

    return process_results
