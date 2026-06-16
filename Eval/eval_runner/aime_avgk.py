"""Custom AIME avg@k metric utilities."""

from __future__ import annotations

import re
from typing import Any, Callable


ANSWER_MARKER_PREFIX_RE = re.compile(
    r"(?is)(?:final\s+answer|the\s+answer\s+is|answer)\s*(?:[:=]|is)?\s*"
)
INTEGER_RE = re.compile(r"(?<![\w.])-?\d+(?!\.\d)(?![\w])")
SPECIAL_TOKEN_RE = re.compile(r"<\|[^|]+?\|>")


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


def last_boxed_only_string(response: str) -> str | None:
    """Extract the final boxed segment from one response when present.

    Args:
        response: Model-generated response text.

    Returns:
        Final boxed substring or `None`.
    """

    boxed_index = response.rfind("\\boxed")
    if "\\boxed " in response:
        return "\\boxed " + response.split("\\boxed ")[-1].split("$")[0]
    if boxed_index < 0:
        return None
    brace_balance = 0
    right_brace_index = None
    for position in range(boxed_index, len(response)):
        if response[position] == "{":
            brace_balance += 1
        elif response[position] == "}":
            brace_balance -= 1
            if brace_balance == 0:
                right_brace_index = position
                break
    if right_brace_index is None:
        return None
    return response[boxed_index : right_brace_index + 1]


def remove_boxed(boxed_value: str) -> str:
    """Remove the outer `\\boxed` wrapper from one extracted answer.

    Args:
        boxed_value: Boxed answer string.

    Returns:
        Inner boxed content.
    """

    if boxed_value.startswith("\\boxed "):
        return boxed_value[len("\\boxed ") :]
    prefix = "\\boxed{"
    assert boxed_value.startswith(prefix), f"Unexpected boxed answer: {boxed_value}"
    assert boxed_value.endswith("}"), f"Unexpected boxed answer: {boxed_value}"
    return boxed_value[len(prefix) : -1]


def normalize_answer_text(value: str) -> str:
    """Normalize one short AIME answer string before equality comparison.

    Args:
        value: Candidate or target answer string.

    Returns:
        Normalized answer string.
    """

    normalized = value.strip()
    normalized = normalized.replace("$", "")
    normalized = normalized.replace(",", "")
    normalized = normalized.replace("\\!", "")
    normalized = normalized.replace("\\,", "")
    normalized = normalized.replace(" ", "")
    normalized = normalized.rstrip(".")
    if re.fullmatch(r"-?\d+", normalized):
        sign = "-" if normalized.startswith("-") else ""
        digits = normalized[1:] if sign else normalized
        normalized = sign + (digits.lstrip("0") or "0")
    return normalized


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
    return response[indices[-2] + 1 : indices[-1]]


def _strip_response_special_tokens(response: str) -> str:
    """Remove model control tokens that should not affect answer parsing."""

    stripped = SPECIAL_TOKEN_RE.sub("", response)
    return stripped.strip()


def _extract_marker_answer(response: str) -> str | None:
    """Extract the immediate short numeric answer after the final answer marker."""

    matches = list(ANSWER_MARKER_PREFIX_RE.finditer(response))
    if not matches:
        return None
    tail = response[matches[-1].end() :]
    boxed_answer = last_boxed_only_string(response=tail)
    if boxed_answer is not None:
        try:
            return remove_boxed(boxed_value=boxed_answer)
        except (AssertionError, IndexError):
            pass
    integer_matches = list(INTEGER_RE.finditer(tail))
    if integer_matches:
        return integer_matches[0].group(0)
    dollar_wrapped = _extract_dollar_wrapped_answer(response=tail)
    if dollar_wrapped != tail:
        return dollar_wrapped
    return tail


def extract_candidate_answer(response: str) -> str:
    """Extract candidate answer text from one model response.

    Args:
        response: Model-generated response text.

    Returns:
        Parsed candidate answer string.
    """
    cleaned_response = _strip_response_special_tokens(response=response)
    boxed_answer = last_boxed_only_string(response=cleaned_response)
    if boxed_answer is None:
        marker_answer = _extract_marker_answer(response=cleaned_response)
        if marker_answer is not None:
            return marker_answer
        return _extract_dollar_wrapped_answer(response=cleaned_response)
    try:
        boxed_content = remove_boxed(boxed_value=boxed_answer)
    except (AssertionError, IndexError):
        marker_answer = _extract_marker_answer(response=cleaned_response)
        if marker_answer is not None:
            return marker_answer
        return _extract_dollar_wrapped_answer(response=cleaned_response)
    return str(boxed_content)


def is_correct_answer(candidate: str, target: str) -> int:
    """Score one candidate answer against one target answer.

    Args:
        candidate: Parsed candidate answer string.
        target: Target answer string.

    Returns:
        `1` for a correct answer and `0` otherwise.
    """
    return int(
        normalize_answer_text(value=candidate) == normalize_answer_text(value=target)
    )


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
