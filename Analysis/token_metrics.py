"""Probability and entropy calculations for token-level analysis."""

from __future__ import annotations

import math
from typing import Iterable

from analysis_types import TokenAlternative


def probability_from_logprob(*, logprob: float) -> float:
    """Convert natural log probability to linear probability.

    Args:
        logprob: Natural log probability value.

    Returns:
        Probability in `[0, 1]`.
    """
    if logprob <= -1e9:
        return 0.0
    return float(math.exp(logprob))


def canonicalize_top_logprobs(*, raw_top_logprobs: object) -> list[tuple[str, float]]:
    """Normalize top-logprob payloads across chat/completions response shapes.

    Args:
        raw_top_logprobs: Response payload fragment for one token.

    Returns:
        List of `(token, logprob)` alternatives.
    """
    if isinstance(raw_top_logprobs, dict):
        return [
            (str(token), float(logprob)) for token, logprob in raw_top_logprobs.items()
        ]
    if isinstance(raw_top_logprobs, list):
        return _from_list_payload(raw_top_logprobs=raw_top_logprobs)
    return []


def _from_list_payload(*, raw_top_logprobs: list[object]) -> list[tuple[str, float]]:
    """Parse list-based top-logprob payload shape.

    Args:
        raw_top_logprobs: List payload from response.

    Returns:
        List of `(token, logprob)` pairs.
    """
    pairs: list[tuple[str, float]] = []
    for item in raw_top_logprobs:
        if not isinstance(item, dict):
            continue
        token = item.get("token")
        logprob = item.get("logprob")
        if token is None or logprob is None:
            continue
        pairs.append((str(token), float(logprob)))
    return pairs


def normalize_probability_entries(
    *, entries: list[tuple[str, float]]
) -> list[tuple[str, float]]:
    """Normalize possibly duplicate token entries to probabilities.

    Args:
        entries: List of `(token, logprob)` entries.

    Returns:
        Unique token probabilities list as `(token, probability)`.
    """
    token_to_probability: dict[str, float] = {}
    for token, logprob in entries:
        probability = probability_from_logprob(logprob=logprob)
        token_to_probability[token] = max(
            token_to_probability.get(token, 0.0), probability
        )
    return list(token_to_probability.items())


def approximate_entropy(
    *,
    selected_token: str,
    selected_logprob: float,
    top_entries: Iterable[tuple[str, float]],
) -> tuple[float, float, tuple[TokenAlternative, ...]]:
    """Compute selected-token probability and approximate entropy.

    Args:
        selected_token: Selected token text.
        selected_logprob: Selected token logprob.
        top_entries: Top alternatives as `(token, logprob)`.

    Returns:
        Tuple `(selected_probability, entropy, alternatives)`.

    Example:
        >>> p, entropy, alts = approximate_entropy(
        ...     selected_token="a",
        ...     selected_logprob=-0.1,
        ...     top_entries=[("a", -0.1), ("b", -2.0)],
        ... )
        >>> round(p, 3) > 0
        True
    """
    entries = list(top_entries)
    entries.append((selected_token, selected_logprob))
    token_probs = normalize_probability_entries(entries=entries)
    selected_probability = probability_from_logprob(logprob=selected_logprob)
    entropy_value = _entropy_with_residual(token_probs=token_probs)
    alternatives = _build_alternatives(entries=entries)
    return selected_probability, entropy_value, alternatives


def _entropy_with_residual(*, token_probs: list[tuple[str, float]]) -> float:
    """Compute entropy with one residual bucket for unseen probability mass.

    Args:
        token_probs: Unique token probabilities.

    Returns:
        Approximate entropy using natural log.
    """
    total_probability = sum(probability for _, probability in token_probs)
    if total_probability > 1.0:
        token_probs = [
            (token, probability / total_probability)
            for token, probability in token_probs
        ]
        total_probability = 1.0
    residual_probability = max(0.0, 1.0 - total_probability)
    entropy_terms = [
        _entropy_term(probability=probability) for _, probability in token_probs
    ]
    if residual_probability > 0.0:
        entropy_terms.append(_entropy_term(probability=residual_probability))
    return sum(entropy_terms)


def _entropy_term(*, probability: float) -> float:
    """Compute one `-p log p` entropy term.

    Args:
        probability: Probability mass for one bucket.

    Returns:
        Entropy contribution.
    """
    if probability <= 0.0:
        return 0.0
    return -probability * math.log(probability)


def _build_alternatives(
    *, entries: list[tuple[str, float]]
) -> tuple[TokenAlternative, ...]:
    """Build sorted token alternatives from raw token logprob entries.

    Args:
        entries: Token and logprob pairs.

    Returns:
        Sorted alternatives by probability descending.
    """
    token_to_logprob: dict[str, float] = {}
    for token, logprob in entries:
        previous_logprob = token_to_logprob.get(token)
        if previous_logprob is None or logprob > previous_logprob:
            token_to_logprob[token] = logprob
    alternatives = [
        TokenAlternative(
            token=token,
            logprob=logprob,
            probability=probability_from_logprob(logprob=logprob),
        )
        for token, logprob in token_to_logprob.items()
    ]
    alternatives.sort(key=lambda alternative: alternative.probability, reverse=True)
    return tuple(alternatives)
