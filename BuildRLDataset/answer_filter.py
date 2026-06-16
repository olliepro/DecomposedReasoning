from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

AnswerFormatClass = Literal[
    "plain_scalar_or_simple_fraction",
    "exact_symbolic_numeric_expression",
    "other",
]

PLAIN_SCALAR_OR_SIMPLE_FRACTION: AnswerFormatClass = "plain_scalar_or_simple_fraction"
EXACT_SYMBOLIC_NUMERIC_EXPRESSION: AnswerFormatClass = (
    "exact_symbolic_numeric_expression"
)
OTHER_ANSWER_FORMAT: AnswerFormatClass = "other"
ALLOWED_ANSWER_FORMAT_CLASSES: frozenset[AnswerFormatClass] = frozenset(
    {PLAIN_SCALAR_OR_SIMPLE_FRACTION, EXACT_SYMBOLIC_NUMERIC_EXPRESSION}
)

OUTER_LATEX_WRAPPER_PATTERN = re.compile(
    r"^\\(?:boxed|text|mathrm|operatorname)\{(.+)\}$"
)
PLAIN_SCALAR_PATTERN = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")
SIMPLE_FRACTION_PATTERN = re.compile(r"[+-]?\d+/[+-]?\d+")
NUMERIC_UNIT_PATTERN = re.compile(
    r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:\\%|%|\^\\circ|\\text\{[^}]+\})"
)
SYMBOLIC_MARKERS = ("\\sqrt", "\\pi", "^", "\\frac", "π")


@dataclass(frozen=True)
class AnswerFormat:
    """Classification for one ground-truth answer string."""

    raw_text: str
    normalized_text: str
    format_class: AnswerFormatClass

    def is_trainable(self) -> bool:
        """Return whether this answer class is kept for RL training."""

        return self.format_class in ALLOWED_ANSWER_FORMAT_CLASSES


def strip_outer_latex_wrappers(*, text: str) -> str:
    """Peel simple whole-answer LaTeX wrappers from an answer string."""

    value = text.strip()
    while True:
        match = OUTER_LATEX_WRAPPER_PATTERN.fullmatch(value)
        if match is None:
            return value
        value = match.group(1).strip()


def normalize_answer_text(*, text: str) -> str:
    """Normalize an answer string for coarse answer-format classification."""

    value = text.strip()
    if value.startswith("$$") and value.endswith("$$") and len(value) >= 4:
        value = value[2:-2].strip()
    if value.startswith("$") and value.endswith("$") and len(value) >= 2:
        value = value[1:-1].strip()
    value = strip_outer_latex_wrappers(text=value)
    value = value.replace(",", "")
    return re.sub(pattern=r"\s+", repl="", string=value)


def classify_answer_text(*, text: str) -> AnswerFormat:
    """Classify a raw ground-truth answer string for RL train filtering."""

    normalized_text = normalize_answer_text(text=text)
    if PLAIN_SCALAR_PATTERN.fullmatch(normalized_text):
        return AnswerFormat(
            raw_text=text,
            normalized_text=normalized_text,
            format_class=PLAIN_SCALAR_OR_SIMPLE_FRACTION,
        )
    if SIMPLE_FRACTION_PATTERN.fullmatch(normalized_text):
        denominator = normalized_text.rsplit("/", 1)[1]
        if int(denominator) != 0:
            return AnswerFormat(
                raw_text=text,
                normalized_text=normalized_text,
                format_class=PLAIN_SCALAR_OR_SIMPLE_FRACTION,
            )
    if NUMERIC_UNIT_PATTERN.fullmatch(normalized_text):
        return AnswerFormat(
            raw_text=text,
            normalized_text=normalized_text,
            format_class=OTHER_ANSWER_FORMAT,
        )
    if any(marker in normalized_text for marker in SYMBOLIC_MARKERS):
        return AnswerFormat(
            raw_text=text,
            normalized_text=normalized_text,
            format_class=EXACT_SYMBOLIC_NUMERIC_EXPRESSION,
        )
    return AnswerFormat(
        raw_text=text, normalized_text=normalized_text, format_class=OTHER_ANSWER_FORMAT
    )


def classify_answer_values(*, values: list[str]) -> AnswerFormat:
    """Classify the first non-empty answer value, matching reward normalization."""

    non_empty_values = [value for value in values if value.strip()]
    if not non_empty_values:
        return AnswerFormat(
            raw_text="", normalized_text="", format_class=OTHER_ANSWER_FORMAT
        )
    return classify_answer_text(text=non_empty_values[0])
