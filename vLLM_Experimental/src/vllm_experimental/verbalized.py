"""Verbalized off-policy option helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from random import Random

ENUMERATE_TEMPLATE = (
    "Enumerate {candidate_count} distinct options for the immediate next decision/step"
)
CONTINUE_TEMPLATE = "Proceed with option {option_number}"
CONTINUE_EXEC_PREFILL_TEMPLATE = "Let's do option {option_number}:"

OPTION_RE = re.compile(
    r"(?:^|\n)\s*(?:option\s*)?(?P<number>\d+)[\).:\-]\s*(?P<text>[^\n]+)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class VerbalizedOption:
    """One parsed verbalized option."""

    option_number: int
    text: str


def enumeration_prompt(*, candidate_count: int) -> str:
    """Return the canonical off-policy enumeration steer text."""

    assert candidate_count >= 1, "candidate_count must be positive"
    return ENUMERATE_TEMPLATE.format(candidate_count=candidate_count)


def continue_steer(*, option_number: int) -> str:
    """Return the canonical selected-option steer text."""

    assert option_number >= 1, "option_number must be positive"
    return CONTINUE_TEMPLATE.format(option_number=option_number)


def continue_exec_prefill(*, option_number: int) -> str:
    """Return the canonical selected-option exec prefill."""

    assert option_number >= 1, "option_number must be positive"
    return CONTINUE_EXEC_PREFILL_TEMPLATE.format(option_number=option_number)


def parse_verbalized_options(*, text: str) -> tuple[VerbalizedOption, ...]:
    """Parse numbered option rows from a model enumeration."""

    options: list[VerbalizedOption] = []
    seen: set[int] = set()
    for match in OPTION_RE.finditer(text):
        option_number = int(match.group("number"))
        if option_number in seen:
            continue
        option_text = match.group("text").strip()
        if not option_text:
            continue
        seen.add(option_number)
        options.append(VerbalizedOption(option_number=option_number, text=option_text))
    return tuple(options)


def sample_option_numbers(
    *, available: tuple[int, ...], count: int, seed: int
) -> tuple[int, ...]:
    """Sample distinct option numbers deterministically."""

    assert count >= 1, "count must be positive"
    assert len(available) >= count, "not enough options to sample"
    shuffled = list(available)
    Random(seed).shuffle(shuffled)
    return tuple(sorted(shuffled[:count]))
