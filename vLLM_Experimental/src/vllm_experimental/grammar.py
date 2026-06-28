"""Token-level steer/exec grammar helpers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable


class GrammarState(str, Enum):
    """High-level control-tag state."""

    NEED_THINK_OPEN = "need_think_open"
    NEED_STEER_OPEN = "need_steer_open"
    IN_STEER = "in_steer"
    NEED_EXEC_OPEN = "need_exec_open"
    IN_EXEC = "in_exec"
    AFTER_EXEC_CLOSE = "after_exec_close"
    AFTER_THINK_CLOSE = "after_think_close"
    DONE = "done"


@dataclass(frozen=True)
class GrammarTokenIds:
    """Atomic control-token ids for the added-token 12ep model."""

    think_open: int
    think_close: int
    steer_open: int
    steer_close: int
    exec_open: int
    exec_close: int
    newline: int | None = None
    eos: int | None = None


@dataclass
class GrammarTracker:
    """Incrementally track the steer/exec grammar over generated token ids."""

    tokens: GrammarTokenIds
    max_steer_tokens: int = 30
    max_exec_tokens: int = 512
    max_final_tokens: int = 2048
    state: GrammarState = GrammarState.NEED_THINK_OPEN
    steer_token_count: int = 0
    exec_token_count: int = 0
    final_token_count: int = 0
    think_newline_seen: bool = False
    steer_close_newline_seen: bool = False
    exec_close_newline_seen: bool = False

    def observe(self, token_id: int, *, strict_limits: bool = True) -> GrammarState:
        """Advance grammar state after one generated token."""

        if self.state == GrammarState.NEED_THINK_OPEN:
            assert token_id == self.tokens.think_open, "expected <think>"
            self.state = GrammarState.NEED_STEER_OPEN
            self.think_newline_seen = False
            return self.state
        if self.state == GrammarState.NEED_STEER_OPEN:
            if self.tokens.newline is not None and not self.think_newline_seen:
                assert token_id == self.tokens.newline, "expected newline after <think>"
                self.think_newline_seen = True
                return self.state
            if token_id == self.tokens.think_close:
                self.state = GrammarState.AFTER_THINK_CLOSE
                self.final_token_count = 0
                return self.state
            assert token_id == self.tokens.steer_open, "expected <steer> or </think>"
            self.state = GrammarState.IN_STEER
            self.steer_token_count = 0
            return self.state
        if self.state == GrammarState.IN_STEER:
            if token_id == self.tokens.steer_close:
                assert self.steer_token_count >= 1, "empty steer block"
                self.state = GrammarState.NEED_EXEC_OPEN
                self.steer_close_newline_seen = False
            else:
                assert (
                    self.tokens.eos is None or token_id != self.tokens.eos
                ), "eos is only allowed after final answer content"
                self.steer_token_count += 1
                if strict_limits:
                    assert (
                        self.steer_token_count <= self.max_steer_tokens
                    ), "steer block exceeded token limit"
            return self.state
        if self.state == GrammarState.NEED_EXEC_OPEN:
            if self.tokens.newline is not None and not self.steer_close_newline_seen:
                assert (
                    token_id == self.tokens.newline
                ), "expected newline after </steer>"
                self.steer_close_newline_seen = True
                return self.state
            assert token_id == self.tokens.exec_open, "expected <exec>"
            self.state = GrammarState.IN_EXEC
            self.exec_token_count = 0
            return self.state
        if self.state == GrammarState.IN_EXEC:
            if token_id == self.tokens.exec_close:
                assert self.exec_token_count >= 1, "empty exec block"
                self.state = GrammarState.AFTER_EXEC_CLOSE
                self.exec_close_newline_seen = False
            else:
                assert (
                    self.tokens.eos is None or token_id != self.tokens.eos
                ), "eos is only allowed after final answer content"
                self.exec_token_count += 1
                if strict_limits:
                    assert (
                        self.exec_token_count <= self.max_exec_tokens
                    ), "exec block exceeded token limit"
            return self.state
        if self.state == GrammarState.AFTER_EXEC_CLOSE:
            if self.tokens.newline is not None and not self.exec_close_newline_seen:
                assert token_id == self.tokens.newline, "expected newline after </exec>"
                self.exec_close_newline_seen = True
                return self.state
            if token_id == self.tokens.steer_open:
                self.state = GrammarState.IN_STEER
                self.steer_token_count = 0
            elif token_id == self.tokens.think_close:
                self.state = GrammarState.AFTER_THINK_CLOSE
                self.final_token_count = 0
            else:
                raise AssertionError("expected <steer> or </think> after </exec>")
            return self.state
        if self.state == GrammarState.AFTER_THINK_CLOSE:
            if self.tokens.eos is not None and token_id == self.tokens.eos:
                assert (
                    self.final_token_count >= 1
                ), "eos requires at least one final answer token"
                self.state = GrammarState.DONE
            else:
                self.final_token_count += 1
                if strict_limits:
                    assert (
                        self.final_token_count <= self.max_final_tokens
                    ), "final answer exceeded token limit"
            return self.state
        return self.state

    def observe_many(
        self, token_ids: Iterable[int], *, strict_limits: bool = True
    ) -> GrammarState:
        """Advance over a sequence and return the final state."""

        for token_id in token_ids:
            self.observe(token_id=token_id, strict_limits=strict_limits)
        return self.state


def temperature_for_state(
    *,
    state: GrammarState,
    steer_temperature: float = 1.0,
    exec_temperature: float = 0.7,
    post_think_temperature: float = 0.7,
) -> float:
    """Return the configured sampling temperature for the grammar state."""

    if state in {
        GrammarState.NEED_STEER_OPEN,
        GrammarState.IN_STEER,
        GrammarState.NEED_EXEC_OPEN,
        GrammarState.AFTER_EXEC_CLOSE,
    }:
        return steer_temperature
    if state == GrammarState.AFTER_THINK_CLOSE:
        return post_think_temperature
    return exec_temperature


def suffix_matches(*, token_ids: list[int], suffix: tuple[int, ...]) -> bool:
    """Return whether token_ids ends with suffix."""

    if not suffix:
        return True
    if len(token_ids) < len(suffix):
        return False
    return tuple(token_ids[-len(suffix) :]) == suffix
