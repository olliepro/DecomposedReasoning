"""vLLM V1 logits processor plugin for the experimental grammar surface."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from vllm_experimental.grammar import (
    GrammarState,
    GrammarTokenIds,
    GrammarTracker,
    temperature_for_state,
)
from vllm_experimental.types import TreeSearchParams


if TYPE_CHECKING:

    class AdapterLogitsProcessorBase:
        """Static fallback for local checks without importing vLLM."""

        pass

else:
    try:
        from vllm.v1.sample.logits_processor import (
            AdapterLogitsProcessor as AdapterLogitsProcessorBase,
        )
    except Exception:  # pragma: no cover - import guard for non-vLLM tooling.

        class AdapterLogitsProcessorBase:
            """Runtime fallback when vLLM is not installed."""

            pass


def _payload_from_params(params: Any) -> dict[str, Any] | None:
    extra_args = getattr(params, "extra_args", None)
    if not isinstance(extra_args, dict):
        return None
    payload = extra_args.get("vllm_experimental")
    if not isinstance(payload, dict):
        return None
    return payload


def _tree_params_from_payload(payload: dict[str, Any]) -> TreeSearchParams:
    return TreeSearchParams(
        mode=str(payload.get("mode", "grammar_temp")),  # type: ignore[arg-type]
        fire_rate=float(payload.get("fire_rate", 0.10)),
        candidate_count=int(payload.get("candidate_count", 50)),
        branch_fanout=int(payload.get("branch_fanout", 2)),
        branch_depth=int(payload.get("branch_depth", 4)),
        off_policy_min_candidates=int(payload.get("off_policy_min_candidates", 3)),
        off_policy_max_candidates=int(payload.get("off_policy_max_candidates", 10)),
        branch_max_tokens=int(payload.get("branch_max_tokens", 700)),
        max_model_len=int(payload.get("max_model_len", 17_408)),
        max_num_batched_tokens=int(payload.get("max_num_batched_tokens", 65_536)),
        max_steer_tokens=int(payload.get("max_steer_tokens", 30)),
        max_exec_tokens=int(payload.get("max_exec_tokens", 512)),
        steer_temperature=float(payload.get("steer_temperature", 1.0)),
        exec_temperature=float(payload.get("exec_temperature", 0.7)),
        post_think_temperature=float(payload.get("post_think_temperature", 0.7)),
        seed=int(payload.get("seed", 1234)),
        native_scheduler_kv_fork=bool(payload.get("native_scheduler_kv_fork", False)),
        native_branch_wave_size=int(payload.get("native_branch_wave_size", 50)),
        native_branch_dynamic_admission=bool(
            payload.get("native_branch_dynamic_admission", True)
        ),
        native_branch_min_free_blocks=int(
            payload.get("native_branch_min_free_blocks", 256)
        ),
        native_branch_free_block_fraction=float(
            payload.get("native_branch_free_block_fraction", 0.05)
        ),
        native_branch_seq_reserve=int(payload.get("native_branch_seq_reserve", 8)),
        native_branch_priority_boost=int(
            payload.get("native_branch_priority_boost", 1000)
        ),
        native_branch_block_safety_multiplier=float(
            payload.get("native_branch_block_safety_multiplier", 1.25)
        ),
        native_branch_blocked_log_interval_s=float(
            payload.get("native_branch_blocked_log_interval_s", 5.0)
        ),
        native_branch_max_live_pools=int(
            payload.get("native_branch_max_live_pools", 2)
        ),
        native_branch_max_queued_pools=int(
            payload.get("native_branch_max_queued_pools", 8)
        ),
    )


def _grammar_tokens_from_payload(payload: dict[str, Any]) -> GrammarTokenIds:
    token_payload = payload.get("control_token_ids")
    assert isinstance(
        token_payload, dict
    ), "vllm_experimental.control_token_ids is required"
    return GrammarTokenIds(
        think_open=int(token_payload["think_open"]),
        think_close=int(token_payload["think_close"]),
        steer_open=int(token_payload["steer_open"]),
        steer_close=int(token_payload["steer_close"]),
        exec_open=int(token_payload["exec_open"]),
        exec_close=int(token_payload["exec_close"]),
        newline=(
            int(token_payload["newline"])
            if token_payload.get("newline") is not None
            else None
        ),
        eos=(
            int(token_payload["eos"]) if token_payload.get("eos") is not None else None
        ),
    )


def _prefix_output_token_ids(payload: dict[str, Any]) -> list[int]:
    token_ids = payload.get("prefix_output_token_ids", [])
    assert isinstance(token_ids, list), "prefix_output_token_ids must be a list"
    return [int(token_id) for token_id in token_ids]


def _token_id_list(*, payload: dict[str, Any], key: str) -> list[int]:
    token_ids = payload.get(key, [])
    assert isinstance(token_ids, list), f"{key} must be a list"
    return [int(token_id) for token_id in token_ids]


def replay_tracker(
    *,
    payload: dict[str, Any],
    output_ids: list[int],
    tokens: GrammarTokenIds,
) -> GrammarTracker:
    """Rebuild grammar state for a vLLM processor call.

    vLLM passes native branch children with replayed parent output. Limit
    overflow during replay should force a close token, not crash the worker
    before the close mask is applied.
    """

    tracker = GrammarTracker(
        tokens=tokens,
        max_steer_tokens=int(payload.get("max_steer_tokens", 30)),
        max_exec_tokens=int(payload.get("max_exec_tokens", 512)),
        max_final_tokens=int(payload.get("max_final_tokens", 2048)),
    )
    for token_id in _prefix_output_token_ids(payload=payload):
        tracker.observe(token_id=token_id, strict_limits=False)
    for token_id in output_ids:
        tracker.observe(token_id=token_id, strict_limits=False)
    return tracker


def _new_tracker(*, payload: dict[str, Any], tokens: GrammarTokenIds) -> GrammarTracker:
    return GrammarTracker(
        tokens=tokens,
        max_steer_tokens=int(payload.get("max_steer_tokens", 30)),
        max_exec_tokens=int(payload.get("max_exec_tokens", 512)),
        max_final_tokens=int(payload.get("max_final_tokens", 2048)),
    )


@dataclass
class IncrementalGrammarReplay:
    """Append-only grammar replay for vLLM's live output-token list."""

    payload: dict[str, Any]
    tokens: GrammarTokenIds
    tracker: GrammarTracker = field(init=False)
    observed_len: int = 0
    payload_identity: int = field(init=False)

    def __post_init__(self) -> None:
        self.payload_identity = id(self.payload)
        self.tracker = _new_tracker(payload=self.payload, tokens=self.tokens)
        self._observe_prefix(payload=self.payload)

    def _observe_prefix(self, *, payload: dict[str, Any]) -> None:
        for token_id in _prefix_output_token_ids(payload=payload):
            self.tracker.observe(token_id=token_id, strict_limits=False)

    def reset(
        self, *, payload: dict[str, Any], output_ids: list[int]
    ) -> GrammarTracker:
        self.payload = payload
        self.payload_identity = id(payload)
        self.tracker = _new_tracker(payload=payload, tokens=self.tokens)
        self._observe_prefix(payload=payload)
        for token_id in output_ids:
            self.tracker.observe(token_id=token_id, strict_limits=False)
        self.observed_len = len(output_ids)
        return self.tracker

    def sync(self, *, payload: dict[str, Any], output_ids: list[int]) -> GrammarTracker:
        if len(output_ids) < self.observed_len or id(payload) != self.payload_identity:
            return self.reset(payload=payload, output_ids=output_ids)
        for token_id in output_ids[self.observed_len :]:
            self.tracker.observe(token_id=token_id, strict_limits=False)
        self.observed_len = len(output_ids)
        return self.tracker


def forced_script_token(
    *,
    payload: dict[str, Any],
    output_ids: list[int],
    tokens: GrammarTokenIds,
) -> int | None:
    """Return the next scripted token for native verbalized branch children."""

    start_index = int(payload.get("forced_output_start_index", 0))
    initial_script = _token_id_list(payload=payload, key="forced_output_token_ids")
    initial_offset = len(output_ids) - start_index
    if 0 <= initial_offset < len(initial_script):
        return initial_script[initial_offset]

    followup_script = _token_id_list(
        payload=payload, key="forced_after_exec_close_token_ids"
    )
    if not followup_script:
        return None
    trigger_after = int(
        payload.get(
            "forced_after_exec_close_trigger_after",
            start_index + len(initial_script),
        )
    )
    search_start = max(0, min(trigger_after, len(output_ids)))
    close_index: int | None = None
    for index in range(search_start, len(output_ids)):
        if output_ids[index] == tokens.exec_close:
            close_index = index
            break
    if close_index is None:
        return None
    followup_offset = len(output_ids) - close_index - 1
    if 0 <= followup_offset < len(followup_script):
        return followup_script[followup_offset]
    return None


def forced_close_token(*, tracker: GrammarTracker) -> int | None:
    """Return a forced close token when a bounded block reaches its cap."""

    if (
        tracker.state == GrammarState.IN_STEER
        and tracker.steer_token_count >= tracker.max_steer_tokens
    ):
        return tracker.tokens.steer_close
    if (
        tracker.state == GrammarState.IN_EXEC
        and tracker.exec_token_count >= tracker.max_exec_tokens
    ):
        return tracker.tokens.exec_close
    if (
        tracker.state == GrammarState.AFTER_THINK_CLOSE
        and tracker.tokens.eos is not None
        and tracker.final_token_count >= tracker.max_final_tokens
    ):
        return tracker.tokens.eos
    return None


def empty_block_close_token(*, tracker: GrammarTracker) -> int | None:
    """Return a close token that must be suppressed for an empty block."""

    if tracker.state == GrammarState.IN_STEER and tracker.steer_token_count == 0:
        return tracker.tokens.steer_close
    if tracker.state == GrammarState.IN_EXEC and tracker.exec_token_count == 0:
        return tracker.tokens.exec_close
    return None


def invalid_control_token_ids(*, tracker: GrammarTracker) -> set[int]:
    """Return control tokens invalid in the current free-text block."""

    tokens = tracker.tokens
    if tracker.state == GrammarState.IN_STEER:
        invalid = {
            tokens.think_open,
            tokens.think_close,
            tokens.steer_open,
            tokens.exec_open,
            tokens.exec_close,
        }
        if tokens.eos is not None:
            invalid.add(tokens.eos)
        return invalid
    if tracker.state == GrammarState.IN_EXEC:
        invalid = {
            tokens.think_open,
            tokens.think_close,
            tokens.steer_open,
            tokens.steer_close,
            tokens.exec_open,
        }
        if tokens.eos is not None:
            invalid.add(tokens.eos)
        return invalid
    if tracker.state == GrammarState.AFTER_THINK_CLOSE:
        invalid = {
            tokens.think_open,
            tokens.think_close,
            tokens.steer_open,
            tokens.steer_close,
            tokens.exec_open,
            tokens.exec_close,
        }
        if tokens.eos is not None and tracker.final_token_count < 1:
            invalid.add(tokens.eos)
        return invalid
    return set()


def forced_open_token(*, tracker: GrammarTracker) -> int | None:
    """Return the next required opening/control token for the current state."""

    tokens = tracker.tokens
    if tracker.state == GrammarState.NEED_THINK_OPEN:
        return tokens.think_open
    if tracker.state == GrammarState.NEED_STEER_OPEN:
        if tokens.newline is not None and not tracker.think_newline_seen:
            return tokens.newline
        return None
    if tracker.state == GrammarState.NEED_EXEC_OPEN:
        if tokens.newline is not None and not tracker.steer_close_newline_seen:
            return tokens.newline
        return tokens.exec_open
    if tracker.state == GrammarState.AFTER_EXEC_CLOSE:
        if tokens.newline is not None and not tracker.exec_close_newline_seen:
            return tokens.newline
    return None


def choice_token_ids(*, tracker: GrammarTracker) -> set[int]:
    """Return allowed branch-choice control tokens for choice states."""

    tokens = tracker.tokens
    if tracker.state == GrammarState.NEED_STEER_OPEN and tracker.think_newline_seen:
        return {tokens.steer_open, tokens.think_close}
    if (
        tracker.state == GrammarState.AFTER_EXEC_CLOSE
        and tracker.exec_close_newline_seen
    ):
        return {tokens.steer_open, tokens.think_close}
    return set()


class ThoughtGrammarLogitsProcessor(AdapterLogitsProcessorBase):
    """Per-request grammar and dynamic-temperature processor."""

    @classmethod
    def validate_params(cls, sampling_params: Any) -> None:
        payload = _payload_from_params(sampling_params)
        if payload is None:
            return
        tree_params = _tree_params_from_payload(payload=payload)
        tree_params.validate()
        _grammar_tokens_from_payload(payload=payload)

    def is_argmax_invariant(self) -> bool:
        return False

    def new_req_logits_processor(self, params: Any) -> Any:
        payload = _payload_from_params(params)
        if payload is None:
            return None
        tree_params = _tree_params_from_payload(payload=payload)
        tokens = _grammar_tokens_from_payload(payload=payload)
        replay_state = IncrementalGrammarReplay(payload=payload, tokens=tokens)

        def apply(output_ids: list[int], logits: Any) -> Any:
            current_payload = _payload_from_params(params) or payload
            tracker = replay_state.sync(
                payload=current_payload,
                output_ids=output_ids,
            )
            scripted_id = forced_script_token(
                payload=current_payload,
                output_ids=output_ids,
                tokens=tokens,
            )
            if scripted_id is not None:
                masked = logits.new_full(logits.shape, -float("inf"))
                masked[scripted_id] = logits[scripted_id]
                return masked
            forced_close = forced_close_token(tracker=tracker)
            if forced_close is not None:
                masked = logits.new_full(logits.shape, -float("inf"))
                masked[forced_close] = logits[forced_close]
                return masked
            temp = temperature_for_state(
                state=tracker.state,
                steer_temperature=tree_params.steer_temperature,
                exec_temperature=tree_params.exec_temperature,
                post_think_temperature=tree_params.post_think_temperature,
            )
            if temp != 1.0:
                logits = logits / temp
            forced_id = forced_open_token(tracker=tracker)
            if forced_id is not None:
                masked = logits.new_full(logits.shape, -float("inf"))
                masked[forced_id] = logits[forced_id]
                return masked
            choices = choice_token_ids(tracker=tracker)
            if choices:
                masked = logits.new_full(logits.shape, -float("inf"))
                for token_id in choices:
                    masked[token_id] = logits[token_id]
                return masked
            empty_close = empty_block_close_token(tracker=tracker)
            if empty_close is not None:
                logits[empty_close] = -float("inf")
            for token_id in invalid_control_token_ids(tracker=tracker):
                logits[token_id] = -float("inf")
            return logits

        return apply


def validate_request_extra_args(*, sampling_params: Any) -> None:
    """Hookable validation used by the patched runtime copy."""

    payload = _payload_from_params(sampling_params)
    if payload is None:
        return
    tree_params = _tree_params_from_payload(payload=payload)
    tree_params.validate()
    _grammar_tokens_from_payload(payload=payload)
