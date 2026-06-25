"""vLLM request bookkeeping and event serialization helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from branching_eval.artifact_store import ArtifactStore
from branching_eval.event_types import EventContext
from vllm_client import GenerationChoice

ASSISTANT_PREFIX_TAIL_CHARS = 200


@dataclass(frozen=True)
class RequestStreamState:
    """Previous request state used for token-prefix invariant validation."""

    request_id: str
    input_token_ids: tuple[int, ...]
    output_token_ids: tuple[int, ...]


def resolve_prefix_base_delta(
    *,
    request_stream_state: dict[str, RequestStreamState],
    request_stream_id: str,
    current_input_token_ids: tuple[int, ...],
    prefix_chain_enabled: bool,
) -> tuple[tuple[int, ...], tuple[int, ...], str | None]:
    """Return cached base tokens, new delta tokens, and previous request id."""

    if not prefix_chain_enabled:
        return (), current_input_token_ids, None
    previous_state = request_stream_state.get(request_stream_id)
    if previous_state is None:
        return (), current_input_token_ids, None
    base = previous_state.input_token_ids + previous_state.output_token_ids
    assert current_input_token_ids[: len(base)] == base, (
        f"request stream prefix mismatch for {request_stream_id}: "
        f"expected base token prefix length {len(base)}, "
        f"got input length {len(current_input_token_ids)}"
    )
    delta = current_input_token_ids[len(base) :]
    return base, delta, previous_state.request_id


def update_request_stream_state_output_ids(
    *,
    request_stream_state: dict[str, RequestStreamState],
    request_stream_id: str,
    consumed_output_token_ids: tuple[int, ...],
) -> None:
    """Replace cached output ids with the token prefix actually consumed."""

    previous_state = request_stream_state.get(request_stream_id)
    if previous_state is None:
        return
    assert previous_state.output_token_ids[: len(consumed_output_token_ids)] == (
        consumed_output_token_ids
    ), f"consumed output ids must prefix cached output ids for {request_stream_id}"
    request_stream_state[request_stream_id] = RequestStreamState(
        request_id=previous_state.request_id,
        input_token_ids=previous_state.input_token_ids,
        output_token_ids=consumed_output_token_ids,
    )


def reset_request_stream_state(
    *,
    request_stream_state: dict[str, RequestStreamState],
    request_stream_id: str,
) -> None:
    """Drop cached prefix-chain state for one request stream."""

    request_stream_state.pop(request_stream_id, None)


def set_request_stream_state(
    *,
    request_stream_state: dict[str, RequestStreamState],
    request_stream_id: str,
    request_id: str,
    input_token_ids: tuple[int, ...],
    output_token_ids: tuple[int, ...],
) -> None:
    """Store the latest prompt/output token ids for a prefix-chain stream."""

    request_stream_state[request_stream_id] = RequestStreamState(
        request_id=request_id,
        input_token_ids=input_token_ids,
        output_token_ids=output_token_ids,
    )


def append_vllm_request_event(
    *,
    artifact_store: ArtifactStore,
    context: EventContext,
    request_id: str,
    request_stream_id: str,
    prev_request_id: str | None,
    request_kind: str,
    assistant_prefix: str,
    temperature: float,
    top_p: float,
    top_k: int | None = None,
    min_p: float | None = None,
    presence_penalty: float | None = None,
    max_tokens: int,
    n: int,
    seed: int,
    stop: tuple[str, ...] | None,
    top_logprobs: int,
    current_input_token_ids: tuple[int, ...],
    base_prefix_token_ids: tuple[int, ...],
    delta_input_token_ids: tuple[int, ...],
    request_priority_value: int | None = None,
    request_branch_number: str | None = None,
    repetition_penalty: float | None = None,
) -> None:
    """Append one typed vLLM request event row."""

    artifact_store.append_event(
        context=context,
        event_type="vllm_request",
        payload={
            "request_id": request_id,
            "request_stream_id": request_stream_id,
            "prev_request_id": prev_request_id,
            "request_kind": request_kind,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "min_p": min_p,
            "presence_penalty": presence_penalty,
            "max_tokens": max_tokens,
            "n": n,
            "seed": seed,
            "stop": list(stop) if stop is not None else None,
            "top_logprobs": top_logprobs,
            "request_priority": request_priority_value,
            "request_branch_number": request_branch_number,
            "repetition_penalty": repetition_penalty,
            "current_input_token_count": len(current_input_token_ids),
            "base_prefix_token_count": len(base_prefix_token_ids),
            "delta_token_count": len(delta_input_token_ids),
            "delta_input_token_ids": list(delta_input_token_ids),
            "assistant_prefix_char_count": len(assistant_prefix),
            "assistant_prefix_tail": assistant_prefix[-ASSISTANT_PREFIX_TAIL_CHARS:],
        },
    )


def append_vllm_response_event(
    *,
    artifact_store: ArtifactStore,
    context: EventContext,
    request_id: str,
    request_stream_id: str,
    request_kind: str,
    latency_seconds: float,
    choices: tuple[GenerationChoice, ...],
    compact_text_preview: Callable[[str, int], str],
) -> None:
    """Append one typed successful vLLM response event row."""

    artifact_store.append_event(
        context=context,
        event_type="vllm_response",
        payload={
            "request_id": request_id,
            "request_stream_id": request_stream_id,
            "request_kind": request_kind,
            "status": "ok",
            "latency_seconds": latency_seconds,
            "choice_count": len(choices),
            "choices": [
                serialize_choice_for_vllm_event(
                    choice=choice,
                    compact_text_preview=compact_text_preview,
                )
                for choice in choices
            ],
        },
    )


def append_vllm_response_error_event(
    *,
    artifact_store: ArtifactStore,
    context: EventContext,
    request_id: str,
    request_stream_id: str,
    request_kind: str,
    error_message: str,
    latency_seconds: float,
) -> None:
    """Append one typed failed vLLM response event row."""

    artifact_store.append_event(
        context=context,
        event_type="vllm_response",
        payload={
            "request_id": request_id,
            "request_stream_id": request_stream_id,
            "request_kind": request_kind,
            "status": "error",
            "error_message": error_message,
            "latency_seconds": latency_seconds,
            "choices": [],
        },
    )


def serialize_choice_for_vllm_event(
    *,
    choice: GenerationChoice,
    compact_text_preview: Callable[[str, int], str],
) -> dict[str, Any]:
    """Serialize one generation choice for runtime request inspection."""

    token_ids = tuple(choice.token_ids or ())
    token_rows = []
    for token_index, parsed_token in enumerate(choice.tokens):
        token_id = token_ids[token_index] if token_index < len(token_ids) else None
        token_rows.append(
            {
                "token_index": token_index,
                "token_id": token_id,
                "token_text": parsed_token.token,
                "selected_logprob": parsed_token.logprob,
            }
        )
    return {
        "index": choice.index,
        "token_ids": list(token_ids),
        "finish_reason": choice.finish_reason,
        "stop_reason": choice.stop_reason,
        "output_token_count": len(token_ids),
        "text_preview": compact_text_preview(choice.text, 160),
        "tokens": token_rows,
    }
