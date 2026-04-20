"""Tests for request-stream prefix bookkeeping."""

from __future__ import annotations

import pytest

from branching_eval.branch_executor import BranchExecutor, _RequestStreamState


def test_update_request_stream_state_output_ids_trims_to_consumed_prefix() -> None:
    """Consumed output-token prefixes should replace cached full output ids."""

    executor = BranchExecutor.__new__(BranchExecutor)
    executor._request_stream_state = {
        "decode:node_x": _RequestStreamState(
            request_id="req_prev",
            input_token_ids=(1, 2, 3),
            output_token_ids=(4, 5, 6, 7),
        )
    }

    executor._update_request_stream_state_output_ids(
        request_stream_id="decode:node_x",
        consumed_output_token_ids=(4, 5),
    )

    updated_state = executor._request_stream_state["decode:node_x"]
    assert updated_state.request_id == "req_prev"
    assert updated_state.input_token_ids == (1, 2, 3)
    assert updated_state.output_token_ids == (4, 5)


def test_update_request_stream_state_output_ids_rejects_non_prefix() -> None:
    """Consumed output ids must remain a prefix of the cached raw output ids."""

    executor = BranchExecutor.__new__(BranchExecutor)
    executor._request_stream_state = {
        "decode:node_x": _RequestStreamState(
            request_id="req_prev",
            input_token_ids=(1, 2, 3),
            output_token_ids=(4, 5, 6, 7),
        )
    }

    with pytest.raises(AssertionError):
        executor._update_request_stream_state_output_ids(
            request_stream_id="decode:node_x",
            consumed_output_token_ids=(4, 9),
        )
