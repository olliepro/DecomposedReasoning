"""Local native-vLLM debugging checks that do not need a Slurm job."""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


CONTROL_TAGS = ("<think>", "</think>", "<steer>", "</steer>", "<exec>", "</exec>")


@dataclass(frozen=True)
class RuntimeSourceReport:
    """Source-level invariants for a materialized vLLM runtime."""

    runtime_path: Path
    checks: dict[str, bool]

    @property
    def failures(self) -> list[str]:
        """Return failed invariant names."""

        return [name for name, passed in self.checks.items() if not passed]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""

        return {
            "runtime_path": str(self.runtime_path),
            "checks": self.checks,
            "failures": self.failures,
        }


@dataclass(frozen=True)
class ControlTokenReport:
    """Tokenizer report for one control tag."""

    text: str
    token_ids: list[int]
    decoded_pieces: list[str]
    is_atomic: bool

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""

        return {
            "text": self.text,
            "token_ids": self.token_ids,
            "decoded_pieces": self.decoded_pieces,
            "is_atomic": self.is_atomic,
        }


@dataclass(frozen=True)
class CandidateAuditReport:
    """Summary of native branch candidate audit evidence."""

    path: Path
    promote_event_count: int
    candidate_count: int
    selected_mismatch_count: int
    layout_only_steer_count: int
    invalid_literal_tag_count: int

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""

        return {
            "path": str(self.path),
            "promote_event_count": self.promote_event_count,
            "candidate_count": self.candidate_count,
            "selected_mismatch_count": self.selected_mismatch_count,
            "layout_only_steer_count": self.layout_only_steer_count,
            "invalid_literal_tag_count": self.invalid_literal_tag_count,
        }


def runtime_source_report(*, runtime_path: Path) -> RuntimeSourceReport:
    """Check final materialized runtime source for hidden-child invariants."""

    scheduler = _read_runtime_file(
        runtime_path=runtime_path,
        relative_path=Path("vllm/v1/core/sched/scheduler.py"),
    )
    runner = _read_runtime_file(
        runtime_path=runtime_path,
        relative_path=Path("vllm/v1/worker/gpu_model_runner.py"),
    )
    engine_init = _read_runtime_file(
        runtime_path=runtime_path,
        relative_path=Path("vllm/v1/engine/__init__.py"),
    )
    output_processor = _read_runtime_file(
        runtime_path=runtime_path,
        relative_path=Path("vllm/v1/engine/output_processor.py"),
    )
    single_type_kv = _read_runtime_file(
        runtime_path=runtime_path,
        relative_path=Path("vllm/v1/core/single_type_kv_cache_manager.py"),
    )
    kv_coordinator = _read_runtime_file(
        runtime_path=runtime_path,
        relative_path=Path("vllm/v1/core/kv_cache_coordinator.py"),
    )
    kv_manager = _read_runtime_file(
        runtime_path=runtime_path,
        relative_path=Path("vllm/v1/core/kv_cache_manager.py"),
    )
    checks = {
        "child_prompt_includes_parent_output": (
            "prompt_token_ids=list(parent.prompt_token_ids) + "
            "list(parent.output_token_ids)" in scheduler
        ),
        "child_output_is_new_only": (
            "child.append_output_token_ids(list(parent.output_token_ids))"
            not in scheduler
        ),
        "child_replays_parent_output_for_grammar": (
            'parent_prefix_output_ids = child_payload.get("prefix_output_token_ids", [])'
            in scheduler
            and "[int(token_id) for token_id in parent_prefix_output_ids]" in scheduler
            and "+ list(parent.output_token_ids)" in scheduler
        ),
        "eps_triggers_after_steer_open": (
            '"eps_on_policy_diverse"' in scheduler
            and '"eps_off_policy_verbalized"' in scheduler
            and 'steer_open = int(token_payload["steer_open"])' in scheduler
            and "request._output_token_ids[-1] == steer_open" in scheduler
        ),
        "on_policy_child_samples_steer_body": (
            "(candidate_id, [], [])" in scheduler
            and 'steer_open = [int(token_payload["steer_open"])]' not in scheduler
        ),
        "boundary_suffix_requires_exec_close_newline": (
            'newline = token_payload.get("newline")' in scheduler
            and "request._output_token_ids[-2] == exec_close" in scheduler
            and "request._output_token_ids[-1] == int(newline)" in scheduler
        ),
        "on_policy_stops_after_exec_open": (
            'return [int(token_payload["exec_open"])]' in scheduler
        ),
        "on_policy_budget_includes_exec_open": (
            'return int(payload.get("max_steer_tokens", 30)) + 3' in scheduler
        ),
        "finished_parent_boundaries_do_not_refire": (
            "if think_close in request._output_token_ids:" in scheduler
        ),
        "branch_suspend_discards_async_placeholders": (
            "async_tokens_to_discard = parent.num_output_placeholders" in scheduler
            and "parent.async_tokens_to_discard = async_tokens_to_discard" in scheduler
            and "parent.num_output_placeholders = 0" in scheduler
            and "self.prev_step_scheduled_req_ids.discard(parent_id)" in scheduler
        ),
        "child_budget_counts_new_tokens_only": (
            "child_params.max_tokens = len(initial_script) + branch_budget + "
            "len(followup_script)" in scheduler
        ),
        "child_uses_forked_computed_prefix": (
            "child.num_computed_tokens = fork_tokens" in scheduler
            and "self.waiting.add_request(child)" in scheduler
        ),
        "full_prefix_child_recomputes_last_token": (
            "if child.num_computed_tokens == child.num_tokens:" in scheduler
            and "child.num_computed_tokens = child.num_tokens - 1" in scheduler
        ),
        "waiting_request_accepts_precomputed_prefix": (
            "if request.num_computed_tokens == 0:" in scheduler
            and "num_computed_tokens = request.num_computed_tokens" in scheduler
        ),
        "worker_new_request_keeps_num_computed_tokens": (
            "num_computed_tokens=new_req_data.num_computed_tokens" in runner
        ),
        "diverse_top_k_random_sampling": (
            "_vllm_exp_diverse_top_k_child_ids" in scheduler
            and '"diverse_top_k_sample"' in scheduler
        ),
        "branch_promote_logs_top_k": (
            "top_k_candidate_ids=self._vllm_exp_candidate_ids(top_k_child_ids)"
            in scheduler
            and "unique_candidate_count=unique_count" in scheduler
        ),
        "hidden_children_do_not_request_logprobs": (
            "child_params.logprobs = max(child_params.logprobs or 0, 1)"
            not in scheduler
        ),
        "promoted_parent_finishes_at_token_cap": (
            "_vllm_exp_finish_parent_if_limit_reached" in scheduler
            and "self._vllm_exp_suspended_parent_ids.discard(parent.request_id)"
            in scheduler
            and "parent_finished_by_limit" in scheduler
            and "finish_reason=parent_finish_reason" in scheduler
            and "and not parent_finished_by_limit" in scheduler
        ),
        "off_policy_enumerates_inline_before_branch": (
            "_vllm_exp_start_off_policy_enumeration" in scheduler
            and '"off_policy_enumerate_inline"' in scheduler
            and "self._vllm_exp_force_parent_script(parent, enum_script)" in scheduler
        ),
        "off_policy_fanout_one_continues_inline": (
            '"off_policy_continue_inline"' in scheduler
            and "self._vllm_exp_force_parent_script(parent, continue_script)"
            in scheduler
        ),
        "off_policy_branches_only_after_enumeration": (
            "_vllm_exp_off_policy_branch_options.pop" in scheduler
            and "self._vllm_exp_continue_script_after_boundary(payload, option_number)"
            in scheduler
            and "enum_script," not in scheduler
        ),
        "off_policy_branch_skip_has_inline_fallback": (
            '"off_policy_continue_inline_fallback"' in scheduler
            and "if self._vllm_exp_start_branches(parent):" in scheduler
            and "self._vllm_exp_off_policy_branch_options.pop(parent.request_id, None)"
            in scheduler
        ),
        "branch_products_use_engine_payload": (
            "vllm_experimental_branch_outputs: list[dict[str, Any]] | None"
            in engine_init
            and "engine_core_output.vllm_experimental_branch_outputs"
            in output_processor
            and "_vllm_exp_make_branch_request_output" in output_processor
        ),
        "fanout_returns_branch_products": (
            "_vllm_exp_return_child_ids" in scheduler
            and "return top_k_child_ids[:branch_fanout]" in scheduler
            and "return child_ids[:branch_fanout]" in scheduler
        ),
        "branch_return_finishes_parent": (
            '"branch_return"' in scheduler
            and "parent.status = RequestStatus.FINISHED_STOPPED" in scheduler
            and "vllm_experimental_branch_outputs=branch_outputs" in scheduler
        ),
        "native_branch_depth_cap": (
            "_vllm_exp_parent_branch_fires" in scheduler
            and "_vllm_exp_branch_depth_used" in scheduler
            and '"branch_skip_depth_limit"' in scheduler
            and 'branch_fanout = int(payload.get("branch_fanout", 1))' in scheduler
            and "if branch_fanout > 1 and branch_depth_used >= branch_depth_limit:"
            in scheduler
            and 'payload.get("branch_depth", 4)' in scheduler
            and 'payload.get("branch_depth_start", 0)' in scheduler
        ),
        "branch_depth_consumed_after_successful_start": (
            "def _vllm_exp_consume_branch_depth(self, request: Request)" in scheduler
            and "started = self._vllm_exp_start_branches(request)" in scheduler
            and "if started:\n            self._vllm_exp_consume_branch_depth(request)"
            in scheduler
            and "if sample >= fire_rate:" in scheduler
            and "return started" in scheduler
        ),
        "completed_hidden_children_are_freed_early": (
            "_vllm_exp_child_products: dict[str, dict[str, Any]]" in scheduler
            and "_vllm_exp_store_child_product(child)" in scheduler
            and "self._free_request(child)" in scheduler
            and "_vllm_exp_child_suffix_token_ids" in scheduler
        ),
        "completed_hidden_child_loads_parent_payload": (
            "parent = self.requests[parent_id]\n"
            "        payload = self._vllm_exp_payload(parent)" in scheduler
            and 'assert payload is not None, "native branch parent payload missing"'
            in scheduler
        ),
        "promoted_parent_replays_selected_suffix": (
            "def _vllm_exp_promote_winner(self, parent: Request, winner_id: str)"
            in scheduler
            and "replay_from_tokens = self._vllm_exp_branch_fork_tokens.get"
            in scheduler
            and "return trimmed_blocks" in scheduler
        ),
        "promoted_parent_trims_stale_blocks_before_replay": (
            "def trim_blocks(self, request_id: str, num_computed_tokens: int) -> int:"
            in single_type_kv
            and "self.block_pool.free_blocks(reversed(stale_blocks))" in single_type_kv
            and "def trim_blocks(self, request_id: str, num_computed_tokens: int) -> int:"
            in kv_coordinator
            and "manager.trim_blocks(request_id, num_computed_tokens)" in kv_coordinator
            and "def trim_blocks(self, request_id: str, num_computed_tokens: int) -> int:"
            in kv_manager
            and "self.kv_cache_manager.trim_blocks(" in scheduler
            and "kv_blocks_trimmed=trimmed_blocks" in scheduler
        ),
        "native_fork_marks_shared_blocks_accounted": (
            "self.num_cached_block[child_request_id] = num_shared_blocks"
            in single_type_kv
            and "self.num_cached_block[target_request_id] = len(source_blocks)"
            in single_type_kv
            and "self.num_cached_block[request_id] = retained_count" in single_type_kv
        ),
        "dynamic_pool_admission_queue": (
            "_vllm_exp_branch_pool_queue: deque[str] = deque()" in scheduler
            and "_vllm_exp_try_admit_branch_pools()" in scheduler
            and '"branch_pool_queued"' in scheduler
        ),
        "dynamic_pool_atomic_launch": (
            "def _vllm_exp_launch_branch_pool(self, parent: Request)" in scheduler
            and "pool_plan = list(pending)" in scheduler
            and "pending.clear()" in scheduler
            and "fast_lane=True" in scheduler
        ),
        "dynamic_pool_immediate_admission": (
            '"branch_pool_queued",\n                candidate_count=len(branch_plan),\n'
            '                reason="queued",\n            )\n'
            "            self._vllm_exp_try_admit_branch_pools()" in scheduler
        ),
        "dynamic_pool_budget_checks": (
            "_vllm_exp_estimated_pool_blocks" in scheduler
            and "native_branch_min_free_blocks" in scheduler
            and "native_branch_free_block_fraction" in scheduler
            and "native_branch_seq_reserve" in scheduler
            and 'reason = "free_blocks"' in scheduler
        ),
        "dynamic_pool_fast_lane": (
            "self.waiting.prepend_request(child)" in scheduler
            and "native_branch_priority_boost" in scheduler
            and "SchedulingPolicy.FCFS" in scheduler
        ),
        "dynamic_pool_backpressure_logging": (
            '"branch_pool_blocked"' in scheduler
            and "native_branch_blocked_log_interval_s" in scheduler
            and "self._vllm_exp_last_blocked_log_at" in scheduler
        ),
        "dynamic_pool_pressure_queues_without_skip": (
            '"branch_skip_backpressure"' not in scheduler
            and "self._vllm_exp_branch_pool_queue.append(parent_id)" in scheduler
            and "self._vllm_exp_try_admit_branch_pools()" in scheduler
            and "def _vllm_exp_clear_pending_branch_pool" not in scheduler
            and "native_branch_max_live_pools" in scheduler
            and "native_branch_max_queued_pools" in scheduler
            and 'reason = "live_pool_limit"' in scheduler
        ),
    }
    return RuntimeSourceReport(runtime_path=runtime_path, checks=checks)


def control_token_reports(*, tokenizer: Any) -> list[ControlTokenReport]:
    """Return atomic-token reports for grammar control tags."""

    reports: list[ControlTokenReport] = []
    for text in (*CONTROL_TAGS, "\n"):
        token_ids = [
            int(token_id)
            for token_id in tokenizer.encode(text, add_special_tokens=False)
        ]
        decoded_pieces = [str(tokenizer.decode([token_id])) for token_id in token_ids]
        reports.append(
            ControlTokenReport(
                text=text,
                token_ids=token_ids,
                decoded_pieces=decoded_pieces,
                is_atomic=len(token_ids) == 1,
            )
        )
    return reports


def candidate_audit_report(
    *, path: Path, decoder: Callable[[list[int]], str] | None = None
) -> CandidateAuditReport:
    """Summarize branch candidate evidence from JSONL or CSV audits."""

    if path.suffix == ".csv":
        decoded_records = _candidate_records_from_csv(path=path)
        return _candidate_audit_from_decoded(path=path, decoded_records=decoded_records)
    records_by_event = _candidate_records_from_events(path=path)
    promote_event_count = len(records_by_event)
    records = [record for records in records_by_event for record in records]
    selected_mismatch_count = sum(
        1 for records in records_by_event if not _selected_matches_policy(records)
    )
    decoded_records = _decode_records(records=records, decoder=decoder)
    decoded_report = _candidate_audit_from_decoded(
        path=path,
        decoded_records=decoded_records,
    )
    return CandidateAuditReport(
        path=path,
        promote_event_count=promote_event_count,
        candidate_count=len(records),
        selected_mismatch_count=selected_mismatch_count,
        layout_only_steer_count=decoded_report.layout_only_steer_count,
        invalid_literal_tag_count=decoded_report.invalid_literal_tag_count,
    )


def main() -> None:
    """Run local native debug checks."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--events", type=Path)
    args = parser.parse_args()
    payload: dict[str, object] = {}
    tokenizer = _load_tokenizer(model_path=args.model) if args.model else None
    if args.runtime:
        payload["runtime"] = runtime_source_report(runtime_path=args.runtime).to_dict()
    if tokenizer is not None:
        payload["control_tokens"] = [
            report.to_dict() for report in control_token_reports(tokenizer=tokenizer)
        ]
    if args.events:
        decoder = None if tokenizer is None else lambda ids: str(tokenizer.decode(ids))
        payload["candidate_audit"] = candidate_audit_report(
            path=args.events,
            decoder=decoder,
        ).to_dict()
    print(json.dumps(payload, indent=2, sort_keys=True))


def _read_runtime_file(*, runtime_path: Path, relative_path: Path) -> str:
    path = runtime_path / relative_path
    assert path.exists(), f"missing runtime file: {path}"
    return path.read_text(encoding="utf-8")


def _candidate_records_from_events(*, path: Path) -> list[list[dict[str, object]]]:
    groups: list[list[dict[str, object]]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        if event.get("event") not in {"branch_promote", "branch_return"}:
            continue
        records = event.get("candidate_audit", [])
        assert isinstance(records, list), "candidate_audit must be a list"
        top_k_ids = _event_top_k_ids(event=event)
        selected_id = (
            int(event["selected_candidate_id"])
            if event.get("event") == "branch_promote"
            else None
        )
        returned_ids = _event_returned_ids(event=event)
        groups.append(
            [
                _annotated_candidate_record(
                    record=dict(record),
                    selected_id=selected_id,
                    returned_ids=returned_ids,
                    top_k_ids=top_k_ids,
                )
                for record in records
            ]
        )
    return groups


def _candidate_records_from_csv(*, path: Path) -> list[dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _candidate_audit_from_decoded(
    *, path: Path, decoded_records: list[dict[str, object]]
) -> CandidateAuditReport:
    decoded_texts = [
        str(record.get("decoded_text_one_line", "")) for record in decoded_records
    ]
    return CandidateAuditReport(
        path=path,
        promote_event_count=_csv_event_count(records=decoded_records),
        candidate_count=len(decoded_records),
        selected_mismatch_count=0,
        layout_only_steer_count=sum(
            _is_layout_only_steer(text) for text in decoded_texts
        ),
        invalid_literal_tag_count=sum(
            _has_invalid_literal_tag(text) for text in decoded_texts
        ),
    )


def _decode_records(
    *, records: list[dict[str, object]], decoder: Callable[[list[int]], str] | None
) -> list[dict[str, object]]:
    if decoder is None:
        return records
    decoded: list[dict[str, object]] = []
    for record in records:
        copied = dict(record)
        copied["decoded_text_one_line"] = decoder(_suffix_token_ids(record=record))
        decoded.append(copied)
    return decoded


def _suffix_token_ids(*, record: dict[str, object]) -> list[int]:
    raw = record.get("suffix_token_ids", [])
    assert isinstance(raw, list), "suffix_token_ids must be a list"
    return [int(token_id) for token_id in raw]


def _selected_matches_policy(records: list[dict[str, object]]) -> bool:
    selected = [record for record in records if bool(record.get("selected", False))]
    returned = [record for record in records if bool(record.get("returned", False))]
    if returned:
        return True
    if not selected:
        return False
    assert len(selected) == 1, "expected exactly one selected candidate"
    selected_record = selected[0]
    return bool(selected_record.get("in_diverse_top_k", False))


def _event_top_k_ids(*, event: dict[str, Any]) -> set[int]:
    raw = event.get("top_k_candidate_ids", [])
    if not isinstance(raw, list):
        return set()
    return {int(candidate_id) for candidate_id in raw}


def _event_returned_ids(*, event: dict[str, Any]) -> set[int]:
    raw = event.get("returned_candidate_ids", [])
    if not isinstance(raw, list):
        return set()
    return {int(candidate_id) for candidate_id in raw}


def _annotated_candidate_record(
    *,
    record: dict[str, object],
    selected_id: int | None,
    returned_ids: set[int],
    top_k_ids: set[int],
) -> dict[str, object]:
    candidate_id = _record_int(record=record, key="candidate_id")
    if "in_diverse_top_k" not in record and top_k_ids:
        record["in_diverse_top_k"] = candidate_id in top_k_ids
    record["selected"] = selected_id is not None and candidate_id == selected_id
    record["returned"] = candidate_id in returned_ids
    return record


def _record_int(*, record: dict[str, object], key: str) -> int:
    value = record[key]
    assert isinstance(value, (int, str)), f"{key} must be int-like"
    return int(value)


def _csv_event_count(*, records: list[dict[str, object]]) -> int:
    event_ids = {str(record.get("event_index", "")) for record in records}
    event_ids.discard("")
    return len(event_ids)


def _is_layout_only_steer(text: str) -> bool:
    return bool(re.fullmatch(r"<steer>(?:\\n|\s)*</steer>", text))


def _has_invalid_literal_tag(text: str) -> bool:
    content = _candidate_content(text=text)
    return any(tag in content for tag in CONTROL_TAGS)


def _candidate_content(*, text: str) -> str:
    suffix_match = re.fullmatch(r"<steer>(?P<content>.*?)</steer>\n<exec>", text, re.S)
    if suffix_match:
        return str(suffix_match.group("content"))
    steer_match = re.fullmatch(r"<steer>(?P<content>.*?)</steer>", text, re.S)
    if steer_match:
        return str(steer_match.group("content"))
    return text


def _load_tokenizer(*, model_path: Path) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)


if __name__ == "__main__":
    main()
