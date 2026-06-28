import json
from pathlib import Path

from vllm_experimental.native_debug import (
    candidate_audit_report,
    control_token_reports,
    runtime_source_report,
)


class FakeTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        assert not add_special_tokens
        mapping = {
            "<think>": [1],
            "</think>": [2],
            "<steer>": [3],
            "</steer>": [4],
            "<exec>": [5],
            "</exec>": [6],
            "\n": [7],
        }
        return mapping[text]

    def decode(self, token_ids: list[int]) -> str:
        mapping = {
            1: "<think>",
            2: "</think>",
            3: "<steer>",
            4: "</steer>",
            5: "<exec>",
            6: "</exec>",
            7: "\n",
            100: "content",
        }
        return "".join(mapping[token_id] for token_id in token_ids)


def test_runtime_source_report_checks_native_child_invariants(tmp_path: Path) -> None:
    scheduler = tmp_path / "vllm/v1/core/sched/scheduler.py"
    engine_init = tmp_path / "vllm/v1/engine/__init__.py"
    output_processor = tmp_path / "vllm/v1/engine/output_processor.py"
    runner = tmp_path / "vllm/v1/worker/gpu_model_runner.py"
    single_type_kv = tmp_path / "vllm/v1/core/single_type_kv_cache_manager.py"
    kv_coordinator = tmp_path / "vllm/v1/core/kv_cache_coordinator.py"
    kv_manager = tmp_path / "vllm/v1/core/kv_cache_manager.py"
    scheduler.parent.mkdir(parents=True)
    engine_init.parent.mkdir(parents=True)
    runner.parent.mkdir(parents=True)
    scheduler.write_text(
        "\n".join(
            [
                "prompt_token_ids=list(parent.prompt_token_ids) + list(parent.output_token_ids)",
                'parent_prefix_output_ids = child_payload.get("prefix_output_token_ids", [])',
                "[int(token_id) for token_id in parent_prefix_output_ids]",
                "+ list(parent.output_token_ids)",
                '"eps_on_policy_diverse"',
                '"eps_off_policy_verbalized"',
                'steer_open = int(token_payload["steer_open"])',
                "request._output_token_ids[-1] == steer_open",
                "(candidate_id, [], [])",
                'newline = token_payload.get("newline")',
                "request._output_token_ids[-2] == exec_close",
                "request._output_token_ids[-1] == int(newline)",
                'return [int(token_payload["exec_open"])]',
                'return int(payload.get("max_steer_tokens", 30)) + 3',
                "if think_close in request._output_token_ids:",
                "async_tokens_to_discard = parent.num_output_placeholders",
                "parent.async_tokens_to_discard = async_tokens_to_discard",
                "parent.num_output_placeholders = 0",
                "self.prev_step_scheduled_req_ids.discard(parent_id)",
                "child_params.max_tokens = len(initial_script) + branch_budget + len(followup_script)",
                "child.num_computed_tokens = fork_tokens",
                "if child.num_computed_tokens == child.num_tokens:",
                "child.num_computed_tokens = child.num_tokens - 1",
                "self.waiting.add_request(child)",
                "if request.num_computed_tokens == 0:",
                "num_computed_tokens = request.num_computed_tokens",
                "_vllm_exp_diverse_top_k_child_ids",
                '"diverse_top_k_sample"',
                "top_k_candidate_ids=self._vllm_exp_candidate_ids(top_k_child_ids)",
                "unique_candidate_count=unique_count",
                "_vllm_exp_finish_parent_if_limit_reached",
                "self._vllm_exp_suspended_parent_ids.discard(parent.request_id)",
                "parent_finished_by_limit",
                "finish_reason=parent_finish_reason",
                "and not parent_finished_by_limit",
                "_vllm_exp_start_off_policy_enumeration",
                '"off_policy_enumerate_inline"',
                "self._vllm_exp_force_parent_script(parent, enum_script)",
                '"off_policy_continue_inline"',
                "self._vllm_exp_force_parent_script(parent, continue_script)",
                "_vllm_exp_off_policy_branch_options.pop",
                "self._vllm_exp_continue_script_after_boundary(payload, option_number)",
                '"off_policy_continue_inline_fallback"',
                "if self._vllm_exp_start_branches(parent):",
                "self._vllm_exp_off_policy_branch_options.pop(parent.request_id, None)",
                "_vllm_exp_return_child_ids",
                "return top_k_child_ids[:branch_fanout]",
                "return child_ids[:branch_fanout]",
                '"branch_return"',
                "parent.status = RequestStatus.FINISHED_STOPPED",
                "vllm_experimental_branch_outputs=branch_outputs",
                "_vllm_exp_parent_branch_fires",
                "_vllm_exp_branch_depth_used",
                '"branch_skip_depth_limit"',
                'branch_fanout = int(payload.get("branch_fanout", 1))',
                "if branch_fanout > 1 and branch_depth_used >= branch_depth_limit:",
                'payload.get("branch_depth", 4)',
                'payload.get("branch_depth_start", 0)',
                "def _vllm_exp_consume_branch_depth(self, request: Request)",
                "started = self._vllm_exp_start_branches(request)",
                "if started:\n            self._vllm_exp_consume_branch_depth(request)",
                "if sample >= fire_rate:",
                "return started",
                "_vllm_exp_child_products: dict[str, dict[str, Any]]",
                "_vllm_exp_store_child_product(child)",
                "self._free_request(child)",
                "_vllm_exp_child_suffix_token_ids",
                "parent = self.requests[parent_id]",
                "        payload = self._vllm_exp_payload(parent)",
                'assert payload is not None, "native branch parent payload missing"',
                "def _vllm_exp_promote_winner(self, parent: Request, winner_id: str)",
                "replay_from_tokens = self._vllm_exp_branch_fork_tokens.get",
                "self.kv_cache_manager.trim_blocks(",
                "kv_blocks_trimmed=trimmed_blocks",
                "return trimmed_blocks",
                "_vllm_exp_branch_pool_queue: deque[str] = deque()",
                "_vllm_exp_try_admit_branch_pools()",
                '"branch_pool_queued"',
                "def _vllm_exp_launch_branch_pool(self, parent: Request)",
                "pool_plan = list(pending)",
                "pending.clear()",
                "fast_lane=True",
                '"branch_pool_queued",\n                candidate_count=len(branch_plan),\n'
                '                reason="queued",\n            )\n'
                "            self._vllm_exp_try_admit_branch_pools()",
                "_vllm_exp_estimated_pool_blocks",
                "native_branch_min_free_blocks",
                "native_branch_free_block_fraction",
                "native_branch_seq_reserve",
                'reason = "free_blocks"',
                "self.waiting.prepend_request(child)",
                "native_branch_priority_boost",
                "SchedulingPolicy.FCFS",
                '"branch_pool_blocked"',
                "native_branch_blocked_log_interval_s",
                "self._vllm_exp_last_blocked_log_at",
                "self._vllm_exp_branch_pool_queue.append(parent_id)",
                "self._vllm_exp_try_admit_branch_pools()",
                "native_branch_max_live_pools",
                "native_branch_max_queued_pools",
                'reason = "live_pool_limit"',
            ]
        ),
        encoding="utf-8",
    )
    engine_init.write_text(
        "vllm_experimental_branch_outputs: list[dict[str, Any]] | None",
        encoding="utf-8",
    )
    output_processor.write_text(
        "\n".join(
            [
                "engine_core_output.vllm_experimental_branch_outputs",
                "_vllm_exp_make_branch_request_output",
            ]
        ),
        encoding="utf-8",
    )
    runner.write_text(
        "num_computed_tokens=new_req_data.num_computed_tokens",
        encoding="utf-8",
    )
    single_type_kv.write_text(
        "\n".join(
            [
                "def trim_blocks(self, request_id: str, num_computed_tokens: int) -> int:",
                "self.block_pool.free_blocks(reversed(stale_blocks))",
                "self.num_cached_block[child_request_id] = num_shared_blocks",
                "self.num_cached_block[target_request_id] = len(source_blocks)",
                "self.num_cached_block[request_id] = retained_count",
            ]
        ),
        encoding="utf-8",
    )
    kv_coordinator.write_text(
        "\n".join(
            [
                "def trim_blocks(self, request_id: str, num_computed_tokens: int) -> int:",
                "manager.trim_blocks(request_id, num_computed_tokens)",
            ]
        ),
        encoding="utf-8",
    )
    kv_manager.write_text(
        "def trim_blocks(self, request_id: str, num_computed_tokens: int) -> int:",
        encoding="utf-8",
    )

    report = runtime_source_report(runtime_path=tmp_path)

    assert report.failures == []


def test_candidate_audit_report_checks_selection_and_blank_spans(
    tmp_path: Path,
) -> None:
    path = tmp_path / "native_events.jsonl"
    path.write_text(
        json.dumps(
            {
                "event": "branch_promote",
                "selected_candidate_id": 1,
                "top_k_candidate_ids": [1],
                "candidate_audit": [
                    {
                        "candidate_id": 0,
                        "suffix_token_ids": [3, 7, 4],
                    },
                    {
                        "candidate_id": 1,
                        "suffix_token_ids": [3, 100, 4],
                    },
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = candidate_audit_report(path=path, decoder=FakeTokenizer().decode)

    assert report.promote_event_count == 1
    assert report.candidate_count == 2
    assert report.selected_mismatch_count == 0
    assert report.layout_only_steer_count == 1
    assert report.invalid_literal_tag_count == 0


def test_candidate_audit_accepts_random_selection_inside_top_k(
    tmp_path: Path,
) -> None:
    path = tmp_path / "native_events.jsonl"
    path.write_text(
        json.dumps(
            {
                "event": "branch_promote",
                "selected_candidate_id": 0,
                "candidate_audit": [
                    {
                        "candidate_id": 0,
                        "in_diverse_top_k": True,
                        "suffix_token_ids": [3, 100, 4],
                    },
                    {
                        "candidate_id": 1,
                        "in_diverse_top_k": True,
                        "suffix_token_ids": [3, 100, 4],
                    },
                    {
                        "candidate_id": 2,
                        "in_diverse_top_k": False,
                        "suffix_token_ids": [3, 100, 4],
                    },
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = candidate_audit_report(path=path, decoder=FakeTokenizer().decode)

    assert report.selected_mismatch_count == 0


def test_candidate_audit_rejects_random_selection_outside_top_k(
    tmp_path: Path,
) -> None:
    path = tmp_path / "native_events.jsonl"
    path.write_text(
        json.dumps(
            {
                "event": "branch_promote",
                "selected_candidate_id": 2,
                "top_k_candidate_ids": [0, 1],
                "candidate_audit": [
                    {
                        "candidate_id": 0,
                        "suffix_token_ids": [3, 100, 4],
                    },
                    {
                        "candidate_id": 1,
                        "suffix_token_ids": [3, 100, 4],
                    },
                    {
                        "candidate_id": 2,
                        "suffix_token_ids": [3, 100, 4],
                    },
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = candidate_audit_report(path=path, decoder=FakeTokenizer().decode)

    assert report.selected_mismatch_count == 1


def test_candidate_audit_accepts_branch_return_events(tmp_path: Path) -> None:
    path = tmp_path / "native_events.jsonl"
    path.write_text(
        json.dumps(
            {
                "event": "branch_return",
                "returned_candidate_ids": [2, 3],
                "top_k_candidate_ids": [2, 3],
                "candidate_audit": [
                    {
                        "candidate_id": 0,
                        "suffix_token_ids": [3, 100, 4],
                    },
                    {
                        "candidate_id": 2,
                        "suffix_token_ids": [3, 100, 4],
                    },
                    {
                        "candidate_id": 3,
                        "suffix_token_ids": [3, 100, 4],
                    },
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = candidate_audit_report(path=path, decoder=FakeTokenizer().decode)

    assert report.promote_event_count == 1
    assert report.selected_mismatch_count == 0


def test_candidate_audit_allows_structural_on_policy_exec_open(
    tmp_path: Path,
) -> None:
    path = tmp_path / "native_events.jsonl"
    path.write_text(
        json.dumps(
            {
                "event": "branch_promote",
                "selected_candidate_id": 0,
                "top_k_candidate_ids": [0],
                "candidate_audit": [
                    {
                        "candidate_id": 0,
                        "suffix_token_ids": [3, 100, 4, 7, 5],
                    },
                    {
                        "candidate_id": 1,
                        "suffix_token_ids": [3, 5, 4, 7, 5],
                    },
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = candidate_audit_report(path=path, decoder=FakeTokenizer().decode)

    assert report.layout_only_steer_count == 0
    assert report.invalid_literal_tag_count == 1


def test_control_token_reports_are_atomic() -> None:
    reports = control_token_reports(tokenizer=FakeTokenizer())

    assert all(report.is_atomic for report in reports)
