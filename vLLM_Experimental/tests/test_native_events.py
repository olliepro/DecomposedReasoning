import json
from pathlib import Path

from vllm_experimental.run_benchmark_job import read_native_event_metrics


def write_events(*, path: Path, events: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(event, sort_keys=True) + "\n" for event in events),
        encoding="utf-8",
    )


def test_read_native_event_metrics_counts_start_and_promote(tmp_path: Path) -> None:
    path = tmp_path / "native_events.jsonl"
    write_events(
        path=path,
        events=[
            {
                "async_tokens_to_discard": 1,
                "candidate_count": 2,
                "event": "branch_start",
                "fork_tokens": 528,
                "shared_blocks": 1,
            },
            {
                "candidate_count": 2,
                "event": "branch_wave_start",
                "fork_tokens": 528,
                "shared_blocks": 1,
            },
            {
                "children_freed": 4,
                "diversity_vector_source": "model_hidden_state",
                "event": "branch_promote",
                "branch_depth_limit": 4,
                "branch_depth_used": 2,
                "hidden_vector_child_count": 4,
                "pool_hidden_pairwise_diversity": 0.25,
                "selected_candidate_id": 1,
                "selected_hidden_diversity": 0.5,
                "top_k_candidate_ids": [1, 3],
                "unique_candidate_count": 3,
                "candidate_bound": 3,
            },
        ],
    )

    metrics = read_native_event_metrics(path=path)

    assert metrics["boundary_fire_count"] == 1
    assert metrics["branch_count"] == 4
    assert metrics["prefill_tokens_avoided"] == 2112
    assert metrics["kv_blocks_allocated"] == 4
    assert metrics["kv_blocks_copied"] == 0
    assert metrics["kv_blocks_freed"] == 4
    assert metrics["async_tokens_discarded"] == 1
    assert metrics["selected_candidate_ids"] == [1]
    assert metrics["returned_candidate_ids"] == []
    assert metrics["returned_branch_counts"] == []
    assert metrics["branch_depth_used"] == [2]
    assert metrics["branch_depth_limits"] == [4]
    assert metrics["top_k_candidate_ids"] == [[1, 3]]
    assert metrics["unique_candidate_counts"] == [3]
    assert metrics["candidate_bounds"] == [3]
    assert metrics["diversity_vector_source"] == "model_hidden_state"
    assert metrics["hidden_vector_child_count"] == 4
    assert metrics["pool_hidden_pairwise_diversity"] == 0.25
    assert metrics["selected_hidden_diversity"] == 0.5


def test_read_native_event_metrics_missing_file_is_zero(tmp_path: Path) -> None:
    metrics = read_native_event_metrics(path=tmp_path / "missing.jsonl")

    assert metrics["boundary_fire_count"] == 0
    assert metrics["branch_count"] == 0
    assert metrics["async_tokens_discarded"] == 0
    assert metrics["branch_pool_queued_count"] == 0
    assert metrics["branch_pool_blocked_count"] == 0
    assert metrics["branch_pool_admitted_count"] == 0
    assert metrics["max_live_branch_pools"] == 0
    assert metrics["max_queued_branch_pools"] == 0
    assert metrics["min_branch_free_blocks"] == 0
    assert metrics["min_branch_seq_slots"] == 0
    assert metrics["selected_candidate_ids"] == []
    assert metrics["returned_candidate_ids"] == []
    assert metrics["returned_branch_counts"] == []
    assert metrics["branch_depth_used"] == []
    assert metrics["branch_depth_limits"] == []
    assert metrics["top_k_candidate_ids"] == []
    assert metrics["unique_candidate_counts"] == []
    assert metrics["candidate_bounds"] == []
    assert metrics["diversity_vector_source"] == "none"
    assert metrics["pool_hidden_pairwise_diversity"] == 0.0


def test_read_native_event_metrics_counts_branch_return(tmp_path: Path) -> None:
    path = tmp_path / "native_events.jsonl"
    write_events(
        path=path,
        events=[
            {
                "async_tokens_to_discard": 1,
                "candidate_count": 3,
                "event": "branch_start",
                "fork_tokens": 128,
                "shared_blocks": 2,
            },
            {
                "candidate_bound": 3,
                "branch_depth_limit": 4,
                "branch_depth_used": 1,
                "children_freed": 3,
                "diversity_vector_source": "model_hidden_state",
                "event": "branch_return",
                "hidden_vector_child_count": 3,
                "pool_hidden_pairwise_diversity": 0.75,
                "returned_branch_count": 2,
                "returned_candidate_ids": [4, 7],
                "top_k_candidate_ids": [4, 7, 8],
                "unique_candidate_count": 3,
            },
        ],
    )

    metrics = read_native_event_metrics(path=path)

    assert metrics["branch_count"] == 3
    assert metrics["kv_blocks_freed"] == 3
    assert metrics["selected_candidate_ids"] == []
    assert metrics["returned_candidate_ids"] == [[4, 7]]
    assert metrics["returned_branch_counts"] == [2]
    assert metrics["branch_depth_used"] == [1]
    assert metrics["branch_depth_limits"] == [4]
    assert metrics["top_k_candidate_ids"] == [[4, 7, 8]]
    assert metrics["unique_candidate_counts"] == [3]
    assert metrics["candidate_bounds"] == [3]


def test_read_native_event_metrics_counts_dynamic_admission(
    tmp_path: Path,
) -> None:
    path = tmp_path / "native_events.jsonl"
    write_events(
        path=path,
        events=[
            {
                "async_tokens_to_discard": 1,
                "candidate_count": 0,
                "event": "branch_start",
                "fork_tokens": 528,
                "shared_blocks": 33,
            },
            {
                "candidate_count": 50,
                "estimated_pool_blocks": 150,
                "event": "branch_pool_queued",
                "free_blocks": 1000,
                "live_branch_pool_count": 0,
                "queued_branch_pool_count": 1,
                "seq_slots": 64,
            },
            {
                "candidate_count": 50,
                "estimated_pool_blocks": 150,
                "event": "branch_pool_blocked",
                "free_blocks": 140,
                "live_branch_pool_count": 0,
                "queued_branch_pool_count": 1,
                "seq_slots": 64,
            },
            {
                "candidate_count": 50,
                "event": "branch_pool_admitted",
                "fork_tokens": 528,
                "free_blocks": 900,
                "live_branch_pool_count": 1,
                "queued_branch_pool_count": 0,
                "seq_slots": 64,
                "shared_blocks": 33,
            },
            {
                "candidate_bound": 6,
                "branch_depth_limit": 4,
                "branch_depth_used": 1,
                "children_freed": 50,
                "diversity_vector_source": "model_hidden_state",
                "event": "branch_promote",
                "hidden_vector_child_count": 50,
                "pool_hidden_pairwise_diversity": 0.75,
                "selected_candidate_id": 7,
                "selected_hidden_diversity": 0.4,
                "top_k_candidate_ids": [0, 7, 9],
                "unique_candidate_count": 19,
            },
        ],
    )

    metrics = read_native_event_metrics(path=path)

    assert metrics["boundary_fire_count"] == 1
    assert metrics["branch_count"] == 50
    assert metrics["branch_pool_queued_count"] == 1
    assert metrics["branch_pool_blocked_count"] == 1
    assert metrics["branch_pool_admitted_count"] == 1
    assert metrics["max_live_branch_pools"] == 1
    assert metrics["max_queued_branch_pools"] == 1
    assert metrics["min_branch_free_blocks"] == 140
    assert metrics["min_branch_seq_slots"] == 64
    assert metrics["prefill_tokens_avoided"] == 26_400
    assert metrics["kv_blocks_allocated"] == 1650
    assert metrics["kv_blocks_freed"] == 50
