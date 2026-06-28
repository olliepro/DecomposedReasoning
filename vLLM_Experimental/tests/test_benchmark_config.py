"""Tests for benchmark config generation."""

from __future__ import annotations

from typing import cast

from vllm_experimental.benchmark_config import build_config
from vllm_experimental.run_benchmark_job import (
    ASSISTANT_PREFILL,
    DEFAULT_MATH_SYSTEM_PROMPT,
    attach_prompt_prefill,
    batch_shape_warmup_enabled,
    batch_shape_warmup_prompts,
    batch_shape_warmup_token_count,
    chat_prompts,
    ensure_assistant_prefill,
    maybe_pad_raw_prompts,
    native_frontier_enabled,
    native_scheduler_enabled,
    verbalized_token_scripts,
)
from vllm_experimental.types import DEFAULT_MODEL_PATH


class FakeTokenizer:
    """Minimal tokenizer for deterministic script-token tests."""

    def apply_chat_template(
        self,
        *,
        conversation: list[dict[str, str]],
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        assert not tokenize
        assert add_generation_prompt
        assert conversation[0] == {
            "role": "system",
            "content": DEFAULT_MATH_SYSTEM_PROMPT,
        }
        assert conversation[1]["role"] == "user"
        return f"system:{conversation[0]['content']}\nuser:{conversation[1]['content']}\nassistant:\n"

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        assert not add_special_tokens
        return [ord(char) for char in text]


def test_default_sweep_shape() -> None:
    config = build_config(run_name="smoke", mode="grammar_temp")
    rows = config.sweep_rows()
    assert len(rows) == 15
    assert rows[0]["model_path"] == str(DEFAULT_MODEL_PATH)
    assert rows[0]["doc_ids"] == [0, 1, 2, 3, 4, 5, 6, 7]
    assert rows[0]["max_num_seqs"] == 384
    assert config.params.native_branch_wave_size == 50
    assert config.params.native_branch_dynamic_admission
    assert config.params.native_branch_min_free_blocks == 256
    assert config.params.native_branch_free_block_fraction == 0.05
    assert config.params.native_branch_seq_reserve == 8
    assert config.params.native_branch_priority_boost == 1000


def test_off_policy_defaults_to_fanout_two() -> None:
    config = build_config(run_name="smoke", mode="eps_off_policy_verbalized")
    assert config.params.branch_fanout == 2
    assert config.params.branch_depth == 4
    assert config.params.native_scheduler_kv_fork
    config.params.validate()


def test_off_policy_honors_fanout_env(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("BRANCH_FANOUT", "1")
    config = build_config(run_name="smoke", mode="eps_off_policy_verbalized")
    assert config.params.branch_fanout == 1
    config.params.validate()


def test_grammar_cap_env_overrides(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("MAX_STEER_TOKENS", "2")
    monkeypatch.setenv("MAX_EXEC_TOKENS", "3")
    monkeypatch.setenv("MAX_MODEL_LEN", "4096")
    monkeypatch.setenv("MAX_NUM_BATCHED_TOKENS", "8192")
    monkeypatch.setenv("MAX_NUM_SEQS", "64")
    monkeypatch.setenv("NATIVE_BRANCH_WAVE_SIZE", "4")
    monkeypatch.setenv("NATIVE_BRANCH_DYNAMIC_ADMISSION", "0")
    monkeypatch.setenv("NATIVE_BRANCH_MIN_FREE_BLOCKS", "12")
    monkeypatch.setenv("NATIVE_BRANCH_FREE_BLOCK_FRACTION", "0.25")
    monkeypatch.setenv("NATIVE_BRANCH_SEQ_RESERVE", "3")
    monkeypatch.setenv("NATIVE_BRANCH_PRIORITY_BOOST", "25")
    monkeypatch.setenv("NATIVE_BRANCH_BLOCK_SAFETY_MULTIPLIER", "1.5")
    monkeypatch.setenv("NATIVE_BRANCH_BLOCKED_LOG_INTERVAL_S", "2.5")
    monkeypatch.setenv("NATIVE_BRANCH_MAX_LIVE_POOLS", "6")
    monkeypatch.setenv("NATIVE_BRANCH_MAX_QUEUED_POOLS", "11")
    monkeypatch.setenv("BRANCH_DEPTH", "7")
    config = build_config(run_name="smoke", mode="eps_on_policy_diverse")
    assert config.params.max_steer_tokens == 2
    assert config.params.max_exec_tokens == 3
    assert config.max_model_len == 4096
    assert config.max_num_batched_tokens == 8192
    assert config.params.max_model_len == 4096
    assert config.params.max_num_batched_tokens == 8192
    assert config.max_num_seqs == 64
    assert config.params.native_branch_wave_size == 4
    assert not config.params.native_branch_dynamic_admission
    assert config.params.native_branch_min_free_blocks == 12
    assert config.params.native_branch_free_block_fraction == 0.25
    assert config.params.native_branch_seq_reserve == 3
    assert config.params.native_branch_priority_boost == 25
    assert config.params.native_branch_block_safety_multiplier == 1.5
    assert config.params.native_branch_blocked_log_interval_s == 2.5
    assert config.params.native_branch_max_live_pools == 6
    assert config.params.native_branch_max_queued_pools == 11
    assert config.params.branch_depth == 7
    assert config.sweep_rows()[0]["max_num_seqs"] == 64


def test_hidden_state_diversity_env_overrides(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("DIVERSITY_VECTOR_SOURCE", "model_hidden_state")
    config = build_config(run_name="smoke", mode="eps_on_policy_diverse")
    assert config.params.diversity_vector_source == "model_hidden_state"
    assert config.params.native_scheduler_kv_fork
    row_tree = cast(dict[str, object], config.sweep_rows()[0]["tree_search"])
    assert row_tree["diversity_vector_source"] == "model_hidden_state"


def test_prompt_padding_env(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("PROMPT_PAD_REPEATS", "2")
    raw_prompts = maybe_pad_raw_prompts(raw_prompts=["prompt"])
    assert raw_prompts == ["prompt\n\nfiller filler"]
    chat_prompt = chat_prompts(prompts=raw_prompts, tokenizer=FakeTokenizer())[0]
    assert chat_prompt.endswith(f"assistant:\n{ASSISTANT_PREFILL}")
    assert not chat_prompt.endswith("filler filler")


def test_chat_prompts_prefill_assistant_think_newline() -> None:
    prompt = chat_prompts(prompts=["question"], tokenizer=FakeTokenizer())[0]
    assert DEFAULT_MATH_SYSTEM_PROMPT in prompt
    assert prompt.endswith(f"assistant:\n{ASSISTANT_PREFILL}")


def test_ensure_assistant_prefill_normalizes_template_suffixes() -> None:
    assert ensure_assistant_prefill(rendered_prompt="assistant:\n") == (
        f"assistant:\n{ASSISTANT_PREFILL}"
    )
    assert ensure_assistant_prefill(rendered_prompt="assistant:\n<think>") == (
        f"assistant:\n{ASSISTANT_PREFILL}"
    )
    assert ensure_assistant_prefill(
        rendered_prompt=f"assistant:\n{ASSISTANT_PREFILL}"
    ) == (f"assistant:\n{ASSISTANT_PREFILL}")


def test_attach_prompt_prefill_sets_replay_tokens() -> None:
    tree_search: dict[str, object] = {}
    attach_prompt_prefill(tree_search=tree_search, tokenizer=FakeTokenizer())
    assert tree_search["prefix_output_token_ids"] == [
        ord("<"),
        ord("t"),
        ord("h"),
        ord("i"),
        ord("n"),
        ord("k"),
        ord(">"),
        ord("\n"),
    ]


def test_off_policy_native_scheduler_routing() -> None:
    row: dict[str, object] = {
        "tree_search": {
            "branch_fanout": 1,
            "native_scheduler_kv_fork": True,
        }
    }
    assert native_scheduler_enabled(row=row, mode="eps_off_policy_verbalized")
    assert not native_frontier_enabled(row=row, mode="eps_off_policy_verbalized")


def test_multi_fanout_uses_native_frontier() -> None:
    row: dict[str, object] = {
        "tree_search": {
            "branch_fanout": 2,
            "native_scheduler_kv_fork": True,
        }
    }
    assert native_scheduler_enabled(row=row, mode="eps_on_policy_diverse")
    assert native_frontier_enabled(row=row, mode="eps_on_policy_diverse")


def test_batch_shape_warmup_defaults_to_first_chunk_shape() -> None:
    row: dict[str, object] = {
        "prompt_concurrency": 5,
        "request_prompt_batch_size": 3,
    }
    prompts = batch_shape_warmup_prompts(prompts=["a", "b"], row=row)
    assert prompts == ["a", "b", "a"]


def test_batch_shape_warmup_env(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    assert batch_shape_warmup_enabled()
    monkeypatch.setenv("BATCH_SHAPE_WARMUP_ENABLED", "0")
    assert not batch_shape_warmup_enabled()
    monkeypatch.setenv("BATCH_SHAPE_WARMUP_MAX_TOKENS", "9")
    assert batch_shape_warmup_token_count(max_tokens=4) == 4
    assert batch_shape_warmup_token_count(max_tokens=12) == 9


def test_verbalized_scripts_are_tokenized_for_scheduler_payload() -> None:
    scripts = verbalized_token_scripts(tokenizer=FakeTokenizer())
    enumerate_text = "".join(chr(token_id) for token_id in scripts["enumerate"]["3"])
    continue_text = "".join(chr(token_id) for token_id in scripts["continue"]["2"])
    assert enumerate_text == (
        "Enumerate 3 distinct options for the immediate next "
        "decision/step</steer>\n<exec>"
    )
    assert continue_text == ("Proceed with option 2</steer>\n<exec>Let's do option 2:")
