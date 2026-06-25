"""Tests for branching-eval configuration parsing."""

from __future__ import annotations

from pathlib import Path

import yaml

from branching_eval.config_types import DecodingConfig, load_branching_eval_config


def test_config_defaults_parse_from_minimal_payload(tmp_path: Path) -> None:
    """Minimal config should parse with defaults and validate."""

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {"models": [{"model_id": "non_sft", "checkpoint_or_repo": "Qwen/Qwen3-8B"}]}
        ),
        encoding="utf-8",
    )
    config = load_branching_eval_config(config_path=config_path)
    assert config.tasks.task_names == ("aime24",)
    assert config.decoding.temperature == 0.6
    assert config.decoding.steer_temperature is None
    assert config.decoding.steer_top_p is None
    assert config.decoding.decode_chunk_tokens == 512
    assert config.decoding.top_k is None
    assert config.decoding.min_p is None
    assert config.decoding.presence_penalty is None
    assert config.decoding.repetition_penalty is None
    assert config.decoding.debug_assert_text_token_alignment is False
    assert config.branching.num_candidates == 100
    assert config.branching.steer_repetition_penalty == 1.01
    assert config.branching.repetition_checking_enabled is True
    assert config.branching.epsilon_greedy_prob == 0.05
    assert config.serve.scheduling_policy == "priority"
    assert config.serve.kv_offloading_size_gb == 64.0
    assert config.serve.kv_offloading_backend == "native"
    assert config.serve.request_timeout_seconds == 600.0
    assert config.run_matrix.include_structured_baselines is False
    assert config.run_matrix.include_epsilon_greedy is False


def test_decoding_sampling_params_parse(tmp_path: Path) -> None:
    """Thinking-mode sampling params should parse into DecodingConfig."""

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "models": [
                    {"model_id": "non_sft", "checkpoint_or_repo": "Qwen/Qwen3-8B"}
                ],
                "decoding": {
                    "temperature": 1.0,
                    "top_p": 0.95,
                    "top_k": 20,
                    "min_p": 0.0,
                    "presence_penalty": 1.5,
                    "repetition_penalty": 1.0,
                    "max_model_len": 33792,
                },
            }
        ),
        encoding="utf-8",
    )
    config = load_branching_eval_config(config_path=config_path)
    assert config.decoding == DecodingConfig(
        temperature=1.0,
        top_p=0.95,
        top_k=20,
        min_p=0.0,
        presence_penalty=1.5,
        repetition_penalty=1.0,
        max_model_len=33792,
    )


def test_output_paths_resolve_relative_to_config(tmp_path: Path) -> None:
    """Relative artifact paths should resolve from config parent directory."""

    config_dir = tmp_path / "configs"
    config_dir.mkdir(parents=True)
    config_path = config_dir / "branching.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "models": [
                    {"model_id": "non_sft", "checkpoint_or_repo": "Qwen/Qwen3-8B"}
                ],
                "artifacts": {"output_root": "runs"},
            }
        ),
        encoding="utf-8",
    )
    config = load_branching_eval_config(config_path=config_path)
    assert config.artifacts.output_root == (config_dir / "runs").resolve()


def test_scheduling_policy_parses_from_serve_block(tmp_path: Path) -> None:
    """Serve scheduling and KV offload settings should parse from YAML."""

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "models": [
                    {"model_id": "non_sft", "checkpoint_or_repo": "Qwen/Qwen3-8B"}
                ],
                "serve": {
                    "scheduling_policy": "priority",
                    "kv_offloading_size_gb": 12.0,
                    "kv_offloading_backend": "lmcache",
                    "request_timeout_seconds": 321.0,
                },
                "branching": {"steer_repetition_penalty": 1.2},
            }
        ),
        encoding="utf-8",
    )
    config = load_branching_eval_config(config_path=config_path)
    assert config.serve.scheduling_policy == "priority"
    assert config.serve.kv_offloading_size_gb == 12.0
    assert config.serve.kv_offloading_backend == "lmcache"
    assert config.serve.request_timeout_seconds == 321.0
    assert config.branching.steer_repetition_penalty == 1.2


def test_branching_repetition_checking_parses_bool_string(tmp_path: Path) -> None:
    """Branching configs should parse false string values for repeat truncation."""

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "models": [
                    {"model_id": "non_sft", "checkpoint_or_repo": "Qwen/Qwen3-8B"}
                ],
                "branching": {"repetition_checking_enabled": "False"},
            }
        ),
        encoding="utf-8",
    )

    config = load_branching_eval_config(config_path=config_path)

    assert config.branching.repetition_checking_enabled is False


def test_external_server_model_spec_parses(tmp_path: Path) -> None:
    """External server model specs should parse and validate."""

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "models": [
                    {
                        "model_id": "remote",
                        "base_url": "http://127.0.0.1:9000/v1",
                        "served_model_name": "served-qwen",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    config = load_branching_eval_config(config_path=config_path)
    model_spec = config.models[0]
    assert model_spec.uses_external_server is True
    assert model_spec.base_url == "http://127.0.0.1:9000/v1"
    assert model_spec.served_model_name == "served-qwen"


def test_clustering_server_model_spec_parses(tmp_path: Path) -> None:
    """Model specs should parse optional dedicated clustering server settings."""

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "models": [
                    {
                        "model_id": "non_sft",
                        "checkpoint_or_repo": "Qwen/Qwen3-8B",
                        "clustering_base_url": "http://127.0.0.1:9001/v1",
                        "clustering_served_model_name": "cluster-qwen",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    config = load_branching_eval_config(config_path=config_path)
    model_spec = config.models[0]
    assert model_spec.uses_external_server is False
    assert model_spec.clustering_base_url == "http://127.0.0.1:9001/v1"
    assert model_spec.clustering_served_model_name == "cluster-qwen"


def test_run_matrix_structured_baselines_parse(tmp_path: Path) -> None:
    """Structured-baseline flag should parse from YAML."""

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "models": [
                    {"model_id": "non_sft", "checkpoint_or_repo": "Qwen/Qwen3-8B"}
                ],
                "run_matrix": {
                    "include_baselines": False,
                    "include_structured_baselines": True,
                    "baseline_rollouts": 32,
                    "include_branching": False,
                },
            }
        ),
        encoding="utf-8",
    )
    config = load_branching_eval_config(config_path=config_path)
    assert config.run_matrix.include_baselines is False
    assert config.run_matrix.include_structured_baselines is True
    assert config.run_matrix.baseline_rollouts == 32
    assert config.run_matrix.include_branching is False


def test_epsilon_greedy_and_new_selector_parse(tmp_path: Path) -> None:
    """Config should parse epsilon-greedy settings and selector lists."""

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "models": [
                    {"model_id": "non_sft", "checkpoint_or_repo": "Qwen/Qwen3-8B"}
                ],
                "branching": {"epsilon_greedy_prob": 0.2},
                "run_matrix": {
                    "include_baselines": False,
                    "include_branching": True,
                    "include_epsilon_greedy": True,
                    "selectors": ["embed_diverse_topk_random", "random"],
                },
            }
        ),
        encoding="utf-8",
    )

    config = load_branching_eval_config(config_path=config_path)

    assert config.branching.epsilon_greedy_prob == 0.2
    assert config.run_matrix.include_epsilon_greedy is True
    assert config.run_matrix.selectors == ("embed_diverse_topk_random", "random")


def test_debug_alignment_flag_parses_from_decoding_block(tmp_path: Path) -> None:
    """Decoding debug alignment flag should parse from YAML."""

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "models": [
                    {"model_id": "non_sft", "checkpoint_or_repo": "Qwen/Qwen3-8B"}
                ],
                "decoding": {"debug_assert_text_token_alignment": True},
            }
        ),
        encoding="utf-8",
    )
    config = load_branching_eval_config(config_path=config_path)
    assert config.decoding.debug_assert_text_token_alignment is True


def test_steer_sampling_parses_and_resolves_by_request_kind(tmp_path: Path) -> None:
    """Steer sampling overrides should affect only steer-generation requests."""

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "models": [{"model_id": "sft", "checkpoint_or_repo": "Qwen/Qwen3-8B"}],
                "decoding": {
                    "temperature": 0.3,
                    "steer_temperature": 0.8,
                    "top_p": 0.91,
                    "steer_top_p": 0.77,
                },
            }
        ),
        encoding="utf-8",
    )

    config = load_branching_eval_config(config_path=config_path)

    assert config.decoding.temperature == 0.3
    assert config.decoding.steer_temperature == 0.8
    assert config.decoding.top_p == 0.91
    assert config.decoding.steer_top_p == 0.77
    assert config.decoding.request_temperature(request_kind="decode_chunk") == 0.3
    assert config.decoding.request_top_p(request_kind="decode_chunk") == 0.91
    assert (
        config.decoding.request_temperature(request_kind="steer_single_candidate")
        == 0.8
    )
    assert config.decoding.request_top_p(request_kind="steer_single_candidate") == 0.77
    assert (
        config.decoding.request_temperature(
            request_kind="candidate_pool_steer_boundary"
        )
        == 0.8
    )
    assert (
        config.decoding.request_top_p(request_kind="candidate_pool_steer_boundary")
        == 0.77
    )


def test_initial_assistant_prefix_parses_from_decoding_block(tmp_path: Path) -> None:
    """Initial assistant prefix should parse from YAML decoding config."""

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "models": [{"model_id": "sft", "checkpoint_or_repo": "Qwen/Qwen3-8B"}],
                "decoding": {"initial_assistant_prefix": "<think>\n<steer>"},
            }
        ),
        encoding="utf-8",
    )

    config = load_branching_eval_config(config_path=config_path)

    assert config.decoding.initial_assistant_prefix == "<think>\n<steer>"


def test_candidate_pool_temperature_keeps_previous_default() -> None:
    """Candidate-pool sampling should stay at 1.0 unless explicitly overridden."""

    assert (
        DecodingConfig().request_temperature(
            request_kind="candidate_pool_steer_boundary"
        )
        == 1.0
    )
