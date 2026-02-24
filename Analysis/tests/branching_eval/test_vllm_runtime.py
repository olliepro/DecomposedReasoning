"""Tests for vLLM serve command construction."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import cast

from branching_eval.config_types import ModelSpec, ServeConfig
from branching_eval.vllm_runtime import (
    RunningVllmServer,
    build_vllm_serve_command,
    managed_vllm_server,
    resolve_generation_model_name,
)


def test_build_serve_command_for_plain_checkpoint() -> None:
    """Plain model spec should not include LoRA serve flags."""

    model_spec = ModelSpec(model_id="non_sft", checkpoint_or_repo="Qwen/Qwen3-8B")
    serve_config = ServeConfig(tensor_parallel_size=2, gpu_memory_utilization=0.9)
    command = build_vllm_serve_command(
        model_spec=model_spec,
        serve_config=serve_config,
        port=8020,
    )
    command_text = " ".join(command)
    assert "--enable-lora" not in command_text
    assert "--scheduling-policy priority" in command_text
    assert "Qwen/Qwen3-8B" in command_text
    assert resolve_generation_model_name(model_spec=model_spec) == "Qwen/Qwen3-8B"


def test_build_serve_command_for_lora_pair() -> None:
    """LoRA spec should include expected module flags and model alias."""

    model_spec = ModelSpec(
        model_id="sft",
        base_model="Qwen/Qwen3-8B",
        lora_adapter="/tmp/lora",
        lora_name="qwen3_8b_sft",
    )
    serve_config = ServeConfig(tensor_parallel_size=2, gpu_memory_utilization=0.9)
    command = build_vllm_serve_command(
        model_spec=model_spec,
        serve_config=serve_config,
        port=8021,
    )
    command_text = " ".join(command)
    assert "--enable-lora" in command_text
    assert "--lora-modules" in command_text
    assert "qwen3_8b_sft=/tmp/lora" in command_text
    assert resolve_generation_model_name(model_spec=model_spec) == "qwen3_8b_sft"


def test_build_serve_command_with_priority_scheduler_policy() -> None:
    """Serve command should forward explicit `priority` scheduler policy."""

    model_spec = ModelSpec(model_id="non_sft", checkpoint_or_repo="Qwen/Qwen3-8B")
    serve_config = ServeConfig(scheduling_policy="priority")
    command = build_vllm_serve_command(
        model_spec=model_spec,
        serve_config=serve_config,
        port=8022,
    )
    command_text = " ".join(command)
    assert "--scheduling-policy priority" in command_text


def test_managed_server_starts_and_stops_once(monkeypatch, tmp_path: Path) -> None:
    """Managed context should call start and stop exactly once."""

    model_spec = ModelSpec(model_id="non_sft", checkpoint_or_repo="Qwen/Qwen3-8B")
    serve_config = ServeConfig()
    started_ports: list[int] = []
    stopped_ports: list[int] = []
    fake_server = RunningVllmServer(
        model_spec=model_spec,
        model_name_for_generation="Qwen/Qwen3-8B",
        base_url="http://127.0.0.1:8020/v1",
        port=8020,
        command=("vllm", "serve", "Qwen/Qwen3-8B"),
        process=cast(subprocess.Popen[str], object()),
    )

    def fake_start(
        *,
        model_spec: ModelSpec,
        serve_config: ServeConfig,
        port: int,
        log_dir: Path,
    ) -> RunningVllmServer:
        _ = model_spec, serve_config, log_dir
        started_ports.append(port)
        return fake_server

    def fake_stop(*, server: RunningVllmServer) -> None:
        stopped_ports.append(server.port)

    monkeypatch.setattr("branching_eval.vllm_runtime.start_vllm_server", fake_start)
    monkeypatch.setattr("branching_eval.vllm_runtime.stop_vllm_server", fake_stop)

    with managed_vllm_server(
        model_spec=model_spec,
        serve_config=serve_config,
        port=8020,
        log_dir=tmp_path,
    ) as running:
        assert running.base_url.endswith("/v1")

    assert started_ports == [8020]
    assert stopped_ports == [8020]
