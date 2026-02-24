"""vLLM process lifecycle helpers for checkpoint-driven branching eval runs."""

from __future__ import annotations

import json
import subprocess
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
from urllib import error as urllib_error
from urllib import request as urllib_request

from branching_eval.config_types import ModelSpec, ServeConfig


@dataclass(frozen=True)
class RunningVllmServer:
    """One running vLLM server process.

    Args:
        model_spec: Source model specification.
        model_name_for_generation: Model name used in completions requests.
        base_url: OpenAI-compatible base URL.
        port: Bound TCP port.
        command: Launch command.
        process: Process handle.

    Returns:
        Dataclass representing one server runtime.
    """

    model_spec: ModelSpec
    model_name_for_generation: str
    base_url: str
    port: int
    command: tuple[str, ...]
    process: subprocess.Popen[str]


def build_vllm_serve_command(
    *, model_spec: ModelSpec, serve_config: ServeConfig, port: int
) -> tuple[str, ...]:
    """Build the `vllm serve` command for one model spec.

    Args:
        model_spec: Model serving specification.
        serve_config: Serving configuration.
        port: Target server port.

    Returns:
        Tuple command suitable for `subprocess.Popen`.
    """

    command = [
        "vllm",
        "serve",
        model_spec.served_model_arg(),
        "--host",
        serve_config.host,
        "--port",
        str(port),
        "--dtype",
        serve_config.dtype,
        "--scheduling-policy",
        serve_config.scheduling_policy,
        "--tensor-parallel-size",
        str(serve_config.tensor_parallel_size),
        "--gpu-memory-utilization",
        str(serve_config.gpu_memory_utilization),
        "--max-logprobs",
        str(serve_config.max_logprobs),
    ]
    if serve_config.trust_remote_code:
        command.append("--trust-remote-code")
    if not model_spec.has_lora:
        return tuple(command)
    command.extend(
        ["--enable-lora", "--lora-modules", _lora_module_arg(model_spec=model_spec)]
    )
    return tuple(command)


def resolve_generation_model_name(*, model_spec: ModelSpec) -> str:
    """Resolve request-time model name used for generation calls.

    Args:
        model_spec: Model serving specification.

    Returns:
        Generation-time model id.
    """

    if model_spec.has_lora:
        assert model_spec.lora_name is not None, "lora_name required for LoRA"
        return model_spec.lora_name
    return model_spec.served_model_arg()


def start_vllm_server(
    *,
    model_spec: ModelSpec,
    serve_config: ServeConfig,
    port: int,
    log_dir: Path,
) -> RunningVllmServer:
    """Start one vLLM server and wait for readiness.

    Args:
        model_spec: Model serving specification.
        serve_config: Serving configuration.
        port: Target TCP port.
        log_dir: Directory where stdout/stderr logs are written.

    Returns:
        Running server metadata.
    """

    command = build_vllm_serve_command(
        model_spec=model_spec,
        serve_config=serve_config,
        port=port,
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"serve_{model_spec.model_id}_{port}.log"
    log_handle = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        command,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    base_url = f"http://{serve_config.host}:{port}/v1"
    try:
        wait_for_server(
            base_url=base_url,
            timeout_seconds=serve_config.startup_timeout_seconds,
            poll_interval_seconds=serve_config.poll_interval_seconds,
            process=process,
        )
    except Exception:
        _terminate_process(process=process)
        raise
    return RunningVllmServer(
        model_spec=model_spec,
        model_name_for_generation=resolve_generation_model_name(model_spec=model_spec),
        base_url=base_url,
        port=port,
        command=command,
        process=process,
    )


def wait_for_server(
    *,
    base_url: str,
    timeout_seconds: float,
    poll_interval_seconds: float,
    process: subprocess.Popen[str],
) -> None:
    """Wait for `/models` health endpoint to respond with JSON.

    Args:
        base_url: OpenAI-compatible server base URL (`.../v1`).
        timeout_seconds: Max wait duration.
        poll_interval_seconds: Poll sleep interval.
        process: Server process handle.

    Returns:
        None.
    """

    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("vLLM server exited before becoming ready")
        if is_server_ready(base_url=base_url):
            return
        time.sleep(max(0.1, poll_interval_seconds))
    raise TimeoutError(f"Timed out waiting for vLLM server: {base_url}")


def is_server_ready(*, base_url: str) -> bool:
    """Check whether vLLM `/models` endpoint is reachable.

    Args:
        base_url: OpenAI-compatible server base URL (`.../v1`).

    Returns:
        True when the endpoint returns a valid JSON payload.
    """

    url = f"{base_url}/models"
    request = urllib_request.Request(url=url, method="GET")
    try:
        with urllib_request.urlopen(request, timeout=2.0) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (
        urllib_error.URLError,
        urllib_error.HTTPError,
        TimeoutError,
        json.JSONDecodeError,
    ):
        return False
    return isinstance(payload, dict)


def stop_vllm_server(*, server: RunningVllmServer) -> None:
    """Terminate one running vLLM server.

    Args:
        server: Running server metadata.

    Returns:
        None.
    """

    _terminate_process(process=server.process)


@contextmanager
def managed_vllm_server(
    *,
    model_spec: ModelSpec,
    serve_config: ServeConfig,
    port: int,
    log_dir: Path,
) -> Iterator[RunningVllmServer]:
    """Context manager that starts and stops one vLLM server.

    Args:
        model_spec: Model serving specification.
        serve_config: Serving configuration.
        port: Target TCP port.
        log_dir: Serve log directory.

    Returns:
        Iterator yielding one running server descriptor.
    """

    server = start_vllm_server(
        model_spec=model_spec,
        serve_config=serve_config,
        port=port,
        log_dir=log_dir,
    )
    try:
        yield server
    finally:
        stop_vllm_server(server=server)


def _lora_module_arg(*, model_spec: ModelSpec) -> str:
    assert model_spec.lora_name is not None, "lora_name required"
    assert model_spec.lora_adapter is not None, "lora_adapter required"
    return f"{model_spec.lora_name}={model_spec.lora_adapter}"


def _terminate_process(*, process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=15.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5.0)
