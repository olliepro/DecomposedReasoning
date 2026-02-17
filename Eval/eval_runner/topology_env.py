"""Topology-aware environment policy for vLLM evaluation."""

from __future__ import annotations

import os
import re
import subprocess
from typing import Mapping, MutableMapping

ANSI_ESCAPE_PATTERN = re.compile(pattern=r"\x1B\[[0-?]*[ -/]*[@-~]")
GPU_NAME_PATTERN = re.compile(pattern=r"GPU\d+")
CROSS_NUMA_ENV_OVERRIDES: Mapping[str, str] = {
    "NCCL_P2P_DISABLE": "1",
    "VLLM_DISABLE_PYNCCL": "1",
    "VLLM_SKIP_P2P_CHECK": "0",
}


def _strip_ansi_sequences(text: str) -> str:
    """Remove ANSI terminal escape sequences.

    Args:
        text: Raw terminal output.

    Returns:
        Plain text without ANSI escape bytes.
    """
    return ANSI_ESCAPE_PATTERN.sub("", text)


def read_topology_output() -> str | None:
    """Read `nvidia-smi topo -m` output when available.

    Args:
        None.

    Returns:
        Topology output text, or `None` if command execution fails.
    """
    try:
        completed = subprocess.run(
            args=["nvidia-smi", "topo", "-m"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return _strip_ansi_sequences(text=completed.stdout)


def _extract_gpu_header_tokens(topology_text: str) -> list[str]:
    """Extract ordered GPU labels from the topology header row.

    Args:
        topology_text: Plain-text topology output.

    Returns:
        Ordered GPU labels from the header.
    """
    for raw_line in topology_text.splitlines():
        clean_line = raw_line.strip()
        if "CPU Affinity" not in clean_line:
            continue
        gpu_names = GPU_NAME_PATTERN.findall(clean_line)
        if len(gpu_names) >= 2:
            return gpu_names
    return []


def _iter_gpu_row_tokens(topology_text: str) -> list[list[str]]:
    """Collect tokenized GPU matrix rows from topology text.

    Args:
        topology_text: Plain-text topology output.

    Returns:
        Tokenized rows for lines starting with `GPU<idx>`.
    """
    gpu_rows: list[list[str]] = []
    for raw_line in topology_text.splitlines():
        clean_line = raw_line.strip()
        if not re.match(pattern=r"^GPU\d+\s", string=clean_line):
            continue
        gpu_rows.append(clean_line.split())
    return gpu_rows


def is_cross_numa_topology(topology_text: str) -> bool:
    """Check whether topology indicates cross-NUMA GPU connectivity.

    Args:
        topology_text: Output from `nvidia-smi topo -m`.

    Returns:
        `True` when at least one GPU pair reports `SYS` connectivity.

    Example:
        >>> is_cross_numa_topology(topology_text="GPU0 GPU1 CPU Affinity\\nGPU0 X SYS")
        True
    """
    cleaned_text = _strip_ansi_sequences(text=topology_text)
    gpu_names = _extract_gpu_header_tokens(topology_text=cleaned_text)
    if len(gpu_names) < 2:
        return False
    for row_tokens in _iter_gpu_row_tokens(topology_text=cleaned_text):
        row_gpu_name = row_tokens[0]
        if row_gpu_name not in gpu_names:
            continue
        link_tokens = row_tokens[1 : 1 + len(gpu_names)]
        if len(link_tokens) < len(gpu_names):
            continue
        for index, column_gpu_name in enumerate(gpu_names):
            if column_gpu_name == row_gpu_name:
                continue
            if link_tokens[index] == "SYS":
                return True
    return False


def maybe_set_cross_numa_vllm_env(
    model_type: str,
    env: MutableMapping[str, str] | None = None,
    topology_text: str | None = None,
) -> tuple[bool, str]:
    """Apply cross-NUMA vLLM env overrides when topology requires it.

    Args:
        model_type: Eval backend model type (`vllm` or `hf`).
        env: Optional environment mapping for mutation in tests.
        topology_text: Optional injected topology text for tests.

    Returns:
        Tuple `(applied, reason)` describing policy outcome.
    """
    if model_type != "vllm":
        return False, "model_type is not vllm"
    active_topology_text = read_topology_output() if topology_text is None else topology_text
    if active_topology_text is None:
        return False, "nvidia-smi topo -m unavailable"
    if not is_cross_numa_topology(topology_text=active_topology_text):
        return False, "topology is not cross-NUMA"
    active_env = os.environ if env is None else env
    for variable_name, variable_value in CROSS_NUMA_ENV_OVERRIDES.items():
        active_env[variable_name] = variable_value
    return True, "cross-NUMA topology detected"
