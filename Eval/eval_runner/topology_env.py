"""Topology-aware environment policy for vLLM evaluation."""

from __future__ import annotations

import os
import re
import subprocess
from typing import Mapping, MutableMapping

ANSI_ESCAPE_PATTERN = re.compile(pattern=r"\x1B\[[0-?]*[ -/]*[@-~]")
GPU_NAME_PATTERN = re.compile(pattern=r"GPU\d+")
CPU_AFFINITY_PATTERN = re.compile(pattern=r"^\d+(?:-\d+)?(?:,\d+(?:-\d+)?)*$")
INTEGER_PATTERN = re.compile(pattern=r"^\d+$")
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


def read_gpu_product_names() -> list[str] | None:
    """Read installed GPU product names via `nvidia-smi`.

    Args:
        None.

    Returns:
        Ordered GPU name list, or `None` if query command fails.
    """
    try:
        completed = subprocess.run(
            args=["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    gpu_names = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    return gpu_names


def has_required_cross_numa_gpu_shape(gpu_product_names: list[str]) -> bool:
    """Check whether host matches the required cross-NUMA GPU shape.

    Args:
        gpu_product_names: GPU product names from `nvidia-smi`.

    Returns:
        `True` only when there are exactly 2 GPUs and both are A100.
    """
    if len(gpu_product_names) != 2:
        return False
    return all("A100" in gpu_name.upper() for gpu_name in gpu_product_names)


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
        if "CPU Affinity" in clean_line:
            continue
        gpu_rows.append(clean_line.split())
    return gpu_rows


def _parse_row_numa_affinity(*, row_tokens: list[str], gpu_count: int) -> int | None:
    """Extract one GPU row NUMA affinity value when present.

    Args:
        row_tokens: Tokenized row for one `GPU<idx>` matrix line.
        gpu_count: Number of GPU columns in the matrix header.

    Returns:
        Parsed NUMA affinity value, or `None` when unavailable.
    """
    row_tail_tokens = row_tokens[1 + gpu_count :]
    for token_index, token in enumerate(row_tail_tokens):
        if not CPU_AFFINITY_PATTERN.fullmatch(token):
            continue
        next_index = token_index + 1
        if next_index >= len(row_tail_tokens):
            continue
        next_token = row_tail_tokens[next_index]
        if INTEGER_PATTERN.fullmatch(next_token):
            return int(next_token)
        continue
    integer_tokens = [int(token) for token in row_tail_tokens if INTEGER_PATTERN.fullmatch(token)]
    if not integer_tokens:
        return None
    return integer_tokens[-1]


def _extract_gpu_numa_affinities(*, topology_text: str, gpu_names: list[str]) -> dict[str, int]:
    """Build per-GPU NUMA affinity mapping from matrix rows.

    Args:
        topology_text: Plain-text topology output.
        gpu_names: Ordered GPU labels from matrix header.

    Returns:
        Mapping from GPU label to NUMA affinity value.
    """
    gpu_count = len(gpu_names)
    parsed_affinities: dict[str, int] = {}
    for row_tokens in _iter_gpu_row_tokens(topology_text=topology_text):
        row_gpu_name = row_tokens[0]
        if row_gpu_name not in gpu_names:
            continue
        row_numa_affinity = _parse_row_numa_affinity(row_tokens=row_tokens, gpu_count=gpu_count)
        if row_numa_affinity is None:
            continue
        parsed_affinities[row_gpu_name] = row_numa_affinity
    return parsed_affinities


def is_cross_numa_topology(topology_text: str) -> bool:
    """Check whether topology indicates cross-NUMA GPU connectivity.

    Args:
        topology_text: Output from `nvidia-smi topo -m`.

    Returns:
        `True` when at least one GPU pair reports `SYS` connectivity, or when
        detected GPU NUMA affinity values are split across multiple NUMA nodes.

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
    gpu_numa_affinities = _extract_gpu_numa_affinities(topology_text=cleaned_text, gpu_names=gpu_names)
    return len(set(gpu_numa_affinities.values())) >= 2


def maybe_set_cross_numa_vllm_env(
    model_type: str,
    env: MutableMapping[str, str] | None = None,
    topology_text: str | None = None,
    gpu_product_names: list[str] | None = None,
) -> tuple[bool, str]:
    """Apply cross-NUMA vLLM env overrides when topology requires it.

    Args:
        model_type: Eval backend model type (`vllm` or `hf`).
        env: Optional environment mapping for mutation in tests.
        topology_text: Optional injected topology text for tests.
        gpu_product_names: Optional injected GPU product names for tests.

    Returns:
        Tuple `(applied, reason)` describing policy outcome.
    """
    if model_type != "vllm":
        return False, "model_type is not vllm"
    active_gpu_product_names = (
        read_gpu_product_names() if gpu_product_names is None else gpu_product_names
    )
    if active_gpu_product_names is None:
        return False, "nvidia-smi --query-gpu=name unavailable"
    if not has_required_cross_numa_gpu_shape(
        gpu_product_names=active_gpu_product_names
    ):
        return False, "requires exactly 2 A100 GPUs"
    active_topology_text = read_topology_output() if topology_text is None else topology_text
    if active_topology_text is None:
        return False, "nvidia-smi topo -m unavailable"
    if not is_cross_numa_topology(topology_text=active_topology_text):
        return False, "topology is not cross-NUMA"
    active_env = os.environ if env is None else env
    for variable_name, variable_value in CROSS_NUMA_ENV_OVERRIDES.items():
        active_env[variable_name] = variable_value
    return True, "cross-NUMA topology detected"
