"""Unit tests for topology-based vLLM env policy."""

from __future__ import annotations

import eval_runner.topology_env as topology_env

CROSS_NUMA_TOPOLOGY = """
GPU0 GPU1 CPU Affinity NUMA Affinity
GPU0 X SYS 0-31 0
GPU1 SYS X 32-63 1
"""

SINGLE_NUMA_TOPOLOGY = """
GPU0 GPU1 CPU Affinity NUMA Affinity
GPU0 X PIX 0-63 0
GPU1 PIX X 0-63 0
"""

NODE_LINK_CROSS_NUMA_TOPOLOGY = """
GPU0 GPU1 NIC0 CPU Affinity NUMA Affinity GPU NUMA ID
GPU0 X NODE NODE 32-42 2 N/A
GPU1 NODE X NODE 7 N/A
"""


def test_is_cross_numa_topology_detects_sys_links() -> None:
    """SYS links between GPU rows should be treated as cross-NUMA.

    Args:
        None.

    Returns:
        None.
    """
    assert topology_env.is_cross_numa_topology(topology_text=CROSS_NUMA_TOPOLOGY)


def test_is_cross_numa_topology_rejects_non_sys_links() -> None:
    """Non-SYS GPU links should not trigger cross-NUMA policy.

    Args:
        None.

    Returns:
        None.
    """
    assert not topology_env.is_cross_numa_topology(topology_text=SINGLE_NUMA_TOPOLOGY)


def test_is_cross_numa_topology_detects_split_numa_affinity_without_sys() -> None:
    """Split NUMA affinity values should trigger cross-NUMA even without SYS.

    Args:
        None.

    Returns:
        None.
    """
    assert topology_env.is_cross_numa_topology(topology_text=NODE_LINK_CROSS_NUMA_TOPOLOGY)


def test_maybe_set_cross_numa_vllm_env_applies_expected_values() -> None:
    """Policy should set all required env vars for vLLM on cross-NUMA.

    Args:
        None.

    Returns:
        None.
    """
    fake_env: dict[str, str] = {}
    applied, reason = topology_env.maybe_set_cross_numa_vllm_env(
        model_type="vllm",
        env=fake_env,
        topology_text=CROSS_NUMA_TOPOLOGY,
        gpu_product_names=["NVIDIA A100-SXM4-80GB", "NVIDIA A100-SXM4-80GB"],
    )

    assert applied is True
    assert reason == "cross-NUMA topology detected"
    assert fake_env["NCCL_P2P_DISABLE"] == "1"
    assert fake_env["VLLM_DISABLE_PYNCCL"] == "1"
    assert fake_env["VLLM_SKIP_P2P_CHECK"] == "0"


def test_maybe_set_cross_numa_vllm_env_skips_for_non_cross_numa() -> None:
    """Policy should skip env mutation when topology is not cross-NUMA.

    Args:
        None.

    Returns:
        None.
    """
    fake_env: dict[str, str] = {}
    applied, reason = topology_env.maybe_set_cross_numa_vllm_env(
        model_type="vllm",
        env=fake_env,
        topology_text=SINGLE_NUMA_TOPOLOGY,
        gpu_product_names=["NVIDIA A100-SXM4-80GB", "NVIDIA A100-SXM4-80GB"],
    )

    assert applied is False
    assert reason == "topology is not cross-NUMA"
    assert fake_env == {}


def test_maybe_set_cross_numa_vllm_env_handles_missing_topology(monkeypatch) -> None:
    """Policy should not fail when topology probing is unavailable.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """

    def _missing_topology() -> str | None:
        return None

    monkeypatch.setattr(topology_env, "read_topology_output", _missing_topology)
    fake_env: dict[str, str] = {}
    applied, reason = topology_env.maybe_set_cross_numa_vllm_env(
        model_type="vllm",
        env=fake_env,
        topology_text=None,
        gpu_product_names=["NVIDIA A100-SXM4-80GB", "NVIDIA A100-SXM4-80GB"],
    )

    assert applied is False
    assert reason == "nvidia-smi topo -m unavailable"
    assert fake_env == {}


def test_maybe_set_cross_numa_vllm_env_skips_for_hf_model() -> None:
    """Policy should be inactive for non-vLLM model backends.

    Args:
        None.

    Returns:
        None.
    """
    fake_env: dict[str, str] = {}
    applied, reason = topology_env.maybe_set_cross_numa_vllm_env(
        model_type="hf",
        env=fake_env,
        topology_text=CROSS_NUMA_TOPOLOGY,
        gpu_product_names=["NVIDIA A100-SXM4-80GB", "NVIDIA A100-SXM4-80GB"],
    )

    assert applied is False
    assert reason == "model_type is not vllm"
    assert fake_env == {}


def test_maybe_set_cross_numa_vllm_env_requires_exactly_two_a100_gpus() -> None:
    """Policy should be inactive when GPU shape is not exactly 2xA100."""
    fake_env: dict[str, str] = {}
    applied, reason = topology_env.maybe_set_cross_numa_vllm_env(
        model_type="vllm",
        env=fake_env,
        topology_text=CROSS_NUMA_TOPOLOGY,
        gpu_product_names=["NVIDIA A100-SXM4-80GB"],
    )

    assert applied is False
    assert reason == "requires exactly 2 A100 GPUs"
    assert fake_env == {}


def test_has_required_cross_numa_gpu_shape_accepts_two_a100_gpus() -> None:
    """Helper should accept only the required 2xA100 host shape."""
    assert topology_env.has_required_cross_numa_gpu_shape(
        gpu_product_names=["NVIDIA A100-SXM4-40GB", "NVIDIA A100-SXM4-80GB"]
    )
    assert not topology_env.has_required_cross_numa_gpu_shape(
        gpu_product_names=["NVIDIA A100-SXM4-80GB", "NVIDIA H100-SXM5-80GB"]
    )
    assert not topology_env.has_required_cross_numa_gpu_shape(
        gpu_product_names=["NVIDIA A100-SXM4-80GB"]
    )
