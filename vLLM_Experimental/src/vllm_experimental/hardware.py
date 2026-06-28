"""Hardware fingerprint helpers for benchmark comparability."""

from __future__ import annotations

import json
import os
import socket
import subprocess
from pathlib import Path

from vllm_experimental.types import HardwareFingerprint


def current_hardware_fingerprint() -> HardwareFingerprint:
    """Read the current node's first visible GPU fingerprint."""

    query = [
        "nvidia-smi",
        "--query-gpu=name,memory.total",
        "--format=csv,noheader,nounits",
    ]
    raw = subprocess.check_output(query, text=True).strip().splitlines()
    assert raw, "nvidia-smi returned no GPU rows"
    gpu_name, memory_total = [part.strip() for part in raw[0].split(",", maxsplit=1)]
    return HardwareFingerprint(
        node=socket.gethostname(),
        gpu_name=gpu_name,
        gpu_memory_total_mib=int(memory_total),
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    )


def write_hardware_manifest(*, path: Path) -> HardwareFingerprint:
    """Write and return the current hardware fingerprint."""

    fingerprint = current_hardware_fingerprint()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "node": fingerprint.node,
                "gpu_name": fingerprint.gpu_name,
                "gpu_memory_total_mib": fingerprint.gpu_memory_total_mib,
                "cuda_visible_devices": fingerprint.cuda_visible_devices,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return fingerprint


def assert_comparable_manifests(*, left_path: Path, right_path: Path) -> None:
    """Assert two hardware manifests represent comparable GPU rows."""

    left = json.loads(left_path.read_text(encoding="utf-8"))
    right = json.loads(right_path.read_text(encoding="utf-8"))
    left_fp = HardwareFingerprint(
        node=str(left["node"]),
        gpu_name=str(left["gpu_name"]),
        gpu_memory_total_mib=int(left["gpu_memory_total_mib"]),
        cuda_visible_devices=str(left.get("cuda_visible_devices", "")),
    )
    right_fp = HardwareFingerprint(
        node=str(right["node"]),
        gpu_name=str(right["gpu_name"]),
        gpu_memory_total_mib=int(right["gpu_memory_total_mib"]),
        cuda_visible_devices=str(right.get("cuda_visible_devices", "")),
    )
    assert left_fp.comparable_to(right_fp), (
        f"GPU mismatch: {left_fp.gpu_name}/{left_fp.gpu_memory_total_mib} MiB "
        f"vs {right_fp.gpu_name}/{right_fp.gpu_memory_total_mib} MiB"
    )
