#!/bin/bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/fs/scratch/PAA0201/ollieproudman/DecomposedReasoning}"
PYTHON_BIN="${PYTHON_BIN:-${SCRATCH_ROOT}/RLTraining/envs/qwen35-rl-cu129/bin/python}"
VLLM_EXPERIMENTAL_DIR="${REPO_ROOT}/vLLM_Experimental"

cd "${REPO_ROOT}"
PYTHONPATH="${VLLM_EXPERIMENTAL_DIR}/src:${PYTHONPATH:-}" \
  "${PYTHON_BIN}" -m vllm_experimental.materialize_runtime \
    --runtime-parent "${SCRATCH_ROOT}/vLLM_Experimental/runtimes" \
    --patches-dir "${VLLM_EXPERIMENTAL_DIR}/patches" \
    "$@"
