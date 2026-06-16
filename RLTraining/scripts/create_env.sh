#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RLTRAINING_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"
VENV_DIR="${VENV_DIR:-${RLTRAINING_DIR}/.venv}"
TORCH_BACKEND="${TORCH_BACKEND:-cu129}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"

uv venv "${VENV_DIR}" --clear --python "${PYTHON_BIN}"
uv pip install --python "${VENV_DIR}/bin/python" pip
uv pip install \
  --python "${VENV_DIR}/bin/python" \
  --torch-backend "${TORCH_BACKEND}" \
  "torch==2.11.0+cu129" \
  "transformers==5.9.0" \
  "triton==3.6.0" \
  "flash-linear-attention==0.5.0" \
  "flashinfer-python==0.6.11.post2" \
  "vllm==0.22.0"
if [[ "${TORCH_BACKEND}" == "cu129" ]]; then
  uv pip install \
    --python "${VENV_DIR}/bin/python" \
    --index-url "https://flashinfer.ai/whl/cu129" \
    --extra-index-url "https://pypi.org/simple" \
    --index-strategy unsafe-best-match \
    "flashinfer-jit-cache==0.6.11.post2+cu129"
fi
uv pip install \
  --python "${VENV_DIR}/bin/python" \
  -r "${RLTRAINING_DIR}/requirements.txt"
if [[ "${INSTALL_CAUSAL_CONV1D_FASTPATH:-1}" == "1" ]]; then
  PYTHON_BIN="${VENV_DIR}/bin/python" \
    SCRATCH_ROOT="${SCRATCH_ROOT:-/fs/scratch/PAA0201/ollieproudman/DecomposedReasoning}" \
    bash "${SCRIPT_DIR}/install_causal_conv1d_fastpath.sh"
fi
uv pip install --python "${VENV_DIR}/bin/python" -e "${RLTRAINING_DIR}/packages/flash_attn_shim"
uv pip install --python "${VENV_DIR}/bin/python" --no-deps -e "${RLTRAINING_DIR}/verl"
