#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RLTRAINING_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"
VENV_DIR="${RLTRAINING_DIR}/.venv"

uv venv "${VENV_DIR}" --clear --python "${PYTHON_BIN}"
uv pip install --python "${VENV_DIR}/bin/python" -r "${RLTRAINING_DIR}/requirements.txt"
uv pip install --python "${VENV_DIR}/bin/python" --no-deps -e "${RLTRAINING_DIR}/verl"
