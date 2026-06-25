#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RLTRAINING_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${RL_SPEC_PYTHON_BIN:-${RL_PYTHON_BIN:-}}"
if [[ -z "${PYTHON_BIN}" ]]; then
  if [[ -x "${RLTRAINING_DIR}/.venv/bin/python" ]]; then
    PYTHON_BIN="${RLTRAINING_DIR}/.venv/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi

export PYTHONPATH="${RLTRAINING_DIR}:${RLTRAINING_DIR}/verl:${RLTRAINING_DIR}/../Analysis:${PYTHONPATH:-}"

exec "${PYTHON_BIN}" -m branching_dapo.run_specs submit-qwen35-matrix "$@"
