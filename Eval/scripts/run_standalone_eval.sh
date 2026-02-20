#!/bin/bash

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: scripts/run_standalone_eval.sh --checkpoint <path-or-repo> --config <config-yaml> [--output <json>] [extra-standalone-args...]"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
EVAL_PYTHON_BIN="${EVAL_PYTHON_BIN:-${PROJECT_DIR}/.venv/bin/python}"

if [[ ! -x "${EVAL_PYTHON_BIN}" ]]; then
  echo "Eval python binary not found: ${EVAL_PYTHON_BIN}"
  exit 1
fi

if [[ " $* " != *" --checkpoint "* ]]; then
  echo "Missing required argument: --checkpoint"
  exit 1
fi
if [[ " $* " != *" --config "* ]]; then
  echo "Missing required argument: --config"
  exit 1
fi

"${EVAL_PYTHON_BIN}" -m eval_runner.standalone "$@"
