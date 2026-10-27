#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(dirname "$HERE")"
PY="${PYTHON:-/opt/conda/bin/python}"
if [ ! -x "$PY" ]; then
  echo "STOP: Python executable not found: $PY. Set PYTHON explicitly; do not alter the scientific recipe."
  exit 1
fi
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_VISIBLE_DEVICES=0
"$PY" "$HERE/study.py" integrity --project "$PROJECT"
"$PY" "$HERE/study.py" launch --project "$PROJECT" "$@"
