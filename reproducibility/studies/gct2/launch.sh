#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(dirname -- "$HERE")"
PY="${PYTHON:-/opt/conda/bin/python}"
STUDY="${1:-}"
case "$STUDY" in gct2|bco1) shift ;; *) echo 'Usage: bash growth_confirmation_cost_v1/launch.sh {gct2|bco1} [--resume]'; exit 2 ;; esac
[[ -x "$PY" ]] || { echo "STOP: Python not found: $PY"; exit 1; }
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_VISIBLE_DEVICES=0
"$PY" "$HERE/study.py" integrity --study "$STUDY" --project "$PROJECT"
"$PY" "$HERE/study.py" launch --study "$STUDY" --project "$PROJECT" "$@"
