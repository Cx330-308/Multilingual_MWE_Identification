#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="${PARSEME_DATA_DIR:-$PROJECT_ROOT/../sharedtask-data/2.0/subtask1}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/artifacts/experiments/llm-qwen3p5-4b-4shot-seed13}"
REFERENCE_DIR="${REFERENCE_DIR:-$PROJECT_ROOT/artifacts/experiments/e2-sqrt-sampling-seed13/official-evaluations/eval-003/predictions}"

export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

exec python -m mwe_roberta.cli llm-baseline \
  --data-dir "$DATA_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --reference-prediction-dir "$REFERENCE_DIR" \
  --local-files-only \
  "$@"
