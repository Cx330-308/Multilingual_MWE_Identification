#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
cd "$PROJECT_DIR"

MODEL_DIR="${1:-$MWE_ARTIFACTS_DIR/xlmr-all/best_model}"

python -m mwe_roberta.cli predict \
  --data-dir "$PARSEME_DATA_DIR" \
  --languages all \
  --split test.blind \
  --model-dir "$MODEL_DIR" \
  --output-dir "$MWE_ARTIFACTS_DIR/predictions"
