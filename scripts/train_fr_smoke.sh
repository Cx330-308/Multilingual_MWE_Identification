#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
cd "$PROJECT_DIR"

python -m mwe_roberta.cli train \
  --data-dir "$PARSEME_DATA_DIR" \
  --languages FR \
  --model-name xlm-roberta-base \
  --output-dir "$MWE_ARTIFACTS_DIR/xlmr-fr-smoke" \
  --epochs 0.05 \
  --batch-size 2 \
  --eval-batch-size 2 \
  --max-length 128 \
  --stride 32 \
  --max-train-sentences-per-language 32 \
  --max-dev-sentences-per-language 16
