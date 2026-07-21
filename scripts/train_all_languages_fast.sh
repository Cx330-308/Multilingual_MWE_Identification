#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
cd "$PROJECT_DIR"

python -m mwe_roberta.cli train \
  --data-dir "$PARSEME_DATA_DIR" \
  --languages all \
  --model-name xlm-roberta-base \
  --output-dir "$MWE_ARTIFACTS_DIR/xlmr-all-1epoch" \
  --labels-json "$MWE_ARTIFACTS_DIR/metadata/labels.json" \
  --epochs 1 \
  --batch-size 8 \
  --eval-batch-size 16 \
  --max-length 256 \
  --stride 64 \
  --learning-rate 2e-5 \
  --weight-decay 0.01 \
  --warmup-ratio 0.1 \
  --evaluation-strategy epoch \
  --save-strategy epoch \
  --load-best-model-at-end
