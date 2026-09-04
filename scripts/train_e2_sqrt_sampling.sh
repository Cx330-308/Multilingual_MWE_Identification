#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
cd "$PROJECT_DIR"

SEED="${SEED:-13}"
E2_OUTPUT_DIR="${E2_OUTPUT_DIR:-$MWE_ARTIFACTS_DIR/experiments/e2-sqrt-sampling-seed${SEED}}"

python -m mwe_roberta.cli train \
  --data-dir "$PARSEME_DATA_DIR" \
  --languages all \
  --model-name xlm-roberta-base \
  --output-dir "$E2_OUTPUT_DIR" \
  --labels-json "$MWE_ARTIFACTS_DIR/metadata/labels.json" \
  --epochs 3 \
  --batch-size 8 \
  --eval-batch-size 16 \
  --max-length 256 \
  --stride 64 \
  --learning-rate 2e-5 \
  --weight-decay 0.01 \
  --warmup-ratio 0.1 \
  --language-sampling-alpha 0.5 \
  --evaluation-strategy epoch \
  --save-strategy epoch \
  --save-total-limit 2 \
  --selection-metric official \
  --early-stopping-patience 1 \
  --load-best-model-at-end \
  --seed "$SEED" \
  --experiment-name "e2-sqrt-sampling-seed${SEED}"
