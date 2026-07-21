#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
cd "$PROJECT_DIR"

python -m mwe_roberta.cli evaluate \
  --data-dir "$PARSEME_DATA_DIR" \
  --languages all \
  --prediction-dir "$MWE_ARTIFACTS_DIR/predictions-dev" \
  --output-dir "$MWE_ARTIFACTS_DIR/evaluation-dev"
