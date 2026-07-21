#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
cd "$PROJECT_DIR"

python -m mwe_roberta.cli prepare \
  --data-dir "$PARSEME_DATA_DIR" \
  --languages all \
  --output-dir "$MWE_ARTIFACTS_DIR/metadata"
