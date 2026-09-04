#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export MODEL_NAME="google-bert/bert-base-multilingual-cased"
export MODEL_TAG="mbert-base-cased"

exec "$SCRIPT_DIR/train_mdeberta_tuning.sh" "$@"
