#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for seed in 42 2026; do
  "$SCRIPT_DIR/run_qwen_lora_confirmation.sh" "$seed"
done

echo "All Qwen LoRA confirmation seeds are complete."
