#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CANDIDATE="${1:-}"
if [[ -z "$CANDIDATE" ]]; then
  echo "Usage: $0 {L0|L1|L2|L3}" >&2
  exit 2
fi
shift

case "$CANDIDATE" in
  L0)
    candidate_tag=l0
    LEARNING_RATE=2e-4
    LANGUAGE_SAMPLING_ALPHA=0.5
    LORA_RANK=16
    LORA_ALPHA=32
    ;;
  L1)
    candidate_tag=l1
    LEARNING_RATE=1e-4
    LANGUAGE_SAMPLING_ALPHA=0.5
    LORA_RANK=16
    LORA_ALPHA=32
    ;;
  L2)
    candidate_tag=l2
    LEARNING_RATE=2e-4
    LANGUAGE_SAMPLING_ALPHA=0.5
    LORA_RANK=8
    LORA_ALPHA=16
    ;;
  L3)
    candidate_tag=l3
    LEARNING_RATE=2e-4
    LANGUAGE_SAMPLING_ALPHA=0.25
    LORA_RANK=16
    LORA_ALPHA=32
    ;;
  *)
    echo "Unknown candidate '$CANDIDATE'; choose L0, L1, L2, or L3." >&2
    exit 2
    ;;
esac

SEED="${SEED:-13}"
RUN_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/artifacts/experiments/lora-screen-${candidate_tag}-n100pl-seed${SEED}}"

printf 'LoRA screening candidate: %s\n' "$CANDIDATE"
printf 'Training output: %s\n' "$RUN_DIR"

MAX_TRAIN_SENTENCES_PER_LANGUAGE=100 \
SAVE_STEPS="${SAVE_STEPS:-25}" \
LEARNING_RATE="$LEARNING_RATE" \
LANGUAGE_SAMPLING_ALPHA="$LANGUAGE_SAMPLING_ALPHA" \
LORA_RANK="$LORA_RANK" \
LORA_ALPHA="$LORA_ALPHA" \
SEED="$SEED" \
OUTPUT_DIR="$RUN_DIR" \
DRY_RUN="${DRY_RUN:-0}" \
"$SCRIPT_DIR/train_qwen_lora.sh" "$@"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf 'Evaluation command: TRAIN_DIR=%q OUTPUT_DIR=%q %q\n' \
    "$RUN_DIR" "$RUN_DIR/dev-subset-seed42" "$SCRIPT_DIR/evaluate_qwen_lora.sh"
  exit 0
fi

TRAIN_DIR="$RUN_DIR" \
TRAIN_SEED="$SEED" \
EVAL_SEED=42 \
OUTPUT_DIR="$RUN_DIR/dev-subset-seed42" \
"$SCRIPT_DIR/evaluate_qwen_lora.sh"
