#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$HOME/.pyenv/versions/3.12.12/bin/python}"
RUN_SUBSET="${RUN_SUBSET:-1}"
RUN_FULL_DEV="${RUN_FULL_DEV:-1}"
DRY_RUN="${DRY_RUN:-0}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "LLM Python not found or not executable: $PYTHON_BIN" >&2
  exit 2
fi

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 {42|2026}" >&2
  echo "Run one confirmation seed at a time because each complete run is long." >&2
  exit 2
fi

TRAIN_SEED="$1"
TRAINING_SUBSET_SEED="${TRAINING_SUBSET_SEED:-13}"
case "$TRAIN_SEED" in
  42|2026) ;;
  *)
    echo "Confirmation seed must be 42 or 2026; received '$TRAIN_SEED'." >&2
    exit 2
    ;;
esac

EXPERIMENT_NAME="lora-qwen3p5-4b-n500pl-ml640-sa0p5-r8-a16-d0p05-lr2em4-ep1-seed${TRAIN_SEED}"
TRAIN_DIR="$PROJECT_ROOT/artifacts/experiments/$EXPERIMENT_NAME"
SUBSET_DIR="$TRAIN_DIR/dev-subset-seed42"
FULL_DIR="$TRAIN_DIR/dev-full"

training_complete() {
  [[ -s "$TRAIN_DIR/adapter/adapter_model.safetensors" ]] || return 1
  [[ -s "$TRAIN_DIR/training_metadata.json" ]] || return 1
  [[ -s "$TRAIN_DIR/experiment_config.json" ]] || return 1
  "$PYTHON_BIN" -c \
    'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8")).get("status", ""))' \
    "$TRAIN_DIR/experiment_config.json" | grep -qx completed
}

latest_checkpoint() {
  "$PYTHON_BIN" -c \
    'from pathlib import Path; import sys
p = Path(sys.argv[1])
items = []
for path in p.glob("checkpoint-*"):
    try:
        items.append((int(path.name.rsplit("-", 1)[1]), path))
    except ValueError:
        pass
print(max(items)[1] if items else "")' \
    "$TRAIN_DIR"
}

run_training() {
  if training_complete; then
    echo "Training already complete: $TRAIN_DIR"
    return
  fi

  if [[ ! -e "$TRAIN_DIR" ]]; then
    echo "Starting LoRA training for seed $TRAIN_SEED."
    SEED="$TRAIN_SEED" TRAINING_SUBSET_SEED="$TRAINING_SUBSET_SEED" \
      "$SCRIPT_DIR/train_qwen_lora.sh"
    return
  fi

  local checkpoint
  checkpoint="$(latest_checkpoint)"
  if [[ -z "$checkpoint" ]]; then
    echo "Incomplete training directory has no resumable checkpoint: $TRAIN_DIR" >&2
    echo "Inspect it manually rather than overwriting recorded artefacts." >&2
    exit 2
  fi
  echo "Resuming LoRA training for seed $TRAIN_SEED from $checkpoint."
  SEED="$TRAIN_SEED" TRAINING_SUBSET_SEED="$TRAINING_SUBSET_SEED" \
    RESUME_FROM_CHECKPOINT="$checkpoint" \
    "$SCRIPT_DIR/train_qwen_lora.sh"
}

run_evaluation() {
  local mode="$1"
  local output_dir="$2"
  if [[ -s "$output_dir/metrics.json" ]]; then
    echo "Evaluation already complete: $output_dir"
    return
  fi

  if [[ "$mode" == "full" ]]; then
    echo "Starting complete-development evaluation for training seed $TRAIN_SEED."
    TRAIN_SEED="$TRAIN_SEED" EVAL_SEED=42 FULL_DEV=1 \
      "$SCRIPT_DIR/evaluate_qwen_lora.sh"
  else
    echo "Starting matched-subset evaluation for training seed $TRAIN_SEED."
    TRAIN_SEED="$TRAIN_SEED" EVAL_SEED=42 \
      "$SCRIPT_DIR/evaluate_qwen_lora.sh"
  fi
}

print_result() {
  "$PYTHON_BIN" -c \
    'import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))["metrics"]["llm"]["average"]
m, t = d["mwe"], d["token"]
print("Full dev seed {seed}: MWE P={p:.4f} R={r:.4f} F1={f:.4f}; "
      "token F1={tf:.4f}".format(
          seed=sys.argv[2], p=m["precision"], r=m["recall"],
          f=m["f1"], tf=t["f1"]))' \
    "$FULL_DIR/metrics.json" "$TRAIN_SEED"
}

echo "Confirmation experiment: $EXPERIMENT_NAME"
echo "Training seed: $TRAIN_SEED; training-subset seed: $TRAINING_SUBSET_SEED; evaluation sample seed: 42"
echo "Output: $TRAIN_DIR"

if [[ "$DRY_RUN" == "1" ]]; then
  SEED="$TRAIN_SEED" TRAINING_SUBSET_SEED="$TRAINING_SUBSET_SEED" DRY_RUN=1 \
    "$SCRIPT_DIR/train_qwen_lora.sh"
  printf 'Matched subset: TRAIN_SEED=%q EVAL_SEED=42 %q\n' \
    "$TRAIN_SEED" "$SCRIPT_DIR/evaluate_qwen_lora.sh"
  printf 'Full dev: TRAIN_SEED=%q EVAL_SEED=42 FULL_DEV=1 %q\n' \
    "$TRAIN_SEED" "$SCRIPT_DIR/evaluate_qwen_lora.sh"
  exit 0
fi

run_training
if [[ "$RUN_SUBSET" == "1" ]]; then
  run_evaluation subset "$SUBSET_DIR"
fi
if [[ "$RUN_FULL_DEV" == "1" ]]; then
  run_evaluation full "$FULL_DIR"
  print_result
fi

echo "Confirmation seed $TRAIN_SEED is complete."
