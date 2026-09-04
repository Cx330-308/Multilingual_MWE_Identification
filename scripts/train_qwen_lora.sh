#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="${PARSEME_DATA_DIR:-$PROJECT_ROOT/../sharedtask-data/2.0/subtask1}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.5-4B}"
SEED="${SEED:-13}"
TRAINING_SUBSET_SEED="${TRAINING_SUBSET_SEED:-$SEED}"
EPOCHS="${EPOCHS:-1}"
LEARNING_RATE="${LEARNING_RATE:-2e-4}"
MAX_LENGTH="${MAX_LENGTH:-640}"
MAX_TRAIN_SENTENCES_PER_LANGUAGE="${MAX_TRAIN_SENTENCES_PER_LANGUAGE:-500}"
LANGUAGE_SAMPLING_ALPHA="${LANGUAGE_SAMPLING_ALPHA:-0.5}"
LORA_RANK="${LORA_RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
SMOKE_TEST="${SMOKE_TEST:-0}"
DRY_RUN="${DRY_RUN:-0}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"

PYTHON_BIN="${PYTHON_BIN:-$HOME/.pyenv/versions/3.12.12/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "LLM Python not found or not executable: $PYTHON_BIN" >&2
  echo "Set PYTHON_BIN to the Python 3.12 interpreter containing the LoRA dependencies." >&2
  exit 2
fi
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

tag_value() {
  local tag="$1"
  tag="${tag//./p}"
  tag="${tag//-/m}"
  tag="${tag//+/p}"
  tag="${tag//E/e}"
  printf '%s' "$tag"
}

if [[ "$SMOKE_TEST" == "1" ]]; then
  experiment_name="smoke-lora-qwen3p5-4b-seed${SEED}"
else
  experiment_name="lora-qwen3p5-4b-n${MAX_TRAIN_SENTENCES_PER_LANGUAGE}pl-ml${MAX_LENGTH}-sa$(tag_value "$LANGUAGE_SAMPLING_ALPHA")-r${LORA_RANK}-a${LORA_ALPHA}-d$(tag_value "$LORA_DROPOUT")-lr$(tag_value "$LEARNING_RATE")-ep$(tag_value "$EPOCHS")-seed${SEED}"
fi
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/artifacts/experiments/$experiment_name}"

ARGS=(
  --data-dir "$DATA_DIR"
  --model-name "$MODEL_NAME"
  --output-dir "$OUTPUT_DIR"
  --seed "$SEED"
  --training-subset-seed "$TRAINING_SUBSET_SEED"
  --epochs "$EPOCHS"
  --batch-size "${BATCH_SIZE:-1}"
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS:-16}"
  --learning-rate "$LEARNING_RATE"
  --max-length "$MAX_LENGTH"
  --language-sampling-alpha "$LANGUAGE_SAMPLING_ALPHA"
  --lora-rank "$LORA_RANK"
  --lora-alpha "$LORA_ALPHA"
  --lora-dropout "$LORA_DROPOUT"
  --max-train-sentences-per-language "$MAX_TRAIN_SENTENCES_PER_LANGUAGE"
  --save-strategy steps
  --save-steps "${SAVE_STEPS:-100}"
  --save-total-limit 1
  --device "${DEVICE:-auto}"
  --local-files-only
)

if [[ "$SMOKE_TEST" == "1" ]]; then
  ARGS=(
    --data-dir "$DATA_DIR"
    --model-name "$MODEL_NAME"
    --output-dir "$OUTPUT_DIR"
    --seed "$SEED"
    --training-subset-seed "$TRAINING_SUBSET_SEED"
    --batch-size 1
    --gradient-accumulation-steps 1
    --max-length "$MAX_LENGTH"
    --max-train-sentences-per-language 1
    --max-steps 1
    --logging-steps 1
    --warmup-ratio 0
    --lr-scheduler-type constant
    --save-strategy no
    --device "${DEVICE:-auto}"
    --local-files-only
  )
fi

if [[ -n "$RESUME_FROM_CHECKPOINT" ]]; then
  ARGS+=(--resume-from-checkpoint "$RESUME_FROM_CHECKPOINT")
fi

printf 'Experiment: %s\n' "$experiment_name"
printf 'Output: %s\n' "$OUTPUT_DIR"
printf 'Python: %s\n' "$PYTHON_BIN"
printf 'Command:'
printf ' %q' "$PYTHON_BIN" -m mwe_roberta.cli llm-lora-train
printf ' %q' "${ARGS[@]}" "$@"
printf '\n'

if [[ "$DRY_RUN" == "1" ]]; then
  exit 0
fi
if [[ -e "$OUTPUT_DIR" && -z "$RESUME_FROM_CHECKPOINT" ]]; then
  echo "Refusing to reuse existing output directory: $OUTPUT_DIR" >&2
  echo "Set OUTPUT_DIR to a new path, or set RESUME_FROM_CHECKPOINT." >&2
  exit 2
fi

exec "$PYTHON_BIN" -m mwe_roberta.cli llm-lora-train "${ARGS[@]}" "$@"
