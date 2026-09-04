#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="${PARSEME_DATA_DIR:-$PROJECT_ROOT/../sharedtask-data/2.0/subtask1}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.5-4B}"
TRAIN_SEED="${TRAIN_SEED:-${SEED:-13}}"
EVAL_SEED="${EVAL_SEED:-42}"
SAMPLE_PURPOSE="${SAMPLE_PURPOSE:-qwen-train-calibration-v1-frozen-dev}"
TRAIN_DIR="${TRAIN_DIR:-$PROJECT_ROOT/artifacts/experiments/lora-qwen3p5-4b-n500pl-ml640-sa0p5-r8-a16-d0p05-lr2em4-ep1-seed${TRAIN_SEED}}"
ADAPTER_DIR="${ADAPTER_DIR:-$TRAIN_DIR/adapter}"
if [[ "${FULL_DEV:-0}" == "1" ]]; then
  DEFAULT_OUTPUT_DIR="$TRAIN_DIR/dev-full"
else
  DEFAULT_OUTPUT_DIR="$TRAIN_DIR/dev-subset-seed${EVAL_SEED}"
fi
OUTPUT_DIR="${OUTPUT_DIR:-$DEFAULT_OUTPUT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-$HOME/.pyenv/versions/3.12.12/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "LLM Python not found or not executable: $PYTHON_BIN" >&2
  echo "Set PYTHON_BIN to the Python 3.12 interpreter containing the LoRA dependencies." >&2
  exit 2
fi
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

ARGS=(
  --data-dir "$DATA_DIR"
  --model-name "$MODEL_NAME"
  --adapter-dir "$ADAPTER_DIR"
  --output-dir "$OUTPUT_DIR"
  --seed "$EVAL_SEED"
  --sample-purpose "$SAMPLE_PURPOSE"
  --shots 0
  --batch-size "${BATCH_SIZE:-4}"
  --max-input-tokens "${MAX_INPUT_TOKENS:-2048}"
  --max-new-tokens "${MAX_NEW_TOKENS:-96}"
  --device "${DEVICE:-auto}"
  --local-files-only
)

if [[ "${FULL_DEV:-0}" == "1" ]]; then
  ARGS+=(--all-targets)
fi

exec "$PYTHON_BIN" -m mwe_roberta.cli llm-baseline "${ARGS[@]}" "$@"
