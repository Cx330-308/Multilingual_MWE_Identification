#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"
cd "$PROJECT_DIR"

MODEL_INPUT="${1:-mdeberta}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SEED="${SEED:-13}"
DRY_RUN="${DRY_RUN:-0}"
SMOKE_TEST="${SMOKE_TEST:-0}"
ALLOW_ENGLISH_ROBERTA="${ALLOW_ENGLISH_ROBERTA:-0}"

case "$MODEL_INPUT" in
  xlmr|xlm-roberta-base|FacebookAI/xlm-roberta-base)
    model_name="FacebookAI/xlm-roberta-base"
    model_tag="xlmr-base"
    english_only=0
    ;;
  mdeberta|mdeberta-v3-base|microsoft/mdeberta-v3-base)
    model_name="microsoft/mdeberta-v3-base"
    model_tag="mdeberta-v3-base"
    english_only=0
    ;;
  mbert|bert-base-multilingual-cased|google-bert/bert-base-multilingual-cased)
    model_name="google-bert/bert-base-multilingual-cased"
    model_tag="mbert-base-cased"
    english_only=0
    ;;
  roberta|roberta-base|FacebookAI/roberta-base)
    model_name="FacebookAI/roberta-base"
    model_tag="roberta-base-english"
    english_only=1
    ;;
  *)
    model_name="$MODEL_INPUT"
    model_tag="${2:-${MODEL_INPUT##*/}}"
    english_only=0
    ;;
esac

integer_pattern='^[0-9]+$'
tag_pattern='^[A-Za-z0-9._-]+$'

if [[ ! "$SEED" =~ $integer_pattern ]]; then
  echo "SEED must be a non-negative integer." >&2
  exit 2
fi
if [[ ! "$model_tag" =~ $tag_pattern ]]; then
  echo "The model tag must contain only letters, numbers, dots, underscores or hyphens." >&2
  echo "For a custom model, pass a safe tag as the second argument." >&2
  exit 2
fi
for flag_name in DRY_RUN SMOKE_TEST ALLOW_ENGLISH_ROBERTA; do
  flag_value="${!flag_name}"
  if [[ "$flag_value" != "0" && "$flag_value" != "1" ]]; then
    echo "$flag_name must be 0 or 1." >&2
    exit 2
  fi
done

if [[ "$english_only" == "1" ]]; then
  echo "Warning: FacebookAI/roberta-base was pretrained for English, while the" >&2
  echo "PARSEME languages in this project do not include English." >&2
  echo "Treat this run as a negative control, not a fair multilingual baseline." >&2
  if [[ "$SMOKE_TEST" == "0" && "$DRY_RUN" == "0" && "$ALLOW_ENGLISH_ROBERTA" != "1" ]]; then
    echo "Set ALLOW_ENGLISH_ROBERTA=1 to confirm a full English RoBERTa run." >&2
    exit 2
  fi
fi

if [[ "$SMOKE_TEST" == "1" ]]; then
  experiment_name="smoke-${model_tag}-fp32-seed${SEED}"
  output_dir="${OUTPUT_DIR:-$MWE_ARTIFACTS_DIR/experiments/$experiment_name}"
  train_command=(
    "$PYTHON_BIN" -m mwe_roberta.cli train
    --data-dir "$PARSEME_DATA_DIR"
    --languages FR
    --model-name "$model_name"
    --output-dir "$output_dir"
    --epochs 1
    --batch-size 2
    --eval-batch-size 2
    --max-length 256
    --stride 64
    --learning-rate 2e-5
    --weight-decay 0.01
    --warmup-ratio 0.1
    --language-sampling-alpha 0.5
    --class-weight-alpha 0.0
    --evaluation-strategy no
    --save-strategy no
    --logging-steps 1
    --max-train-sentences-per-language 128
    --max-dev-sentences-per-language 16
    --experiment-name "$experiment_name"
    --seed "$SEED"
  )
else
  experiment_name="compare-${model_tag}-fp32-a0p5-lr2em5-ep3-seed${SEED}"
  output_dir="${OUTPUT_DIR:-$MWE_ARTIFACTS_DIR/experiments/$experiment_name}"
  train_command=(
    "$PYTHON_BIN" -m mwe_roberta.cli train
    --data-dir "$PARSEME_DATA_DIR"
    --languages all
    --model-name "$model_name"
    --output-dir "$output_dir"
    --epochs 3
    --batch-size 8
    --eval-batch-size 16
    --max-length 256
    --stride 64
    --learning-rate 2e-5
    --weight-decay 0.01
    --warmup-ratio 0.1
    --language-sampling-alpha 0.5
    --class-weight-alpha 0.0
    --evaluation-strategy epoch
    --save-strategy epoch
    --save-total-limit 2
    --selection-metric official
    --early-stopping-patience 1
    --load-best-model-at-end
    --experiment-name "$experiment_name"
    --seed "$SEED"
  )
fi

printf 'Model: %s\n' "$model_name"
printf 'Experiment: %s\n' "$experiment_name"
printf 'Output: %s\n' "$output_dir"
printf 'Command:'
printf ' %q' "${train_command[@]}"
printf '\n'

if [[ "$DRY_RUN" == "1" ]]; then
  exit 0
fi

if [[ -e "$output_dir" ]]; then
  echo "Refusing to reuse existing output directory: $output_dir" >&2
  echo "Set OUTPUT_DIR to a new path if this is an intentional rerun." >&2
  exit 2
fi

"${train_command[@]}"
