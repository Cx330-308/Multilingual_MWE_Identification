#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"
cd "$PROJECT_DIR"

SAMPLING_ALPHA="${SAMPLING_ALPHA:-0.5}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
EPOCHS="${EPOCHS:-3}"
SEED="${SEED:-13}"
DRY_RUN="${DRY_RUN:-0}"

alpha_pattern='^(0([.][0-9]+)?|1([.]0+)?)$'
number_pattern='^[0-9]+([.][0-9]+)?([eE][+-]?[0-9]+)?$'
zero_pattern='^0+([.]0+)?([eE][+-]?[0-9]+)?$'
integer_pattern='^[0-9]+$'

if [[ ! "$SAMPLING_ALPHA" =~ $alpha_pattern ]]; then
  echo "SAMPLING_ALPHA must be a number between 0.0 and 1.0." >&2
  exit 2
fi
if [[ ! "$LEARNING_RATE" =~ $number_pattern || "$LEARNING_RATE" =~ $zero_pattern ]]; then
  echo "LEARNING_RATE must be a positive number, for example 1e-5." >&2
  exit 2
fi
if [[ ! "$EPOCHS" =~ $number_pattern || "$EPOCHS" =~ $zero_pattern ]]; then
  echo "EPOCHS must be a positive number." >&2
  exit 2
fi
if [[ ! "$SEED" =~ $integer_pattern ]]; then
  echo "SEED must be a non-negative integer." >&2
  exit 2
fi
if [[ "$DRY_RUN" != "0" && "$DRY_RUN" != "1" ]]; then
  echo "DRY_RUN must be 0 or 1." >&2
  exit 2
fi

tag_value() {
  local tag="$1"
  tag="${tag//./p}"
  tag="${tag//-/m}"
  tag="${tag//+/p}"
  tag="${tag//E/e}"
  printf '%s' "$tag"
}

alpha_tag="$(tag_value "$SAMPLING_ALPHA")"
learning_rate_tag="$(tag_value "$LEARNING_RATE")"
epochs_tag="$(tag_value "$EPOCHS")"
experiment_name="tune-xlmr-a${alpha_tag}-lr${learning_rate_tag}-ep${epochs_tag}-seed${SEED}"
output_dir="${OUTPUT_DIR:-$MWE_ARTIFACTS_DIR/experiments/$experiment_name}"

train_command=(
  python -m mwe_roberta.cli train
  --data-dir "$PARSEME_DATA_DIR"
  --languages all
  --model-name xlm-roberta-base
  --output-dir "$output_dir"
  --epochs "$EPOCHS"
  --batch-size 8
  --eval-batch-size 16
  --max-length 256
  --stride 64
  --learning-rate "$LEARNING_RATE"
  --weight-decay 0.01
  --warmup-ratio 0.1
  --language-sampling-alpha "$SAMPLING_ALPHA"
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
