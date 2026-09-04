#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"
cd "$PROJECT_DIR"

SAMPLING_ALPHA="${SAMPLING_ALPHA:-0.5}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_NAME="${MODEL_NAME:-microsoft/mdeberta-v3-base}"
MODEL_TAG="${MODEL_TAG:-mdeberta-v3-base}"
CLASS_WEIGHT_ALPHA="${CLASS_WEIGHT_ALPHA:-0.0}"
CLASS_WEIGHT_MIN="${CLASS_WEIGHT_MIN:-0.75}"
CLASS_WEIGHT_MAX="${CLASS_WEIGHT_MAX:-2.5}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
EPOCHS="${EPOCHS:-3}"
SEED="${SEED:-13}"
DRY_RUN="${DRY_RUN:-0}"
SMOKE_TEST="${SMOKE_TEST:-0}"

unit_interval_pattern='^(0([.][0-9]+)?|1([.]0+)?)$'
number_pattern='^[0-9]+([.][0-9]+)?([eE][+-]?[0-9]+)?$'
zero_pattern='^0+([.]0+)?([eE][+-]?[0-9]+)?$'
integer_pattern='^[0-9]+$'
tag_pattern='^[A-Za-z0-9._-]+$'

if [[ -z "$MODEL_NAME" ]]; then
  echo "MODEL_NAME must not be empty." >&2
  exit 2
fi
if [[ ! "$MODEL_TAG" =~ $tag_pattern ]]; then
  echo "MODEL_TAG must contain only letters, numbers, dots, underscores or hyphens." >&2
  exit 2
fi

if [[ ! "$SAMPLING_ALPHA" =~ $unit_interval_pattern ]]; then
  echo "SAMPLING_ALPHA must be a number between 0.0 and 1.0." >&2
  exit 2
fi
if [[ ! "$CLASS_WEIGHT_ALPHA" =~ $unit_interval_pattern ]]; then
  echo "CLASS_WEIGHT_ALPHA must be a number between 0.0 and 1.0." >&2
  exit 2
fi
for variable_name in CLASS_WEIGHT_MIN CLASS_WEIGHT_MAX LEARNING_RATE EPOCHS; do
  variable_value="${!variable_name}"
  if [[ ! "$variable_value" =~ $number_pattern || "$variable_value" =~ $zero_pattern ]]; then
    echo "$variable_name must be a positive number." >&2
    exit 2
  fi
done
if ! awk -v lower="$CLASS_WEIGHT_MIN" -v upper="$CLASS_WEIGHT_MAX" \
  'BEGIN { exit(lower <= upper ? 0 : 1) }'; then
  echo "CLASS_WEIGHT_MAX must be greater than or equal to CLASS_WEIGHT_MIN." >&2
  exit 2
fi
if [[ ! "$SEED" =~ $integer_pattern ]]; then
  echo "SEED must be a non-negative integer." >&2
  exit 2
fi
for flag_name in DRY_RUN SMOKE_TEST; do
  flag_value="${!flag_name}"
  if [[ "$flag_value" != "0" && "$flag_value" != "1" ]]; then
    echo "$flag_name must be 0 or 1." >&2
    exit 2
  fi
done

tag_value() {
  local tag="$1"
  tag="${tag//./p}"
  tag="${tag//-/m}"
  tag="${tag//+/p}"
  tag="${tag//E/e}"
  printf '%s' "$tag"
}

sampling_tag="$(tag_value "$SAMPLING_ALPHA")"
class_alpha_tag="$(tag_value "$CLASS_WEIGHT_ALPHA")"
class_min_tag="$(tag_value "$CLASS_WEIGHT_MIN")"
class_max_tag="$(tag_value "$CLASS_WEIGHT_MAX")"
learning_rate_tag="$(tag_value "$LEARNING_RATE")"
recorded_epochs="$EPOCHS"
if [[ "$SMOKE_TEST" == "1" ]]; then
  recorded_epochs="1"
fi
epochs_tag="$(tag_value "$recorded_epochs")"
configuration_tag="sa${sampling_tag}-cwa${class_alpha_tag}-clip${class_min_tag}to${class_max_tag}-lr${learning_rate_tag}-ep${epochs_tag}-seed${SEED}"

if [[ "$SMOKE_TEST" == "1" ]]; then
  experiment_name="smoke-${MODEL_TAG}-fp32-${configuration_tag}"
else
  experiment_name="tune-${MODEL_TAG}-fp32-${configuration_tag}"
fi
output_dir="${OUTPUT_DIR:-$MWE_ARTIFACTS_DIR/experiments/$experiment_name}"

train_command=(
  "$PYTHON_BIN" -m mwe_roberta.cli train
  --data-dir "$PARSEME_DATA_DIR"
  --languages all
  --model-name "$MODEL_NAME"
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
  --class-weight-alpha "$CLASS_WEIGHT_ALPHA"
  --class-weight-min "$CLASS_WEIGHT_MIN"
  --class-weight-max "$CLASS_WEIGHT_MAX"
  --evaluation-strategy epoch
  --save-strategy epoch
  --save-total-limit 2
  --selection-metric official
  --early-stopping-patience 1
  --load-best-model-at-end
  --experiment-name "$experiment_name"
  --seed "$SEED"
)

if [[ "$SMOKE_TEST" == "1" ]]; then
  train_command=(
    "$PYTHON_BIN" -m mwe_roberta.cli train
    --data-dir "$PARSEME_DATA_DIR"
    --languages FR
    --model-name "$MODEL_NAME"
    --output-dir "$output_dir"
    --epochs 1
    --batch-size 2
    --eval-batch-size 2
    --max-length 256
    --stride 64
    --learning-rate "$LEARNING_RATE"
    --weight-decay 0.01
    --warmup-ratio 0.1
    --language-sampling-alpha "$SAMPLING_ALPHA"
    --class-weight-alpha "$CLASS_WEIGHT_ALPHA"
    --class-weight-min "$CLASS_WEIGHT_MIN"
    --class-weight-max "$CLASS_WEIGHT_MAX"
    --evaluation-strategy no
    --save-strategy no
    --logging-steps 1
    --max-train-sentences-per-language 128
    --max-dev-sentences-per-language 16
    --experiment-name "$experiment_name"
    --seed "$SEED"
  )
fi

printf 'Model: %s\n' "$MODEL_NAME"
printf 'Experiment: %s\n' "$experiment_name"
printf 'Output: %s\n' "$output_dir"
printf 'Command:'
printf ' %q' "${train_command[@]}"
printf '\n'

if [[ "$CLASS_WEIGHT_ALPHA" == "0" || "$CLASS_WEIGHT_ALPHA" == "0.0" ]]; then
  printf 'Loss: standard cross-entropy; clipping bounds are inactive.\n'
fi

if [[ "$DRY_RUN" == "1" ]]; then
  exit 0
fi

if [[ -e "$output_dir" ]]; then
  echo "Refusing to reuse existing output directory: $output_dir" >&2
  echo "Set OUTPUT_DIR to a new path if this is an intentional rerun." >&2
  exit 2
fi

"${train_command[@]}"
