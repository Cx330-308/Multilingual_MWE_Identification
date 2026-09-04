#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"
cd "$PROJECT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
prediction_dir="${1:-$MWE_ARTIFACTS_DIR/predictions/e2-sqrt-sampling-seed13}"
submission_name="${2:-$(basename "$prediction_dir")-codabench.zip}"
validator="$PARSEME_DATA_DIR/tools/parseme_validate.py"

if [[ ! -d "$prediction_dir" ]]; then
  echo "Prediction directory not found: $prediction_dir" >&2
  exit 2
fi
if [[ ! -f "$validator" ]]; then
  echo "Official CUPT validator not found: $validator" >&2
  exit 2
fi
if [[ "$submission_name" = /* ]]; then
  output_path="$submission_name"
else
  output_path="$MWE_ARTIFACTS_DIR/submissions/$submission_name"
fi
if [[ -e "$output_path" ]]; then
  echo "Refusing to overwrite existing submission: $output_path" >&2
  exit 2
fi

languages=(EGY EL FA FR GRC HE JA KA LV NL PL PT RO SL SR SV UK)
submission_files=()
for language in "${languages[@]}"; do
  relative_path="$language/test.system.cupt"
  if [[ ! -f "$prediction_dir/$relative_path" ]]; then
    echo "Missing required prediction: $prediction_dir/$relative_path" >&2
    exit 2
  fi
  language_lower="$(printf '%s' "$language" | tr '[:upper:]' '[:lower:]')"
  "$PYTHON_BIN" "$validator" \
    --quiet \
    --lang "$language_lower" \
    --level 2 \
    "$prediction_dir/$relative_path"
  submission_files+=("$relative_path")
done

mkdir -p "$(dirname "$output_path")"
output_path="$(cd "$(dirname "$output_path")" && pwd)/$(basename "$output_path")"

(
  cd "$prediction_dir"
  zip -q "$output_path" "${submission_files[@]}"
)

printf 'Wrote %s\n' "$output_path"
printf 'Validated and included %d language files with no enclosing directory.\n' "${#submission_files[@]}"
