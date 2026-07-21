#!/usr/bin/env bash

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PARSEME_DATA_DIR="${PARSEME_DATA_DIR:-$PROJECT_DIR/../sharedtask-data/2.0/subtask1}"
MWE_ARTIFACTS_DIR="${MWE_ARTIFACTS_DIR:-$PROJECT_DIR/artifacts}"

export PROJECT_DIR PARSEME_DATA_DIR MWE_ARTIFACTS_DIR
export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

if [[ ! -d "$PARSEME_DATA_DIR" ]]; then
  printf 'PARSEME data directory not found: %s\n' "$PARSEME_DATA_DIR" >&2
  printf 'Set PARSEME_DATA_DIR to the directory containing the language folders and tools/.\n' >&2
  exit 1
fi
