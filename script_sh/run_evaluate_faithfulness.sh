#!/bin/bash
set -euo pipefail

# Run from any directory; activate the desired Python environment beforehand.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [ "$#" -eq 0 ]; then
    echo "Usage: bash script_sh/run_evaluate_faithfulness.sh RESULT_ROOT [RESULT_ROOT ...]"
    exit 1
fi

PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda}"
STEP="${STEP:-224}"
BATCH_SIZE="${BATCH_SIZE:-16}"
OUTPUT_DIR="${OUTPUT_DIR:-faithfulness_aggregates}"

"$PYTHON" evaluate_script/run_evaluate_faithfulness.py \
    --input-dirs "$@" \
    --device "$DEVICE" \
    --step "$STEP" \
    --batch-size "$BATCH_SIZE"

"$PYTHON" evaluate_script/run_aggregate_faithfulness.py \
    --input-dirs "$@" \
    --output-dir "$OUTPUT_DIR"
