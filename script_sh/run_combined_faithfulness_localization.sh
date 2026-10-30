#!/bin/bash

# Join saved per-sample faithfulness and localization scores without inference.
set -euo pipefail

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$REPO_DIR"
export CUDA_VISIBLE_DEVICES=""

PYTHON="${PYTHON:-python}"
INPUT_DIR="${INPUT_DIR:-offical_faithfulness/server_run_seed}"
LOCALIZATION_RECORDS="${LOCALIZATION_RECORDS:-faithfulness_aggregates/localization_records.json}"
TABLE_DIR="${TABLE_DIR:-result_latexs}"

if (( $# > 0 )); then
    INPUT_DIRS=("$@")
else
    INPUT_DIRS=("$INPUT_DIR")
fi

for input_dir in "${INPUT_DIRS[@]}"; do
    if [[ ! -d "$input_dir" ]]; then
        printf 'ERROR: input directory does not exist: %s\n' "$input_dir" >&2
        exit 1
    fi
done
if [[ ! -f "$LOCALIZATION_RECORDS" ]]; then
    printf 'ERROR: localization records do not exist: %s; generate them with run_localization_iou --records-output first\n' "$LOCALIZATION_RECORDS" >&2
    exit 1
fi

ARGS=(--faithfulness-input-dirs "${INPUT_DIRS[@]}"
      --localization-records "$LOCALIZATION_RECORDS"
      --output-dir "$TABLE_DIR" --precision "${PRECISION:-3}" --sd-format subscript)
case "${USE_TTEST:-0}" in
    0) ARGS+=(--no-ttest) ;;
    1) ARGS+=(--use-ttest --ttest-alpha "${TTEST_ALPHA:-0.05}"
              --ttest-correction "${TTEST_CORRECTION:-none}") ;;
    *) printf 'ERROR: USE_TTEST must be 0 or 1\n' >&2; exit 1 ;;
esac
read -r -a EPS_VALUES <<< "${EPS:-}"
if (( ${#EPS_VALUES[@]} > 0 )); then
    ARGS+=(--eps "${EPS_VALUES[@]}")
fi
if [[ -n "${LOSS_TYPE:-}" ]]; then
    ARGS+=(--loss-type "$LOSS_TYPE")
fi
if [[ -n "${STRATEGY:-}" ]]; then
    ARGS+=(--strategy "$STRATEGY")
fi

printf '[JOIN] Combining saved faithfulness.json and %s\n' "$LOCALIZATION_RECORDS"
"$PYTHON" -u -m evaluate_latex_generate.generate_tables "${ARGS[@]}"
printf '[OK] Combined Ins / Del / IMD / IoU tables: %s/faithfulness_*\n' "$TABLE_DIR"
