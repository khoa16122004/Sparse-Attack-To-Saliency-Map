#!/bin/bash
#SBATCH --job-name=faith_agg
#SBATCH --output=ESWA/faith_agg_%j.out
#SBATCH --error=ESWA/faith_agg_%j.err
#SBATCH --account=uit
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=04:00:00

# CPU-only: aggregates saved faithfulness scores and evaluates saved maps for IoU.
set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    REPO_DIR="$SLURM_SUBMIT_DIR"
else
    REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd -- "$REPO_DIR"
export CUDA_VISIBLE_DEVICES=""

PYTHON="${PYTHON:-python}"
INPUT_DIR="${INPUT_DIR:-offical_faithfulness/server_run_seed}"
OUTPUT_DIR="${OUTPUT_DIR:-faithfulness_aggregates}"
TABLE_DIR="${TABLE_DIR:-result_latexs}"
EPS="${EPS:-}"
LOSS_TYPE="${LOSS_TYPE:-}"
STRATEGY="${STRATEGY:-}"
USE_TTEST="${USE_TTEST:-1}"
TTEST_ALPHA="${TTEST_ALPHA:-0.05}"
TTEST_CORRECTION="${TTEST_CORRECTION:-none}"
PRECISION="${PRECISION:-4}"
RUN_IOU="${RUN_IOU:-1}"
IMAGENET_ROOT="${IMAGENET_ROOT:-/datastore/quanphm/datasets/ImageNet/val}"
MASK_ROOT="${MASK_ROOT:-imagenet_mask}"
TOP_K_PERCENT="${TOP_K_PERCENT:-80}"
EMPTY_GT="${EMPTY_GT:-error}"

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

case "$RUN_IOU" in
    0) ;;
    1)
        if [[ ! -d "$IMAGENET_ROOT" || ! -f "$MASK_ROOT/mask_manifest.json" ]]; then
            printf 'ERROR: IoU requires IMAGENET_ROOT and MASK_ROOT/mask_manifest.json; set RUN_IOU=0 for faithfulness only\n' >&2
            exit 1
        fi
        ;;
    *) printf 'ERROR: RUN_IOU must be 0 or 1\n' >&2; exit 1 ;;
esac

TABLE_ARGS=(--faithfulness-input-dirs "${INPUT_DIRS[@]}" --output-dir "$TABLE_DIR")
case "$USE_TTEST" in
    0) TABLE_ARGS+=(--no-ttest) ;;
    1) TABLE_ARGS+=(--use-ttest --ttest-alpha "$TTEST_ALPHA" --ttest-correction "$TTEST_CORRECTION") ;;
    *)
        printf 'ERROR: USE_TTEST must be 0 or 1, got: %s\n' "$USE_TTEST" >&2
        exit 1
        ;;
esac
TABLE_ARGS+=(--precision "$PRECISION")
read -r -a EPS_VALUES <<< "$EPS"
if (( ${#EPS_VALUES[@]} > 0 )); then
    TABLE_ARGS+=(--eps "${EPS_VALUES[@]}")
fi
if [[ -n "$LOSS_TYPE" ]]; then
    TABLE_ARGS+=(--loss-type "$LOSS_TYPE")
fi
if [[ -n "$STRATEGY" ]]; then
    TABLE_ARGS+=(--strategy "$STRATEGY")
fi

if [[ "$RUN_IOU" == 1 ]]; then
    IOU_ARGS=(--input-dirs "${INPUT_DIRS[@]}" --imagenet-root "$IMAGENET_ROOT"
              --mask-root "$MASK_ROOT" --top-k-percent "$TOP_K_PERCENT"
              --empty-gt "$EMPTY_GT" --output-dir "$TABLE_DIR" --precision "$PRECISION"
              --records-output "$OUTPUT_DIR/localization_records.json")
    if (( ${#EPS_VALUES[@]} > 0 )); then
        IOU_ARGS+=(--eps "${EPS_VALUES[@]}")
    fi
    if [[ -n "$LOSS_TYPE" ]]; then
        IOU_ARGS+=(--loss-type "$LOSS_TYPE")
    fi
    if [[ -n "$STRATEGY" ]]; then
        IOU_ARGS+=(--strategy "$STRATEGY")
    fi
    printf '[IoU] Evaluating saved maps and rendering localization LaTeX (CPU)\n'
    "$PYTHON" -u -m evaluate_script.run_localization_iou "${IOU_ARGS[@]}"
    TABLE_ARGS+=(--localization-records "$OUTPUT_DIR/localization_records.json")
fi

printf '[1/2] Aggregating faithfulness metrics and Ins/Del curves (CPU)\n'
"$PYTHON" -u -m evaluate_script.run_aggregate_faithfulness \
    --input-dirs "${INPUT_DIRS[@]}" \
    --output-dir "$OUTPUT_DIR"

printf '[2/2] Rendering LaTeX tables from evaluate_latex_generate/faithfulness.text\n'
"$PYTHON" -u -m evaluate_latex_generate.generate_tables "${TABLE_ARGS[@]}"

printf '[OK] Metrics/curves: %s; LaTeX tables: %s/faithfulness_*\n' "$OUTPUT_DIR" "$TABLE_DIR"
if [[ "$RUN_IOU" == 1 ]]; then
    printf '[OK] Localization IoU tables/statistics: %s/localization_*\n' "$TABLE_DIR"
fi
