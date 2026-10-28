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

# CPU-only: reads existing faithfulness.json files; does not run models.
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

printf '[1/2] Aggregating faithfulness metrics and Ins/Del curves (CPU)\n'
"$PYTHON" -u -m evaluate_script.run_aggregate_faithfulness \
    --input-dirs "${INPUT_DIRS[@]}" \
    --output-dir "$OUTPUT_DIR"

printf '[2/2] Rendering LaTeX tables from evaluate_latex_generate/faithfulness.text\n'
"$PYTHON" -u -m evaluate_latex_generate.generate_tables "${TABLE_ARGS[@]}"

printf '[OK] Metrics/curves: %s; LaTeX tables: %s/faithfulness_*\n' "$OUTPUT_DIR" "$TABLE_DIR"
