#!/bin/bash
#SBATCH --job-name=process_result
#SBATCH --output=evaluate_log/process_result_%j.out
#SBATCH --error=evaluate_log/process_result_%j.err
#SBATCH --account=uit
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=12:00:00

# CPU-only: aggregates per-sample evaluation.json files into CNN LaTeX tables.
set -euo pipefail

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}}"
cd -- "$REPO_DIR"
source "${CONDA_SH:-/datastore/khoatn/miniconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-Sparse_Attack}"

PYTHON="${PYTHON:-python}"
OUTPUT_DIR="${OUTPUT_DIR:-result_latexs/processed}"
DEBUG_FILES="${DEBUG_FILES:-0}"

if (( $# > 0 )); then
    INPUT_DIRS=("$@")
elif [[ -n "${INPUT_DIR:-}" ]]; then
    INPUT_DIRS=("$INPUT_DIR")
else
    INPUT_DIRS=(
        "/datastore/khoatn/Sparse-Attack-To-Saliency-Map/offical_faithfulness/server_run_seed"
        "/datastore/hoangln/Sparse-Attack-To-Saliency-Map/offical_faithfulness/server_run_seed"
    )
fi

for input_dir in "${INPUT_DIRS[@]}"; do
    if [[ ! -d "$input_dir" ]]; then
        printf 'ERROR: input directory does not exist: %s\n' "$input_dir" >&2
        exit 1
    fi
done

ARGS=(--input-dirs "${INPUT_DIRS[@]}" --output-dir "$OUTPUT_DIR")
case "$DEBUG_FILES" in
    0) ;;
    1) ARGS+=(--debug-files) ;;
    *) printf 'ERROR: DEBUG_FILES must be 0 or 1\n' >&2; exit 1 ;;
esac

"$PYTHON" -u process_result.py "${ARGS[@]}"
