#!/bin/bash
#SBATCH --job-name=eval_samples
#SBATCH --output=evaluate_log/eval_samples_%j.out
#SBATCH --error=evaluate_log/eval_samples_%j.err
#SBATCH --account=uit
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=4G
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:2g.35gb:1

set -euo pipefail

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}}"
cd -- "$REPO_DIR"
source "${CONDA_SH:-/datastore/khoatn/miniconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-Sparse_Attack}"
nvidia-smi -L

PYTHON="${PYTHON:-python}"
IMAGENET_ROOT="${IMAGENET_ROOT:-/datastore/quanphm/datasets/ImageNet/val}"
MASK_ROOT="${MASK_ROOT:-imagenet_mask_new_new}"
STEP="${STEP:-224}"
BATCH_SIZE="${BATCH_SIZE:-128}"
KERNEL_SIZE="${KERNEL_SIZE:-11}"
KERNEL_SIGMA="${KERNEL_SIGMA:-5}"
IOU_THRESHOLD="${IOU_THRESHOLD:-0.5}"
TOP_K_PERCENT="${TOP_K_PERCENT:-}"
EMPTY_GT="${EMPTY_GT:-zero}"
REPLACE="${REPLACE:-0}"

if (( $# > 0 )); then
    INPUT_DIRS=("$@")
elif [[ -n "${INPUT_DIR:-}" ]]; then
    INPUT_DIRS=("$INPUT_DIR")
else
    INPUT_DIRS=(
        "/datastore/khangtd/Sparse-Attack-To-Saliency-Map/khangtd/offical_faithfulness/server_run_seed"
        "/datastore/khoatn/Sparse-Attack-To-Saliency-Map/khoatn/offical_faithfulness/server_run_seed"
        "/datastore/hoangln/Sparse-Attack-To-Saliency-Map/hoangln/offical_faithfulness/server_run_seed"
    )
fi

for input_dir in "${INPUT_DIRS[@]}"; do
    if [[ ! -d "$input_dir" ]]; then
        printf 'ERROR: input directory does not exist: %s\n' "$input_dir" >&2
        exit 1
    fi
done
if [[ ! -d "$IMAGENET_ROOT" || ! -f "$MASK_ROOT/mask_manifest.json" ]]; then
    printf 'ERROR: IoU requires IMAGENET_ROOT and MASK_ROOT/mask_manifest.json\n' >&2
    exit 1
fi

ARGS=(
    --input-dirs "${INPUT_DIRS[@]}"
    --imagenet-root "$IMAGENET_ROOT" --mask-root "$MASK_ROOT" --device cuda
    --step "$STEP" --batch-size "$BATCH_SIZE"
    --kernel-size "$KERNEL_SIZE" --kernel-sigma "$KERNEL_SIGMA"
    --empty-gt "$EMPTY_GT"
)
if [[ -n "$TOP_K_PERCENT" ]]; then
    ARGS+=(--top-k-percent "$TOP_K_PERCENT")
else
    ARGS+=(--threshold "$IOU_THRESHOLD")
fi
case "$REPLACE" in
    0) ;;
    1) ARGS+=(--replace) ;;
    *) printf 'ERROR: REPLACE must be 0 or 1\n' >&2; exit 1 ;;
esac
if [[ -n "${MODEL:-}" ]]; then ARGS+=(--model "$MODEL"); fi
if [[ -n "${EXPLAIN_METHOD:-}" ]]; then ARGS+=(--explain-method "$EXPLAIN_METHOD"); fi

"$PYTHON" -u -m evaluate "${ARGS[@]}"
