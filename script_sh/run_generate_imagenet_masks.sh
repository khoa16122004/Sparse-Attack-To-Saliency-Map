#!/bin/bash
#SBATCH --job-name=ESWA
#SBATCH --output=ESWA/normal_%j.out
#SBATCH --error=ESWA/normal_%j.err
#SBATCH --account=uit
#SBATCH --partition=large
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8      # Tối đa 16 CPU
#SBATCH --mem=32G
#SBATCH --time=72:00:00        # Tối đa 3 ngày (72:00:00)
#SBATCH --gres=gpu:2g.35gb:1


nvidia-smi -L


source /datastore/hoangln/miniconda3/etc/profile.d/conda.sh
conda activate Sparse_Attack
cd /datastore/hoangln/Sparse-Attack-To-Saliency-Map
export SSL_CERT_FILE="$(python -c 'import certifi; print(certifi.where())')"
export REQUESTS_CA_BUNDLE="$SSL_CERT_FILE"
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-job$SLURM_JOB_ID
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log-job$SLURM_JOB_ID

rm -rf $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY


set -euo pipefail
REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd -- "$REPO_DIR"
PYTHON="${PYTHON:-python}"
IMAGENET_ROOT="${1:-${IMAGENET_ROOT:-/datastore/quanphm/datasets/ImageNet/val}}"
MASK_ROOT="${2:-${MASK_ROOT:-imagenet_mask}}"
ARGS=(--imagenet-root "$IMAGENET_ROOT" --output-dir "$MASK_ROOT"
      --device cuda --confidence-threshold "${SAM_CONFIDENCE:-0}"
      --max-image-side "${SAM_MAX_IMAGE_SIDE:-1024}")
if [[ -n "${PROMPTS_JSON:-}" ]]; then
    ARGS+=(--prompts-json "$PROMPTS_JSON")
fi
if [[ -n "${SAM_CHECKPOINT:-}" ]]; then
    ARGS+=(--checkpoint "$SAM_CHECKPOINT")
fi
case "${REPLACE:-0}" in
    0) ;;
    1) ARGS+=(--replace) ;;
    *) printf 'ERROR: REPLACE must be 0 or 1\n' >&2; exit 1 ;;
esac
"$PYTHON" -u -m evaluate_script.generate_imagenet_masks "${ARGS[@]}"

