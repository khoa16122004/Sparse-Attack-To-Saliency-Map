#!/bin/bash
#SBATCH --job-name=imagenet_sam3
#SBATCH --output=sam3_masks_%j.out
#SBATCH --error=sam3_masks_%j.err
#SBATCH --account=uit
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:2g.35gb:1

set -euo pipefail
REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd -- "$REPO_DIR"
PYTHON="${PYTHON:-python}"
IMAGENET_ROOT="${1:-${IMAGENET_ROOT:-salient_imagenet_dataset/salient_imagenet_dataset}}"
MASK_ROOT="${2:-${MASK_ROOT:-imagenet_mask}}"
ARGS=(--imagenet-root "$IMAGENET_ROOT" --output-dir "$MASK_ROOT"
      --device cuda --confidence-threshold "${SAM_CONFIDENCE:-0.5}")
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
