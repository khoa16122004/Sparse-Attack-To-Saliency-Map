#!/bin/bash
#SBATCH --job-name=COMPARE_LOSS_50
#SBATCH --output=faithfull/mps_%j.out
#SBATCH --error=faithfull/mps_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=mps:a100:2
#SBATCH --mem=4G
#SBATCH --time=72:00:00

set -euo pipefail

REQUIRED_VRAM=12000

# =========================================================
# CHUAN BI MOI TRUONG
# =========================================================
module clear -f
source /home/elo/miniconda3/etc/profile.d/conda.sh
conda activate bcos_attack

echo "ENV: $CONDA_DEFAULT_ENV"
echo "PREFIX: $CONDA_PREFIX"
which python
python -c "import sys; print(sys.executable)"

mkdir -p faithfull

unset CUDA_VISIBLE_DEVICES
CHECK_OUT=$(/usr/local/bin/gpu_check.sh $REQUIRED_VRAM $SLURM_JOB_ID)
EXIT_CODE=$?

if [ $EXIT_CODE -eq 10 ]; then
    echo "$CHECK_OUT"
    exit 0
elif [ $EXIT_CODE -eq 11 ]; then
    echo "$CHECK_OUT"
    exit 1
fi

BEST_GPU=$CHECK_OUT
echo "Job $SLURM_JOB_ID bat dau tren GPU: $BEST_GPU"

export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-job$SLURM_JOB_ID
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log-job$SLURM_JOB_ID

rm -rf "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"

export CUDA_VISIBLE_DEVICES=$BEST_GPU

# =========================================================
# CHAY CODE: SO SANH 2 FITNESS LOSS
# =========================================================
MODEL_NAMES="${MODEL_NAMES:-densenet121 densenet169 resnet18 resnet50 vgg16 vgg19 vit_b_16 vit_b_32 swin_t}"
IMAGENET_VAL_ROOT="${IMAGENET_VAL_ROOT:-/datastore/elo/quanphm/dataset/ImageNet1K/val}"
EMIT_ROOT="${EMIT_ROOT:-/datastore/elo/quanphm/dataset/ImageNet1K/val}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SEED="${SEED:-22520691}"

echo "[INFO] IMAGENET_VAL_ROOT=$IMAGENET_VAL_ROOT"
echo "[INFO] EMIT_ROOT=$EMIT_ROOT"
echo "[INFO] BATCH_SIZE=$BATCH_SIZE"
echo "[INFO] NUM_WORKERS=$NUM_WORKERS"
echo "[INFO] SEED=$SEED"
echo "[INFO] MODELS=$MODEL_NAMES"

for MODEL_NAME in $MODEL_NAMES; do
    echo "[RUN] classify model=$MODEL_NAME"
    python build_model_eval_json.py \
        --model "$MODEL_NAME" \
        --imagenet-val-root "$IMAGENET_VAL_ROOT" \
        --emit-root "$EMIT_ROOT" \
        --output "model_evaluation_results/${MODEL_NAME}.json" \
        --batch-size "$BATCH_SIZE" \
        --num-workers "$NUM_WORKERS" \
        --device cuda
done

echo "[RUN] create *_selection.json from model_evaluation_results/*.json"
python model_evaluation_results/process.py

echo "[RUN] sample random 100 from each *_selection.json"
python sample_random_selection_100.py \
    --selection-dir model_evaluation_results \
    --k 100 \
    --seed "$SEED" \
    --overwrite

echo "DONE. Generated files in model_evaluation_results"
