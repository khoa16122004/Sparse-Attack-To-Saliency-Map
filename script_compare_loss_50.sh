#!/bin/bash
#SBATCH --job-name=COMPARE_LOSS_50
#SBATCH --output=logs_COMPARE_LOSS_50/mps_%j.out
#SBATCH --error=logs_COMPARE_LOSS_50/mps_%j.err
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

mkdir -p logs_COMPARE_LOSS_50

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
OUTPUT_ROOT="compare_loss_50"
MODEL_NAMES="vgg16 resnet50 densnet121 vit_b_32"
NUM_SAMPLE=50
STRATEGY="uniform"

for MODEL_NAME in $MODEL_NAMES; do
    for STRATEGY in uniform saliency_guided; do
        for FITNESS in margin_saliency cross_entropy_saliency; do
            echo "[RUN] model=$MODEL_NAME strategy=$STRATEGY fitness=$FITNESS eps=50 num_sample=$NUM_SAMPLE output_root=$OUTPUT_ROOT"
            python run_batch.py \
                --model-name "$MODEL_NAME" \
                --num_sample "$NUM_SAMPLE" \
                --operator-strategy "$STRATEGY" \
                --eps 50 \
                --fitness-function "$FITNESS" \
                --output-root "$OUTPUT_ROOT"
        done
    done
done

echo "DONE. Outputs stored under: $OUTPUT_ROOT/$MODEL_NAME"
