#!/bin/bash
#SBATCH --job-name=CAUSAL_BATCH_SBGD
#SBATCH --output=logs_FaithFUll/mps_%j.out
#SBATCH --error=logs_FaithFUll/mps_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=mps:a100:2
#SBATCH --mem=4G
#SBATCH --time=72:00:00

set -euo pipefail

REQUIRED_VRAM=12000

module clear -f
source /home/elo/miniconda3/etc/profile.d/conda.sh
conda activate bcos_attack

echo "ENV: $CONDA_DEFAULT_ENV"
echo "PREFIX: $CONDA_PREFIX"
which python
python -c "import sys; print(sys.executable)"

mkdir -p logs_FaithFUll

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

MODEL_NAMES="vgg16"
NUM_SAMPLE=100
SELECTION_FILE="${SELECTION_FILE:-}"

EPSILON="${EPSILON:-1}"
K="${K:-50}"
ITERATIONS="${ITERATIONS:-200}"
ALPHA="${ALPHA:-0.0039215686}"
BETA="${BETA:-0.1}"
W_MARGIN="${W_MARGIN:-0.0}"
W_SALIENCY="${W_SALIENCY:-1.0}"
EXPLAIN_METHOD="${EXPLAIN_METHOD:-input_gradient}"
SEED="${SEED:-22520691}"
OUTPUT_ROOT="${OUTPUT_ROOT:-rivf_offical/server_run_seed/SBGD_causal/$SEED/}"

SPARSITY_RATIO="${SPARSITY_RATIO:-}"
TAU="${TAU:-0.5}"
DEBUG_GRAD="${DEBUG_GRAD:-1}"
DYNAMIC_MASK="${DYNAMIC_MASK:-1}"
SOFTPLUS_BETA="${SOFTPLUS_BETA:-10.0}"
ZERO_GRAD_PATIENCE="${ZERO_GRAD_PATIENCE:-3}"
ZERO_GRAD_JITTER="${ZERO_GRAD_JITTER:-1e-2}"

for MODEL_NAME in $MODEL_NAMES; do
    echo "[RUN] model=$MODEL_NAME epsilon=$EPSILON k=$K iter=$ITERATIONS alpha=$ALPHA beta=$BETA w_margin=$W_MARGIN w_saliency=$W_SALIENCY explain_method=$EXPLAIN_METHOD num_sample=$NUM_SAMPLE output_root=$OUTPUT_ROOT"

    CMD=(
        python sbgd_batch.py
        --model-name "$MODEL_NAME"
        --num_sample "$NUM_SAMPLE"
        --epsilon "$EPSILON"
        --k "$K"
        --iterations "$ITERATIONS"
        --alpha "$ALPHA"
        --beta "$BETA"
        --tau "$TAU"
        --w-margin "$W_MARGIN"
        --w-saliency "$W_SALIENCY"
        --seed "$SEED"
        --output-root "$OUTPUT_ROOT"
        --explain-method "$EXPLAIN_METHOD"
        --softplus-beta "$SOFTPLUS_BETA"
        --zero-grad-patience "$ZERO_GRAD_PATIENCE"
        --zero-grad-jitter "$ZERO_GRAD_JITTER"
    )

    if [ -n "$SPARSITY_RATIO" ]; then
        CMD+=(--sparsity-ratio "$SPARSITY_RATIO")
    fi
    if [ -n "$SELECTION_FILE" ]; then
        CMD+=(--selection-file "$SELECTION_FILE")
    fi
    if [ "$DEBUG_GRAD" = "1" ]; then
        CMD+=(--debug-grad)
    fi
    if [ "$DYNAMIC_MASK" = "1" ]; then
        CMD+=(--disable-fixed-mask-location)
    fi

    "${CMD[@]}"
done

echo "DONE. Outputs stored under: $OUTPUT_ROOT/<model>/<approach_tag>/"
