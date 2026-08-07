#!/bin/bash
#SBATCH --job-name=CAUSAL_BATCH_PGD
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

ITERATIONS="${ITERATIONS:-80}"
ALPHA="${ALPHA:-1.0}"
TAU="${TAU:-0.5}"
EPS="${EPS:-50}"
LAMBDA_MARGIN="${LAMBDA_MARGIN:-0.7}"
LAMBDA_SPARSE="${LAMBDA_SPARSE:-0.1}"
SPARSE_TARGET="${SPARSE_TARGET:-}"
EXPLAIN_METHOD="${EXPLAIN_METHOD:-simple_gradient}"
SEED="${SEED:-22520691}"
OUTPUT_ROOT="rivf_offical/server_run_seed/PGD_sparse_causal/$SEED/"

STEP="${STEP:-224}"
KERNEL_SIZE="${KERNEL_SIZE:-11}"
KERNEL_SIGMA="${KERNEL_SIGMA:-5}"
VERBOSE="${VERBOSE:-0}"
SAVE_PROCESS="${SAVE_PROCESS:-0}"
AUTOCAST="${AUTOCAST:-1}"
AUTOCAST_DTYPE="${AUTOCAST_DTYPE:-float16}"

for MODEL_NAME in $MODEL_NAMES; do
    echo "[RUN] model=$MODEL_NAME iter=$ITERATIONS alpha=$ALPHA tau=$TAU eps=$EPS lambda_margin=$LAMBDA_MARGIN lambda_sparse=$LAMBDA_SPARSE sparse_target=$SPARSE_TARGET explain_method=$EXPLAIN_METHOD num_sample=$NUM_SAMPLE output_root=$OUTPUT_ROOT"

    CMD=(
        python run_batch_causal_pgd_sparse.py
        --model-name "$MODEL_NAME"
        --num_sample "$NUM_SAMPLE"
        --iterations "$ITERATIONS"
        --alpha "$ALPHA"
        --threshold "$TAU"
        --eps "$EPS"
        --lambda-margin "$LAMBDA_MARGIN"
        --lambda-sparse "$LAMBDA_SPARSE"
        --seed "$SEED"
        --output-root "$OUTPUT_ROOT"
        --explain-method "$EXPLAIN_METHOD"
        --step "$STEP"
        --kernel-size "$KERNEL_SIZE"
        --kernel-sigma "$KERNEL_SIGMA"
        --verbose "$VERBOSE"
        --autocast-dtype "$AUTOCAST_DTYPE"
    )

    if [ "$SAVE_PROCESS" = "1" ]; then
        CMD+=(--save-process)
    fi
    if [ "$AUTOCAST" = "1" ]; then
        CMD+=(--autocast)
    fi
    if [ -n "$SPARSE_TARGET" ]; then
        CMD+=(--sparse-target "$SPARSE_TARGET")
    fi

    "${CMD[@]}"
done

echo "DONE. Outputs stored under: $OUTPUT_ROOT/$MODEL_NAME"
