#!/bin/bash
#SBATCH --job-name=ESWA
#SBATCH --output=ESWA/normal_%j.out
#SBATCH --error=ESWA/normal_%j.err
#SBATCH --account=uit
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8      # Tối đa 16 CPU
#SBATCH --mem=4G               # Tối đa 72 GB RAM
#SBATCH --time=72:00:00        # Tối đa 3 ngày (72:00:00)
#SBATCH --gres=gpu:2g.35gb:1


nvidia-smi -L


source /datastore/khoatn/miniconda3/etc/profile.d/conda.sh
conda activate Sparse_Attack
cd /datastore/khoatn/Sparse-Attack-To-Saliency-Map
export SSL_CERT_FILE="$(python -c 'import certifi; print(certifi.where())')"
export REQUESTS_CA_BUNDLE="$SSL_CERT_FILE"
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-job$SLURM_JOB_ID
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log-job$SLURM_JOB_ID

rm -rf $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY


STEP="${STEP:-224}"
BATCH_SIZE="${BATCH_SIZE:-64}"
DEVICE="${DEVICE:-cuda}"
OUTPUT_DIR="${OUTPUT_DIR:-faithfulness_aggregates}"
INPUT_DIR="${INPUT_DIR:-offical/server_run_seed}"
if (( $# > 0 )); then
    INPUT_DIRS=("$@")
else
    INPUT_DIRS=("$INPUT_DIR")
fi

python evaluate_script/run_evaluate_faithfulness.py \
    --input-dirs "${INPUT_DIRS[@]}" \
    --device "$DEVICE" \
    --step "$STEP" \
    --batch-size "$BATCH_SIZE" \
    --replace \
    --output-dir "$OUTPUT_DIR"
