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

# =========================================================
# CHAY CODE: SO SANH 2 FITNESS LOSS
# =========================================================
MODEL_NAMES="resnet18"
NUM_SAMPLE=100
EPSILONS="100 50 20"

# Objective weights (can override by env or script args)
# Example:
#   W_MARGIN=0.7 W_SALIENCY=0.3 sbatch resnet18.sh
#   sbatch resnet18.sh 0.7 0.3
W_MARGIN="${W_MARGIN:-0.0}"
W_SALIENCY="${W_SALIENCY:-1.0}"
EXPLAIN_METHOD="${EXPLAIN_METHOD:-grad_cam}"
SEED="${SEED:-22520692}"
OUTPUT_ROOT="offical/server_run_seed/GA/$SEED/"

# At wm=0, ws=1, both fitness formulations are equivalent (attack objective is off).
FITNESSES="margin_saliency cross_entropy_saliency"
case "${W_MARGIN}:${W_SALIENCY}" in
    "0:1"|"0:1.0"|"0.0:1"|"0.0:1.0")
        FITNESSES="margin_saliency"
        ;;
esac


for MODEL_NAME in $MODEL_NAMES; do
    for STRATEGY in uniform; do
        for EPS in $EPSILONS; do
            for FITNESS in $FITNESSES; do
                echo "[RUN] model=$MODEL_NAME strategy=$STRATEGY fitness=$FITNESS eps=$EPS w_margin=$W_MARGIN w_saliency=$W_SALIENCY explain_method=$EXPLAIN_METHOD num_sample=$NUM_SAMPLE output_root=$OUTPUT_ROOT"
                python run_batch.py \
                    --model-name "$MODEL_NAME" \
                    --num_sample "$NUM_SAMPLE" \
                    --operator-strategy "$STRATEGY" \
                    --eps "$EPS" \
                    --w-margin "$W_MARGIN" \
                    --w-saliency "$W_SALIENCY" \
                    --seed "$SEED" \
                    --fitness-function "$FITNESS" \
                    --output-root "$OUTPUT_ROOT" \
                    --explain-method "$EXPLAIN_METHOD"
            done
        done
    done
done

echo "DONE. Outputs stored under: $OUTPUT_ROOT/$MODEL_NAME"
