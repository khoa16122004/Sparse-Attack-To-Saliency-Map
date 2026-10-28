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

PYTHON="${PYTHON:-python}"
INPUT_DIR="${INPUT_DIR:-offical_faithfulness/server_run_seed}"
OUTPUT_DIR="${OUTPUT_DIR:-faithfulness_aggregates}"
TABLE_DIR="${TABLE_DIR:-result_latexs}"
STEP="${STEP:-224}"
BATCH_SIZE="${BATCH_SIZE:-16}"
KERNEL_SIZE="${KERNEL_SIZE:-11}"
KERNEL_SIGMA="${KERNEL_SIGMA:-5}"
REPLACE="${REPLACE:-0}"
EPS="${EPS:-}"
LOSS_TYPE="${LOSS_TYPE:-}"
STRATEGY="${STRATEGY:-}"
USE_TTEST="${USE_TTEST:-1}"
TTEST_ALPHA="${TTEST_ALPHA:-0.05}"
TTEST_CORRECTION="${TTEST_CORRECTION:-none}"
PRECISION="${PRECISION:-4}"

if (( $# > 0 )); then
    INPUT_DIRS=("$@")
else
    INPUT_DIRS=("$INPUT_DIR")
fi

for input_dir in "${INPUT_DIRS[@]}"; do
    if [[ ! -d "$input_dir" ]]; then
        printf 'ERROR: input directory does not exist: %s\n' "$input_dir" >&2
        exit 1
    fi
done

EVALUATION_ARGS=()
case "$REPLACE" in
    0) ;;
    1) EVALUATION_ARGS+=(--replace) ;;
    *)
        printf 'ERROR: REPLACE must be 0 or 1, got: %s\n' "$REPLACE" >&2
        exit 1
        ;;
esac
TABLE_ARGS=(--faithfulness-input-dirs "${INPUT_DIRS[@]}" --output-dir "$TABLE_DIR")
case "$USE_TTEST" in
    0) TABLE_ARGS+=(--no-ttest) ;;
    1) TABLE_ARGS+=(--use-ttest --ttest-alpha "$TTEST_ALPHA" --ttest-correction "$TTEST_CORRECTION") ;;
    *)
        printf 'ERROR: USE_TTEST must be 0 or 1, got: %s\n' "$USE_TTEST" >&2
        exit 1
        ;;
esac
TABLE_ARGS+=(--precision "$PRECISION")
read -r -a EPS_VALUES <<< "$EPS"
if (( ${#EPS_VALUES[@]} > 0 )); then
    TABLE_ARGS+=(--eps "${EPS_VALUES[@]}")
fi
if [[ -n "$LOSS_TYPE" ]]; then
    TABLE_ARGS+=(--loss-type "$LOSS_TYPE")
fi
if [[ -n "$STRATEGY" ]]; then
    TABLE_ARGS+=(--strategy "$STRATEGY")
fi

printf '[1/3] Evaluating sample faithfulness on CUDA (replace=%s)\n' "$REPLACE"
"$PYTHON" -u -m evaluate_script.run_evaluate_faithfulness \
    --input-dirs "${INPUT_DIRS[@]}" \
    --device cuda \
    --step "$STEP" \
    --batch-size "$BATCH_SIZE" \
    --kernel-size "$KERNEL_SIZE" \
    --kernel-sigma "$KERNEL_SIGMA" \
    --output-dir "$OUTPUT_DIR" \
    "${EVALUATION_ARGS[@]}"

printf '[2/3] Aggregating faithfulness metrics and Ins/Del curves\n'
"$PYTHON" -u -m evaluate_script.run_aggregate_faithfulness \
    --input-dirs "${INPUT_DIRS[@]}" \
    --output-dir "$OUTPUT_DIR"

printf '[3/3] Rendering LaTeX tables from evaluate_latex_generate/faithfulness.text\n'
"$PYTHON" -u -m evaluate_latex_generate.generate_tables "${TABLE_ARGS[@]}"

printf '[OK] Metrics/curves: %s; LaTeX tables: %s/faithfulness_*\n' "$OUTPUT_DIR" "$TABLE_DIR"
