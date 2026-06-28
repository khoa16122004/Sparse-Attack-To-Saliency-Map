#!/bin/bash
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <project> [entity] [input_root] [group]"
    echo "Example: $0 Sparse-Poshoc 22520691-uit compare_loss_50 compare_loss_50"
    exit 1
fi

PROJECT="$1"
ENTITY="${2:-}"
INPUT_ROOT="${3:-compare_loss_50}"
GROUP="${4:-compare_loss_50}"

echo "[INFO] Cleaning W&B local caches and run artifacts..."
rm -rf ./wandb
rm -rf ./.wandb
rm -rf "$HOME/.cache/wandb"
rm -rf /tmp/wandb

echo "[INFO] Running exporter..."
if [ -n "$ENTITY" ]; then
    python wandb_export.py --project "$PROJECT" --entity "$ENTITY" --input-root "$INPUT_ROOT" --group "$GROUP"
else
    python wandb_export.py --project "$PROJECT" --input-root "$INPUT_ROOT" --group "$GROUP"
fi

echo "[DONE] Cache reset + export finished."
