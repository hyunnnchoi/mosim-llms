#!/usr/bin/env bash
# ────────────────────────────────────────────────────────
# run_solo.sh — Baseline: train a single model on 4 GPUs
# ────────────────────────────────────────────────────────
# Usage:
#   bash A100/run_solo.sh <model_name> [total_steps] [gpu_ids]
#
# Default GPU assignment matches pair topology (cross-NUMA interleaved):
#   solo uses GPUs 0,2,4,6 — same layout as Job A in pair mode
#   This ensures solo vs pair comparison is fair (same PCIe paths).

set -euo pipefail
cd "$(dirname "$0")/.."

# Air-gapped environment: force offline mode for HuggingFace
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

# Match pair NCCL settings for fair comparison
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1

MODEL="${1:?Usage: $0 <model_name> [total_steps] [gpu_ids]}"
TOTAL_STEPS="${2:-100}"
GPU_IDS="${3:-0,2,4,6}"
NPROC=$(echo "$GPU_IDS" | tr ',' '\n' | wc -l | tr -d ' ')
MASTER_PORT="${MASTER_PORT:-29500}"
OUTPUT_DIR="./A100/results"

# Determine training script
case "$MODEL" in
    bert)    SCRIPT="A100/train_bert.py" ;;
    gpt2)    SCRIPT="A100/train_gpt2.py" ;;
    *)       SCRIPT="A100/train_cnn.py" ;;
esac

echo "============================================"
echo " SOLO: ${MODEL} on GPUs [${GPU_IDS}]"
echo " Steps: ${TOTAL_STEPS}, Procs: ${NPROC}"
echo " NCCL: P2P=off, SHM=off (force PCIe host relay)"
echo "============================================"

mkdir -p "$OUTPUT_DIR"

CMD="CUDA_VISIBLE_DEVICES=${GPU_IDS} torchrun \
    --nproc_per_node=${NPROC} \
    --master_port=${MASTER_PORT} \
    ${SCRIPT} \
    --mode solo \
    --total-steps ${TOTAL_STEPS} \
    --output-dir ${OUTPUT_DIR}"

# CNN models need --model flag
if [[ "$MODEL" != "bert" && "$MODEL" != "gpt2" ]]; then
    CMD="${CMD} --model ${MODEL}"
fi

echo "$CMD"
eval "$CMD"

echo ""
echo "[SOLO] ${MODEL} finished."
