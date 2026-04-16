#!/usr/bin/env bash
# ────────────────────────────────────────────────────────
# run_pair.sh — Run two models simultaneously on 4+4 GPUs
# ────────────────────────────────────────────────────────
# Usage:
#   bash A100/run_pair.sh <model_a> <model_b> [total_steps]
#
# GPU topology (A100 8-GPU, no NVLink):
#   NUMA 0: GPU 0,1,2,3   NUMA 1: GPU 4,5,6,7
#   PIX pairs: (0,1) (2,3) (4,5) (6,7)
#
# Default GPU assignment maximizes PCIe contention:
#   Job A: 0,2,4,6  Job B: 1,3,5,7
#   - Splits every PIX pair → both jobs share each PCIe switch
#   - Both jobs span NUMA 0+1 → QPI/UPI contention
#
# Override: GPU_A=0,1,2,3 GPU_B=4,5,6,7 bash A100/run_pair.sh ...

set -euo pipefail
cd "$(dirname "$0")/.."

# Air-gapped environment: force offline mode for HuggingFace
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

# Force NCCL to relay all data through host memory via PCIe
# (no GPU P2P, no shared memory shortcut)
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1

MODEL_A="${1:?Usage: $0 <model_a> <model_b> [total_steps]}"
MODEL_B="${2:?Usage: $0 <model_a> <model_b> [total_steps]}"
TOTAL_STEPS="${3:-100}"

# Interleaved across NUMA + PIX for maximum contention
GPU_A="${GPU_A:-0,2,4,6}"
GPU_B="${GPU_B:-1,3,5,7}"
NPROC_A=$(echo "$GPU_A" | tr ',' '\n' | wc -l | tr -d ' ')
NPROC_B=$(echo "$GPU_B" | tr ',' '\n' | wc -l | tr -d ' ')
PORT_A="${PORT_A:-29500}"
PORT_B="${PORT_B:-29501}"
OUTPUT_DIR="./A100/results"

# Barrier directory for training-start synchronization
BARRIER_DIR="/tmp/a100_barrier"
rm -rf "$BARRIER_DIR"
mkdir -p "$BARRIER_DIR"

# Determine scripts
get_script() {
    case "$1" in
        bert) echo "A100/train_bert.py" ;;
        gpt2) echo "A100/train_gpt2.py" ;;
        *)    echo "A100/train_cnn.py" ;;
    esac
}

SCRIPT_A=$(get_script "$MODEL_A")
SCRIPT_B=$(get_script "$MODEL_B")

echo "============================================"
echo " PAIR: ${MODEL_A} [GPUs ${GPU_A}] + ${MODEL_B} [GPUs ${GPU_B}]"
echo " Steps: ${TOTAL_STEPS}"
echo " NCCL: P2P=off, SHM=off (force PCIe host relay)"
echo "============================================"

mkdir -p "$OUTPUT_DIR"

# Build commands (with barrier dir for start synchronization)
CMD_A="CUDA_VISIBLE_DEVICES=${GPU_A} torchrun \
    --nproc_per_node=${NPROC_A} \
    --master_port=${PORT_A} \
    ${SCRIPT_A} \
    --mode pair \
    --partner ${MODEL_B} \
    --total-steps ${TOTAL_STEPS} \
    --output-dir ${OUTPUT_DIR} \
    --barrier-dir ${BARRIER_DIR}"

CMD_B="CUDA_VISIBLE_DEVICES=${GPU_B} torchrun \
    --nproc_per_node=${NPROC_B} \
    --master_port=${PORT_B} \
    ${SCRIPT_B} \
    --mode pair \
    --partner ${MODEL_A} \
    --total-steps ${TOTAL_STEPS} \
    --output-dir ${OUTPUT_DIR} \
    --barrier-dir ${BARRIER_DIR}"

# CNN models need --model flag
if [[ "$MODEL_A" != "bert" && "$MODEL_A" != "gpt2" ]]; then
    CMD_A="${CMD_A} --model ${MODEL_A}"
fi
if [[ "$MODEL_B" != "bert" && "$MODEL_B" != "gpt2" ]]; then
    CMD_B="${CMD_B} --model ${MODEL_B}"
fi

echo "[Model A] $CMD_A"
echo "[Model B] $CMD_B"
echo ""

# Launch both simultaneously and wait
eval "$CMD_A" &
PID_A=$!

eval "$CMD_B" &
PID_B=$!

echo "[PAIR] Launched ${MODEL_A} (PID=$PID_A) and ${MODEL_B} (PID=$PID_B)"
echo "[PAIR] Waiting for both to finish..."

FAIL=0
wait $PID_A || { echo "[PAIR] ${MODEL_A} failed (exit $?)"; FAIL=1; }
wait $PID_B || { echo "[PAIR] ${MODEL_B} failed (exit $?)"; FAIL=1; }

# Cleanup barrier
rm -rf "$BARRIER_DIR"

if [ $FAIL -eq 0 ]; then
    echo ""
    echo "[PAIR] Both ${MODEL_A} and ${MODEL_B} finished successfully."
else
    echo ""
    echo "[PAIR] WARNING: One or both models failed."
    exit 1
fi
