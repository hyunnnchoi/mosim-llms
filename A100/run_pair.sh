#!/usr/bin/env bash
# ────────────────────────────────────────────────────────
# run_pair.sh — Run two models simultaneously on 4+4 GPUs
# ────────────────────────────────────────────────────────
# Usage:
#   bash A100/run_pair.sh <model_a> <model_b> [total_steps]
#
# Examples:
#   bash A100/run_pair.sh gpt2 bert
#   bash A100/run_pair.sh gpt2 vgg16 200

set -euo pipefail
cd "$(dirname "$0")/.."

MODEL_A="${1:?Usage: $0 <model_a> <model_b> [total_steps]}"
MODEL_B="${2:?Usage: $0 <model_a> <model_b> [total_steps]}"
TOTAL_STEPS="${3:-100}"

GPU_A="${GPU_A:-0,1,2,3}"
GPU_B="${GPU_B:-4,5,6,7}"
NPROC_A=$(echo "$GPU_A" | tr ',' '\n' | wc -l | tr -d ' ')
NPROC_B=$(echo "$GPU_B" | tr ',' '\n' | wc -l | tr -d ' ')
PORT_A="${PORT_A:-29500}"
PORT_B="${PORT_B:-29501}"
OUTPUT_DIR="./A100/results"

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
echo "============================================"

mkdir -p "$OUTPUT_DIR"

# Build commands
CMD_A="CUDA_VISIBLE_DEVICES=${GPU_A} torchrun \
    --nproc_per_node=${NPROC_A} \
    --master_port=${PORT_A} \
    ${SCRIPT_A} \
    --mode pair \
    --partner ${MODEL_B} \
    --total-steps ${TOTAL_STEPS} \
    --output-dir ${OUTPUT_DIR}"

CMD_B="CUDA_VISIBLE_DEVICES=${GPU_B} torchrun \
    --nproc_per_node=${NPROC_B} \
    --master_port=${PORT_B} \
    ${SCRIPT_B} \
    --mode pair \
    --partner ${MODEL_A} \
    --total-steps ${TOTAL_STEPS} \
    --output-dir ${OUTPUT_DIR}"

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

if [ $FAIL -eq 0 ]; then
    echo ""
    echo "[PAIR] Both ${MODEL_A} and ${MODEL_B} finished successfully."
else
    echo ""
    echo "[PAIR] WARNING: One or both models failed."
    exit 1
fi
