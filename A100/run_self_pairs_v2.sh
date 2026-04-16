#!/usr/bin/env bash
# ────────────────────────────────────────────────────────────
# run_self_pairs_v2.sh — Self-pair experiments (A+A, B+B, ...)
# ────────────────────────────────────────────────────────────
# Same model on both sides: primary (0,2,4,6) + interferer (1,3,5,7)
# Uses --job-id / --partner-id to avoid barrier file collisions.
#
# Usage:
#   bash A100/run_self_pairs_v2.sh [total_steps]

set -euo pipefail
cd "$(dirname "$0")/.."

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

TOTAL_STEPS="${1:-100}"
RESULTS_DIR="./A100/results_v2"
LOG_DIR="./A100/logs_v2"

mkdir -p "$RESULTS_DIR" "$LOG_DIR"

MODELS=(gpt2 bert whisper resnet44 resnet110 resnet50 vgg16 googlenet inception3 densenet40_k12 densenet100_k12)

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SUMMARY_LOG="${LOG_DIR}/self_pairs_${TIMESTAMP}.log"

echo "======================================================" | tee "$SUMMARY_LOG"
echo " A100 Self-Pair Experiments (same model interference)" | tee -a "$SUMMARY_LOG"
echo " Primary: GPUs 0,2,4,6 / Interferer: GPUs 1,3,5,7"   | tee -a "$SUMMARY_LOG"
echo " Steps: ${TOTAL_STEPS}" | tee -a "$SUMMARY_LOG"
echo " Experiments: ${#MODELS[@]}" | tee -a "$SUMMARY_LOG"
echo "======================================================" | tee -a "$SUMMARY_LOG"
echo "" | tee -a "$SUMMARY_LOG"

get_script() {
    case "$1" in
        bert)    echo "A100/train_bert.py" ;;
        gpt2)    echo "A100/train_gpt2.py" ;;
        whisper) echo "A100/train_whisper.py" ;;
        *)       echo "A100/train_cnn.py" ;;
    esac
}

PASS=0
FAIL=0

for model in "${MODELS[@]}"; do
    desc="pair_${model}_${model}"
    echo "--------------------------------------------" | tee -a "$SUMMARY_LOG"
    echo "[RUN] $desc" | tee -a "$SUMMARY_LOG"
    start_time=$(date +%s)

    SCRIPT=$(get_script "$model")
    BARRIER_DIR="/tmp/a100_barrier_v2"
    rm -rf "$BARRIER_DIR"
    mkdir -p "$BARRIER_DIR"

    # Common args (model flag only for CNN)
    MODEL_FLAG=""
    if [[ "$model" != "bert" && "$model" != "gpt2" && "$model" != "whisper" ]]; then
        MODEL_FLAG="--model ${model}"
    fi

    # Primary: measured, fixed steps
    CUDA_VISIBLE_DEVICES=0,2,4,6 \
    NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 \
    torchrun --nproc_per_node=4 --master_port=29500 \
    ${SCRIPT} ${MODEL_FLAG} \
        --mode pair --partner "${model}" \
        --total-steps ${TOTAL_STEPS} --output-dir ${RESULTS_DIR} \
        --role primary \
        --job-id "${model}_primary" --partner-id "${model}_interferer" \
        --barrier-dir ${BARRIER_DIR} \
    >> "${LOG_DIR}/${desc}_A.log" 2>&1 &
    PID_A=$!

    # Interferer: same model, runs until primary is done
    CUDA_VISIBLE_DEVICES=1,3,5,7 \
    NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 \
    torchrun --nproc_per_node=4 --master_port=29501 \
    ${SCRIPT} ${MODEL_FLAG} \
        --mode pair --partner "${model}" \
        --total-steps ${TOTAL_STEPS} --output-dir ${RESULTS_DIR} \
        --role interferer \
        --job-id "${model}_interferer" --partner-id "${model}_primary" \
        --barrier-dir ${BARRIER_DIR} \
    >> "${LOG_DIR}/${desc}_B.log" 2>&1 &
    PID_B=$!

    pair_fail=0
    wait $PID_A || pair_fail=1
    wait $PID_B || pair_fail=1
    rm -rf "$BARRIER_DIR"

    elapsed=$(( $(date +%s) - start_time ))
    if [ $pair_fail -eq 0 ]; then
        echo "[OK]  $desc (${elapsed}s)" | tee -a "$SUMMARY_LOG"
        PASS=$((PASS + 1))
    else
        echo "[FAIL] $desc (${elapsed}s)" | tee -a "$SUMMARY_LOG"
        FAIL=$((FAIL + 1))
    fi
done

echo "" | tee -a "$SUMMARY_LOG"
echo "======================================================" | tee -a "$SUMMARY_LOG"
echo " Self-pair experiments complete!" | tee -a "$SUMMARY_LOG"
echo " Passed: ${PASS}" | tee -a "$SUMMARY_LOG"
echo " Failed: ${FAIL}" | tee -a "$SUMMARY_LOG"
echo " Results: ${RESULTS_DIR}/" | tee -a "$SUMMARY_LOG"
echo "======================================================" | tee -a "$SUMMARY_LOG"
