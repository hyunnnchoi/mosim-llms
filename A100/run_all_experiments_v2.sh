#!/usr/bin/env bash
# ────────────────────────────────────────────────────────────
# run_all_experiments_v2.sh — V2: optimal solo vs worst-case pair
# ────────────────────────────────────────────────────────────
# Solo:  GPUs 0,1,2,3 (NUMA 0, P2P/SHM enabled → best case)
# Pair:  Job A 0,2,4,6 / Job B 1,3,5,7 (interleaved, P2P/SHM off → worst case)
#
# Usage:
#   bash A100/run_all_experiments_v2.sh [total_steps]

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
SUMMARY_LOG="${LOG_DIR}/experiment_summary_${TIMESTAMP}.log"

echo "======================================================" | tee "$SUMMARY_LOG"
echo " A100 Interference Experiments V2" | tee -a "$SUMMARY_LOG"
echo " Solo:  GPUs 0,1,2,3 (NUMA 0, P2P+SHM ON)" | tee -a "$SUMMARY_LOG"
echo " Pair:  A=0,2,4,6 B=1,3,5,7 (interleaved, P2P+SHM OFF)" | tee -a "$SUMMARY_LOG"
echo " Steps: ${TOTAL_STEPS}" | tee -a "$SUMMARY_LOG"
echo " Solo baselines: ${#MODELS[@]}" | tee -a "$SUMMARY_LOG"
NUM_PAIRS=$(( ${#MODELS[@]} * (${#MODELS[@]} - 1) ))
echo " Pair combinations: ${NUM_PAIRS} (both directions)" | tee -a "$SUMMARY_LOG"
echo " Total experiments: $(( ${#MODELS[@]} + NUM_PAIRS ))" | tee -a "$SUMMARY_LOG"
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

build_args() {
    local model="$1" mode="$2" partner="$3" role="${4:-primary}"
    local args="--mode ${mode} --total-steps ${TOTAL_STEPS} --output-dir ${RESULTS_DIR} --role ${role}"
    if [[ "$mode" == "pair" ]]; then
        args="${args} --partner ${partner} --barrier-dir ${BARRIER_DIR}"
    fi
    if [[ "$model" != "bert" && "$model" != "gpt2" && "$model" != "whisper" ]]; then
        args="${args} --model ${model}"
    fi
    echo "$args"
}

PASS=0
FAIL=0

# ════════════════════════════════════════
# Phase 1: Solo baselines
#   GPUs 0,1,2,3 (NUMA 0), NCCL P2P+SHM enabled
# ════════════════════════════════════════
echo "=== Phase 1: Solo Baselines (GPUs 0,1,2,3, NUMA 0) ===" | tee -a "$SUMMARY_LOG"

for model in "${MODELS[@]}"; do
    desc="solo_${model}"
    echo "--------------------------------------------" | tee -a "$SUMMARY_LOG"
    echo "[RUN] $desc" | tee -a "$SUMMARY_LOG"
    start_time=$(date +%s)

    SCRIPT=$(get_script "$model")
    ARGS=$(build_args "$model" "solo" "")

    if CUDA_VISIBLE_DEVICES=0,1,2,3 \
       NCCL_P2P_DISABLE=0 NCCL_SHM_DISABLE=0 \
       torchrun --nproc_per_node=4 --master_port=29500 \
       ${SCRIPT} ${ARGS} \
       >> "${LOG_DIR}/${desc}.log" 2>&1; then
        elapsed=$(( $(date +%s) - start_time ))
        echo "[OK]  $desc (${elapsed}s)" | tee -a "$SUMMARY_LOG"
        PASS=$((PASS + 1))
    else
        elapsed=$(( $(date +%s) - start_time ))
        echo "[FAIL] $desc (${elapsed}s)" | tee -a "$SUMMARY_LOG"
        FAIL=$((FAIL + 1))
    fi
done

# ════════════════════════════════════════
# Phase 2: Pair experiments
#   Job A: 0,2,4,6 / Job B: 1,3,5,7 (interleaved)
#   NCCL P2P+SHM disabled → force PCIe host relay
# ════════════════════════════════════════
echo "" | tee -a "$SUMMARY_LOG"
echo "=== Phase 2: Pair Experiments (A=0,2,4,6 B=1,3,5,7) ===" | tee -a "$SUMMARY_LOG"

for ((i=0; i<${#MODELS[@]}; i++)); do
    for ((j=0; j<${#MODELS[@]}; j++)); do
        [[ $i -eq $j ]] && continue
        model_a="${MODELS[$i]}"   # primary (measured)
        model_b="${MODELS[$j]}"   # interferer
        desc="pair_${model_a}_${model_b}"

        echo "--------------------------------------------" | tee -a "$SUMMARY_LOG"
        echo "[RUN] $desc" | tee -a "$SUMMARY_LOG"
        start_time=$(date +%s)

        SCRIPT_A=$(get_script "$model_a")
        SCRIPT_B=$(get_script "$model_b")

        BARRIER_DIR="/tmp/a100_barrier_v2"
        rm -rf "$BARRIER_DIR"
        mkdir -p "$BARRIER_DIR"

        ARGS_A=$(build_args "$model_a" "pair" "$model_b" "primary")
        ARGS_B=$(build_args "$model_b" "pair" "$model_a" "interferer")

        CUDA_VISIBLE_DEVICES=0,2,4,6 \
        NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 \
        torchrun --nproc_per_node=4 --master_port=29500 \
        ${SCRIPT_A} ${ARGS_A} \
        >> "${LOG_DIR}/${desc}_A.log" 2>&1 &
        PID_A=$!

        CUDA_VISIBLE_DEVICES=1,3,5,7 \
        NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 \
        torchrun --nproc_per_node=4 --master_port=29501 \
        ${SCRIPT_B} ${ARGS_B} \
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
done

# ════════════════════════════════════════
# Summary
# ════════════════════════════════════════
echo "" | tee -a "$SUMMARY_LOG"
echo "======================================================" | tee -a "$SUMMARY_LOG"
echo " V2 Experiments complete!" | tee -a "$SUMMARY_LOG"
echo " Passed: ${PASS}" | tee -a "$SUMMARY_LOG"
echo " Failed: ${FAIL}" | tee -a "$SUMMARY_LOG"
echo " Results: ${RESULTS_DIR}/" | tee -a "$SUMMARY_LOG"
echo " Logs:    ${LOG_DIR}/" | tee -a "$SUMMARY_LOG"
echo "======================================================" | tee -a "$SUMMARY_LOG"
