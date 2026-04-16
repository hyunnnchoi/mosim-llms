#!/usr/bin/env bash
# ────────────────────────────────────────────────────────────
# run_all_experiments.sh — Run all solo baselines + pair combos
# ────────────────────────────────────────────────────────────
# This generates all C(10,2)=45 pair experiments + 10 solo baselines.
#
# Usage:
#   bash A100/run_all_experiments.sh [total_steps]
#
# Estimated time: ~55 experiments x ~2-5 min each = 2-5 hours

set -euo pipefail
cd "$(dirname "$0")/.."

# Air-gapped environment: force offline mode for HuggingFace
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

TOTAL_STEPS="${1:-100}"
RESULTS_DIR="./A100/results"
LOG_DIR="./A100/logs"

mkdir -p "$RESULTS_DIR" "$LOG_DIR"

MODELS=(gpt2 bert resnet44 resnet110 resnet50 vgg16 googlenet inception3 densenet40_k12 densenet100_k12)

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SUMMARY_LOG="${LOG_DIR}/experiment_summary_${TIMESTAMP}.log"

echo "======================================================" | tee "$SUMMARY_LOG"
echo " A100 Interference Experiments" | tee -a "$SUMMARY_LOG"
echo " Models: ${MODELS[*]}" | tee -a "$SUMMARY_LOG"
echo " Steps per experiment: ${TOTAL_STEPS}" | tee -a "$SUMMARY_LOG"
echo " Solo baselines: ${#MODELS[@]}" | tee -a "$SUMMARY_LOG"
NUM_PAIRS=$(( ${#MODELS[@]} * (${#MODELS[@]} - 1) / 2 ))
echo " Pair combinations: ${NUM_PAIRS}" | tee -a "$SUMMARY_LOG"
echo " Total experiments: $(( ${#MODELS[@]} + NUM_PAIRS ))" | tee -a "$SUMMARY_LOG"
echo "======================================================" | tee -a "$SUMMARY_LOG"
echo "" | tee -a "$SUMMARY_LOG"

PASS=0
FAIL=0

run_experiment() {
    local desc="$1"
    shift
    echo "--------------------------------------------" | tee -a "$SUMMARY_LOG"
    echo "[RUN] $desc" | tee -a "$SUMMARY_LOG"
    local start_time=$(date +%s)

    if "$@" >> "${LOG_DIR}/${desc// /_}.log" 2>&1; then
        local elapsed=$(( $(date +%s) - start_time ))
        echo "[OK]  $desc (${elapsed}s)" | tee -a "$SUMMARY_LOG"
        PASS=$((PASS + 1))
    else
        local elapsed=$(( $(date +%s) - start_time ))
        echo "[FAIL] $desc (${elapsed}s)" | tee -a "$SUMMARY_LOG"
        FAIL=$((FAIL + 1))
    fi
}

# ════════════════════════════════════════
# Phase 1: Solo baselines (4 GPUs each)
# ════════════════════════════════════════
echo "" | tee -a "$SUMMARY_LOG"
echo "=== Phase 1: Solo Baselines ===" | tee -a "$SUMMARY_LOG"

for model in "${MODELS[@]}"; do
    run_experiment "solo_${model}" \
        bash A100/run_solo.sh "$model" "$TOTAL_STEPS"
done

# ════════════════════════════════════════
# Phase 2: All pair combinations (4+4 GPUs)
# ════════════════════════════════════════
echo "" | tee -a "$SUMMARY_LOG"
echo "=== Phase 2: Pair Experiments ===" | tee -a "$SUMMARY_LOG"

for ((i=0; i<${#MODELS[@]}; i++)); do
    for ((j=i+1; j<${#MODELS[@]}; j++)); do
        model_a="${MODELS[$i]}"
        model_b="${MODELS[$j]}"
        run_experiment "pair_${model_a}_${model_b}" \
            bash A100/run_pair.sh "$model_a" "$model_b" "$TOTAL_STEPS"
    done
done

# ════════════════════════════════════════
# Summary
# ════════════════════════════════════════
echo "" | tee -a "$SUMMARY_LOG"
echo "======================================================" | tee -a "$SUMMARY_LOG"
echo " Experiments complete!" | tee -a "$SUMMARY_LOG"
echo " Passed: ${PASS}" | tee -a "$SUMMARY_LOG"
echo " Failed: ${FAIL}" | tee -a "$SUMMARY_LOG"
echo " Results: ${RESULTS_DIR}/" | tee -a "$SUMMARY_LOG"
echo " Logs:    ${LOG_DIR}/" | tee -a "$SUMMARY_LOG"
echo "======================================================" | tee -a "$SUMMARY_LOG"
