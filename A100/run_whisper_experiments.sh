#!/usr/bin/env bash
# ────────────────────────────────────────────────────────────
# run_whisper_experiments.sh — Whisper-only: solo + all pairs
# ────────────────────────────────────────────────────────────
# 22 experiments total:
#   1  whisper solo baseline
#  10  whisper as primary   (whisper_with_X)
#  10  whisper as interferer (X_with_whisper)
#   1  whisper self-pair    (whisper_with_whisper)
#
# Usage:
#   bash A100/run_whisper_experiments.sh [total_steps]

set -eo pipefail
cd "$(dirname "$0")/.."

export HF_HOME=/home/work/hyunmokchoi/hf_cache
export HF_DATASETS_CACHE=/home/work/hyunmokchoi/hf_cache
export TRANSFORMERS_CACHE=/home/work/hyunmokchoi/hf_cache
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

TOTAL_STEPS="${1:-100}"
RESULTS_DIR="./A100/results_v2"
LOG_DIR="./A100/logs_v2"
BARRIER_BASE="/tmp/a100_barrier_whisper"

mkdir -p "$RESULTS_DIR" "$LOG_DIR"

OTHERS=(gpt2 bert resnet44 resnet110 resnet50 vgg16 googlenet inception3 densenet40_k12 densenet100_k12)

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SUMMARY_LOG="${LOG_DIR}/whisper_experiments_${TIMESTAMP}.log"

echo "======================================================" | tee "$SUMMARY_LOG"
echo " A100 Whisper Interference Experiments"                  | tee -a "$SUMMARY_LOG"
echo " Solo:  GPUs 0,1,2,3 (NUMA 0, P2P+SHM ON)"             | tee -a "$SUMMARY_LOG"
echo " Pair:  A=0,2,4,6 B=1,3,5,7 (interleaved, P2P+SHM OFF)" | tee -a "$SUMMARY_LOG"
echo " Steps: ${TOTAL_STEPS}"                                  | tee -a "$SUMMARY_LOG"
echo " Experiments: 22 (1 solo + 10 primary + 10 interferer + 1 self)" | tee -a "$SUMMARY_LOG"
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

# ════════════════════════════════════════
# Phase 1: Whisper solo baseline  (#1)
# ════════════════════════════════════════
echo "=== Phase 1: Whisper Solo Baseline (GPUs 0,1,2,3) ===" | tee -a "$SUMMARY_LOG"
echo "--------------------------------------------" | tee -a "$SUMMARY_LOG"
echo "[RUN] solo_whisper" | tee -a "$SUMMARY_LOG"
start_time=$(date +%s)

if CUDA_VISIBLE_DEVICES=0,1,2,3 \
   NCCL_P2P_DISABLE=0 NCCL_SHM_DISABLE=0 \
   torchrun --nproc_per_node=4 --master_port=29500 \
   A100/train_whisper.py \
       --mode solo --total-steps ${TOTAL_STEPS} --output-dir ${RESULTS_DIR} --role primary \
   >> "${LOG_DIR}/solo_whisper.log" 2>&1; then
    elapsed=$(( $(date +%s) - start_time ))
    echo "[OK]  solo_whisper (${elapsed}s)" | tee -a "$SUMMARY_LOG"
    PASS=$((PASS + 1))
else
    elapsed=$(( $(date +%s) - start_time ))
    echo "[FAIL] solo_whisper (${elapsed}s)" | tee -a "$SUMMARY_LOG"
    FAIL=$((FAIL + 1))
fi

# ════════════════════════════════════════
# Phase 2: Whisper as primary (#2-#11)
# ════════════════════════════════════════
echo "" | tee -a "$SUMMARY_LOG"
echo "=== Phase 2: Whisper as Primary (whisper_with_X) ===" | tee -a "$SUMMARY_LOG"

for other in "${OTHERS[@]}"; do
    desc="pair_whisper_${other}"
    echo "--------------------------------------------" | tee -a "$SUMMARY_LOG"
    echo "[RUN] $desc" | tee -a "$SUMMARY_LOG"
    start_time=$(date +%s)

    SCRIPT_B=$(get_script "$other")
    BDIR="${BARRIER_BASE}_${desc}"
    rm -rf "$BDIR" && mkdir -p "$BDIR"

    MODEL_FLAG=""
    if [[ "$other" != "bert" && "$other" != "gpt2" && "$other" != "whisper" ]]; then
        MODEL_FLAG="--model ${other}"
    fi

    # Primary: whisper (measured)
    CUDA_VISIBLE_DEVICES=0,2,4,6 \
    NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 \
    torchrun --nproc_per_node=4 --master_port=29500 \
    A100/train_whisper.py \
        --mode pair --partner "${other}" \
        --total-steps ${TOTAL_STEPS} --output-dir ${RESULTS_DIR} \
        --role primary \
        --job-id whisper_primary --partner-id "${other}_interferer" \
        --barrier-dir "$BDIR" \
    >> "${LOG_DIR}/${desc}_A.log" 2>&1 &
    PID_A=$!

    # Interferer: other model
    CUDA_VISIBLE_DEVICES=1,3,5,7 \
    NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 \
    torchrun --nproc_per_node=4 --master_port=29501 \
    ${SCRIPT_B} ${MODEL_FLAG} \
        --mode pair --partner whisper \
        --total-steps ${TOTAL_STEPS} --output-dir ${RESULTS_DIR} \
        --role interferer \
        --job-id "${other}_interferer" --partner-id whisper_primary \
        --barrier-dir "$BDIR" \
    >> "${LOG_DIR}/${desc}_B.log" 2>&1 &
    PID_B=$!

    pair_fail=0
    wait $PID_A || pair_fail=1
    wait $PID_B || pair_fail=1
    rm -rf "$BDIR"

    elapsed=$(( $(date +%s) - start_time ))
    if [ $pair_fail -eq 0 ]; then
        echo "[OK]  $desc (${elapsed}s)" | tee -a "$SUMMARY_LOG"
        PASS=$((PASS + 1))
    else
        echo "[FAIL] $desc (${elapsed}s)" | tee -a "$SUMMARY_LOG"
        FAIL=$((FAIL + 1))
    fi
done

# ════════════════════════════════════════
# Phase 3: Whisper as interferer (#12-#21)
# ════════════════════════════════════════
echo "" | tee -a "$SUMMARY_LOG"
echo "=== Phase 3: Whisper as Interferer (X_with_whisper) ===" | tee -a "$SUMMARY_LOG"

for other in "${OTHERS[@]}"; do
    desc="pair_${other}_whisper"
    echo "--------------------------------------------" | tee -a "$SUMMARY_LOG"
    echo "[RUN] $desc" | tee -a "$SUMMARY_LOG"
    start_time=$(date +%s)

    SCRIPT_A=$(get_script "$other")
    BDIR="${BARRIER_BASE}_${desc}"
    rm -rf "$BDIR" && mkdir -p "$BDIR"

    MODEL_FLAG=""
    if [[ "$other" != "bert" && "$other" != "gpt2" && "$other" != "whisper" ]]; then
        MODEL_FLAG="--model ${other}"
    fi

    # Primary: other model (measured)
    CUDA_VISIBLE_DEVICES=0,2,4,6 \
    NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 \
    torchrun --nproc_per_node=4 --master_port=29500 \
    ${SCRIPT_A} ${MODEL_FLAG} \
        --mode pair --partner whisper \
        --total-steps ${TOTAL_STEPS} --output-dir ${RESULTS_DIR} \
        --role primary \
        --job-id "${other}_primary" --partner-id whisper_interferer \
        --barrier-dir "$BDIR" \
    >> "${LOG_DIR}/${desc}_A.log" 2>&1 &
    PID_A=$!

    # Interferer: whisper
    CUDA_VISIBLE_DEVICES=1,3,5,7 \
    NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 \
    torchrun --nproc_per_node=4 --master_port=29501 \
    A100/train_whisper.py \
        --mode pair --partner "${other}" \
        --total-steps ${TOTAL_STEPS} --output-dir ${RESULTS_DIR} \
        --role interferer \
        --job-id whisper_interferer --partner-id "${other}_primary" \
        --barrier-dir "$BDIR" \
    >> "${LOG_DIR}/${desc}_B.log" 2>&1 &
    PID_B=$!

    pair_fail=0
    wait $PID_A || pair_fail=1
    wait $PID_B || pair_fail=1
    rm -rf "$BDIR"

    elapsed=$(( $(date +%s) - start_time ))
    if [ $pair_fail -eq 0 ]; then
        echo "[OK]  $desc (${elapsed}s)" | tee -a "$SUMMARY_LOG"
        PASS=$((PASS + 1))
    else
        echo "[FAIL] $desc (${elapsed}s)" | tee -a "$SUMMARY_LOG"
        FAIL=$((FAIL + 1))
    fi
done

# ════════════════════════════════════════
# Phase 4: Whisper self-pair (#22)
# ════════════════════════════════════════
echo "" | tee -a "$SUMMARY_LOG"
echo "=== Phase 4: Whisper Self-Pair ===" | tee -a "$SUMMARY_LOG"

desc="pair_whisper_whisper"
echo "--------------------------------------------" | tee -a "$SUMMARY_LOG"
echo "[RUN] $desc" | tee -a "$SUMMARY_LOG"
start_time=$(date +%s)

BDIR="${BARRIER_BASE}_${desc}"
rm -rf "$BDIR" && mkdir -p "$BDIR"

CUDA_VISIBLE_DEVICES=0,2,4,6 \
NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 \
torchrun --nproc_per_node=4 --master_port=29500 \
A100/train_whisper.py \
    --mode pair --partner whisper \
    --total-steps ${TOTAL_STEPS} --output-dir ${RESULTS_DIR} \
    --role primary \
    --job-id whisper_primary --partner-id whisper_interferer \
    --barrier-dir "$BDIR" \
>> "${LOG_DIR}/${desc}_A.log" 2>&1 &
PID_A=$!

CUDA_VISIBLE_DEVICES=1,3,5,7 \
NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 \
torchrun --nproc_per_node=4 --master_port=29501 \
A100/train_whisper.py \
    --mode pair --partner whisper \
    --total-steps ${TOTAL_STEPS} --output-dir ${RESULTS_DIR} \
    --role interferer \
    --job-id whisper_interferer --partner-id whisper_primary \
    --barrier-dir "$BDIR" \
>> "${LOG_DIR}/${desc}_B.log" 2>&1 &
PID_B=$!

pair_fail=0
wait $PID_A || pair_fail=1
wait $PID_B || pair_fail=1
rm -rf "$BDIR"

elapsed=$(( $(date +%s) - start_time ))
if [ $pair_fail -eq 0 ]; then
    echo "[OK]  $desc (${elapsed}s)" | tee -a "$SUMMARY_LOG"
    PASS=$((PASS + 1))
else
    echo "[FAIL] $desc (${elapsed}s)" | tee -a "$SUMMARY_LOG"
    FAIL=$((FAIL + 1))
fi

# ════════════════════════════════════════
# Summary
# ════════════════════════════════════════
echo "" | tee -a "$SUMMARY_LOG"
echo "======================================================" | tee -a "$SUMMARY_LOG"
echo " Whisper experiments complete!" | tee -a "$SUMMARY_LOG"
echo " Passed: ${PASS} / 22" | tee -a "$SUMMARY_LOG"
echo " Failed: ${FAIL}" | tee -a "$SUMMARY_LOG"
echo " Results: ${RESULTS_DIR}/" | tee -a "$SUMMARY_LOG"
echo " Logs:    ${LOG_DIR}/" | tee -a "$SUMMARY_LOG"
echo "======================================================" | tee -a "$SUMMARY_LOG"
