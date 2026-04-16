#!/usr/bin/env bash
# Re-run only the 19 failed experiments (googlenet & inception3 related)
set -euo pipefail
cd "$(dirname "$0")/.."

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

TOTAL_STEPS="${1:-100}"
RESULTS_DIR="./A100/results"
LOG_DIR="./A100/logs"
PASS=0
FAIL=0

run_experiment() {
    local desc="$1"
    shift
    echo "--------------------------------------------"
    echo "[RUN] $desc"
    local start_time=$(date +%s)
    if "$@" >> "${LOG_DIR}/${desc// /_}_retry.log" 2>&1; then
        local elapsed=$(( $(date +%s) - start_time ))
        echo "[OK]  $desc (${elapsed}s)"
        PASS=$((PASS + 1))
    else
        local elapsed=$(( $(date +%s) - start_time ))
        echo "[FAIL] $desc (${elapsed}s)"
        FAIL=$((FAIL + 1))
    fi
}

echo "=== Re-running failed experiments (googlenet/inception3) ==="

# Solo baselines
run_experiment "solo_googlenet" bash A100/run_solo.sh googlenet "$TOTAL_STEPS"
run_experiment "solo_inception3" bash A100/run_solo.sh inception3 "$TOTAL_STEPS"

# All pairs involving googlenet or inception3
MODELS=(gpt2 bert resnet44 resnet110 resnet50 vgg16 densenet40_k12 densenet100_k12)
for m in "${MODELS[@]}"; do
    run_experiment "pair_${m}_googlenet" bash A100/run_pair.sh "$m" googlenet "$TOTAL_STEPS"
    run_experiment "pair_${m}_inception3" bash A100/run_pair.sh "$m" inception3 "$TOTAL_STEPS"
done
run_experiment "pair_googlenet_inception3" bash A100/run_pair.sh googlenet inception3 "$TOTAL_STEPS"

echo ""
echo "=============================="
echo " Retry complete: Passed=${PASS}, Failed=${FAIL}"
echo "=============================="
