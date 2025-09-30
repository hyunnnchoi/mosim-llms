#!/bin/bash

# GPT-2 빠른 학습 (계산 그래프 캡처용)
# Usage: ./run_gpt2_quick.sh [num_gpus] [max_steps]

NUM_GPUS=${1:-1}
MAX_STEPS=${2:-10}  # 기본 10 steps (그래프 캡처에 충분)
TRACE_NAME="gpt2_${NUM_GPUS}gpu_quick_trace"

echo "=========================================="
echo "GPT-2 Quick Training (Graph Capture)"
echo "Number of GPUs: $NUM_GPUS"
echo "Max steps: $MAX_STEPS"
echo "Skip evaluation: YES"
echo "=========================================="

if [ "$NUM_GPUS" -eq 1 ]; then
    # Single GPU
    python gpt2/train.py \
        --model-name gpt2 \
        --batch-size 4 \
        --num-epochs 1 \
        --max-steps $MAX_STEPS \
        --num-gpus 1 \
        --enable-tracing \
        --skip-eval \
        --trace-output-dir ./outputs \
        --trace-name $TRACE_NAME \
        --save-dir ./checkpoints/gpt2
else
    # Multi-GPU with torchrun
    torchrun \
        --nproc_per_node=$NUM_GPUS \
        --nnodes=1 \
        --node_rank=0 \
        --master_addr=localhost \
        --master_port=29500 \
        gpt2/train.py \
        --model-name gpt2 \
        --batch-size 4 \
        --num-epochs 1 \
        --max-steps $MAX_STEPS \
        --num-gpus $NUM_GPUS \
        --enable-tracing \
        --skip-eval \
        --trace-output-dir ./outputs \
        --trace-name $TRACE_NAME \
        --save-dir ./checkpoints/gpt2
fi

echo ""
echo "Training completed!"
echo "Max steps: $MAX_STEPS (graph captured in ~step 2)"
echo "Kineto trace saved to: ./outputs/${TRACE_NAME}_kineto.json"
echo "Chakra ET file saved to: ./outputs/${TRACE_NAME}.et ✓"
