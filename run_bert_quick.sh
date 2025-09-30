#!/bin/bash

# BERT 빠른 학습 (계산 그래프 캡처용)
# Usage: ./run_bert_quick.sh [num_gpus] [max_steps]

NUM_GPUS=${1:-1}
MAX_STEPS=${2:-500}  # 기본 500 steps
TRACE_NAME="bert_${NUM_GPUS}gpu_quick_trace"

echo "=========================================="
echo "BERT Quick Training (Graph Capture)"
echo "Number of GPUs: $NUM_GPUS"
echo "Max steps: $MAX_STEPS"
echo "=========================================="

if [ "$NUM_GPUS" -eq 1 ]; then
    # Single GPU
    python bert/train.py \
        --model-name bert-base-uncased \
        --batch-size 4 \
        --num-epochs 1 \
        --max-steps $MAX_STEPS \
        --num-gpus 1 \
        --enable-tracing \
        --trace-output-dir ./outputs \
        --trace-name $TRACE_NAME \
        --save-dir ./checkpoints/bert
else
    # Multi-GPU with torchrun
    torchrun \
        --nproc_per_node=$NUM_GPUS \
        --nnodes=1 \
        --node_rank=0 \
        --master_addr=localhost \
        --master_port=29501 \
        bert/train.py \
        --model-name bert-base-uncased \
        --batch-size 4 \
        --num-epochs 1 \
        --max-steps $MAX_STEPS \
        --num-gpus $NUM_GPUS \
        --enable-tracing \
        --trace-output-dir ./outputs \
        --trace-name $TRACE_NAME \
        --save-dir ./checkpoints/bert
fi

echo ""
echo "Training completed!"
echo "Max steps: $MAX_STEPS"
echo "Kineto trace saved to: ./outputs/${TRACE_NAME}_kineto.json"
echo "Chakra ET file saved to: ./outputs/${TRACE_NAME}.et ✓"
