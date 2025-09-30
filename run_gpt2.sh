#!/bin/bash

# GPT-2 학습 실행 스크립트
# Usage: ./run_gpt2.sh [num_gpus]

NUM_GPUS=${1:-1}
BATCH_SIZE=${2:-4}
TRACE_NAME="gpt2_${NUM_GPUS}gpu_trace"

echo "=========================================="
echo "GPT-2 Training on SQuAD"
echo "Number of GPUs: $NUM_GPUS"
echo "Batch size per GPU: $BATCH_SIZE"
echo "=========================================="

if [ "$NUM_GPUS" -eq 1 ]; then
    # Single GPU
    python gpt2/train.py \
        --model-name gpt2 \
        --batch-size $BATCH_SIZE \
        --num-epochs 3 \
        --num-gpus 1 \
        --enable-tracing \
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
        --batch-size $BATCH_SIZE \
        --num-epochs 3 \
        --num-gpus $NUM_GPUS \
        --enable-tracing \
        --trace-output-dir ./outputs \
        --trace-name $TRACE_NAME \
        --save-dir ./checkpoints/gpt2
fi

echo ""
echo "Training completed!"
echo "Checkpoints saved to: ./checkpoints/gpt2"
echo "Traces saved to: ./outputs/${TRACE_NAME}*"
