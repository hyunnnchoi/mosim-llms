#!/bin/bash

# CNN 모델 빠른 학습 (계산 그래프 캡처용)
# Usage: ./run_cnn_quick.sh [model_name] [num_gpus] [epochs]
#
# Available models:
#   densenet40_k12, densenet100_k12, googlenet, inception3,
#   resnet110, resnet44, resnet50, vgg16

MODEL=${1:-resnet50}
NUM_GPUS=${2:-1}
EPOCHS=${3:-1}  # 기본 1 epoch (그래프 캡처에 충분)
TRACE_NAME="${MODEL}_${NUM_GPUS}gpu_trace"

# 모델별 기본 설정
case $MODEL in
    densenet40_k12|densenet100_k12|resnet44|resnet110)
        DATASET="cifar10"
        IMAGE_SIZE=32
        NUM_CLASSES=10
        BATCH_SIZE=128
        ;;
    googlenet)
        DATASET="cifar10"
        IMAGE_SIZE=32
        NUM_CLASSES=10
        BATCH_SIZE=128
        AUX_LOGITS="--aux-logits"
        ;;
    inception3)
        DATASET="cifar10"
        IMAGE_SIZE=299  # Inception v3는 299x299 입력
        NUM_CLASSES=10
        BATCH_SIZE=64
        AUX_LOGITS="--aux-logits"
        ;;
    resnet50)
        DATASET="cifar10"
        IMAGE_SIZE=224  # ResNet-50은 ImageNet 크기
        NUM_CLASSES=10
        BATCH_SIZE=128
        ;;
    vgg16)
        DATASET="cifar10"
        IMAGE_SIZE=224
        NUM_CLASSES=10
        BATCH_SIZE=64
        ;;
    *)
        echo "Unknown model: $MODEL"
        echo "Available models: densenet40_k12, densenet100_k12, googlenet, inception3,"
        echo "                  resnet110, resnet44, resnet50, vgg16"
        exit 1
        ;;
esac

echo "=========================================="
echo "CNN Model Quick Training (Graph Capture)"
echo "Model: $MODEL"
echo "Dataset: $DATASET"
echo "Number of GPUs: $NUM_GPUS"
echo "Epochs: $EPOCHS"
echo "Batch size per GPU: $BATCH_SIZE"
echo "Image size: ${IMAGE_SIZE}x${IMAGE_SIZE}"
echo "Output: ./outputs/${TRACE_NAME}/"
echo "=========================================="

# 출력 디렉토리 생성
mkdir -p outputs

if [ "$NUM_GPUS" -eq 1 ]; then
    # Single GPU
    python torch_cnn_benchmarks/benchmark.py \
        --model $MODEL \
        --dataset $DATASET \
        --synthetic \
        --num-classes $NUM_CLASSES \
        --image-size $IMAGE_SIZE \
        --batch-size $BATCH_SIZE \
        --epochs $EPOCHS \
        --no-eval \
        --enable-tracing \
        --trace-name $TRACE_NAME \
        --trace-output-dir ./outputs \
        --trace-wait-steps 0 \
        --trace-warmup-steps 0 \
        --trace-active-steps 1 \
        --log-interval 10 \
        ${AUX_LOGITS:-}
else
    # Multi-GPU with torchrun
    torchrun \
        --nproc_per_node=$NUM_GPUS \
        --nnodes=1 \
        --node_rank=0 \
        --master_addr=localhost \
        --master_port=29500 \
        torch_cnn_benchmarks/benchmark.py \
        --model $MODEL \
        --dataset $DATASET \
        --synthetic \
        --num-classes $NUM_CLASSES \
        --image-size $IMAGE_SIZE \
        --batch-size $BATCH_SIZE \
        --epochs $EPOCHS \
        --no-eval \
        --enable-tracing \
        --trace-name $TRACE_NAME \
        --trace-output-dir ./outputs \
        --trace-wait-steps 0 \
        --trace-warmup-steps 0 \
        --trace-active-steps 1 \
        --log-interval 10 \
        ${AUX_LOGITS:-}
fi

echo ""
echo "✓ Training completed!"
echo ""
echo "Results in: ./outputs/${TRACE_NAME}/"
echo "  - Raw traces: host_*.json, device_*.json"
echo "  - Merged: merged_*.json"
echo "  - Final Chakra ET: ${TRACE_NAME}.*.et"
if [ "$NUM_GPUS" -eq 1 ]; then
    echo "  - Analysis: analysis/stacks.txt"
fi
echo ""
echo "To view in Chrome: chrome://tracing"
echo "  Load: ./outputs/${TRACE_NAME}/device_0.json"
