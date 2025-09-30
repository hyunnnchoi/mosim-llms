#!/bin/bash

# Docker 컨테이너 실행 스크립트

MODE=${1:-gpu}

# 필요한 디렉토리 생성
mkdir -p outputs data checkpoints

if [ "$MODE" = "gpu" ]; then
    echo "Running container with GPU support..."
    docker run -it --rm --gpus all \
        --shm-size=8g \
        -v $(pwd)/outputs:/workspace/mosim-llms/outputs \
        -v $(pwd)/data:/workspace/mosim-llms/data \
        -v $(pwd)/checkpoints:/workspace/mosim-llms/checkpoints \
        mosim-llms:latest
elif [ "$MODE" = "cpu" ]; then
    echo "Running container with CPU only..."
    docker run -it --rm \
        --shm-size=8g \
        -v $(pwd)/outputs:/workspace/mosim-llms/outputs \
        -v $(pwd)/data:/workspace/mosim-llms/data \
        -v $(pwd)/checkpoints:/workspace/mosim-llms/checkpoints \
        mosim-llms:latest
else
    echo "Usage: ./run.sh [gpu|cpu]"
    exit 1
fi
