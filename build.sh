#!/bin/bash

# Docker 이미지 빌드 스크립트

echo "Building mosim-llms Docker image..."

# Docker 이미지 빌드
docker build -t mosim-llms:latest .

if [ $? -eq 0 ]; then
    echo "✓ Docker image built successfully!"
    echo ""
    echo "To run the container:"
    echo "  GPU: ./run.sh gpu"
    echo "  CPU: ./run.sh cpu"
    echo ""
    echo "Or use docker-compose:"
    echo "  docker-compose up -d"
else
    echo "✗ Docker build failed!"
    exit 1
fi
