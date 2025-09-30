#!/bin/bash

# 모든 실험 자동 실행 스크립트
# GPT-2 및 BERT를 1, 2, 8 GPU로 학습하고 Chakra trace 생성

echo "=========================================="
echo "Running All Experiments"
echo "GPT-2 and BERT on SQuAD"
echo "GPU configurations: 1, 2, 8"
echo "=========================================="
echo ""

# 출력 디렉토리 생성
mkdir -p outputs checkpoints/gpt2 checkpoints/bert

# GPT-2 실험
echo "[1/6] GPT-2 with 1 GPU"
./run_gpt2.sh 1 4

echo ""
echo "[2/6] GPT-2 with 2 GPUs"
./run_gpt2.sh 2 4

echo ""
echo "[3/6] GPT-2 with 8 GPUs"
./run_gpt2.sh 8 4

# BERT 실험
echo ""
echo "[4/6] BERT with 1 GPU"
./run_bert.sh 1 4

echo ""
echo "[5/6] BERT with 2 GPUs"
./run_bert.sh 2 4

echo ""
echo "[6/6] BERT with 8 GPUs"
./run_bert.sh 8 4

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "=========================================="
echo ""
echo "Results:"
echo "  Checkpoints: ./checkpoints/{gpt2,bert}"
echo "  Kineto Traces: ./outputs/*_kineto.json"
echo "  Chakra ET Files: ./outputs/*.et ✓"
echo "  Analysis: ./outputs/*_stacks.txt"
echo ""
echo "Chakra ET files are ready for simulation/analysis!"
echo "Use with ASTRA-sim, Timeloop, or other Chakra-compatible tools."
