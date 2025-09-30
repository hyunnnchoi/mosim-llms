#!/bin/bash

# 모든 프로파일링 실험 자동 실행
# GPT-2 & BERT × {1, 2, 4} GPU × 20 steps

STEPS=20

echo "=========================================="
echo "Running All Profiling Experiments"
echo "Models: GPT-2, BERT"
echo "GPU configs: 1, 2, 4"
echo "Steps: $STEPS (skip evaluation)"
echo "Total: 6 experiments"
echo "=========================================="
echo ""

# 출력 디렉토리 생성
mkdir -p outputs checkpoints/gpt2 checkpoints/bert

# GPT-2 프로파일링
echo "[1/6] GPT-2 with 1 GPU ($STEPS steps)"
./run_gpt2_quick.sh 1 $STEPS
echo ""

echo "[2/6] GPT-2 with 2 GPUs ($STEPS steps)"
./run_gpt2_quick.sh 2 $STEPS
echo ""

echo "[3/6] GPT-2 with 4 GPUs ($STEPS steps)"
./run_gpt2_quick.sh 4 $STEPS
echo ""

# BERT 프로파일링
echo "[4/6] BERT with 1 GPU ($STEPS steps)"
./run_bert_quick.sh 1 $STEPS
echo ""

echo "[5/6] BERT with 2 GPUs ($STEPS steps)"
./run_bert_quick.sh 2 $STEPS
echo ""

echo "[6/6] BERT with 4 GPUs ($STEPS steps)"
./run_bert_quick.sh 4 $STEPS
echo ""

# 결과 요약
echo "=========================================="
echo "All profiling experiments completed!"
echo "=========================================="
echo ""
echo "Results:"
echo "  Kineto traces: ./outputs/*_kineto.json"
echo "  Chakra ET files: ./outputs/*.et"
echo "  Analysis: ./outputs/*_stacks.txt"
echo ""
echo "Generated traces:"
ls -lh outputs/*.json 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}'
echo ""
echo "Chakra ET files:"
ls -lh outputs/*.et 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}'
