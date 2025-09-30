#!/bin/bash

# 모든 프로파일링 실험 자동 실행
# GPT-2 & BERT × {1, 2, 4} GPU × 20 steps

STEPS=20

# 사용 가능한 GPU 수 확인
if command -v nvidia-smi &> /dev/null; then
    AVAILABLE_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
else
    AVAILABLE_GPUS=0
fi

echo "=========================================="
echo "Running All Profiling Experiments"
echo "Models: GPT-2, BERT"
echo "Available GPUs: $AVAILABLE_GPUS"
echo "Steps: $STEPS (skip evaluation)"
echo "=========================================="
echo ""

# GPU 설정 결정
if [ $AVAILABLE_GPUS -ge 4 ]; then
    GPU_CONFIGS=(1 2 4)
    TOTAL_EXP=6
elif [ $AVAILABLE_GPUS -ge 2 ]; then
    GPU_CONFIGS=(1 2)
    TOTAL_EXP=4
    echo "⚠️  Only 2 GPUs available, skipping 4 GPU experiments"
else
    GPU_CONFIGS=(1)
    TOTAL_EXP=2
    echo "⚠️  Only 1 GPU available, skipping multi-GPU experiments"
fi

echo "GPU configs to run: ${GPU_CONFIGS[@]}"
echo "Total experiments: $TOTAL_EXP"
echo ""

# 출력 디렉토리 생성
mkdir -p outputs checkpoints/gpt2 checkpoints/bert

# 실험 카운터
counter=1

# GPT-2 프로파일링
for ngpu in "${GPU_CONFIGS[@]}"; do
    echo "[$counter/$TOTAL_EXP] GPT-2 with $ngpu GPU(s) ($STEPS steps)"
    ./run_gpt2_quick.sh $ngpu $STEPS
    echo ""
    ((counter++))
done

# BERT 프로파일링
for ngpu in "${GPU_CONFIGS[@]}"; do
    echo "[$counter/$TOTAL_EXP] BERT with $ngpu GPU(s) ($STEPS steps)"
    ./run_bert_quick.sh $ngpu $STEPS
    echo ""
    ((counter++))
done

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
ls -lh outputs/*_kineto.json 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}'
echo ""
echo "Chakra ET files:"
ls -lh outputs/*.et 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}'
