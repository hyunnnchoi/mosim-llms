#!/bin/bash

# Kineto JSON traces를 Chakra ET 파일로 변환
# Usage: ./convert_to_et.sh

INPUT_DIR="./outputs"
OUTPUT_DIR="./outputs"

echo "=========================================="
echo "Converting Kineto traces to Chakra ET"
echo "Input: $INPUT_DIR/*_kineto.json"
echo "Output: $OUTPUT_DIR/*.et"
echo "=========================================="
echo ""

# Chakra 설치 확인
if ! command -v chakra_converter &> /dev/null; then
    echo "❌ chakra_converter command not found!"
    echo ""
    echo "Please install Chakra first:"
    echo "  pip install https://github.com/mlcommons/chakra/archive/refs/heads/main.zip"
    echo ""
    exit 1
fi

# 변환할 파일 찾기
KINETO_FILES=$(ls $INPUT_DIR/*_kineto.json 2>/dev/null)

if [ -z "$KINETO_FILES" ]; then
    echo "❌ No Kineto trace files found in $INPUT_DIR"
    exit 1
fi

# 파일 개수 세기
FILE_COUNT=$(echo "$KINETO_FILES" | wc -l | tr -d ' ')
echo "Found $FILE_COUNT Kineto trace files"
echo ""

# 각 파일 변환
counter=1
for kineto_file in $KINETO_FILES; do
    # 파일명에서 _kineto.json 제거하고 .et 추가
    base_name=$(basename "$kineto_file" "_kineto.json")
    et_file="$OUTPUT_DIR/${base_name}.et"
    
    echo "[$counter/$FILE_COUNT] Converting: $base_name"
    echo "  Input:  $(basename $kineto_file)"
    echo "  Output: $(basename $et_file)"
    
    # Chakra converter 실행 (CLI 사용)
    chakra_converter PyTorch \
        --input "$kineto_file" \
        --output "${et_file%.et}" \
        > /dev/null 2>&1
    
    if [ $? -eq 0 ]; then
        echo "  ✓ Conversion successful"
        # 파일 크기 확인
        et_size=$(ls -lh "$et_file" 2>/dev/null | awk '{print $5}')
        echo "  Size: $et_size"
    else
        echo "  ✗ Conversion failed"
    fi
    
    echo ""
    ((counter++))
done

# 결과 요약
echo "=========================================="
echo "Conversion completed!"
echo "=========================================="
echo ""
echo "Generated ET files:"
ls -lh $OUTPUT_DIR/*.et 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}'
echo ""
echo "Total ET files: $(ls $OUTPUT_DIR/*.et 2>/dev/null | wc -l | tr -d ' ')"
