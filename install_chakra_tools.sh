#!/bin/bash

# Chakra 및 관련 도구 설치 스크립트

set -e  # 에러 발생시 중단

# pip 명령어 확인 (pip3 또는 pip)
if command -v pip3 &> /dev/null; then
    PIP_CMD="pip3"
elif command -v pip &> /dev/null; then
    PIP_CMD="pip"
else
    echo "✗ Error: pip not found"
    exit 1
fi

echo "Using: $PIP_CMD"
echo ""

echo "=========================================="
echo "Chakra 도구 설치 시작"
echo "=========================================="
echo ""

# 임시 디렉토리 생성
TEMP_DIR=$(mktemp -d)
cd "$TEMP_DIR"

echo "📦 [1/3] PARAM 설치 중..."
echo "--------------------------------------"
git clone https://github.com/facebookresearch/param.git
cd param/et_replay
git checkout 7b19f586dd8b267333114992833a0d7e0d601630
$PIP_CMD install --user .
cd "$TEMP_DIR"
echo "✓ PARAM 설치 완료"
echo ""

echo "📦 [2/3] HolisticTraceAnalysis 설치 중..."
echo "--------------------------------------"
git clone https://github.com/facebookresearch/HolisticTraceAnalysis.git
cd HolisticTraceAnalysis
git checkout d731cc2e2249976c97129d409a83bd53d93051f6
git submodule update --init
$PIP_CMD install --user -r requirements.txt
$PIP_CMD install --user -e .
cd "$TEMP_DIR"
echo "✓ HolisticTraceAnalysis 설치 완료"
echo ""

echo "📦 [3/3] Chakra 설치 중..."
echo "--------------------------------------"
$PIP_CMD install --user git+https://github.com/mlcommons/chakra.git
echo "✓ Chakra 설치 완료"
echo ""

# 임시 디렉토리 정리
cd -
rm -rf "$TEMP_DIR"

echo "=========================================="
echo "✓ 모든 도구 설치 완료!"
echo "=========================================="
echo ""
echo "설치된 명령어:"
echo "  - chakra_trace_link"
echo "  - chakra_converter"
echo ""
echo "이제 다음 명령으로 변환할 수 있습니다:"
echo "  python3 convert_to_et.py"
echo ""

