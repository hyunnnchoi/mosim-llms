#!/bin/bash

# Chakra 도구 설치 및 변환 자동화 스크립트 (venv 사용)

set -e  # 에러 발생시 중단

echo "=========================================="
echo "Chakra ET 변환 자동화"
echo "=========================================="
echo ""

VENV_DIR="chakra-venv"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# 1. 가상 환경 생성 또는 확인
if [ ! -d "$VENV_DIR" ]; then
    echo "📦 [1/4] 가상 환경 생성 중..."
    python3 -m venv "$VENV_DIR"
    echo "✓ 가상 환경 생성 완료"
    echo ""
else
    echo "✓ [1/4] 가상 환경이 이미 존재합니다"
    echo ""
fi

# 2. 가상 환경 활성화
echo "🔌 [2/4] 가상 환경 활성화 중..."
source "$VENV_DIR/bin/activate"
echo "✓ 가상 환경 활성화 완료"
echo ""

# 3. Chakra 도구 설치 확인
if ! command -v chakra_trace_link &> /dev/null; then
    echo "📦 [3/4] Chakra 도구 설치 중..."
    echo "--------------------------------------"
    
    # PARAM 설치
    echo "  → PARAM 설치 중..."
    TEMP_DIR=$(mktemp -d)
    git clone https://github.com/facebookresearch/param.git "$TEMP_DIR/param" > /dev/null 2>&1
    cd "$TEMP_DIR/param/et_replay"
    git checkout 7b19f586dd8b267333114992833a0d7e0d601630 > /dev/null 2>&1
    pip install . > /dev/null 2>&1
    echo "    ✓ PARAM 설치 완료"
    
    # HolisticTraceAnalysis 설치
    echo "  → HolisticTraceAnalysis 설치 중..."
    git clone https://github.com/facebookresearch/HolisticTraceAnalysis.git "$TEMP_DIR/HolisticTraceAnalysis" > /dev/null 2>&1
    cd "$TEMP_DIR/HolisticTraceAnalysis"
    git checkout d731cc2e2249976c97129d409a83bd53d93051f6 > /dev/null 2>&1
    git submodule update --init > /dev/null 2>&1
    pip install -r requirements.txt > /dev/null 2>&1
    pip install -e . > /dev/null 2>&1
    echo "    ✓ HolisticTraceAnalysis 설치 완료"
    
    # Chakra 설치
    echo "  → Chakra 설치 중..."
    pip install git+https://github.com/mlcommons/chakra.git > /dev/null 2>&1
    echo "    ✓ Chakra 설치 완료"
    
    # 정리
    cd "$SCRIPT_DIR"
    rm -rf "$TEMP_DIR"
    
    echo "--------------------------------------"
    echo "✓ 모든 도구 설치 완료"
    echo ""
else
    echo "✓ [3/4] Chakra 도구가 이미 설치되어 있습니다"
    echo ""
fi

# 4. 변환 실행
echo "🔄 [4/4] Trace 파일 변환 중..."
echo "=========================================="
python "$SCRIPT_DIR/convert_to_et.py"

echo ""
echo "=========================================="
echo "✓ 모든 작업 완료!"
echo "=========================================="
echo ""
echo "가상 환경을 비활성화하려면:"
echo "  deactivate"
echo ""
echo "다음에 다시 변환하려면:"
echo "  source $VENV_DIR/bin/activate"
echo "  python convert_to_et.py"
echo "  deactivate"
echo ""

