#!/bin/bash

# setup_vm_environment.sh
# Dockerfile의 설치 과정을 클라우드 VM의 가상환경에서 수행하는 스크립트
# PyTorch 2.6.0 + CUDA 12.4 환경 기준

set -e  # 에러 발생 시 스크립트 중단

# 스크립트 디렉토리 저장 (cd 하기 전에 저장)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "VM Environment Setup for Chakra ET"
echo "=========================================="

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 함수: 색상 출력
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# 1. 시스템 패키지 확인 및 설치
print_status "Checking system packages..."

# 필수 패키지 목록
REQUIRED_PACKAGES="git wget vim build-essential"
MISSING_PACKAGES=""

for pkg in $REQUIRED_PACKAGES; do
    if ! dpkg -l | grep -q "^ii.*$pkg"; then
        MISSING_PACKAGES="$MISSING_PACKAGES $pkg"
    fi
done

if [ -n "$MISSING_PACKAGES" ]; then
    print_warning "Missing packages:$MISSING_PACKAGES"
    print_status "Installing missing packages (requires sudo)..."
    sudo apt-get update
    sudo apt-get install -y $MISSING_PACKAGES
else
    print_status "All required system packages are installed."
fi

# 2. Python 가상환경 생성
VENV_NAME="chakra-vm-venv"
print_status "Creating Python virtual environment: $VENV_NAME"

if [ -d "$VENV_NAME" ]; then
    print_warning "Virtual environment '$VENV_NAME' already exists."
    read -p "Do you want to remove and recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$VENV_NAME"
        python3 -m venv "$VENV_NAME"
    fi
else
    python3 -m venv "$VENV_NAME"
fi

# 가상환경 활성화
print_status "Activating virtual environment..."
source "$VENV_NAME/bin/activate"

# Python 버전 확인
PYTHON_VERSION=$(python --version)
print_status "Python version: $PYTHON_VERSION"

# 3. Python 패키지 업그레이드
print_status "Upgrading pip, setuptools, wheel..."
pip install --upgrade pip setuptools wheel

# 4. PyTorch 설치 (CUDA 12.4 버전)
print_status "Installing PyTorch 2.6.0 with CUDA 12.4..."
pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# PyTorch 설치 확인
print_status "Verifying PyTorch installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

# 5. PARAM 설치 (Chakra 의존성 - et_replay 필요)
print_status "Installing PARAM (et_replay)..."
cd /tmp
if [ -d "param" ]; then
    rm -rf param
fi
git clone https://github.com/facebookresearch/param.git
cd param/et_replay
git checkout 7b19f586dd8b267333114992833a0d7e0d601630
pip install .
cd /tmp
rm -rf param
cd "$SCRIPT_DIR"

# 6. HolisticTraceAnalysis 설치 (chakra_trace_link 필요)
print_status "Installing HolisticTraceAnalysis (chakra_trace_link)..."
cd /tmp
if [ -d "HolisticTraceAnalysis" ]; then
    rm -rf HolisticTraceAnalysis
fi
git clone https://github.com/facebookresearch/HolisticTraceAnalysis.git
cd HolisticTraceAnalysis
git checkout d731cc2e2249976c97129d409a83bd53d93051f6
git submodule update --init
pip install -r requirements.txt
pip install -e .
cd /tmp
rm -rf HolisticTraceAnalysis
cd "$SCRIPT_DIR"

# 7. Chakra 설치 (로컬 소스에서 설치)
CHAKRA_DIR="$SCRIPT_DIR/chakra"

if [ ! -d "$CHAKRA_DIR" ]; then
    print_error "Chakra directory not found: $CHAKRA_DIR"
    print_error "Please ensure the 'chakra' directory exists in the same location as this script."
    exit 1
fi

print_status "Installing Chakra from local source: $CHAKRA_DIR"
cd "$CHAKRA_DIR"
pip install protobuf>=3.19.0 grpcio-tools setuptools-grpc

print_status "Compiling protobuf files..."
python -m grpc_tools.protoc \
    --proto_path=schema/protobuf \
    --python_out=schema/protobuf \
    --grpc_python_out=schema/protobuf \
    schema/protobuf/et_def.proto

# setup.cfg 제거 (있는 경우)
if [ -f "setup.cfg" ]; then
    rm -f setup.cfg
fi

pip install -e .
cd "$SCRIPT_DIR"

# 8. Chakra 설치 확인
print_status "Verifying Chakra tools installation..."
echo "=== Chakra Tools Verification ==="

if chakra_trace_link --help > /dev/null 2>&1; then
    echo -e "${GREEN}✓${NC} chakra_trace_link OK"
else
    echo -e "${RED}✗${NC} chakra_trace_link MISSING"
fi

if chakra_converter --help > /dev/null 2>&1; then
    echo -e "${GREEN}✓${NC} chakra_converter OK"
else
    echo -e "${RED}✗${NC} chakra_converter MISSING"
fi

if python -c "from chakra.src.converter.pytorch_converter import PyTorchConverter" 2>/dev/null; then
    echo -e "${GREEN}✓${NC} chakra.src.converter.pytorch_converter import OK"
else
    echo -e "${RED}✗${NC} Python API import MISSING"
fi

# 9. PyTorch 관련 추가 패키지 설치
print_status "Installing PyTorch-related packages..."
pip install \
    transformers \
    datasets \
    tensorboard \
    tqdm \
    numpy \
    pandas

# 10. 프로젝트 requirements.txt 설치 (있는 경우)
cd "$SCRIPT_DIR"
if [ -f "requirements.txt" ]; then
    print_status "Installing packages from requirements.txt..."
    pip install -r requirements.txt
else
    print_warning "requirements.txt not found, skipping..."
fi

# 11. 사전 학습된 토크나이저 다운로드
PRETRAINED_DIR="$SCRIPT_DIR/pretrained_models"
print_status "Downloading pretrained tokenizers to: $PRETRAINED_DIR"

mkdir -p "$PRETRAINED_DIR"

python -c "
from transformers import AutoTokenizer

print('Downloading GPT-2 tokenizer...')
tok1 = AutoTokenizer.from_pretrained('gpt2')
tok1.save_pretrained('$PRETRAINED_DIR/gpt2')

print('Downloading BERT tokenizer...')
tok2 = AutoTokenizer.from_pretrained('bert-base-uncased')
tok2.save_pretrained('$PRETRAINED_DIR/bert-base-uncased')

print('✓ Tokenizers saved to $PRETRAINED_DIR')
"

# 완료 메시지
echo ""
echo "=========================================="
print_status "Setup completed successfully!"
echo "=========================================="
echo ""
echo "To activate the environment, run:"
echo "  source $VENV_NAME/bin/activate"
echo ""
echo "Installed tools:"
echo "  - PyTorch 2.6.0 (CUDA 12.4)"
echo "  - PARAM (et_replay)"
echo "  - HolisticTraceAnalysis (chakra_trace_link)"
echo "  - Chakra (chakra_converter)"
echo "  - Transformers, Datasets, TensorBoard, etc."
echo ""
echo "Pretrained models location:"
echo "  $PRETRAINED_DIR"
echo ""
