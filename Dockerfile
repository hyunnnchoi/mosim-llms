FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

# 환경 변수 설정
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# 시스템 패키지 업데이트 및 필수 도구 설치
RUN apt-get update && apt-get install -y \
    git \
    wget \
    vim \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 설정
WORKDIR /workspace

# Python 패키지 업그레이드
RUN pip install --upgrade pip setuptools wheel

# Chakra 및 의존성 설치
# 1. PARAM 설치 (Chakra 의존성 - et_replay 필요)
RUN cd /tmp && \
    git clone https://github.com/facebookresearch/param.git && \
    cd param/et_replay && \
    git checkout 7b19f586dd8b267333114992833a0d7e0d601630 && \
    pip install . && \
    cd / && rm -rf /tmp/param

# 2. HolisticTraceAnalysis 설치 (chakra_trace_link 필요)
RUN cd /tmp && \
    git clone https://github.com/facebookresearch/HolisticTraceAnalysis.git && \
    cd HolisticTraceAnalysis && \
    git checkout d731cc2e2249976c97129d409a83bd53d93051f6 && \
    git submodule update --init && \
    pip install -r requirements.txt && \
    pip install -e . && \
    cd / && rm -rf /tmp/HolisticTraceAnalysis

# 3. Chakra 설치 (GitHub에서 직접 설치)
RUN pip install protobuf>=3.19.0 && \
    pip install git+https://github.com/mlcommons/chakra.git

# Chakra 설치 확인
RUN echo "=== Chakra Tools Verification ===" && \
    chakra_trace_link --help > /dev/null 2>&1 && echo "✓ chakra_trace_link OK" || echo "✗ chakra_trace_link MISSING" && \
    chakra_converter --help > /dev/null 2>&1 && echo "✓ chakra_converter OK" || echo "✗ chakra_converter MISSING"

# PyTorch 관련 추가 패키지 설치
RUN pip install \
    transformers \
    datasets \
    tensorboard \
    tqdm \
    numpy \
    pandas

# 사전 학습된 토크나이저 다운로드 (Docker 이미지에 포함)
RUN mkdir -p /workspace/pretrained_models && \
    python -c "from transformers import AutoTokenizer; \
    tok1 = AutoTokenizer.from_pretrained('gpt2'); \
    tok1.save_pretrained('/workspace/pretrained_models/gpt2'); \
    tok2 = AutoTokenizer.from_pretrained('bert-base-uncased'); \
    tok2.save_pretrained('/workspace/pretrained_models/bert-base-uncased'); \
    print('✓ Tokenizers saved to /workspace/pretrained_models')"

# 프로젝트 파일 복사
COPY . /workspace/mosim-llms

# 작업 디렉토리를 프로젝트 루트로 변경
WORKDIR /workspace/mosim-llms

# requirements.txt가 있다면 추가 패키지 설치
RUN if [ -f requirements.txt ]; then pip install -r requirements.txt; fi

# 기본 명령어
CMD ["/bin/bash"]
