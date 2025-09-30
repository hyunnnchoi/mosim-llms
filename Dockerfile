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

# Chakra 설치를 위한 의존성 설치
RUN pip install protobuf>=3.19.0

# Chakra 설치 (GitHub에서 직접 설치)
RUN pip install git+https://github.com/mlcommons/chakra.git

# PyTorch 관련 추가 패키지 설치
RUN pip install \
    transformers \
    datasets \
    tensorboard \
    tqdm \
    numpy \
    pandas

# 프로젝트 파일 복사
COPY . /workspace/mosim-llms

# 작업 디렉토리를 프로젝트 루트로 변경
WORKDIR /workspace/mosim-llms

# requirements.txt가 있다면 추가 패키지 설치
RUN if [ -f requirements.txt ]; then pip install -r requirements.txt; fi

# 기본 명령어
CMD ["/bin/bash"]
