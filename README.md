# MOSIM LLMs - Pretrain with Chakra Tracing

PyTorch 2.4.0 기반 GPT-2 및 BERT Pretraining 프로젝트 with Chakra Execution Trace 캡처

**SQuAD 데이터셋을 사용한 Data Parallel (DDP) 학습 및 Chakra trace 캡처**

## 프로젝트 구조

```
mosim-llms/
├── Dockerfile                  # Docker 이미지 빌드 파일
├── docker-compose.yml          # Docker Compose 설정
├── build.sh                    # 이미지 빌드 스크립트
├── run.sh                      # 컨테이너 실행 스크립트
├── requirements.txt            # Python 의존성
│
├── gpt2/                       # GPT-2 pretrain 구현
│   ├── __init__.py
│   ├── train.py               # GPT-2 학습 스크립트 (DDP 지원)
│   ├── model.py               # GPT-2 모델 정의
│   └── config.py              # GPT-2 설정
│
├── bert/                       # BERT pretrain 구현
│   ├── __init__.py
│   ├── train.py               # BERT 학습 스크립트 (DDP 지원)
│   ├── model.py               # BERT 모델 정의
│   └── config.py              # BERT 설정
│
├── utils/                      # 공통 유틸리티
│   ├── __init__.py
│   ├── distributed.py         # DDP 설정 유틸
│   ├── data_utils.py          # SQuAD 데이터 로딩
│   └── chakra_tracer.py       # Chakra trace 캡처
│
├── run_gpt2_quick.sh           # GPT-2 프로파일링 (1/2/4 GPU)
├── run_bert_quick.sh           # BERT 프로파일링 (1/2/4 GPU)
├── run_all_profiling.sh        # 전체 프로파일링 자동 실행
│
├── Chakra ET 변환 도구 (로컬용)
├── setup_and_convert.sh        # 자동 설치 + 변환
├── fix_host_traces.py          # Host trace JSON 수정
├── convert_to_et.py            # Chakra ET 변환
├── convert_remaining.py        # 실패한 파일 재변환
├── install_chakra_tools.sh     # Chakra 도구 수동 설치
│
├── 문서
├── README.md                   # 프로젝트 가이드
├── USAGE_GUIDE.md              # 상세 사용법
├── PROJECT_SUMMARY.md          # 프로젝트 요약
├── CHAKRA_SETUP.md             # Chakra 설정 가이드
└── CONVERSION_SUMMARY.md       # 변환 결과 요약
```

## 🚀 빠른 시작

### A. 서버에서 프로파일링 실행

서버(GPU 환경)에서 PyTorch 프로파일링을 수행하여 trace 파일을 생성합니다.

#### 1. 코드 업데이트 및 Docker 이미지 빌드

```bash
# 최신 코드 가져오기
sudo git pull

# Docker 이미지 빌드
sudo ./build.sh
```

#### 2. Docker 컨테이너 실행

```bash
# GPU 컨테이너 실행
./run.sh gpu

# 또는 docker-compose 사용
docker-compose up -d
docker-compose exec mosim-llms bash
```

#### 3. 프로파일링 실행 (컨테이너 내부)

```bash
# 모든 프로파일링 자동 실행 (GPT-2, BERT × 1/2/4 GPU)
./run_all_profiling.sh

# 또는 개별 실행
./run_gpt2_quick.sh    # GPT-2: 1/2/4 GPU
./run_bert_quick.sh    # BERT: 1/2/4 GPU
```

**생성되는 파일** (outputs 폴더):
```
outputs/
├── bert_1gpu_quick_trace_host.json      # Host trace
├── bert_1gpu_quick_trace_device.json    # Device trace
├── bert_1gpu_quick_trace_stacks.txt     # 분석용
├── bert_2gpu_quick_trace_*.json
├── bert_4gpu_quick_trace_*.json
├── gpt2_1gpu_quick_trace_*.json
├── gpt2_2gpu_quick_trace_*.json
└── gpt2_4gpu_quick_trace_*.json
```

### B. 로컬(맥북)에서 Chakra ET 변환

서버에서 생성된 trace 파일을 다운로드하여 Chakra ET 형식으로 변환합니다.

#### 1. 서버에서 trace 파일 다운로드

```bash
# 로컬 맥북에서 실행
scp -r your-server:/path/to/mosim-llms/outputs ./
```

#### 2. Chakra 도구 자동 설치 및 변환

```bash
# 한 번에 설치 + 변환 (권장)
./setup_and_convert.sh
```

이 스크립트는 자동으로:
- ✅ Python 가상 환경 생성 (`chakra-venv/`)
- ✅ Chakra 도구 설치 (PARAM, HolisticTraceAnalysis, Chakra)
- ✅ Host trace JSON 수정 (중복 객체 제거)
- ✅ Host + Device trace 병합
- ✅ Chakra ET 파일 생성

#### 3. 변환 결과 확인

```bash
# ET 파일 확인
ls -lh outputs/*.et

# 예상 출력:
# bert_1gpu_quick_trace.et      (10 MB)
# bert_2gpu_quick_trace.et      (12 MB)
# bert_4gpu_quick_trace.et      (12 MB)
# gpt2_1gpu_quick_trace.et      (8.6 MB)
# gpt2_2gpu_quick_trace.et      (9.6 MB)
# gpt2_4gpu_quick_trace.et      (9.6 MB)
```

#### 📝 수동 변환 (문제 발생 시)

자동 변환이 실패한 경우:

```bash
# 1. 가상 환경 활성화
source chakra-venv/bin/activate

# 2. Host trace 수정 (중복 JSON 객체 제거)
python fix_host_traces.py

# 3. ET 변환
python convert_to_et.py

# 4. 가상 환경 비활성화
deactivate
```

**변환 과정:**
1. `fix_host_traces.py` - 손상된 host trace JSON 수정
2. `chakra_trace_link` - host + device trace 병합
3. `chakra_converter` - Chakra ET 형식으로 변환

**문제 해결:**
- 자세한 가이드: [CHAKRA_SETUP.md](CHAKRA_SETUP.md)
- 변환 결과: [CONVERSION_SUMMARY.md](CONVERSION_SUMMARY.md)

## 상세 사용법

### GPT-2 학습

```bash
# 단일 GPU
python gpt2/train.py \
    --model-name gpt2 \
    --batch-size 4 \
    --num-epochs 3 \
    --num-gpus 1 \
    --enable-tracing \
    --trace-output-dir ./outputs \
    --trace-name gpt2_1gpu

# 멀티 GPU (DDP)
torchrun --nproc_per_node=8 gpt2/train.py \
    --model-name gpt2 \
    --batch-size 4 \
    --num-epochs 3 \
    --num-gpus 8 \
    --enable-tracing \
    --trace-output-dir ./outputs \
    --trace-name gpt2_8gpu
```

### BERT 학습

```bash
# 단일 GPU
python bert/train.py \
    --model-name bert-base-uncased \
    --batch-size 4 \
    --num-epochs 3 \
    --num-gpus 1 \
    --enable-tracing

# 멀티 GPU (DDP)
torchrun --nproc_per_node=8 bert/train.py \
    --model-name bert-base-uncased \
    --batch-size 4 \
    --num-epochs 3 \
    --num-gpus 8 \
    --enable-tracing
```

## Data Parallelism 구현

- **방식**: PyTorch DistributedDataParallel (DDP)
- **FSDP 미사용**: TensorFlow의 `MultiWorkerMirroredStrategy`와 유사한 전통적인 DP 방식
- **지원 GPU 수**: 1, 2, 8
- **Batch Size**: GPU당 4 (조정 가능)

### DDP 특징
- 각 GPU가 모델 전체 복사본을 가짐
- Gradient 동기화를 통한 학습
- `torch.nn.parallel.DistributedDataParallel` 사용
- `torchrun`을 통한 멀티프로세스 실행

## 📊 Chakra Execution Trace

### Trace 워크플로우

**서버 → 로컬 2단계 프로세스:**

#### 1단계: 서버에서 PyTorch Profiling
```
PyTorch Profiler
├── ExecutionTraceObserver → *_host.json    (CPU operations)
└── Kineto Profiler → *_device.json         (GPU operations)
```

#### 2단계: 로컬에서 Chakra ET 변환
```
Local Conversion
├── fix_host_traces.py → host.json 수정
├── chakra_trace_link → host + device 병합
└── chakra_converter → .et 파일 생성
```

### 생성되는 파일

#### 서버 (PyTorch Profiling)
```
outputs/
├── *_host.json        # ExecutionTraceObserver 출력
├── *_device.json      # Kineto Profiler 출력
└── *_stacks.txt       # Stack trace 분석
```

#### 로컬 (Chakra 변환 후)
```
outputs/
├── *_host.json        # 원본 (수정됨)
├── *_device.json      # 원본
├── *_merged.json      # 병합된 trace (중간 파일)
└── *.et              # Chakra ET (최종 출력) ✓
```

### 최종 ET 파일

| 모델 | GPU | 파일명 | 크기 |
|------|-----|--------|------|
| BERT | 1 | `bert_1gpu_quick_trace.et` | 10 MB |
| BERT | 2 | `bert_2gpu_quick_trace.et` | 12 MB |
| BERT | 4 | `bert_4gpu_quick_trace.et` | 12 MB |
| GPT-2 | 1 | `gpt2_1gpu_quick_trace.et` | 8.6 MB |
| GPT-2 | 2 | `gpt2_2gpu_quick_trace.et` | 9.6 MB |
| GPT-2 | 4 | `gpt2_4gpu_quick_trace.et` | 9.6 MB |

**캡처 내용:**
- ✅ Compute operations (forward, backward)
- ✅ Memory operations (allocation, transfer)
- ✅ Communication operations (all-reduce, broadcast for DDP)
- ✅ Dependency graph
- ✅ Timing information

### Chakra 도구 설치

#### 서버 (Docker)
Docker 이미지에 **이미 포함**되어 있습니다:
- ✅ PARAM (et_replay)
- ✅ HolisticTraceAnalysis (chakra_trace_link)
- ✅ Chakra (chakra_converter)

별도 설치 불필요!

#### 로컬 (맥북)
`setup_and_convert.sh` 스크립트가 **자동 설치**합니다:

```bash
./setup_and_convert.sh
# 자동으로 설치:
# - Python 가상 환경 (chakra-venv/)
# - PARAM, HolisticTraceAnalysis, Chakra
```

**수동 설치**가 필요한 경우:
```bash
./install_chakra_tools.sh
# 또는
source chakra-venv/bin/activate
pip install git+https://github.com/mlcommons/chakra.git
```

자세한 설치 가이드: [CHAKRA_SETUP.md](CHAKRA_SETUP.md)

## 데이터셋

### SQuAD (Stanford Question Answering Dataset)

- **GPT-2**: Causal Language Modeling
  - Question + Context를 concatenate하여 학습
  - Next token prediction

- **BERT**: Masked Language Modeling
  - `[CLS] question [SEP] context [SEP]` 형식
  - 15% token masking

## Docker 이미지

### 베이스 이미지
- **이미지**: `pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime`
- **PyTorch**: 2.4.0
- **CUDA**: 12.4
- **cuDNN**: 9

### 포함된 도구
- PyTorch 2.4.0
- Transformers (HuggingFace)
- Datasets (HuggingFace)
- Chakra (MLCommons)
- TensorBoard
- Weights & Biases

### 볼륨 마운트
- `./outputs` → `/workspace/mosim-llms/outputs`
- `./data` → `/workspace/mosim-llms/data`
- `./checkpoints` → `/workspace/mosim-llms/checkpoints`

## 출력 파일

### 체크포인트
```
checkpoints/
├── gpt2/
│   └── best_model/
│       ├── config.json
│       ├── pytorch_model.bin
│       └── ...
└── bert/
    └── best_model/
        ├── config.json
        ├── pytorch_model.bin
        └── ...
```

### Chakra Traces
```
outputs/
├── *_kineto.json          # PyTorch Kineto trace (중간 형식)
├── *_stacks.txt           # Stack trace analysis
└── *.et                   # Chakra ET format (최종 출력) ✓
```

## 실험 설정

| Model | Dataset | GPUs | Batch Size/GPU | Effective Batch | DP Method |
|-------|---------|------|----------------|-----------------|-----------|
| GPT-2 | SQuAD   | 1    | 4              | 4               | N/A       |
| GPT-2 | SQuAD   | 2    | 4              | 8               | DDP       |
| GPT-2 | SQuAD   | 8    | 4              | 32              | DDP       |
| BERT  | SQuAD   | 1    | 4              | 4               | N/A       |
| BERT  | SQuAD   | 2    | 4              | 8               | DDP       |
| BERT  | SQuAD   | 8    | 4              | 32              | DDP       |

## 라이선스

Apache-2.0 License (Chakra와 동일)

## 🔧 Docker 이미지 최적화

### 사전 다운로드된 모델 & 토크나이저

Docker 이미지에는 다음이 **미리 포함**되어 있습니다:

- ✅ GPT-2 tokenizer (`/workspace/pretrained_models/gpt2`)
- ✅ BERT tokenizer (`/workspace/pretrained_models/bert-base-uncased`)

**장점:**
- 인터넷 연결 없이 실행 가능
- 매번 다운로드할 필요 없음
- HuggingFace Hub 접근 문제 해결

**자동 폴백:** 이미지 내 토크나이저가 없으면 자동으로 다운로드합니다.
