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
├── run_gpt2.sh                 # GPT-2 실행 스크립트
├── run_bert.sh                 # BERT 실행 스크립트
└── run_all_experiments.sh      # 전체 실험 자동 실행
```

## 빠른 시작

### 1. Docker 이미지 빌드

```bash
./build.sh
```

### 2. 컨테이너 실행

```bash
# GPU 사용
./run.sh gpu

# 또는 docker-compose 사용
docker-compose up -d
docker-compose exec mosim-llms bash
```

### 3. 학습 실행

컨테이너 내부에서:

```bash
# GPT-2: 1 GPU
./run_gpt2.sh 1

# GPT-2: 2 GPUs
./run_gpt2.sh 2

# GPT-2: 8 GPUs
./run_gpt2.sh 8

# BERT: 1 GPU
./run_bert.sh 1

# BERT: 2 GPUs
./run_bert.sh 2

# BERT: 8 GPUs
./run_bert.sh 8

# 모든 실험 자동 실행
./run_all_experiments.sh
```

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

## Chakra Execution Trace

### Trace 캡처

학습 중 자동으로 PyTorch Kineto trace를 생성하고 Chakra ET 파일로 변환합니다:

- **중간 형식**: Kineto trace JSON (PyTorch profiler 출력)
- **최종 형식**: Chakra Execution Trace (`.et`)
- **저장 위치**: `./outputs/`
- **캡처 내용**:
  - Compute operations (forward, backward)
  - Memory operations (allocation, transfer)
  - Communication operations (all-reduce for DDP)
  - Timing information
  - Stack traces

### Trace 파일

```
outputs/
├── gpt2_1gpu_trace_kineto.json      # PyTorch Kineto trace
├── gpt2_1gpu_trace.et               # Chakra ET 파일 ✓
├── gpt2_1gpu_trace_stacks.txt       # 분석용 텍스트
├── gpt2_2gpu_trace_kineto.json
├── gpt2_2gpu_trace.et               # Chakra ET 파일 ✓
├── gpt2_8gpu_trace_kineto.json
├── gpt2_8gpu_trace.et               # Chakra ET 파일 ✓
├── bert_1gpu_trace_kineto.json
├── bert_1gpu_trace.et               # Chakra ET 파일 ✓
├── bert_2gpu_trace_kineto.json
├── bert_2gpu_trace.et               # Chakra ET 파일 ✓
├── bert_8gpu_trace_kineto.json
└── bert_8gpu_trace.et               # Chakra ET 파일 ✓
```

### Chakra ET 변환

자동 변환이 실패하는 경우 수동으로 변환 가능:

```bash
# Chakra Python API 사용
python -m chakra.et_converter.pytorch \
    --input outputs/gpt2_1gpu_trace_kineto.json \
    --output outputs/gpt2_1gpu_trace.et

# 또는 직접 Python 스크립트
from chakra.et_converter.pytorch import PyTorchConverter
converter = PyTorchConverter()
converter.convert("input.json", "output.et")
```

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
