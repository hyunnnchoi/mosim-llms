# MOSIM-LLMs 프로젝트 요약

## 프로젝트 개요

SQuAD 데이터셋으로 GPT-2와 BERT를 Pretrain하면서 Chakra Execution Trace를 캡처하는 프로젝트입니다.

### 핵심 요구사항
- ✅ PyTorch 2.4.0 (Docker: `pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime`)
- ✅ GPT-2 & BERT Pretrain 구현
- ✅ SQuAD 데이터셋 사용
- ✅ Data Parallelism (1, 2, 8 GPU)
- ✅ Chakra Trace 캡처
- ✅ FSDP 미사용 (DDP만 사용)

## 파일 구조

```
mosim-llms/
├── Docker 관련
│   ├── Dockerfile                  # PyTorch 2.4.0 + Chakra
│   ├── docker-compose.yml          # Docker Compose 설정
│   ├── .dockerignore              # Docker 빌드 제외 파일
│   ├── build.sh                   # 이미지 빌드 스크립트
│   └── run.sh                     # 컨테이너 실행 스크립트
│
├── 모델 구현
│   ├── gpt2/
│   │   ├── __init__.py
│   │   ├── config.py              # GPT-2 설정
│   │   ├── model.py               # GPT-2 모델 (HuggingFace 기반)
│   │   └── train.py               # GPT-2 학습 스크립트 (DDP 지원)
│   │
│   └── bert/
│       ├── __init__.py
│       ├── config.py              # BERT 설정
│       ├── model.py               # BERT 모델 (HuggingFace 기반)
│       └── train.py               # BERT 학습 스크립트 (DDP 지원)
│
├── 유틸리티
│   └── utils/
│       ├── __init__.py
│       ├── distributed.py         # DDP 설정 (MultiWorkerMirroredStrategy 유사)
│       ├── data_utils.py          # SQuAD 데이터 로딩
│       └── chakra_tracer.py       # Chakra trace 캡처
│
├── 실행 스크립트
│   ├── run_gpt2.sh                # GPT-2 실행 (1/2/8 GPU)
│   ├── run_bert.sh                # BERT 실행 (1/2/8 GPU)
│   └── run_all_experiments.sh     # 모든 실험 자동 실행
│
├── 문서
│   ├── README.md                  # 프로젝트 README
│   ├── USAGE_GUIDE.md             # 상세 사용 가이드
│   └── PROJECT_SUMMARY.md         # 이 파일
│
└── 설정
    ├── requirements.txt           # Python 의존성
    └── .gitignore                # Git 제외 파일
```

## 주요 기능

### 1. GPT-2 Pretrain
- **모델**: GPT-2 (HuggingFace `transformers`)
- **Task**: Causal Language Modeling
- **데이터**: SQuAD (question + context)
- **지원 GPU**: 1, 2, 8
- **Batch Size**: 4 per GPU (조정 가능)

### 2. BERT Pretrain
- **모델**: BERT (HuggingFace `transformers`)
- **Task**: Masked Language Modeling (15% masking)
- **데이터**: SQuAD ([CLS] question [SEP] context [SEP])
- **지원 GPU**: 1, 2, 8
- **Batch Size**: 4 per GPU (조정 가능)

### 3. Data Parallelism
- **방식**: PyTorch DistributedDataParallel (DDP)
- **특징**: TensorFlow `MultiWorkerMirroredStrategy`와 유사
- **FSDP**: 사용 안 함
- **실행**: `torchrun`으로 멀티프로세스 실행

### 4. Chakra Tracing
- **도구**: PyTorch Profiler + Chakra
- **출력 형식**: Chrome trace JSON (`.json`)
- **캡처 내용**:
  - Compute ops (forward/backward)
  - Memory ops (allocation/transfer)
  - Communication ops (all-reduce for DDP)
  - Timing & stack traces
- **변환**: 향후 Chakra ET 형식으로 변환 가능

## 사용 방법

### 빠른 시작

```bash
# 1. Docker 이미지 빌드
./build.sh

# 2. 컨테이너 실행
./run.sh gpu

# 3. 학습 실행 (컨테이너 내부)
./run_gpt2.sh 1    # GPT-2, 1 GPU
./run_bert.sh 1    # BERT, 1 GPU

# 4. 모든 실험 자동 실행
./run_all_experiments.sh
```

### GPU별 실행

```bash
# 1 GPU
./run_gpt2.sh 1
./run_bert.sh 1

# 2 GPUs
./run_gpt2.sh 2
./run_bert.sh 2

# 8 GPUs
./run_gpt2.sh 8
./run_bert.sh 8
```

## 실험 설정

| Model | Dataset | GPUs | Batch/GPU | Total Batch | Method |
|-------|---------|------|-----------|-------------|--------|
| GPT-2 | SQuAD   | 1    | 4         | 4           | -      |
| GPT-2 | SQuAD   | 2    | 4         | 8           | DDP    |
| GPT-2 | SQuAD   | 8    | 4         | 32          | DDP    |
| BERT  | SQuAD   | 1    | 4         | 4           | -      |
| BERT  | SQuAD   | 2    | 4         | 8           | DDP    |
| BERT  | SQuAD   | 8    | 4         | 32          | DDP    |

## 출력 파일

### 체크포인트
```
checkpoints/
├── gpt2/best_model/
│   ├── config.json
│   └── pytorch_model.bin
└── bert/best_model/
    ├── config.json
    └── pytorch_model.bin
```

### Chakra Traces
```
outputs/
├── gpt2_1gpu_trace_chrome.json
├── gpt2_1gpu_trace_stacks.txt
├── gpt2_2gpu_trace_chrome.json
├── gpt2_8gpu_trace_chrome.json
├── bert_1gpu_trace_chrome.json
├── bert_2gpu_trace_chrome.json
└── bert_8gpu_trace_chrome.json
```

## 기술 스택

### Docker
- Base: `pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime`
- CUDA: 12.4
- cuDNN: 9

### Python 패키지
- PyTorch 2.4.0
- Transformers (HuggingFace)
- Datasets (HuggingFace)
- Chakra (MLCommons)
- TensorBoard
- Weights & Biases

### 분산 학습
- `torch.distributed` (DDP)
- `torch.nn.parallel.DistributedDataParallel`
- `torchrun` launcher

### Profiling
- `torch.profiler`
- Chakra trace export

## 주요 구현 포인트

### 1. Data Parallelism (utils/distributed.py)
```python
# TensorFlow MultiWorkerMirroredStrategy 유사
- setup_distributed(): DDP 환경 초기화
- cleanup_distributed(): 환경 정리
- reduce_value(): 프로세스 간 값 reduce
```

### 2. SQuAD 데이터 로딩 (utils/data_utils.py)
```python
# GPT-2: question + context (CLM)
# BERT: [CLS] question [SEP] context [SEP] (MLM)
- SQuADDataset: 커스텀 데이터셋
- get_dataloaders(): DistributedSampler 지원
```

### 3. Chakra Tracing (utils/chakra_tracer.py)
```python
# PyTorch Profiler 래퍼
- ChakraTracer: Context manager
- Chrome trace export
- Stack trace analysis
```

### 4. 학습 스크립트 (gpt2/train.py, bert/train.py)
```python
# DDP 지원 학습 루프
- setup_distributed()
- Model wrapping with DDP
- DistributedSampler for DataLoader
- Chakra tracing integration
```

## GitHub 업로드 체크리스트

- [x] Dockerfile (PyTorch 2.4.0 + Chakra)
- [x] docker-compose.yml
- [x] requirements.txt
- [x] GPT-2 구현 (train.py, model.py, config.py)
- [x] BERT 구현 (train.py, model.py, config.py)
- [x] Utils (distributed.py, data_utils.py, chakra_tracer.py)
- [x] 실행 스크립트 (run_*.sh)
- [x] .gitignore
- [x] .dockerignore
- [x] README.md
- [x] USAGE_GUIDE.md
- [x] PROJECT_SUMMARY.md

## 다음 단계

1. **GitHub 업로드**
   ```bash
   git init
   git add .
   git commit -m "Initial commit: GPT-2 & BERT pretrain with Chakra tracing"
   git remote add origin <your-repo-url>
   git push -u origin main
   ```

2. **Docker 이미지 테스트**
   ```bash
   ./build.sh
   ./run.sh gpu
   ```

3. **학습 테스트**
   ```bash
   # 컨테이너 내부
   ./run_gpt2.sh 1
   ./run_bert.sh 1
   ```

4. **Trace 분석**
   - Chrome DevTools (`chrome://tracing`)에서 trace 파일 열기
   - 성능 병목 지점 분석
   - Chakra ET 형식으로 변환 (향후)

## 참고 링크

- [Chakra GitHub](https://github.com/mlcommons/chakra)
- [PyTorch DDP Tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [SQuAD Dataset](https://rajpurkar.github.io/SQuAD-explorer/)
