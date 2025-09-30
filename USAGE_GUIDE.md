# 사용 가이드

## 목차
1. [환경 설정](#환경-설정)
2. [학습 실행](#학습-실행)
3. [Chakra Trace 분석](#chakra-trace-분석)
4. [문제 해결](#문제-해결)

## 환경 설정

### 1. Docker 이미지 빌드

```bash
# 프로젝트 루트에서
./build.sh
```

성공 메시지:
```
✓ Docker image built successfully!
```

### 2. 컨테이너 실행

#### GPU 환경
```bash
./run.sh gpu
```

#### CPU 환경 (테스트용)
```bash
./run.sh cpu
```

#### Docker Compose 사용
```bash
# 백그라운드 실행
docker-compose up -d

# 컨테이너 접속
docker-compose exec mosim-llms bash

# 종료
docker-compose down
```

## 학습 실행

### 빠른 시작

컨테이너 내부에서:

```bash
# GPT-2 단일 GPU 학습
./run_gpt2.sh 1

# BERT 단일 GPU 학습
./run_bert.sh 1
```

### GPU 개수별 실행

#### 1 GPU
```bash
# GPT-2
./run_gpt2.sh 1 4  # 1 GPU, batch size 4

# BERT
./run_bert.sh 1 4
```

#### 2 GPUs
```bash
# GPT-2
./run_gpt2.sh 2 4  # 2 GPUs, batch size 4 per GPU

# BERT
./run_bert.sh 2 4
```

#### 8 GPUs
```bash
# GPT-2
./run_gpt2.sh 8 4  # 8 GPUs, batch size 4 per GPU

# BERT
./run_bert.sh 8 4
```

### 전체 실험 자동 실행

```bash
# 모든 조합 실행 (GPT-2, BERT × 1, 2, 8 GPU)
./run_all_experiments.sh
```

### 수동 실행

#### GPT-2

```bash
# 단일 GPU
python gpt2/train.py \
    --model-name gpt2 \
    --batch-size 4 \
    --learning-rate 5e-5 \
    --num-epochs 3 \
    --num-gpus 1 \
    --enable-tracing \
    --trace-output-dir ./outputs \
    --trace-name gpt2_custom \
    --save-dir ./checkpoints/gpt2

# 멀티 GPU (torchrun 사용)
torchrun \
    --nproc_per_node=8 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=29500 \
    gpt2/train.py \
    --model-name gpt2 \
    --batch-size 4 \
    --num-epochs 3 \
    --num-gpus 8 \
    --enable-tracing
```

#### BERT

```bash
# 단일 GPU
python bert/train.py \
    --model-name bert-base-uncased \
    --batch-size 4 \
    --learning-rate 5e-5 \
    --num-epochs 3 \
    --num-gpus 1 \
    --enable-tracing \
    --trace-output-dir ./outputs \
    --trace-name bert_custom \
    --save-dir ./checkpoints/bert

# 멀티 GPU (torchrun 사용)
torchrun \
    --nproc_per_node=8 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=29501 \
    bert/train.py \
    --model-name bert-base-uncased \
    --batch-size 4 \
    --num-epochs 3 \
    --num-gpus 8 \
    --enable-tracing
```

## Chakra Trace 분석

### Trace 파일 위치

학습 후 `outputs/` 디렉토리에 trace 파일이 생성됩니다:

```
outputs/
├── gpt2_1gpu_trace_kineto.json      # PyTorch Kineto trace (중간)
├── gpt2_1gpu_trace.et               # Chakra ET 파일 (최종) ✓
├── gpt2_1gpu_trace_stacks.txt       # 텍스트 분석
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

### Chakra ET 파일 사용

**주요 출력 파일: `*.et` (Chakra Execution Trace)**

```bash
# ET 파일 확인
ls -lh outputs/*.et

# ET 파일은 Chakra 시뮬레이터/분석 도구에서 사용
# 예: ASTRA-sim, Timeloop 등
```

### Kineto Trace 시각화 (디버깅용)

```bash
# Chrome DevTools로 Kineto trace 확인
# 1. Chrome 브라우저 열기
# 2. chrome://tracing 접속
# 3. outputs/*_kineto.json 파일 로드
```

### 텍스트 분석

```bash
# Stack trace 확인
cat outputs/gpt2_1gpu_trace_stacks.txt | head -50

# 주요 연산 시간 확인
grep "cuda_time_total" outputs/gpt2_1gpu_trace_stacks.txt
```

### 수동 ET 변환

자동 변환이 실패한 경우:

```bash
# Chakra Python API 사용
python -m chakra.et_converter.pytorch \
    --input outputs/gpt2_1gpu_trace_kineto.json \
    --output outputs/gpt2_1gpu_trace.et

# 또는 Python 스크립트
python << EOF
from chakra.et_converter.pytorch import PyTorchConverter
converter = PyTorchConverter()
converter.convert(
    "outputs/gpt2_1gpu_trace_kineto.json",
    "outputs/gpt2_1gpu_trace.et"
)
EOF
```

## 출력 파일

### 체크포인트

```bash
# 저장된 모델 확인
ls -lh checkpoints/gpt2/best_model/
ls -lh checkpoints/bert/best_model/

# 모델 로드 예제 (Python)
from transformers import GPT2LMHeadModel, BertForMaskedLM

gpt2_model = GPT2LMHeadModel.from_pretrained("./checkpoints/gpt2/best_model")
bert_model = BertForMaskedLM.from_pretrained("./checkpoints/bert/best_model")
```

### Trace 파일

```bash
# Trace 파일 확인
ls -lh outputs/

# 파일 크기 확인
du -sh outputs/*
```

## 문제 해결

### GPU 메모리 부족

```bash
# Batch size 줄이기
./run_gpt2.sh 1 2  # batch size 2

# Gradient accumulation 사용
python gpt2/train.py --batch-size 2 --gradient-accumulation-steps 2
```

### 멀티 GPU 실행 오류

```bash
# NCCL 환경 변수 설정
export NCCL_DEBUG=INFO
export NCCL_P2P_DISABLE=1

# 재실행
./run_gpt2.sh 8
```

### 데이터셋 다운로드 실패

```bash
# 캐시 디렉토리 지정
python gpt2/train.py --cache-dir ./data/cache

# 또는 환경 변수 설정
export HF_DATASETS_CACHE=./data/cache
```

### Trace 파일이 생성되지 않음

```bash
# Tracing이 활성화되었는지 확인
python gpt2/train.py --enable-tracing

# 출력 디렉토리 확인
ls -la outputs/

# 권한 문제 해결
chmod -R 755 outputs/
```

### CUDA Out of Memory

```bash
# 1. Batch size 줄이기
--batch-size 2

# 2. Sequence length 줄이기
--max-seq-length 256

# 3. 작은 모델 사용
--model-name gpt2  # instead of gpt2-medium
```

## 고급 사용법

### 커스텀 학습 파라미터

```bash
python gpt2/train.py \
    --model-name gpt2-medium \
    --batch-size 2 \
    --learning-rate 3e-5 \
    --weight-decay 0.01 \
    --num-epochs 5 \
    --warmup-steps 1000 \
    --max-grad-norm 1.0 \
    --save-steps 500 \
    --logging-steps 50 \
    --eval-steps 500
```

### Profiling 설정 조정

```python
# utils/chakra_tracer.py 수정
tracer = ChakraTracer(
    output_dir="./outputs",
    trace_name="custom_trace",
    wait_steps=5,      # 대기 스텝 증가
    warmup_steps=5,    # Warmup 스텝 증가
    active_steps=10,   # 프로파일링 스텝 증가
)
```

### 분산 학습 디버깅

```bash
# 단일 노드, 단일 프로세스로 테스트
python gpt2/train.py --num-gpus 1

# 로그 레벨 설정
export LOGLEVEL=DEBUG
torchrun --nproc_per_node=2 gpt2/train.py --num-gpus 2
```

## 참고 자료

- [PyTorch DDP 문서](https://pytorch.org/docs/stable/distributed.html)
- [Chakra GitHub](https://github.com/mlcommons/chakra)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [SQuAD Dataset](https://rajpurkar.github.io/SQuAD-explorer/)
