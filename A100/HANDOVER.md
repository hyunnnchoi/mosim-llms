# Whisper 실험 인수인계 문서

## 작업 개요
A100 8-GPU 간섭(interference) 실험에 **Whisper-small (241.7M params)** 모델을 추가하는 작업.
기존 10개 모델(gpt2, bert, resnet44, resnet110, resnet50, vgg16, googlenet, inception3, densenet40_k12, densenet100_k12)에 whisper를 추가하여
whisper가 포함된 22개 실험 조합을 실행.

## 완료된 작업

### 1. 코드 구현 (완료)

**새로 생성한 파���:**
- `whisper/__init__.py` — 패키지 초기화
- `whisper/config.py` — `WhisperLocalConfig` dataclass (whisper-small: d_model=768, 12 encoder/decoder layers)
- `whisper/model.py` — `WhisperModel` 래퍼 (HF `WhisperForConditionalGeneration`, from scratch 초기화)
- `A100/train_whisper.py` — DDP 학습 스크립트 + `SyntheticWhisperDataset` (랜덤 mel-spectrogram, 외부 데이터 불필요)
- `A100/run_whisper_experiments.sh` — whisper 전용 22개 실험 스크립트

**수정한 파일:**
- `A100/config.py` — `MODEL_CONFIGS`에 whisper 항목 추가 (batch=16, mel_seq_len=3000, lr=5e-4)
- `A100/run_all_experiments_v2.sh` — MODELS 배열, get_script, build_args에 whisper 추가
- `A100/run_self_pairs_v2.sh` — 동일

### 2. 간섭 잘 보이게 한 설계 포인트
- 241.7M params (GPT2 124M, BERT 110M 대비 2배)
- batch_size=16, mel_seq_len=3000 (30초 분량) → 입력 텐서 [16, 80, 3000]으로 메모리/bandwidth 압박
- encoder + decoder 양쪽 12 layers → compute-heavy
- synthetic data → I/O bottleneck 없이 순수 GPU 연산에 집중

### 3. 디버�� 이력
- HF tokenizer 캐시 경로 문제: `CUDA_VISIBLE_DEVICES` 변경 시 HF 캐시 못 찾는 문제 ��� `HF_HOME=/home/work/hyunmokchoi/hf_cache` 환경변수 추가로 해결
- `set -euo pipefail`의 `set -u` (unbound variable) 문제 → `set -eo pipefail`로 변경
- 포트 충돌 (29500 EADDRINUSE): 이전 프로세스 잔재 → `pkill -9` 후 재시작으로 해결
- `build_args` 함수의 `BARRIER_DIR` 스코프 문제 → 함수 대신 인라인으로 펼쳐서 해결

## 실험 실행 결과 (완료!)

**22/22 전부 성공** (`bash A100/run_whisper_experiments.sh 100`)
총 소요시간: ~29분

**실험 22개 구성:**
1. `whisper_solo` — solo baseline
2-11. `whisper_with_X` (10개) — whisper가 primary(측정 대상), X가 interferer
12-21. `X_with_whisper` (10개) — X가 primary, whisper가 interferer
22. `whisper_with_whisper` — self-pair

**실행 속도 실적:**
- solo: 86초
- whisper as primary pair: 98~127초 (평균 ~107초)
- whisper as interferer pair: 28~88초 (평균 ~53초, primary가 CNN이면 빠름)
- whisper self-pair: 120초

### 모니터링 방법
```bash
# 요약 로그 (최신 파일)
cat $(ls -t A100/logs_v2/whisper_experiments_*.log | head -1)

# 결과 파일 수 확인
ls A100/results_v2/whisper_* A100/results_v2/*_with_whisper* 2>/dev/null | wc -l

# 개별 pair 로그
tail -5 A100/logs_v2/pair_whisper_*_A.log
tail -5 A100/logs_v2/pair_*_whisper_A.log
```

### 실험이 끝나면 할 일
1. 결과 파일 22개 생성 확인:
   ```bash
   ls A100/results_v2/whisper_solo.json
   ls A100/results_v2/whisper_with_*.json    # 11개 (10 + self)
   ls A100/results_v2/*_with_whisper.json    # 10개
   ```
2. 실패한 실험이 있으면 로그 확인 후 개별 재실행
3. `analyze_results.py`로 전체 결과 분석 (whisper 포함)

### 만약 실험이 중간에 멈췄거나 세션이 끊겼다면
```bash
# 실행 중인 프로세스 확인
pgrep -f "torchrun|train_whisper|train_gpt2|train_bert|train_cnn"

# 포트 사용 확인
lsof -i :29500; lsof -i :29501

# 모든 학습 프로세스 정리
pkill -9 -f "torchrun|train_whisper|train_gpt2|train_bert|train_cnn"

# 완료된 결과 확인 후 재시작
ls A100/results_v2/whisper_* A100/results_v2/*_with_whisper* 2>/dev/null
bash A100/run_whisper_experiments.sh 100
```
주의: 스크립트가 이미 존재하는 결과를 skip하지 않고 덮어쓰므로, 완료된 결과만 보존하고 싶다면 수동으로 실패분만 재실행할 것.

### 미완료 실험만 재실행하고 싶을 때
이미 `results_v2/` 에 JSON이 있는 실험은 성공한 것. 없는 조합만 골라서 돌리면 됨.
```bash
# 예: whisper_with_inception3 만 돌리기
cd /home/work/hyunmokchoi/mosim-llms
export HF_HOME=/home/work/hyunmokchoi/hf_cache
export HF_DATASETS_CACHE=/home/work/hyunmokchoi/hf_cache
export TRANSFORMERS_CACHE=/home/work/hyunmokchoi/hf_cache
export HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1

BDIR=/tmp/a100_barrier_tmp && rm -rf $BDIR && mkdir -p $BDIR

# Primary (whisper, measured)
CUDA_VISIBLE_DEVICES=0,2,4,6 NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 \
torchrun --nproc_per_node=4 --master_port=29500 \
A100/train_whisper.py --mode pair --partner inception3 --total-steps 100 \
--output-dir A100/results_v2 --role primary \
--job-id whisper_primary --partner-id inception3_interferer --barrier-dir $BDIR &

# Interferer (inception3)
CUDA_VISIBLE_DEVICES=1,3,5,7 NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 \
torchrun --nproc_per_node=4 --master_port=29501 \
A100/train_cnn.py --model inception3 --mode pair --partner whisper --total-steps 100 \
--output-dir A100/results_v2 --role interferer \
--job-id inception3_interferer --partner-id whisper_primary --barrier-dir $BDIR &

wait
```

## 프로젝트 구조 요약
```
mosim-llms/
├── A100/                          # A100 간섭 실험
│   ├── config.py                  # 모델 configs (gpt2, bert, whisper, CNNs)
│   ├── train_bert.py
│   ├── train_gpt2.py
│   ├── train_whisper.py           # NEW
│   ├── train_cnn.py
│   ├── metrics.py                 # ExperimentMetrics (JSON 저장)
│   ├── barrier.py                 # Pair 동기화
│   ├── run_whisper_experiments.sh # NEW: whisper 전용 22개 실험
│   ├── run_all_experiments_v2.sh  # 전체 모델 실험 (whisper 포함)
│   ├── run_self_pairs_v2.sh       # Self-pair 실험 (whisper 포함)
│   ├── results_v2/                # 실험 결과 JSON
│   └── logs_v2/                   # 실험 로그
├── whisper/                       # NEW
│   ├── config.py                  # WhisperLocalConfig
│   └── model.py                   # WhisperModel (HF 래퍼)
├── bert/
├── gpt2/
├── utils/
│   └── data_utils.py              # SQuAD 데이터 (bert/gpt2용, whisper는 자체 synthetic)
└���─ pretrained_bundle.tar.gz       # gpt2, bert, squad ���시 (hf_cache에 풀려있음)
```
