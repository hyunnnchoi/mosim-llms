# Chakra ET 변환 가이드

## 문제 상황

outputs 폴더에 다음 파일들이 있습니다:
- `*_host.json` - ExecutionTraceObserver가 생성한 host trace
- `*_device.json` - Kineto가 생성한 device trace

이 파일들을 Chakra ET (`.et`) 파일로 변환해야 합니다.

## 해결 방법

### 옵션 1: 가상 환경 사용 (권장)

가장 안전하고 깔끔한 방법입니다.

```bash
# 1. 가상 환경 생성
python3 -m venv chakra-venv

# 2. 가상 환경 활성화
source chakra-venv/bin/activate

# 3. 필요한 도구 설치
# PARAM 설치
git clone https://github.com/facebookresearch/param.git /tmp/param
cd /tmp/param/et_replay
git checkout 7b19f586dd8b267333114992833a0d7e0d601630
pip install .
cd -

# HolisticTraceAnalysis 설치
git clone https://github.com/facebookresearch/HolisticTraceAnalysis.git /tmp/HolisticTraceAnalysis
cd /tmp/HolisticTraceAnalysis
git checkout d731cc2e2249976c97129d409a83bd53d93051f6
git submodule update --init
pip install -r requirements.txt
pip install -e .
cd -

# Chakra 설치
pip install git+https://github.com/mlcommons/chakra.git

# 4. 변환 실행
python convert_to_et.py

# 5. 변환 완료 후 가상 환경 비활성화
deactivate
```

### 옵션 2: pipx 사용

```bash
# pipx 설치 (아직 설치하지 않았다면)
brew install pipx

# Chakra 설치
pipx install git+https://github.com/mlcommons/chakra.git

# 변환 실행
python3 convert_to_et.py
```

### 옵션 3: 서버에서 변환

서버에 이미 Chakra가 설치되어 있다면 서버에서 변환하는 것이 가장 간단합니다:

```bash
# 서버 접속
ssh your-server

# 프로젝트 디렉토리로 이동
cd /path/to/mosim-llms

# 변환 실행
python convert_to_et.py
# 또는
./convert_traces.sh

# 생성된 .et 파일을 로컬로 다운로드
# 로컬에서:
scp your-server:/path/to/mosim-llms/outputs/*.et ./outputs/
```

## 변환 프로세스

변환은 두 단계로 진행됩니다:

1. **chakra_trace_link**: host trace + device trace를 merge
   ```bash
   chakra_trace_link \
       --rank 0 \
       --chakra-host-trace outputs/gpt2_1gpu_quick_trace_host.json \
       --chakra-device-trace outputs/gpt2_1gpu_quick_trace_device.json \
       --output-file outputs/gpt2_1gpu_quick_trace_merged.json
   ```

2. **chakra_converter**: merged trace를 .et 파일로 변환
   ```bash
   chakra_converter PyTorch \
       --input outputs/gpt2_1gpu_quick_trace_merged.json \
       --output outputs/gpt2_1gpu_quick_trace
   ```

`convert_to_et.py` 스크립트는 이 과정을 자동으로 수행합니다.

## 예상 결과

변환이 완료되면 outputs 폴더에 다음 파일들이 생성됩니다:

```
outputs/
├── bert_1gpu_quick_trace.et         ✓ 최종 파일
├── bert_1gpu_quick_trace_merged.json  (중간 파일)
├── bert_2gpu_quick_trace.et         ✓
├── bert_4gpu_quick_trace.et         ✓
├── gpt2_1gpu_quick_trace.et         ✓
├── gpt2_2gpu_quick_trace.et         ✓
└── gpt2_4gpu_quick_trace.et         ✓
```

## 문제 해결

### 명령어를 찾을 수 없음
```bash
# 가상 환경이 활성화되어 있는지 확인
which chakra_trace_link

# PATH에 추가 (필요한 경우)
export PATH="$HOME/.local/bin:$PATH"
```

### 메모리 부족
큰 trace 파일의 경우 메모리가 부족할 수 있습니다:
- 한 번에 하나씩 변환
- 타임아웃 증가 (스크립트 내 timeout 값 수정)

### 설치 오류
가상 환경을 사용하는 것이 가장 안전합니다. 시스템 Python에 직접 설치하려고 하면 macOS에서 차단될 수 있습니다.

