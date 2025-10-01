# Chakra ET 변환 결과 요약

## ✅ 성공한 파일 (4개)

| 파일 | 크기 | 상태 |
|------|------|------|
| `bert_1gpu_quick_trace.et` | 10 MB | ✅ 완료 |
| `bert_2gpu_quick_trace.et` | 12 MB | ✅ 완료 |
| `bert_4gpu_quick_trace.et` | 12 MB | ✅ 완료 |
| `gpt2_1gpu_quick_trace.et` | 8.6 MB | ✅ 완료 |

## ❌ 실패한 파일 (2개)

| 파일 | 에러 | 원인 |
|------|------|------|
| `gpt2_2gpu_quick_trace` | RecursionError | 복잡한 multi-GPU 통신 그래프 |
| `gpt2_4gpu_quick_trace` | RecursionError | 복잡한 multi-GPU 통신 그래프 |

## 문제 원인

1. **host trace 중복 객체**
   - ExecutionTraceObserver가 trace를 여러 번 저장
   - 해결: `fix_host_traces.py`로 마지막 객체만 추출

2. **경로 중복 문제**
   - chakra_trace_link의 내부 경로 처리 이슈
   - 해결: 절대 경로 사용

3. **RecursionError**
   - multi-GPU trace의 순환 의존성 체크 과정에서 재귀 한도 초과
   - Chakra converter의 DFS 알고리즘 제한

## 해결 방법

### 옵션 1: 서버에서 변환 (권장)

서버의 Chakra 버전이 다르거나 더 많은 메모리가 있을 수 있습니다:

```bash
# 서버 접속
ssh your-server

# 프로젝트 디렉토리
cd /path/to/mosim-llms

# host trace 수정
python3 fix_host_traces.py

# 변환 실행
source chakra-venv/bin/activate  # 가상 환경이 있다면
python3 convert_to_et.py

# 생성된 .et 파일 다운로드
# 로컬에서:
scp your-server:/path/to/mosim-llms/outputs/*.et ./outputs/
```

### 옵션 2: 1-GPU 결과만 사용

Multi-GPU 결과가 필수가 아니라면:
- BERT: 1, 2, 4 GPU 모두 사용 가능 ✅
- GPT-2: 1 GPU만 사용 ✅

### 옵션 3: Chakra 버전 변경

다른 버전의 Chakra를 시도:

```bash
source chakra-venv/bin/activate
pip uninstall chakra -y
pip install chakra==0.0.3  # 또는 다른 버전
python convert_to_et.py
```

## 생성된 파일 위치

```
outputs/
├── bert_1gpu_quick_trace.et               ✅ 10 MB
├── bert_2gpu_quick_trace.et               ✅ 12 MB
├── bert_4gpu_quick_trace.et               ✅ 12 MB
├── gpt2_1gpu_quick_trace.et               ✅ 8.6 MB
├── gpt2_2gpu_quick_trace_merged.json      (중간 파일, 변환 실패)
└── gpt2_4gpu_quick_trace_merged.json      (중간 파일, 변환 실패)
```

## 다음 단계

성공한 4개 파일로 작업을 진행할 수 있습니다:

```bash
# ET 파일 확인
ls -lh outputs/*.et

# 분석/시뮬레이션에 사용
# 예: ASTRA-sim, Timeloop 등
```

## 사용한 도구

- ✅ `fix_host_traces.py` - host trace JSON 중복 객체 수정
- ✅ `convert_to_et.py` - host + device trace 병합 및 ET 변환
- ✅ `setup_and_convert.sh` - 자동 설치 및 변환 스크립트

## 향후 개선

서버에서 프로파일링을 다시 실행할 때:

1. **chakra_tracer.py 수정** (line 186-188)
   - `_trace_handler`에서 `et_observer.stop()` 호출하지 않기
   - 이미 주석에 명시되어 있음

2. **변환을 서버에서 바로 수행**
   - 프로파일링 직후 변환하면 환경 문제 최소화

3. **단일 GPU 먼저 테스트**
   - Multi-GPU는 복잡도가 훨씬 높음

