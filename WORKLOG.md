# WORKLOG

## [2026-04-19 12:28] A100 results_v2 — whisper 실험 반영한 results_summary.csv 재생성

### 실행 환경 (스크립트 실행 시)
| 항목 | 값 |
|------|-----|
| 스크립트 | `/tmp/update_csv.py` (1회성) |
| 입력 | `A100/results_v2/*_solo.json`, `*_with_*.json` (총 132개 JSON) |
| 출력 | `A100/results_v2/results_summary.csv` |

### 변경 사항
| 파일 | 유형 | 설명 |
|------|------|------|
| `A100/results_v2/results_summary.csv` | 수정 | whisper 모델(solo + pair 22행) 추가하여 전체 CSV 재생성 |
| `A100/results_v2/logs/2026-04-19_*_update_csv_whisper.log` | 생성 | 스크립트 실행 로그 |
| `WORKLOG.md` | 생성 | 작업 기록 파일 신규 생성 |

### 작업 상세
- 배경: 모델 풀에 whisper가 추가되어 `whisper_solo.json`, `whisper_with_{model}.json`, `{model}_with_whisper.json` 실험이 수행됨.
- CSV 스키마를 유지하면서, 모든 JSON을 다시 읽어 CSV를 재생성.
- 행 정렬: pair 그룹(primary 모델 알파벳 순, partner 알파벳 순) → solo 그룹(알파벳 순).
- `slowdown_ratio`는 primary 모델의 solo `iter_times_sec.mean` 대비 비율 (소수점 4자리 반올림).
- 결과: 기존 110행(10×10 pair + 10 solo) → 132행(11×11 pair + 11 solo).
- 기존 non-whisper 행들의 값은 원본과 동일함을 확인.

### 로그
- `A100/results_v2/logs/2026-04-19_*_update_csv_whisper.log`

### 관련 이슈
- 없음

## [2026-04-19 12:32] A100 results_v2 — 히트맵 플롯 스크립트 및 PDF 이동

### 변경 사항
| 파일 | 유형 | 설명 |
|------|------|------|
| `A100/results_v2/plot_heatmap_v3.py` | 이동 | `~/Downloads/files/`에서 결과 폴더로 이동 (10×10 interference 히트맵 플롯) |
| `A100/results_v2/00-interference_heatmap_10x10_final_v3.pdf` | 이동 | 히트맵 출력 PDF 이동 |

### 작업 상세
- 플롯 스크립트와 출력 PDF를 결과 폴더와 함께 보관하기 위해 이동.
- 참고: 스크립트 내부 경로(`/mnt/user-data/uploads/`, `/home/claude/`)는 외부 환경용 경로 그대로 두었음 (현 시점 수정 요청 없음).
