# 개발·검증 안내

## 소스 구조

| 경로 | 역할 |
| --- | --- |
| `v3/contracts.py`, `v3/equations.py` | ZIP 계약·방정식·비교 보고서 |
| `v3/candidate_runtime.py`, `v3/wave_runtime.py`, `v3/wave_inference.py` | 학습·체크포인트·추론 |
| `v3/natural_language.py`, `v3/services.py` | 요청 해석·하이브리드 검색·도구 제어 |
| `v3/api_schemas.py`, `v3/api_routes.py`, `v3/api_services.py` | API 요청 형식·라우터·서비스 |
| `v3/documents.py`, `v3/visualization.py`, `v3/model_registry.py` | 문서·그림·추론 모델 관리 |
| `v3/colab_config.py`, `v3/colab_entry.py`, `v3/interface.py` | Colab 설정·실행 패널 |
| `tools/` | 노트북 동기화·패키징·검사 |

노트북은 평문 모듈로 생성합니다. 결과 ZIP 안의 Python 소스를 실행하지 않습니다.
일반 실행과 개발용 소스는 같은 학습 로직을 사용합니다.

## 소스와 노트북 확인

```bash
python -m compileall -q v3
python tools/sync_v3_notebook.py
python tools/check_public_docs.py
```

`sync_v3_notebook.py`는 정의 셀과 Colab 안내를 다시 생성합니다.
GitHub Actions는 Python 구문, 노트북 동기화, 결과 조회의 TensorFlow 비의존성을 확인합니다.
`check_public_docs.py`는 공개 문서 링크와 게시 목록을 확인하며 학습하지 않습니다.

CLI 결과 조회:

```bash
python -m v3 /path/to/V3_Results.zip
```

자연어 질문은 `--question`, 선택형 Qwen은 `--llm`, BM25만 사용하는 경로는 `--bm25`로 지정합니다.

## 검사 범위

| 검사 | 확인 사항 |
| --- | --- |
| 자연어·API | 쉬움·중간·어려움 각 10문항, 별도 표현 15문항, 도구 순서·수치·근거·응답 상태·허용되지 않은 요청 거절 |
| 저장 모델 | 선택 모델의 유한한 예측, 반복·단일/배치 일치, 저장 참조 오차 재계산 |
| 새 학습·재개 | 축소한 시험용 업데이트, 모델·옵티마이저·SA·표본 상태 복원 |
| Wave 재개 | 복제 체크포인트의 동일 다음 업데이트 |
| LHS 경계 | 실제 샘플러의 5,000/5,001 경계 제어 |
| 서비스 | 문서 생성·시각화·등록·API 계약·FAISS 선택 경로 |
| Colab 패널 | 조건부 설정·명시적 학습 허용·Drive 실패 처리 |

기능 검사의 모델·재개 검사는 선택 항목입니다. 전체 benchmark 재학습과는 다릅니다.
Qwen의 의도 판단과 제어기 보정, 시험용 임베딩 검사와 실제 E5 검색은 구분합니다.
브라우저 화면·Drive 인증·외부 서버 접근은 로컬 모의 검사만으로 확인되지 않습니다.

로컬 검사 예:

```bash
python tools/test_v3_interface.py --bundle /path/to/V3_Results.zip
python tools/test_v3_services.py --bundle /path/to/V3_Results.zip
```

`--numerical`은 추가 시험용 학습을 실행하므로 필요한 경우에만 사용합니다.
서비스 검사에는 선택형 FAISS를 포함해 해당 도구의 의존성이 필요합니다.

## 검사 기록

실행 당시 환경·소스 해시·검사 결과는 결과 ZIP의 `validation/`에 있습니다.
새 실행은 `validation/runs/`에 저장하며 `validation/current.json`으로 찾습니다.
소스 지문과 결과 출처는 `provenance/`에 보존합니다.
과거 검사 통과 기록은 수정된 코드의 새 실행 검사로 취급하지 않습니다.

## 배포

`tools/prepare_v3_publish.py`가 공개 소스 목록을 생성합니다.
계획·작업 메모·로컬 검토 JSON은 게시하지 않습니다. 검사 근거는 결과 ZIP에 보존합니다.
`tools/package_v3.py`는 검증된 결과 ZIP에서 모델·실험 데이터를 유지하고 최신 소스·사용 안내를 묶습니다.
큰 모델 ZIP은 Git에 커밋하지 않고 Release 첨부 파일로 제공합니다.

```bash
python tools/package_v3.py --evidence-zip /path/to/evidence.zip --validation-zip /path/to/check.zip
python tools/prepare_v3_publish.py
```

배포 파일의 SHA-256은 `V3_Release_Checksums.json`에서 확인합니다.
