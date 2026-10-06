# 온라인 PINN PDE 프레임워크 V3

## 시작하기

다음 두 파일을 `MyDrive/PINN`에 함께 둔 뒤 **모두 실행**을 선택하세요.

- 실행 노트북: `Online_PINN_PDE_Framework_V3.ipynb`
- 통합 결과: `Online_PINN_PDE_Framework_V3_Results.zip`

기본 `ACTION="results"`는 9개 방정식·65개 완료 실험의 저장 비교표를 표시합니다.
재학습·GPU·TensorFlow·언어 모델은 필요하지 않습니다. 원본 실험 ZIP을 따로 선택하지 않습니다.

## 기능 선택

| ACTION 값 | 기능 | 학습 여부 |
| --- | --- | --- |
| `results` | 저장 결과와 보고서 조회 | 없음 |
| `predict` | 지정 좌표의 저장 모델 예측 | 없음 |
| `agent` | 자연어 질문에 결과 도구로 응답 | 없음 |
| `evaluate` | 자연어·도구·API 검사 | 학습 검사 옵션을 켠 경우만 있음 |
| `train` | 선택 방정식의 새 학습 또는 재개 | 사용자가 학습을 활성화한 경우만 있음 |

변수명과 선택값은 Python·API 호환성을 위해 영문으로 유지합니다.
학습 설정은 `ACTION="train"`, `TRAINING_ENABLED=False`로 먼저 확인하세요.
설정 확인 후에만 `TRAINING_ENABLED=True`로 변경합니다.

## 선택 기능의 실행 환경

결과 조회는 NumPy만 사용합니다. 예측·학습은 Colab 제공 TensorFlow를 우선 사용하며 특정 버전을 강제하지 않습니다.
도구·API 검사에는 `fastapi httpx`, Qwen·E5 검색에는 `torch transformers>=5`가 필요합니다.
선택한 기능의 의존성만 설치하세요. 단순 결과 조회에서는 모델을 내려받지 않습니다.
GPU 기능은 T4 또는 L4에서 사용할 수 있으며 셀별 장치 전환은 필요하지 않습니다.

## 검사 결과와 추가 학습

`evaluate`는 쉬움·중간·어려움 각 10문항과 별도 표현 15문항을 검사합니다.
`USE_LLM=True`는 실제 Qwen을 사용합니다. `DENSE_RAG=True`는 E5+BM25 검색,
`False`는 BM25 검색을 선택합니다. 시스템 성공과 언어 모델의 판단 일치는 별도로 기록합니다.

`CHECK_TRAINING=True`는 두 방정식에서 총 6회의 시험용 Adam 업데이트와 별도 연결 검사를 수행합니다.
전체 실험을 다시 학습하는 과정은 아닙니다.
검사 출력은 `Online_PINN_PDE_Framework_V3_Validation.zip`입니다.
이 파일을 다음 검사의 입력으로 지정하면 이전 기록을 포함해 갱신합니다. 통합 결과 원본은 보존합니다.

추가 학습의 기본 시드는 3234 하나입니다. 준비와 실험 완료 경계에서 하나의 ZIP을 갱신하며
partial ZIP은 만들지 않습니다. 연결이 끊기면 미완료 실험은 유실될 수 있습니다.

## 결과 해석

V3의 10% 목표는 연구용 물리장 오차 기준이지 산업 인증 기준이 아닙니다.
Wave의 과거 5% 중단 이력은 보존합니다. 서로 다른 방정식의 PDE 잔차 원값은 직접 비교하지 않습니다.
경계·초기조건 및 국소 최악점의 합격 판정은 별도 검증 대상입니다.
