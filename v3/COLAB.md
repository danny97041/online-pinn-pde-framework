# V3 — 완료 결과 조회 / 저장 모델 예측 / 선택 학습

두 파일을 `MyDrive/PINN`에 함께 두고 **모두 실행**하세요.

- `Online_PINN_PDE_Framework_V3_Complete.ipynb`
- `Online_PINN_PDE_Framework_V3_Complete_Results.zip`

기본 `ACTION=results`: 9개 방정식, 65개 완료 실험을 조회합니다. 재학습·GPU·TensorFlow·LLM 불필요.
`predict`: 저장 모델만 로드. `agent`: 결과 도구 질문. `evaluate`: 30+15문항 검사.
`train`: 설정을 표시하고 사용자가 학습을 켠 경우에만 실행. 새 실험은 기본 1 seed.
`CHECK_TRAINING=True`는 `evaluate`에서 6회의 일회성 trial Adam update와 별도 wiring probe를
실행합니다. 이 경우에는 "학습 없음"으로 분류하지 않습니다. Benchmark 재학습은 아닙니다.
새 학습과 checkpoint 재개를 구분하고 모델 구조/샘플링 변경은 재개에서 차단합니다.
V3 10%는 연구 목표이지 산업 승인 기준이 아닙니다. Wave의 과거 5% 중단 이력은 보존합니다.

일반 결과 조회는 Colab 제공 NumPy만 사용합니다. 예측/학습은 제공 TensorFlow를 우선 사용.
Agent/API 검사에는 `fastapi httpx`, Qwen/E5에는 `torch transformers>=5` 필요.
Agent/평가는 기본 E5+BM25 하이브리드 검색. DENSE_RAG=False는 가벼운 BM25 경로.
필요한 기능의 의존성만 설치하세요. 단순 결과 조회는 모델을 내려받지 않습니다.
TensorFlow 특정 버전 설치는 강제하지 않습니다. 실제 버전과 행동 검사 결과를 기록합니다.
코드는 아래에 평문으로 공개되어 있으며 GitHub 소스와 같은 원본입니다.

현재 배포 브랜치는 문구·구조 점검 단계입니다. 저장된 이전 검사 기록과 이번 소스의
새 실행 검증을 구분하며, 실행 검증을 완료하기 전에는 `main`으로 승격하지 않습니다.
