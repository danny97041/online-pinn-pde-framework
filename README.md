# Online PINN PDE Framework

2D Wave-PINN 학습·추론을 FastAPI, Hybrid RAG, AI Agent 및 자동 보고서 생성으로 연결한 AI Engineering Portfolio입니다. 해석해가 있는 PDE를 이용하는 재사용 가능한 framework로 확장하고 있습니다.

**실행 결과부터 살펴보려면 [V1 실행 Notebook](outputs/Online_PINN_2D_Portfolio_V1_Executed.ipynb)을 확인하세요.** 저장된 출력과 그림은 재학습 없이 읽을 수 있습니다.

`2D Wave-PINN → FastAPI → Hybrid RAG → AI Agent → Automated Reporting`

TensorFlow 기반 모델, FastAPI/Pydantic API, multilingual-e5-small·FAISS·BM25·RRF 검색, Local LLM 도구 호출을 연결합니다. Notebook 실행 시 `app/` 아래 API, schema, service, ML, RAG, Agent 모듈을 생성하는 구조이며, 저장소에는 미리 생성된 서버 패키지나 모델 가중치가 포함되지 않습니다.

## Versioned Notebooks

| Version | Notebook | Status | Purpose |
|---|---|---|---|
| V1 | [`Online_PINN_2D_Portfolio_V1_Executed.ipynb`](outputs/Online_PINN_2D_Portfolio_V1_Executed.ipynb) | 실행 및 회귀검증 완료 | 고정 50,000 iteration 기반 2D Wave 실행 결과 보존본 |
| V2 | [`Online_PINN_2D_Portfolio_V2.ipynb`](outputs/Online_PINN_2D_Portfolio_V2.ipynb) | 코드 구현 및 정적 검증 완료 | 최대 100,000 iteration과 Relative L1/L2 목표기반 중단 구조 |
| V3 | [V3 개발 Notebook](https://github.com/danny97041/online-pinn-pde-framework/blob/codex/v3-pde-framework/outputs/Online_PINN_PDE_Framework_V3.ipynb) | 연구 workflow 개발 중 | 단계별 런타임 검증과 재개 가능한 PDE framework |

V1에는 6개 방법과 4개 seed의 24개 비교 실험, API, Hybrid RAG, Agent, 문서 생성 및 최종 통합 테스트 결과가 저장되어 있습니다. V2의 목표기반 중단 방식은 전체 GPU benchmark를 다시 실행하기 전이므로 실행 검증 완료로 표시하지 않습니다.

V3의 현재 구현 범위는 Wave2D입니다. Burgers, Kovasznay Forward/Inverse 및 Taylor-Green Vortex는 exact-solution 검증을 선행한 뒤 단계적으로 추가합니다.

## Artifact Policy

모델 가중치, TensorFlow checkpoint, 학습 데이터, FAISS index 및 단계별 ZIP은 Git에 저장하지 않습니다. 실행 artifact는 Google Drive에서 별도로 관리합니다. 기존 모델 추론이나 학습 재개에는 호환 artifact가 필요하며 저장소 clone만으로 기존 실행 상태가 복원되지는 않습니다.

## Recorded Verification

2026-08-31 V1 실행본에 저장된 결과입니다. 이번 문서 정리에서 재실행한 테스트 결과는 아닙니다.

| 테스트 | 저장된 결과 |
|---|---|
| Wave2D / RAG API 통합 | 통과 |
| RAG Retrieval | 12/12 |
| 기본 Agent Tool Routing | 6/6 |
| Agent·문서 생성 회귀 | 13/13 |
| Schema·Calculator 경계값 | 16/16 |
| Final Integration | 9/9 |

위 값은 한정된 회귀 테스트셋의 통과 건수이며 일반적인 모델 정확도 추정치가 아닙니다. 24/24 비교 실험 완료 역시 수렴 목표 달성이나 PINN의 우위를 뜻하지 않습니다. Supervised-only를 포함한 방법별 오차와 물리 잔차를 함께 해석해야 합니다.

V1 소개 셀의 `lambda_1 = 0.972986`, Relative L2 `11.54%`는 2026-08-28 기록입니다. 2026-08-31 제공 ZIP의 `training_result.json`에는 각각 약 `0.972833`, `11.5084%`가 기록되어 있습니다. 날짜가 다른 실행 기록을 혼합하지 않으며 원본 Notebook과 출력은 보존합니다.

## Execution Guide

1. 실행할 Notebook을 Google Colab에 업로드합니다. 의존 패키지는 Notebook 설치 셀에서 설치합니다.
2. V3 최초 실행은 CPU 런타임에서 Stage `1`, mode `new` 또는 `0`으로 시작합니다.
3. 이후 단계는 앞선 V3 snapshot과 완료 기록을 사용하여 `load` 또는 `1`로 진행합니다.
4. Stage 4·7·8은 GPU 학습 단계로 각각 smoke, pilot, 승인된 benchmark를 수행합니다.

V3 전체 Colab 순차 실행과 중단 후 복구 검증은 남아 있습니다. 현재 최종 snapshot 셀의 `complete` 표시는 앞선 모든 검증의 성공을 독립적으로 보장하지 않으므로 앞선 셀 오류가 있으면 완료 증거로 사용하지 않습니다. 패키지 버전이 모두 고정되어 있지 않아 환경에 따른 재검증이 필요합니다.

V3 브랜치의 `work/verify_v3_research_workflow.py`는 구문·구조와 모의 CPU/GPU 불일치 검사를 수행합니다. 학습이나 실제 checkpoint 복구를 검증하는 테스트는 아닙니다. `work/create_v3_research_workflow.py`는 실행 시 V3 Notebook을 다시 생성하여 덮어쓰므로 일반적인 실행에는 사용하지 않습니다.

## Research Scope

논문 서지정보와 사용자가 제공한 원문 링크는 V1의 Research & Publications 셀에 있습니다. 이 2D Wave 비교를 학위논문의 Navier–Stokes 실험 수치 재현으로 제시하지 않습니다. Burgers, Kovasznay, Taylor–Green은 계획 단계입니다.

## Branch Policy

- `main`: V1 실행본과 V2 소스 보존. 버전별 검증 범위는 위 표를 따릅니다.
- `codex/v3-pde-framework`: V3 PDE framework 연구개발

자동 모델 승격은 허용하지 않습니다. 정식 benchmark는 pilot 결과를 검토하고 명시적으로 승인한 configuration만 실행합니다.

`v2-wave2d`는 기존 역사적 태그이며 이름 자체가 V2 전체 실행 검증을 의미하지 않습니다. 별도 라이선스는 아직 지정하지 않았으며 코드 재사용·재배포 조건은 추후 명시합니다.
