# Online PINN PDE Framework

### 온라인 PINN PDE 프레임워크 V3

**9개 방정식 · 65개 저장 실험 · 5개 비교 방법**

저장 결과를 비교하고, 좌표를 예측하거나 자연어로 질문할 수 있습니다.
필요한 방정식만 선택해 설정을 확인·변경하고 추가 학습을 실행할 수도 있습니다.

[![Colab에서 실행](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/danny97041/online-pinn-pde-framework/blob/main/outputs/Online_PINN_PDE_Framework_V3.ipynb)
[![배포 파일](https://img.shields.io/badge/다운로드-GitHub_Releases-2563eb)](https://github.com/danny97041/online-pinn-pde-framework/releases)
[![사용 안내](https://img.shields.io/badge/안내-한국어-16a34a)](v3/README.md)

## 두 파일로 시작하기

| 파일 | 내용 |
| --- | --- |
| [Online_PINN_PDE_Framework_V3.ipynb](outputs/Online_PINN_PDE_Framework_V3.ipynb) | Colab 실행 노트북 |
| `V3_Results.zip` | 저장 모델·비교 결과·검색 근거·검사 기록 |

1. 노트북과 결과 ZIP을 내려받아 `MyDrive/PINN`에 함께 둡니다.
2. 노트북을 Colab에서 열고 **런타임 → 모두 실행**을 선택합니다.
3. 최초 설정을 적용하거나 60초 기다립니다. 기본 기능은 **저장 결과 보기**입니다.
4. 맨 아래 실행 패널에서 사용할 기능과 필요한 설정만 선택합니다.

결과표 조회에는 GPU·TensorFlow·Qwen·재학습이 필요하지 않습니다.
결과 ZIP은 Git 소스와 분리해 Release 첨부 파일로 배포합니다.

## 무엇을 할 수 있나요?

| 기능 | 사용 예 |
| --- | --- |
| 저장 결과 보기 | 방정식·방법별 물리장 및 계수 오차 비교 |
| 좌표 예측 | `poisson2d baseline seed 3234 좌표 [[0.25,0.65]] 예측해줘` |
| 자연어 질문 | `Poisson에서 방법별 물리장 오차를 비교해줘` |
| 기능 검사 | 자연어·API·저장 모델·재개·LHS 경계 검사 선택 |
| 추가 학습 | 방정식·방법 선택 → 설정 확인·변경 → 실행 허용 |

코드 정의는 분리된 접힘 셀로 제공됩니다. 설정과 결과는 하단 패널에 모여 있습니다.
선택 기능의 의존성만 사용하며, 기본 실행에서는 학습이나 언어 모델 다운로드를 시작하지 않습니다.

## 비교 대상

| 방정식 | 저장 시드 수 |
| --- | ---: |
| 2차원 파동 · 2차원 열전도 | 각각 3 |
| 2차원 포아송 · 버거스 · Kovasznay 순방향/역문제 · Taylor–Green · 2차원 Darcy · 반응–확산 | 각각 1 |

비교 방법은 **기본 PINN / 손실항 SA / 푸리에 특징 / 푸리에 특징 + 손실항 SA / 푸리에 특징 + 커리큘럼**입니다.
표는 저장된 실험 결과를 비교합니다. 세부 Adam·L-BFGS 이력과 설정은 결과 ZIP에 있습니다.

## 자연어·검색·추가 학습

- **Qwen3.5-0.8B**는 요청 의도를 제안하고, 제어기는 허용된 도구를 호출합니다.
- **multilingual-e5-small + BM25 + 가중 RRF**로 근거를 검색합니다. BM25 경량 경로도 지원합니다.
- 답변 수치는 저장 결과·계산·예측 도구에서 가져옵니다.
- 추가 학습은 명시적으로 허용한 경우에만 실행됩니다. 설정만 먼저 확인할 수 있습니다.
- 학습점은 기본 5천 Adam 업데이트마다 LHS로 교체하며 검증·평가점은 고정합니다.
- TensorFlow 버전 일치는 강제하지 않습니다. 완료 경계에서 ZIP 하나를 갱신하며 partial ZIP은 만들지 않습니다.

자세한 설정과 실행 환경은 [V3 사용 안내](v3/README.md)를 참고하세요.

## 확인된 검사

검사 결과·환경·실행 소스의 해시는 결과 ZIP의 `validation/`과 `provenance/`에 있습니다.

| 검사 | 확인 결과 |
| --- | --- |
| 자연어 30문항 + 별도 표현 15문항 · 내부 API | 통과 |
| Qwen + E5/BM25 경로 | 실행 기록 확인 |
| 저장 모델 65개 예측·참조 오차 재계산 | 통과 |
| 후보 새 학습·재개 / Wave 다음 업데이트 / LHS 경계 | 통과 |
| 비교 문구·조건부 패널·ZIP 보존 | 로컬 검사 통과 |

Qwen의 의도 판단 일치와 제어기 보정은 별도로 기록합니다.
자동 검사의 범위는 문법·소스 동기화·기능 동작입니다. 브라우저 화면과 공유 권한은 실행 환경별로 다를 수 있습니다.

## 개발 및 이전 버전

실행 노트북과 `v3/*.py`는 같은 평문 소스를 사용합니다. 개발용과 일반용의 학습 로직은 같습니다.

| 버전 | 파일 | 범위 |
| --- | --- | --- |
| V3 | [Online_PINN_PDE_Framework_V3.ipynb](outputs/Online_PINN_PDE_Framework_V3.ipynb) | 9개 방정식의 저장 비교·추론·도구 호출·선택적 추가 학습 |
| V2 | [Online_PINN_2D_Portfolio_V2.ipynb](outputs/Online_PINN_2D_Portfolio_V2.ipynb) | 목표기반 중단 코드·정적 검사; 전체 benchmark 재실행 전 |
| V1 | [Online_PINN_2D_Portfolio_V1_Executed.ipynb](outputs/Online_PINN_2D_Portfolio_V1_Executed.ipynb) | 24개 Wave 비교 실험과 실행 출력 보존 |

V1·V2 노트북과 과거 결과는 보존합니다. 모델·데이터·캐시·비밀정보는 Git에 넣지 않습니다.
개발 및 배포 확인 절차는 [검증 안내](docs/V3_RELEASE_CHECKLIST.md), 화면·공유 시험 방법은 [Colab 검증 절차](docs/V3_COLAB_TEST_PLAN.md)에 있습니다.

별도 라이선스는 아직 지정하지 않았습니다.
