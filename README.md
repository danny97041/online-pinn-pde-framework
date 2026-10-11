# Online PINN PDE Framework

물리정보 신경망(PINN)의 학습 결과를 비교하고 저장 모델 추론, 검색, 자연어 도구 호출로 연결하는 프레임워크입니다.

[![Colab에서 실행](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/danny97041/online-pinn-pde-framework/blob/main/outputs/Online_PINN_PDE_Framework_V3.ipynb)
[![다운로드](https://img.shields.io/badge/다운로드-GitHub_Releases-2563eb)](https://github.com/danny97041/online-pinn-pde-framework/releases)

## 빠른 시작

일반 실행에는 두 파일이 필요합니다.

| 파일 | 역할 |
| --- | --- |
| [Online_PINN_PDE_Framework_V3.ipynb](outputs/Online_PINN_PDE_Framework_V3.ipynb) | Colab 실행 노트북 |
| `V3_Results.zip` | 저장 모델·비교 결과·검색 근거·검사 기록 |

1. [Releases](https://github.com/danny97041/online-pinn-pde-framework/releases)에서 두 파일을 내려받아 `MyDrive/PINN`에 둡니다.
2. Colab에서 노트북을 열고 **런타임 → 모두 실행**을 선택합니다.
3. 최초 설정을 적용하거나 60초 기다린 뒤, 하단 패널에서 기능을 선택합니다.

기본 기능은 **저장 결과 보기**입니다. 비교표 조회에는 GPU·TensorFlow·Qwen·재학습이 필요하지 않습니다.
자세한 실행 방법은 [Colab 사용법](v3/COLAB.md)에 있습니다.

## V3 기능

- 9개 방정식, 65개 저장 실험의 물리장·계수 오차 비교.
- 저장 모델의 좌표 예측과 자연어 질문.
- Qwen의 도구 호출 제안, E5·BM25 검색, 선택형 FAISS 검색.
- 방정식·방법·설정 선택에 따른 추가 학습과 체크포인트 재개.
- 기술보고서·기능 목록 생성, 결과 시각화.
- 기본 추론 모델 지정·복원과 FastAPI 인터페이스.

비교 방법은 기본 PINN, 손실항 SA, 푸리에 특징, 푸리에 특징 + 손실항 SA, 푸리에 특징 + 커리큘럼입니다.
저장된 Wave·Heat 실험은 각각 3시드, 나머지는 1시드입니다. 새 학습에는 선택형 4시드 실행도 제공합니다.

## 문서와 소스

| 문서 | 내용 |
| --- | --- |
| [V3 기능·실험 조건](v3/README.md) | 비교 대상, 학습 설정, 검색, API |
| [Colab 사용법](v3/COLAB.md) | 파일 배치, 설정, 실행, 재개 |
| [개발·검증 안내](docs/DEVELOPMENT.md) | 소스 구조, 노트북 동기화, 검사 범위 |

실행 노트북과 `v3/*.py`는 같은 평문 소스를 사용합니다.
모델 ZIP은 Release 첨부 파일로 배포하며, 검사 기록은 결과 ZIP의 `validation/`에서 확인할 수 있습니다.
모델·데이터·캐시·비밀정보는 Git에 포함하지 않습니다.

## 이전 버전

| 버전 | 노트북 | 내용 |
| --- | --- | --- |
| V2 | [Online_PINN_2D_Portfolio_V2.ipynb](https://github.com/danny97041/online-pinn-pde-framework/blob/main/outputs/Online_PINN_2D_Portfolio_V2.ipynb) | 목표 기반 중단을 포함한 Wave 실험 코드 |
| V1 | [Online_PINN_2D_Portfolio_V1_Executed.ipynb](https://github.com/danny97041/online-pinn-pde-framework/blob/main/outputs/Online_PINN_2D_Portfolio_V1_Executed.ipynb) | 24개 Wave 비교 실험과 실행 출력 |

별도 라이선스는 지정하지 않았습니다.
