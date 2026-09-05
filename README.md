# Online PINN PDE Framework

Physics-Informed Neural Network 기반 AI Engineering Portfolio의 버전 관리 저장소입니다.

## Versioned Notebooks

- `Online_PINN_2D_Executed_20260831_Original.ipynb`: 기존 실행 결과 보존본
- `Online_PINN_2D_Portfolio_V2.ipynb`: 검증된 2D Wave 연구·비교 기준본
- `Online_PINN_PDE_Framework_V3.ipynb`: 단계별 런타임 검증과 재개 가능한 연구 workflow 개발본

V3의 현재 구현 범위는 Wave2D입니다. Burgers, Kovasznay Forward/Inverse 및 Taylor-Green Vortex는 exact-solution 검증을 선행한 뒤 단계적으로 추가합니다.

## Artifact Policy

모델 가중치, TensorFlow checkpoint, 학습 데이터, FAISS index 및 단계별 ZIP은 Git에 저장하지 않습니다. 이러한 실행 artifact는 Google Drive에 보관하고 Git에는 설정, 검증 코드 및 checksum manifest만 기록합니다.

## Branch Policy

- `main`: 사람이 검토하고 검증한 공개 기준본
- `codex/v3-pde-framework`: V3 PDE framework 연구개발

자동 모델 승격은 허용하지 않습니다. 정식 benchmark는 pilot 결과를 검토하고 명시적으로 승인한 configuration만 실행합니다.
