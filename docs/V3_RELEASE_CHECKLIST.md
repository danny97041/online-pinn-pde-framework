# V3 개발자 배포 검증 체크리스트

이 문서는 유지관리자의 검증·승격 절차입니다. 일반 사용자는 노트북의 빠른 시작 안내를 따르면 됩니다.
저장된 과거 통과 기록만으로 수정본의 실행 검증을 통과 처리하지 않습니다.

## 배포 단계

1. 코드 설명·문법·파일 동기화와 결과 ZIP 무결성 확인.
2. 배포 브랜치에 반영. V1·V2와 main은 보존.
3. Colab에서 아래 실행 경로 검증.
4. 보완 검사와 실패·미실행 검토 후 승인된 커밋만 main으로 승격.
5. 공개 소개와 다운로드 자산 정리.

## 검증에 사용할 파일

- `Online_PINN_PDE_Framework_V3.ipynb`
- `Online_PINN_PDE_Framework_V3_Results.zip`

두 파일은 `MyDrive/PINN`에 둡니다. T4 또는 L4를 유지할 수 있으며 TensorFlow 버전 강제 일치는 필요하지 않습니다.

### 결과 조회

`ACTION="results"`, `ZIP_FILENAME="Online_PINN_PDE_Framework_V3_Results.zip"`로 모두 실행합니다.
9개 방정식·65개 실험과 비교표를 표시하고, TensorFlow·Qwen·원본 ZIP 선택·추가 학습을 요구하지 않아야 합니다.

### 도구·API·새 학습과 재개

```python
ACTION = "evaluate"
ZIP_FILENAME = "Online_PINN_PDE_Framework_V3_Results.zip"
USE_LLM = False
DENSE_RAG = False
CHECK_TRAINING = True
```

기본 30문항·별도 표현 15문항의 응답 상태, 도구 순서, 수치와 안전 처리를 검사합니다.
학습 검사는 `poisson2d/baseline`과 `kovasznay_inverse/ff_loss_sa`에서 새 학습 2회·재개 1회씩 수행합니다.
누적 횟수 3과 고정 검증·평가점의 해시 유지를 확인합니다.
총 6회의 시험용 Adam 업데이트와 별도 연결 검사이며 성능 실험은 아닙니다.
출력은 `Online_PINN_PDE_Framework_V3_Validation.zip`입니다.

### 실제 Qwen·하이브리드 검색

```python
ACTION = "evaluate"
ZIP_FILENAME = "Online_PINN_PDE_Framework_V3_Validation.zip"
USE_LLM = True
DENSE_RAG = True
CHECK_TRAINING = False
```

이전 검사 기록을 포함한 검증 ZIP을 입력으로 사용하고 같은 검증 ZIP을 갱신합니다.
시스템 성공, Qwen 판단 일치, 제어기의 보정을 구분하며 근거의 관련성도 검토합니다.
언어 모델을 실행하지 않은 기록을 실제 모델 통과로 표시하지 않습니다.

## 자동 평가 외 보완 항목

- 65개 저장 모델 전수 예측 및 저장 오차 재계산.
- Wave 학습 재개.
- 실제 5천 회 LHS 교체 경계.
- 실험 완료 경계 저장과 연결 중단 후 복원 범위.
- 실행한 소스와 배포 커밋의 일치 및 검사 환경 기록.

실패·미실행 항목과 과거 검사 범위를 명시하고 승인 후 승격합니다.
V3 10%는 산업 인증이 아니며 경계·초기조건 및 국소 고정밀 합격 판정은 별도 대상입니다.
