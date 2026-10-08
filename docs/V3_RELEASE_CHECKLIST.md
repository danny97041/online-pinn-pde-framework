# V3 검증 범위와 유지관리

일반 실행에는 `Online_PINN_PDE_Framework_V3.ipynb`와 `V3_Results.zip` 두 파일을 사용합니다.
소스·문서는 Git으로, 저장 모델을 포함한 결과 ZIP은 Release 첨부 파일로 배포합니다.

## 완료된 검사

| 범위 | 기록 | 상태 |
| --- | --- | --- |
| Qwen + E5/BM25 자연어 30 + 별도 표현 15문항, 내부 API | `V3_Check_1b70d87f.zip` / `validation/runs/1b124c59.json` | 통과 |
| 규칙·API, 65개 저장 모델의 예측·고정 참조 오차 | `V3_Check_7a4c0860.zip` / `validation/runs/0bacdb8c.json` | 통과 |
| 후보 새 학습·재개, Wave Adam 복원, 4999/5000/5001 LHS 경계 | 동일 `0bacdb8c.json` | 통과 |
| 비교 문구·조건부 패널·타이머 제어 | `validation/form_local.json` | 로컬 통과 |
| 수치 코드·학습 일정·Qwen 의도 판단·검색 경로 보존 | `validation/source_current_local.json` | 소스 비교 통과 |
| ZIP 무결성, 65개 실험·9개 방정식, 두 검사 기록 병합 | `validation/submitted_colab_audit.json` 및 패키지 검사 | 통과 |

Qwen 실행 기록에는 42회 호출, 의도 일치 27회, 제어기 보정 15회가 있습니다.
45문항 시스템 통과를 Qwen 단독 정확도로 해석하지 않습니다.
재개 시험은 폐기용 상태에서 총 8회 업데이트했으며 완료 모델은 변경하지 않았습니다.

두 Colab 실행의 검사 결과는 `V3_Results.zip`에 함께 있습니다.
`provenance/validation_imports.json`에 입력 ZIP과 실행 소스의 해시를 기록합니다.
배포 ZIP에는 현재 소스만 포함하며, 검사 당시의 이전 소스 사본은 포함하지 않습니다.
실행 패널 검사는 로컬 환경의 기능 검사이며 Colab 화면 시험과는 구분합니다.
검사 기록은 실행 시점·소스·환경과 함께 보존하며 과거 검사를 현재 실행의 통과로 표시하지 않습니다.

## 환경별 시험

브라우저 화면·Drive 인증·공유 권한은 자동 코드 검사의 범위에 포함되지 않습니다.
시험 방법과 기대 동작은 [Colab 검증 절차](V3_COLAB_TEST_PLAN.md)에 정의되어 있습니다.
화면이나 권한 시험 결과는 환경·계정·실행 시점과 함께 별도로 기록합니다.

## 공개 배포

- 노트북·평문 소스·문서: Git 저장소의 `main` 및 버전 태그.
- 결과 ZIP·개발용 소스 ZIP: GitHub Release 첨부 파일로 배포. Git 이력에는 모델을 넣지 않음.
- README: Colab 버튼, 정확한 파일명, 다운로드 경로, 실제 완료된 범위 표시.
- CI: 구문·기본 조회·노트북 소스 동기화 검사. GPU 학습·실제 Colab 화면 검사는 아님.
- 가속 커널은 선택 사항이며 기본 실행의 필수 설치로 추가하지 않음.

공개 텍스트 점검 기록은 [V3_Publication_Review.json](../outputs/V3_Publication_Review.json)에 있습니다.
Git 버전·노트북 셀·출력·배포 문서의 점검 범위와 현재 소스 검사를 구분하여 기록합니다.
배포 소스는 현재 공개 문구를 사용합니다. 이전 소스 사본과 중복 배포 기록은 별도로 포함하지 않습니다.
실험 수치·저장 모델·검사 결과와 실행 당시 소스의 해시는 문구 정리와 구분합니다.

## 변경 후 유지관리

```bash
python tools/review_v3_source.py --baseline-zip V3_Check_7a4c0860.zip
python tools/sync_v3_notebook.py
python tools/test_v3_interface.py --bundle V3_Results.zip
python tools/test_v3_services.py --bundle V3_Results.zip --numerical
python tools/package_v3.py --evidence-zip V3_Check_7a4c0860.zip --validation-zip V3_Check_1b70d87f.zip
```

전체 모델·RAG 캐시·비밀정보를 Git에 올리지 않습니다. 결과 ZIP과 소스의 해시를 배포 파일에 함께 제공합니다.
