# V3 배포 기록과 체크리스트

일반 실행에는 `Online_PINN_PDE_Framework_V3.ipynb`와 `V3_Results.zip` 두 파일을 사용합니다.
V1·V2와 완료 모델은 보존합니다. 다음 기록은 전체 재학습을 요청하는 체크리스트가 아닙니다.

## 완료된 검사

| 범위 | 기록 | 상태 |
| --- | --- | --- |
| Qwen + E5/BM25 자연어 30 + 별도 표현 15문항, 내부 API | `V3_Check_1b70d87f.zip` / `validation/runs/1b124c59.json` | 통과 |
| 규칙·API, 65개 저장 모델의 예측·고정 참조 오차 | `V3_Check_7a4c0860.zip` / `validation/runs/0bacdb8c.json` | 통과 |
| 후보 새 학습·재개, Wave Adam 복원, 4999/5000/5001 LHS 경계 | 동일 `0bacdb8c.json` | 통과 |
| 사용자 확인: 60초 무입력 기본 진행, 65개 결과표, 포아송 자연어 비교 및 좌표 예측 | 제공된 Colab 출력 | 확인됨 |
| 최신 간결한 비교 문구·조건부 패널·타이머 제어 | `validation/form_local.json` | 로컬 통과 |
| 수치 코드·학습 일정·Qwen 의도 판단·검색 경로 보존 | `validation/source_current_local.json` | 소스 비교 통과 |
| ZIP 무결성, 65개 실험·9개 방정식, 두 검사 기록 병합 | `validation/submitted_colab_audit.json` 및 패키지 검사 | 통과 |

Qwen 실행 기록에는 42회 호출, 의도 일치 27회, 제어기 보정 15회가 있습니다.
45문항 시스템 통과를 Qwen 단독 정확도로 해석하지 않습니다.
재개 시험은 폐기용 상태에서 총 8회 업데이트했으며 완료 모델은 변경하지 않았습니다.

두 원본 ZIP의 기록과 검사 당시 소스가 `V3_Results.zip`에 함께 있습니다.
`provenance/validation_imports.json`은 원본 ZIP 해시, `provenance/validation_source/`는 검사 당시 노트북을 보존합니다.
최신 문구 수정은 별도 로컬 검사로 구분합니다. 과거 Windows 로컬 실패 기록도 삭제하지 않고 보존합니다.

## 남은 사용자 화면 확인

- [ ] 최신 `Online_PINN_PDE_Framework_V3.ipynb` + `V3_Results.zip`으로 Colab 기본 조회.
- [ ] 자연어 비교의 짧은 답변 및 좌표 예측 확인.
- [ ] 별도 계정에서 공개 뷰어·사본 실행·ZIP 접근 확인.

위 항목은 실제 사용자 환경에서 확인합니다. 자동 검사나 GitHub CI 통과로 대체 표시하지 않습니다.
전수 모델 검사·45문항·전체 학습은 같은 수정 범위에서 반복할 필요가 없습니다.

## 공개 배포

- 노트북·평문 소스·문서: 배포 브랜치 확인 후 `main` 반영.
- 결과 ZIP·개발용 소스 ZIP: GitHub Release 첨부 파일로 배포. Git 이력에는 모델을 넣지 않음.
- README: Colab 버튼, 정확한 파일명, 다운로드 경로, 실제 완료된 범위 표시.
- CI: 구문·기본 조회·노트북 소스 동기화 검사. GPU 학습·실제 Colab 화면 검사는 아님.
- 가속 커널은 선택 사항이며 기본 실행의 필수 설치로 추가하지 않음.

## 변경 후 유지관리

```bash
python tools/review_v3_source.py --baseline-zip V3_Check_7a4c0860.zip
python tools/sync_v3_notebook.py
python tools/test_v3_interface.py --bundle V3_Results.zip
python tools/package_v3.py --evidence-zip V3_Check_7a4c0860.zip --validation-zip V3_Check_1b70d87f.zip
```

전체 모델·RAG 캐시·비밀정보를 Git에 올리지 않습니다. 결과 ZIP과 소스의 해시를 배포 파일에 함께 제공합니다.
