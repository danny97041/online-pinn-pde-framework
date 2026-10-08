"""Deterministic technical reports and feature catalogues from stored evidence."""


def v3_generate_documents(root):
    root = Path(root)
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    target = root / "generated_documents"
    target.mkdir(exist_ok=True)
    lines = [
        "# V3 실험 기술보고서",
        "",
        f"저장 실험 {len(summary['rows'])}개 · 방정식 {len({r['equation'] for r in summary['rows']})}개.",
        "",
        "표의 수치는 저장된 선택 모델의 비교입니다. Adam과 후처리의 계산량은 따로 표시합니다.",
        "",
        "| 방정식 | 방법 | 시드 수 | 평균 L2 (%) | 표본 표준편차 (%p) | 10% 이하 | 평균 Adam 업데이트 | 평균 L-BFGS 목적함수 평가 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in summary["aggregate"]:
        std = r.get("field_l2_sample_std")
        lines.append(
            "| "
            + " | ".join(
                [
                    V3_EQUATION_LABELS.get(r["equation"], r["equation"]),
                    V3_ARM_LABELS.get(r["arm"], r["arm"]),
                    str(r["n"]),
                    f"{100 * r['field_l2_mean']:.4f}",
                    "해당 없음" if std is None else f"{100 * std:.4f}",
                    f"{r['field_10pct_count']}/{r['n']}",
                    f"{r.get('mean_adam_updates', 0):.1f}",
                    f"{r.get('mean_lbfgs_objective_evaluations', 0):.1f}",
                ]
            )
            + " |"
        )
    lines += [
        "",
        "## 평가 범위",
        "",
        "- 방정식별 평가 격자와 출력 성분 집계 방식은 개별 실험 기록을 따릅니다.",
        "- 1시드 결과에는 표본 표준편차를 부여하지 않습니다.",
        "- 저장된 Wave의 과거 중단 기준은 변경하지 않습니다.",
        "- 방법별 계산량이 다르므로 이 표만으로 동일 비용의 우월성을 판단하지 않습니다.",
        "",
        "## 근거 파일",
        "",
        "- `summary.json`: 실험별 결과와 집계",
        "- `comparison.csv`: 비교표",
        "- `evidence/`, `candidates/`: 개별 실험 기록",
        "- `validation/`: 실행 당시 검사 기록",
        "",
    ]
    report = "\n".join(lines)
    catalog = "\n".join(
        [
            "# V3 기능 목록",
            "",
            "| 기능 | 동작 | 선택 의존성 |",
            "| --- | --- | --- |",
            "| 저장 결과 조회 | 완료 모델의 비교표 | NumPy |",
            "| 기술보고서·기능 목록 | 저장 수치로 Markdown 생성, 검색 근거에 편입 | NumPy |",
            "| 결과 시각화 | 비교·시드 분산·해석해/예측/오차·학습 이력 | Matplotlib, 모델 지도에는 TensorFlow |",
            "| 모델 정보 | 방정식·좌표 영역·성분·체크포인트 | NumPy |",
            "| 수동 모델 선택·되돌리기 | 체크섬·모델 로드 확인 후 추론 모델 지정 | TensorFlow |",
            "| 자연어 도구 호출 | 허용된 비교·계산·검색·예측 도구 | 선택적으로 Qwen |",
            "| 하이브리드 검색 | E5/BM25, 가중 RRF, NumPy 또는 FAISS 내적 | E5, FAISS 선택 시 faiss-cpu |",
            "| API | 모델 정보·예측·검색 문맥·도구·명시적 등록 | FastAPI, HTTPX |",
            "| 추가 학습 | 선택 방정식·방법, 기본 1시드 또는 선택형 4시드 | TensorFlow, SciPy |",
            "",
            "## API",
            "",
            "`/health`, `/comparison`, `/report`, `/models`, `/model/info`, `/rag/search`, `/rag/context`,",
            "`/agent/tool`, `/agent/ask`, `/predict`, `/predict/batch`, `/predict/active`, `/registry`,",
            "`/registry/activate`, `/registry/rollback`.",
            "",
            "등록 API의 쓰기는 기본 비활성입니다. 외부 서버 공개 시 인증·접근 제어는 별도로 구성합니다.",
            "",
            "## 학습 프로필",
            "",
            "V3의 새 실험은 기본 L2 10% 목표와 고정 검증·평가점을 사용합니다.",
            "4시드 선택 시 3234·3235·3236·3237을 순차 실행하며 저장 결과를 다시 학습하지 않습니다.",
            "Supervised-only, 실행 데모, L1/L2 동시 5% 목표는 V4 범위입니다.",
            "",
        ]
    )
    documents = {"technical_report.md": report, "feature_catalog.md": catalog}
    for name, text in documents.items():
        (target / name).write_text(text, encoding="utf-8")
    u_json(
        target / "manifest.json",
        {
            "summary_sha256": u_sha(root / "summary.json"),
            "documents": {name: u_sha(target / name) for name in documents},
            "generation": "stored_evidence_templates",
            "llm_used": False,
            "training_executed": False,
        },
    )
    return {"documents": documents, "training_executed": False}
