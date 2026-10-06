# 자동 준비 / 결과 / 선택 동작. ZIP 안의 코드는 실행하지 않습니다.
try:
    from IPython.display import display, Markdown, FileLink
except ModuleNotFoundError:
    # Plain Python smoke tests; Colab already provides rich display.
    display = print
    Markdown = str
    FileLink = str
folder = Path(FOLDER)
if FOLDER.startswith("/content/drive/") and not folder.is_dir():
    from google.colab import drive

    drive.mount("/content/drive")
bundle = release_find(folder, ZIP_FILENAME)
scratch_base = Path("/content") if Path("/content").is_dir() else Path.cwd() / "tmp"
scratch_base.mkdir(exist_ok=True)
tempfile.tempdir = str(scratch_base)
RUN_DIR = scratch_base / ("v3_results_" + uuid.uuid4().hex[:8])
SUMMARY = release_open(bundle, RUN_DIR)
print(
    "저장 완료 결과:",
    SUMMARY["completed_trials"],
    "개 / 방정식",
    len(SUMMARY["equation_routes"]),
    "개 (ZIP 무결성 확인)",
)
verification_file = RUN_DIR / "validation/status.json"
if verification_file.exists():
    verification = json.loads(verification_file.read_text(encoding="utf-8"))
    if verification.get("deployment_runtime_verification") == "pending":
        print("배포 브랜치: 이 소스 수정본의 실행 검증은 예정입니다. 과거 검사 결과와 구분하세요.")
display(Markdown((RUN_DIR / "REPORT.md").read_text(encoding="utf-8")))
if ACTION == "predict":
    print(release_predictor(RUN_DIR)(EQUATION, ARM, SEED, POINTS))
elif ACTION == "agent":
    AGENT, APP = release_services(RUN_DIR, USE_LLM, DENSE_RAG)
    display(Markdown(AGENT.ask(QUESTION)["answer"]))
elif ACTION == "evaluate":
    print("목적: 자연어 30문항 + 별도 표현 15문항 / 도구·API 검사.")
    print(
        "추가 동작: 일회성 새 학습·재개 검사(6 trial Adam updates + 별도 wiring probes)."
        if CHECK_TRAINING
        else "추가 학습 검사 없음; benchmark 재학습 없음."
    )
    checks = release_evaluate(RUN_DIR, real_llm=USE_LLM, dense_rag=DENSE_RAG)
    if CHECK_TRAINING:
        checks["training_behavior"] = release_training_check(RUN_DIR, scratch_base)
        checks["passed"] = checks["passed"] and checks["training_behavior"]["passed"]
        checks["training_executed"] = True
        checks["trial_optimizer_updates_this_check"] = checks["training_behavior"][
            "trial_optimizer_updates"
        ]
        checks["additional_disposable_wiring_probes"] = True
    u_json(RUN_DIR / "validation" / ("actual_llm.json" if USE_LLM else "rules_api.json"), checks)
    OUTPUT_ZIP = folder / "Online_PINN_PDE_Framework_V3_Checked_Results.zip"
    u_pack(RUN_DIR, OUTPUT_ZIP)
    print("요청한 검사 통과:", checks["passed"], "/ 출력:", OUTPUT_ZIP.name)
    display(FileLink(str(OUTPUT_ZIP)))
elif ACTION == "train":
    SETTINGS = release_settings(EQUATION)
    SETTINGS.update(execution=EXECUTION, seed=SEED)
    SETTINGS.update(TRAIN_OVERRIDES)
    print("목적: 선택 방정식의 추가 학습. 완료 결과 ZIP은 변경하지 않습니다.")
    print(json.dumps(SETTINGS, ensure_ascii=False, indent=2))
    if not TRAINING_ENABLED:
        print("설정 표시만 했습니다. 확인 후 TRAINING_ENABLED=True로 변경하면 학습합니다.")
    else:
        PARENT = None
        if EXECUTION == "resume" and RESUME_ZIP_FILENAME:
            resume_zip = folder / RESUME_ZIP_FILENAME
            u_verify_zip(resume_zip, "unified_manifest.json")
            parent_root = scratch_base / ("v3_parent_" + uuid.uuid4().hex[:8])
            with zipfile.ZipFile(resume_zip) as z:
                parent_settings = json.loads(z.read("settings.json"))
                z.extractall(parent_root)
            PARENT = {"settings": parent_settings, "root": parent_root}
        elif EXECUTION == "resume":
            parent_settings = (
                json.loads((RUN_DIR / "settings.json").read_text(encoding="utf-8"))
                if (RUN_DIR / "settings.json").exists()
                else release_settings(EQUATION)
            )
            if EQUATION == "wave2d":
                parent_settings["field_target"] = 0.05
                parent_root = RUN_DIR / "wave_parent"
                with zipfile.ZipFile(RUN_DIR / "inputs/wave_parent.zip") as z:
                    u_members(z)
                    z.extractall(parent_root)
            else:
                parent_root = RUN_DIR
            PARENT = {"settings": parent_settings, "root": parent_root}
        dest = scratch_base / ("v3_additional_" + uuid.uuid4().hex[:8])
        saved = folder / (dest.name + "_latest.zip")
        result = release_training(RUN_DIR, SETTINGS, dest, PARENT, output_zip=saved)
        print("출력:", saved.name)
        display(FileLink(str(saved)))
elif ACTION != "results":
    raise ValueError("Unknown ACTION")
