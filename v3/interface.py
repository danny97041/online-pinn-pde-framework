"""Korean configuration panel and validated action routing, shared by Colab and tests.

Widgets are imported only when displaying the panel. Hidden options are dropped
by the request builder, not merely hidden by the browser.
"""

V3_EQUATION_LABELS = dict(
    zip(
        U_PROBLEMS,
        [
            "2차원 파동",
            "2차원 열전도",
            "2차원 포아송",
            "버거스",
            "Navier–Stokes · Kovasznay 순방향",
            "Navier–Stokes · Kovasznay 역문제",
            "Navier–Stokes · Taylor–Green",
            "2차원 Darcy",
            "반응–확산",
        ],
    )
)
V3_ARM_LABELS = dict(
    zip(
        U_ARMS,
        [
            "기본 PINN",
            "손실항 SA",
            "푸리에 특징",
            "푸리에 특징 + 손실항 SA",
            "푸리에 특징 + 커리큘럼",
        ],
    )
)
V3_ACTION_LABELS = {
    "results": "저장 결과 보기",
    "predict": "좌표 예측",
    "agent": "자연어 질문",
    "evaluate": "기능 검사",
    "train": "추가 학습",
    "documents": "보고서·기능 목록 생성",
    "visualize": "결과 시각화",
    "registry": "추론 모델 관리",
}


def release_model_description(result, equation):
    """Present model variables and selection state without exposing raw JSON."""
    from html import escape

    def safe(value):
        return escape(str(value)).replace("|", "&#124;").replace("\n", " ")

    def selection(record):
        if not record:
            return "미지정"
        return safe(V3_ARM_LABELS.get(record["arm"], record["arm"])) + f" / 시드 {record['seed']}"

    active = result.get("active")
    if isinstance(active, dict) and "arm" not in active:
        active = active.get(equation)
    lines = [
        "### 모델 정보",
        "",
        "| 항목 | 값 |",
        "| --- | --- |",
        f"| 방정식 | {safe(V3_EQUATION_LABELS[equation])} |",
        f"| 기본 추론 모델 | {selection(active)} |",
    ]
    info = result.get("model")
    if info:
        lines.extend([
            f"| 조회한 방법 | {safe(V3_ARM_LABELS[info['arm']])} |",
            f"| 시드 | {info['seed']} |",
            f"| 저장 물리장 L2 오차 | {info['field_l2'] * 100:.6g}% |",
            f"| 저장 모델 파일 | {'있음' if info['checkpoint_available'] else '없음'} |",
            "",
            "#### 입력·출력 변수",
            "",
            "| 구분 | 기호 | 의미 | 범위 |",
            "| --- | --- | --- | --- |",
        ])
        meanings = {"x": "첫 번째 공간 좌표", "y": "두 번째 공간 좌표", "t": "시간"}
        for symbol, bounds in zip(info["coordinates"], info["domain"]):
            lines.append(f"| 입력 | {safe(symbol)} | {meanings[symbol]} | {bounds[0]:.6g} ~ {bounds[1]:.6g} |")
        fluid = equation.startswith("kovasznay") or equation == "taylor_green"
        for symbol in info["outputs"]:
            meaning = {"u": "x방향 속도", "v": "y방향 속도", "p": "압력"}[symbol] if fluid else {
                "wave2d": "변위", "heat2d": "온도장", "poisson2d": "스칼라 해",
                "burgers": "속도장", "darcy2d": "수두·퍼텐셜", "reaction_diffusion": "반응·확산 상태장",
            }[equation]
            lines.append(f"| 출력 | {safe(symbol)} | {meaning} | 모델 예측값 |")
        if equation in {"wave2d", "kovasznay_inverse"}:
            symbol, meaning = ("λ₁", "파동 PDE 계수") if equation == "wave2d" else ("ν", "동점성 계수")
            lines.append(f"| 추정 대상 | {symbol} | {meaning} | 역문제에서 학습 |")
        elif "coefficient" in CANDIDATES.get(equation, {}):
            symbol = "ν" if fluid or equation == "burgers" else "D"
            lines.append(f"| 고정 계수 | {symbol} | PDE에 지정한 계수 | {CANDIDATES[equation]['coefficient']:.6g} |")
        lines += ["", "좌표 범위는 저장 모델의 입력 영역입니다. 단위가 별도 지정되지 않은 합성 실험 좌표이며 SI 단위를 자동 부여하지 않습니다."]
    lines += ["", "기본 추론 모델 지정은 자동 선택 예측 API의 기본값만 바꿉니다. 방법·시드를 직접 고르는 좌표 예측과 학습 결과는 바뀌지 않습니다."]
    return "\n".join(lines)


def release_zip_choices(folder, resume=False):
    """Classify local ZIP contents, without executing any archive code.

    Discovery is cheap; SHA/CRC validation still happens at open time.
    """
    choices = []
    for path in sorted(Path(folder).glob("*.zip")):
        try:
            with zipfile.ZipFile(path) as z:
                if (
                    z.getinfo("contract.json").file_size > 262144
                    or z.getinfo("summary.json").file_size > 16777216
                ):
                    continue
                contract = json.loads(z.read("contract.json"))
                if contract.get("distribution_format") != "v3-two-file-1":
                    continue
                if resume and "settings.json" not in z.namelist():
                    continue
                summary = json.loads(z.read("summary.json"))
                if not summary.get("rows"):
                    continue
            choices.append(path.name)
        except (OSError, ValueError, KeyError, zipfile.BadZipFile):
            continue
    return choices


def release_request(values):
    """Accept only active fields; stale hidden controls cannot start training."""
    action = values.get("action", "results")
    if action not in V3_ACTION_LABELS:
        raise ValueError("목록에서 사용할 기능을 선택하세요.")
    filename = values.get("zip", "")
    if (
        not filename
        or Path(filename).name != filename
        or "/" in filename
        or "\\" in filename
    ):
        raise ValueError("ZIP은 파일 목록에서 선택하세요.")
    request = {"action": action, "zip": filename}
    if action in {"predict", "train", "visualize", "registry"}:
        eq, arm, seed = values.get("equation"), values.get("arm"), values.get("seed")
        if (
            action == "visualize"
            and values.get("plot_kind", "comparison") == "comparison"
        ):
            arm, seed = "baseline", 3234
        if (
            action == "registry"
            and values.get("registry_operation", "inspect") == "rollback"
        ):
            arm, seed = "baseline", 3234
        if (
            action == "train"
            and values.get("execution", "new") == "new"
            and values.get("seed_mode") == "four"
        ):
            seed = V3_BENCHMARK_SEEDS[0]
        if (
            eq not in U_PROBLEMS
            or arm not in U_ARMS
            or type(seed) is not int
            or seed < 0
        ):
            raise ValueError("방정식·방법·시드 선택을 확인하세요.")
        request.update(equation=eq, arm=arm, seed=seed)
    if action == "predict":
        points = values.get("points")
        points = json.loads(points) if isinstance(points, str) else points
        a = np.asarray(points, float)
        if a.ndim != 2 or not 1 <= len(a) <= 1024 or not np.isfinite(a).all():
            raise ValueError("좌표는 유한한 2차원 JSON 배열로 입력하세요.")
        request["points"] = a.tolist()
    if action in {"agent", "evaluate"}:
        request.update(
            use_llm=bool(values.get("use_llm", False)),
            dense_rag=bool(values.get("dense_rag", False)),
        )
        if request["dense_rag"]:
            backend = values.get("dense_backend", "numpy")
            if backend not in {"numpy", "faiss"}:
                raise ValueError("검색 엔진을 선택하세요.")
            request["dense_backend"] = backend
    if action == "visualize":
        kind = values.get("plot_kind", "comparison")
        if kind not in {"comparison", "field", "history"}:
            raise ValueError("그림 종류를 선택하세요.")
        request["kind"] = kind
        if kind == "field":
            request.update(
                resolution=values.get("resolution", 32),
                time_fraction=values.get("time_fraction", 0.5),
                component=values.get("component", 0),
            )
    if action == "registry":
        operation = values.get("registry_operation", "inspect")
        if operation not in {"inspect", "activate", "rollback"}:
            raise ValueError("모델 관리 작업을 선택하세요.")
        request.update(
            operation=operation, approved=values.get("registry_approved") is True
        )
    if action == "agent":
        question = values.get("question", "").strip()
        if not question or len(question) > 2000:
            raise ValueError("질문은 1~2000자로 입력하세요.")
        request["question"] = question
    if action == "evaluate":
        custom = values.get("custom_questions", "")
        if not isinstance(custom, str):
            raise ValueError("추가 질문은 한 줄에 한 문항씩 입력하세요.")
        questions = [q.strip() for q in custom.splitlines() if q.strip()]
        if len(questions) > 10 or any(len(q) > 2000 for q in questions):
            raise ValueError("추가 질문은 최대 10문항, 문항당 2000자입니다.")
        request.update(
            check_models=bool(values.get("check_models", False)),
            check_training=bool(values.get("check_training", False)),
            custom_questions=questions,
        )
    if action == "train":
        s = release_settings(request["equation"])
        execution = values.get("execution", "new")
        if execution not in {"new", "resume"}:
            raise ValueError("새 학습 또는 재개를 선택하세요.")
        overrides = dict(values.get("overrides", {}))
        if set(overrides) - set(s):
            raise ValueError(
                "알 수 없는 학습 설정: " + str(sorted(set(overrides) - set(s)))
            )
        # Architecture/data/SA settings belong to new runs only. Resume inherits
        # immutable fields from its saved parent inside release_execute.
        allowed = {
            "cap",
            "field_target",
            "stop_first",
            "stop_every",
            "stop_consecutive",
            "lbfgs_maxiter",
        }
        if execution == "new":
            allowed |= set(s) - {"equation", "execution", "arms", "seed"}
        if request["arm"] not in {"loss_sa", "ff_loss_sa"}:
            allowed -= {"sa_every", "sa_ema", "sa_bounds"}
        if request["arm"] not in {"ff", "ff_loss_sa", "ff_curriculum"}:
            allowed -= {"ff_features_per_bank", "ff_scales"}
        if request["equation"] == "wave2d":
            allowed -= {
                "network_width",
                "network_depth",
                "collocation_count",
                "data_count",
                "initial_count",
                "boundary_count_per_face",
                "ff_features_per_bank",
                "ff_scales",
                "sa_every",
                "sa_ema",
                "sa_bounds",
            }
        if request["equation"] not in {"wave2d", "kovasznay_inverse"}:
            allowed -= {"initial_coefficient_ratio", "coefficient_ratios"}
        if (
            request["equation"] != "wave2d"
            and CANDIDATES[request["equation"]]["time_axis"] is None
        ):
            allowed -= {"initial_count"}
        request.update(
            execution=execution,
            overrides={k: v for k, v in overrides.items() if k in allowed},
            training_enabled=bool(values.get("training_enabled", False)),
        )
        if execution == "new":
            seed_mode = values.get("seed_mode", "single")
            if seed_mode not in {"single", "four"}:
                raise ValueError("시드 실행 범위를 선택하세요.")
            request.update(
                seed_mode=seed_mode, all_methods=bool(values.get("all_methods", False))
            )
        if execution == "resume":
            name = values.get("resume_zip", "")
            if name and (Path(name).name != name or "/" in name or "\\" in name):
                raise ValueError("재개 ZIP은 파일 목록에서 선택하세요.")
            request["resume_zip"] = name
    return request


def release_execute(request, folder, scratch_base):
    """One action; inputs remain unchanged and exceptions stay visible."""
    from IPython.display import display, Markdown, FileLink

    folder, scratch_base = Path(folder), Path(scratch_base)
    bundle = release_find(folder, request["zip"])
    root = scratch_base / ("v3_view_" + uuid.uuid4().hex[:8])
    summary = release_open(bundle, root)
    action = request["action"]
    print(
        "목적:",
        V3_ACTION_LABELS[action],
        "/ 저장 실험",
        summary["completed_trials"],
        "개",
    )
    if action == "results":
        # Regenerate presentation in this disposable extracted copy, not the input ZIP.
        u_report_markdown(root, summary["aggregate"])
        display(Markdown((root / "REPORT.md").read_text(encoding="utf-8")))
    elif action == "predict":
        print(
            release_predictor(root)(
                request["equation"], request["arm"], request["seed"], request["points"]
            )
        )
    elif action == "agent":
        agent, _ = release_services(
            root,
            request["use_llm"],
            request["dense_rag"],
            request.get("dense_backend", "numpy"),
        )
        result = agent.ask(request["question"])
        display(Markdown(result["answer"]))
        if request["use_llm"]:
            print("Qwen 호출:", "실행" if result.get("llm_used") else "미실행")
    elif action in {"documents", "visualize", "registry"}:
        changed = action != "registry" or request["operation"] != "inspect"
        if action == "documents":
            result = v3_generate_documents(root)
            display(Markdown(result["documents"]["technical_report.md"]))
            UnifiedRAG(root)  # Generated documents join the searchable corpus.
        elif action == "visualize":
            from IPython.display import Image

            result = v3_visualize(
                root,
                request["equation"],
                request["arm"],
                request["seed"],
                **{
                    k: request[k]
                    for k in ["kind", "resolution", "time_fraction", "component"]
                    if k in request
                }
            )
            for name in result["files"]:
                if name.endswith(".png"):
                    display(Image(filename=str(root / name)))
        else:
            registry = V3ModelRegistry(root, release_predictor(root))
            if request["operation"] == "inspect":
                result = {
                    "model": registry.info(
                        request["equation"],
                        request["arm"],
                        request["seed"],
                        checksum=True,
                    ),
                    "active": registry.state["active"],
                }
            elif request["operation"] == "activate":
                record = registry.activate(
                    request["equation"],
                    request["arm"],
                    request["seed"],
                    request["approved"],
                )
                result = {"active": record}
            else:
                result = {
                    "active": registry.rollback(
                        request["equation"], request["approved"]
                    )
                }
            display(Markdown(release_model_description(result, request["equation"])))
        if changed:
            filename = {
                "documents": "V3_Documents.zip",
                "visualize": "V3_Plots.zip",
                "registry": "V3_Registry.zip",
            }[action]
            output = folder / filename
            if output.exists():
                output = folder / (
                    Path(filename).stem + "_" + uuid.uuid4().hex[:8] + ".zip"
                )
            u_pack(root, output)
            print("저장:", output.name)
            display(FileLink(str(output)))
        return result
    elif action == "evaluate":
        output = folder / "V3_Check.zip"
        if output.exists() or output.resolve() == bundle.resolve():
            # Preserve any previous output, not only the selected input.
            output = folder / ("V3_Check_" + uuid.uuid4().hex[:8] + ".zip")
        checks = release_inspection(root, scratch_base, request)
        u_pack(root, output)
        print("요청한 검사 통과:", checks["passed"], "/ 출력:", output.name)
        display(FileLink(str(output)))
    else:
        parent = None
        s = release_settings(request["equation"])
        if request["execution"] == "resume":
            name = request["resume_zip"]
            parent_root = root
            if name:
                parent_root = scratch_base / ("v3_parent_" + uuid.uuid4().hex[:8])
                release_open(release_find(folder, name), parent_root)
            stored_summary = json.loads(
                (parent_root / "summary.json").read_text(encoding="utf-8")
            )
            if not any(
                (r["equation"], r["arm"], r["seed"])
                == (request["equation"], request["arm"], request["seed"])
                for r in stored_summary["rows"]
            ):
                raise ValueError(
                    "선택 방정식·방법·시드의 완료 지점이 이 ZIP에 없습니다."
                )
            settings_path = parent_root / "settings.json"
            if settings_path.exists():
                s = json.loads(settings_path.read_text(encoding="utf-8"))
                benchmark_path = parent_root / "benchmark_settings.json"
                if benchmark_path.exists():
                    seeds = json.loads(benchmark_path.read_text(encoding="utf-8"))[
                        "seeds"
                    ]
                    if request["seed"] not in seeds:
                        raise ValueError("재개 ZIP에 없는 시드입니다.")
                    s["seed"] = request["seed"]
                if (
                    s["equation"] != request["equation"]
                    or s["seed"] != request["seed"]
                    or request["arm"] not in s["arms"]
                ):
                    raise ValueError("재개 ZIP과 선택 방정식·방법·시드가 다릅니다.")
            elif (
                request["equation"] == "wave2d"
                and (parent_root / "inputs/wave_parent.zip").exists()
            ):
                s["field_target"] = 0.05
                extracted = scratch_base / ("v3_wave_parent_" + uuid.uuid4().hex[:8])
                with zipfile.ZipFile(parent_root / "inputs/wave_parent.zip") as z:
                    u_members(z)
                    z.extractall(extracted)
                parent_root = extracted
            elif request["equation"] == "wave2d":
                raise ValueError("이 ZIP에는 Wave Adam 재개 상태가 없습니다.")
            parent = {"root": parent_root, "settings": copy.deepcopy(s)}
        s.update(
            execution=request["execution"], seed=request["seed"], arms=[request["arm"]]
        )
        if request.get("all_methods") and request["execution"] == "new":
            s["arms"] = list(U_ARMS)
        s.update(request["overrides"])
        s = release_validate_settings(s, parent["settings"] if parent else None)
        display(
            Markdown(
                "```json\n" + json.dumps(s, ensure_ascii=False, indent=2) + "\n```"
            )
        )
        seeds = (
            V3_BENCHMARK_SEEDS if request.get("seed_mode") == "four" else [s["seed"]]
        )
        print(
            "실행 범위:",
            len(seeds) * len(s["arms"]),
            "개 실험 / 시드",
            seeds,
            "/ 방법",
            s["arms"],
        )
        if not request["training_enabled"]:
            print("설정 확인만 완료했습니다. ‘학습 실행 허용’을 켜야 학습합니다.")
            return {"status": "settings_only", "settings": s}
        run_id = uuid.uuid4().hex[:8]
        destination = scratch_base / ("v3_train_" + run_id)
        output = folder / ("V3_Train_" + s["equation"] + "_" + run_id + ".zip")
        if request.get("seed_mode") == "four":
            release_training_benchmark(root, s, destination, output)
        else:
            release_training(root, s, destination, parent, output_zip=output)
        print("완료된 실험 경계 저장:", output.name)
        display(FileLink(str(output)))
    return summary


def release_faiss_status():
    """Check the optional binary and a tiny index without downloading packages."""
    import importlib

    try:
        importlib.invalidate_caches()
        faiss = importlib.import_module("faiss")
        index = faiss.IndexFlatIP(2)
        return {"available": index.d == 2, "version": getattr(faiss, "__version__", "")}
    except Exception as exc:
        return {"available": False, "error_type": type(exc).__name__}


def release_install_faiss(runner=None):
    """Install only the fixed optional package after an explicit button click."""
    import subprocess
    import sys

    runner = runner or subprocess.run
    result = runner(
        [
            sys.executable, "-m", "pip", "install", "--quiet",
            "--disable-pip-version-check", "--no-input", "--no-deps",
            "--only-binary=:all:", "faiss-cpu",
        ],
        capture_output=True,
        text=True,
        timeout=240,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(f"faiss-cpu 설치 실패 (종료 코드 {result.returncode}).")
    return {"package": "faiss-cpu", "pip_completed": True}


def release_panel(
    folder, scratch_base, storage_status=None, initial_action="results", show=True
):
    """Compact conditional dropdown settings, with results directly underneath."""
    import ipywidgets as w
    from IPython.display import display

    style = {"description_width": "145px"}

    def control(cls, label, **kwargs):
        return cls(
            description=label, style=style, layout=w.Layout(width="100%"), **kwargs
        )

    controls = {
        "folder": control(w.Text, "파일이 있는 폴더", value=str(folder)),
        "action": control(
            w.Dropdown,
            "사용할 기능",
            options=[(v, k) for k, v in V3_ACTION_LABELS.items()],
            value=initial_action,
        ),
        "zip": control(
            w.Dropdown, "통합 결과 ZIP", options=release_zip_choices(folder)
        ),
        "equation": control(
            w.Dropdown,
            "방정식",
            options=[(v, k) for k, v in V3_EQUATION_LABELS.items()],
            value="poisson2d",
        ),
        "arm": control(
            w.Dropdown, "비교 방법", options=[(v, k) for k, v in V3_ARM_LABELS.items()]
        ),
        "seed": control(w.IntText, "시드", value=3234),
        "points": control(w.Textarea, "예측 좌표 (JSON)", value="[[0.25, 0.65]]"),
        "question": control(
            w.Textarea,
            "직접 입력할 질문",
            value="Poisson에서 방법별 물리장 오차를 비교해줘",
            placeholder="결과 비교, 좌표 예측, 계산, 근거 검색 또는 학습 설정 초안을 자유롭게 질문하세요.",
        ),
        "use_llm": control(w.Checkbox, "Qwen 사용", value=False),
        "dense_rag": control(w.Checkbox, "E5 + BM25 검색", value=False),
        "dense_backend": control(
            w.Dropdown,
            "임베딩 검색 엔진",
            options=[("NumPy (기본)", "numpy"), ("FAISS (선택 설치)", "faiss")],
        ),
        "plot_kind": control(
            w.Dropdown,
            "그림 종류",
            options=[
                ("방법 비교·방정식 히트맵", "comparison"),
                ("해석해·예측·오차 지도", "field"),
                ("저장 학습 이력", "history"),
            ],
        ),
        "resolution": control(w.IntSlider, "지도 격자 크기", value=32, min=8, max=64),
        "time_fraction": control(
            w.FloatSlider, "시간 위치 (0~1)", value=0.5, min=0, max=1, step=0.05
        ),
        "component": control(w.IntText, "출력 성분 (0부터)", value=0),
        "registry_operation": control(
            w.Dropdown,
            "모델 관리 작업",
            options=[
                ("모델 정보 조회", "inspect"),
                ("기본 추론 모델 지정", "activate"),
                ("직전 기본 모델로 복원", "rollback"),
            ],
        ),
        "registry_approved": control(w.Checkbox, "선택 변경 승인", value=False),
        "seed_mode": control(
            w.Dropdown,
            "시드 실행 범위",
            options=[("선택한 1시드 (기본)", "single"), ("4시드: 3234~3237", "four")],
        ),
        "all_methods": control(w.Checkbox, "5개 방법 모두 실행", value=False),
        "check_models": control(w.Checkbox, "저장 모델 전수 검사", value=False),
        "check_training": control(w.Checkbox, "재개·LHS 경계 검사", value=False),
        "custom_questions": control(
            w.Textarea,
            "추가 시험 질문",
            value="",
            placeholder="선택 사항 · 한 줄에 한 문항, 최대 10문항. 답변 의미는 직접 확인해야 합니다.",
        ),
        "execution": control(
            w.Dropdown,
            "학습 시작 방식",
            options=[("새 학습", "new"), ("완료 지점에서 재개", "resume")],
        ),
        "resume_zip": control(
            w.Dropdown,
            "재개 ZIP",
            options=[("선택한 통합 결과에서 재개", "")]
            + [(n, n) for n in release_zip_choices(folder, resume=True)],
        ),
        "training_enabled": control(w.Checkbox, "학습 실행 허용", value=False),
    }
    labels = {
        "cap": "총 Adam 업데이트 상한",
        "field_target": "목표 L2 (0.10 = 10%)",
        "stop_first": "조기중단 허용 시작",
        "stop_every": "중단 검사 주기",
        "stop_consecutive": "연속 통과 횟수",
        "lhs_period": "학습점 LHS 교체 주기",
        "base_lr": "신경망 기준 학습률",
        "network_width": "신경망 너비",
        "network_depth": "신경망 깊이",
        "collocation_count": "PDE 학습점 수",
        "data_count": "관측 학습점 수",
        "initial_count": "초기조건 학습점 수",
        "boundary_count_per_face": "면별 경계 학습점 수",
        "sa_every": "손실항 SA 갱신 주기",
        "sa_ema": "SA 이동평균 계수",
        "ff_features_per_bank": "은행별 FF 개수",
        "lbfgs_maxiter": "L-BFGS 반복 상한",
        "initial_coefficient_ratio": "계수 초기 비율",
    }
    defaults = release_settings("poisson2d")
    settings_controls = {
        key: control(
            w.IntText if type(defaults[key]) is int else w.FloatText,
            label,
            value=defaults[key],
        )
        for key, label in labels.items()
    }
    arrays_controls = {
        key: control(w.Text, label, value=json.dumps(defaults[key]))
        for key, label in {
            "network_ratios": "신경망 LR 비율 3개",
            "coefficient_ratios": "계수 LR 비율 3개",
            "lr_knots": "LR 교차 시점 3개",
            "ff_scales": "FF 척도 (JSON)",
            "sa_bounds": "SA 하한·상한 (JSON)",
        }.items()
    }
    controls["zip"].value = (
        RESULTS_FILENAME
        if RESULTS_FILENAME in controls["zip"].options
        else controls["zip"].value
    )
    output = w.Output(layout=w.Layout(width="100%"))
    notice = w.HTML()
    run = w.Button(description="선택한 기능 실행", button_style="primary")
    refresh = w.Button(description="ZIP 목록 새로고침", icon="refresh")
    connect = w.Button(description="Drive 연결", icon="cloud")
    local = w.Button(description="로컬 ZIP 사용", icon="folder-open")
    storage_notice = w.HTML()
    faiss_notice = w.HTML()
    install_faiss = w.Button(description="FAISS 설치", icon="download")
    faiss_box = w.VBox([faiss_notice, install_faiss])
    run.icon = "play"
    run.layout.width = "190px"
    advanced = w.Accordion(
        children=[
            w.GridBox(
                list(settings_controls.values()) + list(arrays_controls.values()),
                layout=w.Layout(
                    grid_template_columns="repeat(auto-fit, minmax(320px, 1fr))",
                    grid_gap="8px 16px",
                ),
            )
        ]
    )
    advanced.set_title(0, "상세 학습 설정")
    advanced.selected_index = None
    folder_options = w.Accordion(
        children=[
            w.VBox(
                [
                    controls["folder"],
                    storage_notice,
                    w.HBox(
                        [connect, local],
                        layout=w.Layout(flex_flow="row wrap", gap="8px"),
                    ),
                ]
            )
        ]
    )
    folder_options.set_title(0, "저장소 · 파일 위치")
    folder_options.selected_index = None
    grid_layout = lambda: w.Layout(
        grid_template_columns="repeat(auto-fit, minmax(320px, 1fr))",
        grid_gap="8px 16px",
        width="100%",
    )
    header = w.HTML(
        "<h3 style='margin-bottom:6px'>설정</h3>"
        "<p style='margin-top:0'>선택 기능에 해당하는 설정만 표시됩니다. 기본 기능: 저장 결과 조회.</p>"
    )
    panel = w.VBox(
        [
            header,
            w.GridBox([controls["action"], controls["zip"]], layout=grid_layout()),
            folder_options,
            w.GridBox(
                [
                    v
                    for k, v in controls.items()
                    if k not in {"action", "zip", "folder"}
                ],
                layout=grid_layout(),
            ),
            faiss_box,
            advanced,
            notice,
            w.HBox([run, refresh], layout=w.Layout(flex_flow="row wrap", gap="8px")),
        ],
        layout=w.Layout(width="100%", max_width="1150px"),
    )
    results_header = w.HTML("<hr><h3>결과 · 실행 로그</h3>")
    dashboard = w.VBox([panel, results_header, output], layout=w.Layout(width="100%"))
    state = {
        "last_result": None,
        "last_error": None,
        "storage": dict(storage_status or {"status": "local"}),
        "busy": False,
        "faiss": None,
    }
    visible = set()

    def needs_faiss():
        return (
            controls["action"].value in {"agent", "evaluate"}
            and controls["dense_rag"].value
            and controls["dense_backend"].value == "faiss"
        )

    def update(_=None):
        nonlocal visible
        if _ is not None:
            # A changed action/model/parent needs a fresh training opt-in.
            controls["training_enabled"].value = False
            controls["registry_approved"].value = False
        a, arm, eq = (
            controls["action"].value,
            controls["arm"].value,
            controls["equation"].value,
        )
        visible = {"action", "zip", "folder"}
        if a in {"predict", "train", "visualize", "registry"}:
            visible |= {"equation", "arm", "seed"}
        if a == "predict":
            visible |= {"points"}
        if a in {"agent", "evaluate"}:
            visible |= {"use_llm", "dense_rag"}
            if controls["dense_rag"].value:
                visible.add("dense_backend")
        if a == "visualize":
            visible.add("plot_kind")
            if controls["plot_kind"].value == "comparison":
                visible -= {"arm", "seed"}
            if controls["plot_kind"].value == "field":
                visible |= {"resolution", "component"}
                if eq != "burgers" and (
                    eq == "wave2d" or len(CANDIDATES[eq]["domain"]) == 3
                ):
                    visible.add("time_fraction")
        if a == "registry":
            visible.add("registry_operation")
            if controls["registry_operation"].value == "rollback":
                visible -= {"arm", "seed"}
            if controls["registry_operation"].value != "inspect":
                visible.add("registry_approved")
        if a == "agent":
            visible |= {"question"}
        if a == "evaluate":
            visible |= {"check_models", "check_training", "custom_questions"}
        if a == "train":
            visible |= {"execution", "training_enabled"}
            if controls["execution"].value == "resume":
                visible |= {"resume_zip"}
            else:
                visible |= {"seed_mode", "all_methods"}
                if controls["seed_mode"].value == "four":
                    visible.discard("seed")
        for key, widget in controls.items():
            widget.layout.display = "" if key in visible else "none"
            widget.disabled = key not in visible
        active = set(labels) | set(arrays_controls) if a == "train" else set()
        if controls["execution"].value == "resume":
            active &= {
                "cap",
                "field_target",
                "stop_first",
                "stop_every",
                "stop_consecutive",
                "lbfgs_maxiter",
            }
        if arm not in {"loss_sa", "ff_loss_sa"}:
            active -= {"sa_every", "sa_ema", "sa_bounds"}
        if arm not in {"ff", "ff_loss_sa", "ff_curriculum"}:
            active -= {"ff_features_per_bank", "ff_scales"}
        if eq == "wave2d":
            active -= {
                "network_width",
                "network_depth",
                "collocation_count",
                "data_count",
                "initial_count",
                "boundary_count_per_face",
                "sa_every",
                "sa_ema",
                "sa_bounds",
                "ff_features_per_bank",
                "ff_scales",
            }
        if eq not in {"wave2d", "kovasznay_inverse"}:
            active -= {"initial_coefficient_ratio", "coefficient_ratios"}
        if eq != "wave2d" and CANDIDATES[eq]["time_axis"] is None:
            active -= {"initial_count"}
        for key, widget in {**settings_controls, **arrays_controls}.items():
            widget.disabled = key not in active
            widget.layout.display = "" if key in active else "none"
        advanced.layout.display = "" if a == "train" else "none"
        notice.value = {
            "agent": "<p>직접 작성한 질문을 한국어 문장으로 답합니다. 수치는 저장 결과·계산 도구에서 가져오며 자료가 없으면 추가 질문을 요청합니다. 학습·파일 삭제는 질문으로 실행하지 않습니다.</p>",
            "evaluate": "<p>자연어 30문항 + 별도 표현 15문항 / API 검사. 추가 질문은 답변·안전한 도구 호출을 기록하며 의미 정확도는 수동 확인합니다. 전수 검사와 재개 검사는 선택 사항입니다.</p>",
            "train": "<p>완료 결과는 변경하지 않습니다. 재개 시 모델·샘플링 설정은 저장값을 상속하며 미완료 실험은 세션 종료 시 소실될 수 있습니다.</p>",
            "predict": "<p>좌표 순서와 영역: 선택 방정식의 x/y/t 구성에 맞추세요. 잘못된 차원·영역은 실행 시 거절됩니다.</p>",
            "documents": "<p>저장 결과에서 기술보고서와 기능 목록을 생성하고 검색 근거에 포함합니다. 재학습·Qwen 사용 없음.</p>",
            "visualize": "<p>비교표·시드 분산은 저장 수치, 지도는 선택 모델의 새 격자 예측입니다. 시간 선택은 ‘해석해·예측·오차 지도’에서 x/y/t 방정식을 고를 때만 표시됩니다. 버거스 지도는 x–t 전체 영역을 표시합니다. 재학습 없음.</p>",
            "registry": "<p>기본 모델은 /predict/active API가 방법·시드를 자동 선택할 때 사용합니다. 예: 기본 PINN 지정 → 푸리에 특징 지정 → 직전 모델 복원은 기본 PINN으로 돌아갑니다. 직접 방법을 지정하는 좌표 예측은 그대로입니다. 변경 승인 후 생성한 ZIP을 다음 입력으로 선택해야 선택 이력이 이어집니다. 원본 ZIP과 학습 모델은 변경하지 않습니다.</p>",
        }.get(a, "")
        required = needs_faiss()
        if required:
            state["faiss"] = release_faiss_status()
            ready = state["faiss"]["available"]
            faiss_notice.value = (
                "<p>FAISS 사용 가능. 별도 설치 없이 실행할 수 있습니다.</p>"
                if ready else
                "<p>FAISS가 설치되지 않았거나 사용할 수 없습니다. 아래 ‘FAISS 설치’를 누르면 "
                "현재 런타임에 faiss-cpu만 설치합니다. TensorFlow·NumPy는 변경하지 않습니다. "
                "설치 후 ‘선택한 기능 실행’을 다시 누르세요.</p>"
            )
            install_faiss.layout.display = "none" if ready else ""
            install_faiss.disabled = ready or state["busy"]
        else:
            install_faiss.disabled = True
        faiss_box.layout.display = "" if required else "none"
        run.disabled = state["busy"] or (required and not state["faiss"]["available"])
        if state["busy"]:
            for widget in [
                *controls.values(), *settings_controls.values(), *arrays_controls.values()
            ]:
                widget.disabled = True

    def install_faiss_clicked(_=None):
        if state["busy"] or not needs_faiss():
            return
        state["faiss"] = release_faiss_status()
        if state["faiss"]["available"]:
            update()
            return
        state["busy"] = True
        state["last_result"] = None
        state["last_error"] = None
        update()
        refresh.disabled = connect.disabled = local.disabled = True
        with output:
            output.clear_output(wait=True)
            print("FAISS 설치 중… 완료되면 선택한 기능을 다시 실행하세요.")
            try:
                release_install_faiss()
                state["faiss"] = release_faiss_status()
                if not state["faiss"]["available"]:
                    raise RuntimeError(
                        "설치는 완료됐지만 FAISS 로드 검사에 실패했습니다. "
                        "런타임을 다시 시작하고 모두 실행하거나 NumPy 검색을 선택하세요."
                    )
                print("FAISS 설치 완료. ‘선택한 기능 실행’을 누르세요.")
            except Exception as exc:
                state["last_error"] = {"type": type(exc).__name__, "message": str(exc)}
                print("FAISS 설치를 완료하지 못했습니다:", str(exc))
                print("NumPy (기본) 검색은 별도 FAISS 설치 없이 사용할 수 있습니다.")
            finally:
                state["busy"] = False
                refresh.disabled = connect.disabled = local.disabled = False
                update()


    def refresh_choices(_):
        names = release_zip_choices(controls["folder"].value)
        previous = controls["zip"].value
        controls["zip"].options = names
        if previous in names:
            controls["zip"].value = previous
        elif RESULTS_FILENAME in names:
            controls["zip"].value = RESULTS_FILENAME
        else:
            controls["zip"].value = names[0] if len(names) == 1 else None
        controls["resume_zip"].options = [("선택한 통합 결과에서 재개", "")] + [
            (n, n) for n in release_zip_choices(controls["folder"].value, True)
        ]

    def show_storage_status():
        status = state["storage"]["status"]
        messages = {
            "connected": "Drive 연결됨. ZIP 위치: MyDrive/PINN.",
            "local": "로컬 저장소. Colab 로컬 파일과 출력은 세션 종료 시 유실될 수 있습니다.",
            "authentication_failed": "Drive 인증 실패. 재연결 또는 로컬 ZIP 사용이 필요합니다.",
            "connection_failed": "Drive 연결 실패. 저장소 상태 확인이 필요합니다.",
            "unavailable": "현재 환경에서는 Colab Drive 연결을 사용할 수 없습니다.",
            "not_ready": "Drive 연결 후 MyDrive 경로가 확인되지 않았습니다.",
            "not_connected": "Drive 미연결. ‘Drive 연결’ 또는 ‘로컬 ZIP 사용’을 선택하세요.",
        }
        storage_notice.value = (
            "<p>" + messages.get(status, "저장소 연결 미완료.") + "</p>"
        )
        needs_file = status not in {"connected", "local"} or not controls["zip"].value
        if needs_file:
            folder_options.selected_index = 0
        lines = [messages.get(status, "저장소 연결 미완료.")]
        if state["storage"].get("error_type"):
            lines.append("연결 오류 유형: " + state["storage"]["error_type"])
        if needs_file:
            lines.extend(
                [
                    "선택 ZIP 없음. 파일 위치: " + controls["folder"].value,
                    "대체 경로: Colab 왼쪽 파일 패널에 V3_Results.zip 업로드 → ‘로컬 ZIP 사용’.",
                ]
            )
        if status != "connected":
            lines.append("로컬 출력 ZIP은 세션 종료 전에 다운로드가 필요합니다.")
        output.outputs = ()
        output.append_stdout("\n".join(lines) + "\n")

    def connect_drive(_=None, target_folder=None):
        connect.disabled = local.disabled = run.disabled = refresh.disabled = True
        state["last_result"] = None
        state["last_error"] = None
        try:
            state["storage"] = release_drive_status()
            if state["storage"]["status"] == "connected":
                controls["folder"].value = (
                    target_folder or "/content/drive/MyDrive/PINN"
                )
                refresh_choices(None)
            show_storage_status()
        finally:
            connect.disabled = local.disabled = run.disabled = refresh.disabled = False
            update()

    def use_local(_=None):
        state["last_result"] = None
        state["last_error"] = None
        controls["folder"].value = str(
            Path("/content") if Path("/content").is_dir() else Path(scratch_base)
        )
        state["storage"] = {"status": "local"}
        refresh_choices(None)
        show_storage_status()

    def execute(_=None):
        if state["busy"]:
            return
        if needs_faiss() and not release_faiss_status()["available"]:
            update()
            with output:
                output.clear_output(wait=True)
                print("선택한 FAISS 검색을 사용하려면 설정 아래 ‘FAISS 설치’를 먼저 누르세요.")
            return
        state["busy"] = True
        run.disabled = True
        refresh.disabled = True
        connect.disabled = local.disabled = True
        install_faiss.disabled = True
        state["last_result"] = None
        with output:
            output.clear_output(wait=True)
            try:
                selected_folder = Path(controls["folder"].value)
                if not selected_folder.is_dir():
                    raise FileNotFoundError(
                        "파일 위치를 확인하고 ZIP 목록을 새로고침하세요."
                    )
                values = {k: v.value for k, v in controls.items() if k in visible}
                values["overrides"] = {
                    k: v.value for k, v in settings_controls.items() if not v.disabled
                }
                values["overrides"].update(
                    {
                        k: json.loads(v.value)
                        for k, v in arrays_controls.items()
                        if not v.disabled
                    }
                )
                request = release_request(values)
                for widget in [
                    *controls.values(),
                    *settings_controls.values(),
                    *arrays_controls.values(),
                ]:
                    widget.disabled = True
                state["last_result"] = release_execute(
                    request, selected_folder, scratch_base
                )
                state["last_error"] = None
            except Exception as exc:
                state["last_error"] = {"type": type(exc).__name__, "message": str(exc)}
                print("실행하지 못했습니다:", type(exc).__name__, str(exc))
                print(
                    "입력 ZIP은 변경되지 않았습니다. 실행 설정과 파일 위치 확인이 필요합니다."
                )
            finally:
                state["busy"] = False
                update()
                refresh.disabled = False
                connect.disabled = local.disabled = False

    def auto_results():
        """Publish captured read-only output without relying on a comm parent ID."""
        if controls["action"].value != "results":
            return
        from IPython.utils.capture import capture_output

        with capture_output() as captured:
            execute()
        output.outputs = ()
        if captured.stdout:
            output.append_stdout(captured.stdout)
        for item in captured.outputs:
            output.outputs += (
                {
                    "output_type": "display_data",
                    "data": item.data,
                    "metadata": item.metadata,
                },
            )
        if captured.stderr:
            output.append_stderr(captured.stderr)

    for key in [
        "action",
        "arm",
        "equation",
        "execution",
        "zip",
        "folder",
        "dense_rag",
        "dense_backend",
        "plot_kind",
        "registry_operation",
        "seed_mode",
        "all_methods",
    ]:
        controls[key].observe(update, names="value")
    run.on_click(execute)
    refresh.on_click(refresh_choices)
    connect.on_click(connect_drive)
    local.on_click(use_local)
    install_faiss.on_click(install_faiss_clicked)
    update()
    if show:
        display(dashboard)
    show_storage_status()
    return {
        "dashboard": dashboard,
        "output": output,
        "advanced": advanced,
        "run_button": run,
        "faiss_install_button": install_faiss,
        "faiss_install_box": faiss_box,
        "faiss_notice": faiss_notice,
        "controls": controls,
        "settings": settings_controls,
        "arrays": arrays_controls,
        "execute": execute,
        "auto_results": auto_results,
        "refresh": refresh_choices,
        "connect_drive": connect_drive,
        "use_local": use_local,
        "show_storage_status": show_storage_status,
        "state": state,
    }
