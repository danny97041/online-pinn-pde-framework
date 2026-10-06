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
            "Kovasznay 순방향",
            "Kovasznay 역문제",
            "Taylor–Green",
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
}


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
    if not filename or Path(filename).name != filename or "/" in filename or "\\" in filename:
        raise ValueError("ZIP은 파일 목록에서 선택하세요.")
    request = {"action": action, "zip": filename}
    if action in {"predict", "train"}:
        eq, arm, seed = values.get("equation"), values.get("arm"), values.get("seed")
        if eq not in U_PROBLEMS or arm not in U_ARMS or type(seed) is not int or seed < 0:
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
    if action == "agent":
        question = values.get("question", "").strip()
        if not question or len(question) > 4000:
            raise ValueError("질문은 1~4000자로 입력하세요.")
        request["question"] = question
    if action == "evaluate":
        request.update(
            check_models=bool(values.get("check_models", False)),
            check_training=bool(values.get("check_training", False)),
        )
    if action == "train":
        s = release_settings(request["equation"])
        execution = values.get("execution", "new")
        if execution not in {"new", "resume"}:
            raise ValueError("새 학습 또는 재개를 선택하세요.")
        overrides = dict(values.get("overrides", {}))
        if set(overrides) - set(s):
            raise ValueError("알 수 없는 학습 설정: " + str(sorted(set(overrides) - set(s))))
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
    print("목적:", V3_ACTION_LABELS[action], "/ 저장 실험", summary["completed_trials"], "개")
    if action == "results":
        display(Markdown((root / "REPORT.md").read_text(encoding="utf-8")))
        print(
            "검사 기록은 결과 ZIP의 validation/에서 실행 당시 코드와 함께 확인할 수 있습니다."
        )
    elif action == "predict":
        print(
            release_predictor(root)(
                request["equation"], request["arm"], request["seed"], request["points"]
            )
        )
    elif action == "agent":
        agent, _ = release_services(root, request["use_llm"], request["dense_rag"])
        display(Markdown(agent.ask(request["question"])["answer"]))
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
            settings_path = parent_root / "settings.json"
            if settings_path.exists():
                s = json.loads(settings_path.read_text(encoding="utf-8"))
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
        s.update(execution=request["execution"], seed=request["seed"], arms=[request["arm"]])
        s.update(request["overrides"])
        s = release_validate_settings(s, parent["settings"] if parent else None)
        display(Markdown("```json\n" + json.dumps(s, ensure_ascii=False, indent=2) + "\n```"))
        if not request["training_enabled"]:
            print("설정 확인만 완료했습니다. ‘학습 실행 허용’을 켜야 학습합니다.")
            return {"status": "settings_only", "settings": s}
        run_id = uuid.uuid4().hex[:8]
        destination = scratch_base / ("v3_train_" + run_id)
        output = folder / ("V3_Train_" + s["equation"] + "_" + run_id + ".zip")
        release_training(root, s, destination, parent, output_zip=output)
        print("완료된 실험 경계 저장:", output.name)
        display(FileLink(str(output)))
    return summary


def release_panel(folder, scratch_base):
    """Return a two-tab widget; no model import or training on construction."""
    import ipywidgets as w
    from IPython.display import display

    style = {"description_width": "150px"}

    def control(cls, label, **kwargs):
        return cls(description=label, style=style, layout=w.Layout(width="95%"), **kwargs)

    controls = {
        "action": control(
            w.Dropdown, "사용할 기능", options=[(v, k) for k, v in V3_ACTION_LABELS.items()]
        ),
        "zip": control(w.Dropdown, "통합 결과 ZIP", options=release_zip_choices(folder)),
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
            w.Textarea, "질문", value="Poisson에서 방법별 물리장 오차를 비교해줘"
        ),
        "use_llm": control(w.Checkbox, "Qwen 사용", value=False),
        "dense_rag": control(w.Checkbox, "E5 + BM25 검색", value=False),
        "check_models": control(w.Checkbox, "저장 모델 전수 검사", value=False),
        "check_training": control(w.Checkbox, "재개·LHS 경계 검사", value=False),
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
            w.IntText if type(defaults[key]) is int else w.FloatText, label, value=defaults[key]
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
    output = w.Output()
    notice = w.HTML()
    run = w.Button(description="선택한 기능 실행", button_style="primary")
    go = w.Button(description="결과 화면으로 이동")
    refresh = w.Button(description="ZIP 목록 새로고침")
    advanced = w.Accordion(
        children=[w.VBox(list(settings_controls.values()) + list(arrays_controls.values()))]
    )
    advanced.set_title(0, "추가 학습 설정 — 펼쳐서 수정")
    advanced.selected_index = None
    panel = w.VBox(
        [
            w.HTML(
                "<h3>설정</h3><p>필요한 항목만 표시됩니다. 기본 기능은 저장 결과 조회이며 학습하지 않습니다.</p>"
            ),
            controls["action"],
            controls["zip"],
            refresh,
            *[v for k, v in controls.items() if k not in {"action", "zip"}],
            advanced,
            notice,
            w.HBox([run, go]),
        ]
    )
    tabs = w.Tab(children=[panel, output])
    tabs.set_title(0, "설정")
    tabs.set_title(1, "결과 · 실행 로그")
    state = {"last_result": None, "last_error": None}
    visible = set()

    def update(_=None):
        nonlocal visible
        if _ is not None:
            # A changed action/model/parent needs a fresh training opt-in.
            controls["training_enabled"].value = False
        a, arm, eq = controls["action"].value, controls["arm"].value, controls["equation"].value
        visible = {"action", "zip"}
        if a in {"predict", "train"}:
            visible |= {"equation", "arm", "seed"}
        if a == "predict":
            visible |= {"points"}
        if a in {"agent", "evaluate"}:
            visible |= {"use_llm", "dense_rag"}
        if a == "agent":
            visible |= {"question"}
        if a == "evaluate":
            visible |= {"check_models", "check_training"}
        if a == "train":
            visible |= {"execution", "training_enabled"}
            if controls["execution"].value == "resume":
                visible |= {"resume_zip"}
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
            "evaluate": "<p>자연어 30문항 + 별도 표현 15문항 / API 검사. 전수 검사와 재개 검사는 선택 사항입니다.</p>",
            "train": "<p>완료 결과는 변경하지 않습니다. 재개 시 모델·샘플링 설정은 저장값을 상속하며 미완료 실험은 세션 종료 시 소실될 수 있습니다.</p>",
            "predict": "<p>좌표 순서와 영역: 선택 방정식의 x/y/t 구성에 맞추세요. 잘못된 차원·영역은 실행 시 거절됩니다.</p>",
        }.get(a, "")

    def refresh_choices(_):
        names = release_zip_choices(folder)
        previous = controls["zip"].value
        controls["zip"].options = names
        if previous in names:
            controls["zip"].value = previous
        controls["resume_zip"].options = [("선택한 통합 결과에서 재개", "")] + [
            (n, n) for n in release_zip_choices(folder, True)
        ]

    def execute(_=None):
        run.disabled = True
        tabs.selected_index = 1
        with output:
            output.clear_output(wait=True)
            try:
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
                state["last_result"] = release_execute(request, folder, scratch_base)
                state["last_error"] = None
            except Exception as exc:
                state["last_error"] = {"type": type(exc).__name__, "message": str(exc)}
                print("실행하지 못했습니다:", type(exc).__name__, str(exc))
                print("설정 탭에서 항목을 확인하세요. 입력 결과 ZIP은 변경하지 않았습니다.")
            finally:
                run.disabled = False

    for key in ["action", "arm", "equation", "execution", "zip"]:
        controls[key].observe(update, names="value")
    run.on_click(execute)
    go.on_click(lambda _: setattr(tabs, "selected_index", 1))
    refresh.on_click(refresh_choices)
    update()
    display(tabs)
    return {
        "tabs": tabs,
        "controls": controls,
        "settings": settings_controls,
        "arrays": arrays_controls,
        "execute": execute,
        "refresh": refresh_choices,
        "state": state,
    }
