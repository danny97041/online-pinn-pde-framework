"""Korean bottom panel. Defaults are read-only; no implicit training or installs."""

import json
from pathlib import Path
import tempfile
import uuid
from contextlib import nullcontext

from .config import Settings, ARMS, ARM_LABELS, SEEDS, digest
from .data import DataSource
from .equations import EQUATIONS
from .runner import create_run, run_all
from .experiments import experiment_plan, execute_plan
from .reporting import generate_report, make_plots
from .services import Service
from .storage import pack, restore
from .settings_panel import settings_editor


def panel(folder="/content/drive/MyDrive/pinnV4", show=True):
    import ipywidgets as w
    from IPython.display import display, Markdown

    width = w.Layout(width="100%")
    style = {"description_width": "130px"}

    def dropdown(description, options, value):
        return w.Dropdown(
            description=description,
            options=options,
            value=value,
            layout=width,
            style=style,
        )

    action = dropdown(
        "사용할 기능",
        [
            (v, k)
            for k, v in {
                "results": "저장 결과 보기",
                "train": "학습 설정·실행",
                "predict": "좌표 예측",
                "ask": "자연어 질문",
                "check": "선택 기능 확인",
                "documents": "문서 생성",
                "plots": "결과 시각화",
                "registry": "추론 모델 관리",
            }.items()
        ],
        "results",
    )
    storage = w.Text(value=folder, description="저장 폴더", layout=width, style=style)
    zip_name = w.Text(
        value="V4_Results.zip", description="통합 ZIP", layout=width, style=style
    )
    connect = w.Button(description="Drive 연결")
    local = w.Button(description="로컬 ZIP 사용")
    equation = dropdown(
        "방정식", [(v["label"], k) for k, v in EQUATIONS.items()], "wave2d"
    )
    inverse = dropdown("문제 방향", [("순방향", False), ("역문제", True)], False)

    class SelectionChecks(w.VBox):
        def __init__(self, title, options, selected):
            self.checks = {
                k: w.Checkbox(
                    description=label,
                    value=k in selected,
                    layout=w.Layout(width="100%"),
                )
                for label, k in options
            }
            super().__init__([w.HTML(f"<b>{title}</b>"), *self.checks.values()])

        @property
        def value(self):
            return tuple(k for k, box in self.checks.items() if box.value)

        @value.setter
        def value(self, values):
            for k, box in self.checks.items():
                box.value = k in values

        @property
        def disabled(self):
            return all(box.disabled for box in self.checks.values())

        @disabled.setter
        def disabled(self, value):
            for box in self.checks.values():
                box.disabled = value

    class OrderedEquationChecks(SelectionChecks):
        def __init__(self, title, options, selected):
            super().__init__(title, options, selected)
            self.order_list = w.Select(description="실행 순서", layout=width, style=style)
            self.up = w.Button(description="위로")
            self.down = w.Button(description="아래로")
            self.children += (self.order_list, w.HBox([self.up, self.down]))
            self._changing_selection = False
            self.value = selected
            for box in self.checks.values():
                box.observe(self._selection_changed, names="value")
            self.order_list.observe(self._update_buttons, names="value")
            self.up.on_click(lambda _: self._move(-1))
            self.down.on_click(lambda _: self._move(1))

        @property
        def value(self):
            return tuple(self._equation_order)

        @value.setter
        def value(self, values):
            self._equation_order = list(dict.fromkeys(k for k in values if k in self.checks))
            self._changing_selection = True
            try:
                SelectionChecks.value.fset(self, self._equation_order)
            finally:
                self._changing_selection = False
            self._refresh_order()

        def _selection_changed(self, change):
            if self._changing_selection:
                return
            key = next(k for k, box in self.checks.items() if box is change.owner)
            if change.new:
                self._equation_order.append(key)
            else:
                self._equation_order.remove(key)
            self._refresh_order()

        def _refresh_order(self):
            order = self._equation_order
            selected = self.order_list.value
            self.order_list.options = [
                (f"{i}. {self.checks[k].description}", k) for i, k in enumerate(order, 1)
            ]
            self.order_list.value = selected if selected in order else next(iter(order), None)
            self._update_buttons()

        def _update_buttons(self, change=None):
            order = self._equation_order
            selected = self.order_list.value
            locked = self.disabled or selected not in order
            self.up.disabled = locked or selected == order[0]
            self.down.disabled = locked or selected == order[-1]

        def _move(self, step):
            selected = self.order_list.value
            if self.disabled or selected not in self._equation_order:
                return
            i = self._equation_order.index(selected)
            j = i + step
            if 0 <= j < len(self._equation_order):
                self._equation_order.insert(j, self._equation_order.pop(i))
                self._refresh_order()

        @SelectionChecks.disabled.setter
        def disabled(self, value):
            SelectionChecks.disabled.fset(self, value)
            self.order_list.disabled = value
            self._update_buttons()

    arms = SelectionChecks(
        "비교 방법", [(ARM_LABELS[k], k) for k in ARMS], ("baseline",)
    )
    seeds = dropdown(
        "시드 구성", [("1시드 · 3234", (3234,)), ("4시드 · 3234~3237", SEEDS)], (3234,)
    )
    generalization = dropdown(
        "실험 종류",
        [
            ("기본 비교", "basic"),
            ("16구역 관측 제외 · 보간", "interpolation"),
            ("시간 앞80% 학습 · 외삽", "time_extrapolation"),
        ],
        "basic",
    )
    data_mode = dropdown(
        "데이터 연결",
        [("합성 참조 문제", "synthetic"), ("CSV + 알려진 조건", "csv")],
        "synthetic",
    )
    data_contract = w.Text(description="데이터 JSON", layout=width, style=style)
    profile = dropdown(
        "실행 규모", [("본 실험", "research"), ("축소 동작 검사", "smoke")], "research"
    )
    execution = dropdown(
        "시작 방식", [("새 독립 실험", "new"), ("저장 실행에서 재개", "load")], "new"
    )
    batch = w.Checkbox(
        description="여러 방정식·민감도 조건을 한 번에 실행", value=False
    )
    batch_equations = OrderedEquationChecks(
        "방정식 묶음",
        [(v["label"], k) for k, v in EQUATIONS.items() if k != "cylinder_ns"],
        ("wave2d",),
    )
    directions = dropdown(
        "문제 방향",
        [
            ("순방향만", (False,)),
            ("역문제만", (True,)),
            ("순·역문제 모두", (False, True)),
        ],
        (False, True),
    )
    noise_levels = w.Text(
        value="0.0", description="노이즈 비율 목록", layout=width, style=style
    )
    initial_levels = w.Text(
        value="1.0", description="초기 계수 배율 목록", layout=width, style=style
    )
    load_settings = w.Button(description="저장 설정·모델 목록 불러오기")
    authorized = w.Checkbox(description="학습 실행 허용", value=False)
    settings_controls = settings_editor()
    advanced = w.Accordion(children=[settings_controls])
    advanced.set_title(0, "변수별 설명·학습 설정")
    advanced.selected_index = None
    trial = dropdown("완료 실험", [("모델 목록을 불러오세요", "")], "")
    coords = w.Textarea(
        value="[[0.25,0.25,0.5]]", description="좌표 JSON", layout=width, style=style
    )
    question = w.Textarea(
        value="wave2d의 Adam 결과를 비교해줘",
        description="질문",
        layout=width,
        style=style,
    )
    qwen = w.Checkbox(description="Qwen 사용", value=False)
    dense = w.Checkbox(description="E5 + BM25 검색", value=False)
    backend = dropdown("검색 엔진", [("NumPy", "numpy"), ("FAISS", "faiss")], "numpy")
    install = w.Button(description="FAISS 설치")
    install_note = w.HTML()
    time_value = w.FloatText(value=0, description="시간 위치", style=style)
    component = dropdown("표시할 출력", [("u", 0)], 0)
    extra_questions = w.Textarea(description="확인할 질문", layout=width, style=style)
    check_note = w.HTML(
        "Qwen은 기본 3문항, E5/FAISS는 검색 연결만 확인합니다. 질문을 입력하면 해당 질문만 사용합니다. 기존 전체 검사는 재실행하지 않습니다."
    )
    registry_action = dropdown(
        "모델 관리",
        [
            ("정보 조회", "info"),
            ("기본 추론 모델 지정", "select"),
            ("직전 모델 복원", "rollback"),
        ],
        "info",
    )
    approved = w.Checkbox(description="모델 변경 승인", value=False)
    run = w.Button(description="선택한 기능 실행", button_style="primary")
    out = w.Output()
    dependencies = w.Button(description="선택 기능의 필요 패키지 설치")
    dependency_note = w.HTML()
    fields = w.VBox()
    state = {
        "root": None,
        "bundle": None,
        "service": None,
        "busy": False,
        "pending_training": None,
        "loaded_plan": (),
        "loaded_settings_path": None,
    }
    widgets = [
        action,
        storage,
        zip_name,
        equation,
        inverse,
        arms,
        seeds,
        generalization,
        data_mode,
        data_contract,
        profile,
        execution,
        authorized,
        trial,
        coords,
        question,
        qwen,
        dense,
        backend,
        time_value,
        registry_action,
        approved,
        batch,
        batch_equations,
        directions,
        noise_levels,
        initial_levels,
    ]

    def available_faiss():
        try:
            import faiss

            return faiss.IndexFlatIP(2).d == 2
        except (ImportError, AttributeError, OSError):
            return False

    def update(_=None):
        settings_controls.set_context(
            arms.value,
            True in directions.value if batch.value else inverse.value,
            generalization.value,
            data_mode.value,
            EQUATIONS[equation.value]["time_axis"] is not None,
        )
        selected = (
            [
                equation,
                inverse,
                arms,
                seeds,
                generalization,
                data_mode,
                profile,
                execution,
                authorized,
                advanced,
            ]
            if action.value == "train"
            else (
                [trial, coords]
                if action.value == "predict"
                else (
                    [question, qwen, dense, backend]
                    if action.value == "ask"
                    else (
                        [trial, time_value, component]
                        if action.value == "plots"
                        else (
                            [equation, trial, registry_action, approved]
                            if action.value == "registry"
                            else (
                                [qwen, dense, backend, extra_questions, check_note]
                                if action.value == "check"
                                else []
                            )
                        )
                    )
                )
            )
        )
        if action.value == "train" and data_mode.value == "csv":
            selected.insert(-2, data_contract)
        if action.value == "train":
            selected.insert(-2, batch)
            if batch.value:
                selected.remove(equation)
                selected.remove(inverse)
                selected[-2:-2] = [
                    batch_equations,
                    directions,
                    noise_levels,
                    initial_levels,
                ]
            locked = execution.value == "load"
            for control in (
                equation,
                inverse,
                arms,
                seeds,
                generalization,
                data_mode,
                profile,
                batch,
                batch_equations,
                directions,
                noise_levels,
                initial_levels,
                settings_controls,
            ):
                control.disabled = locked or state["busy"]
        elif not state["busy"]:
            equation.disabled = False
        missing = (
            action.value in {"ask", "check"}
            and dense.value
            and backend.value == "faiss"
            and not available_faiss()
        )
        install.layout.display = "" if missing else "none"
        install_note.value = (
            "FAISS 미설치: 아래 버튼을 직접 눌러 설치하세요." if missing else ""
        )
        fields.children = selected
        run.disabled = state["busy"] or missing
        run.description = (
            "학습 설정 확인·실행 준비"
            if action.value == "train"
            else "선택한 기능 실행"
        )
        time_value.layout.display = (
            "" if EQUATIONS[equation.value]["time_axis"] is not None else "none"
        )
        if state["root"] is not None and trial.value:
            selected_model()
        packages = missing_packages()
        dependencies.layout.display = "" if packages else "none"
        dependency_note.value = (
            "필요 패키지: " + ", ".join(packages) if packages else ""
        )

    def mount(_=None):
        with out:
            try:
                from google.colab import drive

                drive.mount("/content/drive")
            except Exception as exc:
                print(
                    "Drive 연결 실패:",
                    type(exc).__name__,
                    ". 로컬 ZIP 사용을 선택할 수 있습니다.",
                )

    def use_local(_=None):
        storage.value = "/content"
        state["root"] = None
        state["bundle"] = None
        state["service"] = None

    def missing_packages():
        import importlib.util

        required = (
            {"tensorflow": "tensorflow", "scipy": "scipy"}
            if action.value in {"predict", "plots"}
            or (action.value == "train" and authorized.value)
            else {}
        )
        if action.value == "plots":
            required["matplotlib"] = "matplotlib"
        if action.value in {"ask", "check"} and qwen.value:
            required.update(torch="torch", transformers="transformers")
        if action.value in {"ask", "check"} and dense.value:
            required["sentence_transformers"] = "sentence-transformers"
        return [
            package
            for module, package in required.items()
            if importlib.util.find_spec(module) is None
        ]

    def dependencies_clicked(_):
        import subprocess, sys

        packages = missing_packages()
        if not packages or state["busy"]:
            return
        state["busy"] = True
        update()
        dependencies.disabled = True
        with out:
            try:
                result = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "pip",
                        "install",
                        "--quiet",
                        "--no-input",
                        *packages,
                    ],
                    capture_output=True,
                    text=True,
                    timeout=600,
                )
                if result.returncode:
                    raise RuntimeError(result.stderr[-1200:])
                print(
                    "필요 패키지 설치 완료. 실행 버튼을 누르세요. 이미 불완전하게 import한 패키지가 있으면 런타임을 재시작하세요."
                )
            except Exception as exc:
                print("설치 실패:", str(exc))
            finally:
                state["busy"] = False
                dependencies.disabled = False
                update()

    def install_clicked(_):
        if state["busy"] or available_faiss():
            return
        import subprocess, sys

        state["busy"] = True
        update()
        install.disabled = True
        with out:
            try:
                r = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "pip",
                        "install",
                        "--quiet",
                        "--no-input",
                        "--no-deps",
                        "--only-binary=:all:",
                        "faiss-cpu",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=240,
                )
                if r.returncode or not available_faiss():
                    raise RuntimeError("설치 후 로드 실패")
                print("FAISS 설치 완료. 실행 버튼을 누르세요.")
            except Exception as exc:
                print(
                    "설치 실패:",
                    type(exc).__name__,
                    ". NumPy 검색 또는 런타임 재시작을 사용할 수 있습니다.",
                )
            finally:
                state["busy"] = False
                install.disabled = False
                update()

    def open_root():
        if Path(
            zip_name.value
        ).name != zip_name.value or not zip_name.value.lower().endswith(".zip"):
            raise ValueError("ZIP에는 파일명만 입력하세요.")
        file = Path(storage.value) / zip_name.value
        if state["root"] is None or state["bundle"] != str(file):
            if not file.is_file():
                raise FileNotFoundError("선택 폴더에 " + file.name + "을 넣으세요.")
            workspace = Path(tempfile.mkdtemp(prefix="v4_view_")) / "results"
            restore(file, workspace)
            state.update(root=workspace, bundle=str(file), service=None)
        return state["root"]

    def current_service(root):
        service = state["service"]
        if service is None or service.root != root:
            service = state["service"] = Service(root)
        return service

    def refresh(_=None):
        state["loaded_settings_path"] = None
        state["pending_training"] = None
        authorized.value = False
        with out:
            try:
                root = open_root()
                kind = json.loads((root / "contract.json").read_text(encoding="utf-8"))[
                    "kind"
                ]
                if kind not in {"results", "catalog"}:
                    raise ValueError(
                        "결과 ZIP을 선택하세요. 재개 ZIP은 같은 실행 ID의 결과 ZIP과 함께 자동으로 사용합니다."
                    )
                service = current_service(root)
                options = [
                    (
                        EQUATIONS[m["equation"]]["label"]
                        + " / "
                        + ("역" if m["inverse"] else "순")
                        + " / "
                        + ARM_LABELS[m["arm"]]
                        + " / "
                        + str(m["seed"])
                        + " / "
                        + m["generalization"]["mode"],
                        m["trial_id"],
                    )
                    for m in service.models()
                ]
                trial.options = options or [("완료된 V4 모델 없음", "")]
                if (root / "settings.json").exists():
                    values = json.loads(
                        (root / "settings.json").read_text(encoding="utf-8")
                    )
                    settings = Settings.from_dict(values)
                    settings_controls.value = values
                    equation.value = settings.equation
                    inverse.value = settings.inverse
                    arms.value = settings.arms
                    if settings.seeds in ((3234,), SEEDS):
                        seeds.value = settings.seeds
                    generalization.value = settings.generalization
                    data_mode.value = settings.data_mode
                    authorized.value = False
                    profile.value = (
                        "smoke" if settings.cap < settings.stop_first else "research"
                    )
                execution.value = "load" if kind == "results" else "new"
                plan_path = root / "plan.json"
                plan = (
                    tuple(
                        Settings.from_dict(s)
                        for s in json.loads(plan_path.read_text(encoding="utf-8"))
                    )
                    if kind == "results" and plan_path.is_file()
                    else ()
                )
                state["loaded_plan"] = plan
                batch.value = bool(plan)
                if plan:
                    batch_equations.value = tuple(
                        dict.fromkeys(s.equation for s in plan)
                    )
                    directions.value = tuple(sorted({s.inverse for s in plan}))
                    noise_levels.value = ", ".join(
                        str(x) for x in dict.fromkeys(s.noise_fraction for s in plan)
                    )
                    if any(s.inverse for s in plan):
                        multipliers = dict.fromkeys(
                            s.initial_coefficient
                            / (
                                settings.initial_coefficient
                                or EQUATIONS[s.equation]["initial"]
                            )
                            for s in plan
                            if s.inverse and s.initial_coefficient is not None
                        )
                        initial_levels.value = (
                            ", ".join(f"{x:g}" for x in multipliers) or "1.0"
                        )
                state["loaded_settings_path"] = (
                    state["bundle"] if kind == "results" else None
                )
                update()
                print(
                    "저장 설정과 모델 목록을 불러왔습니다. 시작 방식: "
                    + (
                        "저장 실행에서 재개"
                        if kind == "results"
                        else "새 독립 실험 · 결과 목록 ZIP"
                    )
                    + ". 학습 허용은 꺼져 있습니다."
                )
            except Exception as exc:
                print("불러오기 실패:", str(exc))

    def selected_model(_=None):
        if state["root"] is None or not trial.value:
            return
        info = current_service(state["root"]).model_info(trial.value)
        component.options = [
            (name, i) for i, name in enumerate(info["output_variables"])
        ]
        time_axis = EQUATIONS[info["equation"]]["time_axis"]
        time_value.layout.display = "" if time_axis is not None else "none"
        if time_axis is not None:
            lo, hi = info["input_variables"][time_axis]["range"]
            if not lo <= time_value.value <= hi:
                time_value.value = (lo + hi) / 2

    def training_signature():
        return digest(
            {
                "settings": settings_controls.value,
                "equation": equation.value,
                "inverse": inverse.value,
                "arms": list(arms.value),
                "seeds": list(seeds.value),
                "generalization": generalization.value,
                "data_mode": data_mode.value,
                "profile": profile.value,
                "execution": execution.value,
                "storage": storage.value,
                "zip": zip_name.value,
                "data_contract": data_contract.value,
                "batch": batch.value,
                "batch_equations": list(batch_equations.value),
                "directions": directions.value,
                "noise_levels": noise_levels.value,
                "initial_levels": initial_levels.value,
            }
        )

    def execute(_=None, *, from_cell=False):
        if state["busy"]:
            return
        state["busy"] = True
        for widget in widgets:
            widget.disabled = True
        settings_controls.disabled = True
        connect.disabled = local.disabled = install.disabled = True
        update()
        with nullcontext() if from_cell else out:
            if not from_cell:
                out.clear_output(wait=True)
            try:
                selected = action.value
                if selected == "train":
                    values = settings_controls.value
                    values.update(
                        equation=equation.value,
                        inverse=inverse.value,
                        arms=list(arms.value),
                        seeds=list(seeds.value),
                        generalization=generalization.value,
                        data_mode=data_mode.value,
                    )
                    settings = Settings.from_dict(values)
                    if profile.value == "smoke":
                        settings = settings.smoke()
                    if execution.value == "load" and state[
                        "loaded_settings_path"
                    ] != str(Path(storage.value) / zip_name.value):
                        raise ValueError(
                            "선택한 ZIP의 저장 설정·모델 목록을 먼저 불러오세요."
                        )
                    if batch.value and not batch_equations.value:
                        raise ValueError("방정식 묶음에서 하나 이상 선택하세요.")
                    configurations = (
                        state["loaded_plan"]
                        if execution.value == "load" and state["loaded_plan"]
                        else (
                            experiment_plan(
                                settings,
                                tuple(batch_equations.value),
                                directions.value,
                                tuple(
                                    float(v.strip())
                                    for v in noise_levels.value.split(",")
                                ),
                                tuple(
                                    float(v.strip())
                                    for v in initial_levels.value.split(",")
                                ),
                            )
                            if batch.value
                            else [settings]
                        )
                    )
                    print(
                        "목적: 독립 V4 "
                        + (
                            "축소 동작 검사"
                            if profile.value == "smoke"
                            else "학습 비교"
                        )
                    )
                    trial_count = sum(
                        len(s.arms) * len(s.seeds) for s in configurations
                    )
                    print(
                        "실행 범위:",
                        trial_count,
                        "실험 / 실험별 Adam 한도",
                        settings.cap,
                        "/ L-BFGS",
                        settings.lbfgs_maxiter,
                    )
                    print(
                        "Adam 최악 한도:",
                        trial_count * settings.cap,
                        "업데이트. 완료된 실험은 제외합니다.",
                    )
                    descriptions = {
                        "baseline": "관측 데이터·물리식·경계 및 초기조건을 함께 학습",
                        "loss_sa": "세 손실 그룹의 가중치를 학습 중 조절",
                        "ff": "입력을 주파수 특징으로 변환해 학습",
                        "ff_loss_sa": "주파수 특징과 손실 가중치 조절을 함께 사용",
                        "ff_curriculum": "주파수 특징을 사용하며 PDE 비중을 점진적으로 증가",
                        "supervised_only": "물리식 없이 관측값만 학습; 계수 추정 없음",
                    }
                    display(
                        Markdown(
                            "| 선택 방법 | 학습 방식 |\n|---|---|\n"
                            + "\n".join(
                                f"| {ARM_LABELS[k]} | {descriptions[k]} |"
                                for k in settings.arms
                            )
                            + "\n\n"
                            + ("방정식 실행 순서: **" if batch.value else "방정식: **")
                            + " → ".join(
                                dict.fromkeys(
                                    EQUATIONS[s.equation]["label"]
                                    for s in configurations
                                )
                            )
                            + "** · "
                            + "·".join(
                                dict.fromkeys(
                                    "역문제" if s.inverse else "순방향"
                                    for s in configurations
                                )
                            )
                            + f" · 시드 {list(settings.seeds)}\n\n"
                            f"목표: 검증 **L1·L2 모두 {settings.target * 100:g}% 이하** · "
                            f"{settings.stop_first:,}회부터 {settings.check_every:,}회마다 확인, "
                            f"연속 {settings.consecutive}회 만족 시 중단.\n\n"
                            f"학습점 갱신: {settings.sampling_every:,}회마다 "
                            "PDE는 RAD, 나머지 합성 학습점은 LHS. "
                            "검증·평가점은 고정합니다. 상세 수치는 접힌 고급 설정에서 확인합니다."
                        )
                    )
                    if not authorized.value:
                        state["pending_training"] = None
                        print(
                            "설정 확인만 완료했습니다. 학습 실행 허용을 켜야 실행합니다."
                        )
                        return
                    if missing_packages():
                        raise RuntimeError("먼저 필요 패키지 설치 버튼을 누르세요.")
                    if not from_cell:
                        state["pending_training"] = training_signature()
                        print(
                            "실행 준비 완료. 아래 ‘학습 실행’ 셀의 ▶를 누르세요. 이 버튼에서는 학습하지 않습니다."
                        )
                        return
                    target = Path(storage.value)
                    if (
                        str(target).startswith("/content/drive/")
                        and not Path("/content/drive/MyDrive").is_dir()
                    ):
                        mount()
                        if not Path("/content/drive/MyDrive").is_dir():
                            raise RuntimeError(
                                "Drive 연결 또는 로컬 ZIP 사용이 필요합니다."
                            )
                    target.mkdir(parents=True, exist_ok=True)
                    if execution.value == "load":
                        root = open_root()
                        stored = Settings.from_dict(
                            json.loads(
                                (root / "settings.json").read_text(encoding="utf-8")
                            )
                        )
                        if (
                            json.loads(
                                (root / "contract.json").read_text(encoding="utf-8")
                            )["kind"]
                            != "results"
                        ):
                            raise ValueError(
                                "결과 목록 ZIP에서는 새 독립 실험을 선택하세요."
                            )
                        if stored.to_dict() != settings.to_dict():
                            raise ValueError(
                                "재개 시 저장 설정·모델 목록을 불러오고 값을 그대로 사용하세요. 설정 변경은 새 실험입니다."
                            )
                        source = DataSource(
                            settings,
                            (
                                root / "inputs/data.json"
                                if settings.data_mode == "csv"
                                else None
                            ),
                        )
                        if (root / "plan.json").is_file():
                            configurations = [
                                Settings.from_dict(v)
                                for v in json.loads(
                                    (root / "plan.json").read_text(encoding="utf-8")
                                )
                            ]
                    else:
                        root = Path(tempfile.mkdtemp(prefix="v4_run_")) / "run"
                        source = create_run(
                            root,
                            settings,
                            (
                                (target / data_contract.value)
                                if settings.data_mode == "csv"
                                else None
                            ),
                        )
                    owner = json.loads(
                        (root / "contract.json").read_text(encoding="utf-8")
                    )
                    suffix = owner["run_id"]
                    result = (
                        (target / zip_name.value)
                        if execution.value == "load"
                        else target / ("V4_Results_" + suffix + ".zip")
                    )
                    resume = target / ("V4_Resume_" + suffix + ".zip")
                    pack(root, result)
                    print("결과 ZIP:", result.name, "/ 재개 ZIP:", resume.name)
                    if execution.value == "load" and not resume.exists():
                        print(
                            "재개 ZIP 없음: 완료 실험은 건너뛰며 저장된 미완료 상태가 없는 실험은 처음부터 시작합니다."
                        )
                    from .progress import training_progress

                    with training_progress():
                        if len(configurations) > 1 or batch.value:
                            execute_plan(root, configurations, result, resume, True)
                        else:
                            run_all(root, settings, source, result, resume, True)
                    trial.options = [("모델 목록을 불러오세요", "")]
                    state.update(root=root, bundle=str(result), service=None)
                    zip_name.value = result.name
                    print("완료:", result.name)
                    return
                root = open_root()
                service = current_service(root)
                if selected == "results":
                    file = root / "REPORT.md"
                    display(
                        Markdown(
                            file.read_text(encoding="utf-8")
                            if file.exists()
                            else "완료된 V4 실험이 없습니다."
                        )
                    )
                    historical = root / "historical/V3_comparison.md"
                    if historical.exists():
                        display(Markdown(historical.read_text(encoding="utf-8")))
                elif selected == "predict":
                    print(service.predict(trial.value, json.loads(coords.value)))
                elif selected == "ask":
                    result = service.ask(
                        question.value, qwen.value, dense.value, backend.value
                    )
                    print(result["answer"])
                    print("Qwen 실제 호출:", result["llm_called"])
                elif selected == "registry":
                    if registry_action.value == "info":
                        info = service.model_info(trial.value)
                        lines = [
                            "### 모델 정보",
                            "",
                            "| 항목 | 값 |",
                            "|---|---|",
                            f"| 방정식 | {EQUATIONS[info['equation']]['label']} |",
                            f"| 방법 | {ARM_LABELS[info['arm']]} |",
                            f"| 시드 | {info['seed']} |",
                            f"| 선택 단계 | {info['selected_stage']} |",
                            "",
                            "| 입력 변수 | 범위 |",
                            "|---|---|",
                        ]
                        lines += [
                            f"| {v['symbol']} | {v['range'][0]} ~ {v['range'][1]} |"
                            for v in info["input_variables"]
                        ]
                        lines += [
                            "",
                            "출력: " + ", ".join(info["output_variables"]),
                            "추정 대상: " + info["coefficient"],
                        ]
                        display(Markdown("\n".join(lines)))
                    else:
                        result = (
                            service.register(
                                equation.value, trial.value, approved.value
                            )
                            if registry_action.value == "select"
                            else service.rollback(equation.value, approved.value)
                        )
                        print(result)
                        output = Path(storage.value) / (
                            "V4_Registry_" + uuid.uuid4().hex[:8] + ".zip"
                        )
                        pack(root, output)
                        print("출력:", output.name)
                elif selected == "check":
                    from .verification import integration_checks
                    from .storage import write_json

                    checks = integration_checks(
                        root,
                        qwen.value,
                        dense.value,
                        backend.value,
                        extra_questions.value.splitlines(),
                        service=service,
                    )
                    write_json(root / "validation/colab.json", checks)
                    output = Path(storage.value) / (
                        "V4_Check_" + uuid.uuid4().hex[:8] + ".zip"
                    )
                    pack(root, output)
                    print(
                        "선택 기능 확인: 질문",
                        checks["question_count"],
                        "개 / 실제 Qwen 호출",
                        checks["actual_llm_calls"],
                    )
                    if dense.value:
                        print(
                            "검색 연결:",
                            checks["retrieval"]["backend"],
                            checks["retrieval"]["passed"],
                        )
                    print(
                        "출력:",
                        output.name,
                        "/ 답변 의미 정확도는 별도 확인 항목입니다.",
                    )
                else:
                    if selected == "documents":
                        generate_report(root)
                    else:
                        for file in make_plots(
                            root, trial.value or None, time_value.value, component.value
                        ):
                            if file.suffix == ".png":
                                from IPython.display import Image

                                display(Image(filename=str(file)))
                    output = Path(storage.value) / (
                        "V4_"
                        + ("Docs" if selected == "documents" else "Plots")
                        + "_"
                        + uuid.uuid4().hex[:8]
                        + ".zip"
                    )
                    pack(root, output)
                    print("출력:", output.name)
            except Exception as exc:
                print("실행하지 못했습니다:", type(exc).__name__, str(exc))
                if from_cell:
                    raise
            finally:
                state["busy"] = False
                for widget in widgets:
                    widget.disabled = False
                settings_controls.disabled = False
                connect.disabled = local.disabled = install.disabled = False
                update()

    def run_training():
        """Called by a regular notebook cell, never by the widget callback."""
        if state["busy"]:
            raise RuntimeError("다른 기능이 실행 중입니다.")
        if action.value != "train" or not authorized.value:
            state["pending_training"] = None
            raise PermissionError(
                "학습 설정·실행을 선택하고 학습 실행 허용을 켠 뒤 실행 준비를 누르세요."
            )
        pending = state["pending_training"]
        if pending is None:
            raise RuntimeError("먼저 패널의 학습 설정 확인·실행 준비 버튼을 누르세요.")
        if pending != training_signature():
            state["pending_training"] = None
            raise RuntimeError(
                "준비 이후 설정이 변경됐습니다. 설정 확인·실행 준비를 다시 누르세요."
            )
        state["pending_training"] = None
        return execute(from_cell=True)

    for widget in (
        action,
        equation,
        data_mode,
        dense,
        backend,
        qwen,
        authorized,
        batch,
        execution,
        directions,
        inverse,
        generalization,
    ):
        widget.observe(update, names="value")
    for checkbox in arms.checks.values():
        checkbox.observe(update, names="value")
    run.on_click(execute)
    connect.on_click(mount)
    local.on_click(use_local)
    install.on_click(install_clicked)
    dependencies.on_click(dependencies_clicked)
    load_settings.on_click(refresh)
    update()
    trial.observe(selected_model, names="value")
    dashboard = w.VBox(
        [
            w.HTML("<h3>설정</h3>"),
            action,
            storage,
            zip_name,
            w.HBox([connect, local, load_settings]),
            fields,
            dependency_note,
            dependencies,
            install_note,
            install,
            run,
            w.HTML("<h3>결과 · 실행 로그</h3>"),
            out,
        ]
    )
    if show:
        display(dashboard)
    return {
        "dashboard": dashboard,
        "state": state,
        "execute": execute,
        "run_training": run_training,
        "load_settings": refresh,
        "run_button": run,
        "output": out,
        "widgets": {
            "action": action,
            "equation": equation,
            "arms": arms,
            "batch": batch,
            "batch_equations": batch_equations,
            "authorized": authorized,
            "settings": settings_controls,
            "execution": execution,
            "zip": zip_name,
            "storage": storage,
            "qwen": qwen,
            "dense": dense,
            "backend": backend,
            "questions": extra_questions,
            "question": question,
        },
    }
