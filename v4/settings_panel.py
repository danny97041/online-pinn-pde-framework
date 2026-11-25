"""Described per-variable controls; serialization stays internal."""

from .config import Settings


def settings_editor():
    import ipywidgets as w

    specs = {
        "학습과 중단": [
            (
                "cap",
                "Adam 최대 반복",
                "각 방법·시드의 총 업데이트 한도입니다. 목표 미달이어도 이 한도에서 종료합니다.",
            ),
            (
                "target",
                "목표 오차 (%)",
                "고정 검증점의 L1과 L2가 모두 이 백분율 이하여야 합니다. 5를 입력하면 5%입니다.",
            ),
            (
                "stop_first",
                "중단 허용 시작",
                "이 반복부터 목표 달성에 의한 조기중단을 허용합니다.",
            ),
            (
                "check_every",
                "목표 확인 간격",
                "이 반복 간격마다 고정 검증점에서 목표를 확인합니다.",
            ),
            (
                "consecutive",
                "연속 만족 횟수",
                "목표를 연속으로 이 횟수만큼 만족해야 Adam을 종료합니다.",
            ),
            (
                "diagnostic_every",
                "진단 기록 간격",
                "손실별 기울기 등 진단을 기록하는 간격입니다. ZIP 저장 간격과 다릅니다.",
            ),
            (
                "base_lr",
                "기준 학습률",
                "시작 신경망 학습률입니다. 신경망/계수 비율은 0·5천·1만 회에서 1/0.001, 0.75/0.75, 0.5/1로 변합니다.",
            ),
            (
                "initial_coefficient",
                "역문제 초기 계수",
                "자동은 방정식별 기본값입니다. 직접 입력은 양수만 허용하며 관측값 회귀에는 적용되지 않습니다.",
            ),
            (
                "lbfgs_maxiter",
                "L-BFGS 최대 반복",
                "Adam 뒤 후처리 반복 한도입니다. 0이면 생략하며 함수 평가 횟수와는 다릅니다.",
            ),
        ],
        "신경망과 푸리에 특징": [
            ("width", "은닉층 뉴런 수", "각 은닉층의 너비입니다. 기본값은 64입니다."),
            ("depth", "은닉층 수", "tanh 은닉층의 개수입니다. 기본값은 5입니다."),
            (
                "ff_features",
                "FF 뱅크별 특징 수",
                "각 주파수 뱅크의 수입니다. 뱅크마다 sin/cos가 생성됩니다. FF 방법에서만 사용합니다.",
            ),
            (
                "ff_scales",
                "FF 주파수 척도",
                "쉼표로 구분한 양수입니다. 각 값이 한 뱅크의 무작위 주파수 표준편차입니다. 예: 0.125, 0.25, 0.5",
            ),
        ],
        "학습점과 고정 평가점": [
            (
                "sampling_every",
                "학습점 갱신 간격",
                "PDE는 RAD, 나머지 합성 학습점은 LHS로 갱신합니다. CSV 관측·조건점은 유지합니다.",
            ),
            (
                "pde_count",
                "PDE 학습점 수",
                "매 주기에 사용할 PDE 좌표 수입니다. 관측값 회귀에는 사용하지 않습니다.",
            ),
            (
                "observation_count",
                "합성 관측점 수",
                "합성 참조값으로 생성할 학습 관측점 수입니다. CSV 행 수를 바꾸지 않습니다.",
            ),
            (
                "boundary_count",
                "경계면당 점 수",
                "합성 문제의 각 공간 경계면에 생성하는 점 수입니다.",
            ),
            (
                "initial_count",
                "초기조건 점 수",
                "합성 시간 의존 문제의 초기 시점 좌표 수입니다. 정상 문제에는 적용하지 않습니다.",
            ),
            (
                "validation_count",
                "고정 검증점 수",
                "합성 문제의 중단 판정용 고정 좌표 수입니다. CSV는 연결한 검증 행을 사용합니다.",
            ),
            (
                "evaluation_count",
                "고정 평가점 수",
                "합성 문제의 결과 보고용 고정 좌표 수입니다. 학습·중단 판정에는 쓰지 않습니다.",
            ),
            (
                "noise_fraction",
                "합성 관측 노이즈 (%)",
                "특성 물리장 크기에 대한 Gaussian 노이즈 표준편차입니다. 0이면 노이즈를 추가하지 않습니다.",
            ),
        ],
        "RAD와 손실항 SA": [
            (
                "rad_candidates",
                "RAD 후보점 수",
                "PDE 학습점을 선택할 후보 좌표 수이며 PDE 학습점 수 이상이어야 합니다.",
            ),
            (
                "rad_power",
                "RAD 잔차 지수",
                "큰 PDE 잔차에 대한 선택 편향을 조절합니다. 기본값 1입니다.",
            ),
            (
                "rad_offset",
                "RAD 균일 혼합값",
                "균일 선택 성분입니다. 값이 커질수록 잔차 집중이 약해집니다. 기본값 1입니다.",
            ),
            (
                "sa_every",
                "SA 갱신 간격",
                "손실항 SA 방법에서 그룹 가중치를 조절하는 반복 간격입니다.",
            ),
            (
                "sa_ema",
                "SA 평활 계수",
                "0 이상 1 미만입니다. 클수록 이전 가중치를 더 유지합니다.",
            ),
            (
                "sa_bounds",
                "SA 배율 하한·상한",
                "쉼표로 구분한 두 양수입니다. 예: 0.25, 4.0. 정규화 전 그룹 배율의 제한입니다.",
            ),
        ],
        "보간 제외 구역": [
            (
                "block_override",
                "제외 구역 번호",
                "자동은 시드로 16구역 중 하나를 선택합니다. 직접 선택은 0~15이며 보간 실험에만 적용합니다.",
            ),
        ],
    }

    class Editor(w.VBox):
        def __init__(self):
            self.controls = {}
            self.auto = {}
            self._disabled = False
            self.context = {}
            self.stored = Settings().to_dict()
            pages = []
            for section, rows in specs.items():
                boxes = []
                for name, label, help_text in rows:
                    value = self.stored[name]
                    options = dict(
                        description=label,
                        style={"description_width": "170px"},
                        layout=w.Layout(width="100%"),
                    )
                    if name in {"initial_coefficient", "block_override"}:
                        self.auto[name] = w.Checkbox(
                            description="자동 선택", value=True
                        )
                        control = (
                            w.IntText(value=0, **options)
                            if name == "block_override"
                            else w.FloatText(value=0.5, **options)
                        )
                        self.auto[name].observe(self._auto_changed, names="value")
                    elif name in {"ff_scales", "sa_bounds"}:
                        control = w.Text(
                            value=", ".join(str(v) for v in value), **options
                        )
                    elif isinstance(value, int):
                        control = w.IntText(value=value, **options)
                    else:
                        control = w.FloatText(
                            value=(
                                value * 100
                                if name in {"target", "noise_fraction"}
                                else value
                            ),
                            **options,
                        )
                    self.controls[name] = control
                    boxes.append(
                        w.VBox(
                            [
                                w.HTML(f"<small>{name} — {help_text}</small>"),
                                *([self.auto[name]] if name in self.auto else []),
                                control,
                            ]
                        )
                    )
                pages.append(w.VBox(boxes))
            accordion = w.Accordion(children=pages)
            for index, title in enumerate(specs):
                accordion.set_title(index, title)
            accordion.selected_index = None
            super().__init__(
                [
                    w.HTML(
                        "항목을 펼쳐 값을 변경하세요. 재개 시 저장 설정을 그대로 불러옵니다. 변경한 설정은 새 독립 실험에 사용하세요."
                    ),
                    accordion,
                ]
            )
            self._apply_enabled()

        def _auto_changed(self, _=None):
            self._apply_enabled()

        def set_context(self, arms, inverse, generalization, data_mode, timed):
            self.context = dict(
                arms=arms,
                inverse=inverse,
                generalization=generalization,
                data_mode=data_mode,
                timed=timed,
            )
            self._apply_enabled()

        def _apply_enabled(self):
            c = self.context
            arms = c.get("arms", ())
            inactive = set()
            if c:
                if not any(k.startswith("ff") for k in arms):
                    inactive.update(("ff_features", "ff_scales"))
                if not any(k in {"loss_sa", "ff_loss_sa"} for k in arms):
                    inactive.update(("sa_every", "sa_ema", "sa_bounds"))
                if not any(k != "supervised_only" for k in arms):
                    inactive.update(
                        ("pde_count", "rad_candidates", "rad_power", "rad_offset")
                    )
                if not c["inverse"] or not any(k != "supervised_only" for k in arms):
                    inactive.add("initial_coefficient")
                if c["generalization"] != "interpolation":
                    inactive.add("block_override")
                if c["data_mode"] == "csv":
                    inactive.update(
                        (
                            "observation_count",
                            "boundary_count",
                            "initial_count",
                            "validation_count",
                            "evaluation_count",
                            "noise_fraction",
                        )
                    )
                if not c["timed"]:
                    inactive.add("initial_count")
            for name, control in self.controls.items():
                control.disabled = (
                    self._disabled
                    or name in inactive
                    or (name in self.auto and self.auto[name].value)
                )
            for name, control in self.auto.items():
                control.disabled = self._disabled or name in inactive

        @property
        def value(self):
            values = dict(self.stored)
            for name, control in self.controls.items():
                value = control.value
                if name in self.auto and self.auto[name].value:
                    value = None
                elif name in {"ff_scales", "sa_bounds"}:
                    try:
                        value = tuple(float(v.strip()) for v in value.split(","))
                    except ValueError as exc:
                        raise ValueError(
                            name + ": 쉼표로 구분한 숫자를 입력하세요."
                        ) from exc
                elif name in {"target", "noise_fraction"}:
                    value /= 100
                values[name] = value
            return values

        @value.setter
        def value(self, values):
            self.stored = dict(values)
            for name, control in self.controls.items():
                value = values[name]
                if name in self.auto:
                    self.auto[name].value = value is None
                    if value is None:
                        continue
                if name in {"ff_scales", "sa_bounds"}:
                    value = ", ".join(str(v) for v in value)
                elif name in {"target", "noise_fraction"}:
                    value *= 100
                control.value = value
            self._apply_enabled()

        @property
        def disabled(self):
            return self._disabled

        @disabled.setter
        def disabled(self, value):
            self._disabled = value
            self._apply_enabled()

    return Editor()
