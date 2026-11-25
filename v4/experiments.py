"""Explicit batch plans and sensitivity cases; no automatic optimization search."""

from dataclasses import replace
from itertools import product
import json
from pathlib import Path
from .config import digest
from .data import DataSource
from .equations import EQUATIONS
from .runner import run_all
from .storage import write_json


def experiment_plan(
    base, equations=None, directions=(False,), noise=(0.0,), initial_multipliers=(1.0,)
):
    base.validate()
    equations = equations or (base.equation,)
    configurations = []
    if base.data_mode == "csv" and tuple(equations) != (base.equation,):
        raise ValueError("CSV 계획은 계약에 지정한 방정식만 사용합니다.")
    if base.data_mode == "csv" and noise != (0.0,):
        raise ValueError(
            "실제 관측은 파일 값을 유지합니다. 노이즈 대조 파일은 별도 계약으로 연결하세요."
        )
    for eq, inverse, level, multiplier in product(
        equations, directions, noise, initial_multipliers
    ):
        if not inverse and multiplier != 1.0:
            continue
        initial = (
            base.initial_coefficient
            if base.initial_coefficient is not None
            else EQUATIONS[eq]["initial"]
        ) * multiplier
        setting = replace(
            base,
            equation=eq,
            inverse=inverse,
            noise_fraction=level,
            initial_coefficient=initial if inverse else None,
        ).validate()
        configurations.append(setting)
    keys = [digest(s.to_dict()) for s in configurations]
    if len(set(keys)) != len(keys):
        raise ValueError("계획에 같은 실험 조건이 중복되었습니다.")
    if not configurations or len(configurations) > 200:
        raise ValueError("계획 조건 수는 1~200입니다.")
    return configurations


def execute_plan(
    root, configurations, results_zip, resume_zip, training_authorized=False, log=print
):
    if training_authorized is not True:
        raise PermissionError("학습 계획 실행 허용이 필요합니다.")
    root = Path(root)
    plan = [s.to_dict() for s in configurations]
    path = root / "plan.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != json.loads(
        json.dumps(plan)
    ):
        raise ValueError("재개 계획이 저장된 계획과 다릅니다.")
    write_json(path, plan)
    reports = []
    for settings in configurations:
        source = DataSource(
            settings, root / "inputs/data.json" if settings.data_mode == "csv" else None
        )
        reports.extend(
            run_all(root, settings, source, results_zip, resume_zip, True, log)
        )
    return reports
