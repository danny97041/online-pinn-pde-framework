"""Synthetic data or explicit CSV contracts; no hidden synthetic fill in CSV mode."""

import csv
import json
from pathlib import Path
import numpy as np

from .equations import EQUATIONS, exact_field, known_forcing
from .generalization import GeneralizationPlan
from .sampling import lhs, permitted_lhs, rng_for
from .storage import sha_file


def make_plan(settings, seed, domain=None):
    spec = EQUATIONS[settings.equation]
    return GeneralizationPlan(
        domain or spec["domain"],
        tuple(a for a in range(len(spec["axes"])) if a != spec["time_axis"]),
        spec["time_axis"],
        settings.generalization,
        seed,
        settings.block_override,
    )


def read_csv(path, axes, outputs, columns=None, transforms=None):
    columns = columns or {}
    transforms = transforms or {}
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        header = reader.fieldnames or []
        rows = list(reader)
    if not rows:
        raise ValueError("관측 CSV가 비었습니다.")

    def column(name, optional=False):
        key = columns.get(name, name)
        if key not in header:
            if optional:
                return np.full(len(rows), np.nan)
            raise ValueError("CSV 열 없음: " + key)
        try:
            values = np.array(
                [float(r[key]) if r[key].strip() else np.nan for r in rows]
            )
        except (ValueError, KeyError):
            raise ValueError("CSV 숫자 형식 오류: " + key) from None
        scale, offset = transforms.get(name, [1.0, 0.0])
        if not np.isfinite([scale, offset]).all() or scale == 0:
            raise ValueError("단위 변환 오류")
        return values * scale + offset

    x = np.column_stack([column(a) for a in axes])
    y = np.column_stack([column(a, True) for a in outputs])
    if (
        not np.isfinite(x).all()
        or np.isinf(y).any()
        or not np.isfinite(y).any(axis=1).all()
    ):
        raise ValueError("CSV 좌표·관측값이 유효하지 않습니다.")
    if len({tuple(v) for v in x}) != len(x):
        raise ValueError("CSV에 중복 좌표가 있습니다. 집계 후 연결하세요.")
    return x, y


class DataSource:
    def __init__(self, settings, contract_path=None):
        self.settings = settings
        self.spec = EQUATIONS[settings.equation]
        self.external = {}
        self.contract = {}
        self.file_hashes = {}
        if settings.data_mode == "csv":
            if not contract_path:
                raise ValueError("데이터 연결 JSON이 필요합니다.")
            path = Path(contract_path)
            self.contract = json.loads(path.read_text(encoding="utf-8-sig"))
            c = self.contract
            if (
                c.get("equation") != settings.equation
                or c.get("canonical_units_confirmed") is not True
            ):
                raise ValueError("방정식·단위 확인이 필요합니다.")
            if (
                "domain" not in c
                or "fixed_coefficient" not in c
                or "field_scales" not in c
                or "residual_scales" not in c
            ):
                raise ValueError("영역·알려진 계수·정규화 척도를 지정하세요.")
            for family, name in c.get("files", {}).items():
                if family not in {
                    "observations",
                    "validation",
                    "evaluation",
                    "boundary",
                    "initial",
                    "initial_velocity",
                    "gauge",
                    "forcing",
                }:
                    raise ValueError("지원하지 않는 데이터 가족")
                file = (path.parent / name).resolve()
                if not file.is_relative_to(path.parent.resolve()):
                    raise ValueError("데이터 CSV는 계약 JSON의 폴더 안에 두세요.")
                self.file_hashes[family] = sha_file(file)
                outputs = (
                    self.residual_names
                    if family == "forcing"
                    else (
                        ["u_t"]
                        if family == "initial_velocity"
                        else self.spec["outputs"]
                    )
                )
                self.external[family] = read_csv(
                    file,
                    self.spec["axes"],
                    outputs,
                    c.get("columns", {}).get(family),
                    c.get("transforms", {}).get(family),
                )
            if not {"observations", "validation"} <= self.external.keys():
                raise ValueError("학습 관측과 별도 검증 CSV가 필요합니다.")
            if (
                set(settings.arms) != {"supervised_only"}
                and "boundary" not in self.external
            ):
                raise ValueError("PINN에는 경계조건 CSV가 필요합니다.")
            if (
                self.spec["time_axis"] is not None
                and set(settings.arms) != {"supervised_only"}
                and "initial" not in self.external
            ):
                raise ValueError("시간 문제에는 초기조건 CSV가 필요합니다.")
            if (
                settings.equation == "wave2d"
                and set(settings.arms) != {"supervised_only"}
                and "initial_velocity" not in self.external
            ):
                raise ValueError("파동 문제에는 초기 속도 u_t CSV가 필요합니다.")
            if len(self.spec["outputs"]) == 3 and set(settings.arms) != {
                "supervised_only"
            }:
                if (
                    "gauge" not in self.external
                    or not np.isfinite(self.external["gauge"][1][:, 2]).all()
                ):
                    raise ValueError(
                        "Navier–Stokes에는 알려진 압력 gauge CSV가 필요합니다."
                    )
            if c.get("forcing_mode") not in {"zero", "csv"}:
                raise ValueError("알려진 외력 zero/csv를 지정하세요.")
            if c["forcing_mode"] == "csv" and "forcing" not in self.external:
                raise ValueError("외력 CSV가 필요합니다.")
            seen = set()
            for family in ("observations", "validation", "evaluation"):
                if family in self.external:
                    points = self.external[family][0]
                    ids = {tuple(row) for row in points}
                    if ids & seen:
                        raise ValueError("관측·검증·평가 CSV 사이 좌표 누수")
                    seen |= ids
            for name, length in [
                ("field_scales", len(self.spec["outputs"])),
                ("residual_scales", len(self.residual_names)),
            ]:
                a = np.array(c[name], float)
                if a.shape != (length,) or not np.isfinite(a).all() or np.any(a <= 0):
                    raise ValueError(name + " 크기·양수 조건 오류")
            if not np.isfinite(c["fixed_coefficient"]) or c["fixed_coefficient"] <= 0:
                raise ValueError("알려진 계수는 양수여야 합니다.")
            if c.get("reference_coefficient") is not None and (
                not np.isfinite(c["reference_coefficient"])
                or c["reference_coefficient"] <= 0
            ):
                raise ValueError("평가용 참조 계수는 양수여야 합니다.")
        self.domain = self.contract.get("domain", self.spec["domain"])
        plan = make_plan(settings, settings.seeds[0], self.domain)
        for family, (x, y) in self.external.items():
            plan.coordinates(x)
            if family in {"initial", "initial_velocity"}:
                axis = self.spec["time_axis"]
                if axis is None or not np.allclose(
                    x[:, axis], self.domain[axis][0], rtol=0, atol=1e-10
                ):
                    raise ValueError("초기조건 좌표는 영역 시작 시간이어야 합니다.")
            if family == "boundary" and settings.equation != "cylinder_ns":
                domain = np.array(self.domain)
                spatial = list(plan.spatial_axes)
                on_edge = np.isclose(
                    x[:, spatial], domain[spatial, 0], atol=1e-10, rtol=0
                ) | np.isclose(x[:, spatial], domain[spatial, 1], atol=1e-10, rtol=0)
                if not on_edge.any(axis=1).all():
                    raise ValueError("경계조건 좌표가 공간 경계에 있지 않습니다.")
        if settings.equation == "cylinder_ns" and not self.contract.get("provenance"):
            raise ValueError(
                "원통 후류 데이터의 출처·사용 조건을 provenance에 기록하세요."
            )
        self.fingerprint = {
            "mode": settings.data_mode,
            "contract": self.contract,
            "files": self.file_hashes,
        }

    @property
    def residual_names(self):
        return (
            ["f_u", "f_v", "f_div"]
            if len(self.spec["outputs"]) == 3
            else ["f_u", "f_v"] if self.settings.equation == "elasticity2d" else ["f_u"]
        )

    @property
    def fixed_coefficient(self):
        return float(self.contract.get("fixed_coefficient", self.spec["fixed"]))

    @property
    def truth(self):
        return (
            self.spec["fixed"]
            if self.settings.data_mode == "synthetic"
            else self.contract.get("reference_coefficient")
        )

    def labels(self, x):
        if self.settings.data_mode != "synthetic":
            raise ValueError("실데이터 모드에서 합성 관측 생성 금지")
        return exact_field(self.settings.equation, x)

    def forcing(self, x):
        if self.settings.data_mode == "synthetic":
            return known_forcing(self.settings.equation, x)
        if self.contract["forcing_mode"] == "zero":
            return np.zeros((len(x), len(self.residual_names)))
        from scipy.interpolate import LinearNDInterpolator

        if not hasattr(self, "_forcing"):
            px, py = self.external["forcing"]
            if not np.isfinite(py).all():
                raise ValueError("모든 외력 성분이 필요합니다.")
            self._forcing = LinearNDInterpolator(px, py)
        values = np.asarray(self._forcing(x))
        if not np.isfinite(values).all():
            raise ValueError(
                "외력 CSV의 보간 영역 밖입니다. 영역 전체를 덮는 알려진 외력이 필요합니다."
            )
        return values

    def scales(self):
        if self.settings.data_mode == "csv":
            return {
                "field": self.contract["field_scales"],
                "residual": self.contract["residual_scales"],
                "provenance": "explicit canonical-units data contract",
            }
        # Prescribed benchmark characteristic scales, not sampled holdout labels.
        characteristic = {
            "reaction_diffusion": [0.2],
            "elasticity2d": [1.0, 0.5],
            "kovasznay": [2.0, 1.0, 1.0],
            "taylor_green": [1.0, 1.0, 0.5],
        }
        field = np.asarray(
            characteristic.get(
                self.settings.equation, [1.0] * len(self.spec["outputs"])
            )
        )
        length = min(
            self.domain[a][1] - self.domain[a][0]
            for a in range(len(self.domain))
            if a != self.spec["time_axis"]
        )
        return {
            "field": field.tolist(),
            "residual": [float(max(field.max() / min(length, length**2), 0.001))]
            * len(self.residual_names),
            "provenance": "prescribed synthetic characteristic field/domain scales; no holdout labels read",
        }

    def training(self, plan, cycle):
        s = self.settings
        arrays = {}
        if self.external:
            for family in ("observations", "boundary", "initial", "gauge"):
                if family in self.external:
                    x, y = self.external[family]
                    mask = plan.training_mask(x, family)
                    if not mask.any():
                        raise ValueError(family + " 학습점이 없습니다.")
                    arrays[family] = x[mask].copy()
                    arrays["label_" + family] = y[mask].copy()
            if "initial_velocity" in self.external:
                x, y = self.external["initial_velocity"]
                mask = plan.training_mask(x, "initial")
                arrays.update(
                    initial_velocity_x=x[mask].copy(), initial_velocity=y[mask].copy()
                )
            if (
                plan.mode == "interpolation"
                and plan.training_mask(self.external["observations"][0]).all()
            ):
                raise ValueError("제외 구역에 실제 관측이 없습니다.")
        else:
            x = permitted_lhs(
                plan, s.observation_count, plan.seed, "observations", cycle
            )
            y = self.labels(x)
            if s.noise_fraction:
                y = (
                    y
                    + rng_for(plan.seed, "noise", cycle).normal(size=y.shape)
                    * np.asarray(self.scales()["field"])
                    * s.noise_fraction
                )
            arrays.update(observations=x, label_observations=y)
            spatial = plan.spatial_axes
            boundaries = []
            for a in spatial:
                for end in (0, 1):
                    b = permitted_lhs(
                        plan,
                        s.boundary_count,
                        plan.seed,
                        "boundary",
                        cycle + 100 * a + end,
                    )
                    b[:, a] = plan.domain[a][end]
                    boundaries.append(b)
            b = np.concatenate(boundaries)
            arrays.update(boundary=b, label_boundary=self.labels(b))
            if plan.time_axis is not None:
                x = permitted_lhs(plan, s.initial_count, plan.seed, "initial", cycle)
                x[:, plan.time_axis] = plan.domain[plan.time_axis][0]
                arrays.update(initial=x, label_initial=self.labels(x))
                if s.equation == "wave2d":
                    arrays["initial_velocity"] = np.zeros((len(x), 1))
            if len(self.spec["outputs"]) == 3:
                x = permitted_lhs(plan, 16, plan.seed, "gauge", cycle)
                x[:, plan.spatial_axes] = np.array(plan.domain)[
                    list(plan.spatial_axes), 0
                ]
                arrays.update(gauge=x, label_gauge=self.labels(x))
        for family in ("observations", "boundary", "initial", "gauge"):
            if family in arrays:
                plan.assert_training(arrays[family], family)
        return arrays

    def validation(self, plan):
        if self.external:
            x, y = self.external["validation"]
            mask = plan.validation_mask(x)
            if not mask.any():
                raise ValueError("허용 학습 영역에 중단 검증점이 없습니다.")
            return {"x": x[mask].copy(), "y": y[mask].copy()}
        x = permitted_lhs(
            plan, self.settings.validation_count, 991823, "observations", 0
        )
        return {"x": x, "y": self.labels(x)}

    def evaluation(self, plan):
        if self.external:
            if "evaluation" not in self.external:
                return None
            x, y = self.external["evaluation"]
            mask = plan.evaluation_mask(x)
            if not mask.any():
                raise ValueError("평가 영역에 별도 관측이 없습니다.")
            return {"x": x[mask].copy(), "y": y[mask].copy()}
        if plan.mode == "basic":
            x = lhs(
                plan.domain,
                self.settings.evaluation_count,
                rng_for(93235, "evaluation"),
            )
        else:
            x = plan.fixed_evaluation_points(per_spatial_axis=16, time_count=9)
        return {"x": x, "y": self.labels(x)}
