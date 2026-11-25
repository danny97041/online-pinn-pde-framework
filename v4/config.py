"""Validated V4 settings and independent experiment identities."""

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math

ARMS = ("baseline", "loss_sa", "ff", "ff_loss_sa", "ff_curriculum", "supervised_only")
ARM_LABELS = {
    "baseline": "기본 PINN",
    "loss_sa": "손실항 SA",
    "ff": "푸리에 특징",
    "ff_loss_sa": "푸리에 특징 + 손실항 SA",
    "ff_curriculum": "푸리에 특징 + 커리큘럼",
    "supervised_only": "관측값 회귀",
}
SEEDS = (3234, 3235, 3236, 3237)


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


@dataclass(frozen=True)
class Settings:
    equation: str = "wave2d"
    inverse: bool = False
    arms: tuple = ("baseline",)
    seeds: tuple = (3234,)
    cap: int = 100000
    target: float = 0.05
    stop_first: int = 15000
    check_every: int = 1000
    consecutive: int = 3
    sampling_every: int = 5000
    diagnostic_every: int = 1000
    base_lr: float = 0.004
    initial_coefficient: float | None = None
    width: int = 64
    depth: int = 5
    ff_features: int = 32
    ff_scales: tuple = (0.125, 0.25, 0.5)
    pde_count: int = 800
    observation_count: int = 200
    boundary_count: int = 80
    initial_count: int = 160
    validation_count: int = 256
    evaluation_count: int = 1024
    rad_candidates: int = 4000
    rad_power: float = 1.0
    rad_offset: float = 1.0
    sa_every: int = 50
    sa_ema: float = 0.9
    sa_bounds: tuple = (0.25, 4.0)
    lbfgs_maxiter: int = 500
    generalization: str = "basic"
    block_override: int | None = None
    data_mode: str = "synthetic"
    noise_fraction: float = 0.0
    revision: str = "v4-independent-1"

    def validate(self):
        from .equations import EQUATIONS

        if self.equation not in EQUATIONS or self.generalization not in {
            "basic",
            "interpolation",
            "time_extrapolation",
        }:
            raise ValueError("방정식 또는 일반화 모드를 확인하세요.")
        if type(self.inverse) is not bool or self.data_mode not in {"synthetic", "csv"}:
            raise ValueError("순·역문제 및 데이터 모드를 확인하세요.")
        if (
            not self.arms
            or len(set(self.arms)) != len(self.arms)
            or any(a not in ARMS for a in self.arms)
        ):
            raise ValueError("비교 방법이 올바르지 않습니다.")
        if (
            not self.seeds
            or len(set(self.seeds)) != len(self.seeds)
            or any(type(s) is not int or s < 0 for s in self.seeds)
        ):
            raise ValueError("시드는 서로 다른 음이 아닌 정수여야 합니다.")
        integers = [
            "cap",
            "stop_first",
            "check_every",
            "consecutive",
            "sampling_every",
            "diagnostic_every",
            "width",
            "depth",
            "ff_features",
            "pde_count",
            "observation_count",
            "boundary_count",
            "initial_count",
            "validation_count",
            "evaluation_count",
            "rad_candidates",
            "sa_every",
        ]
        if any(
            type(getattr(self, n)) is not int or getattr(self, n) <= 0 for n in integers
        ):
            raise ValueError("횟수·크기 설정은 양의 정수여야 합니다.")
        if (
            self.cap > 1000000
            or self.rad_candidates < self.pde_count
            or self.rad_candidates > 1000000
        ):
            raise ValueError("학습 한도 또는 RAD 후보 수가 지원 범위를 벗어납니다.")
        if type(self.lbfgs_maxiter) is not int or not 0 <= self.lbfgs_maxiter <= 10000:
            raise ValueError("L-BFGS 반복 수는 0~10000입니다.")
        for n in ("base_lr", "target", "rad_power", "rad_offset"):
            if not math.isfinite(getattr(self, n)) or getattr(self, n) <= 0:
                raise ValueError(n + " must be finite and positive")
        if (
            not 0 < self.target < 1
            or not 0 <= self.sa_ema < 1
            or not 0 <= self.noise_fraction < 1
        ):
            raise ValueError("오차 목표·EMA·노이즈 범위를 확인하세요.")
        if (
            len(self.sa_bounds) != 2
            or not 0 < self.sa_bounds[0] <= self.sa_bounds[1]
            or not all(math.isfinite(v) for v in self.sa_bounds)
        ):
            raise ValueError("SA 범위를 확인하세요.")
        if not self.ff_scales or any(
            not math.isfinite(v) or v <= 0 for v in self.ff_scales
        ):
            raise ValueError("FF scale을 확인하세요.")
        if self.initial_coefficient is not None and (
            not math.isfinite(self.initial_coefficient) or self.initial_coefficient <= 0
        ):
            raise ValueError("초기 계수는 양수여야 합니다.")
        if (
            self.generalization == "time_extrapolation"
            and EQUATIONS[self.equation]["time_axis"] is None
        ):
            raise ValueError("정상 문제에는 시간 외삽을 사용할 수 없습니다.")
        if self.block_override is not None and (
            type(self.block_override) is not int or not 0 <= self.block_override < 16
        ):
            raise ValueError("구역 번호는 0~15입니다.")
        if self.equation == "cylinder_ns" and self.data_mode != "csv":
            raise ValueError("원통 후류에는 외부 데이터 연결이 필요합니다.")
        return self

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        converted = dict(value)
        for name in ("arms", "seeds", "ff_scales", "sa_bounds"):
            if name in converted:
                converted[name] = tuple(converted[name])
        return cls(**converted).validate()

    def smoke(self):
        return replace(
            self,
            cap=10,
            stop_first=15,
            check_every=5,
            diagnostic_every=5,
            sampling_every=5,
            width=12,
            depth=2,
            ff_features=4,
            pde_count=16,
            observation_count=24,
            boundary_count=8,
            initial_count=12,
            validation_count=32,
            evaluation_count=64,
            rad_candidates=64,
            sa_every=2,
            lbfgs_maxiter=2,
        ).validate()


def learning_rates(completed, base):
    import numpy as np

    return tuple(
        base * float(np.interp(completed, [0, 5000, 10000], ratios))
        for ratios in ([1, 0.75, 0.5], [0.001, 0.75, 1])
    )


def group_weights(arm, completed, factors):
    import numpy as np

    if arm == "supervised_only":
        return np.array([0.0, 1.0, 0.0], np.float32)
    weights = np.ones(3, np.float64)
    if arm in {"loss_sa", "ff_loss_sa"}:
        weights *= factors
    if arm == "ff_curriculum":
        share = min(completed // 100 / 100, 1.0) / 3
        return np.array([share, (1 - share) / 2, (1 - share) / 2], np.float32)
    return (weights / weights.sum()).astype(np.float32)


def checkpoint_due(iteration, cap):
    return iteration in {max(1, math.ceil(cap * i / 5)) for i in range(1, 6)}


def target_reached(history, settings):
    eligible = [
        r
        for r in history
        if r["iteration"] >= settings.stop_first
        and r["iteration"] % settings.check_every == 0
    ]
    tail = eligible[-settings.consecutive :]
    return len(tail) == settings.consecutive and all(
        r["validation"]["relative_l1"] is not None
        and r["validation"]["relative_l2"] is not None
        and max(r["validation"]["relative_l1"], r["validation"]["relative_l2"])
        <= settings.target
        for r in tail
    )
