"""Keyed LHS and RAD. Only PDE points use model-dependent adaptation."""

import hashlib
import numpy as np


def rng_for(seed, family, cycle=0):
    key = f"{seed}:{family}:{cycle}".encode()
    return np.random.default_rng(
        int.from_bytes(hashlib.sha256(key).digest()[:8], "little")
    )


def lhs(bounds, count, rng):
    b = np.asarray(bounds, float)
    if count < 1:
        return np.empty((0, len(b)))
    unit = np.column_stack(
        [(rng.permutation(count) + rng.random(count)) / count for _ in b]
    )
    return b[:, 0] + unit * (b[:, 1] - b[:, 0])


def permitted_lhs(plan, count, seed, family, cycle):
    rng = rng_for(seed, family, cycle)
    result = []
    remaining = count
    for _ in range(32):
        points = lhs(plan.training_domain, max(remaining, 16), rng)
        points = points[plan.training_mask(points, family)]
        result.append(points[:remaining])
        remaining -= min(remaining, len(points))
        if not remaining:
            values = np.concatenate(result)
            plan.assert_training(values, family)
            return values
    raise ValueError("학습 영역에 필요한 좌표를 확보하지 못했습니다.")


def rad_select(pool, scores, count, rng, power=1.0, offset=1.0):
    scores = np.asarray(scores, float)
    if (
        scores.ndim != 1
        or len(scores) != len(pool)
        or len(pool) < count
        or not np.isfinite(scores).all()
        or np.any(scores < 0)
    ):
        raise ValueError("RAD 점수·후보 수 오류")
    maximum = scores.max(initial=0)
    scaled = (scores / maximum) ** power if maximum else np.zeros_like(scores)
    mean = scaled.mean()
    weights = (scaled / mean if mean else np.zeros_like(scores)) + offset
    probabilities = weights / weights.sum()
    indices = rng.choice(len(pool), size=count, replace=False, p=probabilities)
    return np.asarray(pool)[indices].copy(), {
        "power": power,
        "offset": offset,
        "candidate_count": len(pool),
        "score_max": float(maximum),
        "uniform_probability_mass": float(offset / (1 + offset)) if mean else 1.0,
    }
