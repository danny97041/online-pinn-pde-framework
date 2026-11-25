"""Manufactured benchmarks with explicit known forcing, separate from inference.

Truth constants below belong to synthetic data generation/auditing, never to the
learned coefficient initializer. Real-data forcing is supplied by its contract.
"""

import numpy as np

EQUATIONS = {
    "wave2d": dict(
        label="2차원 파동",
        domain=[[0, 1]] * 3,
        axes=["x", "y", "t"],
        outputs=["u"],
        time_axis=2,
        coefficient="lambda1",
        fixed=1.0,
        initial=0.5,
    ),
    "heat2d": dict(
        label="2차원 열전도",
        domain=[[0, 1]] * 3,
        axes=["x", "y", "t"],
        outputs=["u"],
        time_axis=2,
        coefficient="alpha",
        fixed=0.1,
        initial=0.05,
    ),
    "poisson2d": dict(
        label="2차원 포아송",
        domain=[[0, 1]] * 2,
        axes=["x", "y"],
        outputs=["u"],
        time_axis=None,
        coefficient="source_amplitude",
        fixed=1.0,
        initial=0.5,
    ),
    "burgers": dict(
        label="버거스",
        domain=[[-1, 1], [0, 1]],
        axes=["x", "t"],
        outputs=["u"],
        time_axis=1,
        coefficient="nu",
        fixed=0.01 / np.pi,
        initial=0.0015,
    ),
    "kovasznay": dict(
        label="Navier–Stokes · Kovasznay",
        domain=[[-0.5, 1], [-0.5, 1.5]],
        axes=["x", "y"],
        outputs=["u", "v", "p"],
        time_axis=None,
        coefficient="nu",
        fixed=0.025,
        initial=0.0125,
    ),
    "taylor_green": dict(
        label="Navier–Stokes · Taylor–Green",
        domain=[[-np.pi, np.pi], [-np.pi, np.pi], [0, 1]],
        axes=["x", "y", "t"],
        outputs=["u", "v", "p"],
        time_axis=2,
        coefficient="nu",
        fixed=0.01,
        initial=0.005,
    ),
    "darcy2d": dict(
        label="2차원 Darcy",
        domain=[[0, 1]] * 2,
        axes=["x", "y"],
        outputs=["u"],
        time_axis=None,
        coefficient="permeability_multiplier",
        fixed=1.0,
        initial=0.5,
    ),
    "reaction_diffusion": dict(
        label="반응–확산",
        domain=[[0, 1]] * 3,
        axes=["x", "y", "t"],
        outputs=["u"],
        time_axis=2,
        coefficient="D",
        fixed=0.01,
        initial=0.005,
    ),
    "advection_diffusion": dict(
        label="이류–확산",
        domain=[[0, 1]] * 3,
        axes=["x", "y", "t"],
        outputs=["u"],
        time_axis=2,
        coefficient="D",
        fixed=0.1,
        initial=0.05,
    ),
    "helmholtz": dict(
        label="Helmholtz",
        domain=[[0, 1]] * 2,
        axes=["x", "y"],
        outputs=["u"],
        time_axis=None,
        coefficient="wavenumber_squared",
        fixed=1.0,
        initial=0.5,
    ),
    "elasticity2d": dict(
        label="2차원 선형탄성 · 평면변형률",
        domain=[[0, 1]] * 2,
        axes=["x", "y"],
        outputs=["u", "v"],
        time_axis=None,
        coefficient="young_modulus",
        fixed=1.0,
        initial=0.5,
    ),
    "cylinder_ns": dict(
        label="Navier–Stokes · 원통 후류 (외부 데이터)",
        domain=[[1, 8], [-2, 2], [0, 1]],
        axes=["x", "y", "t"],
        outputs=["u", "v", "p"],
        time_axis=2,
        coefficient="nu",
        fixed=0.01,
        initial=0.005,
    ),
}


def exact_field(name, z, xp=np):
    x, y = z[:, 0], z[:, 1]
    pi = np.pi
    if name == "cylinder_ns":
        raise ValueError("원통 후류에는 해석 정답을 생성하지 않습니다.")
    if name == "burgers":
        return (0.5 - 0.5 * xp.tanh((x - 0.5 * y) / (4 * (0.01 / pi))))[:, None]
    if name == "kovasznay":
        lam = 20 - np.sqrt(400 + 4 * pi * pi)
        e = xp.exp(lam * x)
        return xp.stack(
            [
                1 - e * xp.cos(2 * pi * y),
                lam / (2 * pi) * e * xp.sin(2 * pi * y),
                0.5 * (1 - e**2),
            ],
            axis=1,
        )
    if name == "taylor_green":
        e = xp.exp(-0.02 * z[:, 2])
        return xp.stack(
            [
                -xp.cos(x) * xp.sin(y) * e,
                xp.sin(x) * xp.cos(y) * e,
                -0.25 * (xp.cos(2 * x) + xp.cos(2 * y)) * e**2,
            ],
            axis=1,
        )
    s = xp.sin(pi * x) * xp.sin(pi * y)
    if name == "elasticity2d":
        return xp.stack([s, 0.5 * s], axis=1)
    if name in {"poisson2d", "darcy2d", "helmholtz"}:
        return s[:, None]
    if name == "wave2d":
        return (s * xp.cos(np.sqrt(2) * pi * z[:, 2]))[:, None]
    if name == "heat2d":
        return (s * xp.exp(-0.2 * pi * pi * z[:, 2]))[:, None]
    if name == "reaction_diffusion":
        return (0.2 * s * xp.exp(-z[:, 2]))[:, None]
    if name == "advection_diffusion":
        return (s * xp.exp(-z[:, 2]))[:, None]
    raise ValueError(name)


def known_forcing(name, z, xp=np):
    x, y = z[:, 0], z[:, 1]
    s = xp.sin(np.pi * x) * xp.sin(np.pi * y)
    zero = x * 0
    if name in {"kovasznay", "taylor_green", "cylinder_ns"}:
        return xp.stack([zero, zero, zero], axis=1)
    if name == "elasticity2d":
        nu = 0.25  # Known Poisson ratio, not inferred jointly with E.
        lam, mu = nu / ((1 + nu) * (1 - 2 * nu)), 1 / (2 * (1 + nu))
        mixed = np.pi**2 * xp.cos(np.pi * x) * xp.cos(np.pi * y)
        return xp.stack(
            [
                (lam + 3 * mu) * np.pi**2 * s - 0.5 * (lam + mu) * mixed,
                0.5 * (lam + 3 * mu) * np.pi**2 * s - (lam + mu) * mixed,
            ],
            axis=1,
        )
    if name == "poisson2d":
        return (2 * np.pi**2 * s)[:, None]
    if name == "helmholtz":
        return ((2 * np.pi**2 - 1) * s)[:, None]
    if name == "darcy2d":
        sx, sy = xp.sin(np.pi * x), xp.sin(np.pi * y)
        cx, cy = xp.cos(np.pi * x), xp.cos(np.pi * y)
        k = 1 + 0.5 * s
        return (
            2 * np.pi**2 * k * s - 0.5 * np.pi**2 * ((cx * sy) ** 2 + (sx * cy) ** 2)
        )[:, None]
    if name == "reaction_diffusion":
        value = 0.2 * s * xp.exp(-z[:, 2])
        return ((-2 + 0.02 * np.pi**2) * value + value**3)[:, None]
    if name == "advection_diffusion":
        e = xp.exp(-z[:, 2])
        return (
            (-1 + 0.2 * np.pi**2) * s * e
            + 0.5 * np.pi * xp.cos(np.pi * x) * xp.sin(np.pi * y) * e
            + 0.25 * np.pi * xp.sin(np.pi * x) * xp.cos(np.pi * y) * e
        )[:, None]
    return zero[:, None]


def residual(name, x, q, g, h, coefficient, forcing, xp=np):
    """q(N,C), g(N,C,D), full Hessian h(N,C,D,D); all forcing is known input."""
    u, ux, uxx = q[:, 0], g[:, 0, 0], h[:, 0, 0, 0]
    if name == "burgers":
        return (g[:, 0, 1] + u * ux - coefficient * uxx)[:, None] - forcing
    uy, lap = g[:, 0, 1], uxx + h[:, 0, 1, 1]
    if name in {"kovasznay", "taylor_green", "cylinder_ns"}:
        v = q[:, 1]
        ut = g[:, 0, 2] if name != "kovasznay" else u * 0
        vt = g[:, 1, 2] if name != "kovasznay" else u * 0
        return (
            xp.stack(
                [
                    ut + u * ux + v * uy + g[:, 2, 0] - coefficient * lap,
                    vt
                    + u * g[:, 1, 0]
                    + v * g[:, 1, 1]
                    + g[:, 2, 1]
                    - coefficient * (h[:, 1, 0, 0] + h[:, 1, 1, 1]),
                    ux + g[:, 1, 1],
                ],
                axis=1,
            )
            - forcing
        )
    if name == "elasticity2d":
        nu = 0.25
        lam = coefficient * nu / ((1 + nu) * (1 - 2 * nu))
        mu = coefficient / (2 * (1 + nu))
        return (
            xp.stack(
                [
                    -(lam + 2 * mu) * uxx
                    - mu * h[:, 0, 1, 1]
                    - (lam + mu) * h[:, 1, 0, 1],
                    -mu * h[:, 1, 0, 0]
                    - (lam + 2 * mu) * h[:, 1, 1, 1]
                    - (lam + mu) * h[:, 0, 0, 1],
                ],
                axis=1,
            )
            - forcing
        )
    if name == "wave2d":
        return (h[:, 0, 2, 2] - coefficient * lap)[:, None] - forcing
    if name == "heat2d":
        return (g[:, 0, 2] - coefficient * lap)[:, None] - forcing
    if name == "poisson2d":
        return -lap[:, None] - coefficient * forcing
    if name == "helmholtz":
        return (-lap - coefficient * u)[:, None] - forcing
    if name == "darcy2d":
        sx, sy = xp.sin(np.pi * x[:, 0]), xp.sin(np.pi * x[:, 1])
        cx, cy = xp.cos(np.pi * x[:, 0]), xp.cos(np.pi * x[:, 1])
        return (
            -coefficient
            * (
                (1 + 0.5 * sx * sy) * lap
                + 0.5 * np.pi * cx * sy * ux
                + 0.5 * np.pi * sx * cy * uy
            )
        )[:, None] - forcing
    if name == "reaction_diffusion":
        return (g[:, 0, 2] - coefficient * lap - u + u**3)[:, None] - forcing
    if name == "advection_diffusion":
        return (g[:, 0, 2] + 0.5 * ux + 0.25 * uy - coefficient * lap)[
            :, None
        ] - forcing
    raise ValueError(name)
