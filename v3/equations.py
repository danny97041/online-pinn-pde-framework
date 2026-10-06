"""Candidate exact-solution/derivative audits, NOT neural training success.

NumPy finite differences are independent of the TensorFlow autodiff implementation.
Every manufactured forcing is written explicitly, not generated from the tested residual.
"""

import json
from pathlib import Path
import numpy as np

CANDIDATES = {
    "heat2d": {
        "domain": [[0, 1], [0, 1], [0, 1]],
        "outputs": ["u"],
        "priority": 1,
        "coefficient": 0.1,
        "boundary": "homogeneous Dirichlet",
        "time_axis": 2,
    },
    "poisson2d": {
        "domain": [[0, 1], [0, 1]],
        "outputs": ["u"],
        "priority": 2,
        "boundary": "homogeneous Dirichlet",
        "time_axis": None,
    },
    "burgers": {
        "domain": [[-1, 1], [0, 1]],
        "outputs": ["u"],
        "priority": 3,
        "coefficient": 0.01 / np.pi,
        "boundary": "exact nonzero Dirichlet",
        "time_axis": 1,
        "note": "Traveling viscous shock; NOT thesis sine-initial-condition benchmark",
    },
    "kovasznay_forward": {
        "domain": [[-0.5, 1], [-0.5, 1.5]],
        "outputs": ["u", "v", "p"],
        "priority": 4,
        "coefficient": 0.025,
        "boundary": "exact velocity Dirichlet; pressure gauge",
        "time_axis": None,
    },
    "kovasznay_inverse": {
        "domain": [[-0.5, 1], [-0.5, 1.5]],
        "outputs": ["u", "v", "p"],
        "priority": 5,
        "coefficient": 0.025,
        "boundary": "exact velocity Dirichlet; pressure gauge",
        "time_axis": None,
        "note": "Unknown viscosity; exact-field sensitivity is NOT inverse-training identifiability proof",
    },
    "taylor_green": {
        "domain": [[-np.pi, np.pi], [-np.pi, np.pi], [0, 1]],
        "outputs": ["u", "v", "p"],
        "priority": 6,
        "coefficient": 0.01,
        "boundary": "periodic values and first derivatives; pressure gauge",
        "time_axis": 2,
    },
    "darcy2d": {
        "domain": [[0, 1], [0, 1]],
        "outputs": ["u"],
        "priority": 7,
        "boundary": "homogeneous Dirichlet",
        "time_axis": None,
        "note": "Known heterogeneous permeability, manufactured forcing; field-coefficient inverse deferred",
    },
    "reaction_diffusion": {
        "domain": [[0, 1], [0, 1], [0, 1]],
        "outputs": ["u"],
        "priority": 8,
        "coefficient": 0.01,
        "boundary": "homogeneous Dirichlet",
        "time_axis": 2,
        "note": "Forced Allen-Cahn exact audit; NOT unforced pattern-formation benchmark",
    },
}
DEFERRED = {
    "shallow_water": {
        "status": "not_implemented",
        "requirements": [
            "positive depth",
            "well-balanced bed/source",
            "mass conservation",
            "wet/dry and shock treatment",
            "trusted reference solver",
        ],
        "reason": "Do not apply the smooth Wave loss/10% criterion blindly to a hyperbolic system",
    },
    "darcy_inverse_field": {
        "status": "not_implemented",
        "requirements": [
            "permeability observations/prior",
            "boundary flux data",
            "nonuniqueness/regularization study",
        ],
    },
}


def candidate_field(name, z, xp=np):
    x = z[:, 0:1]
    y = z[:, 1:2]
    pi = np.pi
    if name == "burgers":
        nu = 0.01 / pi
        return 0.5 - 0.5 * xp.tanh(0.5 * (x - 0.5 * y) / (2 * nu))
    if name.startswith("kovasznay"):
        lam = 20 - np.sqrt(400 + 4 * pi * pi)
        e = xp.exp(lam * x)
        return xp.concatenate(
            [
                1 - e * xp.cos(2 * pi * y),
                lam / (2 * pi) * e * xp.sin(2 * pi * y),
                0.5 * (1 - e * e),
            ],
            axis=1,
        )
    if name == "taylor_green":
        t = z[:, 2:3]
        e = xp.exp(-0.02 * t)
        return xp.concatenate(
            [
                -xp.cos(x) * xp.sin(y) * e,
                xp.sin(x) * xp.cos(y) * e,
                -0.25 * (xp.cos(2 * x) + xp.cos(2 * y)) * e * e,
            ],
            axis=1,
        )
    s = xp.sin(pi * x) * xp.sin(pi * y)
    if name in ("poisson2d", "darcy2d"):
        return s
    if name == "heat2d":
        return s * xp.exp(-0.2 * pi * pi * z[:, 2:3])
    if name == "reaction_diffusion":
        return 0.2 * s * xp.exp(-z[:, 2:3])
    raise ValueError(name)


def candidate_residual(name, z, q, g, h, coefficient=None):
    """Shapes: q(N,C), g(N,C,D), h diagonal Hessian(N,C,D)."""
    u = q[:, 0]
    ux = g[:, 0, 0]
    uxx = h[:, 0, 0]
    if name == "burgers":
        nu = CANDIDATES[name]["coefficient"] if coefficient is None else coefficient
        return (g[:, 0, 1] + u * ux - nu * uxx)[:, None]
    uy = g[:, 0, 1]
    lap = uxx + h[:, 0, 1]
    if name.startswith("kovasznay") or name == "taylor_green":
        nu = CANDIDATES[name]["coefficient"] if coefficient is None else coefficient
        v = q[:, 1]
        ut = g[:, 0, 2] if name == "taylor_green" else 0
        vt = g[:, 1, 2] if name == "taylor_green" else 0
        return np.column_stack(
            (
                ut + u * ux + v * uy + g[:, 2, 0] - nu * lap,
                vt + u * g[:, 1, 0] + v * g[:, 1, 1] + g[:, 2, 1] - nu * (h[:, 1, 0] + h[:, 1, 1]),
                ux + g[:, 1, 1],
            )
        )
    if name == "heat2d":
        return (g[:, 0, 2] - 0.1 * lap)[:, None]
    if name == "poisson2d":
        return (-lap - 2 * np.pi**2 * np.sin(np.pi * z[:, 0]) * np.sin(np.pi * z[:, 1]))[:, None]
    if name == "darcy2d":
        sx, sy = np.sin(np.pi * z[:, 0]), np.sin(np.pi * z[:, 1])
        cx, cy = np.cos(np.pi * z[:, 0]), np.cos(np.pi * z[:, 1])
        k = 1 + 0.5 * sx * sy
        kx = 0.5 * np.pi * cx * sy
        ky = 0.5 * np.pi * sx * cy
        forcing = 2 * np.pi**2 * k * sx * sy - 0.5 * np.pi**2 * ((cx * sy) ** 2 + (sx * cy) ** 2)
        return (-k * lap - kx * ux - ky * uy - forcing)[:, None]
    if name == "reaction_diffusion":
        exact = 0.2 * np.sin(np.pi * z[:, 0]) * np.sin(np.pi * z[:, 1]) * np.exp(-z[:, 2])
        forcing = (-2 + 0.02 * np.pi**2) * exact + exact**3
        return (g[:, 0, 2] - 0.01 * lap - (u - u**3) - forcing)[:, None]
    raise ValueError(name)


def candidate_fd(name, z, step=1e-4):
    q = candidate_field(name, z)
    g = np.zeros((*q.shape, z.shape[1]))
    h = g.copy()
    for j in range(z.shape[1]):
        dz = np.zeros_like(z)
        dz[:, j] = step
        p, m = candidate_field(name, z + dz), candidate_field(name, z - dz)
        pp, mm = candidate_field(name, z + 2 * dz), candidate_field(name, z - 2 * dz)
        g[:, :, j] = (-pp + 8 * p - 8 * m + mm) / (12 * step)
        h[:, :, j] = (-pp + 16 * p - 30 * q + 16 * m - mm) / (12 * step**2)
    return q, g, h


def candidate_tf(name, z):
    import tensorflow as tf

    # Thin namespace: functions only, so NumPy and TF use the same field expression.
    class Backend:
        sin = staticmethod(tf.sin)
        cos = staticmethod(tf.cos)
        exp = staticmethod(tf.exp)
        tanh = staticmethod(tf.tanh)
        concatenate = staticmethod(lambda values, axis: tf.concat(values, axis))

    X = tf.constant(z, tf.float64)
    with tf.GradientTape(persistent=True) as outer:
        outer.watch(X)
        with tf.GradientTape(persistent=True) as inner:
            inner.watch(X)
            q = candidate_field(name, X, Backend)
            components = tf.unstack(q, axis=1)
        gradients = [
            inner.gradient(v, X, unconnected_gradients=tf.UnconnectedGradients.ZERO)
            for v in components
        ]
        grad_components = [tf.unstack(g, axis=1) for g in gradients]
    second = []
    for row in grad_components:
        second.append(
            tf.stack(
                [
                    outer.gradient(v, X, unconnected_gradients=tf.UnconnectedGradients.ZERO)[:, j]
                    for j, v in enumerate(row)
                ],
                axis=1,
            )
        )
    del inner, outer
    return q.numpy(), tf.stack(gradients, axis=1).numpy(), tf.stack(second, axis=1).numpy()


def candidate_points(name, count=128):
    domain = np.array(CANDIDATES[name]["domain"], float)
    rng = np.random.default_rng(98271 + CANDIDATES[name]["priority"])
    unit = np.column_stack([(rng.permutation(count) + rng.random(count)) / count for _ in domain])
    return domain[:, 0] + unit * (domain[:, 1] - domain[:, 0])


def candidate_audit(name, autodiff=True):
    points = candidate_points(name)
    # Burgers needs points THROUGH the thin shock, not only points that miss it.
    if name == "burgers":
        t = np.linspace(0, 1, 32)
        x = 0.5 * t + np.linspace(-0.02, 0.02, 32)
        points = np.concatenate((points, np.column_stack((x, t))))
    fd = candidate_fd(name, points)
    q, g, h = candidate_tf(name, points) if autodiff else fd
    res = candidate_residual(name, points, q, g, h)
    residual_max = float(np.max(abs(res)))
    # Fixed float64 tolerances. No relaxed threshold inferred from a failed run.
    tolerance = 1e-8 if autodiff else (2e-3 if name == "burgers" else 2e-5)
    gradient_difference = float(np.max(abs(g - fd[1])))
    hessian_difference = float(np.max(abs(h - fd[2])))
    derivative_pass = gradient_difference < 2e-4 and hessian_difference < 0.1
    config = CANDIDATES[name]
    faces = []
    for axis, bounds in enumerate(config["domain"]):
        if axis == config["time_axis"]:
            continue
        lo = points[:32].copy()
        hi = lo.copy()
        lo[:, axis] = bounds[0]
        hi[:, axis] = bounds[1]
        left, right = candidate_field(name, lo), candidate_field(name, hi)
        if name == "taylor_green":
            l = candidate_tf(name, lo) if autodiff else candidate_fd(name, lo)
            r = candidate_tf(name, hi) if autodiff else candidate_fd(name, hi)
            error = max(float(np.max(abs(left - right))), float(np.max(abs(l[1] - r[1]))))
            check = error < 1e-7
        elif config["boundary"] == "homogeneous Dirichlet":
            error = float(max(np.max(abs(left)), np.max(abs(right))))
            check = error < 1e-10
        else:
            # Exact nonzero labels, not a neural BC accuracy claim.
            error = None
            check = bool(np.isfinite(left).all() and np.isfinite(right).all())
        faces.append(
            {
                "axis": axis,
                "passed": check,
                "constraint_error": error,
                "scope": "exact boundary contract, not trained BC error",
            }
        )
    ic = None
    if config["time_axis"] is not None:
        initial = points.copy()
        initial[:, config["time_axis"]] = 0
        values = candidate_field(name, initial)
        ic = {
            "finite": bool(np.isfinite(values).all()),
            "range": [float(values.min()), float(values.max())],
            "scope": "initial labels from stated exact solution, not independent measurement",
        }
    sensitivity = None
    if name == "kovasznay_inverse":
        changed = candidate_residual(name, points, q, g, h, coefficient=0.05)
        sensitivity = {
            "coefficient_true": 0.025,
            "coefficient_probe": 0.05,
            "perturbed_residual_rms": float(np.sqrt(np.mean(changed**2))),
            "passed": bool(np.sqrt(np.mean(changed**2)) > 1e-3),
            "scope": "exact-field sensitivity only; not uniqueness proof",
        }
    passed = (
        residual_max < tolerance
        and derivative_pass
        and all(f["passed"] for f in faces)
        and (ic is None or ic["finite"])
        and (sensitivity is None or sensitivity["passed"])
    )
    return {
        "problem": name,
        "exact_audit_passed": bool(passed),
        "autodiff_executed": autodiff,
        "residual_max": residual_max,
        "residual_tolerance": tolerance,
        "independent_fd_gradient_max_difference": gradient_difference,
        "independent_fd_hessian_max_difference": hessian_difference,
        "boundary_contracts": faces,
        "initial_condition": ic,
        "inverse_sensitivity": sensitivity,
        "point_count": len(points),
        "used_for_training": False,
        "training_executed": False,
        "trained_model_validated": False,
        "benchmark_ready": False,
        "next_required": [
            "neural residual/loss wiring",
            "LHS train and fixed held-out contract",
            "optimizer/optional loss-SA checkpoint roundtrip",
            "GPU smoke",
            "fresh CPU reload",
            "approved paired pilot",
            "problem-specific target and benchmark review",
        ],
    }


def candidate_route(reports):
    return {
        "scope": "Post-Wave candidate exact-equation diagnosis; not all-PDE benchmark",
        "problems": reports,
        "deferred": DEFERRED,
        "training_executed": False,
        "automatic_benchmark_promotion": False,
        "lbfgs_postprocessing": {
            "status": "implemented_in_separate_notebook_runtime_validation_required",
            "notebook": "Online_PINN_PDE_Framework_V3_LBFGS_Postprocess.ipynb",
            "target_reference_l2": None,
            "strategy": "Adam then L-BFGS on the same PINN and inverse coefficient",
            "stage8_training_changed": False,
            "requirements": [
                "CPU-reviewed Stage8 parents; all five methods, no cherry-picked seeds",
                "fixed training points, SA factors and curriculum weights within each solve",
                "no LHS or loss-weight mutation inside objective or line-search callbacks",
                "20-iteration smoke then CPU review; 500-iteration bounded refinement",
                "record accepted iterations, objective/gradient evaluations, elapsed time and line-search failures separately",
                "fixed validation/test/reference; do not tune on reference error",
                "retain parent when training objective or fixed validation acceptance fails",
                "validate field, inverse coefficient, PDE, BC and IC; no 5 percent target",
                "no optimizer superiority claim without equal-budget Adam continuation control",
            ],
        },
    }
