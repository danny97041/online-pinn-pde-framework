"""Read-only diagnostics using the actual float32 residual for gradient checks.

Installed during training by progress.training_progress. The numerical runtime
source and checkpoint identity are unchanged; no downloaded source is executed.
"""

import numpy as np

from .config import group_weights, learning_rates
from .equations import residual

REVISION = "coefficient-observer-v2"


def coefficient_audit(runtime, tensors, gradients):
    """Check d(MSE)/dc against autodiff without subtracting float32 residuals.

    Spatial derivatives are frozen. A float64 affine slope is evaluated from the
    declared equation; its product with the actual float32 normalized residual
    matches the loss's computational path. The separate float64 reconstruction
    remains available for the frozen-field optimum and roundoff diagnostics.
    """
    tf = runtime.tf
    q, g, h = runtime.derivatives(tensors["pde"])
    coefficient = float(runtime.coefficient)
    scale32 = np.asarray(runtime.scales["residual"], dtype=np.float32)
    actual = residual(
        runtime.settings.equation,
        tensors["pde"],
        q,
        g,
        h,
        runtime.coefficient,
        tensors["forcing"],
        tf,
    ) / tf.constant(scale32)
    actual = actual.numpy().astype(np.float64)
    arguments = [
        runtime.settings.equation,
        tensors["pde"].numpy().astype(np.float64),
        q.numpy().astype(np.float64),
        g.numpy().astype(np.float64),
        h.numpy().astype(np.float64),
    ]
    forcing = tensors["forcing"].numpy().astype(np.float64)
    scale = scale32.astype(np.float64)
    intercept = residual(*arguments, 0.0, forcing, np) / scale
    slope = residual(*arguments, 1.0, forcing, np) / scale - intercept
    values = np.asarray(gradients, dtype=np.float64)
    if not all(np.isfinite(v).all() for v in (actual, intercept, slope, values)):
        raise RuntimeError("계수 gradient 검사에 유한하지 않은 값이 있습니다.")
    products = 2.0 * actual * slope
    analytic = float(np.mean(products))
    # Bound reduction/chain-rule rounding by the magnitude before cancellation.
    rounding_allowance = float(
        32 * np.finfo(np.float32).eps * np.mean(np.abs(products))
    )
    tolerance = max(1e-5, 1e-3 * abs(values[0]), rounding_allowance)
    error = abs(analytic - values[0])
    matches = bool(error <= tolerance)
    independent = bool(np.allclose(values[1:], 0.0, rtol=0, atol=1e-8))
    if not matches or not independent:
        raise RuntimeError(
            "계수 gradient 의존성 검사 실패: "
            f"analytic={analytic:.9g}, autodiff={values[0]:.9g}, "
            f"difference={error:.9g}, tolerance={tolerance:.9g}, "
            f"data_constraint_independent={independent}"
        )
    denominator = float(np.mean(slope * slope))
    optimum = (
        -float(np.mean(intercept * slope)) / denominator
        if denominator > 1e-20
        else None
    )
    reconstructed = intercept + coefficient * slope
    return {
        "unconstrained_optimum": optimum,
        "positive_domain_infimum": max(0.0, optimum) if optimum is not None else None,
        "analytic_gradient": analytic,
        "gradient_matches_autodiff": matches,
        "gradient_absolute_difference": error,
        "gradient_absolute_tolerance": tolerance,
        "float32_rounding_allowance": rounding_allowance,
        "reference_float64_gradient": float(2 * np.mean(reconstructed * slope)),
        "residual_roundoff_max_abs": float(np.max(np.abs(actual - reconstructed))),
        "data_constraint_coefficient_independent": independent,
        "field_frozen_only": True,
        "observer_revision": REVISION,
    }


def observe(runtime, tensors, iteration):
    """Observe losses/gradients while retaining model, optimizer and RNG guards."""
    before = runtime.snapshot()
    try:
        terms, vectors, coefficient_gradients = runtime.term_gradients(tensors)
        norms = np.array([np.linalg.norm(v) for v in vectors])
        cosine = np.diag((norms > 1e-30).astype(float))
        for a in range(3):
            for b in range(a):
                cosine[a, b] = cosine[b, a] = float(
                    vectors[a] @ vectors[b] / max(norms[a] * norms[b], 1e-30)
                )
        frozen = (
            coefficient_audit(runtime, tensors, coefficient_gradients)
            if runtime.inverse
            else None
        )
        return {
            "iteration": iteration,
            "raw_normalized_groups": [float(v) for v in terms],
            "network_gradient_norms": norms.tolist(),
            "gradient_cosines": cosine.tolist(),
            "coefficient_term_gradients": coefficient_gradients,
            "coefficient": float(runtime.coefficient),
            "frozen_field_coefficient": frozen,
            "gradient_cosine_defined": (
                (norms > 1e-30)[:, None] & (norms > 1e-30)[None, :]
            ).tolist(),
            "rng_guard_scope": "Python, NumPy legacy RNG, TensorFlow global Generator; deterministic tanh model",
            "effective_weights": group_weights(
                runtime.arm, iteration, runtime.factors
            ).tolist(),
            "sa_factors": runtime.factors.tolist(),
            "next_learning_rates": list(
                learning_rates(iteration, runtime.settings.base_lr)
            ),
            "state_guard": "PASS",
            "observer_revision": REVISION,
            "not_a_causal_proof": True,
        }
    finally:
        runtime.assert_unchanged(before)
