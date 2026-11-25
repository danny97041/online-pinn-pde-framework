"""Field, coefficient, local residual and integrated-equation balance reports."""

import numpy as np
from .generalization import field_metrics


def observed_metrics(reference, prediction):
    reference = np.asarray(reference)
    prediction = np.asarray(prediction)
    components = []
    for a in range(reference.shape[1]):
        valid = np.isfinite(reference[:, a])
        components.append(
            field_metrics(reference[valid, a], prediction[valid, a])
            if valid.any()
            else None
        )
    defined = [c for c in components if c is not None and c["relative_errors_defined"]]
    all_observed_defined = len(defined) == sum(c is not None for c in components)
    return {
        "relative_l1": (
            max((c["relative_l1"] for c in defined), default=None)
            if all_observed_defined
            else None
        ),
        "relative_l2": (
            max((c["relative_l2"] for c in defined), default=None)
            if all_observed_defined
            else None
        ),
        "all_observed_relative_errors_defined": all_observed_defined,
        "components": components,
        "observed_components": len(defined),
        "aggregation": "maximum per observed output; unobserved outputs are not scored",
    }


def constraint_metrics(runtime, arrays):
    results = {}
    scale = np.asarray(runtime.scales["field"])
    for family in ("boundary", "initial", "gauge"):
        if family not in arrays:
            continue
        prediction = runtime.predict(arrays[family])
        labels = arrays["label_" + family]
        observed = np.isfinite(labels)
        if family == "boundary" and len(scale) == 3:
            observed[:, 2] = False
        if family == "gauge":
            observed[:, :2] = False
        error = (prediction - labels) / scale
        results[family] = {
            "normalized_rms": (
                float(np.sqrt(np.mean(error[observed] ** 2)))
                if observed.any()
                else None
            ),
            "normalized_max": (
                float(np.max(abs(error[observed]))) if observed.any() else None
            ),
            "points": len(labels),
            "source": "fixed known constraints; CSV conditions may also be training constraints",
        }
    if "initial_velocity" in arrays:
        points = arrays.get("initial_velocity_x", arrays["initial"])
        _, g, _ = runtime.derivatives(points)
        duration = np.ptp(np.asarray(runtime.source.domain)[-1])
        error = (
            (g.numpy()[:, 0, -1:] - arrays["initial_velocity"]) * duration / scale[0]
        )
        results["initial_velocity"] = {
            "normalized_rms": float(np.sqrt(np.mean(error**2))),
            "normalized_max": float(abs(error).max()),
        }
    return results


def evaluate(runtime, data, physics_points, iteration, stage, constraints=None):
    before = runtime.snapshot()
    field = (
        observed_metrics(data["y"], runtime.predict(data["x"]))
        if data is not None
        else None
    )
    parts = []
    for i in range(0, len(physics_points), 128):
        parts.append(runtime.pde_residual(physics_points[i : i + 128]).numpy())
    residuals = np.concatenate(parts)
    scaled = residuals / np.asarray(runtime.scales["residual"])
    volume = float(np.prod(np.ptp(np.asarray(runtime.source.domain), axis=1)))
    truth = runtime.source.truth
    coefficient = (
        float(runtime.coefficient) if runtime.arm != "supervised_only" else None
    )
    constraint_result = constraint_metrics(runtime, constraints or {})
    from .balance import balance_report

    balance = balance_report(runtime)
    true_view = None
    if runtime.inverse and truth is not None:
        parts = [
            runtime.pde_residual(physics_points[i : i + 128], truth).numpy()
            for i in range(0, len(physics_points), 128)
        ]
        true_view = np.sqrt(
            np.mean(
                (np.concatenate(parts) / np.asarray(runtime.scales["residual"])) ** 2,
                axis=0,
            )
        ).tolist()
    runtime.assert_unchanged(before)
    return {
        "stage": stage,
        "iteration": iteration,
        "field": field,
        "coefficient": coefficient,
        "coefficient_relative_error": (
            abs(coefficient / truth - 1)
            if truth is not None and coefficient is not None and runtime.inverse
            else None
        ),
        "coefficient_estimated": runtime.inverse,
        "constraints": constraint_result,
        "integral_balance": balance,
        "physics": {
            "point_count": len(physics_points),
            "normalized_rms": np.sqrt(np.mean(scaled**2, axis=0)).tolist(),
            "normalized_p99": np.quantile(abs(scaled), 0.99, axis=0).tolist(),
            "normalized_max": abs(scaled).max(axis=0).tolist(),
            "integrated_equation_balance_defect": (
                volume * np.mean(residuals, axis=0)
            ).tolist(),
            "balance_definition": "quadrature estimate of signed governing-equation residual; not an independent surface-flux certificate",
            "reference_coefficient_normalized_rms": true_view,
            "normalization": runtime.scales,
            "used_for_stopping": False,
        },
        "state_guard": "PASS",
    }
