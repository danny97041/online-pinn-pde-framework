"""Optional development audits; not included in the Colab runtime."""

import json
from pathlib import Path
import numpy as np
from v4.config import Settings
from v4.data import DataSource
from v4.equations import EQUATIONS, exact_field, known_forcing, residual
from v4.reporting import reports
from v4.runtime import Runtime
from v4.sampling import lhs, rng_for


def finite_derivatives(name, points, step=1e-4):
    q = exact_field(name, points)
    dim = points.shape[1]
    g = np.zeros((*q.shape, dim))
    h = np.zeros((*q.shape, dim, dim))
    for a in range(dim):
        d = np.zeros_like(points)
        d[:, a] = step
        p, m = exact_field(name, points + d), exact_field(name, points - d)
        pp, mm = exact_field(name, points + 2 * d), exact_field(name, points - 2 * d)
        g[:, :, a] = (-pp + 8 * p - 8 * m + mm) / (12 * step)
        h[:, :, a, a] = (-pp + 16 * p - 30 * q + 16 * m - mm) / (12 * step**2)
        for b in range(a):
            e = np.zeros_like(points)
            e[:, b] = step
            mixed = (
                exact_field(name, points + d + e)
                - exact_field(name, points + d - e)
                - exact_field(name, points - d + e)
                + exact_field(name, points - d - e)
            ) / (4 * step**2)
            h[:, :, a, b] = h[:, :, b, a] = mixed
    return q, g, h


def verify_equations():
    results = []
    for name, spec in EQUATIONS.items():
        if name == "cylinder_ns":
            results.append(
                {"equation": name, "status": "requires_external_reference_data"}
            )
            continue
        bounds = np.array(spec["domain"], float)
        width = np.ptp(bounds, axis=1)
        bounds[:, 0] += 0.2 * width
        bounds[:, 1] -= 0.2 * width
        x = lhs(bounds, 16, rng_for(91923, name))
        q, g, h = finite_derivatives(name, x, step=1e-5 if name == "burgers" else 1e-4)
        f = known_forcing(name, x)
        r = residual(name, x, q, g, h, spec["fixed"], f)
        changed = residual(name, x, q, g, h, spec["fixed"] * 1.2, f)
        maximum = float(np.max(abs(r)))
        scale = max(1.0, float(np.max(abs(f))))
        sensitivity = float(np.sqrt(np.mean((changed - r) ** 2)))
        passed = maximum / scale < 0.003 and sensitivity > 1e-9
        results.append(
            {
                "equation": name,
                "passed": passed,
                "relative_residual_max": maximum / scale,
                "exact_field_coefficient_sensitivity": sensitivity,
                "not_trained_identifiability": True,
            }
        )
    if not all(r.get("passed", True) for r in results):
        raise RuntimeError("방정식 배선 검사 실패: " + str(results))
    return {"passed": True, "results": results, "training_executed": False}


def check_archive_models(root):
    root = Path(root)
    checks = []
    for report in reports(root):
        trial = root / "trials" / report["trial_id"]
        settings = Settings.from_dict(
            json.loads((trial / "trial_settings.json").read_text(encoding="utf-8"))[
                "settings"
            ]
        )
        source = DataSource(
            settings, root / "inputs/data.json" if settings.data_mode == "csv" else None
        )
        runtime = Runtime(settings, report["arm"], source)
        runtime.load(trial / "selected.npz")
        with np.load(trial / "fixed.npz", allow_pickle=False) as z:
            x = z["validation_x"][:4]
        a = runtime.predict(x)
        b = runtime.predict(x)
        if not np.isfinite(a).all() or not np.array_equal(a, b):
            raise RuntimeError("저장 모델 예측 불일치")
        checks.append(
            {"trial_id": report["trial_id"], "finite_repeat_prediction": True}
        )
    return {
        "passed": True,
        "checked_models": len(checks),
        "checks": checks,
        "tensorflow_version_pin_required": False,
    }
