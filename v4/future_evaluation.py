"""Evaluate frozen basic forward models beyond their original time domain."""

import csv
import io
import json
from pathlib import Path
import zipfile

import numpy as np

from .config import Settings
from .data import DataSource
from .equations import EQUATIONS, exact_field
from .metrics import observed_metrics
from .reporting import reports
from .runtime import Runtime
from .storage import sha_file


def future_grid(spec, steps=20):
    """Future time grid at 1% increments of the original time span."""
    if steps not in (20, 200):
        raise ValueError("평가 시점 수는 20 또는 200입니다.")
    time_axis = spec["time_axis"]
    axes = []
    spatial_count = 256 if len(spec["domain"]) == 2 else 32
    for axis, (low, high) in enumerate(spec["domain"]):
        if axis == time_axis:
            values = high + (high - low) * np.arange(1, steps + 1) / 100
        else:
            values = (
                low + (high - low) * (np.arange(spatial_count) + 0.5) / spatial_count
            )
        axes.append(values)
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, len(axes))


def evaluate_saved(root, log=print, *, steps=20):
    root = Path(root)
    run_id = json.loads((root / "contract.json").read_text(encoding="utf-8"))["run_id"]
    rows = []
    for report in reports(root):
        equation = report["equation"]
        spec = EQUATIONS[equation]
        if (
            spec["time_axis"] is None
            or report["inverse"]
            or report["execution_scope"] != "research"
            or report["generalization"]["mode"] != "basic"
        ):
            continue
        trial = root / "trials" / report["trial_id"]
        cfg = json.loads((trial / "trial_settings.json").read_text(encoding="utf-8"))
        settings = Settings.from_dict(cfg["settings"])
        if settings.data_mode != "synthetic" or settings.noise_fraction:
            continue
        source = DataSource(settings)
        runtime = Runtime(settings, report["arm"], source)
        points = future_grid(spec, steps)
        reference = exact_field(equation, points)
        times = points[:, spec["time_axis"]]
        sample_times = np.unique(times)
        last = times == sample_times[-1]
        with np.load(trial / "fixed.npz", allow_pickle=False) as fixed:
            check_x, check_y = fixed["evaluation_x"], fixed["evaluation_y"]
        for stage in ("adam", "lbfgs"):
            if report.get(stage) is None:
                continue
            checkpoint = trial / (stage + ".npz")
            runtime.load(checkpoint)
            replay = observed_metrics(check_y, runtime.predict(check_x))
            original = report[stage]["field"]
            for metric in ("relative_l1", "relative_l2"):
                if not np.isclose(
                    replay[metric], original[metric], rtol=0.002, atol=0.00002
                ):
                    raise ValueError(
                        f"저장 평가 재현 오차: {trial.name} {stage} {metric}"
                    )
            prediction = runtime.predict(points, batch=2048)
            future = observed_metrics(reference, prediction)
            endpoint = observed_metrics(reference[last], prediction[last])
            windows = {}
            for start in range(0, steps, 20):
                lo, hi = sample_times[start], sample_times[start + 19]
                mask = (times >= lo) & (times <= hi)
                block = observed_metrics(reference[mask], prediction[mask])
                for norm in ("l1", "l2"):
                    windows[f"t{lo:.2f}_{hi:.2f}_{norm}_pct"] = (
                        100 * block["relative_" + norm]
                    )
            rows.append(
                dict(
                    run_id=run_id,
                    trial_id=trial.name,
                    equation=equation,
                    arm=report["arm"],
                    seed=report["seed"],
                    stage=stage,
                    checkpoint_sha256=sha_file(checkpoint),
                    points=len(points),
                    original_l1_pct=100 * original["relative_l1"],
                    original_l2_pct=100 * original["relative_l2"],
                    future_l1_pct=100 * future["relative_l1"],
                    future_l2_pct=100 * future["relative_l2"],
                    endpoint_l1_pct=100 * endpoint["relative_l1"],
                    endpoint_l2_pct=100 * endpoint["relative_l2"],
                    **windows,
                )
            )
        log(f"{spec['label']} · {report['arm']} · {report['seed']} 평가 완료")
    if not rows:
        raise ValueError(
            "합성 기본 비교의 무노이즈 순방향 시간 의존 모델을 포함한 결과 ZIP을 선택하세요."
        )
    return rows


def save_evaluation(rows, path, *, steps=20):
    end = 1 + steps / 100
    if f"t{end - .19:.2f}_{end:.2f}_l2_pct" not in rows[0]:
        raise ValueError("시간 범위와 결과 열 구성이 다릅니다.")
    table = io.StringIO(newline="")
    writer = csv.DictWriter(table, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    protocol = dict(
        training_updates=0,
        forecast_fraction=steps / 100,
        time_grid=f"1.01 to {end:.2f} in steps of 0.01 for training domain [0,1]",
        window_columns="non-overlapping 20-sample intervals, inclusive of both labelled times",
        spatial_grid="cell centres: 256 in 1D, 32 x 32 in 2D",
        reference="continued synthetic exact_field",
        scaling="original training domain and field scales",
        metrics="relative errors in percent, maximum per output component",
        endpoint="last future time only",
        stage_rows=len(rows),
    )
    with zipfile.ZipFile(path, "x", zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr("comparison.csv", table.getvalue().encode("utf-8-sig"))
        bundle.writestr("protocol.json", json.dumps(protocol, indent=2))
