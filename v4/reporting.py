"""Reports distinguish Adam, candidate L-BFGS and selected inference stages."""

import csv
import json
import re
from pathlib import Path
import numpy as np
from .config import ARM_LABELS
from .equations import EQUATIONS
from .storage import write_json


def reports(root):
    records = []
    for p in sorted((Path(root) / "trials").glob("*/report.json")):
        record = json.loads(p.read_text(encoding="utf-8"))
        if record.get("trial_id") != p.parent.name or not re.fullmatch(
            "[0-9a-f]{16}", p.parent.name
        ):
            raise ValueError("완료 실험 ID 또는 보고서 경로 오류")
        records.append(record)
    return records


def comparison(root):
    rows = []
    for r in reports(root):
        for stage in ("adam", "lbfgs"):
            result = r.get(stage)
            if result is None:
                continue
            field = result["field"] or {}
            rows.append(
                {
                    "trial_id": r["trial_id"],
                    "equation": r["equation"],
                    "inverse": r["inverse"],
                    "generalization": r["generalization"]["mode"],
                    "arm": r["arm"],
                    "seed": r["seed"],
                    "stage": stage,
                    "l1": field.get("relative_l1"),
                    "l2": field.get("relative_l2"),
                    "coefficient_error": result["coefficient_relative_error"],
                    "adam_updates": r["adam"]["optimizer_updates"],
                    "field_target": r["field_target"],
                    "execution_scope": r["execution_scope"],
                    "protocol_sha": r["protocol_sha"],
                    "noise_fraction": r["condition"]["noise_fraction"],
                    "initial_coefficient": r["condition"]["initial_coefficient"],
                    "lbfgs_iterations": result.get("solver", {}).get("iterations"),
                    "lbfgs_evaluations": result.get("solver", {}).get(
                        "function_evaluations"
                    ),
                }
            )
    return rows


def generate_report(root):
    root = Path(root)
    rows = comparison(root)
    grouped = {}
    for row in rows:
        key = (
            row["equation"],
            row["inverse"],
            row["generalization"],
            row["arm"],
            row["stage"],
            row["execution_scope"],
            row["noise_fraction"],
            row["initial_coefficient"],
            row["protocol_sha"],
        )
        grouped.setdefault(key, []).append(row)
    aggregates = []
    text = [
        "# 온라인 PINN PDE 프레임워크 V4",
        "",
        "Adam과 L-BFGS 종료를 별도로 비교합니다. 아래 수치는 저장된 최종 평가 결과입니다.",
        "",
        "| 방정식 | 문제 | 분할 | 방법 | 단계 | 조건 (노이즈/초기계수) | 시드 | L1 평균 % | L2 평균 % | L2 표본 표준편차 % |",
        "|---|---|---|---|---|---|---:|---:|---:|---:|",
    ]
    for key, group in grouped.items():
        eq, inv, mode, arm, stage, scope, noise, initial, protocol = key
        l1 = [r["l1"] for r in group if r["l1"] is not None]
        l2 = [r["l2"] for r in group if r["l2"] is not None]
        mean1 = float(np.mean(l1)) if l1 else None
        mean2 = float(np.mean(l2)) if l2 else None
        std = float(np.std(l2, ddof=1)) if len(l2) > 1 else None
        aggregates.append(
            {
                "equation": eq,
                "inverse": inv,
                "mode": mode,
                "arm": arm,
                "stage": stage,
                "execution_scope": scope,
                "noise_fraction": noise,
                "initial_coefficient": initial,
                "protocol_sha": protocol,
                "seeds": len(group),
                "mean_l1": mean1,
                "mean_l2": mean2,
                "sample_std_l2": std,
                "field_target_count": sum(
                    r["l1"] is not None
                    and r["l2"] is not None
                    and max(r["l1"], r["l2"]) <= r["field_target"]
                    for r in group
                ),
            }
        )
        fmt = lambda v: "해당 없음" if v is None else f"{100*v:.5f}"
        text.append(
            f"| {EQUATIONS[eq]['label']} | {'역문제' if inv else '순방향'} | {mode} / {scope} | {ARM_LABELS[arm]} | {stage} | {noise}/{initial} ({protocol[:8]}) | {len(group)} | {fmt(mean1)} | {fmt(mean2)} | {fmt(std)} |"
        )
    if not rows:
        text += ["", "아직 완료된 V4 학습 실험이 없습니다."]
    text += [
        "",
        "계수 오차·PDE/경계/초기조건·진단·iter·후처리 선택 근거는 개별 실험 JSON에 있습니다.",
        "보간 시 시드별 제외 구역이 다르면 표본 변동에는 구역 난이도 차이가 포함됩니다.",
        "Supervised-only는 계수를 추정하지 않습니다. 실제 데이터에서 없는 정답은 평가값으로 생성하지 않습니다.",
    ]
    (root / "REPORT.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    if rows:
        with (root / "comparison.csv").open(
            "w", encoding="utf-8-sig", newline=""
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    write_json(
        root / "summary.json",
        {
            "format": "v4-independent-1",
            "completed_trials": len(reports(root)),
            "rows": rows,
            "aggregates": aggregates,
            "historical_v3_not_v4_results": True,
        },
    )
    catalog = "# V4 기능 목록\n\n- 12개 방정식 설정과 순·역문제\n- 6개 방법, 기본 1시드·선택형 4시드\n- PDE RAD / 나머지 합성 학습점 LHS\n- 실제 데이터 CSV 연결\n- 독립 보간·시간 외삽\n- Adam/L-BFGS 분리 평가\n- 20% 간격 단일 재개 ZIP\n- 보고서·그림·API·도구형 질문·모델 등록\n\nOptuna는 포함하지 않습니다. 원통 후류는 외부 데이터가 필요합니다.\n"
    (root / "FEATURES.md").write_text(catalog, encoding="utf-8")
    return root / "REPORT.md"


def make_plots(root, trial_id=None, time_value=None, component=0):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(root)
    out = root / "plots"
    out.mkdir(exist_ok=True)
    rows = comparison(root)
    fig, ax = plt.subplots(figsize=(10, 4))
    for stage in ("adam", "lbfgs"):
        chosen = [r for r in rows if r["stage"] == stage and r["l2"] is not None]
        ax.scatter(range(len(chosen)), [r["l2"] * 100 for r in chosen], label=stage)
    ax.set_ylabel("Field L2 (%)")
    ax.set_xlabel("Saved trial index")
    if rows:
        ax.legend()
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(out / ("comparison." + suffix))
    plt.close(fig)
    if trial_id:
        from .config import Settings
        from .data import DataSource, make_plan
        from .runtime import Runtime
        from .equations import exact_field

        if trial_id not in {r["trial_id"] for r in reports(root)}:
            raise ValueError("완료 실험을 선택하세요.")
        trial = root / "trials" / trial_id
        cfg = json.loads((trial / "trial_settings.json").read_text(encoding="utf-8"))
        settings = Settings.from_dict(cfg["settings"])
        source = DataSource(
            settings, root / "inputs/data.json" if settings.data_mode == "csv" else None
        )
        rt = Runtime(settings, cfg["arm"], source)
        rt.load(trial / "selected.npz")
        plan = make_plan(settings, cfg["seed"], source.domain)
        x = plan.fixed_evaluation_points(32, 1)
        if plan.time_axis is not None:
            if time_value is None:
                time_value = source.domain[plan.time_axis][0]
            if (
                not source.domain[plan.time_axis][0]
                <= time_value
                <= source.domain[plan.time_axis][1]
            ):
                raise ValueError("시간 범위 오류")
            x[:, plan.time_axis] = time_value
        pred = rt.predict(x)
        exact = (
            exact_field(settings.equation, x)
            if settings.data_mode == "synthetic"
            else None
        )
        if type(component) is not int or not 0 <= component < pred.shape[1]:
            raise ValueError("출력 성분을 확인하세요.")
        physics = np.concatenate(
            [rt.pde_residual(x[i : i + 128]).numpy() for i in range(0, len(x), 128)]
        ) / np.asarray(rt.scales["residual"])
        columns = (
            [("Prediction", pred[:, component])]
            if exact is None
            else [
                ("Reference", exact[:, component]),
                ("Prediction", pred[:, component]),
                ("Absolute error", abs(pred[:, component] - exact[:, component])),
            ]
        )
        columns.append(
            ("Normalized residual RMS", np.sqrt(np.mean(physics**2, axis=1)))
        )
        fig, axes = plt.subplots(
            1, len(columns), figsize=(4 * len(columns), 4), squeeze=False
        )
        for ax, (title, z) in zip(axes[0], columns):
            if len(plan.spatial_axes) == 2:
                points = ax.scatter(
                    x[:, plan.spatial_axes[0]], x[:, plan.spatial_axes[1]], c=z, s=8
                )
                fig.colorbar(points, ax=ax)
            else:
                ax.plot(x[:, plan.spatial_axes[0]], z, ".")
            ax.set_title(title)
        fig.tight_layout()
        for suffix in ("png", "pdf"):
            fig.savefig(out / (trial_id + "_field." + suffix))
        plt.close(fig)
        history = json.loads((trial / "report.json").read_text(encoding="utf-8"))[
            "history"
        ]
        fig, ax = plt.subplots()
        ax.plot(
            [r["iteration"] for r in history],
            [r["validation"]["relative_l2"] for r in history],
        )
        ax.set_xlabel("Adam updates")
        ax.set_ylabel("Validation relative L2")
        fig.tight_layout()
        fig.savefig(out / (trial_id + "_history.png"))
        plt.close(fig)
        diagnostics = json.loads((trial / "report.json").read_text(encoding="utf-8"))[
            "diagnostics"
        ]
        if diagnostics:
            fig, axes = plt.subplots(2, 2, figsize=(10, 7))
            iterations = [r["iteration"] for r in diagnostics]
            for a, label in enumerate(("PDE", "data", "constraints")):
                axes[0, 0].plot(
                    iterations,
                    [r["raw_normalized_groups"][a] for r in diagnostics],
                    label=label,
                )
                axes[0, 1].plot(
                    iterations,
                    [r["network_gradient_norms"][a] for r in diagnostics],
                    label=label,
                )
                axes[1, 1].plot(
                    iterations,
                    [r["effective_weights"][a] for r in diagnostics],
                    label=label,
                )
            axes[1, 0].plot(
                iterations, [r["coefficient"] for r in diagnostics], label="coefficient"
            )
            for ax, title in zip(
                axes.flat,
                (
                    "Normalized losses",
                    "Gradient norms",
                    "Coefficient",
                    "Effective group weights",
                ),
            ):
                ax.set_title(title)
                ax.set_xlabel("Adam updates")
                ax.legend()
            fig.tight_layout()
            fig.savefig(out / (trial_id + "_diagnostics.png"))
            plt.close(fig)
    return sorted(out.glob("*"))
