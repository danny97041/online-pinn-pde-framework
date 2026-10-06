"""V3 integrated contracts, evidence, reports and safe one-ZIP persistence.

No TensorFlow, embedding model, server or training is started on import.
"""

import hashlib
import json
import os
import re
import stat
import zipfile
from pathlib import Path, PurePosixPath
import numpy as np

U_ARMS = ["baseline", "loss_sa", "ff", "ff_loss_sa", "ff_curriculum"]
U_PROBLEMS = [
    "wave2d",
    "heat2d",
    "poisson2d",
    "burgers",
    "kovasznay_forward",
    "kovasznay_inverse",
    "taylor_green",
    "darcy2d",
    "reaction_diffusion",
]
U_POLICY = {
    "version": "v3-integrated-2-single-seed",
    "field_target": 0.10,
    "coefficient_report_target": 0.10,
    "stop_first": 15000,
    "stop_every": 1000,
    "stop_consecutive": 3,
    "cap": 100000,
    "lhs_period": 5000,
    "seeds": [3234],
    "arms": U_ARMS,
    "base_lr": 0.004,
    "lr_knots": [0, 5000, 10000],
    "network_ratios": [1.0, 0.75, 0.5],
    "coefficient_ratios": [0.001, 0.75, 1.0],
    "sa_groups": ["PDE", "data", "boundary_and_initial"],
    "sa_every": 50,
    "sa_ema": 0.9,
    "sa_bounds": [0.25, 4.0],
    "ff_features_per_bank": 32,
    "ff_scales": [0.125, 0.25, 0.5],
    "v4_deferred": [
        "mandatory 5% accuracy",
        "local residual peak reduction",
        "BC/IC qualification",
    ],
    "historical_results": "preserve original 5% stopping; never relabel as prospective 10% experiment",
    "overall_physics_approval": False,
    "automatic_promotion": False,
}


def u_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def u_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def u_members(z):
    names = z.namelist()
    if len(names) != len(set(names)):
        raise ValueError("Duplicate ZIP members")
    for i in z.infolist():
        n = i.orig_filename
        p = PurePosixPath(n)
        if (
            p.is_absolute()
            or ".." in p.parts
            or "\\" in n
            or ":" in n
            or "\x00" in n
            or stat.S_ISLNK(i.external_attr >> 16)
        ):
            raise ValueError("Unsafe ZIP entry: " + repr(n))
    if sum(i.file_size for i in z.infolist()) > 4 * 1024**3:
        raise ValueError("ZIP exceeds 4 GiB limit")
    if z.testzip() is not None:
        raise ValueError("ZIP CRC failed")
    return {i.filename for i in z.infolist() if not i.is_dir()}


def u_verify_zip(path, manifest=None):
    with zipfile.ZipFile(path) as z:
        names = u_members(z)
        keys = (
            [manifest]
            if manifest
            else ["unified_manifest.json", "periodic_manifest.json", "audit_manifest.json"]
        )
        key = next((k for k in keys if k in names), None)
        if key is None:
            raise ValueError("No supported manifest in " + Path(path).name)
        entries = json.loads(z.read(key))["files"]
        if names != set(entries) | {key}:
            raise ValueError("ZIP inventory mismatch")
        for n, digest in entries.items():
            if hashlib.sha256(z.read(n)).hexdigest() != digest:
                raise ValueError("Hash mismatch: " + n)
    return u_sha(path)


def u_pack(root, output):
    """Only this run's rolling output is replaced; input ZIPs are never removed."""
    root, output = Path(root), Path(output)
    if root.resolve() in output.resolve().parents:
        raise ValueError("Output must be outside run directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "files": {
            p.relative_to(root).as_posix(): u_sha(p)
            for p in sorted(root.rglob("*"))
            if p.is_file() and p.name != "unified_manifest.json"
        }
    }
    u_json(root / "unified_manifest.json", manifest)
    temp = output.with_name(output.name + ".writing")
    with zipfile.ZipFile(temp, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(root.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(root).as_posix())
    u_verify_zip(temp, "unified_manifest.json")
    os.replace(temp, output)


def u_restore(bundle, root, expected_code):
    u_verify_zip(bundle, "unified_manifest.json")
    root = Path(root)
    if root.exists():
        raise ValueError("Restore destination already exists")
    with zipfile.ZipFile(bundle) as z:
        contract = json.loads(z.read("contract.json"))
        if contract.get("code_sha256") != expected_code:
            raise ValueError("Different integrated code revision; review/export only")
        if contract.get("policy") != U_POLICY:
            raise ValueError("Different training/evaluation contract")
        z.extractall(root)  # fully validated member paths, symlinks and hashes above
    return contract


def u_digest(arrays):
    h = hashlib.sha256()
    for key, a in sorted(arrays.items()):
        a = np.ascontiguousarray(a)
        h.update(key.encode())
        h.update(str(a.shape).encode())
        h.update(a.dtype.str.encode())
        h.update(a.tobytes())
    return h.hexdigest()


def u_cycle(iteration):
    return max(0, (iteration - 1) // U_POLICY["lhs_period"])


def u_lrs(completed):
    return [
        float(np.interp(completed, U_POLICY["lr_knots"], U_POLICY[k]) * U_POLICY["base_lr"])
        for k in ["network_ratios", "coefficient_ratios"]
    ]


def u_weights(arm, completed, factors):
    if arm not in U_ARMS:
        raise ValueError(arm)
    f = np.asarray(factors, float)
    if f.shape != (3,) or not np.isfinite(f).all() or np.any(f <= 0):
        raise ValueError("Invalid SA factors")
    if arm not in ["loss_sa", "ff_loss_sa"] and not np.array_equal(f, np.ones(3)):
        raise ValueError("SA factors leaked into non-SA arm")
    if arm == "ff_curriculum":
        p = min(completed // 100, 100) / 300
        return np.array([p, (1 - p) / 2, (1 - p) / 2])
    return f / f.sum()


def u_target(history):
    every = U_POLICY["stop_every"]
    consecutive = U_POLICY["stop_consecutive"]
    rows = [
        r
        for r in history
        if r["iteration"] >= U_POLICY["stop_first"] and r["iteration"] % every == 0
    ]
    if len(rows) < consecutive:
        return False
    rows = rows[-consecutive:]
    return np.diff([r["iteration"] for r in rows]).tolist() == [every] * (consecutive - 1) and all(
        np.isfinite(r["validation_l2"]) and r["validation_l2"] <= U_POLICY["field_target"]
        for r in rows
    )


def u_plan(profile, mode, equations):
    if profile not in ["review", "preview", "smoke", "pilot", "benchmark", "auto"]:
        raise ValueError("Unknown profile")
    if mode not in ["exp", "full"]:
        raise ValueError("Unknown mode")
    if (
        not equations
        or len(set(equations)) != len(equations)
        or not set(equations) <= set(U_PROBLEMS)
    ):
        raise ValueError("Unknown or duplicate equations")
    caps = {
        "review": 0,
        "preview": 0,
        "smoke": 200,
        "pilot": 10000,
        "benchmark": 100000,
        "auto": 100000,
    }
    seeds = list(U_POLICY["seeds"]) if profile in ["benchmark", "auto"] else [3234]
    return {
        "profile": profile,
        "mode": mode,
        "equations": equations,
        "cap": caps[profile],
        "seeds": seeds,
        "lbfgs_maxiter": 500 if profile in ["benchmark", "auto"] else 20,
        "max_adam_updates": caps[profile] * len(seeds) * len(U_ARMS) * len(equations),
        "service_modules": "skipped_exp" if mode == "exp" else "requested",
        "reporting": "always_enabled",
        "training_authorized": caps[profile] > 0,
    }


def u_quantity_score(adam, selected):
    if adam is None or selected is None:
        return None
    if not np.isfinite([adam, selected]).all() or min(adam, selected) < 0:
        raise ValueError("Invalid score inputs")
    a = 3 if adam <= 0.05 else 2 if adam <= 0.10 else 0
    p = 2 if selected <= 0.05 else 1 if selected <= 0.10 else 0
    retained_cap = 3 if selected <= 0.05 else 2 if selected <= 0.10 else 0
    return max(min(a, retained_cap), p)


def u_import_wave(parent, audit, root):
    """Import evidence, not code/model pickles; no retraining or changed stop history."""
    root = Path(root)
    parent_hash = u_verify_zip(parent)
    with zipfile.ZipFile(parent) as z:
        c = json.loads(z.read("contract.json"))
        summary = json.loads(z.read("summary.json"))
        if c.get("experiment") != "wave_crosslr" or c.get("profile") != "benchmark":
            raise ValueError("Need CrossLR benchmark")
        if summary.get("status") != "complete" or summary.get("completed_trials") != 15:
            raise ValueError("Incomplete Wave benchmark")
        pairs = set()
        rows = []
        for r in summary["rows"]:
            key = (r["seed"], r["arm"])
            pairs.add(key)
            a, p = r["adam"], r["post_selected"]
            rows.append(
                {
                    "equation": "wave2d",
                    "seed": r["seed"],
                    "arm": r["arm"],
                    "origin": "historical_5pct_stop",
                    "adam_iteration": r["iteration"],
                    "adam_field_l2": a["reference_metrics"]["relative_l2"],
                    "field_l2": p["reference_metrics"]["relative_l2"],
                    "adam_lambda_error": abs(a["lambda_1"] - 1.0),
                    "lambda_error": abs(p["lambda_1"] - 1.0),
                    "field_points": "historical_reference",
                    "overall_physics_approved": False,
                    "pde_grade": None,
                }
            )
            for kind, member in [
                ("adam", f"trials/seed_{r['seed']}_{r['arm']}/result.json"),
                ("post", f"postprocess/trials/seed_{r['seed']}_{r['arm']}/result.json"),
            ]:
                detail = json.loads(z.read(member))
                u_json(root / f"evidence/{kind}_seed_{r['seed']}_{r['arm']}.json", detail)
                if kind == "adam":
                    rows[-1]["adam_seconds_including_diagnostics"] = detail["last"].get(
                        "elapsed_seconds"
                    )
                    rows[-1]["trainable_parameters"] = detail.get("trainable_parameters")
                else:
                    rows[-1]["lbfgs_objective_evaluations"] = detail["solver"][
                        "objective_evaluations"
                    ]
                    rows[-1]["lbfgs_seconds_including_diagnostics"] = detail["solver"].get(
                        "elapsed_seconds"
                    )
        if len(rows) != 15 or pairs != {(s, a) for s in [3234, 3235, 3236] for a in U_ARMS}:
            raise ValueError("Wave trial inventory mismatch")
        u_json(root / "evidence/wave_contract.json", c)
        u_json(root / "evidence/wave_summary.json", summary)
    provenance = {
        "parent_filename": Path(parent).name,
        "parent_sha256": parent_hash,
        "contains_model_copy": False,
        "historical_stop_threshold": c["policy"]["target"],
    }
    if audit:
        ah = u_verify_zip(audit)
        with zipfile.ZipFile(audit) as z:
            prov = json.loads(z.read("provenance.json"))
            s = json.loads(z.read("summary.json"))
            if prov["input_sha256"]["parent"] != parent_hash:
                raise ValueError("Physics audit belongs to another parent")
            if s.get("status") != "complete" or s.get("completed_model_stages") != 30:
                raise ValueError("Incomplete physics audit")
            for row in rows:
                trial = json.loads(z.read(f"trials/seed_{row['seed']}_{row['arm']}.json"))
                row["pde_grade"] = trial["score"]["pde_grade"]
                u_json(root / f"evidence/physics_seed_{row['seed']}_{row['arm']}.json", trial)
            u_json(root / "evidence/physics_summary.json", s)
            u_json(root / "evidence/physics_policy.json", json.loads(z.read("policy.json")))
            u_json(
                root / "evidence/physics_normalization.json",
                json.loads(z.read("normalization.json")),
            )
        provenance.update(audit_filename=Path(audit).name, audit_sha256=ah)
    u_json(root / "evidence/provenance.json", provenance)
    u_json(root / "imported_wave_rows.json", rows)
    return rows


def u_report(root, rows, modules, routes):
    """Compare within equation only; do not average unlike PDE residual units."""
    root = Path(root)
    groups = []
    rows = [dict(r) for r in rows]
    for r in rows:
        r["field_attainment_points"] = u_quantity_score(r.get("adam_field_l2"), r.get("field_l2"))
        r["coefficient_attainment_points"] = u_quantity_score(
            r.get("adam_lambda_error"), r.get("lambda_error")
        )
        r["score_out_of_6"] = (
            r["field_attainment_points"] + r["coefficient_attainment_points"]
            if r["coefficient_attainment_points"] is not None
            else None
        )
        r["score_note"] = (
            "Attainment only; 5% is a bonus, not required for V3; forward coefficient N/A; not overall physics approval"
        )
    for eq in U_PROBLEMS:
        for arm in U_ARMS:
            subset = [r for r in rows if r["equation"] == eq and r["arm"] == arm]
            if not subset:
                continue
            fields = [r["field_l2"] for r in subset]
            coeff = [r["lambda_error"] for r in subset if r.get("lambda_error") is not None]
            groups.append(
                {
                    "equation": eq,
                    "arm": arm,
                    "n": len(subset),
                    "field_l2_mean": float(np.mean(fields)),
                    "field_l2_sample_std": (
                        float(np.std(fields, ddof=1)) if len(fields) > 1 else None
                    ),
                    "mean_adam_updates": float(np.mean([r["adam_iteration"] for r in subset])),
                    "mean_lbfgs_objective_evaluations": (
                        float(np.mean([r["lbfgs_objective_evaluations"] for r in subset]))
                        if all("lbfgs_objective_evaluations" in r for r in subset)
                        else None
                    ),
                    "field_10pct_count": sum(v <= 0.10 for v in fields),
                    "coefficient_10pct_count": sum(v <= 0.10 for v in coeff) if coeff else None,
                    "lambda_error_mean": float(np.mean(coeff)) if coeff else None,
                    "score_mean_out_of_6": (
                        float(np.mean([r["score_out_of_6"] for r in subset]))
                        if all(r["score_out_of_6"] is not None for r in subset)
                        else None
                    ),
                    "pde_grades": [r.get("pde_grade") for r in subset],
                }
            )
    updates = sum(r.get("optimizer_updates_this_run", 0) for r in routes.values())
    profile = json.loads((root / "contract.json").read_text())["plan"]["profile"]
    if profile == "review":
        updates = 0
    evaluations = (
        0
        if profile == "review"
        else sum(r.get("lbfgs_evaluations_this_run", 0) for r in routes.values())
    )
    summary = {
        "status": "complete",
        "profile": profile,
        "completed_trials": len(rows),
        "training_executed": updates > 0,
        "optimizer_updates_this_run": updates,
        "lbfgs_evaluations_this_run": evaluations,
        "stored_training_results_present": bool(rows),
        "wiring_probes_are_disposable": True,
        "rows": rows,
        "aggregate": groups,
        "modules": modules,
        "equation_routes": routes,
        "policy": U_POLICY,
        "overall_physics_approved": False,
        "automatic_promotion": False,
        "v3_scope": "10% research milestone, not industrial qualification",
        "model_checkpoint_note": "Imported Wave models remain in the named parent ZIP; this report is not their replacement",
    }
    u_json(root / "summary.json", summary)
    # Machine-readable analysis output, not a preselected winner or a deployment manifest.
    import csv

    with (root / "comparison.csv").open("w", encoding="utf-8-sig", newline="") as f:
        keys = [
            "equation",
            "arm",
            "n",
            "field_l2_mean",
            "field_l2_sample_std",
            "field_10pct_count",
            "coefficient_10pct_count",
            "lambda_error_mean",
            "mean_adam_updates",
            "mean_lbfgs_objective_evaluations",
            "score_mean_out_of_6",
        ]
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(groups)
    u_report_markdown(root, groups, modules)
    return summary


def u_report_markdown(root, groups, modules=None):
    """수치 집계를 바꾸지 않고 사용자용 비교 보고서를 한국어로 표시합니다."""
    equations = {
        "wave2d": "2차원 파동",
        "heat2d": "2차원 열전도",
        "poisson2d": "2차원 포아송",
        "burgers": "버거스",
        "kovasznay_forward": "Kovasznay 순방향",
        "kovasznay_inverse": "Kovasznay 역문제",
        "taylor_green": "Taylor–Green",
        "darcy2d": "2차원 Darcy",
        "reaction_diffusion": "반응–확산",
    }
    methods = {
        "baseline": "기본 PINN",
        "loss_sa": "손실항 SA",
        "ff": "푸리에 특징",
        "ff_loss_sa": "푸리에 특징 + 손실항 SA",
        "ff_curriculum": "푸리에 특징 + 커리큘럼",
    }
    lines = [
        "# 온라인 PINN PDE 프레임워크 V3 — 실험 비교",
        "",
        f"{len({r['equation'] for r in groups})}개 방정식의 저장 실험 결과를 비교합니다. 표의 오차는 백분율입니다.",
        "10%는 연구용 물리장 오차 목표이며 산업 인증이나 전체 물리 정확도 승인을 의미하지 않습니다.",
        "Wave의 과거 5% 중단 이력은 보존하며 새 10% 목표 실험으로 바꾸어 표시하지 않습니다.",
        "",
        "| 방정식 | 방법 | 시드 수 | 평균 물리장 L2 오차 (%) | 10% 이하 실험 수 | 평균 계수 오차 (%) |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in groups:
        coefficient = (
            "해당 없음(순방향)"
            if row["lambda_error_mean"] is None
            else f"{100 * row['lambda_error_mean']:.4f}"
        )
        equation = equations.get(row["equation"], row["equation"])
        method = methods.get(row["arm"], row["arm"])
        lines.append(
            f"| {equation} | {method} | {row['n']} | {100 * row['field_l2_mean']:.4f} | "
            f"{row['field_10pct_count']}/{row['n']} | {coefficient} |"
        )
    lines += [
        "",
        "## 결과 해석의 범위",
        "",
        "- Adam과 선택 후처리 결과는 개별 실험 JSON에서 구분합니다. 동일 비용의 우월성을 추론하지 않습니다.",
        "- Wave 비교표와 도달 점수는 저장된 참조 격자를 사용합니다. 물리 감사는 별도의 고정 진단 격자를 사용합니다.",
        "- 서로 다른 방정식의 PDE 잔차 원값은 단위와 정규화 척도가 달라 직접 순위화하지 않습니다.",
        "- 순방향 문제의 계수 점수는 해당 없음이며 자동 만점이 아닙니다.",
        "- 경계·초기조건 및 국소 최악점의 합격 판정은 별도 검증 대상입니다.",
        "- 해석해로 생성한 합성 학습값은 실제 공학 관측 데이터와 동일하지 않습니다.",
        "- 저장된 실험의 검사 이력과 현재 실행한 검사는 구분해야 합니다.",
    ]
    (Path(root) / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def u_pick(folder, label, predicate):
    folder = Path(folder)
    while True:
        paths = []
        for p in sorted(folder.glob("*.zip")):
            try:
                with zipfile.ZipFile(p) as z:
                    if predicate(z):
                        paths.append(p)
            except (OSError, KeyError, ValueError, zipfile.BadZipFile):
                pass
        print(label)
        for i, p in enumerate(paths, 1):
            print(f"{i}: {p.name}")
        if not paths:
            raise FileNotFoundError("No matching ZIP in PINN for " + label)
        raw = input("Number / filename; Enter=only candidate; rescan / cancel: ").strip()
        if raw == "cancel":
            raise RuntimeError("Cancelled")
        if raw == "rescan":
            continue
        if not raw and len(paths) == 1:
            return paths[0]
        if raw.isdigit() and 1 <= int(raw) <= len(paths):
            return paths[int(raw) - 1]
        matches = [p for p in paths if p.name == raw]
        if len(matches) == 1:
            return matches[0]
        print("Use a displayed number or filename, not a full path.")
