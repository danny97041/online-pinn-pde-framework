"""Lazy numerical runtime for the eight non-Wave equation adapters.

Importing this file defines the factory without loading TensorFlow. Historical
helpers remain grouped inside it so saved model architectures can be restored.
The distribution adapter supplies the current settings and execution policy.
"""


def load_candidate_runtime(policy=None):
    """Load TensorFlow/SciPy helpers and return the candidate runtime namespace.

    Constructing the namespace does not run a benchmark or start an API server.
    Explicit trial calls perform training; quick results viewing never calls it.
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
        "field_target": 0.1,
        "coefficient_report_target": 0.1,
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
            json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
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
                or (":" in n)
                or ("\x00" in n)
                or stat.S_ISLNK(i.external_attr >> 16)
            ):
                raise ValueError("Unsafe ZIP entry: " + repr(n))
        if sum((i.file_size for i in z.infolist())) > 4 * 1024**3:
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
        root, output = (Path(root), Path(output))
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
            z.extractall(root)
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
        if arm not in ["loss_sa", "ff_loss_sa"] and (not np.array_equal(f, np.ones(3))):
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
        return np.diff([r["iteration"] for r in rows]).tolist() == [every] * (
            consecutive - 1
        ) and all(
            (
                np.isfinite(r["validation_l2"]) and r["validation_l2"] <= U_POLICY["field_target"]
                for r in rows
            )
        )

    def u_plan(profile, mode, equations):
        if profile not in ["review", "preview", "smoke", "pilot", "benchmark", "auto"]:
            raise ValueError("Unknown profile")
        if mode not in ["exp", "full"]:
            raise ValueError("Unknown mode")
        if (
            not equations
            or len(set(equations)) != len(equations)
            or (not set(equations) <= set(U_PROBLEMS))
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
        a = 3 if adam <= 0.05 else 2 if adam <= 0.1 else 0
        p = 2 if selected <= 0.05 else 1 if selected <= 0.1 else 0
        retained_cap = 3 if selected <= 0.05 else 2 if selected <= 0.1 else 0
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
                a, p = (r["adam"], r["post_selected"])
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
            r["field_attainment_points"] = u_quantity_score(
                r.get("adam_field_l2"), r.get("field_l2")
            )
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
                            if all(("lbfgs_objective_evaluations" in r for r in subset))
                            else None
                        ),
                        "field_10pct_count": sum((v <= 0.1 for v in fields)),
                        "coefficient_10pct_count": (
                            sum((v <= 0.1 for v in coeff)) if coeff else None
                        ),
                        "lambda_error_mean": float(np.mean(coeff)) if coeff else None,
                        "score_mean_out_of_6": (
                            float(np.mean([r["score_out_of_6"] for r in subset]))
                            if all((r["score_out_of_6"] is not None for r in subset))
                            else None
                        ),
                        "pde_grades": [r.get("pde_grade") for r in subset],
                    }
                )
        updates = sum((r.get("optimizer_updates_this_run", 0) for r in routes.values()))
        profile = json.loads((root / "contract.json").read_text())["plan"]["profile"]
        if profile == "review":
            updates = 0
        evaluations = (
            0
            if profile == "review"
            else sum((r.get("lbfgs_evaluations_this_run", 0) for r in routes.values()))
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
        lines = [
            "# V3 integrated comparison",
            "",
            "10% research milestone. 5% and local high-precision improvement deferred to V4.",
            "No overall physics approval. Historical 5% stopping is preserved, not rewritten as a 10% run.",
            "",
            "| Equation | Method | Seeds | Mean field L2 % | Field <=10% | Mean coefficient error % |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for r in groups:
            ce = (
                "N/A (forward)"
                if r["lambda_error_mean"] is None
                else f"{100 * r['lambda_error_mean']:.4f}"
            )
            lines.append(
                f"| {r['equation']} | {r['arm']} | {r['n']} | {100 * r['field_l2_mean']:.4f} | {r['field_10pct_count']}/{r['n']} | {ce} |"
            )
        lines += [
            "",
            "## Evidence limits",
            "",
            "- Compare Adam and selected postprocessing separately in per-trial JSON; no equal-cost superiority inferred.",
            "- Wave table/attainment scores use the historical reference grid. The physics audit uses its separate fixed diagnostic grids; its original scores remain in evidence JSON.",
            "- Different equations use different residual scales; raw residuals are not ranked across equations.",
            "- Forward problems have no learned-coefficient score, not an automatic full score.",
            "- BC/IC acceptance and local worst-point qualification remain deferred, not PASS.",
            "- Synthetic exact labels are not fixed real engineering observations.",
            "",
            "## Modules",
            "",
        ]
        lines += [f"- {k}: {v}" for k, v in modules.items()]
        (root / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return summary

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

    # Exact-solution and derivative checks validate equation wiring, not trained
    # model accuracy. Manufactured forcing is independent of the tested residual.
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
                    vt
                    + u * g[:, 1, 0]
                    + v * g[:, 1, 1]
                    + g[:, 2, 1]
                    - nu * (h[:, 1, 0] + h[:, 1, 1]),
                    ux + g[:, 1, 1],
                )
            )
        if name == "heat2d":
            return (g[:, 0, 2] - 0.1 * lap)[:, None]
        if name == "poisson2d":
            return (-lap - 2 * np.pi**2 * np.sin(np.pi * z[:, 0]) * np.sin(np.pi * z[:, 1]))[
                :, None
            ]
        if name == "darcy2d":
            sx, sy = (np.sin(np.pi * z[:, 0]), np.sin(np.pi * z[:, 1]))
            cx, cy = (np.cos(np.pi * z[:, 0]), np.cos(np.pi * z[:, 1]))
            k = 1 + 0.5 * sx * sy
            kx = 0.5 * np.pi * cx * sy
            ky = 0.5 * np.pi * sx * cy
            forcing = 2 * np.pi**2 * k * sx * sy - 0.5 * np.pi**2 * (
                (cx * sy) ** 2 + (sx * cy) ** 2
            )
            return (-k * lap - kx * ux - ky * uy - forcing)[:, None]
        if name == "reaction_diffusion":
            exact = 0.2 * np.sin(np.pi * z[:, 0]) * np.sin(np.pi * z[:, 1]) * np.exp(-z[:, 2])
            forcing = (-2 + 0.02 * np.pi**2) * exact + exact**3
            return (g[:, 0, 2] - 0.01 * lap - (u - u**3) - forcing)[:, None]
        raise ValueError(name)

    def candidate_fd(name, z, step=0.0001):
        q = candidate_field(name, z)
        g = np.zeros((*q.shape, z.shape[1]))
        h = g.copy()
        for j in range(z.shape[1]):
            dz = np.zeros_like(z)
            dz[:, j] = step
            p, m = (candidate_field(name, z + dz), candidate_field(name, z - dz))
            pp, mm = (candidate_field(name, z + 2 * dz), candidate_field(name, z - 2 * dz))
            g[:, :, j] = (-pp + 8 * p - 8 * m + mm) / (12 * step)
            h[:, :, j] = (-pp + 16 * p - 30 * q + 16 * m - mm) / (12 * step**2)
        return (q, g, h)

    def candidate_tf(name, z):
        import tensorflow as tf

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
                        outer.gradient(v, X, unconnected_gradients=tf.UnconnectedGradients.ZERO)[
                            :, j
                        ]
                        for j, v in enumerate(row)
                    ],
                    axis=1,
                )
            )
        del inner, outer
        return (q.numpy(), tf.stack(gradients, axis=1).numpy(), tf.stack(second, axis=1).numpy())

    def candidate_points(name, count=128):
        domain = np.array(CANDIDATES[name]["domain"], float)
        rng = np.random.default_rng(98271 + CANDIDATES[name]["priority"])
        unit = np.column_stack(
            [(rng.permutation(count) + rng.random(count)) / count for _ in domain]
        )
        return domain[:, 0] + unit * (domain[:, 1] - domain[:, 0])

    def candidate_audit(name, autodiff=True):
        points = candidate_points(name)
        if name == "burgers":
            t = np.linspace(0, 1, 32)
            x = 0.5 * t + np.linspace(-0.02, 0.02, 32)
            points = np.concatenate((points, np.column_stack((x, t))))
        fd = candidate_fd(name, points)
        q, g, h = candidate_tf(name, points) if autodiff else fd
        res = candidate_residual(name, points, q, g, h)
        residual_max = float(np.max(abs(res)))
        tolerance = 1e-08 if autodiff else 0.002 if name == "burgers" else 2e-05
        gradient_difference = float(np.max(abs(g - fd[1])))
        hessian_difference = float(np.max(abs(h - fd[2])))
        derivative_pass = gradient_difference < 0.0002 and hessian_difference < 0.1
        config = CANDIDATES[name]
        faces = []
        for axis, bounds in enumerate(config["domain"]):
            if axis == config["time_axis"]:
                continue
            lo = points[:32].copy()
            hi = lo.copy()
            lo[:, axis] = bounds[0]
            hi[:, axis] = bounds[1]
            left, right = (candidate_field(name, lo), candidate_field(name, hi))
            if name == "taylor_green":
                l = candidate_tf(name, lo) if autodiff else candidate_fd(name, lo)
                r = candidate_tf(name, hi) if autodiff else candidate_fd(name, hi)
                error = max(float(np.max(abs(left - right))), float(np.max(abs(l[1] - r[1]))))
                check = error < 1e-07
            elif config["boundary"] == "homogeneous Dirichlet":
                error = float(max(np.max(abs(left)), np.max(abs(right))))
                check = error < 1e-10
            else:
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
                "passed": bool(np.sqrt(np.mean(changed**2)) > 0.001),
                "scope": "exact-field sensitivity only; not uniqueness proof",
            }
        passed = (
            residual_max < tolerance
            and derivative_pass
            and all((f["passed"] for f in faces))
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

    if policy is not None:
        U_POLICY.update(policy)
    # Numerical adapters use the equation and archive helpers defined above.
    # Explicit training authorization is enforced by the distribution entrypoint;
    # historical pilot records are evidence, not a new automatic approval.
    import time
    import tempfile
    import tensorflow as tf
    from scipy.optimize import minimize

    class CandidateModel(tf.keras.Model):

        def __init__(self, name, arm, seed):
            super().__init__()
            self.problem, self.arm = (name, arm)
            c = CANDIDATES[name]
            domain = np.asarray(c["domain"], np.float32)
            self.lo = tf.constant(domain[:, 0])
            self.width = tf.constant(domain[:, 1] - domain[:, 0])
            self.output_count = len(c["outputs"])
            self.inverse = name == "kovasznay_inverse"
            self.nominal = 0.0125 if self.inverse else c.get("coefficient", 1.0)
            initial_ratio = U_POLICY.get("initial_coefficient_ratio", 0.5) / 0.5
            self.raw_coefficient = self.add_weight(
                name="log_coefficient_ratio",
                shape=(),
                initializer=tf.keras.initializers.Constant(np.log(initial_ratio)),
                trainable=self.inverse,
            )
            rng = np.random.default_rng(seed + 811)
            self.ff = (
                [
                    tf.constant(
                        rng.normal(
                            0, s, (len(domain), U_POLICY.get("ff_features_per_bank", 32))
                        ).astype(np.float32)
                    )
                    for s in U_POLICY["ff_scales"]
                ]
                if arm.startswith("ff")
                else []
            )
            depth = U_POLICY.get("network_depth", 5)
            self.hidden = [
                tf.keras.layers.Dense(
                    U_POLICY.get("network_width", 64),
                    activation="tanh",
                    kernel_initializer=tf.keras.initializers.GlorotNormal(seed + i),
                )
                for i in range(depth)
            ]
            self.out = tf.keras.layers.Dense(
                self.output_count,
                kernel_initializer=tf.keras.initializers.GlorotNormal(seed + depth),
            )

        @property
        def coefficient(self):
            return (
                tf.cast(self.nominal, tf.float32) * tf.exp(self.raw_coefficient)
                if self.inverse
                else tf.constant(self.nominal, tf.float32)
            )

        def call(self, x, training=False):
            z = 2 * (tf.cast(x, tf.float32) - self.lo) / self.width - 1
            if self.ff:
                parts = [2 * np.pi * tf.matmul(z, b) for b in self.ff]
                z = tf.concat([f(p) for p in parts for f in [tf.sin, tf.cos]], axis=1)
            for layer in self.hidden:
                z = layer(z)
            return self.out(z)

    def cu_derivatives(model, x):
        x = tf.convert_to_tensor(x, tf.float32)
        with tf.GradientTape(persistent=True) as t2:
            t2.watch(x)
            with tf.GradientTape(persistent=True) as t1:
                t1.watch(x)
                q = model(x)
                components = tf.unstack(q, axis=1)
            gradients = [
                t1.gradient(v, x, unconnected_gradients=tf.UnconnectedGradients.ZERO)
                for v in components
            ]
            parts = [tf.unstack(g, axis=1) for g in gradients]
        h = tf.stack(
            [
                tf.stack(
                    [
                        t2.gradient(v, x, unconnected_gradients=tf.UnconnectedGradients.ZERO)[:, j]
                        for j, v in enumerate(row)
                    ],
                    axis=1,
                )
                for row in parts
            ],
            axis=1,
        )
        del t1, t2
        return (q, tf.stack(gradients, axis=1), h)

    def cu_residual(name, x, q, g, h, coefficient):
        u = q[:, 0]
        ux = g[:, 0, 0]
        uxx = h[:, 0, 0]
        pi = np.pi
        if name == "burgers":
            return (g[:, 0, 1] + u * ux - coefficient * uxx)[:, None]
        uy = g[:, 0, 1]
        lap = uxx + h[:, 0, 1]
        if name.startswith("kovasznay") or name == "taylor_green":
            v = q[:, 1]
            ut = g[:, 0, 2] if name == "taylor_green" else 0.0
            vt = g[:, 1, 2] if name == "taylor_green" else 0.0
            return tf.stack(
                [
                    ut + u * ux + v * uy + g[:, 2, 0] - coefficient * lap,
                    vt
                    + u * g[:, 1, 0]
                    + v * g[:, 1, 1]
                    + g[:, 2, 1]
                    - coefficient * (h[:, 1, 0] + h[:, 1, 1]),
                    ux + g[:, 1, 1],
                ],
                axis=1,
            )
        if name == "heat2d":
            return (g[:, 0, 2] - coefficient * lap)[:, None]
        sx, sy = (tf.sin(pi * x[:, 0]), tf.sin(pi * x[:, 1]))
        if name == "poisson2d":
            return (-lap - 2 * pi * pi * sx * sy)[:, None]
        if name == "darcy2d":
            cx, cy = (tf.cos(pi * x[:, 0]), tf.cos(pi * x[:, 1]))
            k = 1 + 0.5 * sx * sy
            forcing = 2 * pi * pi * k * sx * sy - 0.5 * pi * pi * ((cx * sy) ** 2 + (sx * cy) ** 2)
            return (-k * lap - 0.5 * pi * cx * sy * ux - 0.5 * pi * sx * cy * uy - forcing)[:, None]
        if name == "reaction_diffusion":
            exact = 0.2 * sx * sy * tf.exp(-x[:, 2])
            forcing = (-2 + 0.02 * pi * pi) * exact + exact**3
            return (g[:, 0, 2] - coefficient * lap - (u - u**3) - forcing)[:, None]
        raise ValueError(name)

    def cu_lhs(domain, n, rng):
        domain = np.asarray(domain, float)
        unit = np.column_stack([(rng.permutation(n) + rng.random(n)) / n for _ in domain])
        return (domain[:, 0] + unit * (domain[:, 1] - domain[:, 0])).astype(np.float32)

    def cu_samples(name, seed, cycle, heldout=False):
        c = CANDIDATES[name]
        d = np.asarray(c["domain"], float)
        t = c["time_axis"]
        rng = np.random.default_rng(
            np.random.SeedSequence(
                [
                    982731 if heldout else 39183,
                    seed,
                    CANDIDATES[name]["priority"],
                    0 if heldout else cycle,
                ]
            )
        )
        a = {
            "collocation": cu_lhs(
                d, 512 if heldout else U_POLICY.get("collocation_count", 800), rng
            ),
            "data": cu_lhs(d, 512 if heldout else U_POLICY.get("data_count", 200), rng),
        }
        for axis in range(len(d)):
            if axis == t:
                continue
            lower = cu_lhs(d, 80 if heldout else U_POLICY.get("boundary_count_per_face", 80), rng)
            upper = lower.copy()
            lower[:, axis] = d[axis, 0]
            upper[:, axis] = d[axis, 1]
            a[f"lower_{axis}"] = lower
            a[f"upper_{axis}"] = upper
        if t is not None:
            a["initial"] = cu_lhs(d, 160 if heldout else U_POLICY.get("initial_count", 160), rng)
            a["initial"][:, t] = d[t, 0]
        if len(c["outputs"]) == 3:
            a["gauge"] = cu_lhs(d, 32, rng)
            a["gauge"][:, :2] = 0.0
        for k in list(a):
            if k != "collocation":
                a["label_" + k] = candidate_field(name, a[k].astype(float)).astype(np.float32)
        return a

    def cu_scales(name):
        """Same scales for all arms/seeds: explicit characteristic units, not initial loss."""
        c = CANDIDATES[name]
        points = candidate_points(name, 2048)
        q = candidate_field(name, points)
        amplitude = np.maximum(np.sqrt(np.mean(q * q, axis=0)), 0.001)
        length = np.min(np.ptp(np.array(c["domain"]), axis=1)[:2])
        if name == "burgers":
            length = 2.0
        nu = c.get("coefficient", 1.0)
        if name == "kovasznay_inverse":
            nu = 0.0125
        if len(c["outputs"]) == 3:
            speed = max(amplitude[:2])
            r = [speed**2 / length + amplitude[2] / length + nu * speed / length**2] * 2 + [
                speed / length
            ]
        elif name in ["poisson2d", "darcy2d"]:
            r = [amplitude[0] / length**2]
        elif name == "burgers":
            r = [amplitude[0] + amplitude[0] ** 2 / length + nu * amplitude[0] / length**2]
        else:
            r = [amplitude[0] + nu * amplitude[0] / length**2]
        return {
            "field": amplitude.tolist(),
            "residual": np.maximum(r, 1e-06).tolist(),
            "provenance": "fixed synthetic reference amplitudes + domain characteristic units, no model/seed dependence",
            "not_wave_R0": True,
            "pde_grade_transfer": False,
        }

    def cu_groups(model, a, scales):
        name = model.problem
        x = tf.convert_to_tensor(a["collocation"])
        q, g, h = cu_derivatives(model, x)
        residual = cu_residual(name, x, q, g, h, model.coefficient)
        fs = tf.constant(scales["field"], tf.float32)
        rs = tf.constant(scales["residual"], tf.float32)
        mse = lambda v: tf.reduce_mean(tf.square(v))
        physics = mse(residual / rs)
        data = mse((model(a["data"]) - a["label_data"]) / fs)
        bc = []
        initial = []
        for key in a:
            if not key.startswith("lower_"):
                continue
            axis = int(key.split("_")[1])
            upper = "upper_" + str(axis)
            if name == "taylor_green":
                q1, g1, _ = cu_derivatives(model, a[key])
                q2, g2, _ = cu_derivatives(model, a[upper])
                width = CANDIDATES[name]["domain"][axis][1] - CANDIDATES[name]["domain"][axis][0]
                bc += [mse((q1 - q2) / fs), mse((g1[:, :, axis] - g2[:, :, axis]) * width / fs)]
            else:
                n = 2 if len(scales["field"]) == 3 else 1
                bc += [
                    mse((model(a[k])[:, :n] - a["label_" + k][:, :n]) / fs[:n])
                    for k in [key, upper]
                ]
        if "initial" in a:
            initial = [mse((model(a["initial"]) - a["label_initial"]) / fs)]
        if "gauge" in a:
            bc.append(mse((model(a["gauge"])[:, 2] - a["label_gauge"][:, 2]) / fs[2]))
        boundary = tf.reduce_mean(tf.stack(bc))
        constraint = 0.5 * (boundary + initial[0]) if initial else boundary
        return tf.stack([physics, data, constraint])

    def cu_state(name, arm, seed):
        tf.keras.utils.set_random_seed(seed)
        m = CandidateModel(name, arm, seed)
        m(tf.zeros((1, len(CANDIDATES[name]["domain"])), tf.float32))
        net = [v for v in m.trainable_variables if v is not m.raw_coefficient]
        opt = tf.keras.optimizers.Adam(0.004)
        opt.build(net)
        lopt = tf.keras.optimizers.Adam(4e-06)
        lopt.build([m.raw_coefficient])
        factors = tf.Variable(np.ones(3, np.float32), trainable=False)
        count = tf.Variable(0, dtype=tf.int64, trainable=False)
        checkpoint = tf.train.Checkpoint(
            model=m, optimizer=opt, coefficient_optimizer=lopt, factors=factors, sa_updates=count
        )
        return (m, opt, lopt, factors, count, checkpoint, net)

    def cu_values(state):
        return [
            v.numpy().copy()
            for v in list(state[0].variables)
            + list(state[1].variables)
            + list(state[2].variables)
            + [state[3], state[4]]
        ]

    def cu_assert_same(a, b, approx=False):
        if len(a) != len(b) or not all(
            (
                np.allclose(x, y, rtol=1e-05, atol=1e-07) if approx else np.array_equal(x, y)
                for x, y in zip(a, b)
            )
        ):
            raise ValueError("State changed / reload mismatch")

    def cu_stepper(state, scales, compiled=True):
        m, opt, lo, _, _, _, network = state

        def step(a, weights):
            variables = network + [m.raw_coefficient] if m.inverse else network
            with tf.GradientTape() as tape:
                loss = tf.reduce_sum(cu_groups(m, a, scales) * weights)
            grads = tape.gradient(loss, variables)
            for g in grads:
                if g is None:
                    raise ValueError("Disconnected gradient")
                tf.debugging.assert_all_finite(g, "Nonfinite gradient")
            opt.apply_gradients(zip(grads[: len(network)], network))
            if m.inverse:
                lo.apply_gradients([(grads[-1], m.raw_coefficient)])
            return loss

        return tf.function(step, autograph=False, reduce_retracing=True) if compiled else step

    def cu_sa(state, a, scales):
        before = cu_values(state)
        with tf.GradientTape(persistent=True) as tape:
            terms = tf.unstack(cu_groups(state[0], a, scales))
        norms = [
            float(tf.linalg.global_norm([g for g in tape.gradient(t, state[6]) if g is not None]))
            for t in terms
        ]
        del tape
        cu_assert_same(before, cu_values(state))
        norms = np.asarray(norms)
        if not np.isfinite(norms).all():
            raise ValueError("Invalid SA gradients")
        goal = np.clip(np.mean(norms) / np.maximum(norms, 1e-12), *U_POLICY["sa_bounds"])
        ema = U_POLICY["sa_ema"]
        state[3].assign(ema * state[3] + (1 - ema) * goal.astype(np.float32))
        state[4].assign_add(1)

    def cu_metrics(model, points):
        exact = candidate_field(model.problem, np.asarray(points, float))
        pred = np.concatenate(
            [model(points[i : i + 256]).numpy() for i in range(0, len(points), 256)]
        )
        den = np.sum(exact**2, axis=0)
        if np.any(den <= 1e-16):
            raise ValueError("Undefined relative field component error")
        component = np.sqrt(np.sum((pred - exact) ** 2, axis=0) / den)
        if not np.isfinite(component).all():
            raise ValueError("Nonfinite field error")
        return {
            "relative_l2": float(np.max(component)),
            "component_relative_l2": component.tolist(),
            "aggregation": "maximum of per-output relative L2; pressure gauge fixed by training constraint",
        }

    def cu_measure(state, validation, reference, scales, it):
        before = cu_values(state)
        m = state[0]
        val = cu_metrics(m, validation["data"])
        ref = cu_metrics(m, reference)
        losses = cu_groups(m, validation, scales).numpy().astype(float)
        coeff = float(m.coefficient.numpy())
        row = {
            "iteration": it,
            "validation_l2": val["relative_l2"],
            "validation": val,
            "reference": ref,
            "coefficient": coeff,
            "lambda_error": (
                abs(coeff / CANDIDATES[m.problem]["coefficient"] - 1) if m.inverse else None
            ),
            "validation_normalized_groups": losses.tolist(),
            "factors": state[3].numpy().tolist(),
            "sa_updates": int(state[4].numpy()),
            "next_lrs": u_lrs(it),
            "cycle": u_cycle(it),
        }
        cu_assert_same(before, cu_values(state))
        return row

    def cu_physics_report(model, points, scales):
        views = {"learned_or_fixed": float(model.coefficient.numpy())}
        if model.inverse:
            views["true_reference"] = CANDIDATES[model.problem]["coefficient"]
        values = {k: [] for k in views}
        for start in range(0, len(points), 256):
            x = points[start : start + 256]
            q, g, h = cu_derivatives(model, x)
            for key, coefficient in views.items():
                residual = candidate_residual(
                    model.problem, x, q.numpy(), g.numpy(), h.numpy(), coefficient=coefficient
                )
                values[key].append(residual / np.asarray(scales["residual"]))
        report = {}
        for key, parts in values.items():
            v = np.concatenate(parts)
            report[key] = {
                "rms_by_equation": np.sqrt(np.mean(v * v, axis=0)).tolist(),
                "p99_by_equation": np.quantile(abs(v), 0.99, axis=0).tolist(),
                "max_by_equation": np.max(abs(v), axis=0).tolist(),
            }
        return {
            "point_count": len(points),
            "views": report,
            "normalization": scales,
            "grade": None,
            "note": "Equation-specific characteristic units; Wave RMS/p99 grades not transferred",
            "used_for_training_selection_stopping": False,
        }

    def cu_wiring(name):
        exact = candidate_audit(name, autodiff=True)
        if not exact["exact_audit_passed"]:
            raise ValueError("Exact PDE audit failed: " + name)
        z = candidate_points(name, 32)
        q, g, h = candidate_tf(name, z)
        for perturb in [0.0, 0.07]:
            qt, gt, ht = (q + perturb, g + perturb, h - perturb)
            nu = CANDIDATES[name].get("coefficient", 1.0)
            expected = candidate_residual(name, z, qt, gt, ht)
            got = cu_residual(
                name,
                *[tf.constant(v, tf.float64) for v in [z, qt, gt, ht]],
                tf.constant(nu, tf.float64),
            ).numpy()
            if not np.allclose(expected, got, rtol=1e-09, atol=1e-09):
                raise ValueError("Neural PDE formula differs from independent equation")
        a, b = (cu_samples(name, 3234, 0), cu_samples(name, 3234, 1))
        if any(
            (np.array_equal(a[k], b[k]) for k in a if not k.startswith("label_") and k != "gauge")
        ):
            raise ValueError("Training family did not renew")
        if u_digest(cu_samples(name, 3234, 0, True)) != u_digest(cu_samples(name, 3234, 8, True)):
            raise ValueError("Heldout changed")
        small = {k: tf.constant(v[:4]) for k, v in a.items()}
        scales = cu_scales(name)
        checks = []
        initials = {}
        for arm in U_ARMS:
            s = cu_state(name, arm, 3234)
            step = cu_stepper(s, scales, compiled=False)
            initials[arm] = u_digest({f"v{i}": v.numpy() for i, v in enumerate(s[0].variables)})
            step(small, tf.constant(u_weights(arm, 0, np.ones(3)), tf.float32))
            if "loss_sa" in arm:
                cu_sa(s, small, scales)
            with tempfile.TemporaryDirectory() as td:
                s[5].write(str(Path(td) / "state"))
                clone = cu_state(name, arm, 3234)
                clone[5].read(str(Path(td) / "state")).assert_consumed()
                cu_assert_same(cu_values(s), cu_values(clone))
                step(small, tf.constant(u_weights(arm, 1, s[3].numpy()), tf.float32))
                cu_stepper(clone, scales, compiled=False)(
                    small, tf.constant(u_weights(arm, 1, clone[3].numpy()), tf.float32)
                )
                cu_assert_same(cu_values(s), cu_values(clone), True)
            checks.append({"arm": arm, "same_next_update": True})
        if (
            initials["baseline"] != initials["loss_sa"]
            or not initials["ff"] == initials["ff_loss_sa"] == initials["ff_curriculum"]
        ):
            raise ValueError("Paired network initialization differs")
        return {
            "passed": True,
            "exact": exact,
            "neural_formula_equivalence": True,
            "checks": checks,
            "paired_initialization_verified": True,
            "initial_model_sha256": initials,
            "scope": "disposable probes; no convergence or CPU qualification claim",
        }

    def cu_save_weights(model, path):
        np.savez_compressed(path, **{f"v{i}": v.numpy() for i, v in enumerate(model.variables)})

    def cu_load_weights(model, path):
        with np.load(path, allow_pickle=False) as a:
            if set(a.files) != {f"v{i}" for i in range(len(model.variables))}:
                raise ValueError("Model inventory mismatch")
            for i, v in enumerate(model.variables):
                x = a[f"v{i}"]
                if x.shape != tuple(v.shape) or not np.isfinite(x).all():
                    raise ValueError("Model shape/value mismatch")
                v.assign(x)

    def cu_trial(root, name, arm, seed, cap, lbiter, resume=None):
        root = Path(root)
        result = root / "result.json"
        if result.exists():
            return cu_review(root, name, arm, seed, cap)
        if root.exists():
            raise RuntimeError(
                "Incomplete local trial; reconnect and load last completed rolling ZIP"
            )
        root.mkdir(parents=True)
        s = cu_state(name, arm, seed)
        scales = cu_scales(name)
        step = cu_stepper(s, scales)
        validation = cu_samples(name, 917221, 0, True)
        protected = u_digest(validation)
        reference = cu_lhs(CANDIDATES[name]["domain"], 4096, np.random.default_rng(739921))
        history = []
        sampling = []
        start = time.perf_counter()
        current = None
        first = 1
        if resume is not None:
            previous = json.loads((Path(resume) / "result.json").read_text())
            if (previous["equation"], previous["arm"], previous["seed"]) != (name, arm, seed):
                raise ValueError("Resume identity mismatch")
            s[5].read(str(Path(resume) / "adam_state")).assert_consumed()
            first = previous["adam"]["iteration"] + 1
            if first > cap:
                raise ValueError("No additional updates requested")
            history = list(previous["history"])
            sampling = list(previous["sampling"])
            if u_digest(validation) != previous["heldout_sha256"]:
                raise ValueError("Resume heldout mismatch")
        for it in range(first, cap + 1):
            cycle = u_cycle(it)
            if cycle != current:
                a = cu_samples(name, seed, cycle)
                current = cycle
                entry = {"cycle": cycle, "sha256": u_digest(a)}
                if not sampling or sampling[-1] != entry:
                    sampling.append(entry)
                tensors = {k: tf.constant(v) for k, v in a.items()}
            lr, lc = u_lrs(it - 1)
            s[1].learning_rate.assign(lr)
            s[2].learning_rate.assign(lc)
            step(tensors, tf.constant(u_weights(arm, it - 1, s[3].numpy()), tf.float32))
            if arm in ["loss_sa", "ff_loss_sa"] and it % U_POLICY["sa_every"] == 0:
                cu_sa(s, tensors, scales)
            if it % U_POLICY["stop_every"] == 0 or it == cap:
                r = cu_measure(s, validation, reference, scales, it)
                history.append(r)
                print(name, seed, arm, it, "val L2", round(r["validation_l2"], 6), flush=True)
                if u_target(history):
                    break
        adam_seconds = time.perf_counter() - start
        if u_digest(validation) != protected:
            raise ValueError("Heldout mutation")
        s[5].write(str(root / "adam_state"))
        cu_save_weights(s[0], root / "adam_weights.npz")
        np.savez_compressed(root / "current_training.npz", **a)
        adam = history[-1]
        weights = tf.constant(u_weights(arm, it, s[3].numpy()), tf.float32)
        frozen = [
            v.numpy().copy() for v in list(s[1].variables) + list(s[2].variables) + [s[3], s[4]]
        ]
        variables = list(s[0].trainable_variables)
        shapes = [tuple(v.shape) for v in variables]
        sizes = [int(np.prod(sh)) for sh in shapes]

        def vector():
            return np.concatenate([v.numpy().ravel() for v in variables]).astype(float)

        def assign(x):
            offset = 0
            for v, sh, n in zip(variables, shapes, sizes):
                v.assign(x[offset : offset + n].reshape(sh))
                offset += n

        calls = 0

        def objective(x):
            nonlocal calls
            calls += 1
            assign(x)
            with tf.GradientTape() as tape:
                loss = tf.reduce_sum(cu_groups(s[0], tensors, scales) * weights)
            grads = tape.gradient(loss, variables)
            if any((g is None for g in grads)):
                raise ValueError("Disconnected LBFGS gradient")
            grad = np.concatenate([g.numpy().ravel() for g in grads]).astype(float)
            value = float(loss)
            if not np.isfinite(value) or not np.isfinite(grad).all():
                raise ValueError("Nonfinite LBFGS objective")
            return (value, grad)

        x0 = vector()
        before, _ = objective(x0)
        lb_start = time.perf_counter()
        try:
            if lbiter <= 0:
                raise ValueError("L-BFGS disabled by explicit settings")
            solved = minimize(
                objective,
                x0,
                jac=True,
                method="L-BFGS-B",
                options={"maxiter": lbiter, "maxls": 30, "ftol": 1e-12, "gtol": 1e-08},
            )
            assign(solved.x)
            after, _ = objective(solved.x)
            candidate = cu_measure(s, validation, reference, scales, it)
            accepted = bool(after <= before and candidate["validation_l2"] <= adam["validation_l2"])
            solver = {
                "success": bool(solved.success),
                "message": str(solved.message),
                "iterations": int(solved.nit),
            }
        except ValueError as exc:
            accepted = False
            candidate = None
            after = None
            solver = {"success": False, "message": str(exc), "iterations": None}
        if not accepted:
            assign(x0)
        selected = cu_measure(s, validation, reference, scales, it)
        test = cu_samples(name, 917222, 0, True)
        test_before = cu_values(s)
        selected["test"] = cu_metrics(s[0], test["data"])
        selected["test_normalized_groups"] = (
            cu_groups(s[0], test, scales).numpy().astype(float).tolist()
        )
        selected["independent_physics"] = cu_physics_report(s[0], reference, scales)
        cu_assert_same(test_before, cu_values(s))
        cu_assert_same(
            frozen,
            [v.numpy().copy() for v in list(s[1].variables) + list(s[2].variables) + [s[3], s[4]]],
        )
        cu_save_weights(s[0], root / "selected_weights.npz")
        report = {
            "complete": True,
            "equation": name,
            "arm": arm,
            "seed": seed,
            "cap": cap,
            "scales": scales,
            "history": history,
            "sampling": sampling,
            "heldout_sha256": protected,
            "test_sha256": u_digest(test),
            "adam": adam,
            "selected": selected,
            "post_candidate": candidate,
            "post_selected": accepted,
            "solver": {**solver, "evaluations": calls, "before": before, "after": after},
            "seconds": {"adam": adam_seconds, "lbfgs": time.perf_counter() - lb_start},
            "stop_reason": "validation_10pct" if u_target(history) else "cap_reached",
            "target_uses_reference_or_true_coefficient": False,
            "cpu_qualified": False,
            "architecture": "candidate generalized 32-feature multiscale FF; not historical Wave product trunk",
        }
        u_json(result, report)
        return cu_review(root, name, arm, seed, cap)

    def cu_review(root, name, arm, seed, cap):
        root = Path(root)
        r = json.loads((root / "result.json").read_text())
        if (r["equation"], r["arm"], r["seed"], r["cap"], r["complete"]) != (
            name,
            arm,
            seed,
            cap,
            True,
        ):
            raise ValueError("Trial contract mismatch")
        s = cu_state(name, arm, seed)
        s[5].read(str(root / "adam_state")).assert_consumed()
        if r["scales"] != cu_scales(name):
            raise ValueError("Normalization contract changed")
        with np.load(root / "adam_weights.npz", allow_pickle=False) as saved:
            if len(saved.files) != len(s[0].variables) or any(
                (
                    not np.array_equal(saved[f"v{i}"], v.numpy())
                    for i, v in enumerate(s[0].variables)
                )
            ):
                raise ValueError("Adam checkpoint and inference weights disagree")
        it = r["adam"]["iteration"]
        if int(s[1].iterations.numpy()) != it or (
            s[0].inverse and int(s[2].iterations.numpy()) != it
        ):
            raise ValueError("Counter mismatch")
        if int(s[4].numpy()) != (
            it // U_POLICY["sa_every"] if arm in ["loss_sa", "ff_loss_sa"] else 0
        ):
            raise ValueError("SA count mismatch")
        if it < cap and (not u_target(r["history"])):
            raise ValueError("Invalid early stopping")
        validation = cu_samples(name, 917221, 0, True)
        if u_digest(validation) != r["heldout_sha256"]:
            raise ValueError("Heldout mismatch")
        with np.load(root / "current_training.npz", allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
        if u_digest(arrays) != u_digest(cu_samples(name, seed, u_cycle(it))):
            raise ValueError("LHS mismatch")
        expected_sampling = [
            {"cycle": cycle, "sha256": u_digest(cu_samples(name, seed, cycle))}
            for cycle in range(u_cycle(it) + 1)
        ]
        if r["sampling"] != expected_sampling:
            raise ValueError("Sampling cycle history mismatch")
        reference = cu_lhs(CANDIDATES[name]["domain"], 4096, np.random.default_rng(739921))
        a = cu_measure(s, validation, reference, r["scales"], it)
        for key in ["validation_l2", "coefficient"]:
            if not np.isclose(a[key], r["adam"][key], rtol=0.001, atol=1e-06):
                raise ValueError("Adam reload metric mismatch")
        if not np.isclose(
            a["reference"]["relative_l2"],
            r["adam"]["reference"]["relative_l2"],
            rtol=0.001,
            atol=1e-06,
        ):
            raise ValueError("Adam reference metric mismatch")
        cu_load_weights(s[0], root / "selected_weights.npz")
        p = cu_measure(s, validation, reference, r["scales"], it)
        for key in ["validation_l2", "coefficient"]:
            if not np.isclose(p[key], r["selected"][key], rtol=0.001, atol=1e-06):
                raise ValueError("Selected reload metric mismatch")
        if not np.isclose(
            p["reference"]["relative_l2"],
            r["selected"]["reference"]["relative_l2"],
            rtol=0.001,
            atol=1e-06,
        ):
            raise ValueError("Selected reference metric mismatch")
        test = cu_samples(name, 917222, 0, True)
        if u_digest(test) != r["test_sha256"]:
            raise ValueError("Test point mismatch")
        if not np.isclose(
            cu_metrics(s[0], test["data"])["relative_l2"],
            r["selected"]["test"]["relative_l2"],
            rtol=0.001,
            atol=1e-06,
        ):
            raise ValueError("Test metric reload mismatch")
        r["reload_passed"] = True
        u_json(
            root / "reload.json",
            {"passed": True, "cpu_runtime": not bool(tf.config.list_physical_devices("GPU"))},
        )
        return r

    "Explicit additional experiments, loaded with the matching numerical runtime."

    def release_candidate_trial(trial, settings, parent_trial=None):
        return cu_trial(
            trial,
            settings["equation"],
            settings["arms"][0] if len(settings["arms"]) == 1 else trial.name.split("_", 2)[2],
            settings["seed"],
            settings["cap"],
            settings["lbfgs_maxiter"],
            resume=parent_trial,
        )

    return locals()
