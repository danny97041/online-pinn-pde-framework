"""Saved Wave model reconstruction and read-only research diagnostics.

TensorFlow is loaded when the factory is called, not when this file is imported.
Archive contents supply data and weights, never executable Python source.
"""


def load_wave_inference():
    """Return the Wave inference and audit helpers without training a model."""
    import tensorflow as tf

    # Audit residual scaling is separate from training-loss calibration. Saved
    # model normalization is restored where required for prediction. Research
    # thresholds are not literature-wide acceptance limits or industrial standards.
    import hashlib
    import io
    import json
    import math
    import os
    import stat
    import zipfile
    from pathlib import Path, PurePosixPath
    import numpy as np

    SUPPORTED_CODE = "1f1b147ec669ec1322cb9671be016d444f3b60031a097ca7982fcf199bef3877"
    SUPPORTED_SOURCE = "bfa82fe6c50742180eab37b42d73a5c7a25f879e0132313cde7db0f002136564"
    ARMS = ["baseline", "loss_sa", "ff", "ff_loss_sa", "ff_curriculum"]
    AUDIT_POLICY = {
        "version": "wave-common-physics-v1",
        "threshold_provenance": "Proposed research rubric after seeing historical results; not a published acceptance standard",
        "scope": "Post-hoc synthetic Wave benchmark audit; not industrial certification or domain-wide bound",
        "diagnostic_seed": 937261,
        "grid_shape": [32, 32, 32],
        "independent_lhs_count": 16384,
        "surface_shape": [33, 33],
        "initial_shape": [65, 65],
        "batch_size": 256,
        "residual_scale": "Exact uniform-domain RMS(u_true_tt), analytic integral checked by Gauss quadrature",
        "coefficient_views": ["learned", "true"],
        "strict": {"rms": 0.01, "p99_absolute": 0.05},
        "basic": {"rms": 0.05, "p99_absolute": 0.1},
        "sensitivity_rms": [0.01, 0.02, 0.05],
        "threshold_float64_roundoff_rtol": 1e-12,
        "pde_grade": "Require both coefficient views on each of midpoint-grid and independent LHS",
        "constraints_acceptance": None,
        "overall_physics_status": "PENDING: BC/IC tolerances and worst-point assessment not yet approved",
        "field_score_points": "New midpoint grid, identical for both saved stages and all trials",
        "score": "Per quantity max eligible stage score; Adam credit retained only if final model retains that band",
        "adam_points": {"five_percent": 3, "ten_percent": 2},
        "post_points": {"five_percent": 2, "ten_percent": 1},
        "used_for_training_selection_stopping": False,
        "training_executed": False,
        "automatic_promotion": False,
    }

    def audit_sha(path):
        h = hashlib.sha256()
        with Path(path).open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    def audit_json(path, value):
        Path(path).write_text(
            json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    def audit_zip_members(z):
        names = z.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Duplicate ZIP members")
        for info in z.infolist():
            name = info.orig_filename
            p = PurePosixPath(name)
            if (
                p.is_absolute()
                or ".." in p.parts
                or "\\" in name
                or (":" in name)
                or ("\x00" in name)
            ):
                raise ValueError("Unsafe ZIP member: " + name)
            if stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError("ZIP links forbidden")
        if sum((i.file_size for i in z.infolist())) > 4 * 1024**3:
            raise ValueError("Archive exceeds this audit size limit")
        if z.testzip() is not None:
            raise ValueError("Corrupt ZIP")
        return names

    def audit_inputs(source, parent):
        source, parent = (Path(source), Path(parent))
        if source.resolve() == parent.resolve():
            raise ValueError("Source and result must differ")
        if audit_sha(source) != SUPPORTED_SOURCE:
            raise ValueError("Unsupported Stage7 source ZIP")
        with zipfile.ZipFile(source) as z:
            audit_zip_members(z)
            config = json.loads(z.read("config/wave2d_config.json"))
            with np.load(io.BytesIO(z.read("training/wave2d_data.npz")), allow_pickle=False) as a:
                arrays = {k: a[k].copy() for k in a.files}
        with zipfile.ZipFile(parent) as z:
            names = audit_zip_members(z)
            manifest = json.loads(z.read("periodic_manifest.json"))["files"]
            files = {i.filename for i in z.infolist() if not i.is_dir()}
            if files != set(manifest) | {"periodic_manifest.json"}:
                raise ValueError("Manifest inventory mismatch")
            for name, digest in manifest.items():
                if hashlib.sha256(z.read(name)).hexdigest() != digest:
                    raise ValueError("Checksum mismatch: " + name)
            contract, summary = [json.loads(z.read(n + ".json")) for n in ("contract", "summary")]
            if (
                contract.get("experiment") != "wave_crosslr"
                or contract.get("code_sha256") != SUPPORTED_CODE
                or contract.get("source_sha256") != SUPPORTED_SOURCE
                or (contract.get("profile") != "benchmark")
                or (contract.get("seeds") != [3234, 3235, 3236])
                or (contract.get("cap") != 100000)
                or (set(contract.get("arms", {})) != set(ARMS))
            ):
                raise ValueError("Unsupported benchmark contract")
            if summary.get("status") != "complete" or summary.get("completed_trials") != 15:
                raise ValueError("Complete 15-trial benchmark required")
            reports = []
            for seed in contract["seeds"]:
                for arm in ARMS:
                    folder = f"trials/seed_{seed}_{arm}"
                    a = json.loads(z.read(folder + "/result.json"))
                    b = json.loads(z.read("postprocess/" + folder + "/result.json"))
                    if (
                        not a.get("complete")
                        or not b.get("complete")
                        or a.get("arm") != arm
                        or (a.get("sampling_seed") != seed)
                        or (b.get("arm") != arm)
                        or (b.get("seed") != seed)
                        or (
                            b["parent_result_sha256"]
                            != hashlib.sha256(z.read(folder + "/result.json")).hexdigest()
                        )
                    ):
                        raise ValueError("Trial provenance mismatch")
                    for name in (
                        folder + "/last.weights.h5",
                        "postprocess/" + folder + "/selected.weights.h5",
                    ):
                        if name not in names:
                            raise ValueError("Missing model weights")
                    expected = (
                        b["candidate"] if b["decision"]["selected"] == "candidate" else b["before"]
                    )
                    if b["selected"] != expected:
                        raise ValueError("Selected metric provenance mismatch")
                    reports.append({"seed": seed, "arm": arm, "adam": a, "post": b})
        return (config, arrays, contract, reports)

    def audit_bounds(config):
        d = config["domain"]
        return np.array([[d[a + "_min"], d[a + "_max"]] for a in ("x", "y", "t")], dtype=np.float64)

    def audit_exact(x, config):
        x = np.asarray(x, dtype=np.float64)
        e = config["equation"]
        kx, ky = (np.pi * e["kx"], np.pi * e["ky"])
        omega = e["wave_speed"] * math.sqrt(e["lambda_target"] * (kx * kx + ky * ky))
        space = np.sin(kx * x[:, 0]) * np.sin(ky * x[:, 1])
        u = space * np.cos(omega * x[:, 2])
        return {
            "u": u,
            "ut": -omega * space * np.sin(omega * x[:, 2]),
            "utt": -(omega**2) * u,
            "lap": -(kx * kx + ky * ky) * u,
        }

    def audit_trig_mean_square(k, lo, hi, cosine=False):
        if not hi > lo:
            raise ValueError("Empty domain")
        if abs(k) < 1e-14:
            return float(cosine)
        correction = (np.sin(2 * k * hi) - np.sin(2 * k * lo)) / (4 * k * (hi - lo))
        return float(0.5 + correction if cosine else 0.5 - correction)

    def audit_scales(config):
        e, bounds = (config["equation"], audit_bounds(config))
        if e["lambda_target"] <= 0 or e["wave_speed"] <= 0:
            raise ValueError("Positive Wave coefficient required")
        kx, ky = (np.pi * e["kx"], np.pi * e["ky"])
        omega = e["wave_speed"] * math.sqrt(e["lambda_target"] * (kx * kx + ky * ky))
        space2 = audit_trig_mean_square(kx, *bounds[0]) * audit_trig_mean_square(ky, *bounds[1])
        u2 = space2 * audit_trig_mean_square(omega, *bounds[2], cosine=True)
        v2 = omega**2 * space2 * audit_trig_mean_square(omega, *bounds[2])
        ic2 = space2 * np.cos(omega * bounds[2, 0]) ** 2
        scales = {
            "field_rms": math.sqrt(u2),
            "velocity_rms": math.sqrt(v2),
            "initial_displacement_rms": math.sqrt(ic2),
            "residual_rms_scale": omega**2 * math.sqrt(u2),
            "omega": omega,
        }
        if not all((np.isfinite(v) and v > 1e-12 for v in scales.values())):
            raise ValueError(
                "Degenerate reference scale: define a separate protocol, do not add epsilon silently"
            )
        q, w = np.polynomial.legendre.leggauss(48)
        axes = [lo + (q + 1) * (hi - lo) / 2 for lo, hi in bounds]
        xyz = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
        weights = (w[:, None, None] * w[None, :, None] * w[None, None, :] / 8).ravel()
        observed = math.sqrt(float(np.dot(weights, audit_exact(xyz, config)["utt"] ** 2)))
        if not np.isclose(observed, scales["residual_rms_scale"], rtol=1e-11, atol=1e-12):
            raise RuntimeError("Analytic/quadrature normalization mismatch")
        return {
            **scales,
            "quadrature_r0": observed,
            "analytic_quadrature_passed": True,
            "policy": AUDIT_POLICY["residual_scale"],
            "independent_of_model_seed_SA": True,
        }

    def audit_points(config):
        bounds = audit_bounds(config)
        axes = [
            lo + (np.arange(n) + 0.5) * (hi - lo) / n
            for (lo, hi), n in zip(bounds, AUDIT_POLICY["grid_shape"])
        ]
        grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
        rng = np.random.default_rng(AUDIT_POLICY["diagnostic_seed"])
        count = AUDIT_POLICY["independent_lhs_count"]
        lhs = np.column_stack(
            [(rng.permutation(count) + rng.random(count)) / count for _ in range(3)]
        )
        lhs = bounds[:, 0] + lhs * (bounds[:, 1] - bounds[:, 0])
        points = {"grid": grid.astype(np.float32), "lhs": lhs.astype(np.float32)}
        for axis, name in ((0, "x"), (1, "y")):
            other = 1 - axis
            p, t = np.meshgrid(
                np.linspace(*bounds[other], AUDIT_POLICY["surface_shape"][0]),
                np.linspace(*bounds[2], AUDIT_POLICY["surface_shape"][1]),
                indexing="ij",
            )
            for side in (0, 1):
                x = np.empty((p.size, 3))
                x[:, other] = p.ravel()
                x[:, 2] = t.ravel()
                x[:, axis] = bounds[axis, side]
                points[name + ("_min" if side == 0 else "_max")] = x.astype(np.float32)
        x, y = np.meshgrid(
            *[np.linspace(*bounds[i], AUDIT_POLICY["initial_shape"][i]) for i in (0, 1)],
            indexing="ij",
        )
        points["initial"] = np.column_stack(
            (x.ravel(), y.ravel(), np.full(x.size, bounds[2, 0]))
        ).astype(np.float32)
        return points

    def audit_digest_arrays(arrays):
        h = hashlib.sha256()
        for name, a in sorted(arrays.items()):
            a = np.ascontiguousarray(a)
            h.update(name.encode())
            h.update(str((a.shape, a.dtype.str)).encode())
            h.update(a.tobytes())
        return h.hexdigest()

    def audit_within(value, threshold):
        return bool(value <= threshold or np.isclose(value, threshold, rtol=1e-12, atol=0))

    def audit_residual_stats(residual, scale):
        r = np.asarray(residual, dtype=np.float64).ravel()
        if not r.size or not np.isfinite(r).all() or (not np.isfinite(scale)) or (scale <= 0):
            raise ValueError("Invalid residual/scale")
        n = np.abs(r) / scale
        rms, p99 = (float(np.sqrt(np.mean(n * n))), float(np.quantile(n, 0.99)))
        grade = "NOT_MET"
        for name in ("basic", "strict"):
            limit = AUDIT_POLICY[name]
            if audit_within(rms, limit["rms"]) and audit_within(p99, limit["p99_absolute"]):
                grade = name.upper()
        return {
            "raw_mse": float(np.mean(r * r)),
            "raw_rms": float(np.sqrt(np.mean(r * r))),
            "normalized_rms": rms,
            "normalized_mse": rms * rms,
            "p95_absolute_normalized": float(np.quantile(n, 0.95)),
            "p99_absolute_normalized": p99,
            "maximum_absolute_normalized": float(n.max()),
            "grade": grade,
            "rms_threshold_sensitivity": {
                str(t): audit_within(rms, t) for t in AUDIT_POLICY["sensitivity_rms"]
            },
        }

    def audit_field_stats(prediction, exact, global_scale):
        p, e = [np.asarray(v, dtype=np.float64).ravel() for v in (prediction, exact)]
        if (
            p.shape != e.shape
            or not p.size
            or (not np.isfinite(p).all())
            or (not np.isfinite(e).all())
        ):
            raise ValueError("Invalid field")
        delta = p - e
        denominator = float(np.linalg.norm(e))
        return {
            "relative_l2": (
                float(np.linalg.norm(delta) / denominator)
                if denominator > 1e-12 * math.sqrt(e.size) * global_scale
                else None
            ),
            "mse": float(np.mean(delta**2)),
            "rmse_over_global_field_scale": float(np.sqrt(np.mean(delta**2)) / global_scale),
            "maximum_absolute_error": float(np.max(np.abs(delta))),
        }

    def audit_score(adam_error, selected_error):
        if not all((np.isfinite(v) and v >= 0 for v in (adam_error, selected_error))):
            raise ValueError("Invalid score errors")

        def band(e):
            return 5 if audit_within(e, 0.05) else 10 if audit_within(e, 0.1) else None

        a, s = (band(adam_error), band(selected_error))
        eligible_adam = min({5: 3, 10: 2, None: 0}[a], {5: 3, 10: 2, None: 0}[s])
        post = {5: 2, 10: 1, None: 0}[s]
        return {
            "points": max(eligible_adam, post),
            "adam_error": float(adam_error),
            "selected_error": float(selected_error),
            "retained_adam_points": eligible_adam,
            "post_points": post,
        }

    def audit_overall_pde(stages):
        grades = [
            stages[set_name]["residual"][view]["grade"]
            for set_name in ("grid", "lhs")
            for view in ("learned", "true")
        ]
        return (
            "STRICT"
            if all((g == "STRICT" for g in grades))
            else "BASIC" if all((g in ("STRICT", "BASIC") for g in grades)) else "NOT_MET"
        )

    def audit_regional(x, u, exact, residual, config, scales):
        bounds = audit_bounds(config)
        z = (x - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])
        sectors = np.minimum((z * 3).astype(int), 2)
        result = []
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    mask = np.all(sectors == [i, j, k], axis=1)
                    result.append(
                        {
                            "sector": [i, j, k],
                            "count": int(mask.sum()),
                            "field": audit_field_stats(u[mask], exact[mask], scales["field_rms"]),
                            "residual_true": audit_residual_stats(
                                residual[mask], scales["residual_rms_scale"]
                            ),
                        }
                    )
        return result

    def audit_pack(root, output):
        """One report-only rolling ZIP. Never includes input models or modifies inputs."""
        root, output = (Path(root), Path(output))
        files = {
            p.relative_to(root).as_posix(): audit_sha(p)
            for p in root.rglob("*")
            if p.is_file() and p.name != "audit_manifest.json"
        }
        audit_json(root / "audit_manifest.json", {"files": files})
        temporary = output.with_suffix(".writing.zip")
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as z:
            for p in sorted(root.rglob("*")):
                if p.is_file():
                    z.write(p, p.relative_to(root).as_posix())
        with zipfile.ZipFile(temporary) as z:
            if z.testzip() is not None:
                raise RuntimeError("Report ZIP verification failed")
        os.replace(temporary, output)

    class WavePINN2D(tf.keras.Model):

        def __init__(self, model_config):
            super().__init__()
            model_cfg = model_config["model"]
            domain_cfg = model_config["domain"]
            equation_cfg = model_config["equation"]
            self.wave_speed_squared = tf.constant(equation_cfg["wave_speed"] ** 2, dtype=tf.float32)
            self.lb = tf.constant(
                [domain_cfg["x_min"], domain_cfg["y_min"], domain_cfg["t_min"]], dtype=tf.float32
            )
            self.ub = tf.constant(
                [domain_cfg["x_max"], domain_cfg["y_max"], domain_cfg["t_max"]], dtype=tf.float32
            )
            self.hidden_layers = [
                tf.keras.layers.Dense(
                    units,
                    activation=model_cfg["activation"],
                    kernel_initializer="glorot_normal",
                    bias_initializer="zeros",
                )
                for units in model_cfg["layers"][1:-1]
            ]
            self.output_layer = tf.keras.layers.Dense(
                model_cfg["layers"][-1],
                activation=None,
                kernel_initializer="glorot_normal",
                bias_initializer="zeros",
            )
            self.lambda_1 = self.add_weight(
                name="lambda_1",
                shape=(),
                initializer=tf.keras.initializers.Constant(equation_cfg["lambda_initial"]),
                trainable=True,
                dtype=tf.float32,
            )

        def call(self, X, training=False):
            X = tf.cast(X, tf.float32)
            hidden = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
            for layer in self.hidden_layers:
                hidden = layer(hidden, training=training)
            return self.output_layer(hidden, training=training)

        def pde_residual(self, X):
            X = tf.cast(X, tf.float32)
            x_value = X[:, 0:1]
            y_value = X[:, 1:2]
            t_value = X[:, 2:3]
            with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape2:
                tape2.watch([x_value, y_value, t_value])
                with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape1:
                    tape1.watch([x_value, y_value, t_value])
                    prediction = self(tf.concat([x_value, y_value, t_value], axis=1), training=True)
                u_x = tape1.gradient(prediction, x_value)
                u_y = tape1.gradient(prediction, y_value)
                u_t = tape1.gradient(prediction, t_value)
            u_xx = tape2.gradient(u_x, x_value)
            u_yy = tape2.gradient(u_y, y_value)
            u_tt = tape2.gradient(u_t, t_value)
            del tape1
            del tape2
            return u_tt - self.lambda_1 * self.wave_speed_squared * (u_xx + u_yy)

    class WavePINN2DComparison(WavePINN2D):

        def __init__(self, model_config, use_fourier, fourier_seed):
            super().__init__(model_config)
            self.use_fourier = bool(use_fourier)
            fourier_cfg = model_config["method_comparison"]["fourier"]
            if self.use_fourier:
                generator = np.random.default_rng(fourier_seed)
                matrix = generator.normal(
                    loc=0.0, scale=fourier_cfg["scale"], size=(3, fourier_cfg["feature_count"])
                ).astype(np.float32)
                self.fourier_matrix = tf.constant(matrix, dtype=tf.float32)
            else:
                self.fourier_matrix = None

        def call(self, X, training=False):
            X = tf.cast(X, tf.float32)
            normalized = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
            if self.use_fourier:
                projection = 2.0 * np.pi * tf.matmul(normalized, self.fourier_matrix)
                hidden = tf.concat([normalized, tf.sin(projection), tf.cos(projection)], axis=1)
            else:
                hidden = normalized
            for layer in self.hidden_layers:
                hidden = layer(hidden, training=training)
            return self.output_layer(hidden, training=training)

    def ff_banks(spec, seed):
        """Local RNG: never consume training or global numpy random state."""
        if spec["kind"] == "plain":
            return []
        dimensions = (2, 1) if spec["kind"] == "spacetime" else (3,)
        banks = []
        for group, dimension in enumerate(dimensions):
            rng = np.random.default_rng(
                seed if group == 0 else np.random.SeedSequence([seed, group])
            )
            standard = rng.normal(size=(dimension, spec["features_per_bank"]))
            for scale in spec["scales"]:
                banks.append((standard * scale).astype(np.float32))
        return banks

    class WaveFFResearch(WavePINN2D):

        def __init__(self, config, spec, seed):
            super().__init__(config)
            self.ff_kind = spec["kind"]
            self.ff_scale_count = len(spec["scales"])
            self.ff_banks = [
                self.add_weight(
                    name="ff_bank_" + str(i),
                    shape=b.shape,
                    initializer=tf.keras.initializers.Constant(b),
                    trainable=False,
                )
                for i, b in enumerate(ff_banks(spec, seed))
            ]

        def call(self, X, training=False):
            z = 2.0 * (tf.cast(X, tf.float32) - self.lb) / (self.ub - self.lb) - 1.0

            def trunk(h):
                for layer in self.hidden_layers:
                    h = layer(h, training=training)
                return h

            def encoded(local, padded, bank):
                phase = 2.0 * np.pi * tf.matmul(local, bank)
                return trunk(tf.concat([padded, tf.sin(phase), tf.cos(phase)], axis=1))

            if self.ff_kind == "spacetime":
                n = self.ff_scale_count
                space, temporal = (z[:, :2], z[:, 2:3])
                sx = tf.concat([space, tf.zeros_like(temporal)], 1)
                tx = tf.concat([tf.zeros_like(space), temporal], 1)
                hs = [encoded(space, sx, b) for b in self.ff_banks[:n]]
                ht = [encoded(temporal, tx, b) for b in self.ff_banks[n:]]
                hidden = tf.concat([s * t for s in hs for t in ht], 1)
            else:
                hidden = tf.concat([encoded(z, z, b) for b in self.ff_banks], 1)
            return self.output_layer(hidden, training=training)

    "Inference/coordinate-derivative audit only; no optimizer is constructed."
    import copy
    import tempfile
    import time
    import tensorflow as tf

    def audit_model_digest(model):
        return audit_digest_arrays({str(i): v.numpy() for i, v in enumerate(model.variables)})

    def audit_load_model(config, contract, arm, seed, archive, member, scratch):
        spec = contract["arms"][arm]
        cfg = copy.deepcopy(config)
        cfg["model"]["layers"] = [3] + [spec["width"]] * spec["depth"] + [1]
        if spec["kind"] == "plain":
            model = WavePINN2DComparison(cfg, use_fourier=False, fourier_seed=seed)
        elif spec["kind"] == "spacetime":
            model = WaveFFResearch(cfg, spec, seed)
        else:
            raise ValueError("Unsupported architecture")
        model(tf.zeros((1, 3), tf.float32), training=False)
        path = Path(scratch) / "audit_load.weights.h5"
        path.write_bytes(archive.read(member))
        model.load_weights(path)
        if hasattr(model, "ff_banks"):
            for stored, expected in zip(model.ff_banks, ff_banks(spec, seed)):
                np.testing.assert_array_equal(stored.numpy(), expected)
        return model

    def audit_derivative_eager(model, points):
        x = tf.convert_to_tensor(points, dtype=tf.float32)
        with tf.GradientTape(persistent=True, watch_accessed_variables=False) as second:
            second.watch(x)
            with tf.GradientTape(watch_accessed_variables=False) as first:
                first.watch(x)
                u = model(x, training=False)
            g = first.gradient(u, x)
            gx, gy, gt = (g[:, 0:1], g[:, 1:2], g[:, 2:3])
        xx = second.gradient(gx, x)[:, 0:1]
        yy = second.gradient(gy, x)[:, 1:2]
        tt = second.gradient(gt, x)[:, 2:3]
        del second
        return tf.concat([u, gt, tt, xx + yy], axis=1)

    def audit_evaluator(model):

        @tf.function(input_signature=[tf.TensorSpec([None, 3], tf.float32)], autograph=False)
        def derivative(points):
            return audit_derivative_eager(model, points)

        return derivative

    def audit_evaluate(derivative, coordinates):
        chunks = []
        for start in range(0, len(coordinates), AUDIT_POLICY["batch_size"]):
            result = derivative(
                tf.convert_to_tensor(
                    coordinates[start : start + AUDIT_POLICY["batch_size"]], tf.float32
                )
            ).numpy()
            if not np.isfinite(result).all():
                raise RuntimeError("Nonfinite derivative output")
            chunks.append(result.astype(np.float64))
        return np.concatenate(chunks)

    def audit_predict(model, coordinates):
        return np.concatenate(
            [
                model(
                    tf.convert_to_tensor(
                        coordinates[i : i + AUDIT_POLICY["batch_size"]], tf.float32
                    ),
                    training=False,
                )
                .numpy()
                .ravel()
                .astype(np.float64)
                for i in range(0, len(coordinates), AUDIT_POLICY["batch_size"])
            ]
        )

    def audit_assert_close(actual, expected, label):
        if not np.allclose(actual, expected, rtol=0.001, atol=1e-06):
            raise RuntimeError("Reload/numerical guard failed: " + label)

    def audit_exact_probe(config, points, scales):
        e = config["equation"]
        omega = scales["omega"]

        class AnalyticWave(tf.keras.Model):

            def call(self, x, training=False):
                return (
                    tf.sin(np.pi * e["kx"] * x[:, 0:1])
                    * tf.sin(np.pi * e["ky"] * x[:, 1:2])
                    * tf.cos(omega * x[:, 2:3])
                )

        model = AnalyticWave()
        x = points["lhs"][:257]
        out = audit_derivative_eager(model, x).numpy().astype(np.float64)
        exact = audit_exact(x, config)
        expected = np.column_stack([exact[k] for k in ("u", "ut", "utt", "lap")])
        np.testing.assert_allclose(out, expected, rtol=0.0002, atol=0.0001)
        residual = out[:, 2] - e["lambda_target"] * e["wave_speed"] ** 2 * out[:, 3]
        normalized_max = float(np.max(np.abs(residual)) / scales["residual_rms_scale"])
        if normalized_max > 0.0001:
            raise RuntimeError("Exact derivative scale probe failed")
        return {
            "passed": True,
            "normalized_max_residual": normalized_max,
            "points": len(x),
            "model_training": False,
            "scope": "Analytical function derivative test, no learned weights",
        }

    def audit_reload_check(model, derivative, expected, arrays, config, scales):
        reference = audit_field_stats(
            audit_predict(model, arrays["X_exact"]), arrays["u_exact"], scales["field_rms"]
        )
        for key in ("relative_l2", "mse"):
            audit_assert_close(
                reference[key], expected["reference_metrics"][key], "reference " + key
            )
        audit_assert_close(float(model.lambda_1.numpy()), expected["lambda_1"], "lambda")
        physical = {}
        for split in ("val", "test"):
            out = audit_evaluate(derivative, arrays["X_collocation_" + split])
            r = (
                out[:, 2]
                - float(model.lambda_1.numpy()) * config["equation"]["wave_speed"] ** 2 * out[:, 3]
            )
            physical[split] = float(np.mean(r * r))
            audit_assert_close(
                physical[split],
                expected["physical_losses"][split]["physics"],
                split + " physical residual",
            )
        return {
            "passed": True,
            "reference_metrics": reference,
            "physical_pde_mse": physical,
            "rtol": 0.001,
            "atol": 1e-06,
            "cpu_runtime": not bool(tf.config.list_physical_devices("GPU")),
        }

    def audit_volume(x, out, config, coefficient, scales):
        exact = audit_exact(x, config)["u"]
        c2 = config["equation"]["wave_speed"] ** 2
        residuals = {
            "learned": out[:, 2] - coefficient * c2 * out[:, 3],
            "true": out[:, 2] - config["equation"]["lambda_target"] * c2 * out[:, 3],
        }
        result = {
            "point_count": len(x),
            "field": audit_field_stats(out[:, 0], exact, scales["field_rms"]),
            "residual": {
                k: audit_residual_stats(v, scales["residual_rms_scale"])
                for k, v in residuals.items()
            },
        }
        result["worst_points"] = {}
        for name, values in (
            ("field_error", out[:, 0] - exact),
            ("learned_residual", residuals["learned"]),
            ("true_residual", residuals["true"]),
        ):
            indices = np.argsort(np.abs(values))[-20:][::-1]
            result["worst_points"][name] = [
                {"xyz_t": x[i].astype(float).tolist(), "value": float(values[i])} for i in indices
            ]
        result["sectors_equal_volume_3x3x3"] = audit_regional(
            x, out[:, 0], exact, residuals["true"], config, scales
        )
        times = np.unique(x[:, 2])
        if len(times) <= 64:
            result["time_slices"] = []
            for t in times:
                mask = x[:, 2] == t
                result["time_slices"].append(
                    {
                        "time": float(t),
                        "field": audit_field_stats(out[mask, 0], exact[mask], scales["field_rms"]),
                        "residual_true": audit_residual_stats(
                            residuals["true"][mask], scales["residual_rms_scale"]
                        ),
                    }
                )
        return result

    def audit_stage(model, config, points, scales, expected, source_arrays):
        before = audit_model_digest(model)
        derivative = audit_evaluator(model)
        probe_x = points["lhs"][:19]
        probe = audit_evaluate(derivative, probe_x)
        split_probe = np.concatenate(
            [audit_evaluate(derivative, probe_x[:7]), audit_evaluate(derivative, probe_x[7:])]
        )
        np.testing.assert_allclose(probe, split_probe, rtol=0.0002, atol=0.0002)
        builtin = (
            model.pde_residual(tf.convert_to_tensor(probe_x, tf.float32))
            .numpy()
            .ravel()
            .astype(float)
        )
        independent = (
            probe[:, 2]
            - float(model.lambda_1.numpy()) * config["equation"]["wave_speed"] ** 2 * probe[:, 3]
        )
        np.testing.assert_allclose(independent, builtin, rtol=0.001, atol=0.0001)
        reload = audit_reload_check(model, derivative, expected, source_arrays, config, scales)
        coefficient = float(model.lambda_1.numpy())
        results = {}
        for name in ("grid", "lhs"):
            out = audit_evaluate(derivative, points[name])
            results[name] = audit_volume(points[name], out, config, coefficient, scales)
        constraints = {}
        for name in ("x_min", "x_max", "y_min", "y_max", "initial"):
            out = audit_evaluate(derivative, points[name])
            exact = audit_exact(points[name], config)
            scale = scales["initial_displacement_rms"] if name == "initial" else scales["field_rms"]
            error = np.abs(out[:, 0] - exact["u"]) / scale
            constraints[name] = {
                "normalized_rms": float(np.sqrt(np.mean(error**2))),
                "p99_absolute_normalized": float(np.quantile(error, 0.99)),
                "maximum_absolute_normalized": float(error.max()),
                "scale": scale,
                "acceptance": "PENDING",
            }
            if name == "initial":
                velocity = np.abs(out[:, 1] - exact["ut"]) / scales["velocity_rms"]
                constraints["initial_velocity"] = {
                    "normalized_rms": float(np.sqrt(np.mean(velocity**2))),
                    "p99_absolute_normalized": float(np.quantile(velocity, 0.99)),
                    "maximum_absolute_normalized": float(velocity.max()),
                    "scale": scales["velocity_rms"],
                    "zero_target_relative_error_not_used": True,
                    "acceptance": "PENDING",
                }
        if before != audit_model_digest(model):
            raise RuntimeError("Evaluation changed model variables")
        result = {
            **results,
            "constraints": constraints,
            "lambda": coefficient,
            "lambda_relative_error": abs(coefficient - config["equation"]["lambda_target"])
            / abs(config["equation"]["lambda_target"]),
            "model_state_unchanged": True,
            "reload_check": reload,
            "independent_derivatives_match": True,
            "batch_invariance_passed": True,
            "model_variables_sha256": before,
        }
        result["pde_grade"] = audit_overall_pde(result)
        return result

    def audit_make_score(adam, selected):
        field = audit_score(
            adam["grid"]["field"]["relative_l2"], selected["grid"]["field"]["relative_l2"]
        )
        coefficient = audit_score(adam["lambda_relative_error"], selected["lambda_relative_error"])
        return {
            "field": field,
            "lambda": coefficient,
            "total": field["points"] + coefficient["points"],
            "maximum": 6,
            "pde_grade": selected["pde_grade"],
            "overall_physics_status": "PENDING",
            "industrial_qualification": False,
        }

    def audit_summary(rows):
        aggregate = {}
        for arm in ARMS:
            group = [r for r in rows if r["arm"] == arm]
            if not group:
                continue
            total = [r["score"]["total"] for r in group]
            aggregate[arm] = {
                "seeds": len(group),
                "score_mean": float(np.mean(total)),
                "score_min": min(total),
                "score_max": max(total),
                "selected_field_l2_mean": float(
                    np.mean([r["selected"]["grid"]["field"]["relative_l2"] for r in group])
                ),
                "selected_lambda_error_mean": float(
                    np.mean([r["selected"]["lambda_relative_error"] for r in group])
                ),
                "pde_grade_counts": {
                    g: sum((r["score"]["pde_grade"] == g for r in group))
                    for g in ("STRICT", "BASIC", "NOT_MET")
                },
                "both_quantities_within_5_percent": sum(
                    (
                        r["selected"]["grid"]["field"]["relative_l2"] <= 0.05
                        and r["selected"]["lambda_relative_error"] <= 0.05
                        for r in group
                    )
                ),
            }
        return {
            "status": "evaluated_pending_integrity" if len(rows) == 15 else "in_progress",
            "completed_trials": len(rows),
            "training_executed": False,
            "optimizer_updates": 0,
            "lbfgs_evaluations": 0,
            "aggregate": aggregate,
            "overall_physics_status": "PENDING",
            "cpu_qualification": "Not a continuation/same-next-update CPU qualification",
            "automatic_promotion": False,
        }

    def audit_markdown(summary):
        lines = [
            "# Wave common-scale physics audit",
            "",
            "Post-hoc research rubric. No training, model selection, or automatic promotion.",
            "",
            "All error entries below are percentages. PDE grade is not overall physics approval.",
            "",
            "| Method | Mean /6 | Minimum | Field L2 % | Lambda error % | PDE strict/basic/not-met |",
            "|---|---:|---:|---:|---:|---|",
        ]
        for arm, r in summary["aggregate"].items():
            grades = "/".join(
                (str(r["pde_grade_counts"][k]) for k in ("STRICT", "BASIC", "NOT_MET"))
            )
            lines.append(
                f"| {arm} | {r['score_mean']:.3f} | {r['score_min']} | {100 * r['selected_field_l2_mean']:.4f} | {100 * r['selected_lambda_error_mean']:.4f} | {grades} |"
            )
        lines += [
            "",
            "BC/IC acceptance and worst-point assessment remain PENDING.",
            "Finite diagnostic points do not provide a domain-wide error certificate.",
            "Adam and selected stage reports, both residual coefficient views, regional/time metrics and score evidence are in trials/*.json.",
        ]
        return "\n".join(lines) + "\n"

    def audit_run(source, parent, output_root, output_zip):
        start = time.perf_counter()
        hashes = {"source": audit_sha(source), "parent": audit_sha(parent)}
        config, source_arrays, contract, reports = audit_inputs(source, parent)
        if tf.__version__.split(".")[:2] != contract["tensorflow"].split(".")[:2]:
            raise RuntimeError(
                "Use TensorFlow "
                + contract["tensorflow"]
                + " for this audit; current "
                + tf.__version__
            )
        scales, points = (audit_scales(config), audit_points(config))
        exact_error = np.max(
            np.abs(
                audit_exact(source_arrays["X_exact"], config)["u"]
                - source_arrays["u_exact"].ravel()
            )
        )
        if exact_error > 1e-05:
            raise RuntimeError("Analytic solution does not match source labels")
        output_root = Path(output_root)
        output_root.mkdir(parents=True, exist_ok=False)
        (output_root / "trials").mkdir()
        audit_json(output_root / "policy.json", AUDIT_POLICY)
        audit_json(output_root / "normalization.json", scales)
        np.savez_compressed(output_root / "diagnostic_points.npz", **points)
        audit_json(
            output_root / "provenance.json",
            {
                "input_sha256": hashes,
                "source_filename": Path(source).name,
                "parent_filename": Path(parent).name,
                "parent_training_code_sha256": contract["code_sha256"],
                "audit_code_sha256": AUDIT_CODE_SHA256,
                "diagnostic_points_sha256": audit_digest_arrays(points),
                "tensorflow": tf.__version__,
                "numpy": np.__version__,
                "devices": [d.name for d in tf.config.list_physical_devices()],
                "source_label_max_difference": float(exact_error),
                "thresholds_post_hoc": True,
                "equation": config["equation"],
                "domain": config["domain"],
            },
        )
        audit_json(output_root / "wiring.json", audit_exact_probe(config, points, scales))
        print(
            "Normalization PASS; common R0 =",
            format(scales["residual_rms_scale"], ".10g"),
            flush=True,
        )
        rows = []
        with zipfile.ZipFile(parent) as archive, tempfile.TemporaryDirectory() as scratch:
            for index, report in enumerate(reports, 1):
                seed, arm = (report["seed"], report["arm"])
                folder = f"trials/seed_{seed}_{arm}"
                stages = {}
                for stage, member, expected in (
                    ("adam", folder + "/last.weights.h5", report["adam"]["last"]),
                    (
                        "selected",
                        "postprocess/" + folder + "/selected.weights.h5",
                        report["post"]["selected"],
                    ),
                ):
                    model = audit_load_model(config, contract, arm, seed, archive, member, scratch)
                    stages[stage] = audit_stage(
                        model, config, points, scales, expected, source_arrays
                    )
                    del model
                    tf.keras.backend.clear_session()
                row = {
                    "arm": arm,
                    "seed": seed,
                    "adam_iteration": report["adam"]["last"]["iteration"],
                    "lbfgs_accepted_iterations": report["post"]["solver"]["accepted_iterations"],
                    "lbfgs_objective_evaluations": report["post"]["solver"][
                        "objective_evaluations"
                    ],
                    **stages,
                    "score": audit_make_score(stages["adam"], stages["selected"]),
                }
                rows.append(row)
                audit_json(output_root / "trials" / f"seed_{seed}_{arm}.json", row)
                summary = audit_summary(rows)
                audit_json(output_root / "summary.json", summary)
                (output_root / "REPORT.md").write_text(audit_markdown(summary), encoding="utf-8")
                audit_pack(output_root, output_zip)
                print(
                    f"[{index}/15] {seed} {arm}: score {row['score']['total']}/6; PDE {row['score']['pde_grade']}",
                    flush=True,
                )
        if hashes != {"source": audit_sha(source), "parent": audit_sha(parent)}:
            raise RuntimeError("Input ZIP changed during audit")
        summary.update(
            status="complete",
            input_zips_unchanged=True,
            elapsed_seconds=time.perf_counter() - start,
            completed_model_stages=30,
        )
        audit_json(output_root / "summary.json", summary)
        audit_pack(output_root, output_zip)
        print(
            "Complete: 15 trials / 30 saved model stages. Submit ZIP only:", Path(output_zip).name
        )
        return summary

    return locals()
