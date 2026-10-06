"""Lazy Wave training runtime with preserved historical helper definitions.

Section comments identify earlier diagnostic protocols; they do not describe the
current user-facing workflow. The distribution adapter selects the active
CrossLR, periodic-LHS and loss-term-SA policy. No input ZIP code is executed.
"""


def load_wave_training(BUNDLE, settings=None):
    """Load the Wave helpers, source arrays and explicitly supplied settings.

    Calling this factory loads TensorFlow and prepares the runtime. Optimizer
    updates require a subsequent, explicitly authorized trial call.
    """
    REQUIRES_GPU = bool(__import__("tensorflow").config.list_physical_devices("GPU"))
    import tensorflow as tf
    import numpy as np

    if bool(tf.config.list_physical_devices("GPU")) != REQUIRES_GPU:
        raise RuntimeError("TensorFlow runtime mismatch")
    tf.keras.utils.set_random_seed(3234)
    tf.config.experimental.enable_op_determinism()
    DEFINITION_SHA256 = "c0a95fe0d927c7303b6b964777a48e9611cca23fde71ec5097761b6f0d58ec11"
    RUNTIME_SHA256 = "ce0a383ddcb3ae8ee374c697e30e92cc315044d0338b03d07a33c97ad6d0581c"

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

    def calculate_losses_2d(model2d, data2d, training=False):
        initial_coordinates = tf.cast(data2d["X_initial"], tf.float32)
        with tf.GradientTape(watch_accessed_variables=False) as initial_tape:
            initial_tape.watch(initial_coordinates)
            initial_prediction = model2d(initial_coordinates, training=training)
        initial_gradient = initial_tape.gradient(initial_prediction, initial_coordinates)
        initial_velocity_prediction = initial_gradient[:, 2:3]
        loss_initial_displacement = tf.reduce_mean(
            tf.square(data2d["u_initial"] - initial_prediction)
        )
        loss_initial_velocity = tf.reduce_mean(
            tf.square(data2d["ut_initial"] - initial_velocity_prediction)
        )
        loss_initial = 0.5 * (loss_initial_displacement + loss_initial_velocity)
        boundary_prediction = model2d(data2d["X_boundary"], training=training)
        loss_boundary = tf.reduce_mean(tf.square(data2d["u_boundary"] - boundary_prediction))
        residual = model2d.pde_residual(data2d["X_collocation"])
        loss_physics = tf.reduce_mean(tf.square(residual))
        supervised_prediction = model2d(data2d["X_supervised"], training=training)
        loss_supervised = tf.reduce_mean(tf.square(data2d["u_supervised"] - supervised_prediction))
        total = (
            wave2d_weights["initial"] * loss_initial
            + wave2d_weights["boundary"] * loss_boundary
            + wave2d_weights["physics"] * loss_physics
            + wave2d_weights["supervised"] * loss_supervised
        )
        return {
            "total": total,
            "initial": loss_initial,
            "initial_displacement": loss_initial_displacement,
            "initial_velocity": loss_initial_velocity,
            "boundary": loss_boundary,
            "physics": loss_physics,
            "supervised": loss_supervised,
        }

    def _fixed_weights():
        raw = {
            name: tf.constant(value, dtype=tf.float32)
            for name, value in wave2d_config["loss_weights"].items()
        }
        total = tf.add_n(list(raw.values()))
        return {name: value / total for name, value in raw.items()}

    def _curriculum_weights(iteration):
        curriculum = method_comparison_cfg["curriculum"]
        progress = tf.clip_by_value(
            tf.cast(iteration, tf.float32) / curriculum["ramp_iterations"], 0.0, 1.0
        )
        physics = curriculum["physics_start"] + progress * (
            curriculum["physics_end"] - curriculum["physics_start"]
        )
        constraint = curriculum["constraint_start"] + progress * (
            curriculum["constraint_end"] - curriculum["constraint_start"]
        )
        raw = {
            "initial": constraint,
            "boundary": constraint,
            "physics": physics,
            "supervised": tf.constant(curriculum["supervised_weight"], dtype=tf.float32),
        }
        total = tf.add_n(list(raw.values()))
        return {name: value / total for name, value in raw.items()}

    def _self_adaptive_physics_loss(residual):
        adaptive_cfg = method_comparison_cfg["self_adaptive"]
        epsilon = tf.constant(adaptive_cfg["epsilon"], dtype=tf.float32)
        squared = tf.square(residual)
        residual_scale = tf.sqrt(squared + epsilon)
        normalized = residual_scale / (tf.reduce_mean(residual_scale) + epsilon)
        clipped = tf.clip_by_value(
            normalized, adaptive_cfg["minimum_weight"], adaptive_cfg["maximum_weight"]
        )
        weights = clipped / (tf.reduce_mean(clipped) + epsilon)
        weights = tf.stop_gradient(weights)
        return (tf.reduce_mean(weights * squared), tf.reduce_min(weights), tf.reduce_max(weights))

    def calculate_method_losses(model, data, method_spec, iteration, training=False):
        initial_coordinates = tf.cast(data["X_initial"], tf.float32)
        with tf.GradientTape(watch_accessed_variables=False) as initial_tape:
            initial_tape.watch(initial_coordinates)
            initial_prediction = model(initial_coordinates, training=training)
        initial_gradient = initial_tape.gradient(initial_prediction, initial_coordinates)
        initial_velocity_prediction = initial_gradient[:, 2:3]
        loss_initial_displacement = tf.reduce_mean(
            tf.square(data["u_initial"] - initial_prediction)
        )
        loss_initial_velocity = tf.reduce_mean(
            tf.square(data["ut_initial"] - initial_velocity_prediction)
        )
        loss_initial = 0.5 * (loss_initial_displacement + loss_initial_velocity)
        boundary_prediction = model(data["X_boundary"], training=training)
        loss_boundary = tf.reduce_mean(tf.square(data["u_boundary"] - boundary_prediction))
        residual = model.pde_residual(data["X_collocation"])
        if method_spec["self_adaptive"]:
            loss_physics, adaptive_min, adaptive_max = _self_adaptive_physics_loss(residual)
        else:
            loss_physics = tf.reduce_mean(tf.square(residual))
            adaptive_min = tf.constant(1.0, dtype=tf.float32)
            adaptive_max = tf.constant(1.0, dtype=tf.float32)
        supervised_prediction = model(data["X_supervised"], training=training)
        loss_supervised = tf.reduce_mean(tf.square(data["u_supervised"] - supervised_prediction))
        weights = _curriculum_weights(iteration) if method_spec["curriculum"] else _fixed_weights()
        total = (
            weights["initial"] * loss_initial
            + weights["boundary"] * loss_boundary
            + weights["physics"] * loss_physics
            + weights["supervised"] * loss_supervised
        )
        return {
            "total": total,
            "initial": loss_initial,
            "initial_displacement": loss_initial_displacement,
            "initial_velocity": loss_initial_velocity,
            "boundary": loss_boundary,
            "physics": loss_physics,
            "supervised": loss_supervised,
            "adaptive_weight_min": adaptive_min,
            "adaptive_weight_max": adaptive_max,
        }

    METHOD_SPECS = {
        "supervised_only": {
            "supervised_only": True,
            "fourier": False,
            "self_adaptive": False,
            "curriculum": False,
        },
        "baseline": {
            "supervised_only": False,
            "fourier": False,
            "self_adaptive": False,
            "curriculum": False,
        },
        "fourier": {
            "supervised_only": False,
            "fourier": True,
            "self_adaptive": False,
            "curriculum": False,
        },
        "self_adaptive": {
            "supervised_only": False,
            "fourier": False,
            "self_adaptive": True,
            "curriculum": False,
        },
        "fourier_curriculum": {
            "supervised_only": False,
            "fourier": True,
            "self_adaptive": False,
            "curriculum": True,
        },
        "fourier_self_adaptive": {
            "supervised_only": False,
            "fourier": True,
            "self_adaptive": True,
            "curriculum": False,
        },
    }
    # Historical Stage 7 diagnostics. Input archives contain data and weights;
    # their Python source is never executed by this runtime.
    from pathlib import Path, PurePosixPath
    from datetime import datetime, timezone
    import copy
    import csv
    import hashlib
    import io
    import json
    import os
    import time
    import uuid
    import zipfile
    import numpy as np

    SUPPORTED_EXTENSION_RUNTIME = "f9dd9ff05d8f3e1170e85e4886328b1a723f0d06902b17524634ce143553b758"

    def sha256(path):
        digest = hashlib.sha256()
        with Path(path).open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def write_json(path, payload):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)

    def safe_relative(name):
        path = PurePosixPath(name)
        if not name or "\\" in name or ":" in name or path.is_absolute() or (".." in path.parts):
            raise ValueError(f"Unsafe artifact path: {name}")
        return path

    def comparison_digest(config):
        payload = {
            key: config[key]
            for key in ("equation", "domain", "model", "data", "loss_weights", "training")
        }
        payload["comparison"] = config["method_comparison"]
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()

    def load_stage7_bundle(bundle, destination):
        """Validate then extract an allowlist of data, metadata and weights only."""
        bundle, destination = (Path(bundle), Path(destination))
        if destination.exists():
            raise FileExistsError("A fresh extraction directory is required")
        with zipfile.ZipFile(bundle) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)):
                raise ValueError("Duplicate ZIP members")
            for entry in archive.infolist():
                safe_relative(entry.filename)
                if entry.external_attr >> 16 & 61440 == 40960:
                    raise ValueError("ZIP symlinks are not supported")
            if sum((i.file_size for i in archive.infolist())) > 512 * 1024 * 1024:
                raise ValueError("Diagnostic input exceeds the 512 MiB uncompressed limit")
            if archive.testzip() is not None:
                raise ValueError("ZIP CRC verification failed")
            config = json.loads(archive.read("config/wave2d_config.json"))
            results = json.loads(archive.read("report_artifacts/wave2d_method_comparison.json"))
            receipt = json.loads(archive.read("workflow/stage_07_validation.json"))
            if (
                receipt.get("stage") != 7
                or receipt.get("status") != "complete"
                or receipt.get("reasons")
            ):
                raise ValueError("Stage 7 completion receipt is invalid")
            required_cells = receipt.get("required_cells", [])
            if not required_cells or any(
                (receipt.get("cell_status", {}).get(k) != "passed" for k in required_cells)
            ):
                raise ValueError("Stage 7 required cells are incomplete")
            if results.get("status") != "complete" or results.get("protocol_version") != 2:
                raise ValueError("A completed protocol-2 Stage 7 pilot is required")
            if config.get("execution_profile") != "pilot" or config["method_comparison"][
                "seeds"
            ] != [3234]:
                raise ValueError("Expected the one-seed Stage 7 pilot (3234)")
            if results["config_fingerprint"] != comparison_digest(config):
                raise ValueError("Comparison configuration fingerprint mismatch")
            rows = results["results"]
            expected = {(m, 3234) for m in config["method_comparison"]["methods"]}
            if len(rows) != len(expected) or {(r["method"], r["seed"]) for r in rows} != expected:
                raise ValueError("Missing or duplicate pilot trials")
            required = {
                "config/wave2d_config.json",
                "training/wave2d_data.npz",
                "report_artifacts/wave2d_method_comparison.json",
                "workflow/stage_07_validation.json",
            }
            for row in rows:
                for key in (
                    "best",
                    "last",
                    "production_candidate",
                    "history",
                    "convergence_targets",
                ):
                    required.add(row["weights"][key])
                required.update((row["initialization"][key] for key in ("weights", "manifest")))
            for name in required:
                safe_relative(name)
                if archive.getinfo(name).file_size == 0:
                    raise ValueError(f"Empty artifact: {name}")
                if Path(name).suffix not in {".json", ".npz", ".h5", ".csv"}:
                    raise ValueError(f"Unsupported artifact: {name}")
            for row in rows:
                initial = row["initialization"]
                digest = hashlib.sha256(archive.read(initial["weights"])).hexdigest()
                manifest = json.loads(archive.read(initial["manifest"]))
                if digest != initial["sha256"] or digest != manifest["sha256"]:
                    raise ValueError("Initialization checksum mismatch")
                if manifest["seed"] != row["seed"] or manifest["group"] != initial["group"]:
                    raise ValueError("Initialization identity mismatch")
            destination.mkdir(parents=True)
            for name in sorted(required):
                target = destination.joinpath(*safe_relative(name).parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(name))
        return (
            config,
            results,
            {
                "bundle_sha256": sha256(bundle),
                "stage7_receipt": receipt,
                "files": {n: sha256(destination / n) for n in sorted(required)},
            },
        )

    def exact_numpy(x, config):
        e = config["equation"]
        omega = np.pi * e["wave_speed"] * np.sqrt(e["kx"] ** 2 + e["ky"] ** 2)
        return (
            np.sin(np.pi * e["kx"] * x[:, :1])
            * np.sin(np.pi * e["ky"] * x[:, 1:2])
            * np.cos(omega * x[:, 2:3])
        )

    def data_audit(arrays, config):
        checks, details = ({}, {})
        d = config["domain"]
        lo = np.array([d[k + "_min"] for k in ("x", "y", "t")])
        hi = np.array([d[k + "_max"] for k in ("x", "y", "t")])
        for name, value in arrays.items():
            checks[name + ":finite"] = bool(np.isfinite(value).all())
            if name.startswith("X_"):
                checks[name + ":shape_domain"] = bool(
                    value.ndim == 2
                    and value.shape[1] == 3
                    and np.all(value >= lo - 1e-06)
                    and np.all(value <= hi + 1e-06)
                )
                label = "u_" + name[2:]
                if label in arrays:
                    checks[label + ":shape"] = arrays[label].shape == (len(value), 1)
                    error = float(
                        np.max(
                            np.abs(exact_numpy(value.astype(np.float64), config) - arrays[label])
                        )
                    )
                    details[label + ":max_label_error"] = error
                    checks[label + ":exact_label"] = error < 1e-05
                if name.startswith("X_initial"):
                    checks[name + ":t0"] = bool(np.allclose(value[:, 2], 0, atol=1e-07))
                    checks[name + ":ut0"] = bool(
                        np.allclose(arrays["ut_" + name[2:]], 0, atol=1e-07)
                    )
                if name.startswith("X_boundary"):
                    on_face = np.any(
                        np.isclose(value[:, :2], lo[:2], atol=1e-06)
                        | np.isclose(value[:, :2], hi[:2], atol=1e-06),
                        axis=1,
                    )
                    checks[name + ":boundary_face"] = bool(on_face.all())
        for family in ("supervised", "collocation", "initial", "boundary"):
            sets = []
            for split in ("train", "val", "test"):
                a = arrays[f"X_{family}_{split}"]
                checks[f"{family}_{split}:count"] = len(a) == config["data"][f"{family}_{split}"]
                sets.append({tuple(row) for row in a.tolist()})
            counts = [len(sets[i] & sets[j]) for i, j in ((0, 1), (0, 2), (1, 2))]
            details[family + ":split_overlap_counts"] = counts
            checks[family + ":split_disjoint"] = not any(counts)
        checks["full_grid_count"] = len(arrays["X_exact"]) == config["data"]["full_grid_total"]
        return {"checks": checks, "details": details, "passed": all(checks.values())}

    def snapshot(run_dir, backup_dir, status):
        """Diagnostic-only ZIP namespace; excluded from the V3 stage restore glob."""
        run_dir, backup_dir = (Path(run_dir), Path(backup_dir))
        backup_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        target = backup_dir / f"wave2d_diagnostic_{run_dir.name}_{status}_{stamp}.zip"
        temporary = target.with_suffix(".zip.tmp")
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(run_dir.rglob("*")):
                if path.is_symlink():
                    raise ValueError("Diagnostic output symlinks are not supported")
                if path.is_file() and (
                    not any((part.endswith(".tmp") for part in path.relative_to(run_dir).parts))
                ):
                    archive.write(path, path.relative_to(run_dir).as_posix())
        with zipfile.ZipFile(temporary) as archive:
            if archive.testzip() is not None:
                raise RuntimeError("Diagnostic backup CRC failed")
        temporary.replace(target)
        print(f"Diagnostic backup: {target}", flush=True)
        return str(target)

    def arrays_to_tensors(arrays):
        return {
            split: {
                key: tf.convert_to_tensor(arrays[key + "_" + split], tf.float32)
                for key in (
                    "X_initial",
                    "u_initial",
                    "ut_initial",
                    "X_boundary",
                    "u_boundary",
                    "X_collocation",
                    "X_supervised",
                    "u_supervised",
                )
            }
            for split in ("train", "val", "test")
        }

    def field_metrics(model, arrays):
        prediction = model(tf.constant(arrays["X_exact"], tf.float32), training=False).numpy()
        exact = arrays["u_exact"].astype(np.float64)
        delta = prediction.astype(np.float64) - exact
        return {
            "relative_l1": float(np.sum(np.abs(delta)) / np.sum(np.abs(exact))),
            "relative_l2": float(np.linalg.norm(delta) / np.linalg.norm(exact)),
            "mse": float(np.mean(delta**2)),
        }

    def independent_residual(model, coordinates):
        """Vector-coordinate nested tapes, independent of the model's sliced tapes."""
        x = tf.convert_to_tensor(coordinates, tf.float32)
        with tf.GradientTape(persistent=True) as second:
            second.watch(x)
            with tf.GradientTape() as first:
                first.watch(x)
                u = model(x)
            gradient = first.gradient(u, x)
            gx, gy, gt = (gradient[:, 0:1], gradient[:, 1:2], gradient[:, 2:3])
        xx, yy, tt = (
            second.gradient(gx, x)[:, :1],
            second.gradient(gy, x)[:, 1:2],
            second.gradient(gt, x)[:, 2:3],
        )
        del second
        return tt - model.lambda_1 * model.wave_speed_squared * (xx + yy)

    def new_model(config, method, seed, path):
        tf.keras.utils.set_random_seed(seed)
        model = WavePINN2DComparison(config, use_fourier="fourier" in method, fourier_seed=seed)
        model(tf.zeros((1, 3), tf.float32))
        model.load_weights(path)
        return model

    def loss_record(model, tensors):
        return {
            name: float(value.numpy())
            for name, value in calculate_losses_2d(model, tensors).items()
        }

    def residual_statistics(model, points):
        residual = model.pde_residual(tf.constant(points, tf.float32)).numpy()
        return {
            "mean_absolute": float(np.mean(np.abs(residual))),
            "maximum_absolute": float(np.max(np.abs(residual))),
            "mse": float(np.mean(residual**2)),
        }

    def gradient_record(model, tensors, fixed_lambda=False):
        variables = [
            v for v in model.trainable_variables if not (fixed_lambda and v is model.lambda_1)
        ]
        with tf.GradientTape(persistent=True) as tape:
            losses = calculate_losses_2d(model, tensors, training=True)
        result = {}
        for name in ("total", "initial", "boundary", "physics", "supervised"):
            gradients = tape.gradient(losses[name], variables)
            connected = [g for g in gradients if g is not None]
            result[name] = {
                "norm": float(tf.linalg.global_norm(connected).numpy()),
                "connected": len(connected),
                "variables": len(variables),
            }
        del tape
        return result

    def exact_wiring_audit(config, tensors):

        class ExactProbe(WavePINN2D):

            def call(self, x, training=False):
                e = config["equation"]
                w = np.pi * e["wave_speed"] * np.sqrt(e["kx"] ** 2 + e["ky"] ** 2)
                return (
                    tf.sin(np.pi * e["kx"] * x[:, :1])
                    * tf.sin(np.pi * e["ky"] * x[:, 1:2])
                    * tf.cos(w * x[:, 2:3])
                )

        probe = ExactProbe(config)
        probe.lambda_1.assign(config["equation"]["lambda_target"])
        x = tensors["train"]["X_collocation"]
        residual = probe.pde_residual(x).numpy()
        alternate = independent_residual(probe, x).numpy()
        losses = loss_record(probe, tensors["train"])
        checks = {
            "exact_model_residual": bool(np.max(np.abs(residual)) < 0.001),
            "independent_derivatives": bool(
                np.allclose(residual, alternate, atol=0.0001, rtol=0.0001)
            ),
            "exact_BC_IC_labels": all(
                (
                    losses[k] < 1e-08
                    for k in ("initial_displacement", "initial_velocity", "boundary", "supervised")
                )
            ),
        }
        return {
            "checks": checks,
            "losses": losses,
            "residual_max": float(np.max(np.abs(residual))),
            "tolerance_float32": 0.001,
            "passed": all(checks.values()),
        }

    def audit_saved_models(config, payload, arrays, source_dir, run_dir, backup_dir):
        tensors = arrays_to_tensors(arrays)
        report = {
            "data": data_audit(arrays, config),
            "exact_wiring": exact_wiring_audit(config, tensors),
            "trials": [],
            "passed": False,
            "scope": "Artifact consistency and mathematical wiring; not convergence acceptance",
        }
        report["implementation_notes"] = [
            "Self-adaptive weights are residual-derived and stop_gradient is applied; there is no trainable attention optimizer.",
            "Weight clipping precedes mean normalization; final weights need not remain within the pre-normalization clipping bounds.",
            "Fourier matrices are deterministic constants reconstructed from configuration and seed, not H5 variables.",
        ]
        random_points = (
            np.random.default_rng(7301)
            .uniform(
                [config["domain"][k + "_min"] for k in ("x", "y", "t")],
                [config["domain"][k + "_max"] for k in ("x", "y", "t")],
                (1000, 3),
            )
            .astype(np.float32)
        )
        np.savez_compressed(Path(run_dir) / "independent_residual_points.npz", points=random_points)
        for row in payload["results"]:
            print(f"Auditing {row['method']} seed={row['seed']}", flush=True)
            method = row["method"]
            model = new_model(
                config,
                method,
                row["seed"],
                Path(source_dir) / row["weights"]["production_candidate"],
            )
            reloaded = new_model(
                config, method, row["seed"], Path(source_dir) / row["weights"]["best"]
            )
            x = tf.constant(arrays["X_exact"], tf.float32)
            q = tf.constant(random_points[:100], tf.float32)
            metrics = field_metrics(model, arrays)
            checks = {
                "stored_L1": bool(
                    np.isclose(
                        metrics["relative_l1"],
                        row["full_grid_relative_l1"],
                        rtol=0.0002,
                        atol=2e-06,
                    )
                ),
                "stored_L2": bool(
                    np.isclose(
                        metrics["relative_l2"],
                        row["full_grid_relative_l2"],
                        rtol=0.0002,
                        atol=2e-06,
                    )
                ),
                "best_final_prediction": bool(
                    np.allclose(model(x).numpy(), reloaded(x).numpy(), rtol=1e-05, atol=1e-06)
                ),
                "best_final_residual": bool(
                    np.allclose(
                        model.pde_residual(q).numpy(),
                        reloaded.pde_residual(q).numpy(),
                        rtol=0.0001,
                        atol=0.0001,
                    )
                ),
                "independent_derivatives": bool(
                    np.allclose(
                        model.pde_residual(q).numpy(),
                        independent_residual(model, q).numpy(),
                        rtol=0.002,
                        atol=0.002,
                    )
                ),
            }
            fourier = None
            if model.fourier_matrix is not None:
                fourier = hashlib.sha256(model.fourier_matrix.numpy().tobytes()).hexdigest()
                checks["fourier_seed_reconstruction"] = bool(
                    np.array_equal(model.fourier_matrix.numpy(), reloaded.fourier_matrix.numpy())
                )
            history_path = Path(source_dir) / row["weights"]["history"]
            with history_path.open(encoding="utf-8") as stream:
                history = list(csv.DictReader(stream))
            selected = min(history, key=lambda r: float(r["target_score_max_relative_l1_l2"]))
            checks["selected_checkpoint_history"] = bool(
                np.isclose(
                    max(metrics["relative_l1"], metrics["relative_l2"]),
                    float(selected["target_score_max_relative_l1_l2"]),
                    rtol=0.0002,
                    atol=2e-06,
                )
            )
            split_losses = {s: loss_record(model, v) for s, v in tensors.items()}
            checks["stored_test_physics"] = bool(
                np.isclose(
                    split_losses["test"]["physics"],
                    row["test_physics_loss_learned_lambda"],
                    rtol=0.002,
                    atol=2e-05,
                )
            )
            learned = float(model.lambda_1.numpy())
            learned_stats = residual_statistics(model, random_points)
            try:
                model.lambda_1.assign(config["equation"]["lambda_target"])
                target_stats = residual_statistics(model, random_points)
                target_test = loss_record(model, tensors["test"])["physics"]
            finally:
                model.lambda_1.assign(learned)
            checks["stored_target_physics"] = bool(
                np.isclose(
                    target_test, row["test_physics_loss_target_lambda"], rtol=0.002, atol=2e-05
                )
            )
            checks["lambda_restored"] = float(model.lambda_1.numpy()) == learned
            last = new_model(config, method, row["seed"], Path(source_dir) / row["weights"]["last"])
            last_metrics = field_metrics(last, arrays)
            checks["last_checkpoint_history"] = bool(
                np.isclose(
                    last_metrics["relative_l2"],
                    float(history[-1]["full_grid_relative_l2"]),
                    rtol=0.0002,
                    atol=2e-06,
                )
            )
            last_objective = (
                loss_record(last, tensors["train"])["supervised"]
                if method == "supervised_only"
                else float(
                    calculate_method_losses(
                        last,
                        tensors["train"],
                        METHOD_SPECS[method],
                        tf.constant(row["iterations_completed"], tf.int32),
                    )["total"].numpy()
                )
            )
            report["trials"].append(
                {
                    "method": method,
                    "checks": checks,
                    "metrics": metrics,
                    "best_iteration": int(selected["iteration"]),
                    "lambda_1": learned,
                    "split_losses": split_losses,
                    "independent_points_learned_lambda": learned_stats,
                    "independent_points_target_lambda": target_stats,
                    "fourier_matrix_sha256": fourier,
                    "gradient_components": gradient_record(model, tensors["train"]),
                    "last_metrics": last_metrics,
                    "last_train_objective_recomputed": last_objective,
                    "last_train_objective_recorded_pre_update": float(
                        history[-1]["train_objective"]
                    ),
                    "objective_note": "Recorded objective precedes the optimizer update; recomputed objective follows it.",
                }
            )
            write_json(Path(run_dir) / "audit.json", report)
            snapshot(run_dir, backup_dir, "audit_partial")
        report["passed"] = (
            report["data"]["passed"]
            and report["exact_wiring"]["passed"]
            and all((all(r["checks"].values()) for r in report["trials"]))
        )
        write_json(Path(run_dir) / "audit.json", report)
        return report

    def import_forward_extension(bundle, run_dir, contract, config, initialization_sha256):
        """Fork the reviewed 10k forward checkpoint; do not mutate the parent run."""
        destination = Path(run_dir) / "forward_fixed"
        if destination.exists():
            raise FileExistsError("Extension import requires a fresh forward directory")
        with zipfile.ZipFile(bundle) as archive:
            names = archive.namelist()
            if (
                len(names) != len(set(names))
                or sum((i.file_size for i in archive.infolist())) > 512 * 1024 * 1024
            ):
                raise ValueError("Invalid diagnostic ZIP members or size")
            for entry in archive.infolist():
                safe_relative(entry.filename)
                if entry.external_attr >> 16 & 61440 == 40960:
                    raise ValueError("ZIP symlinks are not supported")
            if archive.testzip() is not None:
                raise ValueError("Diagnostic ZIP CRC failed")
            parent = json.loads(archive.read("contract.json"))
            for key in ("source_bundle_sha256", "definition_sha256", "tensorflow", "numpy", "seed"):
                if parent.get(key) != contract.get(key):
                    raise ValueError(f"Extension parent mismatch: {key}")
            if parent.get("runtime_module_sha256") != SUPPORTED_EXTENSION_RUNTIME:
                raise ValueError("Unreviewed parent checkpoint producer")
            for filename, key in (
                ("diagnostic_runtime.py", "runtime_module_sha256"),
                ("reused_definitions.py", "definition_sha256"),
            ):
                if hashlib.sha256(archive.read(filename)).hexdigest() != parent[key]:
                    raise ValueError("Parent embedded code checksum mismatch")
            if (
                parent.get("mode") not in {"pair", "forward"}
                or parent.get("maximum_iterations") != 10000
            ):
                raise ValueError("Expected the original 10,000-iteration diagnostic parent")
            if contract["mode"] != "extend" or not 10000 < contract["maximum_iterations"] <= 30000:
                raise ValueError("Extension total budget must be within 10001..30000")
            if json.loads(archive.read("source_config.json")) != config:
                raise ValueError("Parent PDE/data/model/training configuration mismatch")
            summary = json.loads(archive.read("summary.json"))
            if (
                summary.get("status") != "complete"
                or summary.get("audit_passed") is not True
                or summary.get("training_executed") is not True
            ):
                raise ValueError("Parent diagnostic is incomplete")
            result = json.loads(archive.read("forward_fixed/result.json"))
            if (
                result.get("fixed_lambda") is not True
                or result.get("initialization_sha256") != initialization_sha256
            ):
                raise ValueError("Parent forward identity mismatch")
            version = json.loads(archive.read("forward_fixed/latest.json"))["directory"]
            safe_relative(version)
            state = json.loads(archive.read(f"forward_fixed/{version}/state.json"))
            if (
                state != result["state"]
                or state["iteration"] != 10000
                or state.get("stop_reason") != "diagnostic_cap_reached"
            ):
                raise ValueError(
                    "Expected the completed 10k latest checkpoint, not a selected older model"
                )
            if state.get("target_reached") is not False or [
                r["iteration"] for r in state["history"]
            ] != list(range(0, 10001, 1000)):
                raise ValueError("Parent target/history contract mismatch")
            if any(
                (r["lambda_1"] != config["equation"]["lambda_target"] for r in state["history"])
            ):
                raise ValueError("Parent coefficient was not fixed")
            best_version = state["best_checkpoint"]
            safe_relative(best_version)
            required = {"forward_fixed/latest.json"}
            for directory in {version, best_version}:
                for filename in (
                    "state.index",
                    "state.data-00000-of-00001",
                    "state.json",
                    "model.weights.h5",
                    "history.csv",
                ):
                    required.add(f"forward_fixed/{directory}/{filename}")
            if any((archive.getinfo(n).file_size == 0 for n in required)):
                raise ValueError("Empty parent checkpoint member")
            evidence = {
                "parent_zip_sha256": sha256(bundle),
                "parent_contract": parent,
                "continued_from_iteration": 10000,
                "maximum_total_iterations": contract["maximum_iterations"],
                "parent_forward_result": result,
                "copied_files": {
                    n: hashlib.sha256(archive.read(n)).hexdigest() for n in sorted(required)
                },
            }
            staging = Path(run_dir) / ("forward_import_" + uuid.uuid4().hex + ".tmp")
            staging.mkdir(parents=True)
            for name in required:
                target = staging.joinpath(*PurePosixPath(name).parts[1:])
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(name))
            staging.rename(destination)
            write_json(Path(run_dir) / "extension_provenance.json", evidence)
        return evidence

    SUPPORTED_WARMUP_EXTENSION_RUNTIME = (
        "158989e0e9af93bbf707f67223529a0b515bb746461dfedad2688845f75877d9"
    )

    def import_warmup_extension(bundle, run_dir, contract, config, initialization_sha256):
        """Copy a reviewed latest inverse checkpoint to a new run; never import control training."""
        case = "data_warmup_inverse"
        destination = Path(run_dir) / case
        if destination.exists():
            raise FileExistsError("Warmup extension requires a fresh destination")
        if (
            contract["mode"] != "warmup_extend"
            or not 10000 < contract["maximum_iterations"] <= 30000
        ):
            raise ValueError("Warmup extension total budget must be within 10001..30000")
        with zipfile.ZipFile(bundle) as archive:
            names = archive.namelist()
            if (
                len(names) != len(set(names))
                or sum((i.file_size for i in archive.infolist())) > 512 * 1024 * 1024
            ):
                raise ValueError("Invalid warmup ZIP members or size")
            for entry in archive.infolist():
                safe_relative(entry.filename)
                if entry.external_attr >> 16 & 61440 == 40960:
                    raise ValueError("ZIP symlinks are not supported")
            if archive.testzip() is not None:
                raise ValueError("Warmup parent ZIP CRC failed")
            parent = json.loads(archive.read("contract.json"))
            for key in ("source_bundle_sha256", "definition_sha256", "tensorflow", "numpy", "seed"):
                if parent.get(key) != contract.get(key):
                    raise ValueError("Warmup extension parent mismatch: " + key)
            if parent.get("runtime_module_sha256") != SUPPORTED_WARMUP_EXTENSION_RUNTIME:
                raise ValueError("Unreviewed warmup checkpoint producer")
            for filename, key in (
                ("diagnostic_runtime.py", "runtime_module_sha256"),
                ("reused_definitions.py", "definition_sha256"),
            ):
                if hashlib.sha256(archive.read(filename)).hexdigest() != parent[key]:
                    raise ValueError("Warmup parent embedded code checksum mismatch")
            if parent.get("mode") != "warmup_pair" or parent.get("maximum_iterations") != 10000:
                raise ValueError("Expected a completed 10k warmup_pair parent")
            if json.loads(archive.read("source_config.json")) != config:
                raise ValueError("Warmup parent source configuration mismatch")
            summary = json.loads(archive.read("summary.json"))
            if not (
                summary.get("status") == "complete"
                and summary.get("audit_passed") is True
                and (summary.get("training_executed") is True)
            ):
                raise ValueError("Warmup parent is incomplete")
            results = {k: json.loads(archive.read(k + "/result.json")) for k in WARMUP_CASES}
            if compare_warmup_results(results) != json.loads(
                archive.read("warmup_comparison.json")
            ):
                raise ValueError("Warmup parent comparison report mismatch")
            result = results[case]
            if (
                result["initialization_sha256"] != initialization_sha256
                or result.get("fixed_lambda") is not False
            ):
                raise ValueError("Warmup inverse identity mismatch")
            expected = result["experiment_contract"]
            if json.loads(archive.read(case + "/experiment_contract.json")) != expected:
                raise ValueError("Warmup parent case contract differs from checkpoint result")
            if (
                expected["configuration"] != lr_case_config(config, "inverse_lr_thesis")
                or expected["maximum_iterations"] != 10000
                or expected["seed"] != 3234
                or (expected["warmup_iterations"] != 2000)
            ):
                raise ValueError("Warmup experiment configuration mismatch")
            version = json.loads(archive.read(case + "/latest.json"))["directory"]
            safe_relative(version)
            state = json.loads(archive.read(f"{case}/{version}/state.json"))
            if (
                state != result["state"]
                or state["iteration"] != 10000
                or state.get("complete") is not True
                or (state.get("stop_reason") != "diagnostic_cap_reached")
                or (state.get("target_reached") is not False)
                or (state.get("experiment_contract") != expected)
            ):
                raise ValueError(
                    "Warmup extension requires latest completed 10000, not best weights"
                )
            if (
                state["history"][-1]["phase"] != "joint"
                or state["history"][-1]["optimizer_iterations"] != 10000
                or (not all((all(row["checks"].values()) for row in state["gradient_trace"])))
            ):
                raise ValueError("Warmup parent phase/optimizer/gradient evidence failed")
            best = state["best_checkpoint"]
            safe_relative(best)
            required = {case + "/latest.json", case + "/independent_residual_points.npz"}
            for directory in {version, best}:
                for filename in (
                    "state.index",
                    "state.data-00000-of-00001",
                    "state.json",
                    "model.weights.h5",
                    "history.csv",
                    "gradient_trace.json",
                    "lambda_updates.csv",
                ):
                    required.add(f"{case}/{directory}/{filename}")
            if any((archive.getinfo(n).file_size == 0 for n in required)):
                raise ValueError("Empty warmup checkpoint member")
            evidence = {
                "parent_zip_sha256": sha256(bundle),
                "parent_contract": parent,
                "parent_experiment_contract": expected,
                "parent_result": result,
                "continued_from_iteration": 10000,
                "maximum_total_iterations": contract["maximum_iterations"],
                "latest_directory": version,
                "copied_files": {
                    n: hashlib.sha256(archive.read(n)).hexdigest() for n in sorted(required)
                },
            }
            new_contract = dict(expected, maximum_iterations=contract["maximum_iterations"])
            staging = Path(run_dir) / ("warmup_import_" + uuid.uuid4().hex + ".tmp")
            staging.mkdir(parents=True)
            for name in required:
                target = staging.joinpath(*PurePosixPath(name).parts[1:])
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(name))
            write_json(staging / "experiment_contract.json", new_contract)
            staging.rename(destination)
            write_json(Path(run_dir) / "warmup_extension_provenance.json", evidence)
        return evidence

    def validate_warmup_extension_state(state, expected, evidence, state_path):
        """Permit only a documented cap migration of the immutable imported latest state."""
        if state.get("experiment_contract") == expected:
            return
        parent = evidence["parent_experiment_contract"]
        if (
            state.get("experiment_contract") != parent
            or dict(parent, maximum_iterations=expected["maximum_iterations"]) != expected
            or state["iteration"] != evidence["continued_from_iteration"]
        ):
            raise RuntimeError("Warmup extension checkpoint contract mismatch")
        key = "data_warmup_inverse/" + evidence["latest_directory"] + "/state.json"
        if sha256(state_path) != evidence["copied_files"][key]:
            raise RuntimeError("Imported warmup latest checkpoint metadata changed")

    def summarize_warmup_extension(result, evidence):
        history = result["state"]["history"]
        return {
            "scope": "Continuation of the treatment only; not an equal-budget comparison with the 10k control",
            "continued_from_iteration": evidence["continued_from_iteration"],
            "additional_updates_completed": result["state"]["iteration"]
            - evidence["continued_from_iteration"],
            "parent_last": evidence["parent_result"]["state"]["history"][-1],
            "best_all_phases": next(
                (r for r in history if r["iteration"] == result["state"]["best_iteration"])
            ),
            "best_joint_by_field_error": min(
                (r for r in history if r["phase"] == "joint"),
                key=lambda r: max(r["relative_l1"], r["relative_l2"]),
            ),
            "last": history[-1],
            "stop_reason": result["state"]["stop_reason"],
            "stage8_approved": False,
            "automatic_promotion": False,
        }

    def state_for_budget(state, maximum_iterations, allow_extension=False):
        result = copy.deepcopy(state)
        if result["iteration"] > maximum_iterations:
            raise ValueError("Checkpoint iteration exceeds the requested total budget")
        if (
            allow_extension
            and result["iteration"] < maximum_iterations
            and (result.get("stop_reason") == "diagnostic_cap_reached")
        ):
            if result.get("target_reached") is not False:
                raise ValueError("A target-complete run must not be extended")
            result.update(complete=False, stop_reason="running")
        return result

    def trace_due(iteration, maximum_iterations):
        return iteration == 0 or iteration % 50 == 0 or iteration == maximum_iterations

    LR_CASES = {"inverse_lr_control": 0.001, "inverse_lr_thesis": 0.0005}
    WARMUP_CASES = {"warmup_control": 0, "data_warmup_inverse": 2000}

    def warmup_phase(case, iteration):
        if case not in WARMUP_CASES:
            raise ValueError("Unknown warmup case")
        return (
            "initial"
            if iteration == 0
            else "warmup" if iteration <= WARMUP_CASES[case] else "joint"
        )

    def warmup_objective(losses, weights):
        """Original normalized weights, with only the physics contribution omitted."""
        return sum((losses[k] * weights[k] for k in ("initial", "boundary", "supervised")))

    def warmup_observation_due(iteration, maximum_iterations, interval):
        return (
            lr_observation_due(iteration, maximum_iterations, interval)
            or (2000 <= iteration <= 3000 and iteration % 50 == 0)
            or iteration == 2001
        )

    def warmup_target_allowed(case, iteration):
        return iteration > WARMUP_CASES[case]

    def compare_warmup_results(results):
        if set(results) != set(WARMUP_CASES):
            raise ValueError("Both warmup comparison cases are required")
        control, treatment = (results[k] for k in WARMUP_CASES)
        a, b = (r["experiment_contract"] for r in (control, treatment))
        for key in (
            "configuration",
            "seed",
            "maximum_iterations",
            "initialization_sha256",
            "observation_protocol",
            "optimizer_transition",
        ):
            if a[key] != b[key]:
                raise ValueError("Warmup comparison contract mismatch: " + key)
        if a["configuration"]["training"]["learning_rate"] != 0.0005:
            raise ValueError("Warmup comparison requires learning rate 0.0005")
        for case, r in results.items():
            contract, state = (r["experiment_contract"], r["state"])
            if (
                contract["case"] != case
                or contract["warmup_iterations"] != WARMUP_CASES[case]
                or r["initialization_sha256"] != a["initialization_sha256"]
                or (not state["complete"])
            ):
                raise ValueError("Incomplete or incompatible warmup case")
            for row in state["history"]:
                if row["phase"] != warmup_phase(case, row["iteration"]) or (
                    row["phase"] in {"warmup", "initial"} and row["lambda_1"] != 0.5
                ):
                    raise ValueError("Warmup phase or frozen coefficient invariant failed")
            for update in state["update_trace"]:
                if update["phase"] != warmup_phase(case, update["iteration"]):
                    raise ValueError("Warmup update phase mismatch")
                if update["phase"] == "warmup" and any(
                    (
                        update[k] != expected
                        for k, expected in (
                            ("lambda_before", 0.5),
                            ("lambda_after", 0.5),
                            ("lambda_delta", 0.0),
                            ("total_lambda_gradient_pre_update", 0.0),
                        )
                    )
                ):
                    raise ValueError("Warmup coefficient was updated before release")
        histories = [
            {row["iteration"]: row for row in r["state"]["history"]} for r in (control, treatment)
        ]
        if not all((0 in h for h in histories)) or not all(
            (
                np.isclose(histories[0][0][k], histories[1][0][k], rtol=0.0002, atol=2e-06)
                for k in ("relative_l1", "relative_l2", "lambda_1")
            )
        ):
            raise ValueError("Warmup initial fields do not match")
        shared = []
        for i in sorted(histories[0].keys() & histories[1].keys()):
            row = {"iteration": i}
            for label, history in zip(("control", "warmup"), histories):
                row.update(
                    {
                        label + "_" + k: history[i][k]
                        for k in (
                            "phase",
                            "relative_l1",
                            "relative_l2",
                            "lambda_1",
                            "lambda_absolute_error",
                            "independent_residual_mse",
                        )
                    }
                )
            row["delta_relative_l2_warmup_minus_control"] = (
                row["warmup_relative_l2"] - row["control_relative_l2"]
            )
            shared.append(row)
        return {
            "scope": "Single-seed warmup protocol diagnostic; equal total-update caps, unequal physics-update counts",
            "automatic_promotion": False,
            "stage8_approved": False,
            "shared_observations": shared,
            "last_shared_iteration": shared[-1]["iteration"],
            "cases": {
                case: {
                    "initial_learning_rate": 0.0005,
                    "warmup_iterations": WARMUP_CASES[case],
                    "joint_updates_completed": max(0, r["state"]["iteration"] - WARMUP_CASES[case]),
                    "best_iteration": r["state"]["best_iteration"],
                    "best_metrics": r["best_metrics"],
                    "best": next(
                        (
                            row
                            for row in r["state"]["history"]
                            if row["iteration"] == r["state"]["best_iteration"]
                        )
                    ),
                    "last": r["state"]["history"][-1],
                    "stop_reason": r["state"]["stop_reason"],
                }
                for case, r in results.items()
            },
            "interpretation": "Warmup omits physics and freezes lambda at 0.5. Adam network moments and global step continue at release. Gradient probes describe the original joint objective; only update traces describe the applied objective. Smaller residual or slower lambda drift alone is not success.",
        }

    def lr_case_config(source, case):
        """Only initial Adam learning rate may differ from the audited source."""
        if case not in LR_CASES:
            raise ValueError("Unknown learning-rate case")
        if source["training"]["learning_rate"] != 0.001:
            raise ValueError("Learning-rate control requires source learning_rate=0.001")
        if source["equation"]["lambda_initial"] != 0.5:
            raise ValueError("Learning-rate comparison requires initial lambda=0.5")
        result = copy.deepcopy(source)
        result["training"]["learning_rate"] = LR_CASES[case]
        return result

    def lr_observation_due(iteration, maximum_iterations, interval):
        return (
            iteration <= 1000
            and iteration % 50 == 0
            or iteration % interval == 0
            or iteration == maximum_iterations
        )

    def validate_lr_contract(case_dir, expected):
        """Do not reuse an optimizer checkpoint from another case or configuration."""
        path = Path(case_dir) / "experiment_contract.json"
        if path.exists():
            if json.loads(path.read_text(encoding="utf-8")) != expected:
                raise RuntimeError("Learning-rate experiment contract mismatch")
        elif (Path(case_dir) / "latest.json").exists():
            raise RuntimeError("Learning-rate checkpoint is missing its experiment contract")
        else:
            write_json(path, expected)

    def compare_lr_results(results):
        """Compare equal observation steps; never turn a smaller error into promotion."""
        if set(results) != set(LR_CASES):
            raise ValueError("Both learning-rate cases are required")
        control, thesis = (results[k] for k in LR_CASES)
        if control["initialization_sha256"] != thesis["initialization_sha256"]:
            raise ValueError("Learning-rate comparison initialization mismatch")
        a, b = (r["experiment_contract"] for r in (control, thesis))
        if a["case"] != "inverse_lr_control" or b["case"] != "inverse_lr_thesis":
            raise ValueError("Learning-rate case identities do not match")
        expected_config = lr_case_config(a["configuration"], "inverse_lr_thesis")
        if b["configuration"] != expected_config or any(
            (
                a[k] != b[k]
                for k in (
                    "seed",
                    "maximum_iterations",
                    "initialization_sha256",
                    "observation_protocol",
                )
            )
        ):
            raise ValueError("Learning-rate comparison has changes beyond initial learning rate")
        if not all((r["state"]["complete"] for r in results.values())):
            raise ValueError("Learning-rate comparison requires completed cases")
        histories = [
            {r["iteration"]: r for r in result["state"]["history"]} for result in (control, thesis)
        ]
        if (
            0 not in histories[0]
            or 0 not in histories[1]
            or (
                not all(
                    (
                        np.isclose(histories[0][0][k], histories[1][0][k], rtol=0.0002, atol=2e-06)
                        for k in ("relative_l1", "relative_l2", "lambda_1")
                    )
                )
            )
        ):
            raise ValueError(
                "Learning-rate cases did not start from matching fields and coefficients"
            )
        shared = []
        for iteration in sorted(histories[0].keys() & histories[1].keys()):
            row = {"iteration": iteration}
            for label, history in zip(("control", "thesis"), histories):
                row.update(
                    {
                        label + "_" + k: history[iteration][k]
                        for k in (
                            "relative_l1",
                            "relative_l2",
                            "lambda_1",
                            "lambda_absolute_error",
                            "independent_residual_mse",
                        )
                    }
                )
            row["delta_relative_l2_thesis_minus_control"] = (
                row["thesis_relative_l2"] - row["control_relative_l2"]
            )
            shared.append(row)
        return {
            "scope": "Single-seed initial-learning-rate diagnostic; not full thesis reproduction",
            "stage8_approved": False,
            "automatic_promotion": False,
            "shared_observations": shared,
            "last_shared_iteration": shared[-1]["iteration"],
            "cases": {
                k: {
                    "initial_learning_rate": LR_CASES[k],
                    "best_iteration": r["state"]["best_iteration"],
                    "best_metrics": r["best_metrics"],
                    "best": next(
                        (
                            row
                            for row in r["state"]["history"]
                            if row["iteration"] == r["state"]["best_iteration"]
                        )
                    ),
                    "last": r["state"]["history"][-1],
                    "stop_reason": r["state"]["stop_reason"],
                }
                for k, r in results.items()
            },
            "interpretation": "Compare shared iterations, field error, coefficient error and independent residual together. Slower coefficient drift alone is not success. Timings include diagnostics.",
        }

    def coefficient_trace(model, tensors):
        """Observe gradients without applying updates or assigning model variables."""
        network = [v for v in model.trainable_variables if v is not model.lambda_1]
        with tf.GradientTape(persistent=True) as tape:
            losses = calculate_losses_2d(model, tensors, training=True)
            weighted = {
                k: losses[k] * wave2d_weights[k]
                for k in ("initial", "boundary", "physics", "supervised")
            }
        vectors, gradients = ({}, {})
        for name, loss in {"total": losses["total"], **weighted}.items():
            grad = tape.gradient(loss, network)
            vector = np.concatenate(
                [
                    (np.zeros(v.shape, dtype=np.float32) if g is None else g.numpy()).ravel()
                    for v, g in zip(network, grad)
                ]
            ).astype(np.float64)
            coefficient = tape.gradient(loss, model.lambda_1)
            gradients[name] = {
                "network_norm": float(np.linalg.norm(vector)),
                "lambda_gradient": 0.0 if coefficient is None else float(coefficient.numpy()),
                "lambda_connected": coefficient is not None,
            }
            vectors[name] = vector
        del tape
        cosine = {}
        for name in ("initial", "boundary", "supervised"):
            denominator = np.linalg.norm(vectors["physics"]) * np.linalg.norm(vectors[name])
            cosine["physics_vs_" + name] = (
                None
                if denominator == 0
                else float(np.dot(vectors["physics"], vectors[name]) / denominator)
            )
        x = tf.identity(tensors["X_collocation"])
        with tf.GradientTape(persistent=True) as second:
            second.watch(x)
            with tf.GradientTape() as first:
                first.watch(x)
                value = model(x)
            first_derivative = first.gradient(value, x)
            gx, gy, gt = (
                first_derivative[:, :1],
                first_derivative[:, 1:2],
                first_derivative[:, 2:3],
            )
        a = second.gradient(gt, x)[:, 2:3].numpy().astype(np.float64)
        b = (
            (
                model.wave_speed_squared
                * (second.gradient(gx, x)[:, :1] + second.gradient(gy, x)[:, 1:2])
            )
            .numpy()
            .astype(np.float64)
        )
        del second
        coefficient = float(model.lambda_1.numpy())
        residual = a - coefficient * b
        expected_gradient = float(-2 * np.mean(b * residual) * wave2d_weights["physics"])
        sensitivity = float(np.mean(b * b))
        fixed_field_optimum = None if sensitivity <= 1e-12 else float(np.mean(a * b) / sensitivity)
        checks = {
            "lambda_only_physics_dependency": all(
                (
                    not gradients[k]["lambda_connected"]
                    for k in ("initial", "boundary", "supervised")
                )
            ),
            "lambda_gradient_sum": bool(
                np.isclose(
                    gradients["total"]["lambda_gradient"],
                    gradients["physics"]["lambda_gradient"],
                    rtol=0.0001,
                    atol=1e-06,
                )
            ),
            "analytic_lambda_gradient": bool(
                np.isclose(
                    gradients["physics"]["lambda_gradient"],
                    expected_gradient,
                    rtol=0.002,
                    atol=2e-05,
                )
            ),
            "finite_network_gradients": bool(all((np.isfinite(v).all() for v in vectors.values()))),
        }
        return {
            "gradients": gradients,
            "network_gradient_cosine": cosine,
            "analytic_weighted_lambda_gradient": expected_gradient,
            "fixed_field_least_squares_lambda": fixed_field_optimum,
            "fixed_field_sensitivity_mean_b_squared": sensitivity,
            "checks": checks,
            "interpretation": "Frozen predicted-field optimum only; not an independently identified physical coefficient.",
        }

    def compare_trace_reference(history, updates, reference):
        """Compare shared evaluation steps; finer checkpoint selection is not a new benchmark."""
        shared = []
        by_iteration = {int(row["iteration"]): row for row in reference}
        for row in history:
            old = by_iteration.get(row["iteration"])
            if old is None:
                continue
            checks = {
                key: bool(np.isclose(row[key], float(old[stored]), rtol=0.0002, atol=2e-06))
                for key, stored in (
                    ("relative_l1", "full_grid_relative_l1"),
                    ("relative_l2", "full_grid_relative_l2"),
                    ("lambda_1", "lambda_1"),
                )
            }
            shared.append({"iteration": row["iteration"], "checks": checks})
        crossing = {}
        for level in (0.4, 0.25, 0.1):
            crossing[str(level)] = next(
                (r["iteration"] for r in updates if r["lambda_after"] < level), None
            )
        return {
            "shared_evaluations": shared,
            "shared_evaluation_count": len(shared),
            "shared_evaluations_match": bool(shared)
            and all((all(r["checks"].values()) for r in shared)),
            "first_recorded_lambda_below": crossing,
            "scope": "Matching shared observations is a trajectory check, not proof of identical floating-point updates.",
        }

    def train_diagnostic(
        case,
        config,
        arrays,
        initial_path,
        run_dir,
        backup_dir,
        maximum_iterations,
        allow_extension=False,
    ):
        """Baseline-only controlled experiment; no benchmark or service promotion."""
        if case not in {"forward_fixed", "inverse_control", "inverse_trace"} | set(LR_CASES) | set(
            WARMUP_CASES
        ):
            raise ValueError(case)
        warmup_comparison = case in WARMUP_CASES
        lr_comparison = case in LR_CASES or warmup_comparison
        trace = case == "inverse_trace" or lr_comparison
        cap = 1000 if case == "inverse_trace" else 30000 if allow_extension else 10000
        if not 1 <= maximum_iterations <= cap:
            raise ValueError(f"Diagnostic total iteration cap must be within 1..{cap}")
        if warmup_comparison and maximum_iterations <= 2000:
            raise ValueError(
                "Warmup comparison requires 2001..10000 total iterations to include joint training"
            )
        warmup_extension = allow_extension and case == "data_warmup_inverse"
        if allow_extension:
            provenance_name = (
                "warmup_extension_provenance.json"
                if warmup_extension
                else "extension_provenance.json"
            )
            if (
                case not in {"forward_fixed", "data_warmup_inverse"}
                or not (Path(run_dir) / provenance_name).is_file()
            ):
                raise ValueError("Extension requires a validated compatible parent")
        case_dir = Path(run_dir) / case
        case_dir.mkdir(exist_ok=True)
        experiment_contract = None
        if lr_comparison:
            config = lr_case_config(config, "inverse_lr_thesis" if warmup_comparison else case)
            experiment_contract = {
                "case": case,
                "configuration": config,
                "seed": 3234,
                "maximum_iterations": maximum_iterations,
                "initialization_sha256": sha256(initial_path),
                "observation_protocol": "updates through 1000; evaluations every 50 through 1000 then source interval; independent residual seed 93234",
            }
            if warmup_comparison:
                experiment_contract.update(
                    warmup_iterations=WARMUP_CASES[case],
                    optimizer_transition="same Adam, network moments and global iteration retained; lambda slots frozen until joint",
                    observation_protocol="updates through 3000; shared early and 2000..3000 release window observations; independent residual seed 93234",
                )
            validate_lr_contract(case_dir, experiment_contract)
            print(
                f"Learning-rate case: {case}; initial Adam lr={config['training']['learning_rate']}; seed=3234",
                flush=True,
            )
        fixed = case == "forward_fixed"
        model = new_model(config, "baseline", 3234, initial_path)
        if lr_comparison and (
            not np.isclose(float(model.lambda_1.numpy()), 0.5, rtol=0, atol=1e-07)
        ):
            raise RuntimeError(
                "Learning-rate comparison requires the original initial coefficient, not trained weights"
            )
        if fixed:
            model.lambda_1.assign(config["equation"]["lambda_target"])
        variables = [v for v in model.trainable_variables if not (fixed and v is model.lambda_1)]
        network_variables = [v for v in variables if v is not model.lambda_1]
        cfg = config["training"]
        optimizer = tf.keras.optimizers.Adam(
            tf.keras.optimizers.schedules.ExponentialDecay(
                cfg["learning_rate"], cfg["decay_steps"], cfg["decay_rate"], staircase=True
            )
        )
        optimizer.build(variables)
        if lr_comparison and int(optimizer.iterations.numpy()) != 0:
            raise RuntimeError("New learning-rate case requires a fresh optimizer before restore")
        checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
        tensors = arrays_to_tensors(arrays)
        if lr_comparison:
            rng = np.random.default_rng(93234)
            residual_points = rng.uniform(
                [config["domain"][k + "_min"] for k in ("x", "y", "t")],
                [config["domain"][k + "_max"] for k in ("x", "y", "t")],
                (1000, 3),
            ).astype(np.float32)
            points_path = case_dir / "independent_residual_points.npz"
            if points_path.exists():
                with np.load(points_path, allow_pickle=False) as stored:
                    if not np.array_equal(stored["coordinates"], residual_points):
                        raise RuntimeError("Independent residual points changed on resume")
            else:
                np.savez(points_path, coordinates=residual_points)
        state = {
            "iteration": 0,
            "elapsed_seconds": 0.0,
            "history": [],
            "best_score": None,
            "best_iteration": None,
            "best_checkpoint": None,
            "complete": False,
        }
        if lr_comparison:
            state["experiment_contract"] = experiment_contract
        if trace:
            state.update(update_trace=[], gradient_trace=[])
        pointer = case_dir / "latest.json"
        if pointer.exists():
            version = json.loads(pointer.read_text())["directory"]
            safe_relative(version)
            state = json.loads((case_dir / version / "state.json").read_text())
            if warmup_extension:
                provenance = json.loads(
                    (Path(run_dir) / "warmup_extension_provenance.json").read_text()
                )
                validate_warmup_extension_state(
                    state, experiment_contract, provenance, case_dir / version / "state.json"
                )
            elif lr_comparison and state.get("experiment_contract") != experiment_contract:
                raise RuntimeError(
                    "Restored learning-rate checkpoint belongs to a different experiment"
                )
            checkpoint.read(str(case_dir / version / "state")).assert_consumed()
            if int(optimizer.iterations.numpy()) != state["iteration"]:
                raise RuntimeError("Restored optimizer iteration does not match checkpoint history")
            if fixed and float(model.lambda_1.numpy()) != float(
                config["equation"]["lambda_target"]
            ):
                raise RuntimeError("Restored fixed coefficient mismatch")
            if (
                warmup_comparison
                and state["iteration"] <= WARMUP_CASES[case]
                and (float(model.lambda_1.numpy()) != 0.5)
            ):
                raise RuntimeError("Restored warmup coefficient changed before release")
            restored = field_metrics(model, arrays)
            if warmup_extension and (
                not np.isclose(
                    float(model.lambda_1.numpy()),
                    state["history"][-1]["lambda_1"],
                    rtol=1e-06,
                    atol=1e-07,
                )
            ):
                raise RuntimeError(
                    "Restored inverse coefficient does not match latest checkpoint history"
                )
            for key in ("relative_l1", "relative_l2"):
                if not np.isclose(
                    restored[key], state["history"][-1][key], rtol=0.0002, atol=2e-06
                ):
                    raise RuntimeError("Restored checkpoint metrics do not match its history")
            write_json(
                case_dir / "reload_verification.json",
                {
                    "passed": True,
                    "iteration": state["iteration"],
                    "optimizer_iterations": int(optimizer.iterations.numpy()),
                    "metrics": restored,
                },
            )
            state = state_for_budget(state, maximum_iterations, allow_extension)
            if warmup_extension:
                state["experiment_contract"] = experiment_contract
            if (
                trace
                and state["gradient_trace"]
                and (not all(state["gradient_trace"][-1]["checks"].values()))
            ):
                raise RuntimeError("Cannot resume beyond a failed gradient diagnostic")
            print(
                f"Restored {case}: iteration {state['iteration']} including optimizer", flush=True
            )
            if warmup_extension:
                print(
                    f"Restored lambda={float(model.lambda_1.numpy()):.8f}; next phase=joint; warmup will not repeat",
                    flush=True,
                )
        elif allow_extension:
            raise RuntimeError(
                "Extension checkpoint missing; restart from initialization is forbidden"
            )
        started = time.perf_counter() - state["elapsed_seconds"]
        print(
            f"Training range: {state['iteration'] + 1}..{maximum_iterations} (total budget; target stopping retained)",
            flush=True,
        )

        @tf.function(autograph=False)
        def step():
            with tf.GradientTape() as tape:
                total = calculate_losses_2d(model, tensors["train"], training=True)["total"]
            gradients = tape.gradient(total, variables)
            for gradient in gradients:
                if gradient is None:
                    raise RuntimeError("Disconnected diagnostic training gradient")
                tf.debugging.assert_all_finite(gradient, "Nonfinite diagnostic gradient")
            if trace:
                lambda_gradient = next(
                    (g for g, v in zip(gradients, variables) if v is model.lambda_1)
                )
            optimizer.apply_gradients(zip(gradients, variables))
            if trace:
                return (total, lambda_gradient)

        @tf.function(autograph=False)
        def warmup_step():
            with tf.GradientTape() as tape:
                losses = calculate_losses_2d(model, tensors["train"], training=True)
                objective = warmup_objective(losses, wave2d_weights)
            gradients = tape.gradient(objective, network_variables)
            for gradient in gradients:
                if gradient is None:
                    raise RuntimeError("Disconnected warmup network gradient")
                tf.debugging.assert_all_finite(gradient, "Nonfinite warmup gradient")
            optimizer.apply_gradients(zip(gradients, network_variables))
            tf.debugging.assert_equal(
                model.lambda_1, tf.cast(0.5, model.lambda_1.dtype), "Warmup coefficient changed"
            )
            return (objective, tf.constant(0.0, tf.float32))

        def record(iteration):
            metrics = field_metrics(model, arrays)
            score = max(metrics["relative_l1"], metrics["relative_l2"])
            if not np.isfinite(score):
                raise RuntimeError("Nonfinite diagnostic field metrics")
            if fixed and float(model.lambda_1.numpy()) != float(
                config["equation"]["lambda_target"]
            ):
                raise RuntimeError("Fixed coefficient changed")
            version = f"checkpoint_{iteration:06d}_{uuid.uuid4().hex[:8]}"
            staging = case_dir / (version + ".tmp")
            staging.mkdir(exist_ok=False)
            record_row = {
                "iteration": iteration,
                "elapsed_seconds": time.perf_counter() - started,
                **metrics,
                "lambda_1": float(model.lambda_1.numpy()),
            }
            if lr_comparison:
                stats = residual_statistics(model, residual_points)
                record_row.update({"independent_residual_" + k: v for k, v in stats.items()})
                record_row["lambda_absolute_error"] = abs(
                    record_row["lambda_1"] - config["equation"]["lambda_target"]
                )
            for split, data in tensors.items():
                record_row.update({split + "_" + k: v for k, v in loss_record(model, data).items()})
            if warmup_comparison:
                phase = warmup_phase(case, iteration)
                if phase in {"warmup", "initial"} and record_row["lambda_1"] != 0.5:
                    raise RuntimeError("Warmup coefficient invariant failed")
                record_row.update(
                    phase=phase,
                    optimizer_iterations=int(optimizer.iterations.numpy()),
                    active_train_objective=(
                        warmup_objective(
                            {k: record_row["train_" + k] for k in wave2d_weights}, wave2d_weights
                        )
                        if phase == "warmup"
                        else record_row["train_total"]
                    ),
                )
            if trace:
                detail = coefficient_trace(model, tensors["train"])
                state["gradient_trace"].append({"iteration": iteration, **detail})
            state["history"].append(record_row)
            state.update(iteration=iteration, elapsed_seconds=record_row["elapsed_seconds"])
            if state["best_score"] is None or score < state["best_score"]:
                state.update(best_score=score, best_iteration=iteration, best_checkpoint=version)
            threshold = config["method_comparison"]["stop_rule"]["threshold"]
            target_observation = (
                not trace
                or iteration % config["method_comparison"]["evaluation_interval"] == 0
                or iteration == maximum_iterations
            )
            if warmup_comparison:
                target_observation = target_observation and warmup_target_allowed(case, iteration)
            state["target_reached"] = (
                target_observation
                and metrics["relative_l1"] < threshold
                and (metrics["relative_l2"] < threshold)
            )
            state["complete"] = state["target_reached"] or iteration >= maximum_iterations
            state["stop_reason"] = (
                "target_reached"
                if state["target_reached"]
                else "diagnostic_cap_reached" if state["complete"] else "running"
            )
            checkpoint.write(str(staging / "state"))
            model.save_weights(staging / "model.weights.h5")
            write_json(staging / "state.json", state)
            with (staging / "history.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(state["history"][0]))
                writer.writeheader()
                writer.writerows(state["history"])
            if trace:
                write_json(staging / "gradient_trace.json", state["gradient_trace"])
                if state["update_trace"]:
                    with (staging / "lambda_updates.csv").open(
                        "w", newline="", encoding="utf-8"
                    ) as stream:
                        writer = csv.DictWriter(stream, fieldnames=list(state["update_trace"][0]))
                        writer.writeheader()
                        writer.writerows(state["update_trace"])
            staging.rename(case_dir / version)
            write_json(pointer, {"directory": version})
            snapshot(run_dir, backup_dir, case + "_partial")
            print(
                f"{case} iter={iteration} L1={metrics['relative_l1']:.6e} L2={metrics['relative_l2']:.6e} lambda={float(model.lambda_1.numpy()):.6f}",
                flush=True,
            )
            if warmup_comparison:
                print(
                    f"  phase={record_row['phase']}; active objective={record_row['active_train_objective']:.6e}; optimizer iterations={record_row['optimizer_iterations']}",
                    flush=True,
                )
            if trace:
                print(
                    f"  dL/dlambda={detail['gradients']['total']['lambda_gradient']:.6e}; frozen-field optimum={detail['fixed_field_least_squares_lambda']}; checks={detail['checks']}",
                    flush=True,
                )
                if not all(detail["checks"].values()):
                    raise RuntimeError(
                        "Coefficient gradient diagnostic failed; checkpoint saved, next update blocked"
                    )

        if not state["history"]:
            record(0)
        if not state["complete"]:
            for iteration in range(state["iteration"] + 1, maximum_iterations + 1):
                if trace:
                    before = float(model.lambda_1.numpy())
                    learning_rate = float(tf.convert_to_tensor(optimizer.learning_rate).numpy())
                    warming = warmup_comparison and warmup_phase(case, iteration) == "warmup"
                    pre_objective, pre_gradient = warmup_step() if warming else step()
                    after = float(model.lambda_1.numpy())
                    if not lr_comparison or iteration <= (3000 if warmup_comparison else 1000):
                        state["update_trace"].append(
                            {
                                "iteration": iteration,
                                "lambda_before": before,
                                "lambda_after": after,
                                "lambda_delta": after - before,
                                "total_lambda_gradient_pre_update": float(pre_gradient.numpy()),
                                "total_loss_pre_update": float(pre_objective.numpy()),
                                "learning_rate": learning_rate,
                            }
                        )
                        if warmup_comparison:
                            state["update_trace"][-1]["phase"] = warmup_phase(case, iteration)
                else:
                    step()
                due = (
                    lr_observation_due(
                        iteration,
                        maximum_iterations,
                        config["method_comparison"]["evaluation_interval"],
                    )
                    if lr_comparison
                    else (
                        trace_due(iteration, maximum_iterations)
                        if trace
                        else iteration % config["method_comparison"]["evaluation_interval"] == 0
                        or iteration == maximum_iterations
                    )
                )
                if warmup_comparison:
                    due = warmup_observation_due(
                        iteration,
                        maximum_iterations,
                        config["method_comparison"]["evaluation_interval"],
                    )
                if due:
                    record(iteration)
                    if state["complete"]:
                        break
        model.load_weights(case_dir / state["best_checkpoint"] / "model.weights.h5")
        write_json(
            case_dir / "result.json",
            {
                "case": case,
                "state": state,
                "best_metrics": field_metrics(model, arrays),
                "experiment_contract": experiment_contract,
                "initialization_sha256": sha256(initial_path),
                "fixed_lambda": fixed,
                "automatic_promotion": False,
                "scope": "Single-seed diagnostic, not a replacement for the Stage 7 benchmark",
            },
        )
        return state

    # Historical sector-local LHS helpers. The active distribution adapter chooses
    # the training allocation and renewal period; these definitions alone do not train.
    import hashlib
    import itertools
    import json
    from pathlib import Path
    import numpy as np

    SECTOR_CASES = {
        "volume_lhs": (False, False),
        "supervised_focus": (True, False),
        "collocation_focus": (False, True),
        "both_focus": (True, True),
    }
    SAMPLING_SEEDS = [3234, 3235, 3236, 3237]
    CHANGED_ARRAYS = {"X_supervised_train", "u_supervised_train", "X_collocation_train"}

    def default_sector_design():
        """Editable design; factors multiply when conditions overlap."""
        return {
            "version": 2,
            "normalized_edges": {
                "x": [0.0, 0.2, 0.8, 1.0],
                "y": [0.0, 0.2, 0.8, 1.0],
                "t": [0.0, 0.1, 0.9, 1.0],
            },
            "density_multipliers": {
                "supervised": {"initial_time": 2.0, "spatial_boundary": 2.0, "late_boundary": 2.0},
                "collocation": {"initial_time": 3.0, "spatial_boundary": 2.0, "late_boundary": 1.0},
            },
            "sector_density_overrides": {"supervised": {}, "collocation": {}},
            "sampling_seeds": SAMPLING_SEEDS.copy(),
            "cases": list(SECTOR_CASES),
        }

    def resolve_sector_design(design, config):
        """Resolve density rules to the existing explicit allocation contract.

        Legacy version-1 proportional ratios remain supported without reinterpretation.
        A version-2 sector override REPLACES the density product for that sector.
        """
        if design.get("version") == 1:
            sector_boxes(design, config)
            return json.loads(json.dumps(design))
        if (
            set(design)
            != {
                "version",
                "normalized_edges",
                "density_multipliers",
                "sector_density_overrides",
                "sampling_seeds",
                "cases",
            }
            or design["version"] != 2
        ):
            raise ValueError("Unsupported sampling design fields/version")
        edges = design["normalized_edges"]
        if set(edges) != set("xyt"):
            raise ValueError("Exactly x/y/t edges are required")
        for axis in "xyt":
            values = np.asarray(edges[axis], dtype=float)
            if (
                values.ndim != 1
                or not 3 <= len(values) <= 9
                or (not np.isfinite(values).all())
                or (values[0] != 0)
                or (values[-1] != 1)
                or np.any(np.diff(values) <= 0)
            ):
                raise ValueError(
                    "Density designs require 2..8 sectors per axis with strictly increasing edges in [0,1]"
                )
        if set(design["density_multipliers"]) != {"supervised", "collocation"} or set(
            design["sector_density_overrides"]
        ) != {"supervised", "collocation"}:
            raise ValueError("Both independent sampling families are required")
        indices = list(itertools.product(*(range(len(edges[a]) - 1) for a in "xyt")))
        ids = {"_".join((a + str(i) for a, i in zip("xyt", idx))) for idx in indices}
        protocol = {
            "version": 1,
            "normalized_edges": edges,
            "sampling_seeds": design["sampling_seeds"],
            "cases": design["cases"],
        }
        for family in ("supervised", "collocation"):
            factors = design["density_multipliers"][family]
            overrides = design["sector_density_overrides"][family]
            if (
                set(factors) != {"initial_time", "spatial_boundary", "late_boundary"}
                or not set(overrides) <= ids
            ):
                raise ValueError("Invalid density factor/sector override keys")
            for value in list(factors.values()) + list(overrides.values()):
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (float, int))
                    or (not np.isfinite(value))
                    or (not 0 < value <= 1000000.0)
                ):
                    raise ValueError("Density multipliers must be finite positive numbers <= 1e6")
            masses = {}
            for i, j, k in indices:
                key = f"x{i}_y{j}_t{k}"
                boundary = i in (0, len(edges["x"]) - 2) or j in (0, len(edges["y"]) - 2)
                density = factors["initial_time"] if k == 0 else 1.0
                if boundary:
                    density *= factors["spatial_boundary"]
                    if k == len(edges["t"]) - 2:
                        density *= factors["late_boundary"]
                density = overrides.get(key, density)
                volume = np.prod([edges[a][q + 1] - edges[a][q] for a, q in zip("xyt", (i, j, k))])
                masses[key] = float(volume * density)
            protocol[family + "_ratios"] = masses
        sector_boxes(protocol, config)
        return json.loads(json.dumps(protocol))

    def default_sector_protocol():
        unit = {"domain": {a + s: v for a in "xyt" for s, v in (("_min", 0.0), ("_max", 1.0))}}
        return resolve_sector_design(default_sector_design(), unit)

    def sector_boxes(protocol, config):
        if (
            set(protocol)
            != {
                "version",
                "normalized_edges",
                "supervised_ratios",
                "collocation_ratios",
                "sampling_seeds",
                "cases",
            }
            or protocol["version"] != 1
        ):
            raise ValueError("Unsupported sector protocol fields/version")
        edges = protocol["normalized_edges"]
        if set(edges) != set("xyt"):
            raise ValueError("Exactly x, y, t sector edges are required")
        for a in "xyt":
            v = np.asarray(edges[a], dtype=float)
            if (
                v.ndim != 1
                or not 2 <= len(v) <= 9
                or (not np.isfinite(v).all())
                or (v[0] != 0)
                or (v[-1] != 1)
                or np.any(np.diff(v) <= 0)
            ):
                raise ValueError("Sector edges must strictly increase from 0 to 1")
        boxes = []
        domain = config["domain"]
        for index in itertools.product(*(range(len(edges[a]) - 1) for a in "xyt")):
            lower, upper, volume = ([], [], 1.0)
            for a, i in zip("xyt", index):
                lo, hi = (float(domain[a + "_min"]), float(domain[a + "_max"]))
                if not np.isfinite([lo, hi]).all() or hi <= lo:
                    raise ValueError("Invalid domain")
                lower.append(lo + (hi - lo) * edges[a][i])
                upper.append(lo + (hi - lo) * edges[a][i + 1])
                volume *= edges[a][i + 1] - edges[a][i]
            boxes.append(
                {
                    "id": "_".join((a + str(i) for a, i in zip("xyt", index))),
                    "lower": lower,
                    "upper": upper,
                    "volume": float(volume),
                }
            )
        keys = {b["id"] for b in boxes}
        for family in ("supervised", "collocation"):
            ratios = protocol[family + "_ratios"]
            if set(ratios) != keys:
                raise ValueError("Ratio keys must match every sector exactly")
            values = np.array(list(ratios.values()), dtype=float)
            if not np.isfinite(values).all() or np.any(values <= 0):
                raise ValueError("Every sector ratio must be positive and finite")
        seeds = protocol["sampling_seeds"]
        if not isinstance(seeds, list) or seeds != SAMPLING_SEEDS:
            raise ValueError("This protocol fixes four sampling replications: 3234..3237")
        if protocol["cases"] != list(SECTOR_CASES):
            raise ValueError("The matched four-arm comparison is required")
        return boxes

    def apportion(total, masses):
        """Largest remainder; deterministic sector-order ties, exact total."""
        if type(total) is not int or total <= 0:
            raise ValueError("Positive integer sample count required")
        m = np.asarray(masses, dtype=float)
        if m.ndim != 1 or not len(m) or (not np.isfinite(m).all()) or np.any(m <= 0):
            raise ValueError("Positive finite allocation masses required")
        m = m / m.max()
        ideal = total * m / m.sum()
        counts = np.floor(ideal).astype(int)
        order = np.argsort(-(ideal - counts), kind="stable")
        counts[order[: total - int(counts.sum())]] += 1
        return counts

    def randomized_lhs(count, dimension, seed):
        """Strength-1 randomized LHS; one sample per marginal stratum.

        NumPy implementation: independent permutations plus uniform jitter.
        This guarantee applies within each sector, not globally after weighting.
        """
        if type(count) is not int or count < 0 or type(dimension) is not int or (dimension < 1):
            raise ValueError("Invalid LHS shape")
        rng = np.random.default_rng(seed)
        if count == 0:
            return np.empty((0, dimension), dtype=np.float64)
        return np.column_stack(
            [(rng.permutation(count) + rng.random(count)) / count for _ in range(dimension)]
        )

    def keyed_seed(seed, family, sector):
        token = f"sector-lhs-v1/{seed}/{family}/{sector}".encode()
        return int.from_bytes(hashlib.sha256(token).digest()[:8], "little")

    def family_samples(protocol, config, family, focused, seed, total):
        if family not in {"supervised", "collocation"}:
            raise ValueError("Unsupported sampling family")
        boxes = sector_boxes(protocol, config)
        masses = np.array(
            [protocol[family + "_ratios"][b["id"]] if focused else b["volume"] for b in boxes]
        )
        fractions = masses / masses.max() / (masses / masses.max()).sum()
        counts = apportion(total, masses)
        pieces, evidence = ([], [])
        for b, count, fraction in zip(boxes, counts, fractions):
            unit = randomized_lhs(int(count), 3, keyed_seed(seed, family, b["id"]))
            lo, hi = (np.asarray(b["lower"]), np.asarray(b["upper"]))
            points = (lo + (hi - lo) * unit).astype(np.float32)
            low32, high32 = (lo.astype(np.float32), hi.astype(np.float32))
            if np.any(high32 <= np.nextafter(low32, np.float32(np.inf))):
                raise ValueError("Sector width is too narrow for float32 coordinates")
            points = np.clip(points, np.nextafter(low32, high32), np.nextafter(high32, low32))
            pieces.append(points)
            evidence.append(
                {
                    **b,
                    "family": family,
                    "focused": focused,
                    "allocation_density": float(masses[len(evidence)] / b["volume"]),
                    "requested_fraction": float(fraction),
                    "count": int(count),
                    "actual_fraction": float(count / total),
                    "unit_lhs_marginals_verified": all(
                        (
                            np.array_equal(
                                np.sort(np.floor(unit[:, d] * count).astype(int)), np.arange(count)
                            )
                            for d in range(3)
                        )
                    ),
                }
            )
        points = np.concatenate(pieces)
        if points.shape != (total, 3) or len(np.unique(points, axis=0)) != total:
            raise RuntimeError("LHS count/uniqueness check failed")
        return (points, evidence)

    def arrays_digest(arrays):
        h = hashlib.sha256()
        for key in sorted(arrays):
            a = np.ascontiguousarray(arrays[key])
            h.update(json.dumps([key, a.dtype.str, a.shape]).encode())
            h.update(a.tobytes())
        return h.hexdigest()

    SECTOR_EVALUATION_PROTOCOL = {
        "version": 1,
        "seed": 93235,
        "points_per_sector": 128,
        "batch_size": 512,
        "used_for_training_or_stopping": False,
    }

    def prepare_sector_evaluation(root, protocol, config):
        """Common independent-in-location diagnostics, not a blinded final test."""
        boxes = sector_boxes(protocol, config)
        points, ids = ([], [])
        n = SECTOR_EVALUATION_PROTOCOL["points_per_sector"]
        for index, box in enumerate(boxes):
            u = randomized_lhs(
                n, 3, keyed_seed(SECTOR_EVALUATION_PROTOCOL["seed"], "evaluation", box["id"])
            )
            lo, hi = (np.array(box["lower"]), np.array(box["upper"]))
            p = (lo + (hi - lo) * u).astype(np.float32)
            low32, high32 = (lo.astype(np.float32), hi.astype(np.float32))
            p = np.clip(p, np.nextafter(low32, high32), np.nextafter(high32, low32))
            points.append(p)
            ids.extend([index] * n)
        coordinates = np.concatenate(points)
        arrays = {
            "coordinates": coordinates,
            "sector_index": np.asarray(ids, dtype=np.int64),
            "exact": exact_numpy(coordinates.astype(float), config).reshape(-1),
        }
        path = Path(root) / "sector_evaluation_points.npz"
        if path.exists():
            with np.load(path, allow_pickle=False) as archive:
                stored = {k: archive[k] for k in archive.files}
            if arrays_digest(stored) != arrays_digest(arrays):
                raise RuntimeError("Sector evaluation points changed; continuation blocked")
        else:
            np.savez_compressed(path, **arrays)
        return arrays

    def sector_metrics(evaluation, prediction, residual, boxes):
        prediction, residual = (
            np.asarray(prediction).reshape(-1),
            np.asarray(residual).reshape(-1),
        )
        exact = evaluation["exact"]
        if (
            prediction.shape != exact.shape
            or residual.shape != exact.shape
            or (not all((np.isfinite(v).all() for v in (prediction, residual, exact))))
        ):
            raise ValueError("Invalid sector diagnostic predictions/residuals")
        rows = []
        for index, box in enumerate(boxes):
            mask = evaluation["sector_index"] == index
            if not mask.any():
                raise ValueError("Missing sector evaluation points")
            error = prediction[mask].astype(float) - exact[mask]
            f = residual[mask].astype(float)
            norm = float(np.linalg.norm(exact[mask]))
            rows.append(
                {
                    "sector": box["id"],
                    "point_count": int(mask.sum()),
                    "volume_fraction": box["volume"],
                    "field_rmse": float(np.sqrt(np.mean(error**2))),
                    "field_max_abs_error": float(np.max(np.abs(error))),
                    "relative_l2": (
                        float(np.linalg.norm(error) / norm)
                        if norm > 1e-10 * np.sqrt(mask.sum())
                        else None
                    ),
                    "exact_rms": float(np.sqrt(np.mean(exact[mask] ** 2))),
                    "residual_mse": float(np.mean(f**2)),
                    "residual_mean_abs": float(np.mean(abs(f))),
                    "residual_max_abs": float(np.max(abs(f))),
                }
            )
        return rows

    def checkpoint_at_iteration(trial, iteration):
        case = Path(trial) / "data_warmup_inverse"
        matches = [
            p
            for p in case.glob(f"checkpoint_{iteration:06d}_*")
            if p.is_dir() and (not p.name.endswith(".tmp"))
        ]
        if len(matches) != 1:
            raise RuntimeError("Matched iteration checkpoint is missing or ambiguous")
        checkpoint = matches[0]
        state = json.loads((checkpoint / "state.json").read_text())
        if state["iteration"] != iteration or state["history"][-1]["iteration"] != iteration:
            raise RuntimeError("Matched checkpoint history mismatch")
        return (checkpoint, state)

    def evaluate_sector_checkpoint(trial, config, arrays, protocol, evaluation, iteration):
        """Evaluate the common-iteration checkpoint, never silently use best/last."""
        trial = Path(trial)
        checkpoint, state = checkpoint_at_iteration(trial, iteration)
        weights = checkpoint / "model.weights.h5"
        evidence = {
            "iteration": iteration,
            "weights_sha256": sha256(weights),
            "evaluation_arrays_sha256": arrays_digest(evaluation),
            "protocol": SECTOR_EVALUATION_PROTOCOL,
            "input_manifest_sha256": sha256(trial / "input_manifest.json"),
            "runtime_sha256": sha256(trial / "diagnostic_runtime.py"),
        }
        target = trial / f"sector_evaluation_{iteration:06d}"
        if target.exists():
            manifest = json.loads((target / "complete.json").read_text())
            if manifest["evidence"] != evidence or set(manifest["files"]) != {
                "report.json",
                "sector_metrics.csv",
                "predictions.npz",
            }:
                raise RuntimeError("Cached sector evaluation contract mismatch")
            for name, digest in manifest["files"].items():
                if sha256(target / name) != digest:
                    raise RuntimeError("Cached sector evaluation checksum mismatch")
            return json.loads((target / "report.json").read_text())
        model = new_model(config, "baseline", 3234, weights)
        reload_metrics = field_metrics(model, arrays)
        if not all(
            (
                np.isclose(reload_metrics[k], state["history"][-1][k], rtol=0.0002, atol=2e-06)
                for k in ("relative_l1", "relative_l2", "mse")
            )
        ):
            raise RuntimeError("Matched checkpoint reload metric mismatch")
        if not np.isclose(
            float(model.lambda_1.numpy()), state["history"][-1]["lambda_1"], rtol=1e-06, atol=1e-07
        ):
            raise RuntimeError("Matched checkpoint coefficient mismatch")
        before = [w.numpy().copy() for w in model.weights]
        pred, residual = ([], [])
        batch = SECTOR_EVALUATION_PROTOCOL["batch_size"]
        for i in range(0, len(evaluation["coordinates"]), batch):
            p = evaluation["coordinates"][i : i + batch]
            pred.append(
                model(tf.convert_to_tensor(p, tf.float32), training=False).numpy().reshape(-1)
            )
            residual.append(independent_residual(model, p).numpy().reshape(-1))
        pred, residual = (np.concatenate(pred), np.concatenate(residual))
        if (
            len(before) != len(model.weights)
            or not all((np.array_equal(a, b.numpy()) for a, b in zip(before, model.weights)))
            or sha256(weights) != evidence["weights_sha256"]
        ):
            raise RuntimeError("Model changed during sector evaluation")
        rows = sector_metrics(evaluation, pred, residual, sector_boxes(protocol, config))
        report = {
            "evidence": evidence,
            "reload_checks_passed": True,
            "model_unchanged": True,
            "sectors": rows,
            "scope": "Matched-update per-sector diagnostics. Equal points per sector; not an unweighted whole-domain error estimate.",
        }
        staging = target.with_name(target.name + "_" + uuid.uuid4().hex + ".tmp")
        staging.mkdir(exist_ok=False)
        write_json(staging / "report.json", report)
        with (staging / "sector_metrics.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        np.savez_compressed(
            staging / "predictions.npz", **evaluation, prediction=pred, residual=residual
        )
        write_json(
            staging / "complete.json",
            {
                "evidence": evidence,
                "files": {
                    name: sha256(staging / name)
                    for name in ("report.json", "sector_metrics.csv", "predictions.npz")
                },
            },
        )
        staging.rename(target)
        return report

    def make_sector_trial(source, config, protocol, case, seed):
        if case not in SECTOR_CASES or seed not in protocol["sampling_seeds"]:
            raise ValueError("Unknown sampling trial")
        result = {k: v.copy() for k, v in source.items()}
        rows = []
        for family, focus in zip(("supervised", "collocation"), SECTOR_CASES[case]):
            key = f"X_{family}_train"
            result[key], evidence = family_samples(
                protocol, config, family, focus, seed, len(source[key])
            )
            rows.extend(evidence)
        result["u_supervised_train"] = exact_numpy(
            result["X_supervised_train"].astype(np.float64), config
        ).astype(np.float32)
        unchanged = all(
            (np.array_equal(source[k], result[k]) for k in source if k not in CHANGED_ARRAYS)
        )
        audit = data_audit(result, config)
        if not unchanged or not audit["passed"]:
            raise RuntimeError("Sampling changed protected arrays or failed the data audit")
        return (
            result,
            {
                "case": case,
                "sampling_seed": seed,
                "network_initialization_seed": 3234,
                "source_arrays_sha256": arrays_digest(source),
                "arrays_sha256": arrays_digest(result),
                "protected_arrays_unchanged": unchanged,
                "data_audit": audit,
                "sectors": rows,
                "zero_count_sectors": [
                    r["family"] + ":" + r["id"] for r in rows if r["count"] == 0
                ],
                "scope": "Fixed within-sector LHS per trial; no online resampling. Four sampling seeds, not four network initializations.",
            },
        )

    def prepare_sector_trial(root, source, config, protocol, case, seed, context, initial_path):
        """Persist immutable inputs before optimizer construction; validate on resume."""
        root = Path(root)
        expected, report = make_sector_trial(source, config, protocol, case, seed)
        expected_context = {
            "run_contract": context,
            "sampling": report,
            "initialization_sha256": sha256(initial_path),
        }
        if root.exists():
            stored = json.loads((root / "input_manifest.json").read_text())
            if stored["context"] != expected_context:
                raise RuntimeError("Sampling trial contract mismatch")
            if set(stored["files"]) != {
                "sampling_data.npz",
                "initial.weights.h5",
                "sampling_report.json",
                "source_config.json",
                "sampling_protocol.json",
                "sector_allocation.csv",
            }:
                raise RuntimeError("Sampling manifest file set mismatch")
            for name, digest in stored["files"].items():
                safe_relative(name)
                if sha256(root / name) != digest:
                    raise RuntimeError("Sampling input checksum mismatch")
            with np.load(root / "sampling_data.npz", allow_pickle=False) as archive:
                loaded = {k: archive[k] for k in archive.files}
            if arrays_digest(loaded) != arrays_digest(expected):
                raise RuntimeError("Stored sampling data differ from deterministic reconstruction")
            return (loaded, report)
        staging = root.with_name(root.name + "_" + uuid.uuid4().hex + ".tmp")
        staging.mkdir(parents=True, exist_ok=False)
        np.savez_compressed(staging / "sampling_data.npz", **expected)
        (staging / "initial.weights.h5").write_bytes(Path(initial_path).read_bytes())
        write_json(staging / "sampling_report.json", report)
        write_json(staging / "source_config.json", config)
        write_json(staging / "sampling_protocol.json", protocol)
        with (staging / "sector_allocation.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(report["sectors"][0]))
            writer.writeheader()
            writer.writerows(report["sectors"])
        names = [p.name for p in staging.iterdir() if p.is_file()]
        write_json(
            staging / "input_manifest.json",
            {"context": expected_context, "files": {n: sha256(staging / n) for n in names}},
        )
        staging.rename(root)
        return (expected, report)

    def sector_run_summary(root, protocol, seeds, maximum_iterations):
        trials, rows = ([], [])
        common_by_seed = {}
        metric_keys = (
            "relative_l1",
            "relative_l2",
            "lambda_absolute_error",
            "independent_residual_mse",
            "val_boundary",
            "val_initial_displacement",
            "val_initial_velocity",
            "val_supervised",
        )
        for seed in seeds:
            histories = {}
            for case in protocol["cases"]:
                p = Path(root) / f"seed_{seed}_{case}" / "data_warmup_inverse" / "result.json"
                if not p.is_file():
                    raise RuntimeError("A sampling trial result is missing")
                result = json.loads(p.read_text())
                state = result["state"]
                if (
                    not state["complete"]
                    or not state["history"]
                    or state["iteration"] > maximum_iterations
                ):
                    raise RuntimeError("Sampling trial is incomplete or exceeds its budget")
                history = state["history"]
                joint = [h for h in history if h.get("phase") == "joint"]
                if not joint:
                    raise RuntimeError("At least one joint-training observation is required")
                histories[case] = {h["iteration"]: h for h in joint}
                trials.append(
                    {
                        "case": case,
                        "sampling_seed": seed,
                        "last": history[-1],
                        "best_joint_by_field_error": min(
                            joint, key=lambda r: max(r["relative_l1"], r["relative_l2"])
                        ),
                        "stop_reason": state["stop_reason"],
                        "target_reached": state["target_reached"],
                    }
                )
            common = sorted(set.intersection(*(set(v) for v in histories.values())))
            if not common:
                raise RuntimeError("No matched joint observation exists across cases")
            common_by_seed[str(seed)] = common[-1]
            for case in protocol["cases"]:
                for iteration in common:
                    r = histories[case][iteration]
                    rows.append(
                        {
                            "case": case,
                            "sampling_seed": seed,
                            "iteration": iteration,
                            **{k: r[k] for k in metric_keys},
                        }
                    )
        common_all = sorted(
            set.intersection(
                *(
                    {
                        r["iteration"]
                        for r in rows
                        if r["case"] == case and r["sampling_seed"] == seed
                    }
                    for seed in seeds
                    for case in protocol["cases"]
                )
            )
        )
        if not common_all:
            raise RuntimeError("No common update across sampling replications")
        at = common_all[-1]
        aggregated = {}
        for case in protocol["cases"]:
            selected = [r for r in rows if r["case"] == case and r["iteration"] == at]
            aggregated[case] = {
                k: {
                    "mean": float(np.mean([r[k] for r in selected])),
                    "sample_std": (
                        float(np.std([r[k] for r in selected], ddof=1))
                        if len(selected) > 1
                        else None
                    ),
                }
                for k in metric_keys
            }
        return {
            "status": "complete",
            "stage8_approved": False,
            "automatic_promotion": False,
            "sampling_seeds": seeds,
            "network_initialization_seed": 3234,
            "matched_iteration": at,
            "matched_aggregate": aggregated,
            "trials": trials,
            "shared_observations": rows,
            "scope": "Exploratory sector-local LHS allocation study; fixed initial network, four sampling seeds in pilot. Not thesis reproduction or statistical significance.",
        }

    # Historical periodic-LHS experiment helpers, not a thesis-code reproduction.
    # The current distribution supplies its own explicit training policy.
    import copy
    import hashlib
    import json
    import os
    from pathlib import Path
    import tempfile
    import time
    import zipfile
    import numpy as np

    TERM_NAMES = ("initial", "boundary", "physics", "supervised")
    PERIODIC_POLICY = {
        "version": 1,
        "resample_period": 500,
        "warmup_updates": 2000,
        "sa_update_period": 50,
        "ema": 0.9,
        "factor_min": 0.25,
        "factor_max": 4.0,
        "epsilon": 1e-12,
        "algorithm": "loss_term_gradient_norm_ema_v1",
        "label_policy": "manufactured exact solution; renewed supervised labels",
        "fixed": "all validation/test/reference/diagnostic points",
        "sa_scope": "four aggregate loss terms; enabled only in loss_sa_on",
        "stopping": "joint validation relative L1 and L2 below source threshold at evaluation intervals",
    }
    PERIODIC_ARMS = ("loss_sa_off", "loss_sa_on")

    def sampling_cycle(iteration):
        """Data for update 1..500 is cycle 0, 501..1000 is cycle 1, etc."""
        if type(iteration) is not int or iteration < 0:
            raise ValueError("Nonnegative integer iteration required")
        return max(0, iteration - 1) // PERIODIC_POLICY["resample_period"]

    def sa_update_due(iteration, enabled):
        offset = iteration - PERIODIC_POLICY["warmup_updates"] - 1
        return bool(enabled and offset >= 0 and (offset % PERIODIC_POLICY["sa_update_period"] == 0))

    def balanced_factors(previous, norms):
        """Bounded EMA inverse shared-network gradient norm; no loss-value fitting."""
        previous, norms = (np.asarray(previous, dtype=float), np.asarray(norms, dtype=float))
        if (
            previous.shape != (4,)
            or norms.shape != (4,)
            or (not np.isfinite([previous, norms]).all())
        ):
            raise ValueError("Four finite factors/norms required")
        if np.any(norms < 0) or np.any(previous < 0.25) or np.any(previous > 4):
            raise ValueError("Invalid factor or gradient norm")
        active = norms > PERIODIC_POLICY["epsilon"]
        if not active.any():
            return previous.copy()
        target = np.ones(4)
        scaled = norms[active] / norms[active].max()
        target[active] = np.clip(
            scaled.mean() / scaled, PERIODIC_POLICY["factor_min"], PERIODIC_POLICY["factor_max"]
        )
        return PERIODIC_POLICY["ema"] * previous + (1 - PERIODIC_POLICY["ema"]) * target

    def effective_weights(base, factors, warming=False):
        base, factors = (np.asarray(base, dtype=float), np.asarray(factors, dtype=float))
        if base.shape != (4,) or factors.shape != (4,) or (not np.isfinite([base, factors]).all()):
            raise ValueError("Invalid loss weights")
        if np.any(base <= 0) or np.any(factors < 0.25) or np.any(factors > 4):
            raise ValueError("Loss weights out of bounds")
        base = base / base.sum()
        if warming:
            base[2] = 0.0
            return base
        weighted = base * factors
        return weighted / weighted.sum()

    def training_lhs(source, config, protocol, allocation, seed, cycle):
        """Renew EVERY training family; retain all non-training arrays byte-for-byte.

        Volume samples use sector-local LHS. IC is LHS on the x/y surface;
        BC is separate tangential-space/time LHS on each physical face.
        """
        if (
            allocation not in SECTOR_CASES
            or seed not in SAMPLING_SEEDS
            or type(cycle) is not int
            or (cycle < 0)
        ):
            raise ValueError("Invalid periodic sampling identity")
        result = {k: v.copy() for k, v in source.items()}
        cycle_seed = keyed_seed(seed, "periodic-all-training-v1", str(cycle))
        d = config["domain"]
        evidence = []
        for family, focused in zip(("supervised", "collocation"), SECTOR_CASES[allocation]):
            key = f"X_{family}_train"
            result[key], rows = family_samples(
                protocol, config, family, focused, cycle_seed, len(source[key])
            )
            evidence.extend(rows)

        def surface(count, axes, label):
            unit = randomized_lhs(count, len(axes), keyed_seed(cycle_seed, label, "surface"))
            lo = np.array([d[a + "_min"] for a in axes], dtype=np.float32)
            hi = np.array([d[a + "_max"] for a in axes], dtype=np.float32)
            coords = (lo + (hi - lo) * unit).astype(np.float32)
            return np.clip(coords, np.nextafter(lo, hi), np.nextafter(hi, lo))

        initial = surface(len(source["X_initial_train"]), "xy", "initial")
        result["X_initial_train"] = np.column_stack(
            (initial, np.full(len(initial), d["t_min"], np.float32))
        )
        old = source["X_boundary_train"]
        masks = [
            old[:, axis] == np.float32(d[a + suffix])
            for axis, a in enumerate("xy")
            for suffix in ("_min", "_max")
        ]
        if not np.all(np.sum(masks, axis=0) == 1):
            raise ValueError("Boundary points must have an unambiguous physical face")
        pieces = []
        for face, (axis, a, suffix) in enumerate(
            ((i, a, s) for i, a in enumerate("xy") for s in ("_min", "_max"))
        ):
            count = int(masks[face].sum())
            if count == 0:
                raise ValueError("Every boundary face needs training samples")
            other = "y" if axis == 0 else "x"
            coords = np.empty((count, 3), np.float32)
            coords[:, [1 - axis, 2]] = surface(count, other + "t", f"boundary_{face}")
            coords[:, axis] = d[a + suffix]
            pieces.append(coords)
        result["X_boundary_train"] = np.concatenate(pieces)
        for family in ("initial", "boundary", "supervised"):
            result[f"u_{family}_train"] = exact_numpy(
                result[f"X_{family}_train"].astype(float), config
            ).astype(np.float32)
        if d["t_min"] != 0:
            raise ValueError("This standing-wave protocol requires initial time zero")
        result["ut_initial_train"] = np.zeros_like(result["u_initial_train"])
        protected = {k: v for k, v in source.items() if not k.endswith("_train")}
        if arrays_digest(protected) != arrays_digest({k: result[k] for k in protected}):
            raise RuntimeError("Protected validation/evaluation arrays changed")
        audit = data_audit(result, config)
        if not audit["passed"]:
            raise RuntimeError("Periodic sample domain/labels/overlap audit failed")
        held_out = {
            tuple(row)
            for k, v in source.items()
            if k.startswith("X_") and k.endswith(("_val", "_test"))
            for row in v.tolist()
        }
        for family in ("initial", "boundary", "supervised", "collocation"):
            pts = result[f"X_{family}_train"]
            if len(np.unique(pts, axis=0)) != len(pts) or any(
                (tuple(row) in held_out for row in pts.tolist())
            ):
                raise RuntimeError("Training duplicates/held-out overlap; no silent retry")
        return (
            result,
            {
                "cycle": cycle,
                "sampling_seed": seed,
                "cycle_seed": cycle_seed,
                "allocation": allocation,
                "arrays_sha256": arrays_digest(result),
                "protected_sha256": arrays_digest(protected),
                "sectors": evidence,
                "all_training_families_resampled": True,
                "data_audit_passed": True,
            },
        )

    def validation_field_metrics(model, tensors):
        truth = tensors["u_supervised"].numpy().astype(float)
        prediction = model(tensors["X_supervised"], training=False).numpy().astype(float)
        delta = prediction - truth
        return {
            "relative_l1": float(np.abs(delta).sum() / max(np.abs(truth).sum(), 1e-12)),
            "relative_l2": float(np.linalg.norm(delta) / max(np.linalg.norm(truth), 1e-12)),
        }

    def make_periodic_state(config, initial_path):
        model = new_model(config, "baseline", 3234, initial_path)
        all_vars = list(model.trainable_variables)
        network = [v for v in all_vars if v is not model.lambda_1]
        cfg = config["training"]
        optimizer = tf.keras.optimizers.Adam(
            tf.keras.optimizers.schedules.ExponentialDecay(
                0.0005, cfg["decay_steps"], cfg["decay_rate"], staircase=True
            )
        )
        optimizer.build(all_vars)
        factors = tf.Variable(np.ones(4, np.float32), trainable=False, name="loss_term_factors")
        updates = tf.Variable(0, trainable=False, dtype=tf.int64, name="loss_term_update_count")
        checkpoint = tf.train.Checkpoint(
            model=model, optimizer=optimizer, factors=factors, sa_updates=updates
        )
        return (model, optimizer, factors, updates, checkpoint, all_vars, network)

    def periodic_stepper(model, optimizer, all_vars, network):

        @tf.function(autograph=False, reduce_retracing=True)
        def step(data, weights, warming):
            variables = network if warming else all_vars
            with tf.GradientTape() as tape:
                loss = calculate_losses_2d(model, data, training=True)
                objective = tf.add_n([weights[i] * loss[k] for i, k in enumerate(TERM_NAMES)])
            grads = tape.gradient(objective, variables)
            if any((g is None for g in grads)):
                raise RuntimeError("Disconnected weighted objective")
            tf.debugging.assert_all_finite(objective, "Nonfinite objective")
            for g in grads:
                tf.debugging.assert_all_finite(g, "Nonfinite gradient")
            optimizer.apply_gradients(zip(grads, variables))
            return objective

        return step

    def term_norms(model, network, data):
        """Training-only statistics. Physical lambda excluded from shared-network norms."""
        with tf.GradientTape(persistent=True) as tape:
            losses = calculate_losses_2d(model, data, training=True)
        norms = []
        for name in TERM_NAMES:
            grads = tape.gradient(losses[name], network)
            connected = [g for g in grads if g is not None]
            if not connected:
                raise RuntimeError("Entire loss term disconnected: " + name)
            norms.append(float(tf.linalg.global_norm(connected).numpy()))
        del tape
        return np.asarray(norms)

    def state_values(state):
        model, optimizer, factors, updates = state[:4]
        return [
            v.numpy().copy()
            for v in list(model.variables) + list(optimizer.variables) + [factors, updates]
        ]

    def assert_same_state(a, b, approximate=False):
        if len(a) != len(b):
            raise RuntimeError("Restored state variable count changed")
        for x, y in zip(a, b):
            if x.shape != y.shape or not (
                np.allclose(x, y, rtol=1e-06, atol=1e-07) if approximate else np.array_equal(x, y)
            ):
                raise RuntimeError("Model/optimizer/SA state mismatch")

    def verify_periodic_wiring(config, arrays, initial_path):
        """Disposable CPU-capable Stage 3 checks, including an optimizer/SA roundtrip."""
        state = make_periodic_state(config, initial_path)
        model, optimizer, factors, updates, checkpoint, variables, network = state
        data = arrays_to_tensors(arrays)["train"]
        base = [wave2d_weights[k] for k in TERM_NAMES]
        loss = calculate_losses_2d(model, data)
        off = sum((base[i] * float(loss[k].numpy()) for i, k in enumerate(TERM_NAMES)))
        if not np.isclose(off, float(loss["total"].numpy()), rtol=1e-06, atol=1e-07):
            raise RuntimeError("SA-off objective differs from baseline")
        before = state_values(state)
        norms = term_norms(model, network, data)
        assert_same_state(before, state_values(state))
        factors.assign(balanced_factors(factors.numpy(), norms).astype(np.float32))
        updates.assign_add(1)
        step = periodic_stepper(model, optimizer, variables, network)
        coefficient = float(model.lambda_1.numpy())
        step(data, tf.constant(effective_weights(base, np.ones(4), True), tf.float32), True)
        if float(model.lambda_1.numpy()) != coefficient:
            raise RuntimeError("Warmup changed physical coefficient")
        w = tf.constant(effective_weights(base, factors.numpy()), tf.float32)
        step(data, w, False)
        with tempfile.TemporaryDirectory() as folder:
            checkpoint.write(str(Path(folder) / "state"))
            clone = make_periodic_state(config, initial_path)
            clone[4].read(str(Path(folder) / "state")).assert_consumed()
            assert_same_state(state_values(state), state_values(clone))
            other_step = periodic_stepper(clone[0], clone[1], clone[5], clone[6])
            step(data, w, False)
            other_step(data, w, False)
            assert_same_state(state_values(state), state_values(clone), approximate=True)
        return {
            "passed": True,
            "baseline_loss_equivalence": True,
            "gradient_probe_state_unchanged": True,
            "warmup_lambda_frozen": True,
            "optimizer_and_sa_roundtrip": True,
            "same_next_update": True,
            "term_gradient_norms": norms.tolist(),
            "scope": "Disposable wiring probes, not trained trial performance",
        }

    def run_periodic_trial(
        root, config, source, protocol, allocation, seed, arm, initial_path, cap
    ):
        if arm not in PERIODIC_ARMS or cap not in (2100, 10000):
            raise ValueError("Unknown arm/budget")
        root = Path(root)
        if root.exists():
            raise FileExistsError("Never overwrite or silently continue a changed experiment")
        root.mkdir(parents=True)
        state = make_periodic_state(config, initial_path)
        model, optimizer, factors, updates, checkpoint, variables, network = state
        step = periodic_stepper(model, optimizer, variables, network)
        base = [wave2d_weights[k] for k in TERM_NAMES]
        enabled = arm == "loss_sa_on"
        selected, evidence = training_lhs(source, config, protocol, allocation, seed, 0)
        tensors = arrays_to_tensors(selected)
        history, sampling, adaptive = ([], [evidence], [])
        started = time.perf_counter()
        best_validation, best_iteration = (None, None)

        def record(iteration):
            nonlocal best_validation, best_iteration
            warming = iteration <= PERIODIC_POLICY["warmup_updates"]
            metrics = field_metrics(model, source)
            validation = validation_field_metrics(model, tensors["val"])
            losses = {
                split: loss_record(model, tensors[split]) for split in ("train", "val", "test")
            }
            current = effective_weights(base, factors.numpy(), warming)
            row = {
                "iteration": iteration,
                "phase": "warmup" if warming else "joint",
                "cycle": evidence["cycle"],
                "elapsed_seconds": time.perf_counter() - started,
                "reference_metrics": metrics,
                "validation_metrics": validation,
                "lambda_1": float(model.lambda_1.numpy()),
                "losses": losses,
                "factors": factors.numpy().tolist(),
                "effective_weights": current.tolist(),
                "sa_updates": int(updates.numpy()),
                "optimizer_iterations": int(optimizer.iterations.numpy()),
            }
            if not np.isfinite(
                [
                    *metrics.values(),
                    *validation.values(),
                    row["lambda_1"],
                    *[v for values in losses.values() for v in values.values()],
                ]
            ).all():
                raise RuntimeError("Nonfinite recorded metrics")
            history.append(row)
            if not warming and (
                best_validation is None or losses["val"]["total"] < best_validation
            ):
                best_validation, best_iteration = (losses["val"]["total"], iteration)
                model.save_weights(root / "best_validation.weights.h5")
            threshold = config["method_comparison"]["stop_rule"]["threshold"]
            target = (
                not warming
                and (iteration % 1000 == 0 or iteration == cap)
                and all((v < threshold for v in validation.values()))
            )
            row["target_reached"] = bool(target)
            print(
                f"{arm} seed={seed} iter={iteration} cycle={evidence['cycle']} L2(reference)={metrics['relative_l2']:.5f} lambda={row['lambda_1']:.6f} weights={np.round(current, 4).tolist()}",
                flush=True,
            )
            return target

        record(0)
        for iteration in range(1, cap + 1):
            cycle = sampling_cycle(iteration)
            if cycle != evidence["cycle"]:
                selected, evidence = training_lhs(source, config, protocol, allocation, seed, cycle)
                tensors = arrays_to_tensors(selected)
                sampling.append(evidence)
            warming = iteration <= PERIODIC_POLICY["warmup_updates"]
            if sa_update_due(iteration, enabled):
                norms = term_norms(model, network, tensors["train"])
                factors.assign(balanced_factors(factors.numpy(), norms).astype(np.float32))
                updates.assign_add(1)
                adaptive.append(
                    {
                        "iteration_pre_update": iteration,
                        "cycle": cycle,
                        "norms": norms.tolist(),
                        "factors": factors.numpy().tolist(),
                    }
                )
            weights = tf.constant(effective_weights(base, factors.numpy(), warming), tf.float32)
            step(tensors["train"], weights, warming)
            if warming and float(model.lambda_1.numpy()) != 0.5:
                raise RuntimeError("Warmup lambda invariant failed")
            if iteration % 500 == 0 or iteration in (2001, 2050, cap):
                if record(iteration):
                    break
        if not enabled and (
            int(updates.numpy()) != 0 or not np.array_equal(factors.numpy(), np.ones(4))
        ):
            raise RuntimeError("SA-off arm was modified")
        checkpoint.write(str(root / "state"))
        model.save_weights(root / "last.weights.h5")
        np.savez_compressed(root / "current_training_data.npz", **selected)
        report = {
            "complete": True,
            "arm": arm,
            "sampling_seed": seed,
            "allocation": allocation,
            "history": history,
            "sampling": sampling,
            "adaptive_updates": adaptive,
            "best_validation_iteration": best_iteration,
            "best_validation_loss": best_validation,
            "last": history[-1],
            "automatic_promotion": False,
            "stage8_approved": False,
        }
        write_json(root / "result.json", report)
        return report

    def review_periodic_trial(root, config, source, protocol, allocation, seed, arm, initial_path):
        root = Path(root)
        report = json.loads((root / "result.json").read_text())
        if not report.get("complete") or (
            report["arm"],
            report["sampling_seed"],
            report["allocation"],
        ) != (arm, seed, allocation):
            raise ValueError("Trial identity/completion mismatch")
        last = report["last"]
        restored = make_periodic_state(config, initial_path)
        restored[4].read(str(root / "state")).assert_consumed()
        if (
            int(restored[1].iterations.numpy()) != last["iteration"]
            or int(restored[3].numpy()) != last["sa_updates"]
        ):
            raise RuntimeError("Optimizer/SA counter restore mismatch")
        if not np.array_equal(restored[2].numpy(), np.asarray(last["factors"], np.float32)):
            raise RuntimeError("Adaptive factors restore mismatch")
        regenerated, evidence = training_lhs(
            source, config, protocol, allocation, seed, sampling_cycle(last["iteration"])
        )
        with np.load(root / "current_training_data.npz", allow_pickle=False) as archive:
            stored = {k: archive[k] for k in archive.files}
        if arrays_digest(stored) != evidence["arrays_sha256"] or evidence != report["sampling"][-1]:
            raise RuntimeError("Sampler state/coordinates restore mismatch")
        before = state_values(restored)
        metrics = field_metrics(restored[0], source)
        if not all(
            (
                np.isclose(metrics[k], v, rtol=0.0002, atol=2e-06)
                for k, v in last["reference_metrics"].items()
            )
        ):
            raise RuntimeError("Reload field metric mismatch")
        if not np.isclose(
            float(restored[0].lambda_1.numpy()), last["lambda_1"], rtol=1e-06, atol=1e-07
        ):
            raise RuntimeError("Reload physical coefficient mismatch")
        assert_same_state(before, state_values(restored))
        next_cycle = sampling_cycle(last["iteration"] + 1)
        _, next_evidence = training_lhs(source, config, protocol, allocation, seed, next_cycle)
        return {
            "passed": True,
            "iteration": last["iteration"],
            "model_optimizer_sa_restored": True,
            "current_coordinates_restored": True,
            "next_cycle": next_cycle,
            "next_arrays_sha256": next_evidence["arrays_sha256"],
            "evaluation_state_unchanged": True,
            "cpu_runtime": not bool(tf.config.list_physical_devices("GPU")),
        }

    def inspect_periodic_zip(path):
        with zipfile.ZipFile(path) as z:
            names = z.namelist()
            if (
                len(names) != len(set(names))
                or len(names) > 20000
                or sum((i.file_size for i in z.infolist())) > 512 * 1024**2
            ):
                raise ValueError("Duplicate/oversized periodic ZIP")
            for item in z.infolist():
                relative = safe_relative(item.filename)
                if relative.as_posix() != item.filename:
                    raise ValueError("Noncanonical archive path")
                if item.is_dir() or item.external_attr >> 16 & 61440 == 40960:
                    raise ValueError("Directory/symlink archive members unsupported")
            if z.testzip() is not None:
                raise ValueError("Periodic ZIP CRC error")
            manifest = json.loads(z.read("periodic_manifest.json"))
            if manifest.get("version") != 1 or set(manifest["files"]) != set(names) - {
                "periodic_manifest.json"
            }:
                raise ValueError("Periodic manifest inventory mismatch")
            for name, digest in manifest["files"].items():
                if hashlib.sha256(z.read(name)).hexdigest() != digest:
                    raise ValueError("Periodic ZIP hash mismatch: " + name)
            return json.loads(z.read("contract.json"))

    def save_periodic_zip(root, target):
        """One rolling ZIP; never periodic/exception partial ZIPs or bulk deletion."""
        root, target = (Path(root), Path(target))
        if target.resolve().is_relative_to(root.resolve()):
            raise ValueError("Backup target must be outside the working directory")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".zip.tmp")
        files = {}
        try:
            with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as z:
                for p in sorted(root.rglob("*")):
                    if p.is_symlink():
                        raise ValueError("Symlink output unsupported")
                    if p.is_file() and (not p.name.endswith(".tmp")):
                        name = p.relative_to(root).as_posix()
                        if name == "periodic_manifest.json":
                            continue
                        files[name] = sha256(p)
                        z.write(p, name)
                z.writestr("periodic_manifest.json", json.dumps({"version": 1, "files": files}))
            inspect_periodic_zip(temporary)
            os.replace(temporary, target)
        finally:
            if temporary.exists():
                temporary.unlink()
        print("Rolling ZIP:", target, flush=True)
        return str(target)

    def restore_periodic_zip(path, root, expected):
        if inspect_periodic_zip(path) != expected:
            raise ValueError(
                "New experiment required: source/code/profile/protocol/environment changed"
            )
        root = Path(root)
        root.mkdir(parents=True, exist_ok=False)
        with zipfile.ZipFile(path) as z:
            for name in z.namelist():
                if name == "periodic_manifest.json":
                    continue
                destination = root.joinpath(*safe_relative(name).parts)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(z.read(name))
        return root

    print("TensorFlow:", tf.__version__, "NumPy:", np.__version__)
    W8_CODE_SHA256 = "0119b6498389d5a57133d309ebb22ca0f2891f73fbaaba2fb019ce41d6aab38b"
    "New Stage 8 protocol; historical helpers are injected, never ZIP Python."
    import copy
    import hashlib
    import json
    import tempfile
    import time
    from pathlib import Path
    import numpy as np

    W8_ARMS = {
        "baseline": {"ff": False, "sa": False, "curriculum": False},
        "loss_sa": {"ff": False, "sa": True, "curriculum": False},
        "ff": {"ff": True, "sa": False, "curriculum": False},
        "loss_sa_ff": {"ff": True, "sa": True, "curriculum": False},
        "ff_curriculum": {"ff": True, "sa": False, "curriculum": True},
    }
    W8_SEEDS = [3234, 3235, 3236]
    W8_POLICY = {
        "version": 1,
        "total_update_cap": 100000,
        "warmup_updates": 2000,
        "resample_period": 5000,
        "allocation": "volume_lhs",
        "stopping_metric": "fixed_validation_supervised_relative_l2",
        "threshold": 0.1,
        "consecutive_checks": 3,
        "check_every": 1000,
        "first_check": 3000,
        "initial_lambda": 0.5,
        "initial_learning_rate": 0.0005,
        "seed_design": "three paired network-and-sampling seeds; not factorial uncertainty",
        "curriculum": {
            "kind": "joint-loss-weight-ramp-not-domain-curriculum",
            "ramp_updates": 12000,
            "physics_start": 0.1,
            "physics_end": 1.5,
            "constraint_start": 0.2,
            "constraint_end": 2.0,
            "supervised": 1.0,
        },
        "sa": {
            "algorithm": "four-loss-term-shared-network-gradient-norm-EMA",
            "every": 50,
            "ema": 0.9,
            "factor_min": 0.25,
            "factor_max": 4.0,
        },
        "reference_test_coefficient_used_for_stopping": False,
        "interpretation": "research milestone, not industrial qualification or mathematical proof",
    }

    def activate_w8_policy():
        PERIODIC_POLICY.update(resample_period=5000)
        if sampling_cycle(5000) != 0 or sampling_cycle(5001) != 1:
            raise RuntimeError("5k sampling boundary mismatch")

    def w8_validate_config(config):
        e = config["equation"]
        if e["lambda_initial"] != 0.5 or e["lambda_target"] != 1.0:
            raise ValueError("This comparison contracts lambda initial=.5, target=1")
        if config["method_comparison"]["fourier"] != {"feature_count": 32, "scale": 1.0}:
            raise ValueError("Expected fixed Fourier feature_count=32, scale=1")
        if config["loss_weights"] != {k: 1.0 for k in TERM_NAMES}:
            raise ValueError("Expected four equal baseline weights")
        if config["model"] != {
            "layers": [3, 20, 20, 20, 20, 1],
            "activation": "tanh",
            "dtype": "float32",
        }:
            raise ValueError("Unexpected architecture; requires a new benchmark contract")

    def w8_weights(arm, iteration, factors):
        spec = W8_ARMS[arm]
        warming = iteration <= 2000
        base = np.ones(4)
        if spec["curriculum"] and (not warming):
            c = W8_POLICY["curriculum"]
            f = min(1.0, max(0.0, (iteration - 2001) / c["ramp_updates"]))
            constraint = c["constraint_start"] + f * (c["constraint_end"] - c["constraint_start"])
            physics = c["physics_start"] + f * (c["physics_end"] - c["physics_start"])
            base = [constraint, constraint, physics, c["supervised"]]
        return effective_weights(base, factors if spec["sa"] else np.ones(4), warming)

    def w8_target(history):
        checks = [r for r in history if r["iteration"] >= 3000 and r["iteration"] % 1000 == 0]
        if len(checks) < 3:
            return False
        last = checks[-3:]
        return (
            last[1]["iteration"] - last[0]["iteration"] == 1000
            and last[2]["iteration"] - last[1]["iteration"] == 1000
            and all(
                (
                    np.isfinite(r["validation_metrics"]["relative_l2"])
                    and r["validation_metrics"]["relative_l2"] <= 0.1
                    for r in last
                )
            )
        )

    def w8_make_state(config, arm, seed):
        if arm not in W8_ARMS or seed not in W8_SEEDS:
            raise ValueError("Unknown method or seed")
        tf.keras.utils.set_random_seed(seed)
        model = WavePINN2DComparison(config, use_fourier=W8_ARMS[arm]["ff"], fourier_seed=seed)
        model(tf.zeros((1, 3), tf.float32))
        variables = list(model.trainable_variables)
        network = [v for v in variables if v is not model.lambda_1]
        cfg = config["training"]
        optimizer = tf.keras.optimizers.Adam(
            tf.keras.optimizers.schedules.ExponentialDecay(
                0.0005, cfg["decay_steps"], cfg["decay_rate"], staircase=True
            )
        )
        optimizer.build(variables)
        factors = tf.Variable(np.ones(4, np.float32), trainable=False, name="loss_term_factors")
        updates = tf.Variable(0, dtype=tf.int64, trainable=False, name="sa_updates")
        checkpoint = tf.train.Checkpoint(
            model=model, optimizer=optimizer, factors=factors, sa_updates=updates
        )
        return (model, optimizer, factors, updates, checkpoint, variables, network)

    def w8_ff_digest(model):
        return (
            None
            if model.fourier_matrix is None
            else hashlib.sha256(model.fourier_matrix.numpy().tobytes()).hexdigest()
        )

    def w8_model_digest(model):
        return arrays_digest({str(i): v.numpy() for i, v in enumerate(model.variables)})

    def w8_wiring(config, source, protocol):
        a, ea = training_lhs(source, config, protocol, "volume_lhs", 3234, 0)
        b, eb = training_lhs(source, config, protocol, "volume_lhs", 3234, 1)
        if ea["protected_sha256"] != eb["protected_sha256"]:
            raise RuntimeError("Held-out points changed")
        for family in ("initial", "boundary", "supervised", "collocation"):
            if np.array_equal(a["X_" + family + "_train"], b["X_" + family + "_train"]):
                raise RuntimeError("A training family did not renew")
        checks, initial = ([], {})
        probe = {k: v[:8] for k, v in arrays_to_tensors(a)["train"].items()}
        for arm in W8_ARMS:
            state = w8_make_state(config, arm, 3234)
            model, optimizer, factors, updates, ckpt, variables, network = state
            initial[arm] = w8_model_digest(model)
            step = periodic_stepper(model, optimizer, variables, network)
            before = state_values(state)
            norms = term_norms(model, network, probe)
            assert_same_state(before, state_values(state))
            step(probe, tf.constant(w8_weights(arm, 2000, factors.numpy()), tf.float32), True)
            if float(model.lambda_1.numpy()) != 0.5:
                raise RuntimeError("Warmup changed lambda")
            if int(updates.numpy()) or not np.array_equal(factors.numpy(), np.ones(4)):
                raise RuntimeError("Warmup changed SA state")
            if W8_ARMS[arm]["sa"]:
                norms = term_norms(model, network, probe)
                factors.assign(balanced_factors(factors.numpy(), norms).astype(np.float32))
                updates.assign_add(1)
            weights = tf.constant(w8_weights(arm, 2001, factors.numpy()), tf.float32)
            step(probe, weights, False)
            with tempfile.TemporaryDirectory() as folder:
                path = str(Path(folder) / "state")
                ckpt.write(path)
                clone = w8_make_state(config, arm, 3234)
                clone[4].read(path).assert_consumed()
                assert_same_state(state_values(state), state_values(clone))
                if w8_ff_digest(model) != w8_ff_digest(clone[0]):
                    raise RuntimeError("FF reconstruction mismatch")
                step(probe, weights, False)
                periodic_stepper(clone[0], clone[1], clone[5], clone[6])(probe, weights, False)
                assert_same_state(state_values(state), state_values(clone), approximate=True)
            checks.append(
                {
                    "arm": arm,
                    "passed": True,
                    "same_next_update": True,
                    "warmup_lambda_frozen": True,
                    "ff_sha256": w8_ff_digest(model),
                    "loss_term_sa_enabled": W8_ARMS[arm]["sa"],
                }
            )
        if (
            initial["baseline"] != initial["loss_sa"]
            or len({initial[a] for a in ("ff", "loss_sa_ff", "ff_curriculum")}) != 1
        ):
            raise RuntimeError("Paired initialization differs within architecture")
        return {
            "passed": True,
            "probes": checks,
            "paired_initialization": True,
            "sampler_5000_5001_boundary": True,
            "all_training_families_renew": True,
            "heldout_unchanged": True,
            "disposable_not_performance": True,
        }

    def w8_run_trial(root, config, source, protocol, arm, seed, cap):
        if cap not in (5100, 100000):
            raise ValueError("Smoke must cross 5001; benchmark cap must be 100000")
        root = Path(root)
        root.mkdir(parents=True, exist_ok=False)
        state = w8_make_state(config, arm, seed)
        model, optimizer, factors, updates, ckpt, variables, network = state
        init_digest, ff_digest = (w8_model_digest(model), w8_ff_digest(model))
        step = periodic_stepper(model, optimizer, variables, network)
        arrays, evidence = training_lhs(source, config, protocol, "volume_lhs", seed, 0)
        tensors = arrays_to_tensors(arrays)
        history, samples, adaptive = ([], [evidence], [])
        started = time.perf_counter()

        def record(iteration):
            row = {
                "iteration": iteration,
                "cycle": sampling_cycle(iteration),
                "phase": "warmup" if iteration <= 2000 else "joint",
                "reference_metrics": field_metrics(model, source),
                "validation_metrics": validation_field_metrics(model, tensors["val"]),
                "losses": {s: loss_record(model, tensors[s]) for s in ("train", "val", "test")},
                "lambda_1": float(model.lambda_1.numpy()),
                "elapsed_seconds": time.perf_counter() - started,
                "optimizer_iterations": int(optimizer.iterations.numpy()),
                "sa_updates": int(updates.numpy()),
                "factors": factors.numpy().tolist(),
                "effective_weights": w8_weights(arm, iteration, factors.numpy()).tolist(),
            }
            finite = [
                *row["reference_metrics"].values(),
                *row["validation_metrics"].values(),
                row["lambda_1"],
            ]
            finite += [v for values in row["losses"].values() for v in values.values()]
            if not np.isfinite(finite).all():
                raise RuntimeError("Nonfinite evaluation; trial is not complete")
            history.append(row)
            row["active_train_objective_after_update"] = float(
                sum(
                    (
                        row["effective_weights"][j] * row["losses"]["train"][key]
                        for j, key in enumerate(TERM_NAMES)
                    )
                )
            )
            row["target_reached"] = w8_target(history)
            print(
                f"{arm} seed={seed} iter={iteration} cycle={row['cycle']} valL2={row['validation_metrics']['relative_l2']:.5f} refL2={row['reference_metrics']['relative_l2']:.5f} lambda={row['lambda_1']:.6f}",
                flush=True,
            )
            return row["target_reached"]

        record(0)
        for iteration in range(1, cap + 1):
            cycle = sampling_cycle(iteration)
            if cycle != evidence["cycle"]:
                arrays, evidence = training_lhs(source, config, protocol, "volume_lhs", seed, cycle)
                tensors = arrays_to_tensors(arrays)
                samples.append(evidence)
                print(
                    "All-training LHS renewed before update",
                    iteration,
                    "; cycle",
                    cycle,
                    flush=True,
                )
            if sa_update_due(iteration, W8_ARMS[arm]["sa"]):
                norms = term_norms(model, network, tensors["train"])
                factors.assign(balanced_factors(factors.numpy(), norms).astype(np.float32))
                updates.assign_add(1)
                adaptive.append(
                    {
                        "before_update": iteration,
                        "cycle": cycle,
                        "norms": norms.tolist(),
                        "factors": factors.numpy().tolist(),
                    }
                )
            warming = iteration <= 2000
            step(
                tensors["train"],
                tf.constant(w8_weights(arm, iteration, factors.numpy()), tf.float32),
                warming,
            )
            if warming and float(model.lambda_1.numpy()) != 0.5:
                raise RuntimeError("Warmup coefficient changed")
            if iteration % 1000 == 0 or iteration in (2001, 5001, cap):
                reached = record(iteration)
                if reached and cap == 100000:
                    break
        if not W8_ARMS[arm]["sa"] and (
            int(updates.numpy()) or not np.array_equal(factors.numpy(), np.ones(4))
        ):
            raise RuntimeError("SA-off method acquired adaptive weights")
        ckpt.write(str(root / "state"))
        model.save_weights(root / "last.weights.h5")
        np.savez_compressed(root / "current_training_data.npz", **arrays)
        report = {
            "complete": True,
            "arm": arm,
            "sampling_seed": seed,
            "network_seed": seed,
            "cap": cap,
            "initialization_sha256": init_digest,
            "ff_sha256": ff_digest,
            "history": history,
            "sampling": samples,
            "adaptive_updates": adaptive,
            "last": history[-1],
            "stop_reason": (
                "validation_target"
                if cap == 100000 and w8_target(history)
                else "smoke_cap" if cap == 5100 else "safety_cap_goal_unmet"
            ),
        }
        write_json(root / "result.json", report)
        return report

    def w8_review_trial(root, config, source, protocol, arm, seed, cap):
        root = Path(root)
        r = json.loads((root / "result.json").read_text())
        if not r.get("complete") or (r["arm"], r["sampling_seed"], r["network_seed"], r["cap"]) != (
            arm,
            seed,
            seed,
            cap,
        ):
            raise ValueError("Trial contract mismatch")
        last, it = (r["last"], r["last"]["iteration"])
        if it > cap or (it < cap and (cap != 100000 or not w8_target(r["history"]))):
            raise ValueError("Incomplete trial or invalid early stopping")
        if r["history"][-1] != last or last["target_reached"] != w8_target(r["history"]):
            raise ValueError("Target/history mismatch")
        state = w8_make_state(config, arm, seed)
        if (
            w8_model_digest(state[0]) != r["initialization_sha256"]
            or w8_ff_digest(state[0]) != r["ff_sha256"]
        ):
            raise ValueError("Initial model or FF matrix not reproducible")
        state[4].read(str(root / "state")).assert_consumed()
        expected_updates = len(range(2001, it + 1, 50)) if W8_ARMS[arm]["sa"] else 0
        if (
            int(state[1].iterations.numpy()) != it
            or int(state[3].numpy()) != expected_updates
            or expected_updates != last["sa_updates"]
        ):
            raise ValueError("Optimizer / SA counter mismatch")
        if not np.array_equal(state[2].numpy(), np.asarray(last["factors"], np.float32)):
            raise ValueError("SA state mismatch")
        regenerated, evidence = training_lhs(
            source, config, protocol, "volume_lhs", seed, sampling_cycle(it)
        )
        with np.load(root / "current_training_data.npz", allow_pickle=False) as z:
            stored = {k: z[k] for k in z.files}
        if arrays_digest(stored) != arrays_digest(regenerated) or evidence != r["sampling"][-1]:
            raise ValueError("Restored LHS data mismatch")
        if [s["cycle"] for s in r["sampling"]] != list(range(sampling_cycle(it) + 1)):
            raise ValueError("Missing / duplicated resampling cycle")
        if len({s["protected_sha256"] for s in r["sampling"]}) != 1:
            raise ValueError("Held-out arrays changed across cycles")
        before = state_values(state)
        tensors = arrays_to_tensors(stored)
        actual = field_metrics(state[0], source)
        val = validation_field_metrics(state[0], tensors["val"])
        for actuals, expected in (
            (actual, last["reference_metrics"]),
            (val, last["validation_metrics"]),
        ):
            if not all(
                (np.isclose(actuals[k], v, rtol=0.0003, atol=3e-06) for k, v in expected.items())
            ):
                raise RuntimeError("CPU / GPU reload metric mismatch")
        if not np.isclose(
            float(state[0].lambda_1.numpy()), last["lambda_1"], rtol=1e-06, atol=1e-07
        ):
            raise RuntimeError("Reload lambda mismatch")
        losses = {s: loss_record(state[0], tensors[s]) for s in ("train", "val", "test")}
        if not all(
            (
                np.isclose(losses[s][k], v, rtol=0.001, atol=1e-05)
                for s in losses
                for k, v in last["losses"][s].items()
            )
        ):
            raise RuntimeError("Reload derivative/loss mismatch")
        assert_same_state(before, state_values(state))
        next_cycle = sampling_cycle(it + 1)
        _, next_evidence = training_lhs(source, config, protocol, "volume_lhs", seed, next_cycle)
        return {
            "passed": True,
            "arm": arm,
            "seed": seed,
            "iteration": it,
            "full_state_restored": True,
            "current_arrays_restored": True,
            "next_cycle": next_cycle,
            "next_arrays_sha256": next_evidence["arrays_sha256"],
            "evaluation_state_unchanged": True,
            "cpu_runtime": not bool(tf.config.list_physical_devices("GPU")),
        }

    def w8_summary(reports, reviews, profile, new_updates):
        expected = 15 if profile == "benchmark" else 5 if profile == "smoke" else 0
        if (
            len(reports) != expected
            or len(reviews) != expected
            or any((not r["passed"] for r in reviews))
        ):
            raise ValueError("Missing completed trial/reload check")
        if len({(r["sampling_seed"], r["arm"]) for r in reports}) != expected:
            raise ValueError("Duplicate trial identity")
        for seed in {r["sampling_seed"] for r in reports}:
            group = [r for r in reports if r["sampling_seed"] == seed]
            if {r["arm"] for r in group} != set(W8_ARMS):
                raise ValueError("Missing method")
            count = min((len(r["sampling"]) for r in group))
            if any((r["sampling"][:count] != group[0]["sampling"][:count] for r in group)):
                raise ValueError("Five-arm sampling not paired")
            for architecture in (False, True):
                if (
                    len(
                        {
                            r["initialization_sha256"]
                            for r in group
                            if W8_ARMS[r["arm"]]["ff"] == architecture
                        }
                    )
                    != 1
                ):
                    raise ValueError("Initialization not paired within architecture")
        common = (
            set.intersection(*[{r["iteration"] for r in t["history"]} for t in reports])
            if reports
            else set()
        )
        common = max(common) if common else None
        matched = [
            {
                "arm": t["arm"],
                "seed": t["sampling_seed"],
                **next((r for r in t["history"] if r["iteration"] == common)),
            }
            for t in reports
        ]
        rows = [
            {
                "arm": r["arm"],
                "seed": r["sampling_seed"],
                **r["last"],
                "stop_reason": r["stop_reason"],
                "reference_target_met": r["last"]["reference_metrics"]["relative_l2"] <= 0.1,
                "lambda_absolute_error": abs(r["last"]["lambda_1"] - 1.0),
            }
            for r in reports
        ]
        aggregate = {}
        for arm in W8_ARMS if reports else []:
            m = [r for r in matched if r["arm"] == arm]
            final = [r for r in rows if r["arm"] == arm]
            values = [r["reference_metrics"]["relative_l2"] for r in m]
            aggregate[arm] = {
                "matched_reference_l2_mean": float(np.mean(values)),
                "matched_reference_l2_sample_std": (
                    float(np.std(values, ddof=1)) if len(values) > 1 else None
                ),
                "validation_target_count": sum((r["target_reached"] for r in final)),
                "reference_target_count": sum((r["reference_target_met"] for r in final)),
                "trial_count": len(final),
                "censored_at_100k": sum(
                    (r["stop_reason"] == "safety_cap_goal_unmet" for r in final)
                ),
            }
        cpu = bool(reviews) and all((r["cpu_runtime"] for r in reviews))
        return {
            "status": "complete",
            "profile": profile,
            "completed_trials": len(reports),
            "training_executed": new_updates > 0,
            "optimizer_updates_this_run": new_updates,
            "stage8_execution_authorized": True,
            "automatic_production_promotion": False,
            "paired_sampling_verified": bool(reports),
            "cpu_reload_verified": cpu,
            "matched_iteration": common,
            "matched_rows": matched,
            "final_rows": rows,
            "aggregate": aggregate,
            "reload_checks": reviews,
            "candidate_audit_unlocked": profile == "benchmark" and cpu,
            "all_methods_reached_reference_10pct": bool(rows)
            and all((r["reference_target_met"] for r in rows)),
            "numerical_proof_or_industrial_qualification": False,
        }

    def w8_figures(root, reports, config):
        import matplotlib.pyplot as plt

        folder = Path(root) / "figures"
        folder.mkdir(exist_ok=True)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        colors = dict(zip(W8_ARMS, plt.get_cmap("tab10").colors))
        for r in reports:
            h = r["history"]
            x = [v["iteration"] for v in h]
            label = r["arm"] + " / " + str(r["sampling_seed"])
            axes[0].plot(
                x,
                [v["reference_metrics"]["relative_l2"] for v in h],
                color=colors[r["arm"]],
                alpha=0.65,
                label=label,
            )
            axes[1].plot(x, [v["lambda_1"] for v in h], color=colors[r["arm"]], alpha=0.65)
            axes[2].plot(
                x, [v["losses"]["val"]["physics"] for v in h], color=colors[r["arm"]], alpha=0.65
            )
        for ax, title in zip(
            axes, ("Reference relative L2", "Learned lambda", "Fixed validation PDE MSE")
        ):
            ax.set_title(title)
            ax.set_xlabel("Total optimizer updates")
            ax.grid(alpha=0.2)
        axes[0].axhline(0.1, color="black", linestyle="--")
        axes[0].set_yscale("log")
        axes[1].axhline(1.0, color="black", linestyle="--")
        axes[2].set_yscale("log")
        axes[0].legend(fontsize=6, ncol=2)
        fig.tight_layout()
        for suffix in ("png", "pdf"):
            fig.savefig(folder / ("wave_five_methods." + suffix), dpi=180)
        plt.close(fig)
        d = config["domain"]
        x = np.linspace(d["x_min"], d["x_max"], 65)
        y = np.linspace(d["y_min"], d["y_max"], 65)
        xx, yy = np.meshgrid(x, y, indexing="ij")
        points = np.column_stack((xx.ravel(), yy.ravel(), np.full(xx.size, d["t_max"]))).astype(
            np.float32
        )
        truth = exact_numpy(points.astype(float), config).reshape(xx.shape)
        predictions = []
        for arm in W8_ARMS:
            state = w8_make_state(config, arm, 3234)
            state[4].read(
                str(Path(root) / "trials" / ("seed_3234_" + arm) / "state")
            ).assert_consumed()
            predictions.append(state[0](points, training=False).numpy().reshape(xx.shape))
        limit = max(float(abs(truth).max()), max((float(abs(p).max()) for p in predictions)))
        error_limit = max((float(abs(p - truth).max()) for p in predictions))
        fig, axes = plt.subplots(3, 5, figsize=(16, 9), constrained_layout=True)
        for j, (arm, pred) in enumerate(zip(W8_ARMS, predictions)):
            for i, data in enumerate((truth, pred, abs(pred - truth))):
                im = axes[i, j].imshow(
                    data.T,
                    origin="lower",
                    extent=(x[0], x[-1], y[0], y[-1]),
                    vmin=0 if i == 2 else -limit,
                    vmax=error_limit if i == 2 else limit,
                    cmap="magma" if i == 2 else "coolwarm",
                )
                axes[i, j].set_title(
                    arm + " / " + ("exact", "prediction", "abs error")[i], fontsize=8
                )
                fig.colorbar(im, ax=axes[i, j], shrink=0.7)
        fig.suptitle(
            "Seed 3234, final time; final checkpoints may have different stopping iterations"
        )
        for suffix in ("png", "pdf"):
            fig.savefig(folder / ("wave_field_seed3234." + suffix), dpi=160)
        plt.close(fig)

    def w8_historical_gate(path, source_sha):
        c = inspect_periodic_zip(path)
        with zipfile.ZipFile(path) as z:
            s = json.loads(z.read("summary.json"))
        if (
            c.get("profile") != "target10"
            or s.get("completed_trials") != 8
            or (not s.get("cpu_reload_verified"))
        ):
            raise ValueError("Need CPU-reviewed eight-trial Target10 evidence")
        if (
            not s.get("paired_sampling_verified")
            or len(s.get("reload_checks", [])) != 8
            or (not all((r.get("passed") and r.get("cpu_runtime") for r in s["reload_checks"])))
        ):
            raise ValueError("Target10 CPU reload gate incomplete")
        ancestor = c
        while "parent_contract" in ancestor:
            ancestor = ancestor["parent_contract"]
        if ancestor.get("source_sha256") != source_sha:
            raise ValueError("Historical gate and source ZIP differ")
        return {
            "file": Path(path).name,
            "sha256": sha256(path),
            "cpu_reviewed_trials": 8,
            "authorization": "User explicitly requested Stage 8 five-method execution",
            "not_a_claim_all_parent_reference_errors_are_below_10pct": True,
        }

    activate_w8_policy()
    import scipy

    LB_CODE_SHA256 = "d78b150474272dd0976ef58382e75185b818e0d90a76f673e365588b855cf69f"
    # Bounded L-BFGS refinement helpers. A selected inference snapshot is not an
    # Adam checkpoint; continuation restores the separately retained Adam state.
    import json
    import time
    import hashlib
    from pathlib import Path
    import numpy as np

    LB_POLICY = {
        "version": 1,
        "optimizer": "scipy_L-BFGS-B_without_bounds",
        "stage": "optional_post_training_not_part_of_Adam_100k_budget",
        "field_error_target": None,
        "smoke_iterations": 20,
        "refine_iterations": 500,
        "smoke_evaluations": 500,
        "refine_evaluations": 10000,
        "maxcor": 10,
        "maxls": 30,
        "ftol": 1e-07,
        "gtol": 1e-06,
        "model_dtype": "float32_preserved",
        "solver_vector_dtype": "float64",
        "trainable": "network_and_inverse_lambda",
        "frozen": "parent_last_training_coordinates_labels_SA_factors_curriculum_weights_FF_matrix",
        "selection": "finite_candidate_and_positive_lambda_and_lower_training_objective_and_nonworse_validation_L2",
        "test_reference_coefficient_truth_used_for_selection": False,
        "restart": "completed_trials_reused; unfinished_trial_restarts_from_immutable_parent",
    }

    class LBEvaluationLimit(Exception):
        pass

    class LBNonFinite(Exception):
        pass

    def lb_pack(variables):
        return np.concatenate([v.numpy().reshape(-1).astype(np.float64) for v in variables])

    def lb_assign(variables, position):
        position = np.asarray(position, np.float64)
        sizes = [int(np.prod(v.shape)) for v in variables]
        if position.ndim != 1 or len(position) != sum(sizes) or (not np.isfinite(position).all()):
            raise LBNonFinite("Invalid flat parameter vector")
        offset = 0
        for variable, size in zip(variables, sizes):
            variable.assign(position[offset : offset + size].reshape(variable.shape))
            offset += size

    def lb_value_gradient(model, variables, data, weights):
        """No sampling, weight updates, optimizer calls or validation inside objective."""
        weights = tf.constant(np.asarray(weights, np.float32), tf.float32)

        @tf.function(autograph=False)
        def evaluate():
            with tf.GradientTape() as tape:
                losses = calculate_losses_2d(model, data, training=False)
                objective = tf.add_n([weights[j] * losses[k] for j, k in enumerate(TERM_NAMES)])
            gradients = tape.gradient(objective, variables)
            if any((g is None for g in gradients)):
                raise RuntimeError("Disconnected L-BFGS parameter")
            return (objective, tf.concat([tf.reshape(g, [-1]) for g in gradients], axis=0))

        def value_gradient(position):
            lb_assign(variables, position)
            value, gradient = evaluate()
            value = float(value.numpy())
            gradient = gradient.numpy().astype(np.float64)
            if not np.isfinite(value) or not np.isfinite(gradient).all():
                raise LBNonFinite("Nonfinite loss/gradient from trial line-search point")
            return (value, gradient)

        return value_gradient

    def lb_solve(value_gradient, x0, max_iterations, max_evaluations, minimize_fn=None):
        """Keep last ACCEPTED iterate, not the last rejected line-search point.

        Hard function-evaluation budget is enforced independently of SciPy's options.
        No validation/reference-based stopping and no threshold such as 5%.
        """
        if (
            type(max_iterations) is not int
            or max_iterations < 1
            or type(max_evaluations) is not int
            or (max_evaluations < 1)
        ):
            raise ValueError("Positive integer budgets required")
        if minimize_fn is None:
            from scipy.optimize import minimize

            minimize_fn = minimize
        accepted = np.asarray(x0, np.float64).copy()
        if accepted.ndim != 1 or not np.isfinite(accepted).all():
            raise ValueError("Invalid initial parameters")
        evaluations = 0
        accepted_iterations = 0
        log = []
        cached_x = None
        cached_f = None
        started = time.perf_counter()

        def fun(x):
            nonlocal evaluations, cached_x, cached_f
            if evaluations >= max_evaluations:
                raise LBEvaluationLimit()
            evaluations += 1
            f, g = value_gradient(x)
            if not np.isfinite(f) or not np.isfinite(g).all():
                raise LBNonFinite("Nonfinite objective")
            if np.shape(g) != accepted.shape:
                raise RuntimeError("Gradient shape mismatch")
            cached_x = np.asarray(x).copy()
            cached_f = float(f)
            return (float(f), np.asarray(g, np.float64))

        def callback(xk):
            nonlocal accepted, accepted_iterations
            if not np.isfinite(xk).all():
                raise LBNonFinite("Nonfinite accepted iterate")
            accepted = np.asarray(xk, np.float64).copy()
            accepted_iterations += 1
            value = (
                cached_f if cached_x is not None and np.array_equal(cached_x, accepted) else None
            )
            log.append(
                {
                    "accepted_iteration": accepted_iterations,
                    "objective_evaluations": evaluations,
                    "training_objective": value,
                }
            )
            if accepted_iterations == 1 or accepted_iterations % 50 == 0:
                print(
                    "L-BFGS accepted=",
                    accepted_iterations,
                    "evaluations=",
                    evaluations,
                    "objective=",
                    value,
                    flush=True,
                )

        try:
            result = minimize_fn(
                fun,
                accepted.copy(),
                jac=True,
                method="L-BFGS-B",
                bounds=None,
                callback=callback,
                options={
                    "maxiter": max_iterations,
                    "maxfun": max_evaluations,
                    **{k: LB_POLICY[k] for k in ("maxcor", "maxls", "ftol", "gtol")},
                },
            )
            status = (
                "converged"
                if bool(result.success)
                else "budget_limit" if int(result.status) == 1 else "optimizer_failed"
            )
            message = str(result.message)
            success = bool(result.success)
            if int(result.nit) != accepted_iterations:
                raise RuntimeError("SciPy callback/accepted iteration count mismatch")
        except LBEvaluationLimit:
            status = "evaluation_limit"
            message = "Hard objective-evaluation limit reached"
            success = False
        except LBNonFinite as exc:
            status = "nonfinite_trial_point"
            message = str(exc)
            success = False
        return (
            accepted,
            {
                "status": status,
                "optimizer_converged": success,
                "message": message,
                "accepted_iterations": accepted_iterations,
                "objective_evaluations": evaluations,
                "elapsed_seconds": time.perf_counter() - started,
                "accepted_history": log,
                "field_error_target": None,
                "position_policy": "last_accepted_iterate_or_parent",
            },
        )

    def lb_decision(before, candidate):
        values = [
            candidate["weighted_train_objective"],
            candidate["validation_metrics"]["relative_l2"],
            candidate["lambda_1"],
        ]
        finite = bool(np.isfinite(values).all())
        reasons = []
        if not finite:
            reasons.append("nonfinite_candidate")
        else:
            if candidate["lambda_1"] <= 0:
                reasons.append("nonpositive_physical_coefficient")
            if candidate["weighted_train_objective"] >= before["weighted_train_objective"]:
                reasons.append("training_objective_not_lower")
            if (
                candidate["validation_metrics"]["relative_l2"]
                > before["validation_metrics"]["relative_l2"]
            ):
                reasons.append("validation_L2_worsened")
        return {
            "selected": "candidate" if not reasons else "parent",
            "reasons": reasons,
            "reference_or_test_used": False,
            "optimizer_convergence_is_not_field_accuracy": True,
        }

    def lb_measure(state, arrays, source, weights):
        before = state_values(state)
        tensors = arrays_to_tensors(arrays)
        losses = {s: loss_record(state[0], tensors[s]) for s in ("train", "val", "test")}
        result = {
            "reference_metrics": field_metrics(state[0], source),
            "validation_metrics": validation_field_metrics(state[0], tensors["val"]),
            "lambda_1": float(state[0].lambda_1.numpy()),
            "losses": losses,
            "weighted_train_objective": float(
                sum((weights[j] * losses["train"][k] for j, k in enumerate(TERM_NAMES)))
            ),
        }
        assert_same_state(before, state_values(state))
        vals = [
            *result["reference_metrics"].values(),
            *result["validation_metrics"].values(),
            result["lambda_1"],
            result["weighted_train_objective"],
        ]
        vals += [v for group in losses.values() for v in group.values()]
        if not np.isfinite(vals).all():
            raise RuntimeError("Nonfinite accepted-state evaluation")
        return result

    def lb_frozen_state(state):
        return [v.numpy().copy() for v in state[1].variables] + [
            state[2].numpy().copy(),
            state[3].numpy().copy(),
        ]

    def lb_parent_tree(root):
        manifest = json.loads((Path(root) / "periodic_manifest.json").read_text())
        files = {p.relative_to(root).as_posix() for p in Path(root).rglob("*") if p.is_file()}
        if files != set(manifest["files"]) | {"periodic_manifest.json"}:
            raise ValueError("Parent inventory changed")
        for name, digest in manifest["files"].items():
            if sha256(Path(root).joinpath(*safe_relative(name).parts)) != digest:
                raise ValueError("Parent file changed: " + name)

    def lb_cpu_receipt(summary, seeds, stage8=False):
        expected = {(seed, arm) for seed in seeds for arm in W8_ARMS}
        checks = summary.get("reload_checks", [])
        if (
            summary.get("status") != "complete"
            or not summary.get("cpu_reload_verified")
            or summary.get("completed_trials") != len(expected)
            or (len(checks) != len(expected))
            or ({(r.get("seed"), r.get("arm")) for r in checks} != expected)
            or (not all((r.get("passed") and r.get("cpu_runtime") for r in checks)))
            or (stage8 and (not summary.get("paired_sampling_verified")))
        ):
            raise ValueError("Complete matching CPU review required")

    def lb_solver_probe():

        def quadratic(x):
            return (float(np.dot(x, x)), 2 * x)

        point, result = lb_solve(quadratic, np.array([2.0, -3.0]), 20, 100)
        if np.linalg.norm(point) > 1e-05 or not result["optimizer_converged"]:
            raise RuntimeError("SciPy analytic-gradient solver probe failed")
        return {
            "passed": True,
            "scope": "disposable quadratic, not PINN performance",
            "accepted_iterations": result["accepted_iterations"],
        }

    def lb_trial(root, parent, config, source, protocol, arm, seed, parent_cap, profile):
        root, parent = (Path(root), Path(parent))
        if profile not in ("smoke", "refine"):
            raise ValueError("Invalid L-BFGS profile")
        w8_review_trial(parent, config, source, protocol, arm, seed, parent_cap)
        parent_report = json.loads((parent / "result.json").read_text())
        state = w8_make_state(config, arm, seed)
        state[4].read(str(parent / "state")).assert_consumed()
        with np.load(parent / "current_training_data.npz", allow_pickle=False) as archive:
            arrays = {k: archive[k] for k in archive.files}
        arrays_sha = arrays_digest(arrays)
        frozen = lb_frozen_state(state)
        ff_sha = w8_ff_digest(state[0])
        weights = w8_weights(arm, parent_report["last"]["iteration"], state[2].numpy())
        root.mkdir(parents=True, exist_ok=False)
        before = lb_measure(state, arrays, source, weights)
        x0 = lb_pack(state[5])
        objective = lb_value_gradient(
            state[0], state[5], arrays_to_tensors(arrays)["train"], weights
        )
        first = objective(x0)
        second = objective(x0)
        if first[0] != second[0] or not np.array_equal(first[1], second[1]):
            raise RuntimeError("L-BFGS objective is not deterministic at fixed parameters")
        np.testing.assert_array_equal(lb_pack(state[5]), x0)
        assert_same_state(frozen, lb_frozen_state(state))
        candidate_x, solver = lb_solve(
            objective, x0, LB_POLICY[profile + "_iterations"], LB_POLICY[profile + "_evaluations"]
        )
        lb_assign(state[5], candidate_x)
        candidate = lb_measure(state, arrays, source, weights)
        state[0].save_weights(root / "candidate.weights.h5")
        decision = lb_decision(before, candidate)
        if decision["selected"] == "parent":
            lb_assign(state[5], x0)
        selected = lb_measure(state, arrays, source, weights)
        state[0].save_weights(root / "selected.weights.h5")
        assert_same_state(frozen, lb_frozen_state(state))
        if arrays_digest(arrays) != arrays_sha or w8_ff_digest(state[0]) != ff_sha:
            raise RuntimeError("Frozen data/FF changed")
        report = {
            "complete": True,
            "arm": arm,
            "seed": seed,
            "profile": profile,
            "parent_result_sha256": sha256(parent / "result.json"),
            "parent_adam_iteration": parent_report["last"]["iteration"],
            "parent_arrays_sha256": arrays_sha,
            "ff_sha256": ff_sha,
            "frozen_loss_weights": weights.tolist(),
            "frozen_SA_factors": state[2].numpy().tolist(),
            "before": before,
            "candidate": candidate,
            "selected": selected,
            "decision": decision,
            "solver": solver,
            "objective_repeat_probe_evaluations": 2,
            "sampler_and_adam_and_SA_unchanged": True,
            "optimizer_state_note": "Selected weights are not an Adam continuation checkpoint; parent Adam state is immutable",
            "field_error_target": None,
            "automatic_production_promotion": False,
        }
        write_json(root / "result.json", report)
        print(
            "L-BFGS:",
            seed,
            arm,
            solver["status"],
            "selected=",
            decision["selected"],
            "reference L2:",
            before["reference_metrics"]["relative_l2"],
            "->",
            selected["reference_metrics"]["relative_l2"],
            flush=True,
        )
        return report

    def lb_review(root, parent, config, source, protocol, arm, seed, parent_cap, profile):
        root, parent = (Path(root), Path(parent))
        w8_review_trial(parent, config, source, protocol, arm, seed, parent_cap)
        report = json.loads((root / "result.json").read_text())
        if (
            not report.get("complete")
            or (report["arm"], report["seed"], report["profile"]) != (arm, seed, profile)
            or report["parent_result_sha256"] != sha256(parent / "result.json")
        ):
            raise ValueError("Postprocessing result identity/parent mismatch")
        parent_report = json.loads((parent / "result.json").read_text())
        state = w8_make_state(config, arm, seed)
        state[4].read(str(parent / "state")).assert_consumed()
        frozen = lb_frozen_state(state)
        with np.load(parent / "current_training_data.npz", allow_pickle=False) as archive:
            arrays = {k: archive[k] for k in archive.files}
        weights = w8_weights(arm, parent_report["last"]["iteration"], state[2].numpy())
        if report["frozen_loss_weights"] != weights.tolist() or report[
            "parent_arrays_sha256"
        ] != arrays_digest(arrays):
            raise ValueError("Frozen loss/data mismatch")
        if (
            report["frozen_SA_factors"] != state[2].numpy().tolist()
            or report["parent_adam_iteration"] != parent_report["last"]["iteration"]
        ):
            raise ValueError("Parent iteration/SA mismatch")
        if report["ff_sha256"] != w8_ff_digest(state[0]) or report["decision"] != lb_decision(
            report["before"], report["candidate"]
        ):
            raise ValueError("FF/selection mismatch")
        solver = report["solver"]
        if (
            not 0 <= solver["accepted_iterations"] <= LB_POLICY[profile + "_iterations"]
            or not 0 <= solver["objective_evaluations"] <= LB_POLICY[profile + "_evaluations"]
        ):
            raise ValueError("L-BFGS budget exceeded")

        def compare(a, b):
            if isinstance(a, dict):
                return set(a) == set(b) and all((compare(a[k], b[k]) for k in a))
            return bool(np.isclose(a, b, rtol=0.001, atol=1e-06))

        if not compare(lb_measure(state, arrays, source, weights), report["before"]):
            raise RuntimeError("Before metrics reload mismatch")
        parent_parameters = lb_pack(state[5])
        candidate_parameters = None
        for name in ("candidate", "selected"):
            state[0].load_weights(root / (name + ".weights.h5"))
            if not compare(lb_measure(state, arrays, source, weights), report[name]):
                raise RuntimeError(name + " reload mismatch")
            if name == "candidate":
                candidate_parameters = lb_pack(state[5])
        np.testing.assert_array_equal(
            lb_pack(state[5]),
            (
                candidate_parameters
                if report["decision"]["selected"] == "candidate"
                else parent_parameters
            ),
        )
        expected = (
            report["candidate"]
            if report["decision"]["selected"] == "candidate"
            else report["before"]
        )
        if not compare(report["selected"], expected):
            raise ValueError("Selected metric provenance mismatch")
        assert_same_state(frozen, lb_frozen_state(state))
        return {
            "passed": True,
            "arm": arm,
            "seed": seed,
            "before_candidate_selected_reloaded": True,
            "frozen_states_unchanged": True,
            "cpu_runtime": not bool(tf.config.list_physical_devices("GPU")),
        }

    def lb_summary(reports, reviews, profile, new_evaluations):
        expected = {
            (seed, arm) for seed in ([3234] if profile == "smoke" else W8_SEEDS) for arm in W8_ARMS
        }
        if (
            {(r["seed"], r["arm"]) for r in reports} != expected
            or len(reports) != len(expected)
            or len(reviews) != len(expected)
            or ({(r["seed"], r["arm"]) for r in reviews} != expected)
            or (not all((r["passed"] for r in reviews)))
        ):
            raise ValueError("Missing completed postprocessing trial/review")
        rows = []
        for r in reports:
            rows.append(
                {
                    "arm": r["arm"],
                    "seed": r["seed"],
                    "parent_adam_iteration": r["parent_adam_iteration"],
                    "solver_status": r["solver"]["status"],
                    "accepted_iterations": r["solver"]["accepted_iterations"],
                    "objective_evaluations": r["solver"]["objective_evaluations"],
                    "elapsed_seconds": r["solver"]["elapsed_seconds"],
                    "decision": r["decision"],
                    **{
                        name: {
                            "reference_L2": r[name]["reference_metrics"]["relative_l2"],
                            "validation_L2": r[name]["validation_metrics"]["relative_l2"],
                            "lambda_absolute_error": abs(r[name]["lambda_1"] - 1.0),
                            "validation_PDE_MSE": r[name]["losses"]["val"]["physics"],
                            "test_PDE_MSE": r[name]["losses"]["test"]["physics"],
                            "weighted_train_objective": r[name]["weighted_train_objective"],
                        }
                        for name in ("before", "candidate", "selected")
                    },
                }
            )
        return {
            "status": "complete",
            "profile": profile,
            "completed_trials": len(reports),
            "optimization_executed_this_run": new_evaluations > 0,
            "objective_evaluations_this_run": new_evaluations,
            "field_error_target": None,
            "parent_preserved": True,
            "stage8_metrics_overwritten": False,
            "cpu_reload_verified": all((r["cpu_runtime"] for r in reviews)),
            "reload_checks": reviews,
            "selected_candidate_count": sum(
                (r["decision"]["selected"] == "candidate" for r in reports)
            ),
            "reference_improved_count": sum(
                (
                    r["selected"]["reference_metrics"]["relative_l2"]
                    < r["before"]["reference_metrics"]["relative_l2"]
                    for r in reports
                )
            ),
            "reference_worsened_count": sum(
                (
                    r["selected"]["reference_metrics"]["relative_l2"]
                    > r["before"]["reference_metrics"]["relative_l2"]
                    for r in reports
                )
            ),
            "rows": rows,
            "automatic_production_promotion": False,
            "note": "Numerical optimizer convergence, selection and independent field improvement are separate outcomes",
        }

    print("SciPy:", scipy.__version__)
    INTEGRATED_CODE_SHA256 = "d534bc13d7d40a7a63b7c3a510640dbaaf3a99af62113a93d4c4b24621a4aa5e"
    # Historical Adam-to-L-BFGS orchestration. Defining these helpers does not
    # establish that the current distribution passed a fresh CPU execution check.
    import json
    import copy
    import contextlib
    import re
    import sys
    from pathlib import Path

    LEGACY_INTEGRATED_CODE = "6109a6e60944d9f11f0c6582b2ca3f6df4454bc6edc1a22b3182edb325169fbd"

    def integrated_input_contract(stored, expected, profile):
        if stored == expected:
            return None
        legacy = copy.deepcopy(expected)
        legacy.pop("review_revision", None)
        legacy["integrated_code_sha256"] = LEGACY_INTEGRATED_CODE
        if profile != "review" or stored != legacy:
            raise ValueError(
                "Source/code/profile/policy/environment mismatch; unsupported migration"
            )
        return {
            "from_code": LEGACY_INTEGRATED_CODE,
            "to_code": expected["integrated_code_sha256"],
            "review_revision": expected["review_revision"],
            "scope": "review-only; original trial files unchanged; no training",
        }

    @contextlib.contextmanager
    def integrated_log(root, phase, seed, arm):
        folder = Path(root) / "logs"
        folder.mkdir(exist_ok=True)
        output = sys.stdout

        class TrialLog:

            def __init__(self, file):
                self.file = file
                self.pending = ""

            def write(self, text):
                self.file.write(text)
                self.pending += text
                while "\n" in self.pending:
                    line, self.pending = self.pending.split("\n", 1)
                    match = re.search("\\biter=(\\d+)", line)
                    if phase == "adam" and match and (int(match.group(1)) % 10000 == 0):
                        output.write(line + "\n")
                        output.flush()
                return len(text)

            def flush(self):
                self.file.flush()

        with (folder / f"{phase}_seed_{seed}_{arm}.log").open("a", encoding="utf-8") as file:
            with contextlib.redirect_stdout(TrialLog(file)):
                try:
                    yield
                finally:
                    file.flush()

    INTEGRATED_POLICY = {
        "version": 1,
        "pipeline": "Adam_all_trials_then_reload_then_LBFGS_all_trials",
        "smoke_adam_cap": 5100,
        "benchmark_adam_cap": 100000,
        "smoke_lbfgs_profile": "smoke",
        "benchmark_lbfgs_profile": "refine",
        "cpu_review": "after_both_phases; combined_smoke_CPU_review_required_before_benchmark",
        "storage": "one_rolling_ZIP; completed_Adam_and_LBFGS_trials_saved_separately; no_partial_snapshots",
        "no_field_error_target_for_lbfgs": True,
    }

    def integrated_shape(profile):
        if profile not in ("smoke", "benchmark"):
            raise ValueError("Unknown integrated profile")
        return ([3234], 5100, "smoke") if profile == "smoke" else (W8_SEEDS, 100000, "refine")

    def integrated_adam_lock(root, create=False):
        """Freeze the Adam TRIAL subtree; summary/figures may be re-evaluated on CPU."""
        root = Path(root)
        trial_root = root / "trials"
        files = {
            p.relative_to(root).as_posix(): sha256(p)
            for p in sorted(trial_root.rglob("*"))
            if p.is_file()
        }
        if not files or any((p.is_symlink() for p in trial_root.rglob("*"))):
            raise ValueError("Missing/unsafe Adam trial tree")
        lock = root / "adam_lock.json"
        if lock.exists():
            if json.loads(lock.read_text(encoding="utf-8")) != {"files": files}:
                raise ValueError("Immutable Adam checkpoint/history/data changed")
        elif create:
            write_json(lock, {"files": files})
        else:
            raise ValueError("Missing immutable Adam lock")
        return files

    def integrated_adam(root, config, source, protocol, profile, review=False, boundary=None):
        root = Path(root)
        seeds, cap, _ = integrated_shape(profile)
        if review:
            print("Adam skipped: CPU review")
            return 0
        if (root / "adam_lock.json").exists():
            integrated_adam_lock(root)
        new_updates = 0
        for seed in seeds:
            for arm in W8_ARMS:
                trial = root / "trials" / f"seed_{seed}_{arm}"
                if (trial / "result.json").exists():
                    w8_review_trial(trial, config, source, protocol, arm, seed, cap)
                    print("Reused Adam:", seed, arm)
                    continue
                if (root / "adam_lock.json").exists():
                    raise ValueError("Locked Adam run cannot acquire new trials")
                if trial.exists():
                    raise RuntimeError(
                        "Unfinished local Adam trial: fresh runtime + load last rolling ZIP"
                    )
                print("Adam start:", seed, arm, flush=True)
                with integrated_log(root, "adam", seed, arm):
                    result = w8_run_trial(trial, config, source, protocol, arm, seed, cap)
                new_updates += result["last"]["iteration"]
                w8_review_trial(trial, config, source, protocol, arm, seed, cap)
                print(
                    "Adam complete:",
                    seed,
                    arm,
                    "iteration=",
                    result["last"]["iteration"],
                    flush=True,
                )
                if boundary:
                    boundary()
        return new_updates

    def integrated_postprocess(
        root, config, source, protocol, profile, review=False, boundary=None
    ):
        root = Path(root)
        seeds, cap, lb_profile = integrated_shape(profile)
        for seed in seeds:
            for arm in W8_ARMS:
                w8_review_trial(
                    root / "trials" / f"seed_{seed}_{arm}", config, source, protocol, arm, seed, cap
                )
        integrated_adam_lock(root, create=not review)
        print(
            "Adam -> L-BFGS handoff: PASS; no parent ZIP selection; CPU review follows both phases"
        )
        if boundary and (not review):
            boundary()
        new_evaluations = 0
        for seed in seeds:
            for arm in W8_ARMS:
                relative = Path("trials") / f"seed_{seed}_{arm}"
                parent, trial = (root / relative, root / "postprocess" / relative)
                if (trial / "result.json").exists():
                    lb_review(trial, parent, config, source, protocol, arm, seed, cap, lb_profile)
                    print("Reused L-BFGS:", seed, arm)
                    continue
                if review:
                    raise RuntimeError(
                        "Incomplete L-BFGS trial: GPU load must finish before CPU review"
                    )
                if trial.exists():
                    raise RuntimeError(
                        "Unfinished local L-BFGS trial: fresh runtime + load last rolling ZIP"
                    )
                print("L-BFGS start:", seed, arm, flush=True)
                with integrated_log(root, "lbfgs", seed, arm):
                    result = lb_trial(
                        trial, parent, config, source, protocol, arm, seed, cap, lb_profile
                    )
                new_evaluations += result["solver"]["objective_evaluations"]
                lb_review(trial, parent, config, source, protocol, arm, seed, cap, lb_profile)
                integrated_adam_lock(root)
                print("L-BFGS complete:", seed, arm, flush=True)
                if boundary:
                    boundary()
        integrated_adam_lock(root)
        return new_evaluations

    def integrated_collect(
        root, config, source, protocol, profile, new_updates=0, new_evaluations=0
    ):
        root = Path(root)
        seeds, cap, lb_profile = integrated_shape(profile)
        integrated_adam_lock(root)
        adam_reports, adam_reviews, post_reports, post_reviews = ([], [], [], [])
        for seed in seeds:
            for arm in W8_ARMS:
                relative = Path("trials") / f"seed_{seed}_{arm}"
                parent, trial = (root / relative, root / "postprocess" / relative)
                adam_reviews.append(
                    w8_review_trial(parent, config, source, protocol, arm, seed, cap)
                )
                post_reviews.append(
                    lb_review(trial, parent, config, source, protocol, arm, seed, cap, lb_profile)
                )
                adam_reports.append(
                    json.loads((parent / "result.json").read_text(encoding="utf-8"))
                )
                post_reports.append(json.loads((trial / "result.json").read_text(encoding="utf-8")))
        adam = w8_summary(adam_reports, adam_reviews, profile, new_updates)
        post = lb_summary(post_reports, post_reviews, lb_profile, new_evaluations)
        cpu = adam["cpu_reload_verified"] and post["cpu_reload_verified"]
        summary = {
            "status": "complete",
            "profile": profile,
            "completed_trial_pairs": len(adam_reports),
            "cpu_reload_verified": cpu,
            "adam_unchanged": True,
            "adam": adam,
            "lbfgs": post,
            "lbfgs_field_error_target": None,
            "candidate_audit_unlocked": profile == "benchmark"
            and cpu
            and adam["candidate_audit_unlocked"],
            "automatic_production_promotion": False,
            "note": "Adam method comparison and optional L-BFGS postprocessing are separate results",
        }
        integrated_adam_lock(root)
        return (summary, adam_reports)

    def integrated_cpu_gate(summary, profile="smoke"):
        seeds, cap, lb_profile = integrated_shape(profile)
        if (
            summary.get("status") != "complete"
            or summary.get("profile") != profile
            or summary.get("completed_trial_pairs") != len(seeds) * len(W8_ARMS)
            or (not summary.get("cpu_reload_verified"))
            or (not summary.get("adam_unchanged"))
        ):
            raise ValueError("Need completed integrated pipeline CPU review")
        adam, post = (summary.get("adam", {}), summary.get("lbfgs", {}))
        if adam.get("profile") != profile or post.get("profile") != lb_profile:
            raise ValueError("Integrated child profiles mismatch")
        lb_cpu_receipt(adam, seeds, stage8=True)
        lb_cpu_receipt(post, seeds)
        if profile == "smoke" and (
            not all((r.get("iteration") == cap for r in adam["reload_checks"]))
        ):
            raise ValueError("Smoke must exercise the actual 5001 LHS renewal")

    # Historical checkpoint-first review: restoring saved weights is different
    # from reproducing random initialization bit-for-bit. Record initialization
    # differences separately from checkpoint/FF consistency checks.
    import json
    from pathlib import Path
    import numpy as np

    REVIEW_REVISION = "checkpoint-first-v2"

    def checkpoint_initialization_evidence(state, report, arm, seed):
        cpu = not bool(tf.config.list_physical_devices("GPU"))
        initial_hash = w8_model_digest(state[0])
        ff_hash = w8_ff_digest(state[0])
        if ff_hash != report["ff_sha256"]:
            raise ValueError(f"FF matrix reconstruction mismatch: {arm} seed={seed}")
        match = initial_hash == report["initialization_sha256"]
        if not cpu and (not match):
            raise ValueError(f"GPU fresh initialization hash mismatch: {arm} seed={seed}")
        return {
            "review_revision": REVIEW_REVISION,
            "fresh_initialization_hash_matches": match,
            "stored_initialization_sha256": report["initialization_sha256"],
            "review_initialization_sha256": initial_hash,
            "ff_reconstruction_verified": True,
            "initialization_note": "Fresh initialization bytewise equality is recorded separately from CPU checkpoint restoration; no initial arrays available to measure its difference",
        }

    def checkpoint_weights_agree(state, path):
        before = state_values(state)
        state[0].load_weights(Path(path) / "last.weights.h5")
        assert_same_state(before, state_values(state))

    def w8_review_trial(root, config, source, protocol, arm, seed, cap):
        root = Path(root)
        r = json.loads((root / "result.json").read_text(encoding="utf-8"))
        if not r.get("complete") or (r["arm"], r["sampling_seed"], r["network_seed"], r["cap"]) != (
            arm,
            seed,
            seed,
            cap,
        ):
            raise ValueError("Trial contract mismatch")
        last, it = (r["last"], r["last"]["iteration"])
        if it > cap or (it < cap and (cap != 100000 or not w8_target(r["history"]))):
            raise ValueError("Incomplete trial or invalid early stopping")
        if r["history"][-1] != last or last["target_reached"] != w8_target(r["history"]):
            raise ValueError("Target/history mismatch")
        state = w8_make_state(config, arm, seed)
        initialization = checkpoint_initialization_evidence(state, r, arm, seed)
        state[4].read(str(root / "state")).assert_consumed()
        checkpoint_weights_agree(state, root)
        expected_updates = len(range(2001, it + 1, 50)) if W8_ARMS[arm]["sa"] else 0
        if (
            int(state[1].iterations.numpy()) != it
            or int(state[3].numpy()) != expected_updates
            or expected_updates != last["sa_updates"]
        ):
            raise ValueError("Optimizer / SA counter mismatch")
        if not np.array_equal(state[2].numpy(), np.asarray(last["factors"], np.float32)):
            raise ValueError("SA state mismatch")
        regenerated, evidence = training_lhs(
            source, config, protocol, "volume_lhs", seed, sampling_cycle(it)
        )
        with np.load(root / "current_training_data.npz", allow_pickle=False) as z:
            stored = {k: z[k] for k in z.files}
        if arrays_digest(stored) != arrays_digest(regenerated) or evidence != r["sampling"][-1]:
            raise ValueError("Restored LHS data mismatch")
        if [s["cycle"] for s in r["sampling"]] != list(range(sampling_cycle(it) + 1)):
            raise ValueError("Missing / duplicated resampling cycle")
        if len({s["protected_sha256"] for s in r["sampling"]}) != 1:
            raise ValueError("Held-out arrays changed across cycles")
        before = state_values(state)
        tensors = arrays_to_tensors(stored)
        actual = field_metrics(state[0], source)
        val = validation_field_metrics(state[0], tensors["val"])
        for actuals, expected in (
            (actual, last["reference_metrics"]),
            (val, last["validation_metrics"]),
        ):
            if not all(
                (np.isclose(actuals[k], v, rtol=0.0003, atol=3e-06) for k, v in expected.items())
            ):
                raise RuntimeError("CPU / GPU reload metric mismatch")
        if not np.isclose(
            float(state[0].lambda_1.numpy()), last["lambda_1"], rtol=1e-06, atol=1e-07
        ):
            raise RuntimeError("Reload lambda mismatch")
        losses = {s: loss_record(state[0], tensors[s]) for s in ("train", "val", "test")}
        if not all(
            (
                np.isclose(losses[s][k], v, rtol=0.001, atol=1e-05)
                for s in losses
                for k, v in last["losses"][s].items()
            )
        ):
            raise RuntimeError("Reload derivative/loss mismatch")
        assert_same_state(before, state_values(state))
        next_cycle = sampling_cycle(it + 1)
        _, next_evidence = training_lhs(source, config, protocol, "volume_lhs", seed, next_cycle)
        return {
            "passed": True,
            "arm": arm,
            "seed": seed,
            "iteration": it,
            "full_state_restored": True,
            "current_arrays_restored": True,
            "next_cycle": next_cycle,
            "next_arrays_sha256": next_evidence["arrays_sha256"],
            "evaluation_state_unchanged": True,
            "cpu_runtime": not bool(tf.config.list_physical_devices("GPU")),
            "checkpoint_h5_agree": True,
            **initialization,
        }

    print("Review revision:", REVIEW_REVISION)
    FF_CODE_SHA256 = "68a9d585ecb44b20a37025301801655ea3b700830b92985974e2d14dd377fbfa"
    # Historical FF research helpers with their original aggregate-loss definitions.
    # Current three-group loss-SA and CrossLR settings are selected later. These
    # adaptations are not claimed as exact thesis or literature implementations.
    import copy
    import hashlib
    import json
    import tempfile
    from pathlib import Path
    import numpy as np

    def ff_spec(kind, lr=0.004, width=64, depth=5, scales=(1.0,), sa=False):
        return dict(
            kind=kind,
            lr=lr,
            width=width,
            depth=depth,
            scales=list(scales),
            features_per_bank=32,
            sa=sa,
            ff=kind != "plain",
            curriculum=False,
        )

    FF_ARMS = {
        "ff20_lr0005": ff_spec("single", 0.0005, 20, 4),
        "ff20_lr004": ff_spec("single", 0.004, 20, 4),
        "plain64_lr004": ff_spec("plain"),
        "ff64_sigma1": ff_spec("single"),
        "ff64_sigma025": ff_spec("single", scales=(0.25,)),
        "multiscale64": ff_spec("multiscale", scales=(0.125, 0.25, 0.5)),
        "spacetime64": ff_spec("spacetime", scales=(0.125, 0.25, 0.5)),
        "spacetime64_loss_sa": ff_spec("spacetime", scales=(0.125, 0.25, 0.5), sa=True),
    }
    FF_POLICY = {
        "version": 1,
        "learning_rate_provenance": "0.004 user-reported likely thesis run; not independently verified",
        "normalization": "each physical coordinate to [-1,1]",
        "phase": "2*pi*normalized_coordinate@B; B in cycles per normalized coordinate",
        "scale_selection": "a priori exploratory .125/.25/.5; no exact-solution frequencies injected",
        "banks": "fixed Gaussian matrices; same standardized draws across scale branches",
        "multiscale": "shared 5x64 tanh trunk per joint XYZ-time bank, concatenate branch outputs",
        "spacetime": "separate XY and t banks; shared trunk; all 3x3 elementwise latent products",
        "coordinate_path": "single FF: raw normalized coordinates concatenated before trunk; multi/ST: zero-padded local coordinates before shared trunk",
        "comparison": "paired sampling; matched iteration and parameter counts reported; not parameter/FLOP matched",
        "scope": "Wave2D inverse architecture exploration; not thesis reproduction or production approval",
        "references": [
            "https://bmild.github.io/fourfeat/",
            "https://arxiv.org/abs/2012.10047",
            "https://github.com/PredictiveIntelligenceLab/MultiscalePINNs/blob/main/wave1D/wave_models_tf.py",
        ],
    }
    W8_ARMS = copy.deepcopy(FF_ARMS)

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

    def w8_make_state(config, arm, seed):
        if arm not in W8_ARMS or seed not in W8_SEEDS:
            raise ValueError("Unknown FF trial")
        tf.keras.utils.set_random_seed(seed)
        spec = W8_ARMS[arm]
        cfg = copy.deepcopy(config)
        cfg["model"]["layers"] = [3] + [spec["width"]] * spec["depth"] + [1]
        if spec["kind"] in ("plain", "single"):
            cfg["method_comparison"]["fourier"] = {"feature_count": 32, "scale": spec["scales"][0]}
            model = WavePINN2DComparison(cfg, use_fourier=spec["ff"], fourier_seed=seed)
        else:
            model = WaveFFResearch(cfg, spec, seed)
        model(tf.zeros((1, 3), tf.float32))
        variables = list(model.trainable_variables)
        network = [v for v in variables if v is not model.lambda_1]
        optimizer = tf.keras.optimizers.Adam(
            tf.keras.optimizers.schedules.ExponentialDecay(
                spec["lr"],
                cfg["training"]["decay_steps"],
                cfg["training"]["decay_rate"],
                staircase=True,
            )
        )
        optimizer.build(variables)
        factors = tf.Variable(np.ones(4, np.float32), trainable=False)
        updates = tf.Variable(0, dtype=tf.int64, trainable=False)
        ckpt = tf.train.Checkpoint(
            model=model, optimizer=optimizer, factors=factors, sa_updates=updates
        )
        return (model, optimizer, factors, updates, ckpt, variables, network)

    def w8_ff_digest(model):
        if hasattr(model, "ff_banks"):
            return arrays_digest({str(i): b.numpy() for i, b in enumerate(model.ff_banks)})
        return (
            None
            if model.fourier_matrix is None
            else hashlib.sha256(model.fourier_matrix.numpy().tobytes()).hexdigest()
        )

    def ff_observe(state, data):
        """Read-only training-term diagnostics, not extra stopping/selection checks."""
        before = state_values(state)
        model, _, _, _, _, variables, network = state
        norms = term_norms(model, network, data)
        with tf.GradientTape() as tape:
            physics = calculate_losses_2d(model, data, training=False)["physics"]
        derivative = tape.gradient(physics, model.lambda_1)
        if derivative is None or not np.isfinite([*norms, float(derivative.numpy())]).all():
            raise RuntimeError("Nonfinite/disconnected FF diagnostic")
        assert_same_state(before, state_values(state))
        return {
            "term_order": list(TERM_NAMES),
            "network_gradient_norms": norms.tolist(),
            "physics_lambda_gradient": float(derivative.numpy()),
            "state_unchanged": True,
            "trainable_parameters": sum((int(np.prod(v.shape)) for v in variables)),
            "nontrainable_parameters": sum(
                (int(np.prod(v.shape)) for v in model.non_trainable_variables)
            ),
            "used_for_stopping_or_selection": False,
        }

    def w8_wiring(config, source, protocol):
        a, ea = training_lhs(source, config, protocol, "volume_lhs", 3234, 0)
        b, eb = training_lhs(source, config, protocol, "volume_lhs", 3234, 1)
        if (
            sampling_cycle(5000) != 0
            or sampling_cycle(5001) != 1
            or ea["protected_sha256"] != eb["protected_sha256"]
        ):
            raise RuntimeError("LHS/fixed-point contract failed")
        for family in ("initial", "boundary", "supervised", "collocation"):
            if np.array_equal(a["X_" + family + "_train"], b["X_" + family + "_train"]):
                raise RuntimeError("Training family failed to renew")
        data = {k: v[:8] for k, v in arrays_to_tensors(a)["train"].items()}
        probes, initial = ([], {})
        for arm in W8_ARMS:
            state = w8_make_state(config, arm, 3234)
            model, optimizer, factors, updates, ckpt, variables, network = state
            initial[arm] = w8_model_digest(model)
            diagnostic = ff_observe(state, data)
            step = periodic_stepper(model, optimizer, variables, network)
            step(data, tf.constant(w8_weights(arm, 2000, factors.numpy()), tf.float32), True)
            if float(model.lambda_1.numpy()) != 0.5:
                raise RuntimeError("Warmup changed lambda")
            if W8_ARMS[arm]["sa"]:
                factors.assign(
                    balanced_factors(factors.numpy(), term_norms(model, network, data)).astype(
                        np.float32
                    )
                )
                updates.assign_add(1)
            weights = tf.constant(w8_weights(arm, 2001, factors.numpy()), tf.float32)
            step(data, weights, False)
            digest = w8_ff_digest(model)
            with tempfile.TemporaryDirectory() as temp:
                path = Path(temp)
                ckpt.write(str(path / "state"))
                model.save_weights(path / "last.weights.h5")
                clone = w8_make_state(config, arm, 3234)
                clone[4].read(str(path / "state")).assert_consumed()
                assert_same_state(state_values(state), state_values(clone))
                checkpoint_weights_agree(clone, path)
                if w8_ff_digest(clone[0]) != digest:
                    raise RuntimeError("FF banks changed on restore")
                step(data, weights, False)
                periodic_stepper(clone[0], clone[1], clone[5], clone[6])(data, weights, False)
                assert_same_state(state_values(state), state_values(clone), approximate=True)
            probes.append(
                {
                    "arm": arm,
                    "passed": True,
                    "same_next_update": True,
                    "checkpoint_h5_agree": True,
                    "warmup_lambda_frozen": True,
                    "ff_sha256": digest,
                    "diagnostics": diagnostic,
                }
            )
        for pair in [("ff20_lr0005", "ff20_lr004"), ("spacetime64", "spacetime64_loss_sa")]:
            if initial[pair[0]] != initial[pair[1]]:
                raise RuntimeError("Paired initialization differs")
        return {
            "passed": True,
            "probes": probes,
            "paired_initialization": True,
            "all_training_families_renew": True,
            "heldout_unchanged": True,
            "disposable_not_performance": True,
        }

    def integrated_shape(profile):
        if profile == "smoke":
            return ([3234], 5100, "smoke")
        if profile == "pilot":
            return ([3234], 10000, "smoke")
        if profile == "benchmark":
            return (W8_SEEDS, 100000, "refine")
        raise ValueError("Unknown FF profile")

    def w8_summary(reports, reviews, profile, new_updates):
        seeds, cap, _ = integrated_shape(profile)
        expected = {(s, a) for s in seeds for a in W8_ARMS}
        if (
            len(reports) != len(expected)
            or {(r["sampling_seed"], r["arm"]) for r in reports} != expected
            or len(reviews) != len(expected)
            or ({(r["seed"], r["arm"]) for r in reviews} != expected)
            or (not all((r["passed"] for r in reviews)))
        ):
            raise ValueError("Missing/duplicate trial or review")
        for seed in seeds:
            group = {r["arm"]: r for r in reports if r["sampling_seed"] == seed}
            length = min((len(r["sampling"]) for r in group.values()))
            if any(
                (
                    r["sampling"][:length] != group["ff20_lr0005"]["sampling"][:length]
                    for r in group.values()
                )
            ):
                raise ValueError("Sampling is not paired")
            for x, y in [("ff20_lr0005", "ff20_lr004"), ("spacetime64", "spacetime64_loss_sa")]:
                if group[x]["initialization_sha256"] != group[y]["initialization_sha256"]:
                    raise ValueError("Paired initialization mismatch")
        common = max(set.intersection(*[{r["iteration"] for r in t["history"]} for t in reports]))
        matched = [
            {
                "arm": t["arm"],
                "seed": t["sampling_seed"],
                **next((r for r in t["history"] if r["iteration"] == common)),
            }
            for t in reports
        ]
        final = [
            {
                "arm": t["arm"],
                "seed": t["sampling_seed"],
                **t["last"],
                "stop_reason": t["stop_reason"],
            }
            for t in reports
        ]
        aggregate = {}
        for arm in W8_ARMS:
            values = [r["reference_metrics"]["relative_l2"] for r in matched if r["arm"] == arm]
            aggregate[arm] = {
                "matched_reference_l2_mean": float(np.mean(values)),
                "sample_std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                "validation_target_count": sum(
                    (r["target_reached"] for r in final if r["arm"] == arm)
                ),
            }
        return {
            "status": "complete",
            "profile": profile,
            "completed_trials": len(reports),
            "training_executed": new_updates > 0,
            "optimizer_updates_this_run": new_updates,
            "paired_sampling_verified": True,
            "matched_iteration": common,
            "matched_rows": matched,
            "final_rows": final,
            "aggregate": aggregate,
            "reload_checks": reviews,
            "cpu_reload_verified": all((r["cpu_runtime"] for r in reviews)),
            "candidate_audit_unlocked": False,
            "automatic_production_promotion": False,
            "scope": FF_POLICY["scope"],
        }

    def ff_gate(summary, profile):
        integrated_cpu_gate(summary, profile)
        if not summary["adam"].get("paired_sampling_verified"):
            raise ValueError("Unpaired gate")

    def ff_figures(root, reports, config):
        import matplotlib.pyplot as plt

        folder = Path(root) / "figures"
        folder.mkdir(exist_ok=True)
        fig, axes = plt.subplots(1, 3, figsize=(16, 4))
        for r in reports:
            h = r["history"]
            x = [v["iteration"] for v in h]
            label = r["arm"] + "/" + str(r["sampling_seed"])
            for ax, vals in zip(
                axes,
                (
                    [v["validation_metrics"]["relative_l2"] for v in h],
                    [v["lambda_1"] for v in h],
                    [v["losses"]["val"]["physics"] for v in h],
                ),
            ):
                ax.plot(x, vals, label=label, linewidth=1)
        for ax, title in zip(
            axes, ["Fixed validation L2", "Inverse lambda", "Fixed validation PDE MSE"]
        ):
            ax.set_title(title)
            ax.set_xlabel("Adam updates")
            ax.grid(alpha=0.2)
        axes[0].set_yscale("log")
        axes[0].axhline(0.1, color="black", linestyle="--")
        axes[2].set_yscale("log")
        axes[1].axhline(1, color="black", linestyle="--")
        axes[0].legend(fontsize=5)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(folder / ("ff_learning_curves." + ext), dpi=160)
        plt.close(fig)
        d = config["domain"]
        x, y = [np.linspace(d[a + "_min"], d[a + "_max"], 65) for a in "xy"]
        xx, yy = np.meshgrid(x, y, indexing="ij")
        pts = np.column_stack([xx.ravel(), yy.ravel(), np.full(xx.size, d["t_max"])]).astype(
            np.float32
        )
        truth = exact_numpy(pts.astype(float), config).reshape(xx.shape)
        predictions = []
        for arm in W8_ARMS:
            state = w8_make_state(config, arm, 3234)
            state[0].load_weights(
                Path(root) / "postprocess/trials" / ("seed_3234_" + arm) / "selected.weights.h5"
            )
            predictions.append(state[0](pts, training=False).numpy().reshape(xx.shape))
        limit = max(np.abs(truth).max(), max((np.abs(p).max() for p in predictions)))
        error = max((np.abs(p - truth).max() for p in predictions))
        for arm, pred in zip(W8_ARMS, predictions):
            fig, axes = plt.subplots(1, 3, figsize=(10, 3), constrained_layout=True)
            for i, (ax, v, title) in enumerate(
                zip(axes, [truth, pred, abs(pred - truth)], ["exact", "selected", "absolute error"])
            ):
                im = ax.imshow(
                    v.T,
                    origin="lower",
                    extent=(x[0], x[-1], y[0], y[-1]),
                    vmin=0 if i == 2 else -limit,
                    vmax=error if i == 2 else limit,
                    cmap="magma" if i == 2 else "coolwarm",
                )
                ax.set_title(title)
                fig.colorbar(im, ax=ax)
            fig.suptitle(arm + "; seed3234; final time; final iterations may differ")
            for ext in ("png", "pdf"):
                fig.savefig(folder / (arm + "_selected_field." + ext), dpi=150)
            plt.close(fig)

    def w8_run_trial(root, config, source, protocol, arm, seed, cap):
        if cap not in (5100, 10000, 100000):
            raise ValueError("FF budgets: 5100/10000/100000")
        root = Path(root)
        root.mkdir(parents=True, exist_ok=False)
        state = w8_make_state(config, arm, seed)
        model, optimizer, factors, updates, ckpt, variables, network = state
        init_digest, ff_digest = (w8_model_digest(model), w8_ff_digest(model))
        step = periodic_stepper(model, optimizer, variables, network)
        arrays, evidence = training_lhs(source, config, protocol, "volume_lhs", seed, 0)
        tensors = arrays_to_tensors(arrays)
        history, samples, adaptive = ([], [evidence], [])
        started = time.perf_counter()

        def record(iteration):
            row = {
                "iteration": iteration,
                "cycle": sampling_cycle(iteration),
                "phase": "warmup" if iteration <= 2000 else "joint",
                "reference_metrics": field_metrics(model, source),
                "validation_metrics": validation_field_metrics(model, tensors["val"]),
                "losses": {s: loss_record(model, tensors[s]) for s in ("train", "val", "test")},
                "lambda_1": float(model.lambda_1.numpy()),
                "elapsed_seconds": time.perf_counter() - started,
                "optimizer_iterations": int(optimizer.iterations.numpy()),
                "sa_updates": int(updates.numpy()),
                "factors": factors.numpy().tolist(),
                "effective_weights": w8_weights(arm, iteration, factors.numpy()).tolist(),
            }
            finite = [
                *row["reference_metrics"].values(),
                *row["validation_metrics"].values(),
                row["lambda_1"],
            ]
            finite += [v for values in row["losses"].values() for v in values.values()]
            if not np.isfinite(finite).all():
                raise RuntimeError("Nonfinite evaluation; trial is not complete")
            if iteration in (0, 2000, 2001, 5001, cap) or w8_target(history + [row]):
                observed_at = time.perf_counter()
                row["ff_diagnostics"] = ff_observe(state, tensors["train"])
                row["ff_diagnostics"]["seconds"] = time.perf_counter() - observed_at
            history.append(row)
            row["active_train_objective_after_update"] = float(
                sum(
                    (
                        row["effective_weights"][j] * row["losses"]["train"][key]
                        for j, key in enumerate(TERM_NAMES)
                    )
                )
            )
            row["target_reached"] = w8_target(history)
            print(
                f"{arm} seed={seed} iter={iteration} cycle={row['cycle']} valL2={row['validation_metrics']['relative_l2']:.5f} refL2={row['reference_metrics']['relative_l2']:.5f} lambda={row['lambda_1']:.6f}",
                flush=True,
            )
            return row["target_reached"]

        record(0)
        for iteration in range(1, cap + 1):
            cycle = sampling_cycle(iteration)
            if cycle != evidence["cycle"]:
                arrays, evidence = training_lhs(source, config, protocol, "volume_lhs", seed, cycle)
                tensors = arrays_to_tensors(arrays)
                samples.append(evidence)
                print(
                    "All-training LHS renewed before update",
                    iteration,
                    "; cycle",
                    cycle,
                    flush=True,
                )
            if sa_update_due(iteration, W8_ARMS[arm]["sa"]):
                norms = term_norms(model, network, tensors["train"])
                factors.assign(balanced_factors(factors.numpy(), norms).astype(np.float32))
                updates.assign_add(1)
                adaptive.append(
                    {
                        "before_update": iteration,
                        "cycle": cycle,
                        "norms": norms.tolist(),
                        "factors": factors.numpy().tolist(),
                    }
                )
            warming = iteration <= 2000
            step(
                tensors["train"],
                tf.constant(w8_weights(arm, iteration, factors.numpy()), tf.float32),
                warming,
            )
            if warming and float(model.lambda_1.numpy()) != 0.5:
                raise RuntimeError("Warmup coefficient changed")
            if iteration % 1000 == 0 or iteration in (2001, 5001, cap):
                reached = record(iteration)
                if reached and cap == 100000:
                    break
        if not W8_ARMS[arm]["sa"] and (
            int(updates.numpy()) or not np.array_equal(factors.numpy(), np.ones(4))
        ):
            raise RuntimeError("SA-off method acquired adaptive weights")
        ckpt.write(str(root / "state"))
        model.save_weights(root / "last.weights.h5")
        np.savez_compressed(root / "current_training_data.npz", **arrays)
        report = {
            "complete": True,
            "arm": arm,
            "sampling_seed": seed,
            "network_seed": seed,
            "cap": cap,
            "initialization_sha256": init_digest,
            "ff_sha256": ff_digest,
            "history": history,
            "sampling": samples,
            "adaptive_updates": adaptive,
            "last": history[-1],
            "stop_reason": (
                "validation_target"
                if cap == 100000 and w8_target(history)
                else (
                    "smoke_cap"
                    if cap == 5100
                    else "pilot_cap" if cap == 10000 else "safety_cap_goal_unmet"
                )
            ),
        }
        write_json(root / "result.json", report)
        return report

    "Training-only fixed three-group normalization; physical derivatives unchanged."
    import copy
    import json
    import numpy as np
    from pathlib import Path

    NORMALIZATION_POLICY = {
        "version": 1,
        "groups": ["physics", "data", "boundary_data"],
        "group_weights": [1 / 3, 1 / 3, 1 / 3],
        "boundary_data": "mean(IC, BC); IC=mean(displacement, velocity)",
        "input": "existing differentiable affine map of physical xyz/t to [-1,1]",
        "physical_scales": "U=RMS pooled initial/boundary/supervised training labels; T=domain duration; residual scale=U/T^2",
        "calibration": "group dimensionless MSE at initial model and cycle-0 training arrays, once per paired seed",
        "floor": 1e-08,
        "adaptive": False,
        "heldout_used": False,
        "physical_PDE_changed": False,
        "output_units_changed": False,
    }
    _physical_losses = calculate_losses_2d
    _unscaled_lb_measure = lb_measure
    _norm_context = None
    _norm_bank = {}

    def configure_normalization(root, config, source, protocol, allow_new):
        nonlocal _norm_context, _norm_bank
        root = Path(root)
        _norm_context = (root, config, source, protocol, allow_new)
        path = root / "normalization.json"
        stored = (
            json.loads(path.read_text())
            if path.exists()
            else {"policy": NORMALIZATION_POLICY, "seeds": {}}
        )
        if stored["policy"] != NORMALIZATION_POLICY:
            raise ValueError("Normalization policy mismatch")
        _norm_bank = stored["seeds"]

    def dimensionless_terms(raw, amplitude, time_scale):
        u2 = amplitude**2
        displacement = raw["initial_displacement"] / u2
        velocity = raw["initial_velocity"] / (amplitude / time_scale) ** 2
        return {
            "initial_displacement": displacement,
            "initial_velocity": velocity,
            "initial": 0.5 * (displacement + velocity),
            "boundary": raw["boundary"] / u2,
            "physics": raw["physics"] / (amplitude / time_scale**2) ** 2,
            "supervised": raw["supervised"] / u2,
        }

    def fixed_group_scales(terms):
        values = {
            "physics": float(terms["physics"]),
            "data": float(terms["supervised"]),
            "boundary_data": float(0.5 * (terms["initial"] + terms["boundary"])),
        }
        if not np.isfinite(list(values.values())).all() or min(values.values()) < 0:
            raise ValueError("Invalid normalization calibration")
        return {k: max(v, NORMALIZATION_POLICY["floor"]) for k, v in values.items()}

    def normalize_terms(raw, record):
        d = dimensionless_terms(raw, record["amplitude"], record["time_scale"])
        scales = record["group_scales"]
        out = {
            k: v / scales["boundary_data"]
            for k, v in d.items()
            if k.startswith("initial") or k == "boundary"
        }
        out["physics"] = d["physics"] / scales["physics"]
        out["supervised"] = d["supervised"] / scales["data"]
        out["total"] = (
            0.5 * (out["initial"] + out["boundary"]) + out["physics"] + out["supervised"]
        ) / 3.0
        return out

    def attach_normalization(model, seed):
        if _norm_context is None:
            raise RuntimeError("T3 normalization context is required")
        root, config, source, protocol, allow_new = _norm_context
        arrays, _ = training_lhs(source, config, protocol, "volume_lhs", seed, 0)
        train = {k: v for k, v in arrays.items() if k.endswith("_train")}
        identity = {
            "seed": seed,
            "training_arrays_sha256": arrays_digest(train),
            "initialization_sha256": w8_model_digest(model),
        }
        key = str(seed) + "_" + model.normalization_family
        if key not in _norm_bank:
            if not allow_new:
                raise ValueError("Stored normalization missing; no recalibration during review")
            labels = np.concatenate(
                [
                    train["u_" + family + "_train"].astype(float).ravel()
                    for family in ("initial", "boundary", "supervised")
                ]
            )
            amplitude = float(np.sqrt(np.mean(labels**2)))
            duration = float(config["domain"]["t_max"] - config["domain"]["t_min"])
            if not np.isfinite([amplitude, duration]).all() or amplitude <= 1e-12 or duration <= 0:
                raise ValueError(
                    "Physical scales require nonzero training amplitude and positive duration"
                )
            before = [v.numpy().copy() for v in model.variables]
            raw = _physical_losses(model, arrays_to_tensors(arrays)["train"], training=False)
            raw = {k: float(v.numpy()) for k, v in raw.items()}
            for a, b in zip(before, model.variables):
                np.testing.assert_array_equal(a, b.numpy())
            terms = dimensionless_terms(raw, amplitude, duration)
            record = {
                **identity,
                "amplitude": amplitude,
                "time_scale": duration,
                "group_scales": fixed_group_scales(terms),
                "initial_raw_losses": raw,
                "calibration_cycle": 0,
                "heldout_used": False,
                "frozen": True,
            }
            _norm_bank[key] = record
            write_json(
                root / "normalization.json", {"policy": NORMALIZATION_POLICY, "seeds": _norm_bank}
            )
        record = _norm_bank[key]
        if any((record[k] != v for k, v in identity.items())):
            raise ValueError("Normalization identity mismatch")
        scales = record["group_scales"]
        if (
            set(scales) != set(NORMALIZATION_POLICY["groups"])
            or not np.isfinite(list(scales.values())).all()
            or min(scales.values()) <= 0
        ):
            raise ValueError("Invalid stored normalization scales")
        if (
            not np.isfinite([record["amplitude"], record["time_scale"]]).all()
            or min(record["amplitude"], record["time_scale"]) <= 0
        ):
            raise ValueError("Invalid stored physical scales")
        model.loss_normalization = copy.deepcopy(record)

    def calculate_losses_2d(model, data, training=False):
        raw = _physical_losses(model, data, training=training)
        record = getattr(model, "loss_normalization", None)
        return raw if record is None else normalize_terms(raw, record)

    def physical_loss_record(model, data):
        return {
            k: float(v.numpy()) for k, v in _physical_losses(model, data, training=False).items()
        }

    def lb_measure(state, arrays, source, weights):
        before = state_values(state)
        result = _unscaled_lb_measure(state, arrays, source, weights)
        tensors = arrays_to_tensors(arrays)
        result["physical_losses"] = {
            s: physical_loss_record(state[0], tensors[s]) for s in ("train", "val", "test")
        }
        result["normalization_scales"] = copy.deepcopy(state[0].loss_normalization["group_scales"])
        result["group_losses"] = {
            s: {
                "physics": d["physics"],
                "data": d["supervised"],
                "boundary_data": 0.5 * (d["initial"] + d["boundary"]),
            }
            for s, d in result["losses"].items()
        }
        assert_same_state(before, state_values(state))
        return result

    CROSS_CODE_SHA256 = "1f1b147ec669ec1322cb9671be016d444f3b60031a097ca7982fcf199bef3877"
    "Five-arm Wave inverse experiment; injected after FF and normalization helpers."
    import time
    import tempfile

    CROSS_POLICY = {
        "version": 1,
        "base_lr": 0.004,
        "lr_provenance": "thesis section 3.3.3 printed p59; Table 3.4 p56 differs",
        "network_ratios": [1.0, 0.75, 0.5],
        "lambda_ratios": [0.001, 0.75, 1.0],
        "knots": [0, 5000, 10000],
        "after_last_knot": "constant; no additional decay",
        "iteration_convention": "LR for update n uses completed updates n-1",
        "initial_lambda": 0.5,
        "resample_period": 5000,
        "curriculum": "PDE=floor(completed/100)/300 capped at 1/3; remainder split equally",
        "curriculum_lambda_note": "tracked from start; zero physics gradient for updates 1..100",
        "sa": "3 normalized semantic groups; shared-network gradient-norm inverse balancing",
        "sa_every": 50,
        "sa_ema": 0.9,
        "sa_factor_bounds": [0.25, 4.0],
        "target": 0.05,
        "first_target_check": 15000,
        "target_every": 1000,
        "target_checks": 3,
        "cpu_validation": "final candidates only",
        "automatic_promotion": False,
    }
    _cross_factory = w8_make_state
    W8_ARMS = {
        "baseline": ff_spec("plain"),
        "loss_sa": ff_spec("plain", sa=True),
        "ff": ff_spec("spacetime", scales=(0.125, 0.25, 0.5)),
        "ff_loss_sa": ff_spec("spacetime", scales=(0.125, 0.25, 0.5), sa=True),
        "ff_curriculum": {**ff_spec("spacetime", scales=(0.125, 0.25, 0.5)), "curriculum": True},
    }

    def cross_lr(completed):
        if completed < 0:
            raise ValueError("Negative iteration")
        return tuple(
            (
                float(np.interp(completed, [0, 5000, 10000], v) * 0.004)
                for v in ([1.0, 0.75, 0.5], [0.001, 0.75, 1.0])
            )
        )

    def cross_groups(arm, completed, factors):
        if arm not in W8_ARMS or completed < 0:
            raise ValueError("Unknown arm/iteration")
        f = np.asarray(factors, dtype=float)
        if f.shape != (3,) or not np.isfinite(f).all() or np.any(f <= 0):
            raise ValueError("Invalid SA factors")
        spec = W8_ARMS[arm]
        if not spec["sa"] and (not np.array_equal(f, np.ones(3))):
            raise ValueError("SA leakage")
        if spec["curriculum"]:
            p = min(completed // 100, 100) / 300.0
            return np.array([p, (1 - p) / 2, (1 - p) / 2])
        return f / f.sum() if spec["sa"] else np.ones(3) / 3

    def w8_weights(arm, iteration, factors):
        p, d, b = cross_groups(arm, iteration, factors)
        return np.array([b / 2, b / 2, p, d])

    def cross_target(history):
        rows = [r for r in history if r["iteration"] >= 15000 and r["iteration"] % 1000 == 0]
        if len(rows) < 3:
            return False
        rows = rows[-3:]
        return (
            rows[1]["iteration"] - rows[0]["iteration"] == 1000
            and rows[2]["iteration"] - rows[1]["iteration"] == 1000
            and all((r["validation_metrics"]["relative_l2"] <= 0.05 for r in rows))
        )

    def w8_make_state(config, arm, seed):
        old = _cross_factory(config, arm, seed)
        model, _, _, _, _, variables, network = old
        model.lambda_1.assign(0.5)
        model.normalization_family = W8_ARMS[arm]["kind"]
        attach_normalization(model, seed)
        optimizer = tf.keras.optimizers.Adam(0.004)
        optimizer.build(network)
        optimizer.lambda_optimizer = tf.keras.optimizers.Adam(4e-06)
        optimizer.lambda_optimizer.build([model.lambda_1])
        factors = tf.Variable(np.ones(3, np.float32), trainable=False)
        count = tf.Variable(0, dtype=tf.int64, trainable=False)
        ckpt = tf.train.Checkpoint(
            model=model,
            optimizer=optimizer,
            lambda_optimizer=optimizer.lambda_optimizer,
            factors=factors,
            sa_updates=count,
        )
        return (model, optimizer, factors, count, ckpt, variables, network)

    def state_values(state):
        return [
            v.numpy().copy()
            for v in list(state[0].variables)
            + list(state[1].variables)
            + list(state[1].lambda_optimizer.variables)
            + [state[2], state[3]]
        ]

    def lb_frozen_state(state):
        return [
            v.numpy().copy()
            for v in list(state[1].variables)
            + list(state[1].lambda_optimizer.variables)
            + [state[2], state[3]]
        ]

    def cross_stepper(state):
        model, opt, _, _, _, variables, network = state
        indices = [next((i for i, v in enumerate(variables) if v is n)) for n in network]
        li = next((i for i, v in enumerate(variables) if v is model.lambda_1))

        @tf.function(autograph=False)
        def step(data, weights):
            with tf.GradientTape() as tape:
                terms = calculate_losses_2d(model, data, training=True)
                loss = tf.add_n([weights[i] * terms[k] for i, k in enumerate(TERM_NAMES)])
            grads = tape.gradient(loss, variables)
            for g in grads:
                if g is None:
                    raise RuntimeError("Disconnected trainable variable")
                tf.debugging.assert_all_finite(g, "Nonfinite gradient")
            opt.apply_gradients([(grads[i], variables[i]) for i in indices])
            opt.lambda_optimizer.apply_gradients([(grads[li], model.lambda_1)])
            return loss

        return step

    def cross_sa(state, data):
        before = state_values(state)
        with tf.GradientTape(persistent=True) as tape:
            d = calculate_losses_2d(state[0], data, training=False)
            groups = [d["physics"], d["supervised"], 0.5 * (d["initial"] + d["boundary"])]
        norms = []
        for loss in groups:
            gs = tape.gradient(loss, state[6])
            norms.append(float(tf.linalg.global_norm([g for g in gs if g is not None]).numpy()))
        del tape
        assert_same_state(before, state_values(state))
        norms = np.asarray(norms)
        if not np.isfinite(norms).all():
            raise RuntimeError("Nonfinite SA probe")
        desired = np.clip(np.mean(norms) / np.maximum(norms, 1e-12), 0.25, 4.0)
        state[2].assign((0.9 * state[2].numpy() + 0.1 * desired).astype(np.float32))
        state[3].assign_add(1)
        return norms.tolist()

    def cross_row(state, arrays, source, arm, iteration):
        before = state_values(state)
        row = lb_measure(state, arrays, source, w8_weights(arm, iteration, state[2].numpy()))
        row.update(
            iteration=iteration,
            cycle=sampling_cycle(iteration),
            optimizer_iterations=int(state[1].iterations.numpy()),
            lambda_optimizer_iterations=int(state[1].lambda_optimizer.iterations.numpy()),
            factors=state[2].numpy().tolist(),
            sa_updates=int(state[3].numpy()),
            next_lrs=list(cross_lr(iteration)),
            next_group_weights=cross_groups(arm, iteration, state[2].numpy()).tolist(),
            effective_weights=w8_weights(arm, iteration, state[2].numpy()).tolist(),
        )
        assert_same_state(before, state_values(state))
        return row

    def cross_save(root, state, arrays, report):
        state[4].write(str(root / "state"))
        state[0].save_weights(root / "last.weights.h5")
        np.savez_compressed(root / "current_training_data.npz", **arrays)
        report["loss_normalization"] = copy.deepcopy(state[0].loss_normalization)
        write_json(root / "result.json", report)

    def cross_trial(root, config, source, protocol, arm, seed, cap):
        root.mkdir(parents=True, exist_ok=False)
        state = w8_make_state(config, arm, seed)
        step = cross_stepper(state)
        arrays, evidence = training_lhs(source, config, protocol, "volume_lhs", seed, 0)
        protected = evidence["protected_sha256"]
        data = arrays_to_tensors(arrays)["train"]
        report = {
            "complete": False,
            "arm": arm,
            "sampling_seed": seed,
            "network_seed": seed,
            "cap": cap,
            "initialization_sha256": w8_model_digest(state[0]),
            "ff_sha256": w8_ff_digest(state[0]),
            "trainable_parameters": sum((int(np.prod(v.shape)) for v in state[5])),
            "history": [],
            "sampling": [evidence],
            "adaptive_updates": [],
        }
        row = cross_row(state, arrays, source, arm, 0)
        row["target_reached"] = False
        report["history"].append(row)
        start = time.perf_counter()
        for it in range(1, cap + 1):
            cycle = sampling_cycle(it)
            if cycle != evidence["cycle"]:
                arrays, evidence = training_lhs(source, config, protocol, "volume_lhs", seed, cycle)
                if evidence["protected_sha256"] != protected:
                    raise RuntimeError("Heldout changed")
                report["sampling"].append(evidence)
                data = arrays_to_tensors(arrays)["train"]
            a, b = cross_lr(it - 1)
            state[1].learning_rate.assign(a)
            state[1].lambda_optimizer.learning_rate.assign(b)
            step(data, tf.constant(w8_weights(arm, it - 1, state[2].numpy()), tf.float32))
            if W8_ARMS[arm]["sa"] and it % 50 == 0:
                norms = cross_sa(state, data)
                report["adaptive_updates"].append(
                    {"iteration": it, "norms": norms, "factors": state[2].numpy().tolist()}
                )
            if it % 1000 == 0 or it in (1, 100, 101, 5001, 10001, 15001) or it == cap:
                row = cross_row(state, arrays, source, arm, it)
                row["last_lrs"] = [a, b]
                row["elapsed_seconds"] = time.perf_counter() - start
                report["history"].append(row)
                row["target_reached"] = cross_target(report["history"])
                if it % 1000 == 0 or it == cap:
                    print(
                        seed,
                        arm,
                        it,
                        "val L2",
                        row["validation_metrics"]["relative_l2"],
                        flush=True,
                    )
                if row["target_reached"]:
                    break
        report.update(
            complete=True,
            last=report["history"][-1],
            stop_reason=(
                "validation_target" if report["history"][-1]["target_reached"] else "cap_reached"
            ),
        )
        cross_save(root, state, arrays, report)
        return report

    def w8_review_trial(root, config, source, protocol, arm, seed, cap):
        root = Path(root)
        r = json.loads((root / "result.json").read_text())
        if not r["complete"] or (r["arm"], r["sampling_seed"], r["cap"]) != (arm, seed, cap):
            raise ValueError("Trial identity mismatch")
        state = w8_make_state(config, arm, seed)
        ff = w8_ff_digest(state[0])
        state[4].read(str(root / "state")).assert_consumed()
        checkpoint_weights_agree(state, root)
        if ff != w8_ff_digest(state[0]) or ff != r["ff_sha256"]:
            raise ValueError("FF changed")
        if state[0].loss_normalization != r["loss_normalization"]:
            raise ValueError("Normalization changed")
        it = r["last"]["iteration"]
        if not 0 < it <= cap or r["history"][-1] != r["last"]:
            raise ValueError("Invalid completed iteration/history")
        if it < cap and (not cross_target(r["history"])):
            raise ValueError("Invalid early termination")
        if (
            int(state[1].iterations.numpy()) != it
            or int(state[1].lambda_optimizer.iterations.numpy()) != it
        ):
            raise ValueError("Optimizer counters mismatch")
        if int(state[3].numpy()) != (it // 50 if W8_ARMS[arm]["sa"] else 0):
            raise ValueError("SA schedule mismatch")
        with np.load(root / "current_training_data.npz", allow_pickle=False) as a:
            arrays = {k: a[k] for k in a.files}
        expected, e = training_lhs(source, config, protocol, "volume_lhs", seed, sampling_cycle(it))
        if arrays_digest(expected) != arrays_digest(arrays) or e != r["sampling"][-1]:
            raise ValueError("Saved arrays differ")
        measured = cross_row(state, arrays, source, arm, it)
        if not np.isclose(measured["lambda_1"], r["last"]["lambda_1"], rtol=1e-06, atol=1e-07):
            raise ValueError("Lambda reload differs")
        for row in r["history"]:
            if row["next_lrs"] != list(cross_lr(row["iteration"])):
                raise ValueError("LR schedule record differs")
            if (
                row["next_group_weights"]
                != cross_groups(arm, row["iteration"], row["factors"]).tolist()
            ):
                raise ValueError("Group weight record differs")
        for k in ("reference_metrics", "validation_metrics", "physical_losses", "losses"):

            def compare(a, b):
                if isinstance(a, dict):
                    return set(a) == set(b) and all((compare(a[x], b[x]) for x in a))
                return bool(np.isclose(a, b, rtol=0.001, atol=1e-06))

            if not compare(measured[k], r["last"][k]):
                raise ValueError("Reload metrics differ: " + k)
        if cross_target(r["history"]) != r["last"]["target_reached"]:
            raise ValueError("Stopping evidence mismatch")
        return {
            "passed": True,
            "arm": arm,
            "seed": seed,
            "iteration": it,
            "two_optimizers_restored": True,
            "cpu_runtime": not bool(tf.config.list_physical_devices("GPU")),
        }

    def cross_wiring(config, source, protocol):
        a, ea = training_lhs(source, config, protocol, "volume_lhs", 3234, 0)
        b, eb = training_lhs(source, config, protocol, "volume_lhs", 3234, 1)
        assert ea["protected_sha256"] == eb["protected_sha256"]
        for f in ("initial", "boundary", "supervised", "collocation"):
            assert not np.array_equal(a["X_" + f + "_train"], b["X_" + f + "_train"])
        data = {k: v[:8] for k, v in arrays_to_tensors(a)["train"].items()}
        initial = {}
        checks = []
        for arm in W8_ARMS:
            state = w8_make_state(config, arm, 3234)
            initial[arm] = w8_model_digest(state[0])
            step = cross_stepper(state)
            start = float(state[0].lambda_1.numpy())
            step(data, tf.constant(w8_weights(arm, 0, np.ones(3)), tf.float32))
            if W8_ARMS[arm]["curriculum"]:
                assert float(state[0].lambda_1.numpy()) == start
            else:
                assert float(state[0].lambda_1.numpy()) != start
            if W8_ARMS[arm]["sa"]:
                cross_sa(state, data)
            state[1].learning_rate.assign(0.003)
            state[1].lambda_optimizer.learning_rate.assign(0.003)
            with tempfile.TemporaryDirectory() as tmp:
                state[4].write(str(Path(tmp) / "state"))
                clone = w8_make_state(config, arm, 3234)
                clone[4].read(str(Path(tmp) / "state")).assert_consumed()
                assert_same_state(state_values(state), state_values(clone))
                weights = tf.constant(w8_weights(arm, 5000, state[2].numpy()), tf.float32)
                step(data, weights)
                cross_stepper(clone)(data, weights)
                assert_same_state(state_values(state), state_values(clone), approximate=True)
            checks.append({"arm": arm, "two_optimizer_roundtrip": True, "same_next_update": True})
        assert initial["baseline"] == initial["loss_sa"]
        assert initial["ff"] == initial["ff_loss_sa"] == initial["ff_curriculum"]
        return {
            "passed": True,
            "checks": checks,
            "all_training_families_renew": True,
            "heldout_fixed": True,
            "scope": "disposable probes, not trained performance",
        }

    import platform

    SOURCE_DIR = Path(tempfile.gettempdir()) / ("periodic_source_" + uuid.uuid4().hex)
    wave2d_config, SOURCE_RESULTS, SOURCE_EVIDENCE = load_stage7_bundle(BUNDLE, SOURCE_DIR)
    with np.load(SOURCE_DIR / "training/wave2d_data.npz", allow_pickle=False) as archive:
        SOURCE_ARRAYS = {k: archive[k] for k in archive.files}
    if not data_audit(SOURCE_ARRAYS, wave2d_config)["passed"]:
        raise RuntimeError("Source data audit failed")
    method_comparison_cfg = wave2d_config["method_comparison"]
    wave2d_weights = {
        k: v / sum(wave2d_config["loss_weights"].values())
        for k, v in wave2d_config["loss_weights"].items()
    }
    baseline = next((r for r in SOURCE_RESULTS["results"] if r["method"] == "baseline"))
    INITIAL_PATH = SOURCE_DIR / baseline["initialization"]["weights"]
    PROTOCOL = resolve_sector_design(default_sector_design(), wave2d_config)
    if settings is not None:
        CROSS_POLICY["target"] = settings["field_target"]
        _configured_wave_factory = w8_make_state

        def w8_make_state(config, arm, seed):
            state = _configured_wave_factory(config, arm, seed)
            state[0].lambda_1.assign(settings["initial_coefficient_ratio"])
            return state

        def cross_lr(completed):
            return tuple(
                float(np.interp(completed, settings["lr_knots"], settings[k]) * settings["base_lr"])
                for k in ["network_ratios", "coefficient_ratios"]
            )

        def sampling_cycle(iteration):
            return max(0, (iteration - 1) // settings["lhs_period"])

        def cross_target(history):
            n = settings["stop_consecutive"]
            every = settings["stop_every"]
            eligible = [
                r
                for r in history
                if r["iteration"] >= settings["stop_first"] and r["iteration"] % every == 0
            ][-n:]
            return (
                len(eligible) == n
                and all(
                    b["iteration"] - a["iteration"] == every for a, b in zip(eligible, eligible[1:])
                )
                and all(
                    r["validation_metrics"]["relative_l2"] <= settings["field_target"]
                    for r in eligible
                )
            )

    return locals()
