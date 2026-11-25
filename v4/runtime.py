"""Lazy TensorFlow PINN, separate optimizers, observer and frozen L-BFGS."""

import numpy as np
import random

from .config import group_weights, learning_rates
from .equations import EQUATIONS, residual


class State(list):
    """Model/optimizer state plus observer-only RNG guards, never a checkpoint."""

    pass


def load_tensorflow():
    try:
        import tensorflow as tf

        if not hasattr(tf, "GradientTape"):
            raise ImportError("incomplete TensorFlow package")
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "TensorFlow를 사용할 수 없습니다. 새 런타임에서 tensorflow 설치 후 다시 실행하세요. 버전 일치는 요구하지 않습니다."
        ) from exc
    for device in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(device, True)
        except RuntimeError:
            pass  # A caller may already have initialized this device.
    return tf


class Runtime:
    def __init__(self, settings, arm, source):
        self.tf = tf = load_tensorflow()
        self.settings = settings
        self.arm = arm
        self.source = source
        self.scales = source.scales()
        spec = EQUATIONS[settings.equation]
        self.inverse = settings.inverse and arm != "supervised_only"
        initial = (
            settings.initial_coefficient
            if settings.initial_coefficient is not None
            else spec["initial"]
        )
        domain = np.array(source.domain, np.float32)
        field = np.array(self.scales["field"], np.float32)

        class PINN(tf.keras.Model):
            def __init__(self):
                super().__init__()
                self.log_coefficient = self.add_weight(
                    name="log_coefficient",
                    shape=(),
                    initializer=tf.keras.initializers.Constant(
                        np.log(
                            initial if settings.inverse else source.fixed_coefficient
                        )
                    ),
                    trainable=self_inverse,
                )
                self.banks = []
                rng = np.random.default_rng(4045)
                if arm.startswith("ff"):
                    for i, scale in enumerate(settings.ff_scales):
                        bank = rng.normal(
                            0, scale, (len(domain), settings.ff_features)
                        ).astype(np.float32)
                        self.banks.append(
                            self.add_weight(
                                name=f"ff_bank_{i}",
                                shape=bank.shape,
                                initializer=tf.keras.initializers.Constant(bank),
                                trainable=False,
                            )
                        )
                self.hidden = [
                    tf.keras.layers.Dense(
                        settings.width,
                        activation="tanh",
                        kernel_initializer=tf.keras.initializers.GlorotNormal(3234 + i),
                    )
                    for i in range(settings.depth)
                ]
                self.output_layer = tf.keras.layers.Dense(
                    len(spec["outputs"]),
                    kernel_initializer=tf.keras.initializers.GlorotNormal(
                        3234 + settings.depth
                    ),
                )

            def call(self, x, training=False):
                z = (
                    2
                    * (tf.cast(x, tf.float32) - domain[:, 0])
                    / (domain[:, 1] - domain[:, 0])
                    - 1
                )
                if self.banks:
                    phases = [2 * np.pi * tf.matmul(z, b) for b in self.banks]
                    z = tf.concat(
                        [f(p) for p in phases for f in (tf.sin, tf.cos)], axis=1
                    )
                for layer in self.hidden:
                    z = layer(z)
                return self.output_layer(z) * field

        self_inverse = self.inverse
        self.model = PINN()
        self.model(tf.zeros((1, len(domain))))
        self.network = [
            v
            for v in self.model.trainable_variables
            if v is not self.model.log_coefficient
        ]
        self.opt = tf.keras.optimizers.Adam(settings.base_lr)
        self.opt.build(self.network)
        self.coefficient_opt = tf.keras.optimizers.Adam(settings.base_lr * 0.001)
        self.coefficient_opt.build([self.model.log_coefficient])
        self.factors = np.ones(3, np.float32)
        self.sa_updates = 0
        self._step = tf.function(self._update, autograph=False, reduce_retracing=True)

    @property
    def coefficient(self):
        return (
            self.tf.exp(self.model.log_coefficient)
            if self.inverse
            else self.tf.constant(self.source.fixed_coefficient, self.tf.float32)
        )

    def derivatives(self, x, model=None):
        tf = self.tf
        x = tf.convert_to_tensor(x, tf.float32)
        model = model or self.model
        with tf.GradientTape(persistent=True) as outer:
            outer.watch(x)
            with tf.GradientTape(persistent=True) as inner:
                inner.watch(x)
                q = model(x)
                components = tf.unstack(q, axis=1)
            g = [
                inner.gradient(c, x, unconnected_gradients=tf.UnconnectedGradients.ZERO)
                for c in components
            ]
            parts = [tf.unstack(v, axis=1) for v in g]
        h = tf.stack(
            [
                tf.stack(
                    [
                        outer.gradient(
                            c, x, unconnected_gradients=tf.UnconnectedGradients.ZERO
                        )
                        for c in row
                    ],
                    axis=1,
                )
                for row in parts
            ],
            axis=1,
        )
        del inner, outer
        return q, tf.stack(g, axis=1), h

    def pde_residual(self, x, coefficient=None):
        q, g, h = self.derivatives(x)
        f = self.tf.constant(self.source.forcing(np.asarray(x)), self.tf.float32)
        return residual(
            self.settings.equation,
            self.tf.convert_to_tensor(x, self.tf.float32),
            q,
            g,
            h,
            self.coefficient if coefficient is None else coefficient,
            f,
            self.tf,
        )

    def tensor_data(self, arrays):
        return {
            k: self.tf.constant(np.nan_to_num(v), self.tf.float32)
            for k, v in arrays.items()
        }

    def prepare(self, arrays):
        data = dict(arrays)
        for key in list(arrays):
            if key.startswith("label_"):
                data["mask_" + key[6:]] = np.isfinite(arrays[key]).astype(np.float32)
        data["forcing"] = self.source.forcing(arrays["pde"])
        return self.tensor_data(data)

    def groups(self, a, include_physics=True):
        tf = self.tf
        fs = tf.constant(self.scales["field"], tf.float32)

        def observed(key, components=None):
            pred = self.model(a[key])
            label = a["label_" + key]
            mask = a["mask_" + key]
            if components is not None:
                pred = pred[:, :components]
                label = label[:, :components]
                mask = mask[:, :components]
                scale = fs[:components]
            else:
                scale = fs
            return tf.reduce_sum(tf.square((pred - label) / scale) * mask) / tf.maximum(
                tf.reduce_sum(mask), 1.0
            )

        if include_physics:
            q, g, h = self.derivatives(a["pde"])
            r = residual(
                self.settings.equation,
                a["pde"],
                q,
                g,
                h,
                self.coefficient,
                a["forcing"],
                tf,
            )
            physics = tf.reduce_mean(
                tf.square(r / tf.constant(self.scales["residual"], tf.float32))
            )
        else:
            physics = tf.constant(0.0, tf.float32)
        data = observed("observations")
        terms = []
        if "boundary" in a:
            terms.append(
                observed("boundary", 2 if len(self.scales["field"]) == 3 else None)
            )
        if "initial" in a:
            terms.append(observed("initial"))
        if "initial_velocity" in a:
            _, g, _ = self.derivatives(a.get("initial_velocity_x", a["initial"]))
            time_axis = EQUATIONS[self.settings.equation]["time_axis"]
            duration = np.ptp(np.asarray(self.source.domain)[time_axis])
            terms.append(
                tf.reduce_mean(
                    tf.square(
                        (g[:, 0, time_axis : time_axis + 1] - a["initial_velocity"])
                        * duration
                        / fs[0]
                    )
                )
            )
        if "gauge" in a:
            pred = self.model(a["gauge"])[:, 2:3]
            terms.append(
                tf.reduce_mean(tf.square((pred - a["label_gauge"][:, 2:3]) / fs[2]))
            )
        constraint = (
            tf.reduce_mean(tf.stack(terms)) if terms else tf.constant(0.0, tf.float32)
        )
        return tf.stack([physics, data, constraint])

    def _update(self, data, weights):
        tf = self.tf
        variables = self.network + (
            [self.model.log_coefficient] if self.inverse else []
        )
        with tf.GradientTape() as tape:
            objective = tf.reduce_sum(
                self.groups(data, self.arm != "supervised_only") * weights
            )
        grads = tape.gradient(objective, variables)
        grads = [tf.zeros_like(v) if g is None else g for g, v in zip(grads, variables)]
        for g in grads:
            tf.debugging.assert_all_finite(g, "Nonfinite training gradient")
        tf.debugging.assert_all_finite(objective, "Nonfinite objective")
        self.opt.apply_gradients(zip(grads[: len(self.network)], self.network))
        if self.inverse:
            self.coefficient_opt.apply_gradients(
                [(grads[-1], self.model.log_coefficient)]
            )
            self.model.log_coefficient.assign(
                tf.clip_by_value(self.model.log_coefficient, -20.0, 20.0)
            )
        return objective

    def step(self, tensors, completed):
        net, coef = learning_rates(completed, self.settings.base_lr)
        self.opt.learning_rate.assign(net)
        self.coefficient_opt.learning_rate.assign(coef)
        return float(
            self._step(
                tensors,
                self.tf.constant(group_weights(self.arm, completed, self.factors)),
            )
        )

    def variables(self):
        return (
            list(self.model.variables)
            + list(self.opt.variables)
            + list(self.coefficient_opt.variables)
        )

    def snapshot(self):
        state = State(
            [v.numpy().copy() for v in self.variables()]
            + [self.factors.copy(), np.array(self.sa_updates)]
        )
        state.python_rng = random.getstate()
        state.numpy_rng = np.random.get_state()
        state.tf_generator = self.tf.random.get_global_generator().state.numpy().copy()
        return state

    def assert_unchanged(self, before):
        after = self.snapshot()
        if len(before) != len(after) or not all(
            np.array_equal(a, b) for a, b in zip(before, after)
        ):
            raise RuntimeError("Observer changed model/optimizer state")
        if isinstance(before, State):
            if (
                before.python_rng != after.python_rng
                or not all(
                    np.array_equal(a, b)
                    for a, b in zip(before.numpy_rng, after.numpy_rng)
                )
                or not np.array_equal(before.tf_generator, after.tf_generator)
            ):
                raise RuntimeError(
                    "Observer changed Python/NumPy/global TensorFlow Generator state"
                )

    def term_gradients(self, tensors, include_coefficient=True):
        tf = self.tf
        with tf.GradientTape(persistent=True) as tape:
            terms = tf.unstack(self.groups(tensors))
        gradients = [tape.gradient(term, self.network) for term in terms]
        gradients = [
            [tf.zeros_like(v) if g is None else g for g, v in zip(row, self.network)]
            for row in gradients
        ]
        coefficient = [
            (
                tape.gradient(term, self.model.log_coefficient)
                if include_coefficient
                else None
            )
            for term in terms
        ]
        del tape
        vectors = [
            np.concatenate([g.numpy().ravel() for g in row]) for row in gradients
        ]
        return (
            terms,
            vectors,
            [
                0.0 if g is None else float(g) / float(self.coefficient)
                for g in coefficient
            ],
        )

    def sa_update(self, tensors):
        if self.arm not in {"loss_sa", "ff_loss_sa"}:
            raise ValueError("SA 비교군에서만 SA를 갱신합니다.")
        # This read-only derivative path is covered by development tests. Full
        # model/optimizer/RNG snapshots remain in scheduled observations.
        _, vectors, _ = self.term_gradients(tensors, include_coefficient=False)
        norms = np.array([np.linalg.norm(v) for v in vectors])
        goal = np.clip(
            norms.mean() / np.maximum(norms, 1e-12), *self.settings.sa_bounds
        )
        if not np.isfinite(goal).all():
            raise ValueError("SA gradient invalid")
        self.factors = self.settings.sa_ema * self.factors + (
            1 - self.settings.sa_ema
        ) * goal.astype(np.float32)
        self.sa_updates += 1

    def observe(self, tensors, iteration):
        before = self.snapshot()
        terms, vectors, coef = self.term_gradients(tensors)
        norms = np.array([np.linalg.norm(v) for v in vectors])
        cosine = np.diag((norms > 1e-30).astype(float))
        for a in range(3):
            for b in range(a):
                cosine[a, b] = cosine[b, a] = float(
                    vectors[a] @ vectors[b] / max(norms[a] * norms[b], 1e-30)
                )
        frozen = None
        if self.inverse:
            scale = np.asarray(self.scales["residual"])
            intercept = self.pde_residual(tensors["pde"], 0.0).numpy() / scale
            slope = self.pde_residual(tensors["pde"], 1.0).numpy() / scale - intercept
            denominator = float(np.mean(slope * slope))
            optimum = (
                -float(np.mean(intercept * slope)) / denominator
                if denominator > 1e-20
                else None
            )
            analytic = float(
                2 * np.mean((intercept + float(self.coefficient) * slope) * slope)
            )
            gradient_matches = bool(np.isclose(analytic, coef[0], rtol=1e-3, atol=1e-5))
            if not gradient_matches or not np.allclose(coef[1:], 0.0, atol=1e-8):
                raise RuntimeError("계수 gradient 의존성 검사 실패")
            frozen = {
                "unconstrained_optimum": optimum,
                "positive_domain_infimum": (
                    max(0.0, optimum) if optimum is not None else None
                ),
                "analytic_gradient": analytic,
                "gradient_matches_autodiff": gradient_matches,
                "data_constraint_coefficient_independent": True,
                "field_frozen_only": True,
            }
        self.assert_unchanged(before)
        return {
            "iteration": iteration,
            "raw_normalized_groups": [float(v) for v in terms],
            "network_gradient_norms": norms.tolist(),
            "gradient_cosines": cosine.tolist(),
            "coefficient_term_gradients": coef,
            "coefficient": float(self.coefficient),
            "frozen_field_coefficient": frozen,
            "gradient_cosine_defined": (
                (norms > 1e-30)[:, None] & (norms > 1e-30)[None, :]
            ).tolist(),
            "rng_guard_scope": "Python, NumPy legacy RNG, TensorFlow global Generator; deterministic tanh model",
            "effective_weights": group_weights(
                self.arm, iteration, self.factors
            ).tolist(),
            "sa_factors": self.factors.tolist(),
            "next_learning_rates": list(
                learning_rates(iteration, self.settings.base_lr)
            ),
            "state_guard": "PASS",
            "not_a_causal_proof": True,
        }

    def save(self, path):
        arrays = {f"v{i}": v.numpy() for i, v in enumerate(self.variables())}
        arrays.update(factors=self.factors, sa_updates=np.array(self.sa_updates))
        np.savez_compressed(path, **arrays)

    def load(self, path):
        with np.load(path, allow_pickle=False) as z:
            variables = self.variables()
            if set(z.files) != {
                *(f"v{i}" for i in range(len(variables))),
                "factors",
                "sa_updates",
            }:
                raise ValueError("Checkpoint variable inventory mismatch")
            for i, v in enumerate(variables):
                value = z[f"v{i}"]
                if tuple(v.shape) != value.shape or not np.isfinite(value).all():
                    raise ValueError("Checkpoint shape/finite check failed")
            for i, v in enumerate(variables):
                v.assign(z[f"v{i}"])
            self.factors = z["factors"].astype(np.float32)
            self.sa_updates = int(z["sa_updates"])
            if (
                self.factors.shape != (3,)
                or not np.isfinite(self.factors).all()
                or (self.factors <= 0).any()
                or self.sa_updates < 0
            ):
                raise ValueError("Checkpoint SA state invalid")

    def predict(self, x, batch=256):
        return np.concatenate(
            [self.model(x[i : i + batch]).numpy() for i in range(0, len(x), batch)]
        )

    def lbfgs(self, tensors, completed):
        from scipy.optimize import minimize

        tf = self.tf
        variables = self.network + (
            [self.model.log_coefficient] if self.inverse else []
        )
        shapes = [tuple(v.shape) for v in variables]
        sizes = [int(np.prod(s)) for s in shapes]

        def vector():
            return np.concatenate([v.numpy().ravel() for v in variables]).astype(float)

        def assign(position):
            if not np.isfinite(position).all():
                raise ValueError("Nonfinite L-BFGS position")
            offset = 0
            for v, shape, size in zip(variables, shapes, sizes):
                v.assign(position[offset : offset + size].reshape(shape))
                offset += size

        weights = tf.constant(group_weights(self.arm, completed, self.factors))
        x0 = vector()
        calls = 0

        def objective(position):
            nonlocal calls
            calls += 1
            assign(position)
            with tf.GradientTape() as tape:
                loss = tf.reduce_sum(
                    self.groups(tensors, self.arm != "supervised_only") * weights
                )
            grads = tape.gradient(loss, variables)
            grads = [
                tf.zeros_like(v) if g is None else g for g, v in zip(grads, variables)
            ]
            grad = np.concatenate([v.numpy().ravel() for v in grads]).astype(float)
            value = float(loss)
            if not np.isfinite(value) or not np.isfinite(grad).all():
                raise ValueError("Nonfinite L-BFGS objective")
            return value, grad

        baseline, _ = objective(x0)
        try:
            result = minimize(
                objective,
                x0,
                jac=True,
                method="L-BFGS-B",
                options={
                    "maxiter": self.settings.lbfgs_maxiter,
                    "maxls": 20,
                    "maxfun": max(20, self.settings.lbfgs_maxiter * 30),
                    "ftol": 1e-12,
                    "gtol": 1e-8,
                },
            )
            assign(result.x)
            candidate, _ = objective(result.x)
            return {
                "iterations": int(result.nit),
                "function_evaluations": calls,
                "converged": bool(result.success),
                "message": str(result.message),
                "objective_before": baseline,
                "objective_after": candidate,
                "finite": True,
                "frozen_sampling_and_sa": True,
            }
        except (ValueError, FloatingPointError) as exc:
            assign(x0)
            return {
                "iterations": None,
                "function_evaluations": calls,
                "converged": False,
                "message": str(exc),
                "objective_before": baseline,
                "objective_after": None,
                "finite": False,
                "frozen_sampling_and_sa": True,
            }
