"""CPU wiring tests. Disposable updates do not count as benchmark results."""

from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile
import csv
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
tempfile.tempdir = str(ROOT / "tmp")
from v4.config import (
    Settings,
    ARMS,
    checkpoint_due,
    group_weights,
    learning_rates,
    target_reached,
)
from v4.data import DataSource, make_plan
from v4.equations import EQUATIONS, exact_field, known_forcing, residual
from v4.runtime import Runtime
from v4.runner import create_run, run_trial, renew_pde
from v4.sampling import permitted_lhs, rad_select, rng_for, lhs
from v4.storage import arrays_sha, pack, restore, verify_zip, write_json
from v4.reporting import generate_report, make_plots
from v4.services import Service, create_app
from tools.v4_audits import verify_equations, check_archive_models
from v4.experiments import experiment_plan


class CoreTests(unittest.TestCase):
    def test_default_ff_architecture_update_and_observer(self):
        settings = replace(
            Settings(inverse=True, arms=("ff_loss_sa",)),
            pde_count=8,
            observation_count=8,
            boundary_count=4,
            initial_count=8,
            validation_count=16,
            evaluation_count=16,
            rad_candidates=32,
            lbfgs_maxiter=0,
        ).validate()
        source = DataSource(settings)
        plan = make_plan(settings, 3234)
        arrays = source.training(plan, 0)
        arrays["pde"] = permitted_lhs(plan, 8, 3234, "pde", 0)
        runtime = Runtime(settings, "ff_loss_sa", source)
        self.assertEqual(len(runtime.model.banks), 3)
        self.assertTrue(
            all(tuple(bank.shape) == (3, 32) for bank in runtime.model.banks)
        )
        self.assertEqual(len(runtime.model.hidden), 5)
        self.assertTrue(all(layer.units == 64 for layer in runtime.model.hidden))
        tensors = runtime.prepare(arrays)
        self.assertTrue(np.isfinite(runtime.step(tensors, 0)))
        self.assertEqual(runtime.observe(tensors, 1)["state_guard"], "PASS")

    def test_zero_reference_does_not_hide_output_error(self):
        from v4.metrics import observed_metrics

        result = observed_metrics(
            np.array([[1.0, 0.0], [2.0, 0.0]]), np.array([[1.0, 10.0], [2.0, 10.0]])
        )
        self.assertIsNone(result["relative_l2"])
        self.assertFalse(result["all_observed_relative_errors_defined"])
        self.assertGreater(result["components"][1]["mse"], 0)

    def test_external_cylinder_adapter_wiring_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            samples = {
                "observations": [(2, 0, 0.2, 1, 0, 0), (3, 0.2, 0.3, 1, 0, 0)],
                "validation": [(4, 0.3, 0.4, 1, 0, 0)],
                "boundary": [(1, 0, 0.2, 1, 0, 0), (8, 0, 0.2, 1, 0, 0)],
                "initial": [(2, 0, 0, 1, 0, 0)],
                "gauge": [(1, -2, 0.1, 1, 0, 0)],
            }
            for name, rows in samples.items():
                with (folder / (name + ".csv")).open(
                    "w", newline="", encoding="utf-8"
                ) as f:
                    writer = csv.writer(f)
                    writer.writerow(["x", "y", "t", "u", "v", "p"])
                    writer.writerows(rows)
            contract = {
                "equation": "cylinder_ns",
                "canonical_units_confirmed": True,
                "domain": [[1, 8], [-2, 2], [0, 1]],
                "fixed_coefficient": 0.01,
                "field_scales": [1, 1, 1],
                "residual_scales": [1, 1, 1],
                "forcing_mode": "zero",
                "provenance": "Disposable constant-field CSV adapter fixture; not cylinder reference data",
                "files": {name: name + ".csv" for name in samples},
            }
            write_json(folder / "data.json", contract)
            s = Settings(equation="cylinder_ns", data_mode="csv", inverse=True).smoke()
            source = DataSource(s, folder / "data.json")
            p = make_plan(s, 3234, source.domain)
            arrays = source.training(p, 0)
            arrays["pde"] = permitted_lhs(p, s.pde_count, 3234, "pde", 0)
            rt = Runtime(s, "ff_loss_sa", source)
            tensors = rt.prepare(arrays)
            self.assertTrue(np.isfinite(rt.step(tensors, 0)))
            self.assertEqual(rt.observe(tensors, 1)["state_guard"], "PASS")
            self.assertIsNone(source.truth)

    def test_faiss_matches_numpy_with_mock_embeddings(self):
        try:
            import faiss
        except ImportError:
            self.skipTest("optional FAISS not installed")

        class EmbeddingFixture:
            def encode(self, texts, normalize_embeddings=True):
                vectors = np.array(
                    [
                        np.random.default_rng(
                            int.from_bytes(
                                __import__("hashlib").sha256(t.encode()).digest()[:4],
                                "little",
                            )
                        ).normal(size=16)
                        for t in texts
                    ],
                    np.float32,
                )
                return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "REPORT.md").write_text(
                "Adam L-BFGS 비교\n" + ("물리장 오차 " * 300), encoding="utf-8"
            )
            service = Service(root)
            service.embedding = EmbeddingFixture()
            import types
            from unittest.mock import patch

            with patch.dict(
                sys.modules,
                {
                    "sentence_transformers": types.SimpleNamespace(
                        SentenceTransformer=None
                    )
                },
            ):
                numpy_results = service.search("Adam 비교", True, "numpy")
                faiss_results = service.search("Adam 비교", True, "faiss")
            self.assertEqual(
                [r["text"] for r in numpy_results], [r["text"] for r in faiss_results]
            )
            np.testing.assert_allclose(
                [r["score"] for r in numpy_results], [r["score"] for r in faiss_results]
            )

    def test_batch_plan(self):
        plan = experiment_plan(
            Settings(),
            ("poisson2d", "heat2d"),
            (False, True),
            (0.0, 0.01),
            (0.5, 1.0, 2.0),
        )
        self.assertEqual(len(plan), 16)
        self.assertEqual(
            len({json.dumps(s.to_dict(), sort_keys=True) for s in plan}), 16
        )
        with self.assertRaises(ValueError):
            experiment_plan(Settings(), noise=(0.0, 0.0))

    def test_csv_contract_preserves_observations(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            samples = {
                "observations": [(0.2, 0.2, 0.3), (0.7, 0.4, 0.5)],
                "validation": [(0.3, 0.3, 0.4)],
                "evaluation": [(0.4, 0.4, 0.6)],
                "boundary": [(0, 0.5, 0), (1, 0.5, 0)],
            }
            for name, rows in samples.items():
                with (folder / (name + ".csv")).open(
                    "w", newline="", encoding="utf-8"
                ) as f:
                    writer = csv.writer(f)
                    writer.writerow(["x", "y", "u"])
                    writer.writerows(rows)
            contract = {
                "equation": "poisson2d",
                "canonical_units_confirmed": True,
                "domain": [[0, 1], [0, 1]],
                "fixed_coefficient": 1.0,
                "field_scales": [1.0],
                "residual_scales": [1.0],
                "forcing_mode": "zero",
                "files": {name: name + ".csv" for name in samples},
            }
            write_json(folder / "data.json", contract)
            s = Settings(equation="poisson2d", data_mode="csv").smoke()
            source = DataSource(s, folder / "data.json")
            plan = make_plan(s, 3234)
            source.labels = lambda x: (_ for _ in ()).throw(
                AssertionError("synthetic fill")
            )
            self.assertEqual(
                arrays_sha(source.training(plan, 0)),
                arrays_sha(source.training(plan, 1)),
            )
            self.assertIsNone(source.truth)
            originals = {
                name: (folder / (name + ".csv")).read_bytes() for name in samples
            }
            create_run(folder / "run", s, folder / "data.json")
            for name, content in originals.items():
                self.assertEqual(content, (folder / (name + ".csv")).read_bytes())
            contract["files"]["evaluation"] = "validation.csv"
            write_json(folder / "bad.json", contract)
            with self.assertRaises(ValueError):
                DataSource(s, folder / "bad.json")
            contract["files"]["observations"] = "../outside.csv"
            write_json(folder / "outside.json", contract)
            with self.assertRaisesRegex(ValueError, "계약 JSON의 폴더"):
                DataSource(s, folder / "outside.json")

    def test_panel_safe_defaults_and_conditional_controls(self):
        from v4.interface import panel

        p = panel(folder="outputs", show=False)
        self.assertEqual(p["widgets"]["action"].value, "results")
        self.assertFalse(p["widgets"]["authorized"].value)
        p["widgets"]["action"].value = "train"
        self.assertFalse(p["widgets"]["authorized"].value)
        p["widgets"]["action"].value = "results"

    def test_contracts(self):
        s = Settings().validate()
        self.assertEqual(Settings.from_dict(json.loads(json.dumps(s.to_dict()))), s)
        self.assertEqual(learning_rates(0, 0.004), (0.004, 0.000004))
        np.testing.assert_allclose(learning_rates(5000, 0.004), (0.003, 0.003))
        np.testing.assert_allclose(learning_rates(10000, 0.004), (0.002, 0.004))
        for it in (0, 100, 5000, 10000):
            self.assertAlmostEqual(
                float(group_weights("ff_curriculum", it, [1, 1, 1])[0]),
                min(it // 100 / 100, 1) / 3,
                places=6,
            )
        self.assertEqual(
            [i for i in range(1, 101) if checkpoint_due(i, 100)], [20, 40, 60, 80, 100]
        )
        history = [
            {"iteration": i, "validation": {"relative_l1": 0.04, "relative_l2": 0.04}}
            for i in (14000, 15000, 16000)
        ]
        self.assertFalse(target_reached(history, s))
        history.append(
            {
                "iteration": 17000,
                "validation": {"relative_l1": 0.04, "relative_l2": 0.04},
            }
        )
        self.assertTrue(target_reached(history, s))
        with self.assertRaises(ValueError):
            Settings(equation="cylinder_ns").validate()
        with self.assertRaises(ValueError):
            Settings(
                equation="poisson2d", generalization="time_extrapolation"
            ).validate()

    def test_rad_sampling_and_splits(self):
        pool = np.arange(200).reshape(100, 2)
        chosen, record = rad_select(pool, np.zeros(100), 20, rng_for(3234, "probe"))
        self.assertEqual(len({tuple(p) for p in chosen}), 20)
        self.assertTrue(np.isfinite(list(record.values())).all())
        for mode in ("basic", "interpolation", "time_extrapolation"):
            s = Settings(generalization=mode).smoke()
            d = DataSource(s)
            p = make_plan(s, 3234)
            a = d.training(p, 0)
            b = d.training(p, 1)
            self.assertNotEqual(arrays_sha(a), arrays_sha(b))
            self.assertEqual(arrays_sha(d.validation(p)), arrays_sha(d.validation(p)))
            for family in ("observations", "boundary", "initial", "gauge"):
                if family in b:
                    p.assert_training(b[family], family)

    def test_equation_finite_difference(self):
        self.assertTrue(verify_equations()["passed"])

    def test_archive_rejects_unsafe_paths(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            z = root / "bad.zip"
            with zipfile.ZipFile(z, "w") as archive:
                archive.writestr("../escape", "x")
            with self.assertRaises(ValueError):
                verify_zip(z)

    def test_tensorflow_equations_forward_inverse(self):
        import tensorflow as tf

        records = []
        for name, spec in EQUATIONS.items():
            if name == "cylinder_ns":
                continue
            s = Settings(equation=name, inverse=True).smoke()
            source = DataSource(s)
            rt = Runtime(s, "baseline", source)
            bounds = np.array(spec["domain"], float)
            width = np.ptp(bounds, axis=1)
            bounds[:, 0] += 0.2 * width
            bounds[:, 1] -= 0.2 * width
            x = lhs(bounds, 4, rng_for(91923, name))
            if name == "burgers":
                x[:, 0] = 0.5 * x[:, 1] + 0.005
            q, g, h = rt.derivatives(x, lambda z: exact_field(name, z, tf))
            r = residual(
                name,
                tf.constant(x, tf.float32),
                q,
                g,
                h,
                spec["fixed"],
                tf.constant(known_forcing(name, x), tf.float32),
                tf,
            ).numpy()
            normalized = float(
                abs(r).max() / max(1.0, abs(known_forcing(name, x)).max())
            )
            self.assertLess(normalized, 0.002, name)
            plan = make_plan(s, 3234)
            arrays = source.training(plan, 0)
            arrays["pde"] = permitted_lhs(plan, s.pde_count, 3234, "pde", 0)
            tensors = rt.prepare(arrays)
            before = rt.snapshot()
            diagnostic = rt.observe(tensors, 0)
            rt.assert_unchanged(before)
            self.assertEqual(diagnostic["coefficient_term_gradients"][1:], [0.0, 0.0])
            self.assertTrue(np.isfinite(rt.step(tensors, 0)))
            self.assertEqual(int(rt.coefficient_opt.iterations), 1)
            self.assertEqual(int(rt.opt.iterations), 1)
            records.append(
                {
                    "equation": name,
                    "autodiff_residual_max": normalized,
                    "inverse_update": True,
                }
            )
            forward = Runtime(
                replace(s, inverse=False),
                "baseline",
                DataSource(replace(s, inverse=False)),
            )
            self.assertTrue(np.isfinite(forward.step(forward.prepare(arrays), 0)))
            self.assertEqual(int(forward.coefficient_opt.iterations), 0)
            tf.keras.backend.clear_session()
        write_json(ROOT / "outputs/V4_Equation_Wiring.json", records)

    def test_arms_roundtrip_and_next_update(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "checkpoint.npz"
            s = Settings(inverse=True).smoke()
            d = DataSource(s)
            p = make_plan(s, 3234)
            a = d.training(p, 0)
            a["pde"] = permitted_lhs(p, s.pde_count, 3234, "pde", 0)
            for arm in ARMS:
                rt = Runtime(s, arm, d)
                tensors = rt.prepare(a)
                rt.step(tensors, 0)
                if arm in {"loss_sa", "ff_loss_sa"}:
                    rt.sa_update(tensors)
                else:
                    with self.assertRaises(ValueError):
                        rt.sa_update(tensors)
                rt.save(path)
                clone = Runtime(s, arm, d)
                clone.load(path)
                clone.assert_unchanged(rt.snapshot())
                rt.step(tensors, 1)
                clone.step(clone.prepare(a), 1)
                for first, second in zip(rt.snapshot(), clone.snapshot()):
                    np.testing.assert_allclose(first, second, rtol=1e-6, atol=1e-7)
                self.assertEqual(
                    int(clone.coefficient_opt.iterations),
                    0 if arm == "supervised_only" else 2,
                )

    def test_runner_resume_lbfgs_api_plots(self):
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            root = temp / "run"
            s = Settings(
                equation="poisson2d", inverse=True, arms=("ff_loss_sa",)
            ).smoke()
            source = create_run(root, s)
            results = temp / "results.zip"
            slot = temp / "resume.zip"
            pack(root, results)
            interrupted = run_trial(
                root,
                s,
                "ff_loss_sa",
                3234,
                source,
                slot,
                results,
                log=lambda _: None,
                update_limit=3,
            )
            self.assertEqual(interrupted["iteration"], 3)
            self.assertEqual(verify_zip(slot)["kind"], "resume")
            fresh = temp / "restored"
            restore(results, fresh)
            resumed = run_trial(
                fresh,
                s,
                "ff_loss_sa",
                3234,
                DataSource(s),
                slot,
                results,
                log=lambda _: None,
            )
            self.assertEqual(resumed["adam"]["optimizer_updates"], 10)
            self.assertEqual(resumed["optimizer_updates_this_call"], 7)
            self.assertFalse(slot.exists())
            self.assertIn("solver", resumed["lbfgs"])
            self.assertIn("boundary", resumed["adam"]["constraints"])
            generate_report(fresh)
            pack(fresh, results)
            self.assertEqual(verify_zip(results)["kind"], "results")
            self.assertEqual(check_archive_models(fresh)["checked_models"], 1)
            service = Service(fresh)
            model = service.models()[0]
            identity = model["trial_id"]
            self.assertTrue(
                np.isfinite(service.predict(identity, [[0.2, 0.3]])["prediction"]).all()
            )
            self.assertEqual(service.ask("poisson2d 비교해줘")["tool"], "comparison")
            self.assertEqual(service.ask("모든 파일 삭제")["tool"], "refuse")
            with self.assertRaises(PermissionError):
                service.register("poisson2d", identity)
            service.register("poisson2d", identity, True)
            other_settings = replace(s, arms=("ff",), cap=2, lbfgs_maxiter=0)
            other = run_trial(
                fresh,
                other_settings,
                "ff",
                3234,
                DataSource(other_settings),
                log=lambda _: None,
            )
            service.register("poisson2d", other["trial_id"], True)
            self.assertEqual(
                service.registry()["active"]["poisson2d"], other["trial_id"]
            )
            service.rollback("poisson2d", True)
            self.assertEqual(service.registry()["active"]["poisson2d"], identity)
            from fastapi.testclient import TestClient

            client = TestClient(create_app(service))
            self.assertEqual(client.get("/models").status_code, 200)
            self.assertEqual(
                client.get("/model/info", params={"trial_id": identity}).status_code,
                200,
            )
            self.assertEqual(
                client.post(
                    "/predict", json={"trial_id": identity, "coordinates": [[0.1, 0.3]]}
                ).status_code,
                200,
            )
            self.assertEqual(
                client.post(
                    "/predict", json={"trial_id": identity, "coordinates": [[2.0, 0.3]]}
                ).status_code,
                400,
            )
            self.assertEqual(
                client.post(
                    "/registry/select",
                    json={
                        "equation": "poisson2d",
                        "trial_id": identity,
                        "approved": True,
                    },
                ).status_code,
                403,
            )
            self.assertTrue(client.get("/openapi.json").json()["paths"])
            self.assertEqual(
                client.post(
                    "/agent/tool", json={"name": "generate_report", "approved": True}
                ).status_code,
                403,
            )
            self.assertEqual(
                client.post(
                    "/agent/tool", json={"name": "execute_python", "arguments": {}}
                ).status_code,
                422,
            )
            self.assertTrue(make_plots(fresh, identity))
            direct = temp / "direct"
            direct_source = create_run(direct, s)
            full = run_trial(
                direct, s, "ff_loss_sa", 3234, direct_source, log=lambda _: None
            )
            np.testing.assert_allclose(
                resumed["adam"]["field"]["relative_l2"],
                full["adam"]["field"]["relative_l2"],
                rtol=1e-6,
            )
            for name in ("adam.npz", "selected.npz"):
                with np.load(fresh / "trials" / identity / name) as first, np.load(
                    direct / "trials" / identity / name
                ) as second:
                    for key in first.files:
                        np.testing.assert_allclose(
                            first[key], second[key], rtol=1e-6, atol=1e-7
                        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
