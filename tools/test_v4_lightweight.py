"""Scoped regression for the lightweight revision; no full suite or live downloads."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch, Mock
import zipfile
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
tempfile.tempdir = str(ROOT / "tmp")
from v4.config import Settings
from v4.data import DataSource, make_plan
from v4.runtime import Runtime
from v4.runner import create_run, run_all, training_source_digest
from v4.sampling import permitted_lhs
from v4.storage import pack, restore, verify_zip, write_json
from v4.verification import integration_checks


class LightweightTests(unittest.TestCase):
    def test_panel_reuses_service_only_for_the_same_archive(self):
        from v4.interface import panel
        from v4.services import Service

        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            root = folder / "run"
            create_run(root, Settings())
            pack(root, folder / "V4_Results.zip")
            pack(root, folder / "other.zip")
            p = panel(folder=str(folder), show=False)
            p["widgets"]["action"].value = "ask"
            instances = []

            def service(root):
                instance = Service(root)
                instances.append(instance)
                return instance

            with patch("v4.interface.Service", side_effect=service):
                p["execute"]()
                p["execute"]()
                self.assertEqual(len(instances), 1)
                p["widgets"]["zip"].value = "other.zip"
                p["execute"]()
                self.assertEqual(len(instances), 2)
                self.assertIs(p["state"]["service"], instances[-1])
                self.assertNotEqual(instances[0].root, instances[1].root)

    def test_no_selection_does_not_run_service(self):
        with patch("v4.verification.Service") as service:
            with self.assertRaises(ValueError):
                integration_checks(ROOT)
            service.assert_not_called()

    def test_custom_questions_do_not_trigger_full_suite(self):
        service = Mock()
        service.ask.return_value = {
            "answer": "비교",
            "tool": "comparison",
            "llm_called": False,
        }
        with patch("v4.verification.Service", return_value=service):
            result = integration_checks(ROOT, questions=("파동 비교",))
        self.assertEqual(result["question_count"], 1)
        self.assertEqual(result["actual_llm_calls"], 0)
        service.search.assert_not_called()
        service.ask.assert_called_once_with("파동 비교", False, False, "numpy")
        self.assertFalse(result["equation_audit_executed"])
        self.assertFalse(result["all_model_audit_executed"])

    def test_selected_integrations_only_with_mock_service(self):
        service = Mock()
        service.ask.return_value = {
            "answer": "자료",
            "tool": "search",
            "llm_called": True,
        }
        service.search.return_value = [{"text": "자료"}]
        with patch("v4.verification.Service", return_value=service):
            result = integration_checks(ROOT, True)
            self.assertEqual(service.ask.call_count, 3)
            service.search.assert_not_called()
            self.assertEqual(result["actual_llm_calls"], 3)
            service.reset_mock()
            result = integration_checks(ROOT, dense=True, backend="faiss")
            service.search.assert_called_once()
            service.ask.assert_not_called()
            self.assertEqual(result["question_count"], 0)
            self.assertTrue(result["retrieval"]["passed"])
        self.assertFalse(result["semantic_accuracy_verified"])

    def test_streaming_crc_sha_and_owner_guard(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            root = folder / "run"
            create_run(root, Settings())
            bundle = folder / "data.zip"
            with patch.object(
                zipfile.ZipFile,
                "testzip",
                side_effect=AssertionError("duplicate CRC pass"),
            ):
                pack(root, bundle)
                contract = verify_zip(bundle)
                restore(
                    bundle, folder / "fresh", expected={"run_id": contract["run_id"]}
                )
            with self.assertRaises(ValueError):
                restore(bundle, folder / "wrong", expected={"run_id": "wrong"})
            self.assertFalse((folder / "wrong").exists())
            invalid = folder / "invalid.zip"
            with zipfile.ZipFile(bundle) as source, zipfile.ZipFile(
                invalid, "w"
            ) as target:
                for name in source.namelist():
                    value = source.read(name)
                    target.writestr(
                        name, value + b" " if name == "settings.json" else value
                    )
            with self.assertRaisesRegex(ValueError, "해시"):
                verify_zip(invalid)
            bad_crc = folder / "crc.zip"
            with zipfile.ZipFile(bundle) as source, zipfile.ZipFile(
                bad_crc, "w", zipfile.ZIP_STORED
            ) as target:
                for name in source.namelist():
                    target.writestr(name, source.read(name))
            with zipfile.ZipFile(bad_crc) as archive:
                item = archive.getinfo("settings.json")
            content = bytearray(bad_crc.read_bytes())
            offset = (
                item.header_offset + 30 + len(item.filename.encode()) + len(item.extra)
            )
            content[offset] ^= 1
            bad_crc.write_bytes(content)  # Test fixture, not an input archive.
            with self.assertRaises(zipfile.BadZipFile):
                verify_zip(bad_crc)

    def test_sa_numerics_without_redundant_state_copies(self):
        settings = Settings(equation="poisson2d", inverse=True).smoke()
        source = DataSource(settings)
        plan = make_plan(settings, 3234)
        arrays = source.training(plan, 0)
        arrays["pde"] = permitted_lhs(plan, settings.pde_count, 3234, "pde", 0)
        runtime = Runtime(settings, "ff_loss_sa", source)
        tensors = runtime.prepare(arrays)
        before = runtime.snapshot()
        _, gradients, _ = runtime.term_gradients(tensors)
        norms = np.array([np.linalg.norm(g) for g in gradients])
        expected = settings.sa_ema * runtime.factors + (1 - settings.sa_ema) * np.clip(
            norms.mean() / np.maximum(norms, 1e-12), *settings.sa_bounds
        ).astype(np.float32)
        with patch.object(
            runtime, "snapshot", side_effect=AssertionError("SA full copy")
        ):
            runtime.sa_update(tensors)
        after = runtime.snapshot()
        for old, new in zip(before[:-2], after[:-2]):
            np.testing.assert_array_equal(old, new)
        np.testing.assert_allclose(runtime.factors, expected, rtol=1e-7)
        self.assertEqual(runtime.sa_updates, 1)
        self.assertEqual(runtime.observe(tensors, 1)["state_guard"], "PASS")

    def test_completion_packs_once_per_trial_and_skips_completed(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            root = folder / "run"
            settings = replace(
                Settings(equation="poisson2d", arms=("baseline", "ff_loss_sa")).smoke(),
                cap=2,
                lbfgs_maxiter=0,
            )
            source = create_run(root, settings)
            bundle = folder / "results.zip"
            slot = folder / "resume.zip"
            calls = []

            def counted_pack(root, output):
                calls.append(Path(output))
                return pack(root, output)

            with patch("v4.runner.pack", side_effect=counted_pack):
                completed = run_all(
                    root, settings, source, bundle, slot, True, lambda _: None
                )
                self.assertEqual(calls.count(bundle), 2)
                self.assertFalse(slot.exists())
                self.assertEqual(len(completed), 2)
                with zipfile.ZipFile(bundle) as archive:
                    self.assertEqual(
                        json.loads(archive.read("summary.json"))["completed_trials"], 2
                    )
                initial_calls = len(calls)
                run_all(root, settings, source, bundle, slot, True, lambda _: None)
                self.assertEqual(len(calls), initial_calls)

    def test_panel_excludes_legacy_controls(self):
        from v4.interface import panel

        p = panel(folder="outputs", show=False)
        p["widgets"]["action"].value = "check"

        def descriptions(widget):
            yield getattr(widget, "description", "")
            for child in getattr(widget, "children", ()):
                yield from descriptions(child)

        labels = tuple(descriptions(p["dashboard"]))
        self.assertNotIn("완료 모델 전수 로드 검사", labels)
        self.assertIn("확인할 질문", labels)
        self.assertFalse(p["widgets"]["qwen"].value)
        self.assertFalse(p["widgets"]["dense"].value)


if __name__ == "__main__":
    from tools.test_v4 import CoreTests

    suite = unittest.TestLoader().loadTestsFromTestCase(LightweightTests)
    for name in (
        "test_arms_roundtrip_and_next_update",
        "test_runner_resume_lbfgs_api_plots",
    ):
        suite.addTest(CoreTests(name))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    write_json(
        ROOT / "outputs/V4_Lightweight_Check.json",
        {
            "passed": result.wasSuccessful(),
            "tests_run": result.testsRun,
            "scope": "changed selection, SA, ZIP, completion and resume paths only",
            "full_baseline_suite_executed": False,
            "training_source_sha": training_source_digest(),
            "source_sha256": {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (ROOT / "v4").glob("*.py")
            },
            "actual_qwen_e5_calls": False,
            "research_training_executed": False,
        },
    )
    sys.exit(0 if result.wasSuccessful() else 1)
