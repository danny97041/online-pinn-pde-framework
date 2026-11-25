"""Focused read-only checks for checkbox selection and Korean plan summaries."""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from v4.config import ARMS
from v4.config import Settings
from v4.settings_panel import settings_editor
from v4.interface import panel
from v4.equations import EQUATIONS
from v4.experiments import experiment_plan
from v4.storage import write_json, sha_file


class CheckboxTests(unittest.TestCase):
    def test_variable_defaults_and_percentages(self):
        editor = settings_editor()
        self.assertEqual(editor.value, Settings().to_dict())
        self.assertEqual(editor.controls["target"].value, 5)
        editor.controls["target"].value = 10
        self.assertEqual(editor.value["target"], 0.1)
        editor.controls["ff_scales"].value = "0.125, 0.5"
        self.assertEqual(editor.value["ff_scales"], (0.125, 0.5))

    def test_saved_variable_roundtrip_and_automatic_values(self):
        saved = Settings(
            inverse=True,
            arms=ARMS,
            initial_coefficient=0.7,
            block_override=3,
            generalization="interpolation",
            noise_fraction=0.02,
        ).to_dict()
        editor = settings_editor()
        editor.value = saved
        self.assertEqual(editor.value, saved)
        editor.auto["initial_coefficient"].value = True
        self.assertIsNone(editor.value["initial_coefficient"])

    def test_contextual_controls_and_busy_guards(self):
        editor = settings_editor()
        editor.set_context(("baseline",), False, "basic", "synthetic", False)
        for name in (
            "ff_features",
            "sa_every",
            "initial_coefficient",
            "initial_count",
            "block_override",
        ):
            self.assertTrue(editor.controls[name].disabled)
        editor.set_context(("ff_loss_sa",), True, "interpolation", "synthetic", True)
        self.assertFalse(editor.controls["ff_features"].disabled)
        self.assertFalse(editor.controls["sa_every"].disabled)
        editor.auto["initial_coefficient"].value = False
        self.assertFalse(editor.controls["initial_coefficient"].disabled)
        editor.disabled = True
        self.assertTrue(all(x.disabled for x in editor.controls.values()))
        editor.disabled = False
        self.assertFalse(editor.controls["base_lr"].disabled)
        editor.set_context(("ff_loss_sa",), True, "basic", "csv", True)
        self.assertTrue(editor.controls["observation_count"].disabled)

    def test_invalid_list_is_rejected(self):
        editor = settings_editor()
        editor.controls["ff_scales"].value = "0.125, bad"
        with self.assertRaises(ValueError):
            editor.value

    def test_selection_and_restore(self):
        methods = panel(show=False)["widgets"]["arms"]
        self.assertEqual(methods.value, ("baseline",))
        methods.value = ARMS
        self.assertEqual(methods.value, tuple(ARMS))
        methods.value = ("ff", "loss_sa")
        self.assertEqual(set(methods.value), {"ff", "loss_sa"})
        methods.value = ()
        self.assertEqual(methods.value, ())

    def test_busy_state_disables_all_methods(self):
        methods = panel(show=False)["widgets"]["arms"]
        methods.disabled = True
        self.assertTrue(all(box.disabled for box in methods.checks.values()))
        methods.disabled = False
        self.assertFalse(any(box.disabled for box in methods.checks.values()))

    def test_equation_checkboxes_selection_and_restore(self):
        equations = panel(show=False)["widgets"]["batch_equations"]
        self.assertEqual(equations.value, ("wave2d",))
        self.assertEqual(len(equations.checks), 11)
        self.assertNotIn("cylinder_ns", equations.checks)
        equations.value = ("poisson2d", "heat2d")
        self.assertEqual(set(equations.value), {"poisson2d", "heat2d"})
        equations.checks["wave2d"].value = True
        self.assertIn("wave2d", equations.value)
        equations.value = ()
        self.assertEqual(equations.value, ())

    def test_busy_state_disables_all_equations(self):
        equations = panel(show=False)["widgets"]["batch_equations"]
        equations.disabled = True
        self.assertTrue(all(box.disabled for box in equations.checks.values()))
        equations.disabled = False
        self.assertFalse(any(box.disabled for box in equations.checks.values()))

    def test_remaining_equations_make_120_trial_plan_without_training(self):
        with patch("IPython.display.display"), patch("builtins.print") as output, patch(
            "v4.interface.experiment_plan", wraps=experiment_plan
        ) as plan, patch("v4.interface.execute_plan") as train:
            p = panel(show=False)
            p["widgets"]["action"].value = "train"
            p["widgets"]["batch"].value = True
            p["widgets"]["arms"].value = ARMS
            selected = tuple(k for k in EQUATIONS if k not in {"wave2d", "cylinder_ns"})
            p["widgets"]["batch_equations"].value = selected
            p["execute"]()
            self.assertEqual(plan.call_args.args[1], selected)
            self.assertEqual(plan.call_args.args[2], (False, True))
            self.assertTrue(
                any(c.args[:2] == ("실행 범위:", 120) for c in output.call_args_list)
            )
            train.assert_not_called()

    def test_empty_equation_selection_is_rejected_without_fallback(self):
        with patch("IPython.display.display"), patch("builtins.print") as output, patch(
            "v4.interface.experiment_plan"
        ) as plan:
            p = panel(show=False)
            p["widgets"]["action"].value = "train"
            p["widgets"]["batch"].value = True
            p["widgets"]["batch_equations"].value = ()
            p["widgets"]["authorized"].value = True
            p["execute"]()
            plan.assert_not_called()
            self.assertIsNone(p["state"]["pending_training"])
            self.assertTrue(
                any("하나 이상 선택" in str(c.args) for c in output.call_args_list)
            )

    def test_equation_change_invalidates_prepared_training(self):
        with patch("IPython.display.display"), patch("builtins.print"), patch(
            "v4.interface.execute_plan"
        ) as train:
            p = panel(show=False)
            p["widgets"]["action"].value = "train"
            p["widgets"]["batch"].value = True
            p["widgets"]["authorized"].value = True
            p["execute"]()
            self.assertIsNotNone(p["state"]["pending_training"])
            p["widgets"]["batch_equations"].checks["heat2d"].value = True
            with self.assertRaises(RuntimeError):
                p["run_training"]()
            train.assert_not_called()

    def test_plan_uses_readable_summary_without_training(self):
        with patch("IPython.display.display") as display, patch(
            "v4.interface.run_all"
        ) as train:
            p = panel(show=False)
            p["widgets"]["action"].value = "train"
            p["widgets"]["arms"].value = ARMS
            p["execute"]()
            train.assert_not_called()
            text = "\n".join(str(c.args[0].data) for c in display.call_args_list)
            self.assertIn("| 선택 방법 | 학습 방식 |", text)
            self.assertIn("L1·L2", text)
            self.assertNotIn("```json", text)


if __name__ == "__main__":
    result = unittest.TextTestRunner().run(
        unittest.defaultTestLoader.loadTestsFromTestCase(CheckboxTests)
    )
    write_json(
        ROOT / "outputs/V4_Interface_Check.json",
        {
            "passed": result.wasSuccessful(),
            "tests": result.testsRun,
            "training_executed": False,
            "scope": "Method/equation checkbox selection, 120-trial dry-run, empty-selection and changed-preparation guards, variable defaults, restore and Korean summary; no training",
            "interface_sha256": sha_file(ROOT / "v4/interface.py"),
            "settings_panel_sha256": sha_file(ROOT / "v4/settings_panel.py"),
        },
    )
    sys.exit(0 if result.wasSuccessful() else 1)
