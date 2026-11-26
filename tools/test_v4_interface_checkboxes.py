"""Focused read-only checks for checkbox selection and Korean plan summaries."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
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
from v4.experiments import execute_plan, experiment_plan
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
        self.assertEqual(equations.value, ("poisson2d", "heat2d"))
        equations.checks["wave2d"].value = True
        self.assertIn("wave2d", equations.value)
        equations.value = ()
        self.assertEqual(equations.value, ())
        self.assertTrue(equations.up.disabled and equations.down.disabled)

    def test_order_buttons_control_plan_preview_and_prepared_execution(self):
        with patch("IPython.display.display") as display, patch("builtins.print"), patch(
            "v4.interface.execute_plan"
        ) as train:
            p = panel(show=False)
            widgets = p["widgets"]
            widgets["action"].value = "train"
            widgets["batch"].value = True
            widgets["authorized"].value = True
            eqs = widgets["batch_equations"]
            eqs.checks["heat2d"].value = True
            eqs.order_list.value = "heat2d"
            self.assertTrue(eqs.down.disabled)
            eqs.up.click()
            self.assertEqual(eqs.value, ("heat2d", "wave2d"))
            self.assertTrue(eqs.up.disabled)
            p["execute"]()
            self.assertIn("2차원 열전도 → 2차원 파동", str(display.call_args.args[0].data))
            self.assertIsNotNone(p["state"]["pending_training"])
            eqs.down.click()
            with self.assertRaises(RuntimeError):
                p["run_training"]()
            train.assert_not_called()
            eqs.checks["wave2d"].value = False
            eqs.checks["wave2d"].value = True
            self.assertEqual(eqs.value, ("heat2d", "wave2d"))
            eqs.disabled = True
            eqs.down.click()
            self.assertEqual(eqs.value, ("heat2d", "wave2d"))

    def test_extrapolation_order_is_saved_and_replayed(self):
        order = ("heat2d", "wave2d")
        plan = experiment_plan(Settings(generalization="time_extrapolation"), order)
        with TemporaryDirectory() as folder, patch("v4.experiments.DataSource"), patch(
            "v4.experiments.run_all", return_value=[]
        ) as runner:
            execute_plan(folder, plan, "result.zip", "resume.zip", True)
            self.assertEqual(tuple(c.args[1].equation for c in runner.call_args_list), order)
            saved = json.loads((Path(folder) / "plan.json").read_text(encoding="utf-8"))
            equations = panel(show=False)["widgets"]["batch_equations"]
            equations.value = [s["equation"] for s in saved]
            self.assertEqual(equations.value, order)
            runner.reset_mock()
            execute_plan(folder, experiment_plan(plan[0], equations.value), "result.zip", "resume.zip", True)
            self.assertEqual(tuple(c.args[1].equation for c in runner.call_args_list), order)

    def test_busy_state_disables_all_equations(self):
        equations = panel(show=False)["widgets"]["batch_equations"]
        equations.disabled = True
        self.assertTrue(all(box.disabled for box in equations.checks.values()))
        equations.disabled = False
        self.assertFalse(any(box.disabled for box in equations.checks.values()))

    def test_goal_filters_single_and_batch_and_preserves_priority(self):
        p = panel(show=False)
        w = p["widgets"]
        w["action"].value = "train"
        w["batch"].value = True
        eqs = w["batch_equations"]
        self.assertEqual(len(w["equation"].options), 11)
        eqs.value = ("burgers", "poisson2d", "heat2d")
        w["equation"].value = "poisson2d"
        w["generalization"].value = "time_extrapolation"
        expected = {k for k, v in EQUATIONS.items() if v["time_axis"] is not None and k != "cylinder_ns"}
        self.assertEqual(expected, {k for _, k in w["equation"].options})
        self.assertEqual(expected, {k for k, box in eqs.checks.items() if box.layout.display != "none"})
        self.assertEqual(eqs.value, ("burgers", "heat2d"))
        w["directions"].value = (False,)
        with patch("v4.interface.experiment_plan", wraps=experiment_plan) as plan:
            p["execute"]()
            self.assertEqual(plan.call_args.args[1], ("burgers", "heat2d"))
        w["generalization"].value = "interpolation"
        self.assertEqual(len(eqs.available), 11)
        self.assertEqual(eqs.value, ("burgers", "heat2d"))

    def test_restoring_steady_plan_after_temporal_selection(self):
        with TemporaryDirectory() as folder, patch("v4.interface.Service") as service:
            root = Path(folder)
            saved = Settings(equation="poisson2d", generalization="interpolation")
            write_json(root / "contract.json", {"kind": "results"})
            write_json(root / "settings.json", saved.to_dict())
            write_json(root / "plan.json", [s.to_dict() for s in experiment_plan(saved, ("poisson2d", "heat2d"))])
            service.return_value.models.return_value = []
            p = panel(folder=folder, show=False)
            w = p["widgets"]
            w["action"].value = "train"
            w["generalization"].value = "time_extrapolation"
            p["state"].update(root=root, bundle=str(root / w["zip"].value))
            p["load_settings"]()
            self.assertEqual(w["execution"].value, "load")
            self.assertEqual(w["equation"].value, "poisson2d")
            self.assertEqual(w["batch_equations"].value, ("poisson2d", "heat2d"))
            self.assertTrue(w["generalization"].disabled)

    def test_csv_extrapolation_uses_only_chosen_contract_equation(self):
        p = panel(show=False)
        w = p["widgets"]
        w["action"].value = "train"
        w["generalization"].value = "time_extrapolation"
        w["data_mode"].value = "csv"
        w["equation"].value = "cylinder_ns"
        w["batch"].value = True
        with patch("v4.interface.experiment_plan", wraps=experiment_plan) as plan:
            p["execute"]()
            self.assertEqual(plan.call_args.args[1], ("cylinder_ns",))
        w["action"].value = "registry"
        self.assertEqual(len(w["equation"].options), len(EQUATIONS))

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
