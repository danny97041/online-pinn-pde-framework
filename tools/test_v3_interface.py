"""Local UI/backend regression. No LLM download or benchmark retraining."""

import argparse
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main(bundle, numerical=False, scratch_base=None):
    from v3 import context

    started = time.perf_counter()
    ns = context()
    temp = Path(scratch_base) if scratch_base else ROOT / "tmp/patch_tests"
    temp.mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(temp)
    root = Path(tempfile.mkdtemp(prefix="results_", dir=temp)) / "contents"
    summary = ns["release_open"](bundle, root)
    assert summary["completed_trials"] == 65
    assert "tensorflow" not in sys.modules
    base = {
        "zip": Path(bundle).name,
        "action": "results",
        "training_enabled": True,
        "check_training": True,
        "use_llm": True,
        "execution": "resume",
        "resume_zip": "../evil.zip",
    }
    assert ns["release_request"](base) == {"zip": Path(bundle).name, "action": "results"}
    for action in ["predict", "train"]:
        data = {
            **base,
            "action": action,
            "equation": "poisson2d",
            "arm": "baseline",
            "seed": 3234,
            "points": "[[0.25,0.65]]",
            "execution": "new",
            "overrides": {"sa_every": 1, "ff_scales": [99], "cap": 20},
        }
        request = ns["release_request"](data)
        assert "use_llm" not in request and "check_training" not in request
        if action == "train":
            assert request["overrides"] == {"cap": 20} and "resume_zip" not in request
    data.update(action="train", arm="ff_loss_sa")
    request = ns["release_request"](data)
    assert request["overrides"]["sa_every"] == 1
    data.update(execution="resume", resume_zip="", overrides={"base_lr": 0.1, "cap": 120000})
    assert ns["release_request"](data)["overrides"] == {"cap": 120000}
    for invalid in ["../file.zip", "C:\\file.zip", "/tmp/file.zip"]:
        try:
            ns["release_request"]({**base, "zip": invalid})
        except ValueError:
            pass
        else:
            raise AssertionError("Unsafe UI filename accepted")
    with contextlib.redirect_stdout(io.StringIO()):
        panel = ns["release_panel"](Path(bundle).parent, temp)
    controls, settings = panel["controls"], panel["settings"]
    assert controls["equation"].disabled and controls["training_enabled"].disabled
    assert (
        controls["zip"].layout.display == "" and controls["equation"].layout.display == "none"
    )
    controls["action"].value = "train"
    assert not controls["equation"].disabled and not controls["training_enabled"].disabled
    assert controls["resume_zip"].disabled and settings["sa_every"].disabled
    assert (
        controls["equation"].layout.display == ""
        and controls["resume_zip"].layout.display == "none"
    )
    controls["arm"].value = "ff_loss_sa"
    assert not settings["sa_every"].disabled
    controls["equation"].value = "wave2d"
    assert settings["sa_every"].disabled and settings["network_width"].disabled
    controls["execution"].value = "resume"
    assert not controls["resume_zip"].disabled and settings["base_lr"].disabled
    controls["action"].value = "evaluate"
    assert controls["training_enabled"].disabled and not controls["check_models"].disabled
    # Default Run all performs only results viewing and lands on results tab.
    notebook = json.loads(
        (ROOT / "outputs/Online_PINN_PDE_Framework_V3.ipynb").read_text(encoding="utf-8")
    )
    cells = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
    run_all = {"__name__": "notebook_test"}
    with contextlib.redirect_stdout(io.StringIO()):
        for code in cells:
            if "folder = Path(저장_폴더)" in code:
                run_all["저장_폴더"] = str(Path(bundle).parent)
            exec(compile(code, "notebook_test", "exec"), run_all)
    assert "tensorflow" not in sys.modules
    assert run_all["PANEL"]["tabs"].selected_index == 1
    assert run_all["PANEL"]["controls"]["action"].value == "results"
    assert run_all["PANEL"]["state"]["last_error"] is None, run_all["PANEL"]["state"]
    assert run_all["PANEL"]["state"]["last_result"]["completed_trials"] == 65
    hashes = ns["V3_EXECUTED_SOURCE_HASHES"]
    assert hashes == run_all["V3_EXECUTED_SOURCE_HASHES"]
    checks = {
        "archive_65_results": True,
        "inactive_options_excluded": True,
        "path_guards": True,
        "conditional_widget_state": True,
        "default_run_all_no_tensorflow": True,
        "automatic_results_tab": True,
        "notebook_source_hashes_match": True,
    }
    if numerical:
        request = ns["release_request"](
            {
                "action": "evaluate",
                "zip": Path(bundle).name,
                "use_llm": False,
                "dense_rag": False,
                "check_models": True,
                "check_training": True,
            }
        )
        with contextlib.redirect_stdout(io.StringIO()) as log:
            numerical_checks = ns["release_inspection"](root, temp, request)
        (ROOT / "outputs/V3_Patch_Numerical_Log.txt").write_text(
            log.getvalue(), encoding="utf-8"
        )
        checks["numerical"] = numerical_checks
    report = {
        "passed": all(v is True for k, v in checks.items() if k != "numerical")
        and checks.get("numerical", {}).get("passed", True),
        "source_hashes": hashes,
        "checks": checks,
        "colab_frontend": "not_run",
        "benchmark_retraining": False,
        "seconds": time.perf_counter() - started,
    }
    ns["u_json"](ROOT / "outputs/V3_Patch_Verification.json", report)
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "checks": {
                    k: (
                        v
                        if k != "numerical"
                        else {name: result["status"] for name, result in v["checks"].items()}
                    )
                    for k, v in checks.items()
                },
                "colab_frontend": report["colab_frontend"],
                "seconds": report["seconds"],
            },
            ensure_ascii=False,
        )
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--numerical", action="store_true")
    parser.add_argument(
        "--scratch",
        type=Path,
        help="Optional scratch directory, e.g. ASCII path for Windows TensorFlow",
    )
    args = parser.parse_args()
    if not main(args.bundle, args.numerical, args.scratch)["passed"]:
        raise SystemExit(1)
