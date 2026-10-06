"""Disposable historical Wave Adam -> one-iteration L-BFGS adapter check."""

import argparse
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main(bundle):
    from v3 import context

    ns = context()
    scratch = ROOT / "tmp/wave_handoff"
    scratch.mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(scratch)
    root = Path(tempfile.mkdtemp(dir=scratch))
    ns["release_open"](bundle, root / "results")
    source = root / "results/inputs/wave_source.zip"
    parent = root / "parent"
    with ns["zipfile"].ZipFile(root / "results/inputs/wave_parent.zip") as z:
        ns["u_members"](z)
        z.extractall(parent)
    settings = ns["release_settings"]("wave2d")
    settings["field_target"] = 0.05  # historical stopping policy, not relabelled
    runtime = ns["load_wave_training"](source, settings)
    runtime["configure_normalization"](
        parent,
        runtime["wave2d_config"],
        runtime["SOURCE_ARRAYS"],
        runtime["PROTOCOL"],
        allow_new=False,
    )
    trial = parent / "trials/seed_3234_baseline"
    old = runtime["set_wave_state_factory"](
        lambda cfg, arm, seed: ns["release_wave_checkpoint_state"](
            runtime, cfg, arm, seed, trial, parent
        )
    )
    runtime["LB_POLICY"]["refine_iterations"] = 1
    runtime["LB_POLICY"]["refine_evaluations"] = 10
    try:
        result = runtime["lb_trial"](
            root / "post",
            trial,
            runtime["wave2d_config"],
            runtime["SOURCE_ARRAYS"],
            runtime["PROTOCOL"],
            "baseline",
            3234,
            100000,
            "refine",
        )
        review = runtime["lb_review"](
            root / "post",
            trial,
            runtime["wave2d_config"],
            runtime["SOURCE_ARRAYS"],
            runtime["PROTOCOL"],
            "baseline",
            3234,
            100000,
            "refine",
        )
    finally:
        runtime["set_wave_state_factory"](old)
    report = {
        "passed": review["passed"],
        "review": review,
        "accepted_iterations": result["solver"]["accepted_iterations"],
        "objective_evaluations": result["solver"]["objective_evaluations"],
        "scope": "Disposable one-iteration adapter check; no benchmark continuation or promotion",
        "source_hashes": ns["V3_EXECUTED_SOURCE_HASHES"],
    }
    ns["u_json"](ROOT / "outputs/V3_Wave_Handoff_Verification.json", report)
    print(json.dumps(report, ensure_ascii=False))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    args = parser.parse_args()
    if not main(args.bundle)["passed"]:
        raise SystemExit(1)
