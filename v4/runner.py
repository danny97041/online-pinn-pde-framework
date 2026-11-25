"""Explicitly authorized, sequential V4 experiments with one rolling resume slot."""

import json
import hashlib
from pathlib import Path
import shutil
import tempfile
import time
import uuid
import numpy as np

from .config import checkpoint_due, digest, target_reached
from .data import DataSource, make_plan
from .metrics import evaluate, observed_metrics
from .runtime import Runtime
from .sampling import permitted_lhs, rad_select, rng_for
from .storage import arrays_sha, pack, restore, verify_zip, write_json


def training_source_digest():
    directory = Path(__file__).parent
    names = (
        "runner",
        "runtime",
        "config",
        "data",
        "equations",
        "generalization",
        "sampling",
        "metrics",
        "balance",
    )
    return digest(
        {
            name: hashlib.sha256(
                (directory / (name + ".py")).read_text(encoding="utf-8").encode()
            ).hexdigest()
            for name in names
        }
    )


def trial_identity(settings, arm, seed, source):
    return digest(
        {
            "settings": settings.to_dict(),
            "arm": arm,
            "seed": seed,
            "data": source.fingerprint,
        }
    )[:16]


def renew_pde(runtime, plan, cycle):
    s = runtime.settings
    pool = permitted_lhs(plan, s.rad_candidates, plan.seed, "pde", cycle)
    before = runtime.snapshot()
    scores = []
    for start in range(0, len(pool), 128):
        r = runtime.pde_residual(pool[start : start + 128]).numpy() / np.asarray(
            runtime.scales["residual"]
        )
        scores.append(np.sqrt(np.mean(r * r, axis=1)))
    runtime.assert_unchanged(before)
    selected, report = rad_select(
        pool,
        np.concatenate(scores),
        s.pde_count,
        rng_for(plan.seed, "rad_choice", cycle),
        s.rad_power,
        s.rad_offset,
    )
    report.update(
        cycle=cycle,
        candidate_sha=arrays_sha({"pool": pool}),
        selected_sha=arrays_sha({"pde": selected}),
    )
    plan.assert_training(selected, "pde")
    return selected, report


def save_state(trial, runtime, arrays, fixed, state):
    runtime.save(trial / "checkpoint.npz")
    np.savez_compressed(trial / "arrays.npz", **arrays)
    np.savez_compressed(trial / "fixed.npz", **fixed)
    state = dict(state)
    state.update(array_sha=arrays_sha(arrays), fixed_sha=arrays_sha(fixed))
    write_json(trial / "state.json", state)


def resume_slot(root, trial, output):
    with tempfile.TemporaryDirectory(prefix="v4_resume_", dir=root.parent) as temporary:
        scratch = Path(temporary)
        contract = json.loads((root / "contract.json").read_text(encoding="utf-8"))
        contract.update(kind="resume", trial_id=trial.name)
        write_json(scratch / "contract.json", contract)
        for name in (
            "checkpoint.npz",
            "arrays.npz",
            "fixed.npz",
            "state.json",
            "trial_settings.json",
        ):
            shutil.copyfile(trial / name, scratch / name)
        if (root / "inputs").exists():
            shutil.copytree(root / "inputs", scratch / "inputs")
        return pack(scratch, output)


def restore_slot(bundle, root, expected_id):
    owner = json.loads((root / "contract.json").read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory(
        prefix="v4_resume_read_", dir=root.parent
    ) as temporary:
        scratch = Path(temporary) / "state"
        restore(
            bundle,
            scratch,
            expected={
                "kind": "resume",
                "run_id": owner["run_id"],
                "trial_id": expected_id,
            },
        )
        trial = root / "trials" / expected_id
        trial.mkdir(parents=True, exist_ok=True)
        for name in (
            "checkpoint.npz",
            "arrays.npz",
            "fixed.npz",
            "state.json",
            "trial_settings.json",
        ):
            shutil.copyfile(scratch / name, trial / name)


def run_trial(
    root,
    settings,
    arm,
    seed,
    source,
    resume_zip=None,
    results_zip=None,
    log=print,
    update_limit=None,
):
    settings.validate()
    root = Path(root)
    plan = make_plan(settings, seed, source.domain)
    contract = json.loads((root / "contract.json").read_text(encoding="utf-8"))
    if contract.get("training_source_sha") != training_source_digest():
        raise ValueError("학습 코드가 저장 실행과 다릅니다. 새 독립 실험을 사용하세요.")
    identity = trial_identity(settings, arm, seed, source)
    trial = root / "trials" / identity
    if (trial / "report.json").is_file():
        report = json.loads((trial / "report.json").read_text(encoding="utf-8"))
        log("완료 실험 건너뜀: " + identity)
        return report
    if resume_zip and Path(resume_zip).is_file():
        contract = verify_zip(resume_zip)
        if contract.get("trial_id") == identity:
            restore_slot(resume_zip, root, identity)
    trial.mkdir(parents=True, exist_ok=True)
    expected = {
        "settings": settings.to_dict(),
        "arm": arm,
        "seed": seed,
        "source": source.fingerprint,
        "plan": plan.to_dict(),
    }
    expected_digest = digest(expected)
    runtime = Runtime(settings, arm, source)
    if (trial / "state.json").exists():
        state = json.loads((trial / "state.json").read_text(encoding="utf-8"))
        if state["contract_sha"] != expected_digest:
            raise ValueError("재개 설정·데이터·분할 계약 불일치")
        runtime.load(trial / "checkpoint.npz")
        with np.load(trial / "arrays.npz", allow_pickle=False) as z:
            arrays = {k: z[k].copy() for k in z.files}
        with np.load(trial / "fixed.npz", allow_pickle=False) as z:
            fixed = {k: z[k].copy() for k in z.files}
        if (
            arrays_sha(arrays) != state["array_sha"]
            or arrays_sha(fixed) != state["fixed_sha"]
        ):
            raise ValueError("재개 좌표 해시 불일치")
        if int(runtime.opt.iterations) != state["iteration"]:
            raise ValueError("optimizer iteration 불일치")
        for family in ("pde", "observations", "boundary", "initial", "gauge"):
            if family in arrays:
                plan.assert_training(arrays[family], family)
    else:
        arrays = source.training(plan, 0)
        if arm == "supervised_only":
            arrays["pde"] = permitted_lhs(plan, settings.pde_count, seed, "pde", 0)
            rad = []
        else:
            arrays["pde"], rad_record = renew_pde(runtime, plan, 0)
            rad = [rad_record]
        validation = source.validation(plan)
        evaluation = source.evaluation(plan)
        physics = permitted_lhs(
            plan, min(settings.evaluation_count, 1024), 93235, "pde", 0
        )
        # In extrapolation, field and PDE evaluation are both in the unseen time range.
        if plan.mode == "time_extrapolation":
            physics = plan.fixed_evaluation_points(16, 5)
        fixed = {
            "validation_x": validation["x"],
            "validation_y": validation["y"],
            "physics": physics,
        }
        constraint_arrays = source.training(plan, 93235)
        for name, value in constraint_arrays.items():
            if "observations" not in name:
                fixed["constraint_" + name] = value
        if evaluation is not None:
            fixed.update(evaluation_x=evaluation["x"], evaluation_y=evaluation["y"])
        state = {
            "contract_sha": expected_digest,
            "iteration": 0,
            "cycle": 0,
            "phase": "adam",
            "history": [],
            "diagnostics": [],
            "rad_history": rad,
            "training_seconds": 0.0,
            "observer_seconds": 0.0,
            "save_seconds": 0.0,
        }
        write_json(trial / "trial_settings.json", expected)
    tensors = runtime.prepare(arrays)
    updates = 0
    while state["phase"] == "adam" and state["iteration"] < settings.cap:
        if target_reached(state["history"], settings):
            break
        completed = state["iteration"]
        cycle = completed // settings.sampling_every
        if cycle != state["cycle"]:
            arrays = source.training(plan, cycle)
            if arm == "supervised_only":
                arrays["pde"] = permitted_lhs(
                    plan, settings.pde_count, seed, "pde", cycle
                )
            else:
                arrays["pde"], record = renew_pde(runtime, plan, cycle)
                state["rad_history"].append(record)
            tensors = runtime.prepare(arrays)
            state["cycle"] = cycle
        start = time.perf_counter()
        if (
            arm in {"loss_sa", "ff_loss_sa"}
            and completed
            and completed % settings.sa_every == 0
        ):
            runtime.sa_update(tensors)
        runtime.step(tensors, completed)
        state["training_seconds"] += time.perf_counter() - start
        state["iteration"] += 1
        updates += 1
        it = state["iteration"]
        if it % settings.check_every == 0 or it == settings.cap:
            start = time.perf_counter()
            validation = observed_metrics(
                fixed["validation_y"], runtime.predict(fixed["validation_x"])
            )
            state["history"].append(
                {
                    "iteration": it,
                    "validation": validation,
                    "coefficient": (
                        float(runtime.coefficient) if runtime.inverse else None
                    ),
                }
            )
            log(
                f"{settings.equation} {arm} {seed} {it} 검증 L1={validation['relative_l1']} L2={validation['relative_l2']}"
            )
            state["observer_seconds"] += time.perf_counter() - start
        if it % settings.diagnostic_every == 0 or it == settings.cap:
            start = time.perf_counter()
            diagnostic_arrays = {k: v[: min(len(v), 32)] for k, v in arrays.items()}
            state["diagnostics"].append(
                runtime.observe(runtime.prepare(diagnostic_arrays), it)
            )
            state["observer_seconds"] += time.perf_counter() - start
        if checkpoint_due(it, settings.cap) or (
            update_limit is not None and updates >= update_limit
        ):
            start = time.perf_counter()
            save_state(trial, runtime, arrays, fixed, state)
            if resume_zip:
                resume_slot(root, trial, resume_zip)
            state["save_seconds"] += time.perf_counter() - start
        if update_limit is not None and updates >= update_limit:
            return {
                "status": "interrupted_probe",
                "trial_id": identity,
                "iteration": it,
                "optimizer_updates_this_call": updates,
            }
    evaluation = (
        {"x": fixed["evaluation_x"], "y": fixed["evaluation_y"]}
        if "evaluation_x" in fixed
        else None
    )
    validation = {"x": fixed["validation_x"], "y": fixed["validation_y"]}
    constraints = {
        k.removeprefix("constraint_"): v
        for k, v in fixed.items()
        if k.startswith("constraint_")
    }
    if state["phase"] == "adam":
        state["phase"] = "adam_done"
        runtime.save(trial / "adam.npz")
        save_state(trial, runtime, arrays, fixed, state)
        if resume_zip:
            resume_slot(root, trial, resume_zip)
    runtime.load(
        trial / "adam.npz"
        if (trial / "adam.npz").exists()
        else trial / "checkpoint.npz"
    )
    final_evaluation_start = time.perf_counter()
    adam = evaluate(
        runtime, evaluation, fixed["physics"], state["iteration"], "adam", constraints
    )
    adam["validation"] = observed_metrics(
        validation["y"], runtime.predict(validation["x"])
    )
    adam["normalized_groups"] = (
        runtime.groups(tensors, arm != "supervised_only").numpy().tolist()
    )
    adam["optimizer_updates"] = state["iteration"]
    adam_evaluation_seconds = time.perf_counter() - final_evaluation_start
    state["phase"] = "adam_done"
    before = runtime.snapshot()
    start = time.perf_counter()
    lb = None
    solver_seconds = 0.0
    if settings.lbfgs_maxiter:
        solver = runtime.lbfgs(tensors, state["iteration"])
        solver_seconds = time.perf_counter() - start
        lb = evaluate(
            runtime,
            evaluation,
            fixed["physics"],
            state["iteration"],
            "lbfgs",
            constraints,
        )
        lb["validation"] = observed_metrics(
            validation["y"], runtime.predict(validation["x"])
        )
        lb["normalized_groups"] = (
            runtime.groups(tensors, arm != "supervised_only").numpy().tolist()
        )
        lb["solver"] = solver
        runtime.save(trial / "lbfgs.npz")
        a = adam["validation"]
        b = lb["validation"]
        selected = (
            solver["finite"]
            and solver["objective_after"] <= solver["objective_before"]
            and b["relative_l1"] is not None
            and b["relative_l2"] is not None
            and a["relative_l1"] is not None
            and a["relative_l2"] is not None
            and max(b["relative_l1"], b["relative_l2"])
            <= max(a["relative_l1"], a["relative_l2"])
        )
        if not selected:
            for v, value in zip(runtime.variables(), before[:-2]):
                v.assign(value)
            runtime.factors = before[-2].copy()
            runtime.sa_updates = int(before[-1])
    else:
        selected = False
    runtime.save(trial / "selected.npz")
    elapsed_lb = time.perf_counter() - start
    report = {
        "status": "complete",
        "trial_id": identity,
        "equation": settings.equation,
        "inverse": settings.inverse,
        "arm": arm,
        "seed": seed,
        "generalization": plan.summary(),
        "adam": adam,
        "lbfgs": lb,
        "selected_stage": "lbfgs" if selected else "adam",
        "stop_reason": (
            "target_reached"
            if target_reached(state["history"], settings)
            else "cap_reached"
        ),
        "target_reached": target_reached(state["history"], settings),
        "history": state["history"],
        "diagnostics": state["diagnostics"],
        "field_target": settings.target,
        "stop_policy": {
            "first": settings.stop_first,
            "every": settings.check_every,
            "consecutive": settings.consecutive,
            "both_l1_l2": True,
        },
        "execution_scope": (
            "disposable_smoke" if settings.cap < settings.stop_first else "research"
        ),
        "condition": {
            "noise_fraction": settings.noise_fraction,
            "initial_coefficient": settings.initial_coefficient,
        },
        "protocol_sha": digest(
            {
                "settings": {
                    k: v
                    for k, v in settings.to_dict().items()
                    if k not in {"arms", "seeds"}
                },
                "data": source.fingerprint,
            }
        ),
        "rad_history": state["rad_history"],
        "fixed_arrays_sha": arrays_sha(fixed),
        "training_arrays_sha": arrays_sha(arrays),
        "timing": {
            "adam_seconds": state["training_seconds"],
            "observer_seconds": state["observer_seconds"],
            "save_seconds": state["save_seconds"],
            "adam_final_evaluation_seconds": adam_evaluation_seconds,
            "lbfgs_solver_seconds": solver_seconds,
            "lbfgs_stage_seconds_including_evaluation": elapsed_lb,
        },
        "optimizer_updates_this_call": updates,
        "scales": runtime.scales,
        "runtime_versions": {
            "tensorflow": runtime.tf.__version__,
            "numpy": np.__version__,
        },
        "coefficient_identification": (
            "not_estimated"
            if arm == "supervised_only"
            else "single-positive-coefficient; sensitivity audit separate from trained identifiability"
        ),
    }
    write_json(trial / "report.json", report)
    if results_zip:
        from .reporting import generate_report

        generate_report(root)
        pack(root, results_zip)
    if resume_zip and Path(resume_zip).exists():
        receipt = verify_zip(resume_zip)
        owner = json.loads((root / "contract.json").read_text(encoding="utf-8"))
        if (
            results_zip
            and receipt.get("run_id") == owner["run_id"]
            and receipt.get("trial_id") == identity
        ):
            Path(
                resume_zip
            ).unlink()  # Only this verified, completed trial's single resume slot.
    return report


def create_run(root, settings, contract_path=None):
    settings.validate()
    root = Path(root)
    if root.exists():
        raise ValueError("새 실험 작업폴더가 필요합니다.")
    root.mkdir(parents=True)
    source = DataSource(settings, contract_path)
    if contract_path:
        inputs = root / "inputs"
        inputs.mkdir()
        original = Path(contract_path)
        copied = dict(source.contract)
        copied["files"] = {}
        for family, name in source.contract["files"].items():
            file = (original.parent / name).resolve()
            target = inputs / (family + ".csv")
            shutil.copyfile(file, target)
            copied["files"][family] = target.name
        write_json(inputs / "data.json", copied)
        source = DataSource(settings, inputs / "data.json")
    write_json(root / "settings.json", settings.to_dict())
    snapshot = root / "source_snapshot/v4"
    snapshot.mkdir(parents=True)
    for file in Path(__file__).parent.glob("*.py"):
        shutil.copyfile(file, snapshot / file.name)
    write_json(
        root / "contract.json",
        {
            "format": "v4-independent-1",
            "kind": "results",
            "run_id": uuid.uuid4().hex[:8],
            "settings_sha": digest(settings.to_dict()),
            "data_fingerprint": source.fingerprint,
            "training_source_sha": training_source_digest(),
        },
    )
    return source


def run_all(
    root,
    settings,
    source,
    results_zip,
    resume_zip,
    training_authorized=False,
    log=print,
):
    if training_authorized is not True:
        raise PermissionError("학습 실행 허용이 필요합니다.")
    if resume_zip and Path(resume_zip).is_file():
        receipt = verify_zip(resume_zip)
        owner = json.loads((Path(root) / "contract.json").read_text(encoding="utf-8"))
        if receipt["kind"] != "resume" or receipt["run_id"] != owner["run_id"]:
            raise ValueError("다른 실행의 재개 ZIP입니다.")
    completed = []
    for seed in settings.seeds:
        for arm in settings.arms:
            report = run_trial(
                root, settings, arm, seed, source, resume_zip, results_zip, log
            )
            completed.append(report)
            import tensorflow as tf

            tf.keras.backend.clear_session()
    if results_zip and not Path(results_zip).is_file():
        from .reporting import generate_report

        generate_report(root)
        pack(root, results_zip)
    return completed
