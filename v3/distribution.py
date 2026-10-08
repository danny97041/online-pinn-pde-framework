"""Two-file distribution: read-only results, lazy prediction, explicit training.

This module is also rendered as ordinary Python in the Colab notebook. Nothing
from a downloaded result ZIP is executed as Python. TensorFlow is loaded lazily
for model prediction, model-based evaluation or explicitly authorized training.
Default results viewing never imports TensorFlow.
"""

import copy
import importlib.util
import tempfile
import time
import uuid

RESULTS_FILENAME = "V3_Results.zip"
LOCAL_MODEL = "Qwen/Qwen3.5-0.8B"


def release_open(bundle, workspace):
    bundle, workspace = Path(bundle), Path(workspace)
    u_verify_zip(bundle, "unified_manifest.json")
    with zipfile.ZipFile(bundle) as archive:
        contract = json.loads(archive.read("contract.json"))
        if contract.get("distribution_format") != "v3-two-file-1":
            raise ValueError(
                "이 노트북에는 "
                + RESULTS_FILENAME
                + " 형식의 통합 결과 ZIP을 사용하세요."
            )
        if workspace.exists():
            raise ValueError(
                "새 압축 해제 폴더가 필요합니다. 입력 ZIP은 덮어쓰지 않습니다."
            )
        archive.extractall(workspace)
    return json.loads((workspace / "summary.json").read_text(encoding="utf-8"))


def release_find(folder, explicit=""):
    if explicit:
        path = Path(explicit)
        if not path.is_absolute():
            path = Path(folder) / path
        if not path.is_file():
            raise FileNotFoundError(path.name)
        return path
    path = Path(folder) / RESULTS_FILENAME
    if path.is_file():
        return path
    for name in [
        "Online_PINN_PDE_Framework_V3_Results.zip",
        "Online_PINN_PDE_Framework_V3_Validation.zip",
    ]:
        path = Path(folder) / name
        if path.is_file():
            return path
    raise FileNotFoundError(
        str(folder) + "에 노트북과 " + RESULTS_FILENAME + "을 함께 두세요."
    )


def release_predictor(root):
    root = Path(root)
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    identities = {(r["equation"], r["arm"], r["seed"]) for r in summary["rows"]}
    cache = {}

    def predict(equation, arm, seed, points):
        if (equation, arm, seed) not in identities:
            raise ValueError("No completed model for this equation/method/seed")
        x = np.asarray(points, dtype=np.float32)
        if x.ndim != 2 or not 1 <= len(x) <= 1024 or not np.isfinite(x).all():
            raise ValueError("Coordinates must be a finite batch of 1..1024 points")
        key = (equation, arm, seed)
        if key not in cache:
            if equation == "wave2d":
                source = root / "inputs/wave_source.zip"
                parent = root / "inputs/wave_parent.zip"
                if parent.exists():
                    ns = load_wave_inference()
                    config, _, contract, _ = ns["audit_inputs"](source, parent)
                    with zipfile.ZipFile(
                        parent
                    ) as archive, tempfile.TemporaryDirectory() as scratch:
                        model = ns["audit_load_model"](
                            config,
                            contract,
                            arm,
                            seed,
                            archive,
                            f"postprocess/trials/seed_{seed}_{arm}/selected.weights.h5",
                            scratch,
                        )
                else:
                    settings = json.loads(
                        (root / "settings.json").read_text(encoding="utf-8")
                    )
                    ns = load_wave_training(source, settings)
                    config = ns["wave2d_config"]
                    ns["configure_normalization"](
                        root,
                        config,
                        ns["SOURCE_ARRAYS"],
                        ns["PROTOCOL"],
                        allow_new=False,
                    )
                    model = ns["w8_make_state"](config, arm, seed)[0]
                    model.load_weights(
                        root
                        / "postprocess/trials"
                        / f"seed_{seed}_{arm}"
                        / "selected.weights.h5"
                    )
                d = config["domain"]
                domain = [[d[k + "_min"], d[k + "_max"]] for k in ["x", "y", "t"]]
            else:
                custom = (
                    json.loads((root / "settings.json").read_text(encoding="utf-8"))
                    if (root / "settings.json").exists()
                    else None
                )
                ns = load_candidate_runtime(
                    release_training_policy(custom) if custom else None
                )
                model = ns["cu_state"](equation, arm, seed)[0]
                ns["cu_load_weights"](
                    model,
                    root
                    / "candidates"
                    / equation
                    / f"seed_{seed}_{arm}"
                    / "selected_weights.npz",
                )
                domain = ns["CANDIDATES"][equation]["domain"]
            cache[key] = model, np.asarray(domain, float)
        model, domain = cache[key]
        if (
            x.shape[1] != len(domain)
            or np.any(x < domain[:, 0])
            or np.any(x > domain[:, 1])
        ):
            raise ValueError(
                "Wrong coordinate dimension or coordinates outside physical domain"
            )
        prediction = model(x, training=False).numpy()
        if not np.isfinite(prediction).all():
            raise ValueError("Nonfinite model prediction")
        return {
            "equation": equation,
            "arm": arm,
            "seed": seed,
            "prediction": prediction.tolist(),
            "research_only": True,
        }

    return predict


def release_settings(equation):
    if equation not in U_PROBLEMS:
        raise ValueError("Unknown equation")
    return {
        "equation": equation,
        "execution": "new",
        "arms": list(U_ARMS),
        "seed": 3234,
        "cap": 100000,
        "field_target": 0.10,
        "stop_first": 15000,
        "stop_every": 1000,
        "stop_consecutive": 3,
        "lhs_period": 5000,
        "base_lr": 0.004,
        "network_ratios": [1.0, 0.75, 0.5],
        "coefficient_ratios": [0.001, 0.75, 1.0],
        "lr_knots": [0, 5000, 10000],
        "initial_coefficient_ratio": 0.5,
        "ff_features_per_bank": 32,
        "ff_scales": [0.125, 0.25, 0.5],
        "network_width": 64,
        "network_depth": 5,
        "collocation_count": 800,
        "data_count": 200,
        "initial_count": 160,
        "boundary_count_per_face": 80,
        "sa_every": 50,
        "sa_ema": 0.9,
        "sa_bounds": [0.25, 4.0],
        "lbfgs_maxiter": 500,
    }


def release_validate_settings(settings, parent_settings=None):
    s = copy.deepcopy(settings)
    defaults = release_settings(s.get("equation"))
    if set(s) - set(defaults):
        raise ValueError("Unknown settings: " + str(set(s) - set(defaults)))
    s = {**defaults, **s}
    if s["execution"] not in ["new", "resume"]:
        raise ValueError("execution: new/resume")
    if (
        not s["arms"]
        or len(s["arms"]) != len(set(s["arms"]))
        or any(a not in U_ARMS for a in s["arms"])
    ):
        raise ValueError("Unknown or duplicate comparison methods")
    for key in [
        "seed",
        "cap",
        "stop_first",
        "stop_every",
        "stop_consecutive",
        "lhs_period",
        "network_width",
        "network_depth",
        "collocation_count",
        "data_count",
        "initial_count",
        "boundary_count_per_face",
        "sa_every",
        "ff_features_per_bank",
        "lbfgs_maxiter",
    ]:
        if type(s[key]) is not int or s[key] < (
            0 if key in ["seed", "lbfgs_maxiter"] else 1
        ):
            raise ValueError("Invalid positive integer: " + key)
    if not 0 < s["field_target"] <= 1 or not 0 < s["base_lr"] < 1:
        raise ValueError("field_target must be in (0,1]; base_lr must be in (0,1)")
    if not 0 < s["initial_coefficient_ratio"] or not np.isfinite(
        s["initial_coefficient_ratio"]
    ):
        raise ValueError("Initial coefficient ratio must be positive and finite")
    if not 0 <= s["sa_ema"] < 1:
        raise ValueError("SA EMA in [0,1)")
    if (
        len(s["sa_bounds"]) != 2
        or not np.isfinite(s["sa_bounds"]).all()
        or not 0 < s["sa_bounds"][0] <= s["sa_bounds"][1]
    ):
        raise ValueError("SA bounds must be ordered positive values")
    if (
        len(s["lr_knots"]) != 3
        or any(type(v) is not int for v in s["lr_knots"])
        or s["lr_knots"][0] != 0
        or any(b <= a for a, b in zip(s["lr_knots"], s["lr_knots"][1:]))
    ):
        raise ValueError("Three increasing LR knots starting at zero required")
    for key in ["network_ratios", "coefficient_ratios"]:
        if len(s[key]) != 3 or not np.isfinite(s[key]).all() or min(s[key]) <= 0:
            raise ValueError("Three positive finite LR ratios required")
    if (
        not s["ff_scales"]
        or not np.isfinite(s["ff_scales"]).all()
        or min(s["ff_scales"]) <= 0
    ):
        raise ValueError("Positive finite FF scales required")
    if (
        s["cap"] > 1000000
        or s["network_depth"] > 20
        or s["network_width"] > 1024
        or s["ff_features_per_bank"] > 1024
        or any(
            s[k] > 250000
            for k in [
                "collocation_count",
                "data_count",
                "initial_count",
                "boundary_count_per_face",
            ]
        )
    ):
        raise ValueError("Requested resources exceed explicit safety limits")
    if s["equation"] == "wave2d":
        # Wave keeps the historical product trunk and its paired calibration.
        # Editable knobs supported by its adapter, not silently ignored settings.
        unsupported = [
            "network_width",
            "network_depth",
            "collocation_count",
            "data_count",
            "initial_count",
            "boundary_count_per_face",
            "ff_features_per_bank",
            "ff_scales",
            "sa_every",
            "sa_ema",
            "sa_bounds",
        ]
        changed = [k for k in unsupported if s[k] != defaults[k]]
        if changed:
            raise ValueError(
                "Wave adapter preserves historical architecture/sampling/SA; unsupported changes: "
                + str(changed)
            )
    if s["execution"] == "resume":
        if parent_settings is None:
            raise ValueError("Resume requires saved parent settings")
        allowed = {
            "cap",
            "field_target",
            "stop_first",
            "stop_every",
            "stop_consecutive",
            "lbfgs_maxiter",
            "execution",
            "arms",
        }
        changed = [
            k for k in defaults if k not in allowed and s[k] != parent_settings[k]
        ]
        if changed:
            raise ValueError(
                "Use new training for changed model/data/optimizer settings: "
                + str(changed)
            )
        if s["cap"] <= parent_settings.get("completed_iteration", 0):
            raise ValueError("Total update cap must exceed completed Adam iteration")
    return s


def release_training_policy(settings):
    return {
        **copy.deepcopy(U_POLICY),
        **{
            k: v
            for k, v in settings.items()
            if k not in ["equation", "execution", "arms", "seed", "lbfgs_maxiter"]
        },
    }


def release_training(
    root, settings, destination, parent=None, output_zip=None, completed_boundary=None
):
    settings = release_validate_settings(
        settings, parent.get("settings") if parent else None
    )
    destination = Path(destination)
    if destination.exists():
        raise ValueError("Choose a new output directory")
    destination.mkdir(parents=True)
    policy = release_training_policy(settings)
    u_json(destination / "settings.json", settings)
    if settings["equation"] == "wave2d":
        (destination / "inputs").mkdir()
        (destination / "inputs/wave_source.zip").write_bytes(
            (Path(root) / "inputs/wave_source.zip").read_bytes()
        )
    u_json(
        destination / "contract.json",
        {
            "distribution_format": "v3-two-file-1",
            "experiment": "v3_additional",
            "plan": {"profile": "auto"},
            "policy": policy,
        },
    )
    u_json(
        destination / "runtime.json",
        {
            "purpose": "User-selected additional experiment",
            "source_results_sha256": u_sha(Path(root) / "summary.json"),
            "fixed_evaluation_points": True,
            "input_preserved": True,
            "tensorflow_version_equality_required": False,
        },
    )
    output = (
        Path(output_zip)
        if output_zip
        else destination.with_name(destination.name + "_latest.zip")
    )
    rows = []
    run_status = "running"

    def save():
        u_json(
            destination / "training_summary.json",
            {"status": run_status, "rows": rows, "settings": settings},
        )
        report_rows = []
        updates = 0
        evaluations = 0
        for result in rows:
            previous_iteration = 0
            if parent:
                family = (
                    "trials"
                    if settings["equation"] == "wave2d"
                    else "candidates/" + settings["equation"]
                )
                previous_path = (
                    Path(parent["root"])
                    / family
                    / f"seed_{settings['seed']}_{result['arm']}"
                    / "result.json"
                )
                previous = json.loads(previous_path.read_text(encoding="utf-8"))
                previous_iteration = (
                    previous["last"]["iteration"]
                    if settings["equation"] == "wave2d"
                    else previous["adam"]["iteration"]
                )
            if settings["equation"] == "wave2d":
                a, p = result["adam"], result["selected"]
                report_rows.append(
                    {
                        "equation": "wave2d",
                        "arm": result["arm"],
                        "seed": settings["seed"],
                        "adam_iteration": a["iteration"],
                        "adam_field_l2": a["reference_metrics"]["relative_l2"],
                        "field_l2": p["reference_metrics"]["relative_l2"],
                        "adam_lambda_error": abs(a["lambda_1"] - 1.0),
                        "lambda_error": abs(p["lambda_1"] - 1.0),
                        "field_points": "historical_reference_fixed",
                    }
                )
            else:
                a, p = result["adam"], result["selected"]
                report_rows.append(
                    {
                        "equation": settings["equation"],
                        "arm": result["arm"],
                        "seed": settings["seed"],
                        "adam_iteration": a["iteration"],
                        "adam_field_l2": a["reference"]["relative_l2"],
                        "field_l2": p["reference"]["relative_l2"],
                        "adam_lambda_error": a["lambda_error"],
                        "lambda_error": p["lambda_error"],
                        "field_points": "fixed_4096_LHS",
                    }
                )
            updates += a["iteration"] - previous_iteration
            evaluations += result.get("solver", {}).get(
                "evaluations", result.get("lbfgs_evaluations", 0)
            )
        u_report(
            destination,
            report_rows,
            {"report": "complete"},
            {
                settings["equation"]: {
                    "completed_trials": len(rows),
                    "settings": settings,
                    "optimizer_updates_this_run": updates,
                    "lbfgs_evaluations_this_run": evaluations,
                }
            },
        )
        summary = json.loads((destination / "summary.json").read_text(encoding="utf-8"))
        summary.update(
            status=run_status,
            policy=policy,
            model_checkpoint_note="All completed models and Adam checkpoints are included in this ZIP",
        )
        u_json(destination / "summary.json", summary)
        u_pack(destination, output)
        if completed_boundary is not None:
            completed_boundary(destination, summary)

    save()
    if settings["equation"] == "wave2d":
        ns = load_wave_training(Path(root) / "inputs/wave_source.zip", settings)
        u_json(
            destination / "runtime_versions.json",
            {"tensorflow": ns["tf"].__version__, "numpy": np.__version__},
        )
        release_wave_train(ns, root, destination, settings, parent, save, rows)
    else:
        ns = load_candidate_runtime(policy)
        u_json(
            destination / "runtime_versions.json",
            {"tensorflow": ns["tf"].__version__, "numpy": np.__version__},
        )
        wiring = ns["cu_wiring"](settings["equation"])
        u_json(destination / "wiring.json", wiring)
        for arm in settings["arms"]:
            trial = (
                destination
                / "candidates"
                / settings["equation"]
                / f"seed_{settings['seed']}_{arm}"
            )
            parent_trial = (
                Path(parent["root"])
                / "candidates"
                / settings["equation"]
                / f"seed_{settings['seed']}_{arm}"
                if parent
                else None
            )
            result = ns["release_candidate_trial"](trial, settings, parent_trial)
            rows.append(result)
            save()
    run_status = "complete"
    save()
    return output


def release_services(root, use_llm=False, dense_rag=False, dense_backend="numpy"):
    root = Path(root)
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    rag = UnifiedRAG(root, dense=dense_rag, dense_backend=dense_backend)
    agent = UnifiedAgent(summary, rag, release_predictor(root))
    if use_llm:
        agent.enable_local_llm(LOCAL_MODEL)
    return agent, us_app(summary, rag, agent)


def release_evaluate(
    root, real_llm=False, dense_rag=False, custom_questions=None, dense_backend="numpy"
):
    from fastapi.testclient import TestClient

    total_started = time.perf_counter()
    agent, app = release_services(
        root, use_llm=real_llm, dense_rag=dense_rag, dense_backend=dense_backend
    )
    load_seconds = time.perf_counter() - total_started
    started = time.perf_counter()
    agent.force_llm = real_llm
    direct = un_evaluate(agent)
    heldout = un_evaluate(agent, paraphrases=True)
    custom = un_user_question_checks(agent, custom_questions or [])
    # LLM inspection: do not double expensive generation to test HTTP transport.
    with TestClient(app) as client:
        api = us_regression(app, agent)
        transport = client.post("/agent/ask", json={"question": "전체 보고서 알려줘"})
        api["natural_language_endpoint"] = transport.status_code == 200
    return {
        "real_llm_requested": real_llm,
        "model": LOCAL_MODEL if real_llm else None,
        "retrieval_backend": agent.rag.backend,
        "mock_llm_used": False,
        "main": direct,
        "heldout": heldout,
        "custom": custom,
        "api": api,
        "passed": direct["passed"]
        and heldout["passed"]
        and custom["passed"]
        and api["passed"]
        and api["natural_language_endpoint"],
        "seconds": time.perf_counter() - started,
        "load_seconds": load_seconds,
        "total_seconds": time.perf_counter() - total_started,
        "timing_scope": "total includes service/model loading; seconds is evaluator body only",
        "training_executed": False,
    }


def release_training_check(root, scratch_base):
    """Six trial updates plus disposable wiring probes; never benchmark results."""
    checks = []
    for eq, arm in [("poisson2d", "baseline"), ("kovasznay_inverse", "ff_loss_sa")]:
        s = release_settings(eq)
        s.update(arms=[arm], cap=2, lbfgs_maxiter=0)
        first = Path(scratch_base) / ("v3_probe_" + uuid.uuid4().hex[:8])
        release_training(root, s, first)
        completed_zip = first.with_name(first.name + "_latest.zip")
        restored = Path(scratch_base) / ("v3_probe_restore_" + uuid.uuid4().hex[:8])
        release_open(completed_zip, restored)
        r = json.loads(
            (first / "candidates" / eq / f"seed_3234_{arm}" / "result.json").read_text(
                encoding="utf-8"
            )
        )
        second = Path(scratch_base) / ("v3_probe_resume_" + uuid.uuid4().hex[:8])
        release_training(
            root,
            {**s, "execution": "resume", "cap": 3},
            second,
            {"settings": s, "root": restored},
        )
        q = json.loads(
            (second / "candidates" / eq / f"seed_3234_{arm}" / "result.json").read_text(
                encoding="utf-8"
            )
        )
        checks.append(
            {
                "equation": eq,
                "new_updates": 2,
                "additional_updates": 1,
                "resumed_from_completed_boundary_zip": True,
                "completed_zip_sha256": u_sha(completed_zip),
                "passed": q["adam"]["iteration"] == 3
                and q["heldout_sha256"] == r["heldout_sha256"],
            }
        )
    return {
        "passed": all(c["passed"] for c in checks),
        "checks": checks,
        "disposable": True,
        "trial_optimizer_updates": 6,
        "additional_disposable_wiring_probes": True,
        "scope": "Not trained trial performance",
    }


def release_wave_checkpoint_state(ns, config, arm, seed, trial, normalization_root):
    """Restore saved Wave state without requiring the old random initializer.

    Saved calibration, cycle-0 training arrays, FF matrix, Adam field metrics
    and coefficient must agree. No calibration is recomputed on a trained model.
    New training still uses the strict initialization calibration path.
    """
    trial, normalization_root = Path(trial), Path(normalization_root)
    report = json.loads((trial / "result.json").read_text(encoding="utf-8"))
    if (report["arm"], report["sampling_seed"]) != (arm, seed):
        raise ValueError("Wave checkpoint identity mismatch")
    stored = json.loads(
        (normalization_root / "normalization.json").read_text(encoding="utf-8")
    )
    if stored["policy"] != ns["NORMALIZATION_POLICY"]:
        raise ValueError("Wave normalization policy mismatch")
    record = stored["seeds"][str(seed) + "_" + ns["W8_ARMS"][arm]["kind"]]
    if record != report["loss_normalization"]:
        raise ValueError("Wave calibration differs from saved Adam report")
    arrays, _ = ns["training_lhs"](
        ns["SOURCE_ARRAYS"], config, ns["PROTOCOL"], "volume_lhs", seed, 0
    )
    train = {k: v for k, v in arrays.items() if k.endswith("_train")}
    if record["seed"] != seed or record["training_arrays_sha256"] != ns[
        "arrays_digest"
    ](train):
        raise ValueError("Wave calibration training coordinates changed")
    scales = record["group_scales"]
    if (
        set(scales) != set(ns["NORMALIZATION_POLICY"]["groups"])
        or not np.isfinite(list(scales.values())).all()
        or min(scales.values()) <= 0
    ):
        raise ValueError("Invalid saved Wave group scales")
    if (
        not np.isfinite([record["amplitude"], record["time_scale"]]).all()
        or min(record["amplitude"], record["time_scale"]) <= 0
    ):
        raise ValueError("Invalid saved Wave physical scales")
    old = ns["_cross_factory"](config, arm, seed)
    model, variables, network = old[0], old[5], old[6]
    model.loss_normalization = copy.deepcopy(record)
    model.normalization_family = ns["W8_ARMS"][arm]["kind"]
    tf = ns["tf"]
    optimizer = tf.keras.optimizers.Adam(0.004)
    optimizer.build(network)
    optimizer.lambda_optimizer = tf.keras.optimizers.Adam(4e-6)
    optimizer.lambda_optimizer.build([model.lambda_1])
    factors = tf.Variable(np.ones(3, np.float32), trainable=False)
    count = tf.Variable(0, dtype=tf.int64, trainable=False)
    checkpoint = tf.train.Checkpoint(
        model=model,
        optimizer=optimizer,
        lambda_optimizer=optimizer.lambda_optimizer,
        factors=factors,
        sa_updates=count,
    )
    state = (model, optimizer, factors, count, checkpoint, variables, network)
    checkpoint.read(os.path.relpath(trial / "state", Path.cwd())).assert_consumed()
    if int(optimizer.iterations.numpy()) != report["last"]["iteration"]:
        raise ValueError("Wave restored optimizer iteration mismatch")
    if ns["w8_ff_digest"](model) != report["ff_sha256"]:
        raise ValueError("Wave restored FF matrix mismatch")
    metrics = ns["field_metrics"](model, ns["SOURCE_ARRAYS"])
    for key, value in metrics.items():
        if not np.isclose(
            value, report["last"]["reference_metrics"][key], rtol=1e-3, atol=1e-6
        ):
            raise ValueError("Wave restored Adam field metric mismatch: " + key)
    if not np.isclose(
        float(model.lambda_1.numpy()), report["last"]["lambda_1"], rtol=1e-6, atol=1e-7
    ):
        raise ValueError("Wave restored coefficient mismatch")
    return state


def release_wave_train(ns, root, destination, settings, parent, boundary, results):
    """Retain the Wave product architecture and paired normalization contract.

    Resume restores Adam, not the L-BFGS-selected inference snapshot.
    """
    tf = ns["tf"]
    config = ns["wave2d_config"]
    arrays = ns["SOURCE_ARRAYS"]
    protocol = ns["PROTOCOL"]
    destination = Path(destination)
    if parent:
        calibration = Path(parent["root"]) / "normalization.json"
    else:
        calibration = None
    if calibration and calibration.exists():
        (destination / "normalization.json").write_bytes(calibration.read_bytes())
    ns["configure_normalization"](
        destination, config, arrays, protocol, allow_new=not bool(parent)
    )
    ns["CROSS_POLICY"]["target"] = settings["field_target"]
    ns["cross_lr"] = lambda completed: tuple(
        float(
            np.interp(completed, settings["lr_knots"], settings[k])
            * settings["base_lr"]
        )
        for k in ["network_ratios", "coefficient_ratios"]
    )

    def target(history):
        eligible = [
            r
            for r in history
            if r["iteration"] >= settings["stop_first"]
            and r["iteration"] % settings["stop_every"] == 0
        ]
        n = settings["stop_consecutive"]
        eligible = eligible[-n:]
        return (
            len(eligible) == n
            and all(
                b["iteration"] - a["iteration"] == settings["stop_every"]
                for a, b in zip(eligible, eligible[1:])
            )
            and all(
                r["validation_metrics"]["relative_l2"] <= settings["field_target"]
                for r in eligible
            )
        )

    ns["cross_target"] = target
    # The lazy factory also binds these schedules in its closure. These public
    # entries are used by this adapter; no module-global mutation is required.
    for arm in settings["arms"]:
        seed = settings["seed"]
        trial = destination / "trials" / f"seed_{seed}_{arm}"
        state = None
        first = 1
        history = []
        sampling = []
        if parent:
            previous_root = Path(parent["root"]) / "trials" / f"seed_{seed}_{arm}"
            previous = json.loads(
                (previous_root / "result.json").read_text(encoding="utf-8")
            )
            if (previous["arm"], previous["sampling_seed"]) != (arm, seed):
                raise ValueError("Wave resume identity mismatch")
            state = release_wave_checkpoint_state(
                ns, config, arm, seed, previous_root, parent["root"]
            )
            initial_digest = previous["initialization_sha256"]
            first = previous["last"]["iteration"] + 1
            history = list(previous["history"])
            sampling = list(previous["sampling"])
            if first > settings["cap"]:
                raise ValueError("Total cap must exceed parent iteration")
        else:
            state = ns["w8_make_state"](config, arm, seed)
            initial_digest = ns["w8_model_digest"](state[0])
            state[0].lambda_1.assign(settings["initial_coefficient_ratio"])
        trial.mkdir(parents=True)
        step = ns["cross_stepper"](state)
        cycle = None
        started = time.perf_counter()
        report = {
            "complete": False,
            "arm": arm,
            "sampling_seed": seed,
            "network_seed": seed,
            "cap": settings["cap"],
            "initialization_sha256": initial_digest,
            "ff_sha256": ns["w8_ff_digest"](state[0]),
            "trainable_parameters": sum(int(np.prod(v.shape)) for v in state[5]),
            "history": history,
            "sampling": sampling,
            "adaptive_updates": [],
        }
        for it in range(first, settings["cap"] + 1):
            next_cycle = max(0, (it - 1) // settings["lhs_period"])
            if next_cycle != cycle:
                selected, evidence = ns["training_lhs"](
                    arrays, config, protocol, "volume_lhs", seed, next_cycle
                )
                if (
                    sampling
                    and evidence["protected_sha256"] != sampling[0]["protected_sha256"]
                ):
                    raise ValueError("Wave heldout coordinates changed")
                if not sampling or sampling[-1] != evidence:
                    sampling.append(evidence)
                data = ns["arrays_to_tensors"](selected)["train"]
                cycle = next_cycle
            lr, lc = ns["cross_lr"](it - 1)
            state[1].learning_rate.assign(lr)
            state[1].lambda_optimizer.learning_rate.assign(lc)
            step(
                data,
                tf.constant(
                    ns["w8_weights"](arm, it - 1, state[2].numpy()), tf.float32
                ),
            )
            if ns["W8_ARMS"][arm]["sa"] and it % 50 == 0:
                ns["cross_sa"](state, data)
            if it % settings["stop_every"] == 0 or it == settings["cap"]:
                row = ns["cross_row"](state, selected, arrays, arm, it)
                row["factors"] = state[2].numpy().tolist()
                row["next_lrs"] = list(ns["cross_lr"](it))
                row["next_group_weights"] = ns["cross_groups"](
                    arm, it, state[2].numpy()
                ).tolist()
                row["elapsed_seconds"] = time.perf_counter() - started
                history.append(row)
                row["target_reached"] = target(history)
                print(
                    "wave2d",
                    seed,
                    arm,
                    it,
                    "val L2",
                    round(row["validation_metrics"]["relative_l2"], 6),
                )
                if row["target_reached"]:
                    break
        report.update(
            complete=True,
            last=history[-1],
            stop_reason=(
                "validation_target" if history[-1]["target_reached"] else "cap_reached"
            ),
        )
        ns["cross_save"](trial, state, selected, report)
        boundary()
        post = destination / "postprocess/trials" / f"seed_{seed}_{arm}"
        ns["LB_POLICY"]["refine_iterations"] = settings["lbfgs_maxiter"]
        if settings["lbfgs_maxiter"] > 0:
            previous_factory = ns["set_wave_state_factory"](
                lambda cfg, method, sampling_seed: release_wave_checkpoint_state(
                    ns, cfg, method, sampling_seed, trial, destination
                )
            )
            try:
                result = ns["lb_trial"](
                    post,
                    trial,
                    config,
                    arrays,
                    protocol,
                    arm,
                    seed,
                    settings["cap"],
                    "refine",
                )
            finally:
                ns["set_wave_state_factory"](previous_factory)
        else:
            post.mkdir(parents=True)
            (post / "selected.weights.h5").write_bytes(
                (trial / "last.weights.h5").read_bytes()
            )
            result = {
                "selected": report["last"],
                "post_selected": False,
                "scope": "L-BFGS explicitly disabled",
            }
            u_json(post / "result.json", result)
        results.append(
            {
                "equation": "wave2d",
                "arm": arm,
                "seed": seed,
                "adam": report["last"],
                "selected": result["selected"],
                "lbfgs_evaluations": result.get("solver", {}).get(
                    "objective_evaluations", 0
                )
                + result.get("objective_repeat_probe_evaluations", 0),
            }
        )
        boundary()
    return results
