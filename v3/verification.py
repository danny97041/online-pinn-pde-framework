"""Scoped deployment inspections; saved models and stopping policies stay intact."""


def release_model_check(root):
    """All selected models: repeat/batch checks and reference L2 reconstruction."""
    import io

    summary = json.loads((Path(root) / "summary.json").read_text(encoding="utf-8"))
    predictor = release_predictor(root)
    regression = us_prediction_regression(predictor, summary["rows"])
    runtime = load_candidate_runtime(summary.get("policy", U_POLICY))
    wave_arrays = None
    if any(r["equation"] == "wave2d" for r in summary["rows"]):
        with zipfile.ZipFile(Path(root) / "inputs/wave_source.zip") as z:
            with np.load(
                io.BytesIO(z.read("training/wave2d_data.npz")), allow_pickle=False
            ) as a:
                wave_arrays = {k: a[k].copy() for k in ["X_exact", "u_exact"]}
    rows = []
    for row in summary["rows"]:
        eq, arm, seed = row["equation"], row["arm"], row["seed"]
        if eq == "wave2d":
            points, exact = wave_arrays["X_exact"], wave_arrays["u_exact"]
        else:
            points = runtime["cu_lhs"](
                runtime["CANDIDATES"][eq]["domain"], 4096, np.random.default_rng(739921)
            )
            exact = runtime["candidate_field"](eq, points.astype(float))
        prediction = np.concatenate(
            [
                np.asarray(
                    predictor(eq, arm, seed, points[i : i + 1024])["prediction"], float
                )
                for i in range(0, len(points), 1024)
            ]
        )
        exact = np.asarray(exact, float).reshape(prediction.shape)
        if eq == "wave2d":
            l2 = float(np.linalg.norm(prediction - exact) / np.linalg.norm(exact))
        else:
            # Candidate reports use the worst output component, not a combined
            # norm that could hide pressure or velocity errors.
            l2 = float(
                np.max(
                    np.sqrt(
                        np.sum((prediction - exact) ** 2, axis=0)
                        / np.sum(exact**2, axis=0)
                    )
                )
            )
        rows.append(
            {
                "equation": eq,
                "arm": arm,
                "seed": seed,
                "reference_point_count": len(points),
                "stored_l2": row["field_l2"],
                "reloaded_l2": l2,
                "passed": bool(np.isclose(l2, row["field_l2"], rtol=1e-3, atol=1e-6)),
            }
        )
        if (
            len(rows) % 10 == 0
            or len(rows) == len(summary["rows"])
            or not rows[-1]["passed"]
        ):
            print(
                "모델 검사",
                len(rows),
                "/",
                len(summary["rows"]),
                "불일치" if not rows[-1]["passed"] else "진행",
            )
    return {
        "passed": regression["passed"] and all(r["passed"] for r in rows),
        "prediction": regression,
        "reference_reconstruction": rows,
        "training_executed": False,
        "scope": "Stored inference snapshots and fixed reference-grid error reconstruction",
    }


def release_lhs_boundary_check(root):
    """Real sampler/control at 5000/5001; not 5001 optimizer updates."""
    runtime = load_candidate_runtime()
    checks = []
    for eq in U_PROBLEMS:
        if eq == "wave2d":
            continue
        cycles = [runtime["u_cycle"](it) for it in [4999, 5000, 5001]]
        a, b, c = [runtime["cu_samples"](eq, 3234, cycle) for cycle in cycles]
        changed = {
            k: not np.array_equal(b[k], c[k])
            for k in b
            if not k.startswith("label_") and k != "gauge"
        }
        heldout_a = runtime["cu_samples"](eq, 917221, 0, True)
        heldout_b = runtime["cu_samples"](eq, 917221, 1, True)
        checks.append(
            {
                "equation": eq,
                "cycles": cycles,
                "all_lhs_training_families_changed": changed,
                "pressure_gauge_anchor_not_a_spatial_sampling_family": "gauge" in b,
                "fixed_heldout_sha256": u_digest(heldout_a),
                "passed": cycles == [0, 0, 1]
                and u_digest(a) == u_digest(b)
                and all(changed.values())
                and u_digest(heldout_a) == u_digest(heldout_b),
            }
        )
    if (Path(root) / "inputs/wave_source.zip").exists():
        ns = load_wave_training(
            Path(root) / "inputs/wave_source.zip", release_settings("wave2d")
        )
        cycles = [ns["sampling_cycle"](it) for it in [4999, 5000, 5001]]
        pairs = [
            ns["training_lhs"](
                ns["SOURCE_ARRAYS"],
                ns["wave2d_config"],
                ns["PROTOCOL"],
                "volume_lhs",
                3234,
                cycle,
            )
            for cycle in cycles
        ]
        a, b, c = [v[0] for v in pairs]
        changed = {
            k: not np.array_equal(b[k], c[k])
            for k in b
            if k.startswith("X_") and k.endswith("_train")
        }
        checks.append(
            {
                "equation": "wave2d",
                "cycles": cycles,
                "all_training_families_changed": changed,
                "passed": cycles == [0, 0, 1]
                and u_digest(a) == u_digest(b)
                and all(changed.values())
                and pairs[1][1]["protected_sha256"] == pairs[2][1]["protected_sha256"],
            }
        )
    return {
        "passed": all(c["passed"] for c in checks),
        "checks": checks,
        "trial_optimizer_updates": 0,
        "scope": "Real sampler + iteration-to-cycle control at 5000/5001; no simulated optimizer counters and no full 5001-update trajectory",
    }


def release_wave_resume_check(root, scratch):
    """Historical Wave Adam resume: compare two disposable next updates."""
    root, scratch = Path(root), Path(scratch)
    source, parent_zip = (
        root / "inputs/wave_source.zip",
        root / "inputs/wave_parent.zip",
    )
    if not source.exists() or not parent_zip.exists():
        return {
            "status": "not_run",
            "passed": False,
            "reason": "Historical Wave parent checkpoint absent",
        }
    parent = scratch / ("v3_wave_probe_" + uuid.uuid4().hex[:8])
    u_verify_zip(parent_zip, "periodic_manifest.json")
    with zipfile.ZipFile(parent_zip) as z:
        z.extractall(parent)
    ns = load_wave_training(source, release_settings("wave2d"))
    ns["configure_normalization"](
        parent,
        ns["wave2d_config"],
        ns["SOURCE_ARRAYS"],
        ns["PROTOCOL"],
        allow_new=False,
    )
    arm, seed = "baseline", 3234
    trial = parent / "trials" / f"seed_{seed}_{arm}"
    report = json.loads((trial / "result.json").read_text(encoding="utf-8"))
    first = release_wave_checkpoint_state(
        ns, ns["wave2d_config"], arm, seed, trial, parent
    )
    before = ns["state_values"](first)
    second = release_wave_checkpoint_state(
        ns, ns["wave2d_config"], arm, seed, trial, parent
    )
    ns["assert_same_state"](before, ns["state_values"](second))
    next_iteration = report["last"]["iteration"] + 1
    arrays, evidence = ns["training_lhs"](
        ns["SOURCE_ARRAYS"],
        ns["wave2d_config"],
        ns["PROTOCOL"],
        "volume_lhs",
        seed,
        ns["sampling_cycle"](next_iteration),
    )
    data = ns["arrays_to_tensors"](arrays)["train"]
    for state in [first, second]:
        lr, coefficient_lr = ns["cross_lr"](next_iteration - 1)
        state[1].learning_rate.assign(lr)
        state[1].lambda_optimizer.learning_rate.assign(coefficient_lr)
        ns["cross_stepper"](state)(
            data,
            ns["tf"].constant(
                ns["w8_weights"](arm, next_iteration - 1, state[2].numpy()),
                ns["tf"].float32,
            ),
        )
    ns["assert_same_state"](ns["state_values"](first), ns["state_values"](second))
    if int(first[1].iterations.numpy()) != next_iteration:
        raise ValueError("Wave Adam iteration does not match saved history")
    return {
        "passed": True,
        "arm": arm,
        "seed": seed,
        "parent_iteration": next_iteration - 1,
        "next_iteration": next_iteration,
        "same_next_update": True,
        "adam_not_selected_lbfgs": True,
        "both_states_loaded_from_completed_zip": True,
        "new_checkpoint_write_in_this_probe": False,
        "training_data_sha256": u_digest(arrays),
        "protected_sha256": evidence["protected_sha256"],
        "trial_optimizer_updates": 2,
        "disposable": True,
    }


def release_inspection(root, scratch_base, request):
    """Independent run evidence; preserve earlier reports, including Qwen."""
    import platform
    import sys

    started = time.perf_counter()
    run_id = uuid.uuid4().hex[:8]
    scratch = Path(scratch_base) / ("v3_check_" + run_id)
    scratch.mkdir(parents=True)
    report = {
        "revision": "concise-comparison-v3",
        "run_id": run_id,
        "input_summary_sha256": u_sha(Path(root) / "summary.json"),
        "source_hashes": globals().get("V3_EXECUTED_SOURCE_HASHES", {}),
        "requested": request,
        "checks": {},
        "environment": {
            "python": platform.python_version(),
            "platform": platform.system(),
            "numpy": np.__version__,
            "tensorflow_version_equality_required": False,
        },
        "historical_evidence_is_not_fresh_validation": True,
        "full_benchmark_retraining": False,
    }
    stages = {
        "rules_or_llm_api": lambda: release_evaluate(
            root,
            request["use_llm"],
            request["dense_rag"],
            request.get("custom_questions", []),
            request.get("dense_backend", "numpy"),
        )
    }
    if request["check_models"]:
        stages["saved_models"] = lambda: release_model_check(root)
    if request["check_training"]:

        def candidate_probe():
            previous_temp = tempfile.tempdir
            try:
                # Scope the Windows workaround to disposable sequential probes.
                # No cwd change and no process-wide optimizer mutation.
                local = Path(os.path.relpath(scratch, Path.cwd()))
                tempfile.tempdir = str(local)
                return release_training_check(root, local)
            finally:
                tempfile.tempdir = previous_temp

        stages["candidate_new_resume"] = candidate_probe
        stages["wave_adam_resume"] = lambda: release_wave_resume_check(root, scratch)
        stages["lhs_boundary"] = lambda: release_lhs_boundary_check(root)
    for name, run in stages.items():
        stage_started = time.perf_counter()
        print("검사 시작:", name)
        try:
            result = run()
            result["status"] = result.get(
                "status", "passed" if result.get("passed") else "failed"
            )
        except Exception as exc:
            result = {
                "status": "failed",
                "passed": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            if (
                name == "candidate_new_resume"
                and platform.system() == "Windows"
                and type(exc).__name__ == "FailedPreconditionError"
                and "not a directory" in str(exc)
            ):
                result["status"] = "not_run_windows_checkpoint_io"
                result["reason"] = (
                    "Windows TensorFlow checkpoint writer limitation; repeat on Colab. This is not PASS."
                )
        result["elapsed_seconds_including_load"] = time.perf_counter() - stage_started
        report["checks"][name] = result
        print("검사 결과:", name, result["status"])
    report["passed"] = all(v.get("passed", False) for v in report["checks"].values())
    report["trial_optimizer_updates"] = sum(
        v.get("trial_optimizer_updates", 0) for v in report["checks"].values()
    )
    failed_training = request["check_training"] and any(
        not report["checks"][k].get("passed", False)
        for k in ["candidate_new_resume", "wave_adam_resume"]
    )
    report["training_executed"] = (
        True
        if report["trial_optimizer_updates"] > 0
        else (None if failed_training else False)
    )
    report["update_count_scope"] = (
        "Completed probe reports only; failed probes may have executed unreported updates"
    )
    report["total_seconds_including_load"] = time.perf_counter() - started
    if "tensorflow" in sys.modules:
        report["environment"]["tensorflow"] = sys.modules["tensorflow"].__version__
    target = Path(root) / "validation/runs" / (run_id + ".json")
    u_json(target, report)
    u_json(
        Path(root) / "validation/current.json",
        {
            "latest_run": target.relative_to(root).as_posix(),
            "passed": report["passed"],
            "source_hashes": report["source_hashes"],
            "scope": "Only requested checks; earlier reports remain historical",
        },
    )
    return report
