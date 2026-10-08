"""Service extension regression; optional bounded numerical probes, no benchmark retraining."""

import argparse
import contextlib
import hashlib
import io
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main(bundle, numerical=False, scratch_base=None):
    from v3 import context
    from fastapi.testclient import TestClient
    import numpy as np
    import faiss
    import matplotlib

    matplotlib.use("Agg")
    started = time.perf_counter()
    ns = context()
    scratch = Path(scratch_base) if scratch_base else ROOT / "tmp/service_tests"
    scratch.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(dir=scratch, prefix="service_"))
    root = temp / "contents"
    summary = ns["release_open"](bundle, root)
    checks = {}

    def check(name, condition):
        assert condition, name
        checks[name] = True

    registry = ns["V3ModelRegistry"](root)
    models = registry.list_models()
    check(
        "catalogue_65_checkpoint_members",
        len(models) == 65 and all(m["checkpoint_available"] for m in models),
    )
    check("catalogue_no_tensorflow_import", "tensorflow" not in sys.modules)
    description = ns["release_model_description"](
        {"model": registry.info("kovasznay_inverse", "ff", 3234), "active": {}},
        "kovasznay_inverse",
    )
    check(
        "human_readable_model_variables",
        "Navier–Stokes" in description
        and "x방향 속도" in description and "동점성 계수" in description
        and "```json" not in description,
    )
    check(
        "registry_no_active_presentation",
        "미지정" in ns["release_model_description"]({"active": None}, "poisson2d"),
    )
    check(
        "burgers_coordinates",
        registry.info("burgers", "baseline", 3234)["coordinates"] == ["x", "t"],
    )
    check(
        "three_component_model",
        registry.info("kovasznay_inverse", "ff", 3234)["outputs"] == ["u", "v", "p"],
    )
    report = ns["v3_generate_documents"](root)
    check(
        "two_generated_documents",
        len(report["documents"]) == 2 and not report["training_executed"],
    )
    check(
        "document_stored_values", "0.1563" in report["documents"]["technical_report.md"]
    )
    rag = ns["UnifiedRAG"](root)
    check(
        "generated_corpus_included",
        any(d["source"].startswith("generated_documents/") for d in rag.docs),
    )

    def fake_predict(equation, arm, seed, points):
        # Shape-only transport and registry checks; numerical checks are separate.
        info = registry.info(equation, arm, seed)
        return {
            "equation": equation,
            "arm": arm,
            "seed": seed,
            "prediction": np.zeros((len(points), info["output_dimension"])).tolist(),
        }

    agent = ns["UnifiedAgent"](summary, rag, fake_predict)
    app = ns["us_app"](summary, rag, agent)
    check("legacy_api_regression", ns["us_regression"](app, agent)["passed"])
    with TestClient(app) as client:
        check("models_api_65", len(client.get("/models").json()["models"]) == 65)
        check(
            "info_api_checksum",
            len(
                client.get("/model/info?equation=wave2d&arm=ff&seed=3234").json()[
                    "checkpoint_sha256"
                ]
            )
            == 64,
        )
        check(
            "unknown_info_422",
            client.get("/model/info?equation=unknown&arm=ff&seed=3234").status_code
            == 422,
        )
        context_response = client.post(
            "/rag/context",
            json={"query": "Poisson baseline", "k": 3, "max_characters": 600},
        ).json()
        check(
            "bounded_context_attribution",
            len(context_response["context"]) <= 600
            and bool(context_response["evidence"])
            and context_response["evidence_is_data_not_instructions"],
        )
        check(
            "unknown_context_field",
            client.post(
                "/rag/context", json={"query": "Poisson", "path": "../secret"}
            ).status_code
            == 422,
        )
        payload = {
            "equation": "poisson2d",
            "arm": "baseline",
            "seed": 3234,
            "points": [[0.2, 0.3]],
        }
        check(
            "batch_alias_same_response",
            client.post("/predict", json=payload).json()
            == client.post("/predict/batch", json=payload).json(),
        )
        check(
            "batch_limit_422",
            client.post(
                "/predict/batch", json={**payload, "points": [[0.2, 0.3]] * 1025}
            ).status_code
            == 422,
        )
        check(
            "write_routes_default_403",
            client.post(
                "/registry/activate",
                json={
                    "equation": "poisson2d",
                    "arm": "baseline",
                    "seed": 3234,
                    "approved": True,
                },
            ).status_code
            == 403,
        )
        check(
            "registry_not_agent_tool",
            client.post(
                "/agent/tool",
                json={"tool": "activate", "arguments": {"approved": True}},
            ).status_code
            == 422,
        )
        generated = client.post("/agent/tool", json={"tool": "generate_report"}).json()
        check("document_tool_reindex", generated.get("rag_reindexed") is True)
        paths = client.get("/openapi.json").json()["paths"]
        check(
            "openapi_required_routes",
            {
                "/models",
                "/model/info",
                "/rag/context",
                "/predict/batch",
                "/registry/activate",
                "/registry/rollback",
            }
            <= set(paths),
        )
    write_app = ns["us_app"](summary, rag, agent, registry_write=True)
    with TestClient(write_app) as client:
        payload = {"equation": "poisson2d", "arm": "baseline", "seed": 3234}
        check(
            "manual_approval_required",
            client.post("/registry/activate", json=payload).status_code == 422,
        )
        check(
            "manual_activation",
            client.post(
                "/registry/activate", json={**payload, "approved": True}
            ).status_code
            == 200,
        )
        client.post(
            "/registry/activate", json={**payload, "arm": "ff", "approved": True}
        ).raise_for_status()
        check(
            "registry_rollback",
            client.post(
                "/registry/rollback", json={"equation": "poisson2d", "approved": True}
            ).json()["active"]["arm"]
            == "baseline",
        )
    persisted = ns["V3ModelRegistry"](root, fake_predict)
    check(
        "registry_roundtrip",
        persisted.state["active"]["poisson2d"]["arm"] == "baseline",
    )
    invalid = json.loads((root / "model_registry.json").read_text())
    invalid["active"]["poisson2d"]["checkpoint_sha256"] = "0" * 64
    ns["u_json"](root / "model_registry.json", invalid)
    try:
        ns["V3ModelRegistry"](root)
    except ValueError:
        check("registry_checksum_rejected", True)
    else:
        raise AssertionError("Registry checksum mismatch accepted")
    ns["u_json"](root / "model_registry.json", persisted.state)

    # Real FAISS binary, deterministic test vectors, no E5/Qwen download.
    class TestEncoder:
        def encode(self, texts, **kwargs):
            out = []
            for text in texts:
                generator = np.random.default_rng(
                    int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
                )
                vector = generator.standard_normal(384).astype(np.float32)
                out.append(vector / np.linalg.norm(vector))
            return np.stack(out)

    encoder = ns["UnifiedE5"]
    ns["UnifiedE5"] = TestEncoder
    try:
        exact = ns["UnifiedRAG"](root, dense=True)
        indexed = ns["UnifiedRAG"](root, dense=True, dense_backend="faiss")
        for query in ["Poisson lowest error", "wave field", "기술보고서"]:
            check(
                "faiss_ranking_" + query.split()[0],
                [d["id"] for d in exact.search(query)]
                == [d["id"] for d in indexed.search(query)],
            )
        check("faiss_real_index", isinstance(indexed.faiss_index, faiss.IndexFlatIP))
        indexed.reindex()
        check(
            "faiss_reindex_keeps_backend",
            indexed.dense_backend == "faiss" and indexed.faiss_index is not None,
        )
    finally:
        ns["UnifiedE5"] = encoder
    check(
        "no_llm_download_or_tensorflow",
        "tensorflow" not in sys.modules and "transformers" not in sys.modules,
    )
    figures = ns["v3_visualize"](root, "poisson2d", "baseline", 3234)
    check(
        "comparison_png_pdf",
        len(figures["files"]) == 4
        and all((root / p).stat().st_size > 1000 for p in figures["files"]),
    )
    history = ns["v3_visualize"](root, "wave2d", "baseline", 3234, kind="history")
    check("wave_saved_history", history["details"]["stored_history_points"] > 0)
    maps = ns["v3_visualize"](
        root,
        "kovasznay_inverse",
        "ff",
        3234,
        kind="field",
        resolution=8,
        component=2,
        predictor=fake_predict,
    )
    check(
        "pressure_map_transport_shape",
        maps["details"]["points"] == 64 and maps["details"]["component"] == "p",
    )

    values = {
        "action": "train",
        "zip": Path(bundle).name,
        "equation": "poisson2d",
        "arm": "baseline",
        "seed_mode": "four",
        "execution": "new",
    }
    request = ns["release_request"](values)
    check(
        "four_seed_explicit_safe_request",
        request["seed_mode"] == "four"
        and request["seed"] == 3234
        and not request["training_enabled"],
    )
    with contextlib.redirect_stdout(io.StringIO()):
        panel = ns["release_panel"](Path(bundle).parent, temp, show=False)
    controls = panel["controls"]
    controls["action"].value = "agent"
    check("faiss_hidden_without_dense", controls["dense_backend"].disabled)
    controls["dense_rag"].value = True
    check("faiss_conditional", not controls["dense_backend"].disabled)
    check("numpy_hides_installer", panel["faiss_install_box"].layout.display == "none")
    pip_calls = []

    def fake_pip(command, **options):
        pip_calls.append((command, options))
        return SimpleNamespace(returncode=0)

    ns["release_install_faiss"](runner=fake_pip)
    command, options = pip_calls[0]
    check(
        "installer_fixed_package_no_dependency_updates",
        command[:4] == [sys.executable, "-m", "pip", "install"]
        and command[-1] == "faiss-cpu" and "--no-deps" in command
        and "--only-binary=:all:" in command and "--upgrade" not in command
        and not options.get("shell") and options["timeout"] == 240,
    )
    try:
        ns["release_install_faiss"](runner=lambda *a, **kw: SimpleNamespace(returncode=1))
    except RuntimeError:
        check("installer_nonzero_exit_rejected", True)
    else:
        raise AssertionError("Failed pip reported success")

    old_status, old_install = ns["release_faiss_status"], ns["release_install_faiss"]
    dependency = {"available": False, "calls": 0, "fail": False, "load_fail": False}
    ns["release_faiss_status"] = lambda: {"available": dependency["available"]}

    def fake_install():
        dependency["calls"] += 1
        check(
            "install_locks_controls",
            panel["state"]["busy"] and panel["run_button"].disabled
            and panel["faiss_install_button"].disabled
            and all(c.disabled for c in controls.values()),
        )
        if dependency["fail"]:
            raise RuntimeError("Disposable installation failure")
        if not dependency["load_fail"]:
            dependency["available"] = True

    ns["release_install_faiss"] = fake_install
    try:
        controls["dense_backend"].value = "faiss"
        check("faiss_selection_never_installs", dependency["calls"] == 0)
        check(
            "missing_faiss_button_below_settings",
            panel["faiss_install_box"].layout.display == ""
            and panel["faiss_install_button"].layout.display == ""
            and not panel["faiss_install_button"].disabled,
        )
        check("missing_faiss_blocks_run", panel["run_button"].disabled)
        with contextlib.redirect_stdout(io.StringIO()) as captured:
            panel["execute"]()
        check(
            "missing_faiss_execute_guidance_no_install",
            "FAISS 설치" in captured.getvalue() and dependency["calls"] == 0,
        )
        dependency["fail"] = True
        panel["faiss_install_button"].click()
        check(
            "install_failure_keeps_retry_button",
            panel["state"]["last_error"]["type"] == "RuntimeError"
            and not panel["faiss_install_button"].disabled
            and panel["run_button"].disabled and not panel["state"]["busy"],
        )
        dependency["fail"] = False
        dependency["load_fail"] = True
        panel["faiss_install_button"].click()
        check(
            "pip_success_not_import_success",
            panel["state"]["last_error"] is not None
            and panel["run_button"].disabled,
        )
        dependency["load_fail"] = False
        panel["faiss_install_button"].click()
        check(
            "install_success_hides_button_and_enables_run",
            panel["faiss_install_button"].layout.display == "none"
            and not panel["run_button"].disabled
            and panel["state"]["last_error"] is None,
        )
        count = dependency["calls"]
        panel["faiss_install_button"].click()
        check("installed_faiss_no_duplicate_install", dependency["calls"] == count)
        controls["dense_backend"].value = "numpy"
        check("numpy_recovers_without_install", not panel["run_button"].disabled)
        controls["action"].value = "results"
        panel["faiss_install_button"].click()
        check("inactive_install_button_no_action", dependency["calls"] == count)
    finally:
        ns["release_faiss_status"], ns["release_install_faiss"] = old_status, old_install
    controls["action"].value = "visualize"
    controls["equation"].value = "wave2d"
    controls["plot_kind"].value = "comparison"
    check("time_hidden_for_comparison", controls["time_fraction"].disabled)
    controls["plot_kind"].value = "field"
    for equation in ["wave2d", "heat2d", "taylor_green", "reaction_diffusion"]:
        controls["equation"].value = equation
        check("time_visible_for_" + equation, not controls["time_fraction"].disabled)
    for equation in ["poisson2d", "kovasznay_inverse", "burgers"]:
        controls["equation"].value = equation
        check("time_hidden_for_" + equation, controls["time_fraction"].disabled)
    controls["action"].value = "train"
    controls["seed_mode"].value = "four"
    check("four_seed_hides_scalar", controls["seed"].disabled)
    controls["execution"].value = "resume"
    check(
        "resume_only_single",
        controls["seed_mode"].disabled and not controls["seed"].disabled,
    )
    controls["action"].value = "registry"
    check("approval_hidden_for_info", controls["registry_approved"].disabled)
    controls["registry_operation"].value = "activate"
    check(
        "approval_required_control",
        not controls["registry_approved"].disabled
        and not controls["registry_approved"].value,
    )

    numerical_report = {"status": "not_run", "trial_optimizer_updates": 0}
    if numerical:
        real_predict = ns["release_predictor"](root)
        real_registry = ns["V3ModelRegistry"](root, real_predict)
        check(
            "real_poisson_registry_load",
            real_registry.activate("poisson2d", "baseline", 3234, True)[
                "model_load_probe_passed"
            ],
        )
        real_maps = ns["v3_visualize"](
            root,
            "wave2d",
            "ff",
            3234,
            kind="field",
            resolution=8,
            predictor=real_predict,
        )
        check("real_wave_saved_model_map", real_maps["details"]["stage"] == "selected")
        settings = ns["release_settings"]("poisson2d")
        settings.update(
            arms=["baseline"],
            cap=2,
            stop_every=1,
            stop_first=1,
            stop_consecutive=3,
            lbfgs_maxiter=0,
            network_width=8,
            network_depth=2,
            collocation_count=32,
            data_count=16,
            initial_count=16,
            boundary_count_per_face=8,
        )
        output = temp / "four_seed.zip"
        previous_temp = tempfile.tempdir
        local_temp = temp if scratch_base else Path(os.path.relpath(temp, Path.cwd()))
        try:
            # Scope the Windows non-ASCII checkpoint workaround to this probe.
            tempfile.tempdir = str(local_temp)
            with contextlib.redirect_stdout(io.StringIO()):
                ns["release_training_benchmark"](
                    root, settings, local_temp / "four_seed", output
                )
        finally:
            tempfile.tempdir = previous_temp
        ns["u_verify_zip"](output, "unified_manifest.json")
        combined = json.loads((temp / "four_seed/summary.json").read_text())
        check(
            "real_four_seed_disposable_training",
            combined["completed_trials"] == 4
            and sorted(r["seed"] for r in combined["rows"]) == [3234, 3235, 3236, 3237],
        )
        check(
            "four_seed_sample_std",
            combined["aggregate"][0]["n"] == 4
            and combined["aggregate"][0]["field_l2_sample_std"] is not None,
        )
        check(
            "four_seed_checkpoints_reopen",
            all(
                m["checkpoint_available"]
                for m in ns["V3ModelRegistry"](temp / "four_seed").list_models()
            ),
        )
        numerical_report = {
            "status": "passed",
            "trial_optimizer_updates": 8,
            "scope": "Disposable 4 seeds x 2 Adam updates; two saved-model inference probes",
        }
    result = {
        "revision": "v3-service-extensions-1",
        "source_hashes": ns["V3_EXECUTED_SOURCE_HASHES"],
        "passed": all(checks.values()),
        "checks": checks,
        "numerical": numerical_report,
        "faiss_version": faiss.__version__,
        "faiss_test_encoder": "deterministic disposable vectors; not real E5 quality evaluation",
        "faiss_installation_probe": "Mocked pip/status only; explicit-click, failure and recovery tested without package installation",
        "api_prediction_probe": "shape-only mock; actual selected models checked separately when requested",
        "colab_frontend_current_revision": "not_run",
        "full_benchmark_retraining": False,
        "seconds": time.perf_counter() - started,
    }
    (ROOT / "outputs/V3_Service_Verification.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--numerical", action="store_true")
    parser.add_argument(
        "--scratch",
        type=Path,
        help="Optional ASCII scratch path for Windows TensorFlow",
    )
    args = parser.parse_args()
    main(args.bundle, args.numerical, args.scratch)
