"""Inspect submitted check ZIPs without executing their embedded source."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def inspect(path):
    from v3 import context

    ns = context()
    ns["u_verify_zip"](path, "unified_manifest.json")
    with zipfile.ZipFile(path) as archive:
        current = json.loads(archive.read("validation/current.json"))
        run = json.loads(archive.read(current["latest_run"]))
        summary = json.loads(archive.read("summary.json"))
        checks = run["checks"]
        language = checks["rules_or_llm_api"]
        responses = [
            item["response"]
            for name in ("main", "heldout")
            for item in language[name]["checks"]
        ]
        llm = [r for r in responses if r.get("llm_used")]
        snapshot_matches = {
            name: hashlib.sha256(archive.read("source_snapshot/" + name)).hexdigest()
            == value
            for name, value in run["source_hashes"].items()
        }
        # Runtime evidence and the ZIP snapshot may differ; report both scopes.
        local_matches = {
            name: hashlib.sha256((ROOT / "v3" / name).read_bytes()).hexdigest() == value
            for name, value in run["source_hashes"].items()
        }
        output = {
            "filename": Path(path).name,
            "sha256": ns["u_sha"](path),
            "latest_run": current["latest_run"],
            "revision": run["revision"],
            "passed": run["passed"],
            "requested": run["requested"],
            "environment": run["environment"],
            "stored_trials": summary["completed_trials"],
            "equations": len(summary["equation_routes"]),
            "source_hashes": run["source_hashes"],
            "executed_source_matches_zip_snapshot": snapshot_matches,
            "executed_source_matches_current_local": local_matches,
            "checks": {
                k: {"passed": v["passed"], "status": v["status"]}
                for k, v in checks.items()
            },
            "language": {
                "model": language["model"],
                "retrieval_backend": language["retrieval_backend"],
                "main_questions": language["main"]["question_count"],
                "heldout_questions": language["heldout"]["question_count"],
                "custom_questions": language["custom"]["question_count"],
                "llm_calls": len(llm),
                "intent_agreements": sum(r.get("planner_agreed", False) for r in llm),
                "controller_corrections": sum(
                    r.get("controller_fallback", False) for r in llm
                ),
                "api_passed": language["api"]["passed"],
            },
            "disposable_optimizer_updates": run["trial_optimizer_updates"],
        }
        if "saved_models" in checks:
            models = checks["saved_models"]
            output["saved_model_predictions"] = len(models["prediction"]["checks"])
        return output


def main(paths, source_notebook=None):
    results = [inspect(Path(path)) for path in paths]
    report = {
        "scope": "Submitted Colab run records; current presentation edits require separate local checks",
        "passed": all(r["passed"] for r in results),
        "archives": results,
    }
    if source_notebook:
        source_notebook = Path(source_notebook)
        notebook = json.loads(source_notebook.read_text(encoding="utf-8"))
        expected = results[0]["source_hashes"]
        assert all(r["source_hashes"] == expected for r in results)
        matched = {}
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            code = "".join(cell["source"])
            for name, digest in expected.items():
                marker = "\nV3_LOADED_MODULES[" + repr(name) + "] = "
                if marker not in code:
                    continue
                body = code.split("\n", 1)[1].split(marker, 1)[0]
                variants = (
                    body.encode("utf-8"),
                    body.replace("\n", "\r\n").encode("utf-8"),
                )
                matched[name] = any(
                    hashlib.sha256(v).hexdigest() == digest for v in variants
                )
        if set(matched) != set(expected) or not all(matched.values()):
            raise ValueError(
                "Notebook source does not match the submitted executed-source hashes"
            )
        preserved = ROOT / "outputs/V3_Validation_Source.ipynb"
        shutil.copyfile(source_notebook, preserved)
        report["executed_source_notebook"] = {
            "filename": preserved.name,
            "sha256": hashlib.sha256(preserved.read_bytes()).hexdigest(),
            "all_module_hashes_match": True,
        }
    target = ROOT / "outputs/V3_Validation_Audit.json"
    target.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {"passed": report["passed"], "files": [r["filename"] for r in results]},
            ensure_ascii=False,
        )
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("bundles", nargs="+", type=Path)
    parser.add_argument("--source-notebook", type=Path)
    args = parser.parse_args()
    if not main(args.bundles, args.source_notebook)["passed"]:
        raise SystemExit(1)
