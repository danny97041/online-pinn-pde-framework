"""Package the two-file release from verified evidence without changing model payloads."""

import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main(evidence_zip, validation_zips=()):
    from v3 import context

    ns = context()
    baseline = Path(evidence_zip)
    ns["u_verify_zip"](baseline, "unified_manifest.json")
    output = ROOT / "outputs"
    scratch = ROOT / "tmp"
    scratch.mkdir(exist_ok=True)
    data = Path(tempfile.mkdtemp(prefix="v3_package_", dir=scratch))
    summary = ns["release_open"](baseline, data / "contents")
    if summary["completed_trials"] != 65 or len(summary["equation_routes"]) != 9:
        raise ValueError(
            "The distribution bundle must contain the completed 65-trial / 9-equation collection"
        )
    data = data / "contents"
    with zipfile.ZipFile(baseline) as z:
        originals = {
            n: ns["hashlib"].sha256(z.read(n)).hexdigest()
            for n in z.namelist()
            if not n.endswith("/")
        }
    # Preserve imported source and its status as evidence, not the current code.
    original_source = data / "source_snapshot"
    source_id = ns["u_sha"](baseline)[:16]
    history = data / "provenance/source_history" / source_id
    if original_source.exists():
        shutil.copytree(original_source, history, dirs_exist_ok=True)
    replaced = {
        "README.md",
        "REPORT.md",
        "docs/V3_COLAB_TEST_PLAN.md",
        "provenance/distribution_source.json",
        "provenance/validation_imports.json",
        "validation/status.json",
        "validation/current.json",
        "validation/patch_local.json",
        "validation/wave_handoff_local.json",
        "validation/form_local.json",
        "validation/source_current_local.json",
        "validation/submitted_colab_audit.json",
    }
    for name in replaced:
        previous = data / name
        if previous.exists():
            saved = history / "previous_payloads" / name
            saved.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(previous, saved)
    baseline_current = json.loads(
        (data / "validation/current.json").read_text(encoding="utf-8")
    )
    imported = []
    immutable = {
        name: digest
        for name, digest in originals.items()
        if name not in {"unified_manifest.json", "README.md", "REPORT.md"}
        and not name.startswith(
            ("source_snapshot/", "provenance/", "validation/", "docs/")
        )
    }
    for bundle in [baseline, *map(Path, validation_zips)]:
        ns["u_verify_zip"](bundle, "unified_manifest.json")
        with zipfile.ZipFile(bundle) as archive:
            for name, digest in immutable.items():
                if ns["hashlib"].sha256(archive.read(name)).hexdigest() != digest:
                    raise ValueError(
                        "Validation ZIP contains different experiment data: " + name
                    )
            current = json.loads(archive.read("validation/current.json"))
            latest = current.get("latest_run")
            if not latest or not latest.startswith("validation/runs/"):
                raise ValueError("An independent submitted runtime record is required")
            run = json.loads(archive.read(latest))
            if not run.get("passed"):
                raise ValueError("Submitted runtime checks did not pass")
            for name in archive.namelist():
                if not name.startswith("validation/runs/") or name.endswith("/"):
                    continue
                destination = data / name
                content = archive.read(name)
                if destination.exists() and destination.read_bytes() != content:
                    raise ValueError("Conflicting validation run record: " + name)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(content)
            imported.append(
                {
                    "filename": bundle.name,
                    "sha256": ns["u_sha"](bundle),
                    "latest_run": latest,
                    "passed": run["passed"],
                    "revision": run["revision"],
                    "source_hashes": run["source_hashes"],
                }
            )
    ns["u_json"](data / "provenance/validation_imports.json", imported)
    audit = output / "V3_Validation_Audit.json"
    if audit.exists():
        audit_value = json.loads(audit.read_text(encoding="utf-8"))
        expected_digests = {item["sha256"] for item in imported}
        if {item["sha256"] for item in audit_value["archives"]} != expected_digests:
            raise ValueError("The submitted audit does not match the packaging inputs")
        shutil.copyfile(audit, data / "validation/submitted_colab_audit.json")
        notebook_record = audit_value.get("executed_source_notebook")
        if notebook_record:
            preserved_notebook = output / notebook_record["filename"]
            if ns["u_sha"](preserved_notebook) != notebook_record["sha256"]:
                raise ValueError("Executed-source notebook changed after audit")
            destination = (
                data / "provenance/validation_source" / preserved_notebook.name
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(preserved_notebook, destination)
    for path in (ROOT / "v3").glob("*.py"):
        shutil.copyfile(path, original_source / path.name)
    shutil.copyfile(ROOT / "v3/README.md", data / "README.md")
    ns["u_report_markdown"](data, summary["aggregate"])
    (data / "docs").mkdir(exist_ok=True)
    shutil.copyfile(
        ROOT / "docs/V3_COLAB_TEST_PLAN.md", data / "docs/V3_COLAB_TEST_PLAN.md"
    )
    for name in ["V3_RELEASE_CHECKLIST.md", "V3_RELEASE_NOTES.md"]:
        shutil.copyfile(ROOT / "docs" / name, data / "docs" / name)
    notebook = output / "Online_PINN_PDE_Framework_V3.ipynb"
    ns["u_json"](
        data / "provenance/distribution_source.json",
        {
            "files": {p.name: ns["u_sha"](p) for p in (ROOT / "v3").glob("*.py")},
            "notebook_sha256": ns["u_sha"](notebook),
            "source_snapshot_executed_from_zip": False,
            "imported_source_history": history.relative_to(data).as_posix(),
        },
    )
    fresh = output / "V3_Patch_Verification.json"
    if fresh.exists():
        value = json.loads(fresh.read_text(encoding="utf-8"))

        def redact(item):
            if isinstance(item, dict):
                return {k: redact(v) for k, v in item.items()}
            if isinstance(item, list):
                return [redact(v) for v in item]
            if isinstance(item, str):
                return item.replace(str(ROOT), "<local-workspace>").replace(
                    ROOT.as_posix(), "<local-workspace>"
                )
            return item

        ns["u_json"](data / "validation/patch_local.json", redact(value))
    handoff = output / "V3_Wave_Handoff_Verification.json"
    if handoff.exists():
        shutil.copyfile(handoff, data / "validation/wave_handoff_local.json")
    form_check = output / "V3_Form_Verification.json"
    if form_check.exists():
        shutil.copyfile(form_check, data / "validation/form_local.json")
    source_review = output / "V3_Source_Text_Review.json"
    if source_review.exists():
        shutil.copyfile(source_review, data / "validation/source_current_local.json")
    publication_review = output / "V3_Publication_Review.json"
    if publication_review.exists():
        shutil.copyfile(publication_review, data / "validation/publication_current_local.json")
    checks = {}
    for path in (data / "validation").glob("*.json"):
        value = json.loads(path.read_text(encoding="utf-8"))
        checks[path.name] = {
            "passed": value.get("passed"),
            "source_scope": (
                "current_local_form"
                if path.name in {"form_local.json", "source_current_local.json", "publication_current_local.json"}
                else (
                    "previous_local_patch"
                    if path.name in {"patch_local.json", "wave_handoff_local.json"}
                    else "imported_historical"
                )
            ),
            "sha256": ns["u_sha"](path),
        }
    status = {
        "revision": "publication-text-v4",
        "distinct_trials": 65,
        "equations": 9,
        "archive": "passed",
        "checks": checks,
        "colab_frontend_current_revision": "not_run",
        "overall_physics_approved": False,
        "imported_history_not_fresh_validation": True,
        "submitted_colab_runs": imported,
        "current_edit_scope": "Documentation and descriptive metadata only; submitted Colab checks preserve their executed source and revision",
    }
    ns["u_json"](data / "validation/status.json", status)
    ns["u_json"](
        data / "validation/current.json",
        {
            "latest_run": baseline_current["latest_run"],
            "current_local_check": (
                "validation/form_local.json" if form_check.exists() else None
            ),
            "submitted_runs": [item["latest_run"] for item in imported],
            "scope": "Submitted Colab runs preserve their executed revision. Current presentation checks are recorded separately.",
        },
    )
    allowed = replaced | {"unified_manifest.json"}
    unchanged = [
        n
        for n in originals
        if n not in allowed and not n.startswith("source_snapshot/")
    ]
    for name in unchanged:
        if ns["u_sha"](data / name) != originals[name]:
            raise ValueError("Original payload changed: " + name)
    target = output / "V3_Results.zip"
    if target.resolve() == baseline.resolve():
        raise ValueError("Packaging never overwrites its evidence input")
    ns["u_pack"](data, target)
    ns["u_verify_zip"](target, "unified_manifest.json")
    ns["u_json"](output / "V3_Release_Status.json", status)
    public = [p for p in (ROOT / "v3").glob("*") if p.is_file()]
    public += [
        notebook,
        ROOT / "docs/V3_RELEASE_CHECKLIST.md",
        ROOT / "docs/V3_COLAB_TEST_PLAN.md",
        ROOT / "docs/V3_RELEASE_NOTES.md",
        ROOT / ".github/workflows/v3-source.yml",
    ]
    if publication_review.exists():
        public.append(publication_review)
    public += [
        ROOT / "tools" / name
        for name in [
            "sync_v3_notebook.py",
            "review_v3_source.py",
            "test_v3_interface.py",
            "package_v3.py",
            "audit_v3_checks.py",
            "prepare_v3_publish.py",
            "test_v3_wave_handoff.py",
        ]
    ]
    with zipfile.ZipFile(output / "V3_Source.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for p in public:
            z.write(p, p.relative_to(ROOT).as_posix())
        z.writestr("README.md", (ROOT / "README.md").read_text(encoding="utf-8"))
        z.writestr(
            "RELEASE_ASSET.txt",
            "실행 노트북과 V3_Results.zip을 MyDrive/PINN에 함께 둡니다.\n",
        )
    ns["u_json"](
        output / "V3_Release_Checksums.json",
        {p.name: ns["u_sha"](p) for p in [notebook, target, output / "V3_Source.zip"]},
    )
    print(
        json.dumps(
            {
                "packaged": target.name,
                "bytes": target.stat().st_size,
                "original_payloads_preserved": len(unchanged),
                "summary_unchanged": "summary.json" in unchanged,
                "submitted_colab_runs_preserved": len(imported),
                "imported_qwen_and_rules_preserved": all(
                    k in unchanged
                    for k in ["validation/actual_llm.json", "validation/rules_api.json"]
                ),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-zip", required=True)
    parser.add_argument("--validation-zip", action="append", default=[])
    args = parser.parse_args()
    main(args.evidence_zip, args.validation_zip)
