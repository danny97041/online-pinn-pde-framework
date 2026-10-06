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


def main(evidence_zip):
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
        shutil.copytree(original_source, history)
    for name in ["status.json", "current.json"]:
        previous = data / "validation" / name
        if previous.exists():
            shutil.copyfile(previous, history / name)
    for path in (ROOT / "v3").glob("*.py"):
        shutil.copyfile(path, original_source / path.name)
    shutil.copyfile(ROOT / "v3/README.md", data / "README.md")
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
    checks = {}
    for path in (data / "validation").glob("*.json"):
        value = json.loads(path.read_text(encoding="utf-8"))
        checks[path.name] = {
            "passed": value.get("passed"),
            "source_scope": (
                "current_local_patch"
                if path.name in {"patch_local.json", "wave_handoff_local.json"}
                else "imported_historical"
            ),
            "sha256": ns["u_sha"](path),
        }
    status = {
        "revision": "korean-panel-v1",
        "distinct_trials": 65,
        "equations": 9,
        "archive": "passed",
        "checks": checks,
        "colab_frontend_current_revision": "not_run",
        "main_promoted": False,
        "overall_physics_approved": False,
        "imported_history_not_fresh_validation": True,
    }
    ns["u_json"](data / "validation/status.json", status)
    ns["u_json"](
        data / "validation/current.json",
        {
            "latest_run": "validation/patch_local.json" if fresh.exists() else None,
            "scope": "See per-check runtime, source hashes and not_run markers; Colab frontend pending",
        },
    )
    allowed = {
        "unified_manifest.json",
        "README.md",
        "provenance/distribution_source.json",
        "validation/status.json",
        "validation/current.json",
    }
    unchanged = [
        n for n in originals if n not in allowed and not n.startswith("source_snapshot/")
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
        ROOT / ".github/workflows/v3-source.yml",
    ]
    public += [
        ROOT / "tools" / name
        for name in [
            "sync_v3_notebook.py",
            "review_v3_source.py",
            "test_v3_interface.py",
            "package_v3.py",
            "test_v3_wave_handoff.py",
        ]
    ]
    with zipfile.ZipFile(output / "V3_Source.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for p in public:
            z.write(p, p.relative_to(ROOT).as_posix())
        z.writestr("README.md", (ROOT / "README.md").read_text(encoding="utf-8"))
        z.writestr(
            "RELEASE_ASSET.txt", "실행 노트북과 V3_Results.zip을 MyDrive/PINN에 함께 둡니다.\n"
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
    main(parser.parse_args().evidence_zip)
