"""Build an explicit text-only Git publication list; model ZIPs remain release assets."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    paths = [ROOT / "README.md", ROOT / ".gitignore", ROOT / ".github/workflows/v3-source.yml"]
    paths += sorted(p for p in (ROOT / "v3").glob("*") if p.is_file())
    paths += [ROOT / "docs" / n for n in ["V3_RELEASE_CHECKLIST.md", "V3_COLAB_TEST_PLAN.md", "V3_RELEASE_NOTES.md"]]
    paths += [ROOT / "tools" / n for n in ["sync_v3_notebook.py", "review_v3_source.py", "test_v3_interface.py", "test_v3_wave_handoff.py", "package_v3.py", "audit_v3_checks.py", "prepare_v3_publish.py"]]
    paths += [ROOT / "outputs" / n for n in ["Online_PINN_PDE_Framework_V3.ipynb", "V3_Release_Checksums.json", "V3_Release_Status.json", "V3_Source_Text_Review.json", "V3_Form_Verification.json", "V3_Validation_Audit.json", "V3_Publication_Review.json"]]
    rows = []
    for path in paths:
        content = path.read_text(encoding="utf-8").replace("\r\n", "\n").encode("utf-8")
        rows.append({"path": path.relative_to(ROOT).as_posix(), "bytes": len(content), "git_sha": hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()})
    target = ROOT / "outputs/V3_Publish_Manifest.json"
    target.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"text_files": len(rows), "model_files_in_git": 0, "manifest": target.name}))


if __name__ == "__main__":
    main()
