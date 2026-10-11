"""Check public links, publication scope and packaging without running models."""

import ast
import json
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit

from prepare_v3_publish import ROOT, publication_paths
from package_v3 import PUBLIC_DOCS, RETIRED_DOCS, result_readme

HISTORICAL_NOTEBOOKS = {
    "outputs/Online_PINN_2D_Portfolio_V1_Executed.ipynb",
    "outputs/Online_PINN_2D_Portfolio_V2.ipynb",
}
CURRENT_NOTEBOOK = "outputs/Online_PINN_PDE_Framework_V3.ipynb"
FORBIDDEN_PROSE = (
    "나한테",
    "너가",
    "사용자에게 제출",
    "최소 제출물",
    "다음 수정 때",
    "화면 버전",
)


def check():
    files = publication_paths()
    paths = {p.relative_to(ROOT).as_posix() for p in files}
    missing = [p for p in paths if not (ROOT / p).is_file()]
    if missing:
        raise AssertionError("Missing publication files: " + ", ".join(missing))
    if len(paths) != len(files):
        raise AssertionError("Duplicate publication paths")
    retired = {"docs/" + name for name in RETIRED_DOCS}
    assert not paths.intersection(retired)
    assert {"docs/" + name for name in PUBLIC_DOCS} <= paths
    assert not any(p.startswith(("tmp/", "work/")) for p in paths)
    assert not any(p.startswith("outputs/") and p != CURRENT_NOTEBOOK for p in paths)

    docs = sorted(p for p in files if p.suffix == ".md")
    link_count = 0
    for path in docs:
        content = path.read_text(encoding="utf-8")
        for phrase in (*FORBIDDEN_PROSE, *RETIRED_DOCS):
            if phrase in content:
                raise AssertionError(f"Internal prose in {path.name}: {phrase}")
        for match in re.finditer(r"\[[^\]]*\]\(([^)]+)\)", content):
            target = match.group(1).strip()
            if urlsplit(target).scheme or target.startswith("#"):
                continue
            target = unquote(target.split("#", 1)[0])
            resolved = (path.parent / target).resolve()
            if not resolved.is_relative_to(ROOT):
                raise AssertionError(f"Link escapes repository: {path.name}: {target}")
            relative = resolved.relative_to(ROOT).as_posix()
            if relative not in paths | HISTORICAL_NOTEBOOKS:
                raise AssertionError(f"Unpublished link: {path.name}: {target}")
            link_count += 1

    notebook = json.loads((ROOT / CURRENT_NOTEBOOK).read_text(encoding="utf-8"))
    guide = (ROOT / "v3/COLAB.md").read_text(encoding="utf-8").split("\n", 1)[1]
    markdown = [
        "".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "markdown"
    ]
    assert "## 사용 안내\n" + guide in markdown, "Colab guide is not synchronized"
    for source in markdown:
        assert not any(name in source for name in RETIRED_DOCS)
    code_cells = [c for c in notebook["cells"] if c["cell_type"] == "code"]
    for cell in code_cells:
        ast.parse("".join(cell["source"]))
    assert all(c["outputs"] == [] and c["execution_count"] is None for c in code_cells)
    for path in files:
        if path.suffix == ".py":
            ast.parse(path.read_text(encoding="utf-8"))
    result_files = {"COLAB.md", *["docs/" + name for name in PUBLIC_DOCS]}
    for match in re.finditer(r"\[[^\]]*\]\(([^)]+)\)", result_readme()):
        target = match.group(1).split("#", 1)[0]
        if urlsplit(target).scheme or not target:
            continue
        assert target in result_files, "Broken result-archive link: " + target
    return {
        "passed": True,
        "publication_files": len(files),
        "public_markdown_files": len(docs),
        "repository_links": link_count,
        "notebook_guide_synchronized": True,
        "result_archive_manual_links": "passed",
        "model_or_training_execution": False,
        "historical_notebooks_preserved_separately": sorted(HISTORICAL_NOTEBOOKS),
    }


if __name__ == "__main__":
    print(json.dumps(check(), ensure_ascii=False, indent=2))
