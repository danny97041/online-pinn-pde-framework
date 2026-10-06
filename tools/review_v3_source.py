"""Record a source-text review without claiming model or Colab execution tests.

Run before repackaging: the current result ZIP supplies the prior source snapshot.
Only comments, docstrings and historical unused string headings are ignored in
the numerical-runtime AST comparison. Changed numerical code fails that check.
"""

import ast
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
FILES = sorted((ROOT / "v3").glob("*.py"))
NUMERICAL_MODULES = [
    "candidate_runtime.py", "wave_inference.py", "wave_runtime.py",
    "equations.py", "contracts.py",
]


class IgnoreDescriptionStrings(ast.NodeTransformer):
    def visit_Expr(self, node):
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return None
        return self.generic_visit(node)


def computational_ast(source):
    return ast.dump(IgnoreDescriptionStrings().visit(ast.parse(source)), include_attributes=False)


def main():
    previous = ROOT / "outputs/Online_PINN_PDE_Framework_V3_Complete_Results.zip"
    equivalence = {}
    unchanged_planner = {}
    with zipfile.ZipFile(previous) as archive:
        for name in NUMERICAL_MODULES:
            old = archive.read("source_snapshot/" + name).decode("utf-8")
            new = (ROOT / "v3" / name).read_text(encoding="utf-8")
            equivalence[name] = computational_ast(old) == computational_ast(new)
        for name in ["services.py", "natural_language.py"]:
            unchanged_planner[name] = archive.read("source_snapshot/" + name) == (ROOT / "v3" / name).read_bytes()
    assert all(equivalence.values()), equivalence
    assert all(unchanged_planner.values()), unchanged_planner
    for path in FILES:
        compile(path.read_text(encoding="utf-8"), str(path), "exec")
    report = {
        "scope": "Source descriptions, formatting, encoding and optional-check accounting; no model execution",
        "python_files_reviewed": len(FILES),
        "syntax_compilation": "passed",
        "numerical_ast_unchanged": equivalence,
        "planner_and_hybrid_rag_source_byte_unchanged": unchanged_planner,
        "runtime_verification": "pending",
        "training_executed": False,
        "changes": [
            "Historical escaped headings replaced by readable comments",
            "Module and factory descriptions distinguish active policies from historical helpers",
            "Archive integrity and prior evidence are not described as fresh runtime approval",
            "UTF-8 explicit for distribution JSON reads",
            "Optional training check is included in aggregate pass/fail and update accounting",
        ],
        "files_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in FILES},
    }
    target = ROOT / "outputs/V3_Source_Text_Review.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "files_sha256"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
