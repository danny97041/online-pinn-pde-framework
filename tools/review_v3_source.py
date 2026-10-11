"""Record a source-text review without claiming model or Colab execution tests.

Run before repackaging: the current result ZIP supplies the prior source snapshot.
Comments, docstrings, report presentation and an explicit list of descriptive
string replacements are excluded from the numerical-runtime AST comparison.
All other string values and numerical code remain protected.
"""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
FILES = sorted((ROOT / "v3").glob("*.py"))
NUMERICAL_MODULES = [
    "candidate_runtime.py",
    "wave_inference.py",
    "wave_runtime.py",
    "equations.py",
    "contracts.py",
]

# Exact presentation-only replacements, identified by prior-value SHA-256.
# Contract identifiers, hashes, equations and schedules still compare verbatim.
DESCRIPTION_REPLACEMENTS = {
    "wave_runtime.py": {
        "Explicit five-method Stage 8 experiment configuration": "9e3deb587e677ab111b58a3929ac3bb5f5c829c0100313b98cc05326cce2251c",
        "Historical exploratory setting 0.004; literature provenance not independently verified": "595f411135c77abb2c249169a4bb3a5050139ad2ff66e97e4fc230207baf9b12",
    },
    "wave_inference.py": {
        "Complete: 15 trials / 30 saved model stages. Output ZIP:": "36858be384bd0529b85960969d37b72822932d89294b8fcdc3351e4070f7fb55",
        "Post-hoc synthetic Wave residual and reference-field audit": "5ae63beb5ab7d9fe8ee69960470c92df5b295dfa10245a364c6f14b966224eae",
        "Residual statistics describe the fixed diagnostic points, not every point in the domain.": "8d67611311143c180d3cd92cb765b2d5b6470b5803ed9e13ac8e9696795b55cf",
    },
    "services.py": {
        "In-process API/agent allowlist and schema regression": "f5c8ecd65dd98ebfae95ac115835ff3c9d93266f6028877402576c4c8bcd2087",
        "Stored-model inference: finite, repeated and single/batch predictions": "d00f180275aa2820de3e4a7b0f56bb17a579467925685aebba8b8949c8dbe5a5",
    },
}


class RestoreDescriptiveBaseline(ast.NodeTransformer):
    def __init__(self, module_name):
        self.replacements = DESCRIPTION_REPLACEMENTS.get(module_name, {})

    def visit_Constant(self, node):
        if isinstance(node.value, str):
            digest = hashlib.sha256(node.value.encode("utf-8")).hexdigest()
            for current, previous_digest in self.replacements.items():
                if node.value == current or digest == previous_digest:
                    return ast.copy_location(ast.Constant(current), node)
        return node


class IgnoreDescriptionStrings(ast.NodeTransformer):
    def visit_FunctionDef(self, node):
        if node.name == "set_wave_state_factory":
            # New restoration hook only. Its behavior is tested separately;
            # historical loss/gradient/sampling/solver AST remains protected.
            return None
        if node.name == "u_report_markdown":
            return None
        if node.name == "u_report":
            kept = []
            for statement in node.body:
                report_start = (
                    isinstance(statement, ast.Assign)
                    and any(
                        isinstance(t, ast.Name) and t.id == "lines"
                        for t in statement.targets
                    )
                ) or (
                    isinstance(statement, ast.Expr)
                    and isinstance(statement.value, ast.Call)
                    and isinstance(statement.value.func, ast.Name)
                    and statement.value.func.id == "u_report_markdown"
                )
                if report_start:
                    break
                kept.append(statement)
            node.body = kept
        return self.generic_visit(node)

    def visit_Expr(self, node):
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return None
        return self.generic_visit(node)


def source_ast(source, module_name):
    return RestoreDescriptiveBaseline(module_name).visit(ast.parse(source))


def computational_ast(source, module_name):
    return ast.dump(
        IgnoreDescriptionStrings().visit(source_ast(source, module_name)),
        include_attributes=False,
    )


def main(baseline_zip):
    previous = Path(baseline_zip)
    equivalence = {}
    unchanged_planner = {}
    with zipfile.ZipFile(previous) as archive:
        for name in NUMERICAL_MODULES:
            old = archive.read("source_snapshot/" + name).decode("utf-8")
            new = (ROOT / "v3" / name).read_text(encoding="utf-8")
            equivalence[name] = computational_ast(old, name) == computational_ast(
                new, name
            )
        for name in ["services.py", "natural_language.py"]:
            unchanged_planner[name] = ast.dump(
                source_ast(
                    archive.read("source_snapshot/" + name).decode("utf-8"), name
                ),
                include_attributes=False,
            ) == ast.dump(
                source_ast((ROOT / "v3" / name).read_text(encoding="utf-8"), name),
                include_attributes=False,
            )
    assert all(equivalence.values()), equivalence
    # Natural-language presentation is intentionally extended in this revision.
    # Protect Qwen intent classification and the allowlisted service implementation.
    with zipfile.ZipFile(previous) as archive:
        old_nl = ast.parse(
            archive.read("source_snapshot/natural_language.py").decode("utf-8")
        )
    new_nl = ast.parse((ROOT / "v3/natural_language.py").read_text(encoding="utf-8"))
    protected_nl = ["un_llm_ask", "un_local_llm", "un_equations"]
    intent_equivalence = {
        name: ast.dump(
            next(
                n
                for n in old_nl.body
                if isinstance(n, ast.FunctionDef) and n.name == name
            ),
            include_attributes=False,
        )
        == ast.dump(
            next(
                n
                for n in new_nl.body
                if isinstance(n, ast.FunctionDef) and n.name == name
            ),
            include_attributes=False,
        )
        for name in protected_nl
    }
    assert all(intent_equivalence.values())
    for path in FILES:
        compile(path.read_text(encoding="utf-8"), str(path), "exec")
    report = {
        "scope": "배포 소스의 구문과 계산·도구 호출 코드 비교. 문서·응답 표시·명시적으로 열거한 설명 문자열은 별도 검토하며 실행 검사는 독립 보고서로 기록합니다.",
        "python_files_reviewed": len(FILES),
        "syntax_compilation": "passed",
        "numerical_ast_unchanged": equivalence,
        "report_presentation_excluded_from_ast_comparison": True,
        "new_checkpoint_factory_hook_excluded_from_ast_comparison": True,
        "planner_and_hybrid_rag_ast_unchanged": unchanged_planner,
        "qwen_intent_functions_ast_unchanged": intent_equivalence,
        "descriptive_string_replacements": {
            name: len(values) for name, values in DESCRIPTION_REPLACEMENTS.items()
        },
        "runtime_verification": "recorded_separately",
        "training_executed": False,
        "changes": [
            "공개 문서·비교 보고서·출력 안내 정리",
            "수치 코드와 도구 호출 함수의 변경 범위 확인",
            "소스·노트북 동기화 및 설명 문자열 비교",
        ],
        "files_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in FILES
        },
    }
    target = ROOT / "outputs/V3_Source_Text_Review.json"
    target.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {k: v for k, v in report.items() if k != "files_sha256"}, ensure_ascii=False
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="이전 결과 ZIP의 소스와 현재 수치 코드를 비교합니다."
    )
    parser.add_argument(
        "--baseline-zip", required=True, help="비교 기준이 되는 이전 통합 결과 ZIP 경로"
    )
    main(parser.parse_args().baseline_zip)
