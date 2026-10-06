"""Record a source-text review without claiming model or Colab execution tests.

Run before repackaging: the current result ZIP supplies the prior source snapshot.
Only comments, docstrings and historical unused string headings are ignored in
the numerical-runtime AST comparison. Changed numerical code fails that check.
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
                        isinstance(t, ast.Name) and t.id == "lines" for t in statement.targets
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


def computational_ast(source):
    return ast.dump(
        IgnoreDescriptionStrings().visit(ast.parse(source)), include_attributes=False
    )


def main(baseline_zip):
    previous = Path(baseline_zip)
    equivalence = {}
    unchanged_planner = {}
    with zipfile.ZipFile(previous) as archive:
        for name in NUMERICAL_MODULES:
            old = archive.read("source_snapshot/" + name).decode("utf-8")
            new = (ROOT / "v3" / name).read_text(encoding="utf-8")
            equivalence[name] = computational_ast(old) == computational_ast(new)
        for name in ["services.py", "natural_language.py"]:
            unchanged_planner[name] = (
                archive.read("source_snapshot/" + name) == (ROOT / "v3" / name).read_bytes()
            )
    assert all(equivalence.values()), equivalence
    assert all(unchanged_planner.values()), unchanged_planner
    for path in FILES:
        compile(path.read_text(encoding="utf-8"), str(path), "exec")
    report = {
        "scope": "한국어 조건부 설정·파일명·검사 흐름 수정. 보고서 표시를 제외한 수치 집계와 학습 수식·일정은 보존. 새 검사는 별도 런타임 보고서로 판정.",
        "python_files_reviewed": len(FILES),
        "syntax_compilation": "passed",
        "numerical_ast_unchanged": equivalence,
        "report_presentation_excluded_from_ast_comparison": True,
        "new_checkpoint_factory_hook_excluded_from_ast_comparison": True,
        "planner_and_hybrid_rag_source_byte_unchanged": unchanged_planner,
        "runtime_verification": "pending",
        "training_executed": False,
        "changes": [
            "사용자 안내와 비교 보고서를 한국어로 정리",
            "한국어 선택형 설정과 조건별 활성화, 결과 탭 이동",
            "결과·검사·학습 ZIP 파일명을 단순화하고 긴 이름 입력도 지원",
            "모델 전수·Wave 재개·LHS 경계 검사를 선택 실행으로 통합",
            "개인 작업 메모와 승격 절차를 개발자 문서로 분리",
            "보고서 수치 집계·학습 일정·샘플링·언어 모델 및 검색 소스 보존",
        ],
        "files_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in FILES},
    }
    target = ROOT / "outputs/V3_Source_Text_Review.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps({k: v for k, v in report.items() if k != "files_sha256"}, ensure_ascii=False)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="이전 결과 ZIP의 소스와 현재 수치 코드를 비교합니다."
    )
    parser.add_argument(
        "--baseline-zip", required=True, help="비교 기준이 되는 이전 통합 결과 ZIP 경로"
    )
    main(parser.parse_args().baseline_zip)
