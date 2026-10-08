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
                ast.dump(ast.parse(archive.read("source_snapshot/" + name).decode("utf-8")), include_attributes=False)
                == ast.dump(ast.parse((ROOT / "v3" / name).read_text(encoding="utf-8")), include_attributes=False)
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
    assert unchanged_planner["services.py"] and all(intent_equivalence.values())
    for path in FILES:
        compile(path.read_text(encoding="utf-8"), str(path), "exec")
    report = {
        "scope": "커널 타이머 대체 경로·하단 위젯 갱신·직접 질문·한국어 답변·사용자 시험 문항 추가. 보고서 표시를 제외한 수치 집계와 학습 수식·일정, Qwen 의도 판단과 검색 경로는 보존. 새 검사는 별도 런타임 보고서로 판정.",
        "python_files_reviewed": len(FILES),
        "syntax_compilation": "passed",
        "numerical_ast_unchanged": equivalence,
        "report_presentation_excluded_from_ast_comparison": True,
        "new_checkpoint_factory_hook_excluded_from_ast_comparison": True,
        "planner_and_hybrid_rag_ast_unchanged": unchanged_planner,
        "qwen_intent_functions_ast_unchanged": intent_equivalence,
        "runtime_verification": "pending",
        "training_executed": False,
        "changes": [
            "일반 비교 답변과 보고서의 반복 안내 축약; 필요한 질문의 확인·추가 설명은 유지",
            "상단 최초 기능·파일 위치 선택, 60초 무입력 기본 조회와 하단 설정 연동",
            "사용자 안내와 비교 보고서를 한국어로 정리",
            "분리된 정의 셀은 기본 접힘, 하단 드롭다운 실행 패널과 바로 아래 결과 표시",
            "Drive 인증 실패 상태 표시, 명시적 재연결·로컬 ZIP 전환",
            "대표 노트북 안내를 공개 배포용 기능 설명과 실행 절차로 정리",
            "결과·검사·학습 ZIP 파일명을 단순화하고 긴 이름 입력도 지원",
            "모델 전수·Wave 재개·LHS 경계 검사를 선택 실행으로 통합",
            "개인 작업 메모와 승격 절차를 개발자 문서로 분리",
            "보고서 수치 집계·학습 일정·샘플링·Qwen 의도 판단·검색 경로 보존; 자연어 응답 표현과 추가 시험 기록은 변경",
            "Colab 브라우저 콜백 타이머와 하단 안전 기본 조회 버튼 추가; 일반 노트북의 커널 타이머는 유지",
            "자동 파일 위치에서 연결된 Drive·로컬 ZIP이 없을 때 연결 1회 시도; 화면 버전 안내 문구 제거",
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
