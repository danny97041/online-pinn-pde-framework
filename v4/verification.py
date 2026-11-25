"""Explicit checks for optional live integrations, never a repeated full audit."""

from .services import Service


def integration_checks(
    root, use_llm=False, dense=False, backend="numpy", questions=(), service=None
):
    questions = tuple(q.strip() for q in questions if q.strip())
    if not (use_llm or dense or questions):
        raise ValueError(
            "Qwen·E5 연결 또는 확인할 질문을 선택하세요. 전체 검사는 실행하지 않습니다."
        )
    service = service or Service(root)
    result = {
        "scope": "selected live integrations and explicit questions only",
        "training_executed": False,
        "equation_audit_executed": False,
        "all_model_audit_executed": False,
        "semantic_accuracy_verified": False,
    }
    if dense:
        matches = service.search("V4 RAD LHS Adam L-BFGS 비교", True, backend)
        result["retrieval"] = {
            "passed": bool(matches),
            "backend": "E5/" + backend + " + BM25 + RRF",
            "matches": matches,
        }
    else:
        result["retrieval"] = {"status": "not_requested"}
    if use_llm and not questions:
        questions = (
            "wave2d 순방향 Adam의 저장 결과를 비교해줘",
            "wave2d 역문제 L-BFGS의 저장 결과를 비교해줘",
            "V4의 RAD와 LHS 설정 자료를 찾아줘",
        )
    answers = [
        {"question": q, **service.ask(q, use_llm, dense, backend)} for q in questions
    ]
    calls = sum(r["llm_called"] for r in answers)
    result["questions"] = answers
    result["question_count"] = len(answers)
    result["actual_llm_calls"] = calls
    result["qwen"] = (
        {"actual_call_confirmed": bool(calls)}
        if use_llm
        else {"status": "not_requested"}
    )
    return result
