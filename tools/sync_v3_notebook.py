"""Generate a self-contained Colab notebook with folded, separated definitions."""

import json
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "v3"
MODULES = [
    "contracts.py",
    "equations.py",
    "natural_language.py",
    "services.py",
    "model_registry.py",
    "api_schemas.py",
    "api_services.py",
    "api_routes.py",
    "documents.py",
    "visualization.py",
    "distribution.py",
    "benchmark.py",
    "candidate_runtime.py",
    "wave_inference.py",
    "wave_runtime.py",
    "verification.py",
    "session.py",
    "interface.py",
]
TITLES = [
    "공통 계약 · ZIP 저장",
    "방정식 · 비교 보고서",
    "자연어 요청 · 도구 계획",
    "검색 · API · Agent",
    "저장 모델 목록 · 수동 등록",
    "API 요청 형식",
    "API 서비스",
    "API 라우터",
    "기술보고서 · 기능 목록 생성",
    "저장 결과 시각화",
    "모델 결과 · 추가 학습",
    "선택형 4시드 실행",
    "후보 방정식 학습 런타임",
    "Wave 모델 추론",
    "Wave 학습 · L-BFGS",
    "기능 검증",
    "Colab 저장소 연결",
    "조건부 설정 패널",
]


def main():
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 온라인 PINN PDE 프레임워크 V3\n",
                "9개 방정식 · 65개 완료 실험의 조회, 추론, 자연어 도구 호출 및 선택적 추가 학습.\n",
                "**런타임 → 모두 실행** 후 아래에서 최초 기능만 선택합니다. 상세 설정과 결과는 맨 아래 실행 패널에 있습니다.\n",
            ],
        }
    ]
    hashes = {
        name: hashlib.sha256((SOURCE / name).read_bytes()).hexdigest()
        for name in MODULES
    }

    def add_code(title, code, definitions=True):
        source = "#@title " + title + "\n" + code
        compile(source, title, "exec")
        metadata = {"cellView": "form"}
        if definitions:
            metadata.update(collapsed=True, jupyter={"source_hidden": True})
        cells.append(
            {
                "cell_type": "code",
                "metadata": metadata,
                "source": source.splitlines(True),
                "outputs": [],
                "execution_count": None,
            }
        )

    config = (SOURCE / "colab_config.py").read_text(encoding="utf-8")
    config += (
        "\nV3_EXECUTED_SOURCE_HASHES = " + repr(hashes) + "\nV3_LOADED_MODULES = {}\n"
    )
    config += (
        "\nV3_PANEL_REQUEST_ID = object()\n"
        'if isinstance(globals().get("PANEL"), dict):\n'
        '    PANEL["dashboard"].close()\n'
        "    PANEL = None\n"
        '\nif "V3_INITIAL_SETUP" in globals():\n'
        '    V3_INITIAL_SETUP["cancel"]()\n'
        "V3_INITIAL_SETUP = v3_initial_setup(V3_SELECTION_TIMEOUT)\n"
    )
    add_code("00. 최초 설정 · 60초 무입력 기본값", config)
    guide = (SOURCE / "COLAB.md").read_text(encoding="utf-8")
    cells.append(
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ("## 사용 안내\n" + guide.split("\n", 1)[1]).splitlines(True),
        }
    )
    for number, (name, title) in enumerate(zip(MODULES, TITLES), 1):
        code = (SOURCE / name).read_text(encoding="utf-8")
        code += "\nV3_LOADED_MODULES[" + repr(name) + "] = " + repr(hashes[name]) + "\n"
        add_code(f"{number:02d}. {title}", code)
    cells.append(
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 실행 패널\n",
                "정의 셀은 모두 실행 시 초기화되며, 코드가 기본적으로 접힌 상태로 표시됩니다.\n",
                "기능 선택과 결과 조회는 아래 패널에서 수행합니다.\n",
            ],
        }
    )
    add_code(
        "실행 패널 · 설정과 결과",
        (SOURCE / "colab_entry.py").read_text(encoding="utf-8"),
        definitions=False,
    )
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 4,
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "name": "python3",
                "display_name": "Python 3",
                "language": "python",
            },
            "language_info": {"name": "python"},
        },
    }
    target = ROOT / "outputs/Online_PINN_PDE_Framework_V3.ipynb"
    target.parent.mkdir(exist_ok=True)
    target.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(target)


if __name__ == "__main__":
    main()
