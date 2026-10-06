"""Regenerate the self-contained Colab notebook from readable V3 source files."""

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
    "distribution.py",
    "candidate_runtime.py",
    "wave_inference.py",
    "wave_runtime.py",
    "verification.py",
    "interface.py",
]


def main():
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": (SOURCE / "COLAB.md").read_text(encoding="utf-8").splitlines(True),
        }
    ]

    def add_code(path):
        code = path.read_text(encoding="utf-8")
        compile(code, str(path), "exec")
        cells.append(
            {
                "cell_type": "code",
                "metadata": {},
                "source": code.splitlines(True),
                "outputs": [],
                "execution_count": None,
            }
        )

    for name in MODULES:
        cells.append(
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### " + name + "\n",
                    "정의만 로드합니다. 학습은 시작하지 않습니다.\n",
                ],
            }
        )
        add_code(SOURCE / name)
        cells[-1]["metadata"] = {"cellView": "form"}
    cells.append(
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 설정 · 실행\n",
                "파일 위치를 확인하고 아래 패널의 설정 탭에서 기능을 선택하세요.\n",
            ],
        }
    )
    add_code(SOURCE / "colab_config.py")
    hashes = {
        name: hashlib.sha256((SOURCE / name).read_bytes()).hexdigest() for name in MODULES
    }
    cells[-1]["source"].append("\nV3_EXECUTED_SOURCE_HASHES = " + repr(hashes) + "\n")
    add_code(SOURCE / "colab_entry.py")
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 4,
        "cells": cells,
        "metadata": {
            "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
            "language_info": {"name": "python"},
        },
    }
    target = ROOT / "outputs/Online_PINN_PDE_Framework_V3.ipynb"
    target.parent.mkdir(exist_ok=True)
    target.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(target)


if __name__ == "__main__":
    main()
