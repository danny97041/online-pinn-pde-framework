"""Regenerate the self-contained Colab notebook from readable V3 source files."""

import json
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

    add_code(SOURCE / "colab_config.py")
    for name in MODULES:
        cells.append(
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["### " + name + "\n", "정의만 로드합니다. 학습은 시작하지 않습니다.\n"],
            }
        )
        add_code(SOURCE / name)
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
    target = ROOT / "outputs/Online_PINN_PDE_Framework_V3_Complete.ipynb"
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(target)


if __name__ == "__main__":
    main()
