"""Read saved results from the CLI; optional questions do not authorize training.

Usage: python -m v3 result.zip [--question ...] [--llm] [--bm25]
Hybrid retrieval is the question-mode default; --bm25 selects keyword-only search.
"""

import argparse
from pathlib import Path
import tempfile
from . import context


def main():
    parser = argparse.ArgumentParser(
        description="V3 completed results; no implicit training"
    )
    parser.add_argument("bundle", type=Path, help="Completed V3 two-file result ZIP")
    parser.add_argument(
        "--question", help="Question about the stored results and evidence"
    )
    parser.add_argument(
        "--llm", action="store_true", help="Enable optional local Qwen intent advice"
    )
    parser.add_argument(
        "--bm25", action="store_true", help="Explicit keyword-only retrieval"
    )
    args = parser.parse_args()
    ns = context()
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "results"
        ns["release_open"](args.bundle, root)
        if args.question:
            agent, _ = ns["release_services"](
                root, use_llm=args.llm, dense_rag=not args.bm25
            )
            print(agent.ask(args.question)["answer"])
        else:
            print((root / "REPORT.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
