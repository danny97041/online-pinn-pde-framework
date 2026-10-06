"""Readable V3 modules shared with the self-contained Colab notebook.

The modules intentionally share one bounded namespace to retain the historical
TensorFlow helper contracts. Only the fixed module list below is evaluated from
this trusted local package. This is not execution of code from a result ZIP.
"""

from pathlib import Path


def context():
    """Load the fixed source modules; optional numerical/model imports stay lazy."""
    namespace = {"__name__": "pinn_v3_runtime"}
    directory = Path(__file__).resolve().parent
    modules = [
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
    for name in modules:
        source = (directory / name).read_text(encoding="utf-8")
        exec(compile(source, str(directory / name), "exec"), namespace)
    import hashlib

    namespace["V3_EXECUTED_SOURCE_HASHES"] = {
        name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in modules
    }
    return namespace
