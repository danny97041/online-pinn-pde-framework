from __future__ import annotations

import builtins
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "outputs" / "Online_PINN_2D_Portfolio_V2.ipynb"
V3 = ROOT / "outputs" / "Online_PINN_PDE_Framework_V3.ipynb"


def source(cell):
    return "".join(cell.get("source", []))


def find_cell(notebook, prefix):
    for index, cell in enumerate(notebook["cells"]):
        if source(cell).startswith(prefix):
            return index, cell
    raise AssertionError(f"Missing cell: {prefix}")


def execute_until_expected_runtime_failure(controller, responses, fake_gpus):
    response_iterator = iter(responses)
    original_input = builtins.input
    original_run = subprocess.run

    def fake_input(_prompt=""):
        return next(response_iterator)

    def fake_run(*_args, **_kwargs):
        stdout = "\n".join(fake_gpus)
        return SimpleNamespace(returncode=0 if fake_gpus else 1, stdout=stdout)

    builtins.input = fake_input
    subprocess.run = fake_run
    try:
        try:
            exec(compile(controller, "v3_controller", "exec"), {})
        except RuntimeError as error:
            message = str(error)
            assert "Runtime validation failed" in message
            assert "No project artifact was created or restored" in message
            return message
        raise AssertionError("Runtime mismatch did not stop the controller")
    finally:
        builtins.input = original_input
        subprocess.run = original_run


def main():
    assert V2.exists()
    notebook = json.loads(V3.read_text(encoding="utf-8"))
    assert notebook["metadata"]["v3_research_workflow"]["stages"] == 8
    assert notebook["metadata"]["v3_research_workflow"]["runtime_gate"] is True

    guide_index, guide_cell = find_cell(
        notebook, "## V3 Research Workflow Controller"
    )
    controller_index, controller_cell = find_cell(
        notebook, "# Cell 1 - V3 연구 Workflow 및 Colab 런타임 검증"
    )
    assert guide_index < controller_index
    controller = source(controller_cell)
    assert controller.index("Runtime validation failed") < controller.index(
        'PROJECT_NAME = "AI-Engineering-Portfolio-PDE-V3"'
    )
    assert '"runtime": "CPU"' in controller
    assert '"runtime": "GPU"' in controller
    assert "approved_benchmark_plan.json" in controller
    assert 'approval_confirmation != "APPROVE"' in controller

    cpu_on_gpu_message = execute_until_expected_runtime_failure(
        controller,
        responses=["2", "new"],
        fake_gpus=["NVIDIA T4"],
    )
    assert "Hardware accelerator > None" in cpu_on_gpu_message

    gpu_on_cpu_message = execute_until_expected_runtime_failure(
        controller,
        responses=["4", "new"],
        fake_gpus=[],
    )
    assert "T4 GPU" in gpu_on_cpu_message

    _, config_cell = find_cell(notebook, "# Cell 10B -")
    config_source = source(config_cell)
    assert 'if workflow_stage_enabled(2, 3, 4, 5, 6, 7, 8):' in config_source
    assert 'wave2d_config["training"]["maximum_iterations"] = 200' in config_source
    assert 'wave2d_config["method_comparison"]["seeds"] = [3234]' in config_source
    assert "Automatic method promotion is blocked" in config_source

    _, training_cell = find_cell(notebook, "# Cell 10C -")
    training_source = source(training_cell)
    assert "RUN_WAVE2D_BASELINE_TRAINING" in training_source
    assert "save_workflow_progress" in training_source
    assert "baseline_resume_manager" in training_source
    assert "tf.train.CheckpointManager" in training_source

    _, comparison_cell = find_cell(notebook, "# Cell 10F -")
    comparison_source = source(comparison_cell)
    assert "Execution verification only" in comparison_source
    assert "save_workflow_progress" in comparison_source
    assert "tf.train.CheckpointManager" in comparison_source
    assert "resume_manager.latest_checkpoint" in comparison_source
    assert "V3_Training_State" in comparison_source
    assert 'if workflow_stage_enabled(4, 7, 8):' in comparison_source

    _, final_cell = find_cell(notebook, "# Final V3 Stage Snapshot")
    final_source = source(final_cell)
    assert "online_pinn_pde_v3_stage_" in final_source
    assert '"workflow"' in final_source
    assert "latest_v3_checkpoint.json" in final_source

    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") == "code":
            compile(source(cell), f"cell_{index}", "exec")

    print(f"V2 source preserved : {V2}")
    print(f"V3 notebook valid   : {V3}")
    print("Runtime CPU gate    : PASS")
    print("Runtime GPU gate    : PASS")
    print("Stage 8 approval    : STATIC CHECK ONLY (not executed)")
    print("Checkpoint design   : STATIC CHECK ONLY (restore not executed)")
    print("Code cell syntax    : PASS")


if __name__ == "__main__":
    main()
