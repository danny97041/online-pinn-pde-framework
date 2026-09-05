from __future__ import annotations

import copy
import json
from pathlib import Path
from textwrap import indent


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs" / "Online_PINN_2D_Portfolio_V2.ipynb"
TARGET = ROOT / "outputs" / "Online_PINN_PDE_Framework_V3.ipynb"


def source_text(cell):
    return "".join(cell.get("source", []))


def source_lines(text):
    return text.splitlines(keepends=True)


def first_line(cell):
    return source_text(cell).splitlines()[0] if source_text(cell).splitlines() else ""


def wrap_for_stages(cell, stages):
    original = source_text(cell)
    lines = original.splitlines(keepends=True)
    if not lines:
        return
    heading = lines[0]
    body = "".join(lines[1:])
    if body and not body.endswith("\n"):
        body += "\n"
    stage_values = ", ".join(str(value) for value in stages)
    label = heading.lstrip("# ").strip().replace('"', "'")
    wrapped = (
        heading
        + "\n"
        + f"if workflow_stage_enabled({stage_values}):\n"
        + indent(body, "    ")
        + "else:\n"
        + f"    print_workflow_skip(\"{label}\", ({stage_values},))\n"
    )
    cell["source"] = source_lines(wrapped)
    cell.setdefault("metadata", {})["workflow_stages"] = list(stages)


GUIDE_MARKDOWN = r"""## V3 Research Workflow Controller

이 Notebook은 연구 개발과 회귀검증 기간 동안 8단계 workflow를 사용합니다. 첫 번째 코드 셀에서 실행할 단계를 먼저 선택하며, 선택된 단계와 Colab 런타임이 일치하지 않으면 프로젝트 파일을 생성하거나 복구하기 전에 실행을 중단합니다.

현재 V3 사본에는 기존 동작을 보존한 Wave2D 실행 경로와 연구 workflow 기반 구조가 우선 통합되어 있습니다. Burgers, Kovasznay Forward/Inverse, Taylor-Green Vortex는 후속 구현·검증 단계에서 같은 contract에 추가하며, 추가 전에는 전체 PDE framework가 완료된 것으로 표시하지 않습니다.

| Stage | Purpose | Required Colab runtime |
|---:|---|---|
| 1 | Environment and artifact contract verification | CPU (`Hardware accelerator: None`) |
| 2 | Exact solution, PDE residual, BC/IC verification | CPU (`Hardware accelerator: None`) |
| 3 | Sampling, model, autodiff and loss wiring verification | CPU (`Hardware accelerator: None`) |
| 4 | Short execution-only Smoke Training | GPU (`T4 recommended`, L4 supported) |
| 5 | Checkpoint reload, inference, metrics and plotting | CPU (`Hardware accelerator: None`) |
| 6 | API, RAG, Agent and reporting regression | CPU (`Hardware accelerator: None`) |
| 7 | One-seed benchmark pilot and resource measurement | GPU (`T4 recommended`, L4 supported) |
| 8 | Explicitly approved full benchmark | GPU (`T4 recommended`, L4 supported) |

`new`와 `0`은 복구 없이 시작하며, `load`와 `1`은 Drive의 최신 V3 단계 snapshot을 복구합니다. Stage 4의 짧은 고정 iteration은 실행 가능성 확인 전용이며 수렴 또는 비교 성능을 입증하지 않습니다. Stage 8은 Stage 7 결과를 사람이 검토한 뒤 작성한 `config/approved_benchmark_plan.json`이 없으면 시작하지 않습니다.

학습 중에는 evaluation interval마다 경량 checkpoint와 진행 manifest를 저장하고, 각 단계 또는 문제 구간이 끝날 때 versioned ZIP snapshot을 생성합니다. 연구 workflow 1~8이 완료된 뒤 공개용 Notebook은 `방정식 선택 → new/load` 인터페이스로 별도 생성합니다.
"""


CONTROLLER_CODE = r'''# Cell 1 - V3 연구 Workflow 및 Colab 런타임 검증

from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
import hashlib
import importlib.util
import json
import os
import random
import shutil
import subprocess
import sys
import zipfile

import numpy as np


WORKFLOW_STAGES = {
    1: {
        "name": "environment_and_artifact_contract",
        "runtime": "CPU",
        "runtime_instruction": "Hardware accelerator: None",
        "profile": "verification",
    },
    2: {
        "name": "exact_residual_and_constraints",
        "runtime": "CPU",
        "runtime_instruction": "Hardware accelerator: None",
        "profile": "verification",
    },
    3: {
        "name": "sampling_model_autodiff_wiring",
        "runtime": "CPU",
        "runtime_instruction": "Hardware accelerator: None",
        "profile": "verification",
    },
    4: {
        "name": "execution_only_smoke_training",
        "runtime": "GPU",
        "runtime_instruction": "T4 recommended; L4 supported",
        "profile": "smoke",
    },
    5: {
        "name": "artifact_reload_inference_and_metrics",
        "runtime": "CPU",
        "runtime_instruction": "Hardware accelerator: None",
        "profile": "smoke",
    },
    6: {
        "name": "api_rag_agent_reporting_regression",
        "runtime": "CPU",
        "runtime_instruction": "Hardware accelerator: None",
        "profile": "smoke",
    },
    7: {
        "name": "benchmark_pilot",
        "runtime": "GPU",
        "runtime_instruction": "T4 recommended; L4 supported",
        "profile": "pilot",
    },
    8: {
        "name": "approved_full_benchmark",
        "runtime": "GPU",
        "runtime_instruction": "T4 recommended; L4 supported",
        "profile": "benchmark",
    },
}

WORKFLOW_PREREQUISITES = {
    1: (),
    2: (1,),
    3: (2,),
    4: (3,),
    5: (4,),
    6: (5,),
    7: (6,),
    8: (7,),
}

IMPLEMENTED_PROBLEMS = ("wave2d",)
PLANNED_PROBLEMS = (
    "wave2d",
    "burgers",
    "kovasznay_forward",
    "kovasznay_inverse",
    "taylor_green",
)

print("V3 Research Workflow")
print(f"Implemented problems: {', '.join(IMPLEMENTED_PROBLEMS)}")
print(f"Planned problems    : {', '.join(PLANNED_PROBLEMS)}")
for stage_number, stage_definition in WORKFLOW_STAGES.items():
    print(
        f"  {stage_number}: {stage_definition['name']} "
        f"[{stage_definition['runtime']} - "
        f"{stage_definition['runtime_instruction']}]"
    )

while True:
    stage_input = input("Select research workflow stage [1-8] (required): ").strip()
    if stage_input.isdigit() and int(stage_input) in WORKFLOW_STAGES:
        RESEARCH_STAGE = int(stage_input)
        break
    print("Enter one integer from 1 through 8.")

mode_aliases = {
    "1": "load",
    "load": "load",
    "0": "new",
    "new": "new",
}
while True:
    mode_input = input(
        "Select execution mode [new/load/0/1] (required): "
    ).strip().lower()
    if mode_input in mode_aliases:
        EXECUTION_MODE = mode_aliases[mode_input]
        break
    print("Enter load/1 or new/0 to continue.")

SELECTED_STAGE = WORKFLOW_STAGES[RESEARCH_STAGE]
ACTIVE_PROFILE = SELECTED_STAGE["profile"]


def _detect_nvidia_gpu():
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name",
                "--format=csv,noheader",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return []
    if result.returncode != 0:
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


detected_gpus = _detect_nvidia_gpu()
DETECTED_RUNTIME = "GPU" if detected_gpus else "CPU"

print(f"Selected stage   : {RESEARCH_STAGE} - {SELECTED_STAGE['name']}")
print(f"Execution profile: {ACTIVE_PROFILE}")
print(
    f"Required runtime : {SELECTED_STAGE['runtime']} "
    f"({SELECTED_STAGE['runtime_instruction']})"
)
print(
    "Detected runtime : "
    + (
        f"GPU ({', '.join(detected_gpus)})"
        if detected_gpus
        else "CPU (no NVIDIA GPU attached)"
    )
)

# This validation intentionally occurs before project creation, Drive restoration,
# configuration writes, or model initialization.
if DETECTED_RUNTIME != SELECTED_STAGE["runtime"]:
    if SELECTED_STAGE["runtime"] == "CPU":
        correction = (
            "Runtime > Change runtime type > Hardware accelerator > None"
        )
    else:
        correction = (
            "Runtime > Change runtime type > Hardware accelerator > T4 GPU "
            "(recommended) or L4 GPU"
        )
    raise RuntimeError(
        "Runtime validation failed. No project artifact was created or restored. "
        f"Required: {SELECTED_STAGE['runtime']}. "
        f"Detected: {DETECTED_RUNTIME}. Select {correction} and run again."
    )

if importlib.util.find_spec("google.colab") is None:
    raise RuntimeError(
        "The managed V3 research workflow requires a Google Colab runtime. "
        "No project artifact was created or restored."
    )

print("Runtime validation: PASS")

PROJECT_NAME = "AI-Engineering-Portfolio-PDE-V3"
PROJECT_DIR = Path("/content") / PROJECT_NAME
SEED = 1234
BACKUP_ROOT = Path("/content/drive/MyDrive/Online_PINN_Backups")

if EXECUTION_MODE == "new" and PROJECT_DIR.exists() and any(PROJECT_DIR.iterdir()):
    raise RuntimeError(
        "New mode requires a clean runtime because the V3 project directory "
        "already contains files. Restart the runtime or select load/1."
    )

from google.colab import drive

if EXECUTION_MODE == "load":
    drive.mount("/content/drive")

directories = [
    "config",
    "workflow/progress",
    "workflow/events",
    "training",
    "training_state",
    "model_artifacts/wave_pinn_2d",
    "model_artifacts/wave_pinn_2d_comparison_smoke",
    "model_artifacts/wave_pinn_2d_comparison_pilot",
    "model_artifacts/wave_pinn_2d_comparison_v3",
    "model_artifacts/wave_pinn_2d_initialization",
    "model_artifacts/burgers",
    "model_artifacts/kovasznay_forward",
    "model_artifacts/kovasznay_inverse",
    "model_artifacts/taylor_green",
    "problem_artifacts/smoke",
    "problem_artifacts/benchmark",
    "report_artifacts/figures",
    "data/documents",
    "rag_artifacts",
    "app/api/routes",
    "app/schemas",
    "app/services",
    "app/ml",
    "app/rag",
    "app/agent",
    "app/core",
    "app/reporting",
    "tests",
    "outputs",
]

ALLOWED_RESTORE_ROOTS = {
    "config",
    "workflow",
    "training",
    "training_state",
    "model_artifacts",
    "problem_artifacts",
    "report_artifacts",
    "rag_artifacts",
    "data",
    "app",
    "tests",
    "outputs",
}


def _find_latest_v3_bundle():
    candidates = []
    for root in (Path("/content"), BACKUP_ROOT):
        if root.exists():
            candidates.extend(root.glob("online_pinn_pde_v3_stage_*.zip"))
    candidates = sorted(
        set(candidates),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for candidate in candidates:
        try:
            with zipfile.ZipFile(candidate) as archive:
                names = {PurePosixPath(item.filename).as_posix() for item in archive.infolist()}
                if "workflow/workflow_manifest.json" in names:
                    return candidate
        except (zipfile.BadZipFile, OSError):
            print(f"Skipped invalid V3 backup: {candidate}")
    return None


def _restore_v3_bundle(bundle_path):
    restored = []
    project_root = PROJECT_DIR.resolve()
    with zipfile.ZipFile(bundle_path) as archive:
        for member in archive.infolist():
            parts = PurePosixPath(member.filename).parts
            if (
                not parts
                or ".." in parts
                or parts[0] not in ALLOWED_RESTORE_ROOTS
            ):
                continue
            target = (PROJECT_DIR / Path(*parts)).resolve()
            if target != project_root and project_root not in target.parents:
                raise ValueError(f"Unsafe artifact path: {member.filename}")
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, target.open("wb") as destination:
                shutil.copyfileobj(source, destination)
            restored.append(str(target.relative_to(PROJECT_DIR)))
    return restored


restored_bundle = None
restored_files = []
if EXECUTION_MODE == "load":
    restored_bundle = _find_latest_v3_bundle()
    if restored_bundle is None:
        raise FileNotFoundError(
            "Load mode could not find a V3 stage backup in /content or "
            "/content/drive/MyDrive/Online_PINN_Backups. Run Stage 1 with new/0 "
            "for the first V3 execution."
        )
    PROJECT_DIR.mkdir(parents=True, exist_ok=True)
    restored_files = _restore_v3_bundle(restored_bundle)

for directory in directories:
    (PROJECT_DIR / directory).mkdir(parents=True, exist_ok=True)

for package in [
    "app", "app/api", "app/api/routes", "app/schemas", "app/services",
    "app/ml", "app/rag", "app/agent", "app/core", "app/reporting",
]:
    (PROJECT_DIR / package / "__init__.py").touch(exist_ok=True)

WORKFLOW_MANIFEST_PATH = PROJECT_DIR / "workflow" / "workflow_manifest.json"


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


if WORKFLOW_MANIFEST_PATH.exists():
    workflow_manifest = json.loads(
        WORKFLOW_MANIFEST_PATH.read_text(encoding="utf-8")
    )
else:
    workflow_manifest = {
        "schema_version": 1,
        "project": PROJECT_NAME,
        "created_at": _utc_now(),
        "completed_stages": [],
        "stage_status": {},
        "stage_history": [],
    }

missing_prerequisites = [
    stage
    for stage in WORKFLOW_PREREQUISITES[RESEARCH_STAGE]
    if stage not in workflow_manifest.get("completed_stages", [])
]
if missing_prerequisites:
    raise RuntimeError(
        f"Stage {RESEARCH_STAGE} prerequisites are incomplete: "
        f"{missing_prerequisites}. No stage code was executed."
    )

if RESEARCH_STAGE == 8:
    approval_path = PROJECT_DIR / "config" / "approved_benchmark_plan.json"
    if not approval_path.exists():
        pilot_result_path = (
            PROJECT_DIR
            / "report_artifacts"
            / "wave2d_method_comparison.json"
        )
        if not pilot_result_path.exists():
            raise FileNotFoundError(
                "Stage 8 requires the completed Stage 7 pilot result."
            )
        pilot_payload = json.loads(
            pilot_result_path.read_text(encoding="utf-8")
        )
        if pilot_payload.get("status") != "complete":
            raise RuntimeError(
                "Stage 8 requires a complete Stage 7 pilot result."
            )
        available_methods = [
            "supervised_only",
            "baseline",
            "fourier",
            "self_adaptive",
            "fourier_curriculum",
            "fourier_self_adaptive",
        ]
        print("Stage 7 methods available for explicit benchmark approval:")
        for method_name in available_methods:
            print(f"  - {method_name}")
        while True:
            method_input = input(
                "Enter comma-separated methods approved for Stage 8: "
            ).strip()
            approved_methods = [
                value.strip()
                for value in method_input.split(",")
                if value.strip()
            ]
            if (
                approved_methods
                and len(approved_methods) == len(set(approved_methods))
                and set(approved_methods).issubset(available_methods)
            ):
                break
            print("Enter one or more unique methods from the displayed list.")
        approval_confirmation = input(
            "Type APPROVE to authorize the selected Stage 8 benchmark plan: "
        ).strip()
        if approval_confirmation != "APPROVE":
            raise RuntimeError(
                "Stage 8 benchmark approval was not granted. No benchmark "
                "configuration was created."
            )
        atomic_approval = {
            "schema_version": 1,
            "approved": True,
            "approved_at": _utc_now(),
            "source_stage": 7,
            "problem": "wave2d",
            "methods": approved_methods,
            "seeds": [3234, 3235, 3236, 3237],
            "automatic_promotion": False,
        }
        atomic_write_json(approval_path, atomic_approval)
        print(f"Approved benchmark plan: {approval_path}")
    else:
        existing_approval = json.loads(
            approval_path.read_text(encoding="utf-8")
        )
        if existing_approval.get("approved") is not True:
            raise RuntimeError(
                "The restored Stage 8 benchmark plan is not explicitly approved."
            )
        print(f"Approved benchmark plan restored: {approval_path}")

stage_event = {
    "stage": RESEARCH_STAGE,
    "name": SELECTED_STAGE["name"],
    "profile": ACTIVE_PROFILE,
    "execution_mode": EXECUTION_MODE,
    "required_runtime": SELECTED_STAGE["runtime"],
    "detected_runtime": DETECTED_RUNTIME,
    "detected_gpus": detected_gpus,
    "started_at": _utc_now(),
    "status": "running",
}
workflow_manifest["active_stage"] = RESEARCH_STAGE
workflow_manifest["stage_status"][str(RESEARCH_STAGE)] = "running"
workflow_manifest["stage_history"].append(stage_event)
workflow_manifest["updated_at"] = _utc_now()
atomic_write_json(WORKFLOW_MANIFEST_PATH, workflow_manifest)


def workflow_stage_enabled(*stages):
    return RESEARCH_STAGE in {int(stage) for stage in stages}


def print_workflow_skip(label, stages):
    allowed = ",".join(str(stage) for stage in stages)
    print(
        f"Workflow skip: {label} "
        f"(selected Stage {RESEARCH_STAGE}; active Stages {allowed})"
    )


def save_workflow_progress(problem, case, iteration, payload=None):
    record = {
        "stage": RESEARCH_STAGE,
        "profile": ACTIVE_PROFILE,
        "problem": str(problem),
        "case": str(case),
        "iteration": int(iteration),
        "saved_at": _utc_now(),
    }
    if payload:
        record.update(payload)
    safe_name = f"{problem}_{case}".replace("/", "_").replace(" ", "_")
    path = PROJECT_DIR / "workflow" / "progress" / f"{safe_name}.json"
    atomic_write_json(path, record)
    if Path("/content/drive/MyDrive").exists():
        live_path = (
            BACKUP_ROOT
            / "V3_Live_Progress"
            / f"stage_{RESEARCH_STAGE:02d}"
            / f"{safe_name}.json"
        )
        atomic_write_json(live_path, record)
    return path


def mirror_file_to_drive(local_path, relative_live_path):
    local_path = Path(local_path)
    if not local_path.exists() or not Path("/content/drive/MyDrive").exists():
        return None
    destination = BACKUP_ROOT / "V3_Training_State" / relative_live_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copy2(local_path, temporary)
    temporary.replace(destination)
    return destination


def restore_file_from_drive(relative_live_path, local_path):
    source = BACKUP_ROOT / "V3_Training_State" / relative_live_path
    local_path = Path(local_path)
    if not source.exists():
        return False
    local_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, local_path)
    return True


RUN_WAVE2D_BASELINE_TRAINING = RESEARCH_STAGE == 4
REUSE_VERIFIED_WAVE2D_BASELINE = (
    EXECUTION_MODE == "load" and RESEARCH_STAGE in {4, 5, 6}
)
RUN_WAVE2D_METHOD_COMPARISON = RESEARCH_STAGE in {4, 7, 8}
REUSE_COMPLETED_WAVE2D_METHODS = EXECUTION_MODE == "load"
MAX_NEW_COMPARISON_TRIALS_PER_RUN = 6 if RESEARCH_STAGE == 4 else 1
AUTO_RESTORE_LATEST_ARTIFACT_BUNDLE = False

random.seed(SEED)
np.random.seed(SEED)
os.chdir(PROJECT_DIR)

print(f"Project          : {PROJECT_NAME}")
print(f"Path             : {PROJECT_DIR}")
print(f"Python           : {sys.version.split()[0]}")
print(f"Seed             : {SEED}")
print(f"Workflow stage   : {RESEARCH_STAGE}")
print(f"Execution mode   : {EXECUTION_MODE}")
print(f"Active profile   : {ACTIVE_PROFILE}")
print(f"Required runtime : {SELECTED_STAGE['runtime_instruction']}")
if restored_bundle is not None:
    print(f"Restored bundle  : {restored_bundle}")
    print(f"Restored files   : {len(restored_files)}")
else:
    print("Artifact status  : fresh V3 stage execution")
'''


FINAL_BACKUP_CODE = r'''# Final V3 Stage Snapshot to Google Drive

from datetime import datetime
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

from google.colab import drive


def _stage_completion_status():
    if RESEARCH_STAGE not in {4, 7, 8}:
        return "complete"
    comparison_path = (
        PROJECT_DIR
        / "report_artifacts"
        / "wave2d_method_comparison.json"
    )
    if not comparison_path.exists():
        return "partial"
    payload = json.loads(comparison_path.read_text(encoding="utf-8"))
    return "complete" if payload.get("status") == "complete" else "partial"


stage_completion_status = _stage_completion_status()
workflow_manifest = json.loads(
    WORKFLOW_MANIFEST_PATH.read_text(encoding="utf-8")
)
workflow_manifest["stage_status"][str(RESEARCH_STAGE)] = stage_completion_status
if (
    stage_completion_status == "complete"
    and RESEARCH_STAGE not in workflow_manifest["completed_stages"]
):
    workflow_manifest["completed_stages"].append(RESEARCH_STAGE)
    workflow_manifest["completed_stages"].sort()
workflow_manifest["active_stage"] = None
workflow_manifest["updated_at"] = _utc_now()
workflow_manifest["stage_history"][-1]["status"] = stage_completion_status
workflow_manifest["stage_history"][-1]["finished_at"] = _utc_now()
atomic_write_json(WORKFLOW_MANIFEST_PATH, workflow_manifest)

if not Path("/content/drive/MyDrive").exists():
    drive.mount("/content/drive")

BACKUP_ROOT.mkdir(parents=True, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
backup_path = BACKUP_ROOT / (
    f"online_pinn_pde_v3_stage_{RESEARCH_STAGE:02d}_"
    f"{stage_completion_status}_{timestamp}.zip"
)

artifact_directories = [
    "workflow",
    "config",
    "training",
    "training_state",
    "model_artifacts",
    "problem_artifacts",
    "report_artifacts",
    "rag_artifacts",
    "data",
    "app",
    "tests",
    "outputs",
]
saved_files = []

with ZipFile(backup_path, mode="w", compression=ZIP_DEFLATED) as archive:
    for directory_name in artifact_directories:
        directory_path = PROJECT_DIR / directory_name
        if not directory_path.exists():
            continue
        for file_path in directory_path.rglob("*"):
            if not file_path.is_file() or file_path.suffix == ".tmp":
                continue
            archive_name = file_path.relative_to(PROJECT_DIR).as_posix()
            archive.write(file_path, archive_name)
            saved_files.append(archive_name)

if not saved_files:
    raise RuntimeError("No V3 artifacts were available for the stage snapshot.")

latest_pointer = {
    "schema_version": 1,
    "project": PROJECT_NAME,
    "stage": RESEARCH_STAGE,
    "stage_name": SELECTED_STAGE["name"],
    "status": stage_completion_status,
    "execution_profile": ACTIVE_PROFILE,
    "required_runtime": SELECTED_STAGE["runtime_instruction"],
    "archive": backup_path.name,
    "saved_files": len(saved_files),
    "created_at": _utc_now(),
}
atomic_write_json(BACKUP_ROOT / "latest_v3_checkpoint.json", latest_pointer)

print(f"Stage status     : {stage_completion_status}")
print(f"Required runtime : {SELECTED_STAGE['runtime_instruction']}")
print(f"Backup file      : {backup_path}")
print(f"Saved files      : {len(saved_files)}")
print(f"Size             : {backup_path.stat().st_size / 1024**2:.2f} MB")
print("Stage snapshot   : completed")
'''


def apply_profile_configuration(text):
    marker = "\n\nwave2d_config_path = PROJECT_DIR / \"config\" / \"wave2d_config.json\""
    if marker not in text:
        raise RuntimeError("Could not locate the Wave2D configuration insertion point")
    profile_code = r'''

# V3 execution profiles are isolated by comparison artifact directory. The
# canonical equation, domain, architecture, optimizer, and loss definitions are
# not changed by the Smoke profile.
wave2d_config["execution_profile"] = ACTIVE_PROFILE
if ACTIVE_PROFILE == "smoke":
    wave2d_config["training"]["maximum_iterations"] = 200
    wave2d_config["training"]["evaluation_interval"] = 50
    wave2d_config["method_comparison"]["maximum_iterations"] = 200
    wave2d_config["method_comparison"]["evaluation_interval"] = 50
    wave2d_config["method_comparison"]["seeds"] = [3234]
    wave2d_config["method_comparison"]["artifact_directory"] = (
        "model_artifacts/wave_pinn_2d_comparison_smoke"
    )
elif ACTIVE_PROFILE == "pilot":
    wave2d_config["method_comparison"]["seeds"] = [3234]
    wave2d_config["method_comparison"]["artifact_directory"] = (
        "model_artifacts/wave_pinn_2d_comparison_pilot"
    )
elif ACTIVE_PROFILE == "benchmark":
    approval_path = PROJECT_DIR / "config" / "approved_benchmark_plan.json"
    if not approval_path.exists():
        raise FileNotFoundError(
            "Stage 8 requires config/approved_benchmark_plan.json created "
            "after human review of Stage 7. Automatic method promotion is blocked."
        )
    approved_plan = json.loads(approval_path.read_text(encoding="utf-8"))
    if approved_plan.get("approved") is not True:
        raise RuntimeError("The Stage 8 benchmark plan is not explicitly approved.")
    allowed_methods = set(wave2d_config["method_comparison"]["methods"])
    selected_methods = approved_plan.get("methods", [])
    selected_seeds = approved_plan.get("seeds", [])
    if not selected_methods or not set(selected_methods).issubset(allowed_methods):
        raise ValueError("The approved benchmark method list is empty or invalid.")
    if len(selected_seeds) != 4 or len(set(selected_seeds)) != 4:
        raise ValueError("The approved benchmark requires exactly four unique seeds.")
    wave2d_config["method_comparison"]["methods"] = selected_methods
    wave2d_config["method_comparison"]["seeds"] = selected_seeds
    wave2d_config["method_comparison"]["artifact_directory"] = (
        "model_artifacts/wave_pinn_2d_comparison_v3"
    )
'''
    return text.replace(marker, profile_code + marker, 1)


def patch_baseline_training_gate(text):
    marker = "\nelse:\n    baseline_protocol_version = 2\n"
    if marker not in text:
        raise RuntimeError("Could not locate the baseline training branch")
    replacement = r'''
elif not RUN_WAVE2D_BASELINE_TRAINING:
    baseline_protocol_version = 2
    baseline_stop_reason = "definition_only_no_training"
    baseline_iterations_completed = 0
    best_validation2d = float("nan")
    print("2D API baseline training: not selected for this workflow stage")
    print("Model, autodiff, loss and optimizer definitions are available.")

else:
    baseline_protocol_version = 2
'''
    return text.replace(marker, "\n" + replacement, 1)


def patch_baseline_resume_checkpoint(text):
    paths_marker = r'''wave2d_result_path = PROJECT_DIR / "report_artifacts" / "wave2d_result.json"

history2d = []
'''
    if paths_marker not in text:
        raise RuntimeError("Could not locate baseline checkpoint paths")
    paths_replacement = r'''wave2d_result_path = PROJECT_DIR / "report_artifacts" / "wave2d_result.json"

baseline_live_relative_root = Path(ACTIVE_PROFILE) / "wave2d_api_baseline"
baseline_live_checkpoint_dir = (
    BACKUP_ROOT
    / "V3_Training_State"
    / baseline_live_relative_root
    / "tensorflow"
)
baseline_live_history_relative = (
    baseline_live_relative_root / "wave2d_training_history.csv"
)
baseline_live_target_relative = (
    baseline_live_relative_root / "wave2d_convergence_targets.json"
)
baseline_live_best_relative = (
    baseline_live_relative_root / "best_target.weights.h5"
)
baseline_resume_step = tf.Variable(0, trainable=False, dtype=tf.int64)
baseline_resume_elapsed = tf.Variable(0.0, trainable=False, dtype=tf.float64)
baseline_resume_checkpoint = tf.train.Checkpoint(
    step=baseline_resume_step,
    elapsed_seconds=baseline_resume_elapsed,
    optimizer=optimizer2d,
    model=model2d,
)
baseline_resume_manager = None
if RUN_WAVE2D_BASELINE_TRAINING:
    baseline_live_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    baseline_resume_manager = tf.train.CheckpointManager(
        baseline_resume_checkpoint,
        str(baseline_live_checkpoint_dir),
        max_to_keep=3,
    )
    if not history2d_path.exists():
        restore_file_from_drive(
            baseline_live_history_relative,
            history2d_path,
        )
    if not baseline_target_path.exists():
        restore_file_from_drive(
            baseline_live_target_relative,
            baseline_target_path,
        )
    if not best2d_path.exists():
        restore_file_from_drive(
            baseline_live_best_relative,
            best2d_path,
        )
    if EXECUTION_MODE == "load" and baseline_resume_manager.latest_checkpoint:
        baseline_resume_checkpoint.restore(
            baseline_resume_manager.latest_checkpoint
        ).expect_partial()
        print(
            "Resumed 2D API baseline from iteration "
            f"{int(baseline_resume_step.numpy())}"
        )

history2d = []
'''
    text = text.replace(paths_marker, paths_replacement, 1)

    state_marker = r'''    best_validation2d = np.inf
    best_target_score2d = np.inf
    baseline_started_at = time.perf_counter()
'''
    if state_marker not in text:
        raise RuntimeError("Could not locate baseline training state")
    state_replacement = r'''    if history2d_path.exists():
        with history2d_path.open("r", encoding="utf-8") as stream:
            history2d = list(csv.DictReader(stream))
        for previous_row in history2d:
            baseline_tracker.observe(
                iteration=int(previous_row["iteration"]),
                elapsed_seconds=float(previous_row["elapsed_seconds"]),
                metrics={
                    "relative_l1": float(
                        previous_row["full_grid_relative_l1"]
                    ),
                    "relative_l2": float(
                        previous_row["full_grid_relative_l2"]
                    ),
                },
            )
    best_validation2d = (
        min(float(row["validation_loss"]) for row in history2d)
        if history2d
        else np.inf
    )
    best_target_score2d = (
        min(
            float(row["target_score_max_relative_l1_l2"])
            for row in history2d
        )
        if history2d
        else np.inf
    )
    baseline_iterations_completed = int(baseline_resume_step.numpy())
    baseline_elapsed_offset = float(baseline_resume_elapsed.numpy())
    baseline_started_at = time.perf_counter() - baseline_elapsed_offset
'''
    text = text.replace(state_marker, state_replacement, 1)

    checkpoint_marker = r'''        baseline_target_path.write_text(
            json.dumps(snapshot, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
'''
    if checkpoint_marker not in text:
        raise RuntimeError("Could not locate baseline target save")
    checkpoint_replacement = checkpoint_marker + r'''        baseline_resume_step.assign(iteration)
        baseline_resume_elapsed.assign(elapsed2d)
        baseline_resume_manager.save(checkpoint_number=int(iteration))
        mirror_file_to_drive(
            history2d_path,
            baseline_live_history_relative,
        )
        mirror_file_to_drive(
            baseline_target_path,
            baseline_live_target_relative,
        )
'''
    text = text.replace(checkpoint_marker, checkpoint_replacement, 1)

    start_marker = r'''    current_validation2d, target_score2d, baseline_convergence = (
        record_baseline_checkpoint(iteration=0, losses=None)
    )
    best_validation2d = current_validation2d
    best_target_score2d = target_score2d
    model2d.save_weights(best2d_path)
    baseline_iterations_completed = 0
    baseline_target_reached = (
        baseline_convergence["targets"][baseline_primary_key]["reached"] is True
    )
'''
    if start_marker not in text:
        raise RuntimeError("Could not locate baseline initial checkpoint")
    start_replacement = r'''    if baseline_iterations_completed > 0 and history2d:
        baseline_convergence = baseline_tracker.snapshot(
            iteration_budget=training2d_cfg["maximum_iterations"]
        )
        baseline_target_reached = (
            baseline_convergence["targets"][baseline_primary_key]["reached"]
            is True
        )
    else:
        baseline_iterations_completed = 0
        current_validation2d, target_score2d, baseline_convergence = (
            record_baseline_checkpoint(iteration=0, losses=None)
        )
        best_validation2d = current_validation2d
        best_target_score2d = target_score2d
        model2d.save_weights(best2d_path)
        mirror_file_to_drive(
            best2d_path,
            baseline_live_best_relative,
        )
        baseline_target_reached = (
            baseline_convergence["targets"][baseline_primary_key]["reached"]
            is True
        )
'''
    text = text.replace(start_marker, start_replacement, 1)
    text = text.replace(
        "        for iteration in range(\n"
        "            1,\n"
        "            training2d_cfg[\"maximum_iterations\"] + 1,\n",
        "        for iteration in range(\n"
        "            baseline_iterations_completed + 1,\n"
        "            training2d_cfg[\"maximum_iterations\"] + 1,\n",
        1,
    )
    improvement_marker = r'''                    model2d.save_weights(best2d_path)
'''
    if improvement_marker not in text:
        raise RuntimeError("Could not locate baseline best-weight save")
    improvement_replacement = improvement_marker + r'''                    mirror_file_to_drive(
                        best2d_path,
                        baseline_live_best_relative,
                    )
'''
    text = text.replace(improvement_marker, improvement_replacement, 1)
    return text


def patch_progress_saves(text):
    baseline_marker = "        baseline_target_path.write_text(\n"
    if baseline_marker in text:
        insertion = r'''        save_workflow_progress(
            problem="wave2d",
            case="api_baseline",
            iteration=iteration,
            payload={
                "relative_l1": field_metrics2d["relative_l1"],
                "relative_l2": field_metrics2d["relative_l2"],
                "validation_loss": current_validation2d,
            },
        )
'''
        text = text.replace(baseline_marker, insertion + baseline_marker, 1)

    comparison_marker = "        target_path.write_text(\n"
    if comparison_marker in text:
        insertion = r'''        save_workflow_progress(
            problem="wave2d",
            case=f"{method_name}_seed_{seed}",
            iteration=iteration,
            payload={
                "relative_l1": field_metrics["relative_l1"],
                "relative_l2": field_metrics["relative_l2"],
                "supervised_validation_mse": current_validation,
            },
        )
'''
        text = text.replace(comparison_marker, insertion + comparison_marker, 1)
    return text


def patch_method_subset_and_notice(text):
    marker = "\nconfigured_methods = method_comparison_cfg[\"methods\"]\n"
    if marker not in text:
        raise RuntimeError("Could not locate the method configuration block")
    replacement = r'''
configured_methods = method_comparison_cfg["methods"]
method_specs = {
    method_name: method_specs[method_name]
    for method_name in configured_methods
}
'''
    text = text.replace(marker, "\n" + replacement, 1)
    text = text.replace(
        "if len(configured_seeds) != 4 or len(set(configured_seeds)) != 4:\n"
        "    raise ValueError(\"Exactly four unique comparison seeds are required\")\n",
        "if ACTIVE_PROFILE == \"benchmark\":\n"
        "    if len(configured_seeds) != 4 or len(set(configured_seeds)) != 4:\n"
        "        raise ValueError(\"Exactly four unique comparison seeds are required\")\n"
        "elif ACTIVE_PROFILE in {\"smoke\", \"pilot\"}:\n"
        "    if configured_seeds != [3234]:\n"
        "        raise ValueError(\"Smoke and pilot profiles require seed 3234 only\")\n",
        1,
    )
    notice_marker = "method_comparison_cfg = wave2d_config[\"method_comparison\"]\n"
    notice = (
        notice_marker
        + "print(f\"Comparison profile : {ACTIVE_PROFILE}\")\n"
        + "if ACTIVE_PROFILE == \"smoke\":\n"
        + "    print(\"Execution verification only. Convergence and comparative performance were not evaluated.\")\n"
    )
    return text.replace(notice_marker, notice, 1)


def patch_comparison_resume_checkpoint(text):
    path_marker = r'''    target_path = method_dir / "convergence_targets.json"

    tracker = ConvergenceTargetTracker(
'''
    if path_marker not in text:
        raise RuntimeError("Could not locate comparison checkpoint paths")
    path_replacement = r'''    target_path = method_dir / "convergence_targets.json"

    live_relative_root = Path(ACTIVE_PROFILE) / method_name / f"seed_{seed}"
    live_checkpoint_dir = (
        BACKUP_ROOT / "V3_Training_State" / live_relative_root / "tensorflow"
    )
    live_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    live_history_relative = live_relative_root / "training_history.csv"
    live_target_relative = live_relative_root / "convergence_targets.json"
    live_best_relative = live_relative_root / "best_target.weights.h5"
    if not history_path.exists():
        restore_file_from_drive(live_history_relative, history_path)
    if not target_path.exists():
        restore_file_from_drive(live_target_relative, target_path)
    if not best_path.exists():
        restore_file_from_drive(live_best_relative, best_path)

    resume_step = tf.Variable(0, trainable=False, dtype=tf.int64)
    resume_elapsed = tf.Variable(0.0, trainable=False, dtype=tf.float64)
    resume_checkpoint = tf.train.Checkpoint(
        step=resume_step,
        elapsed_seconds=resume_elapsed,
        optimizer=optimizer,
        model=comparison_model,
    )
    resume_manager = tf.train.CheckpointManager(
        resume_checkpoint,
        str(live_checkpoint_dir),
        max_to_keep=3,
    )
    if EXECUTION_MODE == "load" and resume_manager.latest_checkpoint:
        resume_checkpoint.restore(
            resume_manager.latest_checkpoint
        ).expect_partial()
        print(
            f"Resumed {method_name} seed={seed} from iteration "
            f"{int(resume_step.numpy())}"
        )

    tracker = ConvergenceTargetTracker(
'''
    text = text.replace(path_marker, path_replacement, 1)

    state_marker = r'''    best_validation = _supervised_validation_mse(comparison_model)
    best_target_score = np.inf
    history_rows = []
    started_at = time.perf_counter()
'''
    if state_marker not in text:
        raise RuntimeError("Could not locate comparison training state")
    state_replacement = r'''    history_rows = []
    if history_path.exists():
        with history_path.open("r", encoding="utf-8") as stream:
            history_rows = list(csv.DictReader(stream))
        for previous_row in history_rows:
            tracker.observe(
                iteration=int(previous_row["iteration"]),
                elapsed_seconds=float(previous_row["elapsed_seconds"]),
                metrics={
                    "relative_l1": float(
                        previous_row["full_grid_relative_l1"]
                    ),
                    "relative_l2": float(
                        previous_row["full_grid_relative_l2"]
                    ),
                },
            )
    best_validation = (
        min(
            float(row["supervised_validation_mse"])
            for row in history_rows
        )
        if history_rows
        else _supervised_validation_mse(comparison_model)
    )
    best_target_score = (
        min(
            float(row["target_score_max_relative_l1_l2"])
            for row in history_rows
        )
        if history_rows
        else np.inf
    )
    completed_iterations = int(resume_step.numpy())
    elapsed_offset = float(resume_elapsed.numpy())
    started_at = time.perf_counter() - elapsed_offset
'''
    text = text.replace(state_marker, state_replacement, 1)

    checkpoint_marker = r'''        target_path.write_text(
            json.dumps(snapshot, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
'''
    if checkpoint_marker not in text:
        raise RuntimeError("Could not locate comparison target save")
    checkpoint_replacement = checkpoint_marker + r'''        resume_step.assign(iteration)
        resume_elapsed.assign(elapsed_at_checkpoint)
        resume_manager.save(checkpoint_number=int(iteration))
        mirror_file_to_drive(history_path, live_history_relative)
        mirror_file_to_drive(target_path, live_target_relative)
        mirror_file_to_drive(best_path, live_best_relative)
'''
    text = text.replace(checkpoint_marker, checkpoint_replacement, 1)

    start_marker = r'''    completed_iterations = 0
    target_reached = record_checkpoint(iteration=0, train_objective=None)
    if not target_reached:
        for iteration in range(
            1,
            method_comparison_cfg["maximum_iterations"] + 1,
        ):
'''
    if start_marker not in text:
        raise RuntimeError("Could not locate comparison iteration loop")
    start_replacement = r'''    primary_key = threshold_key(comparison_stop_cfg["threshold"])
    if completed_iterations > 0 and history_rows:
        target_reached = (
            tracker.snapshot(
                iteration_budget=method_comparison_cfg["maximum_iterations"]
            )["targets"][primary_key]["reached"] is True
        )
    else:
        completed_iterations = 0
        target_reached = record_checkpoint(
            iteration=0,
            train_objective=None,
        )
    if not target_reached:
        for iteration in range(
            completed_iterations + 1,
            method_comparison_cfg["maximum_iterations"] + 1,
        ):
'''
    text = text.replace(start_marker, start_replacement, 1)
    return text


def main():
    notebook = json.loads(SOURCE.read_text(encoding="utf-8"))
    notebook = copy.deepcopy(notebook)

    notebook["metadata"].setdefault("v3_research_workflow", {})
    notebook["metadata"]["v3_research_workflow"] = {
        "schema_version": 1,
        "stages": 8,
        "runtime_gate": True,
        "source_notebook": SOURCE.name,
    }

    notebook["cells"][0]["source"] = source_lines(
        source_text(notebook["cells"][0]).replace(
            "# AI Engineering Portfolio",
            "# AI Engineering Portfolio — V3 PDE Research Workflow",
            1,
        )
    )

    controller_index = next(
        index
        for index, cell in enumerate(notebook["cells"])
        if first_line(cell).startswith("# Cell 1 -")
    )
    guide_cell = {
        "cell_type": "markdown",
        "metadata": {"id": "v3-research-workflow-guide"},
        "source": source_lines(GUIDE_MARKDOWN.strip() + "\n"),
    }
    notebook["cells"].insert(controller_index, guide_cell)
    controller_index += 1
    notebook["cells"][controller_index]["source"] = source_lines(
        CONTROLLER_CODE.strip() + "\n"
    )
    notebook["cells"][controller_index]["metadata"]["id"] = (
        "v3-research-workflow-controller"
    )

    stage_map = {
        "# Cell 10B -": (2, 3, 4, 5, 6, 7, 8),
        "# Cell 10B-V -": (2, 3, 4, 7, 8),
        "# Cell 10B-T -": (2, 3, 4, 5, 6, 7, 8),
        "# Cell 10C -": (3, 4, 5, 6, 7, 8),
        "# Cell 10D -": (4, 5),
        "# Cell 10D-V -": (4, 5),
        "# Cell 10F -": (4, 7, 8),
        "# Cell 10E -": (5, 6),
        "# Cell 15 -": (6,),
        "# Cell 16 -": (6,),
        "# Cell 17 -": (6,),
        "# Cell 18 -": (6,),
        "# Cell 19 -": (6,),
        "# Cell 20 -": (6,),
        "# Cell 21 -": (6,),
        "# Cell 22 -": (6,),
        "# Cell 23 -": (6,),
        "# Cell 24 -": (6,),
        "# Cell 25 -": (6,),
        "# Cell 26 -": (6,),
        "# Cell 27 -": (6,),
        "# Cell 28 -": (6,),
        "# Cell 29 -": (6,),
        "# Cell 30 -": (6,),
        "# Cell 31 -": (6,),
        "# Cell 32 -": (6,),
        "# Cell 33 -": (6,),
        "# Cell 34 -": (6,),
        "# Cell 35 -": (6,),
        "# Cell 36 -": (6,),
        "# Cell 37 -": (6,),
        "# Cell 38 -": (6,),
        "# Cell 39 -": (6,),
        "# Cell 40 -": (6,),
        "# Cell 41 -": (6,),
        "# Cell 42 -": (6,),
        "# Cell 42A -": (6,),
        "# Cell 43 -": (6,),
        "# Cell 44 -": (6,),
    }

    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        title = first_line(cell)
        if title.startswith("# Cell 10B -"):
            cell["source"] = source_lines(
                apply_profile_configuration(source_text(cell))
            )
        elif title.startswith("# Cell 10C -"):
            patched = patch_baseline_training_gate(source_text(cell))
            patched = patch_baseline_resume_checkpoint(patched)
            cell["source"] = source_lines(patch_progress_saves(patched))
        elif title.startswith("# Cell 10F -"):
            patched = patch_method_subset_and_notice(source_text(cell))
            patched = patch_comparison_resume_checkpoint(patched)
            cell["source"] = source_lines(patch_progress_saves(patched))

        matched_stages = None
        for prefix, stages in stage_map.items():
            if title.startswith(prefix):
                matched_stages = stages
                break
        if matched_stages is not None:
            wrap_for_stages(cell, matched_stages)

    free_question_cell = next(
        cell
        for cell in notebook["cells"]
        if first_line(cell).startswith("# Free Question -")
    )
    free_question_cell["source"] = source_lines(
        "# Free Question - Manual public demonstration\n\n"
        "print(\n"
        "    \"The interactive free-question demonstration is disabled during \"\n"
        "    \"automated V3 research stages. Run it manually after Stage 6.\"\n"
        ")\n"
    )
    free_question_cell["metadata"]["workflow_manual_only"] = True

    final_cell = next(
        cell
        for cell in notebook["cells"]
        if first_line(cell).startswith("# Final 2D Artifact Backup")
    )
    final_cell["source"] = source_lines(FINAL_BACKUP_CODE.strip() + "\n")
    final_cell["metadata"]["id"] = "v3-final-stage-snapshot"

    TARGET.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    print(TARGET)


if __name__ == "__main__":
    main()
