"""Launch the bottom panel after definitions and the initial decision are ready."""

from pathlib import Path
import tempfile
from IPython.display import display
import ipywidgets as w

if (
    "release_panel" not in globals()
    or "release_drive_status" not in globals()
    or "release_initial_storage" not in globals()
    or not globals().get("V3_EXECUTED_SOURCE_HASHES")
    or globals().get("V3_LOADED_MODULES") != V3_EXECUTED_SOURCE_HASHES
    or "V3_INITIAL_SETUP" not in globals()
):
    raise RuntimeError(
        "정의 셀 초기화가 필요합니다. 런타임 → 모두 실행 후 실행 패널을 사용할 수 있습니다."
    )

scratch_base = Path("/content") if Path("/content").is_dir() else Path.cwd() / "tmp"
scratch_base.mkdir(exist_ok=True)
tempfile.tempdir = str(scratch_base)
try:
    from google.colab import output as colab_output

    colab_output.enable_custom_widget_manager()
except ImportError:
    pass
if isinstance(globals().get("PANEL"), dict) and "dashboard" in PANEL:
    PANEL["dashboard"].close()
PANEL = None
if "V3_PANEL_WAITING" in globals():
    V3_PANEL_WAITING.close()
V3_OPEN_DEFAULTS_BUTTON = w.Button(
    description="기본 조회로 바로 진행",
    button_style="primary",
    layout=w.Layout(width="220px"),
)
V3_PANEL_WAITING = w.VBox(
    [
        w.HTML(
            "<p>최초 설정 대기 중. 자동 대기가 멈추면 아래 버튼으로 진행하세요. 학습·검사는 시작하지 않습니다.</p>"
        ),
        V3_OPEN_DEFAULTS_BUTTON,
    ]
)
display(V3_PANEL_WAITING)
V3_PANEL_REQUEST_ID = object()


def v3_launch_panel(choice, request_id=V3_PANEL_REQUEST_ID):
    global PANEL, STORAGE_STATUS
    if globals().get("V3_PANEL_REQUEST_ID") is not request_id:
        return
    override = globals().get("V3_FOLDER_OVERRIDE")
    drive_folder = Path(
        globals().get("V3_DEFAULT_FOLDER", "/content/drive/MyDrive/PINN")
    )
    local_folder = scratch_base
    resolution = release_initial_storage(
        choice["storage"], local_folder, drive_folder, override=override
    )
    folder = resolution["folder"]
    STORAGE_STATUS = resolution["storage_status"]
    try:
        PANEL = release_panel(
            folder,
            scratch_base,
            storage_status=STORAGE_STATUS,
            initial_action=choice["action"],
            show=False,
        )
        V3_PANEL_WAITING.children = (PANEL["dashboard"],)
        if resolution["connect_once"]:
            # Show the panel first. Cancellation/failure stays visible; no retry loop.
            PANEL["connect_drive"](target_folder=str(folder))
            STORAGE_STATUS = PANEL["state"]["storage"]
        if (
            choice["action"] == "results"
            and folder.is_dir()
            and PANEL["controls"]["zip"].value
        ):
            PANEL["auto_results"]()  # Read-only; reliable timer-triggered rendering.
    except Exception as exc:
        from html import escape

        V3_PANEL_WAITING.children = (
            w.HTML(
                "<p>패널 초기화 실패: "
                + escape(type(exc).__name__ + ": " + str(exc))
                + "</p><p>맨 아래 실행 패널 셀을 다시 실행하세요. 학습은 실행하지 않았습니다.</p>"
            ),
        )
        raise


V3_INITIAL_SETUP["on_ready"](v3_launch_panel)


def v3_open_defaults(_=None, request_id=V3_PANEL_REQUEST_ID):
    if globals().get("V3_PANEL_REQUEST_ID") is request_id:
        V3_INITIAL_SETUP["open_defaults"]()


V3_OPEN_DEFAULTS_BUTTON.on_click(v3_open_defaults)
