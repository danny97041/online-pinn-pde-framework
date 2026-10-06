#@title 2. 설정 · 결과
folder = Path(저장_폴더)
if str(folder).startswith("/content/drive/") and not folder.is_dir():
    from google.colab import drive

    drive.mount("/content/drive")
if not folder.is_dir():
    raise FileNotFoundError("저장 폴더가 없습니다. 파일 위치를 확인하세요.")
scratch_base = Path("/content") if Path("/content").is_dir() else Path.cwd() / "tmp"
scratch_base.mkdir(exist_ok=True)
tempfile.tempdir = str(scratch_base)
try:
    from google.colab import output as colab_output

    colab_output.enable_custom_widget_manager()
except ImportError:
    pass
PANEL = release_panel(folder, scratch_base)
PANEL["execute"]()  # 기본 결과 조회만 실행. 검사와 학습은 자동으로 시작하지 않습니다.
