"""Colab storage connection status, independent of model and training code."""


def release_initial_storage(
    storage,
    local_folder,
    drive_folder="/content/drive/MyDrive/PINN",
    override=None,
    drive_root="/content/drive",
    zip_choices=None,
):
    """Resolve first; mount later after the panel exists. No authentication here."""
    if storage not in {"auto", "local", "drive"}:
        raise ValueError("Invalid initial storage selection")
    local_folder, drive_folder, drive_root = map(
        Path, (local_folder, drive_folder, drive_root)
    )
    mounted = (drive_root / "MyDrive").is_dir()
    choices = zip_choices or release_zip_choices
    if override:
        folder = Path(override)
        is_drive = folder.is_relative_to(drive_root)
    elif storage == "local":
        folder, is_drive = local_folder, False
    elif storage == "drive" or mounted:
        folder, is_drive = drive_folder, True
    elif choices(local_folder):
        folder, is_drive = local_folder, False
    else:
        folder, is_drive = drive_folder, True
    if not is_drive:
        return {
            "folder": folder,
            "storage_status": {"status": "local"},
            "connect_once": False,
        }
    return {
        "folder": folder,
        "storage_status": {"status": "connected" if mounted else "not_connected"},
        "connect_once": not mounted and storage != "local",
    }


def release_drive_status(mount=None, drive_root="/content/drive"):
    """One connection attempt; authentication failures remain explicit.

    Connection success is checked against MyDrive, not a project subfolder.
    No forced remount, credential collection, auth bypass or retry loop occurs.
    ``mount`` and ``drive_root`` are injectable for offline regression tests.
    """
    root = Path(drive_root)
    if (root / "MyDrive").is_dir():
        return {"status": "connected", "already_connected": True}
    try:
        if mount is None:
            from google.colab import drive

            mount = drive.mount
        mount(str(root))
    except ImportError:
        return {"status": "unavailable", "error_type": "ImportError"}
    except Exception as exc:
        return {
            "status": (
                "authentication_failed"
                if "credential propagation" in str(exc).lower()
                else "connection_failed"
            ),
            "error_type": type(exc).__name__,
        }
    return {
        "status": "connected" if (root / "MyDrive").is_dir() else "not_ready",
        "already_connected": False,
    }
