"""Manifest-checked archives. Never executes source from downloaded results."""

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import tempfile
import zipfile


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        if Path(temp).exists():
            Path(temp).unlink()


def verify_zip(path):
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        if len(names) != len(set(names)):
            raise ValueError("ZIP 항목 중복")
        for item in z.infolist():
            p = PurePosixPath(item.filename)
            if (
                p.is_absolute()
                or ".." in p.parts
                or "\\" in item.filename
                or ":" in item.filename
                or "\x00" in item.filename
                or stat.S_ISLNK(item.external_attr >> 16)
            ):
                raise ValueError("ZIP 경로가 안전하지 않습니다.")
        if sum(i.file_size for i in z.infolist()) > 4 * 1024**3:
            raise ValueError("ZIP 크기 한도 초과")
        manifest = json.loads(z.read("v4_manifest.json"))
        files = {i.filename for i in z.infolist() if not i.is_dir()}
        if files != set(manifest["files"]) | {"v4_manifest.json"}:
            raise ValueError("ZIP 목록 불일치")
        for name, digest in manifest["files"].items():
            # Reading to EOF validates CRC while SHA is computed in bounded memory.
            h = hashlib.sha256()
            with z.open(name) as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    h.update(block)
            if h.hexdigest() != digest:
                raise ValueError("ZIP 해시 불일치: " + name)
        contract = json.loads(z.read("contract.json"))
        if contract.get("format") != "v4-independent-1":
            raise ValueError("V4 ZIP이 아닙니다.")
    return contract


def pack(root, output):
    root, output = Path(root), Path(output)
    if root.resolve() in output.resolve().parents:
        raise ValueError("출력 ZIP은 작업폴더 밖에 있어야 합니다.")
    output.parent.mkdir(parents=True, exist_ok=True)
    files = {
        p.relative_to(root).as_posix(): sha_file(p)
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.name != "v4_manifest.json"
    }
    write_json(root / "v4_manifest.json", {"files": files})
    fd, temp = tempfile.mkstemp(
        prefix=output.name + ".", suffix=".tmp", dir=output.parent
    )
    os.close(fd)
    try:
        with zipfile.ZipFile(temp, "w", zipfile.ZIP_DEFLATED) as z:
            for name in [*files, "v4_manifest.json"]:
                z.write(root / name, name)
        verify_zip(temp)
        os.replace(temp, output)
    finally:
        if Path(temp).exists():
            Path(temp).unlink()
    return output


def restore(bundle, root, expected=None):
    contract = verify_zip(bundle)
    if expected and any(contract.get(key) != value for key, value in expected.items()):
        raise ValueError("ZIP의 실행·실험 식별자가 다릅니다.")
    root = Path(root)
    if root.exists():
        raise ValueError("새 작업폴더가 필요합니다.")
    with zipfile.ZipFile(bundle) as z:
        z.extractall(root)
    return contract


def arrays_sha(arrays):
    import numpy as np

    h = hashlib.sha256()
    for name, value in sorted(arrays.items()):
        a = np.ascontiguousarray(value)
        h.update(name.encode())
        h.update(str(a.shape).encode())
        h.update(str(a.dtype).encode())
        h.update(a.tobytes())
    return h.hexdigest()
