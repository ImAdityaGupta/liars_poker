from __future__ import annotations

import gzip
import json
import os
import tempfile
from typing import Any, Dict


def save_json(path: str, obj: Dict[str, Any]) -> None:
    data = json.dumps(obj, indent=2, sort_keys=True).encode("utf-8")
    _atomic_write(path, data)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_jsonz(path: str, obj: Dict[str, Any]) -> None:
    data = json.dumps(obj, indent=2, sort_keys=True).encode("utf-8")
    _atomic_write(path, data, compressed=True)


def load_jsonz(path: str) -> Dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def _atomic_write(path: str, data: bytes, *, compressed: bool = False) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    suffix = ".tmp.gz" if compressed else ".tmp"
    fd, tmp_path = tempfile.mkstemp(dir=directory or None, prefix=".tmp-", suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as tmp:
            if compressed:
                with gzip.GzipFile(fileobj=tmp, mode="wb") as gz:
                    gz.write(data)
                    gz.flush()
            else:
                tmp.write(data)
                tmp.flush()
            os.fsync(tmp.fileno())
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

