"""Strict JSON and atomic file replacement shared by evaluation artifacts."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any


def strict_json_dumps(value: Any, *, indent: int | None = None) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False,
        separators=None if indent is not None else (',', ':'), indent=indent)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, strict_json_dumps(value, indent=2) + '\n')


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f'.{path.name}.', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
