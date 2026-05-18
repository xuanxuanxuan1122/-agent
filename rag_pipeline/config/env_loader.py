from __future__ import annotations

import os
from pathlib import Path
from threading import Lock
from typing import Iterable


PIPELINE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENV_FILES = [
    PIPELINE_ROOT / ".env",
]

_ENV_LOAD_LOCK = Lock()
_ENV_LOADED = False


def _strip_wrapped_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _parse_env_line(raw_line: str) -> tuple[str, str] | None:
    line = str(raw_line or "").strip()
    if not line or line.startswith("#"):
        return None
    if line.startswith("export "):
        line = line[len("export ") :].strip()
    if "=" not in line:
        return None
    key, value = line.split("=", 1)
    key = key.strip()
    if not key:
        return None
    return key, _strip_wrapped_quotes(value.strip())


def load_env_files(paths: Iterable[Path] | None = None, override: bool = False) -> None:
    global _ENV_LOADED
    with _ENV_LOAD_LOCK:
        if _ENV_LOADED and not override:
            return
        for path in list(paths or DEFAULT_ENV_FILES):
            env_path = Path(path)
            if not env_path.exists() or not env_path.is_file():
                continue
            for raw_line in env_path.read_text(encoding="utf-8").splitlines():
                parsed = _parse_env_line(raw_line)
                if parsed is None:
                    continue
                key, value = parsed
                if override or key not in os.environ:
                    os.environ[key] = value
        _ENV_LOADED = True
