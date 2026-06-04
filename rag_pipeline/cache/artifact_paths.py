from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any


DEFAULT_LEDGER_PATH = "output/cache/artifact_ledger.sqlite"
DEFAULT_OBJECT_ROOT = "output/cache/artifacts"


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def env_flag(name: str, default: bool = True) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on", "y"}


def env_int(name: str, default: int, *, min_value: int = 0, max_value: int = 10_000_000) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(float(raw))
    except (TypeError, ValueError):
        return default
    return max(min_value, min(max_value, value))


def resolve_project_path(raw_path: str | os.PathLike[str] | None, default: str) -> Path:
    path = Path(str(raw_path or default).strip() or default)
    if path.is_absolute():
        return path
    return project_root() / path


def artifact_ledger_path() -> Path:
    return resolve_project_path(os.getenv("ARTIFACT_LEDGER_PATH"), DEFAULT_LEDGER_PATH)


def artifact_object_root() -> Path:
    return resolve_project_path(os.getenv("ARTIFACT_OBJECT_ROOT"), DEFAULT_OBJECT_ROOT)


def safe_path_part(value: Any, *, max_chars: int = 120) -> str:
    text = re.sub(r"[^\w\u4e00-\u9fff.-]+", "_", str(value or "").strip(), flags=re.U)
    text = re.sub(r"_+", "_", text).strip("._")
    return (text or "artifact")[:max_chars]
