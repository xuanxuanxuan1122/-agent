from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional


PIPELINE_ROOT = Path(__file__).resolve().parents[1]


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def configure_pipeline_logging(*, log_file: Optional[str] = None, level: Optional[str] = None) -> None:
    if not _env_flag("PIPELINE_LOG_ENABLED", True):
        return

    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        if getattr(handler, "_rag_pipeline_handler", False):
            return

    log_level = str(level or os.getenv("PIPELINE_LOG_LEVEL") or "INFO").strip().upper()
    numeric_level = getattr(logging, log_level, logging.INFO)
    target = Path(log_file or os.getenv("PIPELINE_LOG_FILE") or PIPELINE_ROOT / "logs" / "pipeline.log")
    if not target.is_absolute():
        target = PIPELINE_ROOT / target
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(
            target,
            maxBytes=max(256_000, int(os.getenv("PIPELINE_LOG_MAX_BYTES", "5242880") or 5_242_880)),
            backupCount=max(1, int(os.getenv("PIPELINE_LOG_BACKUP_COUNT", "5") or 5)),
            encoding="utf-8",
        )
    except OSError:
        return

    handler._rag_pipeline_handler = True  # type: ignore[attr-defined]
    handler.setLevel(numeric_level)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s:%(lineno)d %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root_logger.addHandler(handler)
    if root_logger.level == logging.NOTSET or root_logger.level > numeric_level:
        root_logger.setLevel(numeric_level)
