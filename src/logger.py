"""Stage loggers writing to logs/ directory."""

import logging
from pathlib import Path
from src.config import config

_STAGE_HANDLERS: dict[str, logging.FileHandler] = {}
_initialized = False


def _ensure_logs_dir() -> Path:
    logs_dir = config.logs_dir
    logs_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir


def get_stage_logger(stage: str) -> logging.Logger:
    global _initialized
    name = f"pipeline.{stage}"
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logs_dir = _ensure_logs_dir()
    handler = logging.FileHandler(logs_dir / f"{stage}.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(handler)

    if stage == "errors":
        err_handler = logging.FileHandler(logs_dir / "errors.log", encoding="utf-8")
        err_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        err_handler.setLevel(logging.WARNING)
        logger.addHandler(err_handler)

    logger.setLevel(logging.DEBUG)
    _STAGE_HANDLERS[stage] = handler
    _initialized = True
    return logger
