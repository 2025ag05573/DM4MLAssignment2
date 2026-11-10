"""Project-wide logging: console + a persistent log file under reports/logs."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

from src.common.config import PROJECT_ROOT

LOG_FILE = PROJECT_ROOT / "reports" / "logs" / "pipeline.log"
_FMT = "%(asctime)s | %(levelname)-7s | %(name)-28s | %(message)s"


def get_logger(name: str) -> logging.Logger:
    """INFO logger writing to the console and reports/logs/pipeline.log.

    Orchestrators may attach their own handlers to these loggers (Prefect does,
    via PREFECT_LOGGING_EXTRA_LOGGERS), so ours are identified by a marker
    attribute rather than by "has any handler"."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if any(getattr(h, "_recpipe", False) for h in logger.handlers):
        return logger
    logger.propagate = False   # Prefect attaches only its API handler here, so no duplicates
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter(_FMT, "%H:%M:%S"))
    console._recpipe = True
    logger.addHandler(console)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setFormatter(logging.Formatter(_FMT, "%Y-%m-%d %H:%M:%S"))
    fh._recpipe = True
    logger.addHandler(fh)
    return logger


def relpath(p: Path | str) -> str:
    """Path relative to the repo root, for readable logs and reports."""
    try:
        return str(Path(p).resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(p)
