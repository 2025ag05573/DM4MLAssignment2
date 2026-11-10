"""Configuration loading and project paths.

All stages call :func:`load_config` so that every parameter of a run comes from
one YAML file. Relative paths in the config are resolved against the
repository root, which makes the pipeline runnable from any working directory.
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "pipeline.yaml"


@dataclass(frozen=True)
class Paths:
    """Resolved locations of every dataset / artifact the pipeline touches."""

    root: Path
    raw: Path
    processed: Path
    features: Path
    errors: Path
    models: Path
    reports: Path

    @property
    def validated(self) -> Path:
        return self.processed / "validated"

    @property
    def clean(self) -> Path:
        return self.processed / "clean"

    @property
    def splits(self) -> Path:
        return self.processed / "splits"

    @property
    def derived(self) -> Path:
        return self.processed / "derived"

    def ensure(self) -> "Paths":
        for p in (self.raw, self.processed, self.features, self.errors, self.models,
                  self.reports, self.validated, self.clean, self.splits, self.derived,
                  self.reports / "validation", self.reports / "evaluation",
                  self.reports / "logs", self.reports / "figures"):
            p.mkdir(parents=True, exist_ok=True)
        return self


def load_config(path: str | os.PathLike | None = None) -> dict[str, Any]:
    """Load the YAML config (``PIPELINE_CONFIG`` env var overrides the default)."""
    cfg_path = Path(path or os.environ.get("PIPELINE_CONFIG", DEFAULT_CONFIG))
    with open(cfg_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["_config_path"] = str(cfg_path.resolve())
    return cfg


def get_paths(cfg: dict[str, Any]) -> Paths:
    p = cfg["paths"]

    def res(key: str) -> Path:
        q = Path(p[key])
        return q if q.is_absolute() else PROJECT_ROOT / q

    return Paths(root=PROJECT_ROOT, raw=res("raw_dir"), processed=res("processed_dir"),
                 features=res("features_dir"), errors=res("errors_dir"),
                 models=res("models_dir"), reports=res("reports_dir")).ensure()


def set_seed(seed: int) -> None:
    """Seed every RNG used in the project for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
