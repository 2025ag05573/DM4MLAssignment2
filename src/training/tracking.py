"""MLflow experiment tracking helpers (local SQLite backend, file artifact store)."""
from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
import mlflow  # noqa: E402

from src.common.config import PROJECT_ROOT  # noqa: E402


def _resolve_uri(uri: str) -> str:
    if uri.startswith("sqlite:///") and not uri.startswith("sqlite:////"):
        db = PROJECT_ROOT / uri[len("sqlite:///"):]
        db.parent.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{db}"
    return uri


def setup(cfg: dict[str, Any]) -> str:
    """Point MLflow at the project store and return the experiment id."""
    m = cfg["mlflow"]
    uri = os.environ.get("MLFLOW_TRACKING_URI") or _resolve_uri(m["tracking_uri"])
    mlflow.set_tracking_uri(uri)
    exp = mlflow.get_experiment_by_name(m["experiment_name"])
    if exp is None:
        art = (PROJECT_ROOT / m["artifact_location"]).resolve()
        art.mkdir(parents=True, exist_ok=True)
        return mlflow.create_experiment(m["experiment_name"], artifact_location=art.as_uri())
    return exp.experiment_id


@contextmanager
def start_run(cfg: dict[str, Any], run_name: str, nested: bool = False, run_id: str | None = None,
              tags: dict[str, str] | None = None) -> Iterator[mlflow.ActiveRun]:
    exp_id = setup(cfg)
    kwargs = {"run_id": run_id} if run_id else {"experiment_id": exp_id, "run_name": run_name}
    with mlflow.start_run(nested=nested, tags=tags, **kwargs) as run:
        yield run


def log_params(params: dict[str, Any], prefix: str = "") -> None:
    flat = {f"{prefix}{k}": ("None" if v is None else v) for k, v in params.items()}
    mlflow.log_params(flat)


def log_metrics(metrics: dict[str, float], prefix: str = "") -> None:
    # MLflow metric keys may not contain '@'
    mlflow.log_metrics({f"{prefix}{k.replace('@', '_at_')}": float(v) for k, v in metrics.items()})


def log_artifact(path: Path, artifact_path: str | None = None) -> None:
    mlflow.log_artifact(str(path), artifact_path=artifact_path)
