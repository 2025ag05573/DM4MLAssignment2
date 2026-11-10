"""Common model interface + the context every model is fitted with."""
from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.common.config import get_paths
from src.features.store import FeatureStore


@dataclass
class FitContext:
    """Everything a model may use at fit time - all restricted to the fit window."""

    stage: str                    # "tune" (fit=train) or "final" (fit=train+val)
    X: sp.csr_matrix              # users x items explicit ratings (0 = unobserved)
    interactions: pd.DataFrame    # fit-window interactions (user_id, movie_id, rating, timestamp)
    user_index: pd.DataFrame      # user_id -> uidx
    item_index: pd.DataFrame      # movie_id -> iidx
    store: FeatureStore
    snapshot: str                 # feature snapshot matching the fit window
    cfg: dict[str, Any]

    @property
    def n_users(self) -> int:
        return self.X.shape[0]

    @property
    def n_items(self) -> int:
        return self.X.shape[1]


def load_context(cfg: dict[str, Any], stage: str) -> FitContext:
    paths = get_paths(cfg)
    X = sp.load_npz(paths.derived / f"interactions_{stage}.npz").tocsr()
    X.sort_indices()                      # canonical in-row order (see transform.interaction_matrix)
    parts = ["train"] if stage == "tune" else ["train", "val"]
    inter = pd.concat([pd.read_parquet(paths.splits / f"{p}.parquet") for p in parts], ignore_index=True)
    return FitContext(stage=stage, X=X, interactions=inter,
                      user_index=pd.read_parquet(paths.derived / "user_index.parquet"),
                      item_index=pd.read_parquet(paths.derived / "item_index.parquet"),
                      store=FeatureStore(cfg), snapshot=stage, cfg=cfg)


class Recommender(ABC):
    name: str = "base"
    predicts_ratings: bool = False

    @abstractmethod
    def fit(self, ctx: FitContext) -> "Recommender": ...

    @abstractmethod
    def score(self, uidx: np.ndarray) -> np.ndarray:
        """Return an (len(uidx), n_items) matrix; higher = more relevant."""

    def predict_ratings(self, uidx: np.ndarray, iidx: np.ndarray) -> np.ndarray:
        raise NotImplementedError(f"{self.name} does not predict explicit ratings")

    def diagnostics(self) -> dict[str, float]:
        """Fit diagnostics logged as metrics (e.g. whether MF factors are actually used)."""
        return {}

    def params(self) -> dict[str, Any]:
        return {}

    # The fitted model must not keep the (large) FitContext around.
    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)
        return path

    @staticmethod
    def load(path: Path) -> "Recommender":
        with open(path, "rb") as fh:
            return pickle.load(fh)
