"""Feature store access layer (a lightweight, file-based alternative to Feast).

Training code and the inference CLI both read features through this class,
choosing a snapshot (``tune`` / ``final`` / ``serving``). On-demand features
that depend on a (user, item) pair - ``ui_genre_match`` - are computed here so
they are defined exactly once.
"""
from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd

from src.common.config import PROJECT_ROOT, get_paths
from src.common.io import read_json


class FeatureStore:
    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        self.paths = get_paths(cfg)
        self.registry = read_json(self.paths.features / "registry.json")
        self._cache: dict[tuple[str, str], pd.DataFrame] = {}

    # ------------------------------------------------------------------ metadata
    def snapshots(self) -> list[str]:
        return list(self.registry["snapshots"])

    def resolve(self, snapshot: str | None) -> str:
        if snapshot in (None, "latest"):
            return max(self.registry["snapshots"], key=lambda s: self.registry["snapshots"][s]["as_of"])
        if snapshot not in self.registry["snapshots"]:
            raise KeyError(f"unknown snapshot {snapshot!r}; available: {self.snapshots()}")
        return snapshot

    def as_of(self, snapshot: str | None) -> int:
        return int(self.registry["snapshots"][self.resolve(snapshot)]["as_of"])

    # ------------------------------------------------------------------ retrieval
    def _load(self, group: str, snapshot: str | None) -> pd.DataFrame:
        snap = self.resolve(snapshot)
        key = (group, snap)
        if key not in self._cache:
            path = PROJECT_ROOT / self.registry["snapshots"][snap]["groups"][group]["path"]
            self._cache[key] = pd.read_parquet(path)
        return self._cache[key]

    def get_user_features(self, user_ids: Iterable[int] | None = None, snapshot: str | None = "latest",
                          columns: list[str] | None = None) -> pd.DataFrame:
        df = self._load("user_features", snapshot).set_index("user_id")
        df = df if user_ids is None else df.reindex(list(user_ids))
        return df if columns is None else df[columns]

    def get_item_features(self, movie_ids: Iterable[int] | None = None, snapshot: str | None = "latest",
                          columns: list[str] | None = None) -> pd.DataFrame:
        df = self._load("item_features", snapshot).set_index("movie_id")
        df = df if movie_ids is None else df.reindex(list(movie_ids))
        return df if columns is None else df[columns]

    def get_interaction_features(self, snapshot: str | None = "latest") -> pd.DataFrame:
        return self._load("interaction_features", snapshot)

    # ------------------------------------------------------------------ on-demand features
    def user_genre_affinity(self, user_ids: Iterable[int], snapshot: str | None) -> np.ndarray:
        df = self.get_user_features(user_ids, snapshot)
        return df[[c for c in df.columns if c.startswith("user_genre_affinity_")]].to_numpy(np.float32)

    def item_genre_matrix(self, movie_ids: Iterable[int], snapshot: str | None) -> np.ndarray:
        df = self.get_item_features(movie_ids, snapshot)
        return df[[c for c in df.columns if c.startswith("item_genre_")]].to_numpy(np.float32)

    def genre_match(self, user_ids: Iterable[int], movie_ids: Iterable[int],
                    snapshot: str | None = "latest") -> np.ndarray:
        """``ui_genre_match`` for every (user, item) pair: cosine(user affinity, item genres)."""
        U = self.user_genre_affinity(user_ids, snapshot)
        G = self.item_genre_matrix(movie_ids, snapshot)
        Un = U / np.maximum(np.linalg.norm(U, axis=1, keepdims=True), 1e-9)
        Gn = G / np.maximum(np.linalg.norm(G, axis=1, keepdims=True), 1e-9)
        return Un @ Gn.T
