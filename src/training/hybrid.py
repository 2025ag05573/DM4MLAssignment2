"""Hybrid recommender: collaborative + content + popularity, using the feature store.

    score(u, i) = w_cf      * z_u( ALS score(u, i) )
                + w_content * ui_genre_match(u, i)          (on-demand feature)
                + w_pop     * z( log1p(item_rating_count_i) )  (item feature)

``z_u`` standardises the ALS scores per user so the three signals are on a
comparable scale. The content and popularity terms come from the feature store
snapshot that matches the fit window - so the same code path serves training
(``tune``/``final`` snapshots) and inference (``serving`` snapshot). The weights
are tuned on the validation window.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from src.features.store import FeatureStore
from src.training.base import FitContext, Recommender
from src.training.mf import ImplicitALS


class HybridRecommender(Recommender):
    name = "hybrid"

    def __init__(self, cf_model: ImplicitALS, w_cf: float = 1.0, w_content: float = 0.2,
                 w_pop: float = 0.1):
        self.cf = cf_model
        self.w_cf, self.w_content, self.w_pop = w_cf, w_content, w_pop

    def fit(self, ctx: FitContext) -> "HybridRecommender":
        """The CF component is fitted by the caller (re-used, not refitted); here we only
        pull the item-side features and remember which snapshot to use for users."""
        movie_ids = ctx.item_index["movie_id"].to_numpy()
        self.user_ids = ctx.user_index["user_id"].to_numpy()
        self.movie_ids = movie_ids
        cnt = ctx.store.get_item_features(movie_ids, ctx.snapshot, ["item_rating_count"])
        lp = np.log1p(cnt["item_rating_count"].to_numpy(np.float64))
        self.pop_z = ((lp - lp.mean()) / (lp.std() + 1e-9)).astype(np.float32)
        G = ctx.store.item_genre_matrix(movie_ids, ctx.snapshot)
        self.item_genres = G / np.maximum(np.linalg.norm(G, axis=1, keepdims=True), 1e-9)
        A = ctx.store.user_genre_affinity(self.user_ids, ctx.snapshot)
        self.user_affinity = A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-9)
        self.snapshot = ctx.snapshot
        return self

    def refresh_features(self, store: FeatureStore, snapshot: str = "serving") -> "HybridRecommender":
        """Swap in newer feature values (e.g. the serving snapshot) without retraining."""
        A = store.user_genre_affinity(self.user_ids, snapshot)
        self.user_affinity = A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-9)
        cnt = store.get_item_features(self.movie_ids, snapshot, ["item_rating_count"])
        lp = np.log1p(cnt["item_rating_count"].to_numpy(np.float64))
        self.pop_z = ((lp - lp.mean()) / (lp.std() + 1e-9)).astype(np.float32)
        self.snapshot = snapshot
        return self

    def components(self, uidx: np.ndarray) -> dict[str, np.ndarray]:
        cf = self.cf.score(uidx)
        cf_z = (cf - cf.mean(axis=1, keepdims=True)) / (cf.std(axis=1, keepdims=True) + 1e-9)
        content = self.user_affinity[uidx] @ self.item_genres.T       # ui_genre_match
        return {"cf": cf_z, "content": content.astype(np.float32), "pop": self.pop_z[None, :]}

    def score(self, uidx: np.ndarray) -> np.ndarray:
        c = self.components(uidx)
        return self.w_cf * c["cf"] + self.w_content * c["content"] + self.w_pop * c["pop"]

    def params(self) -> dict[str, Any]:
        return {"w_cf": self.w_cf, "w_content": self.w_content, "w_pop": self.w_pop,
                "feature_snapshot": getattr(self, "snapshot", None),
                **{f"cf_{k}": v for k, v in self.cf.params().items()}}
