"""Baseline: non-personalised popularity recommender.

Ranks items by how many ratings they received in the fit window (read from the
feature store's ``item_rating_count``) and recommends the most popular items
each user has not yet seen. As a rating predictor it returns the item's
damped mean rating (``item_bayesian_avg``), the standard non-personalised
baseline for RMSE/MAE.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from src.training.base import FitContext, Recommender

DAY = 86_400


class PopularityRecommender(Recommender):
    name = "popularity"
    predicts_ratings = True

    def __init__(self, window_days: int | None = None):
        self.window_days = window_days
        self.item_scores: np.ndarray | None = None
        self.item_rating: np.ndarray | None = None

    def fit(self, ctx: FitContext) -> "PopularityRecommender":
        movie_ids = ctx.item_index["movie_id"].to_numpy()
        feats = ctx.store.get_item_features(movie_ids, ctx.snapshot,
                                            ["item_rating_count", "item_bayesian_avg"])
        if self.window_days is None:
            counts = feats["item_rating_count"].to_numpy(np.float64)
        else:  # popularity over a recent window only
            r = ctx.interactions
            r = r[r["timestamp"] >= r["timestamp"].max() - self.window_days * DAY]
            counts = (r.groupby("movie_id").size().reindex(movie_ids).fillna(0).to_numpy(np.float64))
        self.item_scores = counts.astype(np.float32)
        self.item_rating = feats["item_bayesian_avg"].to_numpy(np.float32)
        return self

    def refresh_features(self, store, snapshot: str = "serving") -> "PopularityRecommender":
        """Serving: re-read popularity from a newer feature snapshot (no refit needed)."""
        if self.window_days is None:
            ids = store.get_item_features(snapshot=snapshot).index
            f = store.get_item_features(ids, snapshot, ["item_rating_count", "item_bayesian_avg"])
            self.item_scores = f["item_rating_count"].to_numpy(np.float32)
            self.item_rating = f["item_bayesian_avg"].to_numpy(np.float32)
        return self

    def score(self, uidx: np.ndarray) -> np.ndarray:
        return np.broadcast_to(self.item_scores, (len(uidx), len(self.item_scores))).copy()

    def predict_ratings(self, uidx: np.ndarray, iidx: np.ndarray) -> np.ndarray:
        return self.item_rating[iidx]

    def params(self) -> dict[str, Any]:
        return {"window_days": self.window_days, "ranking_signal": "item_rating_count",
                "rating_signal": "item_bayesian_avg"}
