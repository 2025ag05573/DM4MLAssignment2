"""Batch / on-demand recommendation serving.

Uses a trained model plus the **serving** feature snapshot (all interactions up
to "now"), read through the same FeatureStore used in training - so the
features a model sees at prediction time are computed by exactly the same
code as during training.

Examples
--------
    python -m src.inference.recommend --user-id 42                  # top-10 for one user
    python -m src.inference.recommend --user-id 42 --model item_knn --k 5
    python -m src.inference.recommend --batch                       # all users -> parquet
"""
from __future__ import annotations

import argparse
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.common.config import get_paths, load_config
from src.common.io import write_parquet
from src.common.logging_utils import get_logger, relpath
from src.features.store import FeatureStore
from src.training.base import Recommender
from src.training.hybrid import HybridRecommender
from src.transformation.transform import interaction_matrix

log = get_logger("inference")


class RecommendationService:
    def __init__(self, cfg: dict[str, Any], model_name: str = "hybrid", snapshot: str = "serving"):
        self.cfg = cfg
        paths = get_paths(cfg)
        self.store = FeatureStore(cfg)
        self.snapshot = self.store.resolve(snapshot)
        self.model = Recommender.load(paths.models / f"{model_name}.pkl")
        if hasattr(self.model, "refresh_features"):   # popularity / hybrid read features at serve time
            self.model.refresh_features(self.store, self.snapshot)
        self.user_index = pd.read_parquet(paths.derived / "user_index.parquet")
        self.item_index = pd.read_parquet(paths.derived / "item_index.parquet")
        ratings = pd.read_parquet(paths.clean / "ratings.parquet")
        ratings = ratings[ratings["timestamp"] < self.store.as_of(self.snapshot)]
        self.seen = interaction_matrix(ratings, self.user_index, self.item_index)       # exclude watched
        self.recommendable = self.seen.getnnz(axis=0) > 0
        # A model can only personalise for users it was trained on, i.e. with a rating in its fit
        # window (train + val). Users whose first rating came later go to the fallback, like
        # brand-new users (ids not in the user table).
        fit = sp.load_npz(paths.derived / "interactions_final.npz")
        self.model_knows = np.diff(fit.tocsr().indptr) > 0
        self.movies = pd.read_parquet(paths.clean / "movies.parquet").set_index("movie_id")
        # Cold-start fallback: current popularity from the same feature snapshot.
        pop = self.store.get_item_features(self.item_index["movie_id"], self.snapshot, ["item_rating_count"])
        self.popular = pop["item_rating_count"].to_numpy(np.float64)
        self._uidx = self.user_index.set_index("user_id")["uidx"]

    def recommend(self, user_ids: list[int], k: int = 10) -> pd.DataFrame:
        uidx = self._uidx.reindex(user_ids)
        known = uidx.notna().to_numpy()
        u = uidx.fillna(0).to_numpy(np.int64)
        scores = np.tile(self.popular, (len(u), 1))
        has_history = known & self.model_knows[u]
        if has_history.any():
            scores[has_history] = np.asarray(self.model.score(u[has_history]), dtype=np.float64)
        scores[:, ~self.recommendable] = -np.inf
        seen = self.seen[u[known]]
        rows = np.flatnonzero(known)
        scores[np.repeat(rows, np.diff(seen.indptr)), seen.indices] = -np.inf
        top = np.argsort(-scores, axis=1, kind="stable")[:, :k]
        movie_ids = self.item_index["movie_id"].to_numpy()[top]
        out = pd.DataFrame({"user_id": np.repeat(user_ids, k), "rank": np.tile(np.arange(1, k + 1), len(u)),
                            "movie_id": movie_ids.ravel(),
                            "score": np.take_along_axis(scores, top, axis=1).ravel()})
        out["source"] = np.repeat(np.where(has_history, self.model.name, "popularity_fallback"), k)
        out["feature_snapshot"] = self.snapshot
        return out

    def explain(self, user_id: int, recs: pd.DataFrame) -> pd.DataFrame:
        recs = recs.join(self.movies[["title", "genres"]], on="movie_id")
        if user_id in self._uidx.index:
            uf = self.store.get_user_features([user_id], self.snapshot).iloc[0]
            recs["user_preferred_genre"] = uf["user_preferred_genre"]
        if isinstance(self.model, HybridRecommender) and (recs["source"] == self.model.name).all():
            comp = self.model.components(np.array([self._uidx[user_id]]))
            iidx = self.item_index.set_index("movie_id")["iidx"].reindex(recs["movie_id"]).to_numpy()
            recs["cf"] = comp["cf"][0, iidx]
            recs["genre_match"] = comp["content"][0, iidx]
            recs["popularity"] = comp["pop"][0, iidx]
        return recs


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--user-id", type=int, action="append", help="user id (repeatable)")
    ap.add_argument("--model", default="hybrid",
                    choices=["popularity", "explicit_mf", "item_knn", "implicit_als", "hybrid"])
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--batch", action="store_true", help="score every user and write a parquet file")
    args = ap.parse_args(argv)
    cfg = load_config()
    k = args.k or cfg["evaluation"]["k"]
    svc = RecommendationService(cfg, args.model)
    if args.batch:
        users = svc.user_index["user_id"].tolist()
        recs = pd.concat([svc.recommend(users[s:s + 1024], k) for s in range(0, len(users), 1024)])
        path = write_parquet(recs, get_paths(cfg).processed / "recommendations" / f"{args.model}_top{k}.parquet")
        n_fb = int((recs["source"] == "popularity_fallback").sum() / k)
        log.info("batch: %d users x %d -> %s (%d users scored by %s, %d on the popularity fallback: "
                 "no rating in the model's training window)", len(users), k, relpath(path), len(users) - n_fb,
                 args.model, n_fb)
        return
    for uid in args.user_id or [1]:
        recs = svc.explain(uid, svc.recommend([uid], k))
        pd.set_option("display.width", 200, "display.max_colwidth", 45)
        pref = recs["user_preferred_genre"].iat[0] if "user_preferred_genre" in recs else "n/a (new user)"
        print(f"\nTop-{k} for user {uid} | {recs['source'].iat[0]} | features={svc.snapshot} | "
              f"preferred genre: {pref}")
        cols = [c for c in ["rank", "title", "genres", "score", "cf", "genre_match", "popularity"] if c in recs]
        print(recs[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
