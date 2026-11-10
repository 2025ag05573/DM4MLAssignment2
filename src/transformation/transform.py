"""Stage 4 - Data transformation: derived datasets.

The aggregation functions here are *pure* and take an ``as_of`` timestamp, so
the feature layer (``src/features``) re-uses exactly the same logic to build
point-in-time features - one definition of "item popularity" for analysis,
training and serving.

Outputs (data/processed/derived/)
---------------------------------
user_index.parquet / item_index.parquet   id <-> matrix-row/column mapping
interactions_tune.npz                     user x item rating matrix of ``train``
interactions_final.npz                    user x item rating matrix of ``train + val``
user_activity_summary.parquet             per-user activity (final fit window)
item_popularity.parquet                   per-item popularity (final fit window)
genre_summary.parquet                     per-genre volume and mean rating
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.common.config import get_paths, load_config
from src.common.io import write_json, write_parquet
from src.common.logging_utils import get_logger
from src.preparation.prepare import genre_column

log = get_logger("transformation")
DAY = 86_400


# --------------------------------------------------------------------------- pure functions
def window(ratings: pd.DataFrame, as_of: int) -> pd.DataFrame:
    """Only interactions strictly before ``as_of`` (point-in-time correctness)."""
    return ratings[ratings["timestamp"] < as_of]


def user_activity_summary(ratings: pd.DataFrame, as_of: int) -> pd.DataFrame:
    r = window(ratings, as_of)
    g = r.groupby("user_id")
    out = pd.DataFrame({
        "n_ratings": g.size(),
        "mean_rating": g["rating"].mean(),
        "std_rating": g["rating"].std(ddof=0),
        "n_positive": g["rating"].apply(lambda s: int((s >= 4).sum())),
        "first_ts": g["timestamp"].min(),
        "last_ts": g["timestamp"].max(),
    })
    out["active_span_days"] = (out["last_ts"] - out["first_ts"]) / DAY
    out["days_since_last"] = (as_of - out["last_ts"]) / DAY
    out["tenure_days"] = (as_of - out["first_ts"]) / DAY
    out["ratings_per_active_day"] = out["n_ratings"] / np.maximum(out["active_span_days"], 1.0)
    return out.reset_index()


def item_popularity(ratings: pd.DataFrame, as_of: int, prior_m: float, trend_days: int) -> pd.DataFrame:
    r = window(ratings, as_of)
    global_mean = float(r["rating"].mean())
    g = r.groupby("movie_id")
    out = pd.DataFrame({"n_ratings": g.size(), "sum_rating": g["rating"].sum(),
                        "mean_rating": g["rating"].mean(), "last_ts": g["timestamp"].max()})
    # Damped (Bayesian) mean: shrinks items with few ratings towards the global mean.
    out["bayesian_avg"] = (out["sum_rating"] + prior_m * global_mean) / (out["n_ratings"] + prior_m)
    out["popularity_rank"] = out["n_ratings"].rank(ascending=False, method="min").astype(int)
    out["popularity_pct"] = out["n_ratings"].rank(pct=True)
    out["share_of_interactions"] = out["n_ratings"] / len(r)
    recent = r[r["timestamp"] >= as_of - trend_days * DAY].groupby("movie_id").size()
    out["n_ratings_recent"] = recent.reindex(out.index).fillna(0).astype(int)
    out["days_since_last"] = (as_of - out["last_ts"]) / DAY
    return out.drop(columns="sum_rating").reset_index()


def genre_matrix(movies: pd.DataFrame, genres: list[str]) -> pd.DataFrame:
    return movies.set_index("movie_id")[[genre_column(g) for g in genres]]


def user_genre_affinity(ratings: pd.DataFrame, movies: pd.DataFrame, genres: list[str], as_of: int,
                        threshold: int = 4) -> pd.DataFrame:
    """Share of a user's *liked* items (rating >= threshold) in each genre (rows sum to 1).
    Users with no liked items fall back to all their ratings."""
    r = window(ratings, as_of)
    gm = genre_matrix(movies, genres)
    liked = r[r["rating"] >= threshold]
    fallback = r[~r["user_id"].isin(liked["user_id"])]
    base = pd.concat([liked, fallback])
    G = gm.reindex(base["movie_id"]).to_numpy(dtype=float)
    agg = pd.DataFrame(G, columns=genres).groupby(base["user_id"].to_numpy()).sum()
    agg = agg.div(agg.sum(axis=1).replace(0, 1), axis=0)
    agg.index.name = "user_id"
    return agg


def index_maps(users: pd.DataFrame, movies: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ui = pd.DataFrame({"user_id": np.sort(users["user_id"].unique())})
    ui["uidx"] = np.arange(len(ui), dtype=np.int32)
    ii = pd.DataFrame({"movie_id": np.sort(movies["movie_id"].unique())})
    ii["iidx"] = np.arange(len(ii), dtype=np.int32)
    return ui, ii


def interaction_matrix(ratings: pd.DataFrame, user_index: pd.DataFrame, item_index: pd.DataFrame
                       ) -> sp.csr_matrix:
    u = ratings["user_id"].map(user_index.set_index("user_id")["uidx"]).to_numpy()
    i = ratings["movie_id"].map(item_index.set_index("movie_id")["iidx"]).to_numpy()
    shape = (len(user_index), len(item_index))
    X = sp.csr_matrix((ratings["rating"].to_numpy(np.float32), (u, i)), shape=shape)
    # Canonical CSR (sorted column indices, no duplicates): scipy versions differ in the
    # in-row order they produce, and downstream tie-breaking must not depend on it.
    X.sum_duplicates()
    X.sort_indices()
    return X


# --------------------------------------------------------------------------- stage
def run(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = cfg or load_config()
    paths = get_paths(cfg)
    fcfg = cfg["features"]
    users = pd.read_parquet(paths.clean / "users.parquet")
    movies = pd.read_parquet(paths.clean / "movies.parquet")
    train = pd.read_parquet(paths.splits / "train.parquet")
    val = pd.read_parquet(paths.splits / "val.parquet")
    test = pd.read_parquet(paths.splits / "test.parquet")
    fit_final = pd.concat([train, val], ignore_index=True)
    t_test = int(test["timestamp"].min())

    ui, ii = index_maps(users, movies)
    write_parquet(ui, paths.derived / "user_index.parquet")
    write_parquet(ii, paths.derived / "item_index.parquet")
    stats = {}
    for stage, df in (("tune", train), ("final", fit_final)):
        X = interaction_matrix(df, ui, ii)
        sp.save_npz(paths.derived / f"interactions_{stage}.npz", X)
        stats[stage] = {"shape": list(X.shape), "nnz": int(X.nnz),
                        "density": float(X.nnz / (X.shape[0] * X.shape[1]))}
        log.info("interaction matrix [%s] %s nnz=%d density=%.4f", stage, X.shape, X.nnz,
                 stats[stage]["density"])

    uas = user_activity_summary(fit_final, t_test).merge(users, on="user_id", how="left")
    ipop = item_popularity(fit_final, t_test, fcfg["bayesian_prior_count"], fcfg["trend_window_days"]) \
        .merge(movies[["movie_id", "title", "release_year", "genres"]], on="movie_id", how="left") \
        .sort_values("popularity_rank")
    genres = cfg["validation"]["genres"]
    gm = genre_matrix(movies, genres)
    rg = fit_final.join(gm, on="movie_id")
    genre_summary = pd.DataFrame([{"genre": g, "n_ratings": int(rg[genre_column(g)].sum()),
                                   "mean_rating": float(rg.loc[rg[genre_column(g)] == 1, "rating"].mean()),
                                   "n_movies": int(gm[genre_column(g)].sum())} for g in genres]) \
        .sort_values("n_ratings", ascending=False)
    write_parquet(uas, paths.derived / "user_activity_summary.parquet")
    write_parquet(ipop, paths.derived / "item_popularity.parquet")
    write_parquet(genre_summary, paths.derived / "genre_summary.parquet")
    stats.update({"user_activity_summary_rows": len(uas), "item_popularity_rows": len(ipop),
                  "top5_items": ipop.head(5)[["title", "n_ratings"]].to_dict("records")})
    write_json(stats, paths.reports / "transformation_summary.json")
    log.info("derived: user_activity_summary=%d item_popularity=%d genre_summary=%d",
             len(uas), len(ipop), len(genre_summary))
    return stats


if __name__ == "__main__":
    run()
