"""Stage 5 - Feature materialisation (point-in-time snapshots).

Features are computed *as of* a timestamp using only interactions strictly
before it, and written as versioned snapshots::

    data/features/<group>/<snapshot>.parquet
    data/features/registry.json

Three snapshots are built from the same code:

=========  =====================  ==========================================
snapshot   as_of                  consumer
=========  =====================  ==========================================
tune       start of validation    models fitted on train, scored on val
final      start of test          models fitted on train+val, scored on test
serving    after last interaction online/batch prediction (src/inference)
=========  =====================  ==========================================

Because training and serving read the *same* feature definitions through the
same store, there is no training/serving skew.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
import pandas as pd

from src.common.config import get_paths, load_config
from src.common.io import utc_now, write_json, write_parquet
from src.common.logging_utils import get_logger, relpath
from src.features.definitions import FEATURES, by_group
from src.preparation.prepare import genre_column
from src.transformation.transform import (item_popularity, user_activity_summary,
                                          user_genre_affinity, window)

log = get_logger("features")
DAY = 86_400


def compute_item_features(ratings: pd.DataFrame, movies: pd.DataFrame, as_of: int,
                          cfg: dict[str, Any]) -> pd.DataFrame:
    f = cfg["features"]
    pop = item_popularity(ratings, as_of, f["bayesian_prior_count"], f["trend_window_days"])
    global_mean = float(window(ratings, as_of)["rating"].mean())
    genres = cfg["validation"]["genres"]
    base = movies[["movie_id", "release_year", "n_genres"] + [genre_column(g) for g in genres]]
    df = base.merge(pop, on="movie_id", how="left")
    as_of_year = pd.Timestamp(as_of, unit="s", tz="UTC").year
    out = pd.DataFrame({
        "movie_id": df["movie_id"],
        "item_rating_count": df["n_ratings"].fillna(0).astype("int32"),
        "item_avg_rating": df["mean_rating"].astype("float32"),
        "item_bayesian_avg": df["bayesian_avg"].fillna(global_mean).astype("float32"),
        "item_popularity_rank": df["popularity_rank"].fillna(len(pop) + 1).astype("int32"),
        "item_popularity_pct": df["popularity_pct"].fillna(0.0).astype("float32"),
        "item_recent_count": df["n_ratings_recent"].fillna(0).astype("int32"),
        "item_days_since_last_interaction": df["days_since_last"].astype("float32"),
        "item_release_year": df["release_year"].astype("Int16"),
        "item_age_years": (as_of_year - df["release_year"]).astype("Float32"),
        "item_n_genres": df["n_genres"].astype("int8"),
    })
    for g in genres:
        out[f"item_genre_{genre_column(g)[6:]}"] = df[genre_column(g)].astype("int8")
    return out


def compute_user_features(ratings: pd.DataFrame, users: pd.DataFrame, movies: pd.DataFrame,
                          item_feats: pd.DataFrame, as_of: int, cfg: dict[str, Any]) -> pd.DataFrame:
    genres = cfg["validation"]["genres"]
    r = window(ratings, as_of)
    global_mean = float(r["rating"].mean())
    act = user_activity_summary(ratings, as_of)
    aff = user_genre_affinity(ratings, movies, genres, as_of, cfg["evaluation"]["relevance_threshold"])
    main = (r.merge(item_feats[["movie_id", "item_popularity_pct"]], on="movie_id")
            .groupby("user_id")["item_popularity_pct"].mean().rename("user_mainstreamness"))
    df = users.merge(act, on="user_id", how="left").merge(main, on="user_id", how="left")
    out = pd.DataFrame({
        "user_id": df["user_id"],
        "user_rating_count": df["n_ratings"].fillna(0).astype("int32"),
        "user_avg_rating": df["mean_rating"].fillna(global_mean).astype("float32"),
        "user_rating_std": df["std_rating"].fillna(0.0).astype("float32"),
        "user_positive_share": (df["n_positive"] / df["n_ratings"]).fillna(0.0).astype("float32"),
        "user_days_since_last_interaction": df["days_since_last"].astype("float32"),
        "user_tenure_days": df["tenure_days"].astype("float32"),
        "user_activity_rate": df["ratings_per_active_day"].fillna(0.0).astype("float32"),
    })
    affv = aff.reindex(out["user_id"]).fillna(0.0)
    has_hist = affv.sum(axis=1).to_numpy() > 0
    out["user_preferred_genre"] = np.where(has_hist, affv.idxmax(axis=1).to_numpy(), "unknown")
    for g in genres:
        out[f"user_genre_affinity_{genre_column(g)[6:]}"] = affv[g].to_numpy(dtype="float32")
    out["user_mainstreamness"] = df["user_mainstreamness"].astype("float32").to_numpy()
    out["user_gender"] = df["gender"].to_numpy()
    out["user_age_group"] = df["age_group"].to_numpy()
    out["user_occupation"] = df["occupation"].to_numpy()
    out["user_zip_region"] = df["zip_region"].to_numpy()
    for c in ("user_preferred_genre", "user_gender", "user_age_group", "user_occupation", "user_zip_region"):
        out[c] = out[c].astype("string")
    return out


def compute_interaction_features(ratings: pd.DataFrame, user_feats: pd.DataFrame,
                                 item_feats: pd.DataFrame, as_of: int, cfg: dict[str, Any]) -> pd.DataFrame:
    r = window(ratings, as_of)[["user_id", "movie_id", "rating", "timestamp"]]
    df = (r.merge(user_feats[["user_id", "user_avg_rating"]], on="user_id", how="left")
           .merge(item_feats[["movie_id", "item_avg_rating"]], on="movie_id", how="left"))
    age = (as_of - df["timestamp"]) / DAY
    return pd.DataFrame({
        "user_id": df["user_id"], "movie_id": df["movie_id"],
        "ui_rating": df["rating"].astype("int8"),
        "ui_rating_dev_user": (df["rating"] - df["user_avg_rating"]).astype("float32"),
        "ui_rating_dev_item": (df["rating"] - df["item_avg_rating"]).astype("float32"),
        "ui_days_before_as_of": age.astype("float32"),
        "ui_recency_weight": (0.5 ** (age / cfg["features"]["recency_half_life_days"])).astype("float32"),
        "ui_is_positive": (df["rating"] >= cfg["evaluation"]["relevance_threshold"]).astype("int8"),
    })


def expected_columns(group: str, genres: list[str]) -> list[str]:
    cols = []
    for f in by_group(group):
        if f.n_columns > 1:
            cols += [f.name.replace("<genre>", genre_column(g)[6:]) for g in genres]
        elif f.name != "ui_genre_match":  # on-demand, never materialised
            cols.append(f.name)
    return cols


def definitions_version() -> str:
    payload = json.dumps([f.to_dict() for f in FEATURES], sort_keys=True).encode()
    return hashlib.md5(payload).hexdigest()[:12]


def materialise(cfg: dict[str, Any], snapshot: str, as_of: int) -> dict[str, Any]:
    paths = get_paths(cfg)
    genres = cfg["validation"]["genres"]
    ratings = pd.read_parquet(paths.clean / "ratings.parquet")
    users = pd.read_parquet(paths.clean / "users.parquet")
    movies = pd.read_parquet(paths.clean / "movies.parquet")

    item_f = compute_item_features(ratings, movies, as_of, cfg)
    user_f = compute_user_features(ratings, users, movies, item_f, as_of, cfg)
    inter_f = compute_interaction_features(ratings, user_f, item_f, as_of, cfg)
    meta = {"as_of": as_of, "as_of_iso": pd.Timestamp(as_of, unit="s", tz="UTC").isoformat(),
            "created_at": utc_now(), "groups": {}}
    for group, df, key in (("user_features", user_f, ["user_id"]), ("item_features", item_f, ["movie_id"]),
                           ("interaction_features", inter_f, ["user_id", "movie_id"])):
        # Contract between registry and materialised data.
        missing = set(expected_columns(group, genres)) - set(df.columns)
        extra = set(df.columns) - set(expected_columns(group, genres)) - set(key)
        if missing or extra:
            raise ValueError(f"{group}: registry mismatch missing={missing} extra={extra}")
        path = write_parquet(df, paths.features / group / f"{snapshot}.parquet")
        meta["groups"][group] = {"path": relpath(path), "rows": len(df), "entity": key,
                                 "n_features": df.shape[1] - len(key)}
    return meta


def run(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = cfg or load_config()
    paths = get_paths(cfg)
    val = pd.read_parquet(paths.splits / "val.parquet", columns=["timestamp"])
    test = pd.read_parquet(paths.splits / "test.parquet", columns=["timestamp"])
    snapshots = {"tune": int(val["timestamp"].min()), "final": int(test["timestamp"].min()),
                 "serving": int(test["timestamp"].max()) + 1}
    registry = {"definitions_version": definitions_version(), "snapshots": {}}
    for name, as_of in snapshots.items():
        registry["snapshots"][name] = materialise(cfg, name, as_of)
        g = registry["snapshots"][name]["groups"]
        log.info("snapshot %-7s as_of=%s users=%d items=%d interactions=%d", name,
                 registry["snapshots"][name]["as_of_iso"][:19], g["user_features"]["rows"],
                 g["item_features"]["rows"], g["interaction_features"]["rows"])
    write_json(registry, paths.features / "registry.json")
    from src.features.catalog import write_catalog
    write_catalog(cfg)
    return registry


if __name__ == "__main__":
    run()
