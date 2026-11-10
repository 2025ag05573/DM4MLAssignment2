"""Evaluation protocol shared by tuning (val window) and final evaluation (test window).

* **Users evaluated** - users with history in the fit window (warm) who have at
  least one relevant (rating >= threshold) held-out item. Cold-start users are
  counted and reported but cannot be scored by collaborative models.
* **Candidates** - items with at least one interaction in the fit window, minus
  everything the user already rated in the fit window.
* **Ground truth** - held-out items rated >= threshold (warm items only).
* **Rating metrics** - every held-out (warm user, warm item) rating.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.evaluation import metrics as M
from src.training.base import FitContext, Recommender


@dataclass
class EvalSet:
    name: str
    uidx: np.ndarray
    relevant: list[set]
    candidate_mask: np.ndarray
    item_vectors: np.ndarray
    item_pop_share: np.ndarray
    pairs_u: np.ndarray
    pairs_i: np.ndarray
    pairs_r: np.ndarray
    stats: dict[str, Any]


def build_eval_set(ctx: FitContext, heldout: pd.DataFrame, name: str) -> EvalSet:
    thr = ctx.cfg["evaluation"]["relevance_threshold"]
    u_map = ctx.user_index.set_index("user_id")["uidx"]
    i_map = ctx.item_index.set_index("movie_id")["iidx"]
    h = heldout.assign(uidx=heldout["user_id"].map(u_map).to_numpy(),
                       iidx=heldout["movie_id"].map(i_map).to_numpy())
    user_hist = np.diff(ctx.X.indptr) > 0
    item_hist = ctx.X.getnnz(axis=0) > 0
    warm = h[user_hist[h["uidx"]] & item_hist[h["iidx"]]]
    rel = warm[warm["rating"] >= thr].groupby("uidx")["iidx"].apply(lambda s: set(s.tolist()))
    G = ctx.store.item_genre_matrix(ctx.item_index["movie_id"], ctx.snapshot)
    G = G / np.maximum(np.linalg.norm(G, axis=1, keepdims=True), 1e-9)
    counts = ctx.X.getnnz(axis=0).astype(np.float64)
    stats = {"heldout_rows": int(len(h)), "heldout_users": int(h["user_id"].nunique()),
             "cold_start_users": int((~user_hist[h["uidx"].drop_duplicates()]).sum()),
             "rows_on_cold_items": int((~item_hist[h["iidx"]]).sum()),
             "eval_users": int(len(rel)), "candidate_items": int(item_hist.sum()),
             "mean_relevant_per_user": float(rel.map(len).mean()) if len(rel) else 0.0,
             "rating_pairs": int(len(warm))}
    return EvalSet(name, rel.index.to_numpy(np.int64), rel.tolist(), item_hist, G,
                   counts / counts.sum(), warm["uidx"].to_numpy(np.int64), warm["iidx"].to_numpy(np.int64),
                   warm["rating"].to_numpy(np.float64), stats)


def top_k(model: Recommender, ctx: FitContext, es: EvalSet, k: int, batch: int = 1024) -> np.ndarray:
    out = np.empty((len(es.uidx), k), dtype=np.int64)
    for s in range(0, len(es.uidx), batch):
        u = es.uidx[s:s + batch]
        scores = np.asarray(model.score(u), dtype=np.float64)
        scores[:, ~es.candidate_mask] = -np.inf
        seen = ctx.X[u]
        scores[np.repeat(np.arange(len(u)), np.diff(seen.indptr)), seen.indices] = -np.inf
        # Full stable sort rather than argpartition: tied scores (common for popularity counts
        # and kNN sums) are broken by item index, so results do not depend on the numpy version.
        out[s:s + batch] = np.argsort(-scores, axis=1, kind="stable")[:, :k]
    return out


def evaluate(model: Recommender, ctx: FitContext, es: EvalSet, k_values: list[int],
             per_user: bool = False) -> dict[str, float] | tuple[dict[str, float], pd.DataFrame]:
    """Aggregate metrics; with ``per_user=True`` also the per-user ranking metrics at max(K)
    (used for bootstrap confidence intervals and user-segment breakdowns)."""
    kmax = max(k_values)
    recs = top_k(model, ctx, es, kmax)
    res: dict[str, float] = {}
    for k in k_values:
        rk = recs[:, :k]
        res[f"precision@{k}"] = float(np.mean([M.precision_at_k(r, g, k) for r, g in zip(rk, es.relevant)]))
        res[f"recall@{k}"] = float(np.mean([M.recall_at_k(r, g, k) for r, g in zip(rk, es.relevant)]))
        res[f"ndcg@{k}"] = float(np.mean([M.ndcg_at_k(r, g, k) for r, g in zip(rk, es.relevant)]))
        res[f"hit_rate@{k}"] = float(np.mean([M.hit_rate_at_k(r, g, k) for r, g in zip(rk, es.relevant)]))
        res[f"coverage@{k}"] = M.catalog_coverage(rk, int(es.candidate_mask.sum()))
        res[f"diversity@{k}"] = M.intra_list_diversity(rk, es.item_vectors)
        res[f"novelty@{k}"] = M.novelty(rk, es.item_pop_share)
    if model.predicts_ratings:
        pred = model.predict_ratings(es.pairs_u, es.pairs_i)
        res["rmse"] = M.rmse(es.pairs_r, pred)
        res["mae"] = M.mae(es.pairs_r, pred)
    if per_user:
        k = max(k_values)
        rk = recs[:, :k]
        pu = pd.DataFrame({
            "uidx": es.uidx,
            f"ndcg@{k}": [M.ndcg_at_k(r, g, k) for r, g in zip(rk, es.relevant)],
            f"recall@{k}": [M.recall_at_k(r, g, k) for r, g in zip(rk, es.relevant)],
            f"hit_rate@{k}": [M.hit_rate_at_k(r, g, k) for r, g in zip(rk, es.relevant)],
        })
        return res, pu
    return res
