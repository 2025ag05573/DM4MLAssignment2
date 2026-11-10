"""Ranking, beyond-accuracy and rating metrics. Pure NumPy, unit-tested in tests/.

Notation: ``recs`` is the ranked top-K list for one user, ``relevant`` the set of
held-out items that user rated >= the relevance threshold.
"""
from __future__ import annotations

import numpy as np


def precision_at_k(recs: np.ndarray, relevant: set, k: int) -> float:
    """|top-K ∩ relevant| / K"""
    return sum(1 for i in recs[:k] if i in relevant) / k


def recall_at_k(recs: np.ndarray, relevant: set, k: int) -> float:
    """|top-K ∩ relevant| / |relevant|"""
    return sum(1 for i in recs[:k] if i in relevant) / len(relevant) if relevant else 0.0


def ndcg_at_k(recs: np.ndarray, relevant: set, k: int) -> float:
    """Binary-gain NDCG: DCG = sum 1/log2(rank+1) over hits; IDCG = best possible DCG."""
    dcg = sum(1.0 / np.log2(r + 2) for r, i in enumerate(recs[:k]) if i in relevant)
    idcg = sum(1.0 / np.log2(r + 2) for r in range(min(len(relevant), k)))
    return dcg / idcg if idcg > 0 else 0.0


def hit_rate_at_k(recs: np.ndarray, relevant: set, k: int) -> float:
    return float(any(i in relevant for i in recs[:k]))


def catalog_coverage(all_recs: np.ndarray, n_catalog: int) -> float:
    """Share of the recommendable catalog that appears in at least one top-K list."""
    return len(np.unique(all_recs)) / n_catalog


def intra_list_diversity(all_recs: np.ndarray, item_vectors: np.ndarray) -> float:
    """Mean pairwise (1 - cosine similarity) of the items inside each list, averaged over users.
    ``item_vectors`` are L2-normalised content vectors (here: genre one-hots)."""
    V = item_vectors[all_recs]                       # (users, K, d)
    S = np.einsum("ukd,ujd->ukj", V, V)              # pairwise cosine similarities
    k = all_recs.shape[1]
    off_diag = (S.sum(axis=(1, 2)) - np.einsum("ukk->u", S)) / (k * (k - 1))
    return float(np.mean(1.0 - off_diag))


def novelty(all_recs: np.ndarray, item_pop_share: np.ndarray) -> float:
    """Mean self-information -log2(p(i)) of recommended items (higher = less mainstream)."""
    p = np.clip(item_pop_share[all_recs], 1e-12, None)
    return float(np.mean(-np.log2(p)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))
