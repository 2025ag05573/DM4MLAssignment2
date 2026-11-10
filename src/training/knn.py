"""Item-based collaborative filtering (item-kNN).

Two movies are similar when the same users rated both. Similarity is cosine on
the binary user x item "has rated" matrix, shrunk towards zero for item pairs
with few common raters (Koren 2008)::

    sim(i, j) = |U_i ∩ U_j| / (sqrt(|U_i| |U_j|) + shrink)

Only each item's ``neighbours`` most similar items are kept (a sparse matrix,
so the model file stays small), and a user's score for item j is the sum of
the similarities between j and the items the user has already rated (Sarwar
et al. 2001). It needs no latent factors and its recommendations are easy to
explain ("because you watched ...").
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np
import scipy.sparse as sp

from src.common.logging_utils import get_logger
from src.training.base import FitContext, Recommender

log = get_logger("training.knn")


class ItemKNNRecommender(Recommender):
    name = "item_knn"

    def __init__(self, shrink: float = 50.0, neighbours: int = 100):
        self.shrink, self.neighbours = shrink, neighbours

    def fit(self, ctx: FitContext) -> "ItemKNNRecommender":
        t0 = time.perf_counter()
        B = ctx.X.copy().tocsr()
        B.data[:] = 1.0                                   # binary: rated or not
        B = B.astype(np.float32)
        co = (B.T @ B).tocsr()                            # common-rater counts
        n = np.asarray(B.sum(axis=0)).ravel()
        co.sort_indices()                                 # ties in _top_k break by item index
        rows = np.repeat(np.arange(co.shape[0]), np.diff(co.indptr))
        co.data[rows == co.indices] = 0                   # drop self-similarity (same on every scipy)
        co.eliminate_zeros()
        rows = np.repeat(np.arange(co.shape[0]), np.diff(co.indptr))
        co.data = co.data / (np.sqrt(n[rows] * n[co.indices]) + self.shrink)
        self.S = self._top_k(co, self.neighbours)
        self.B = B
        log.info("ItemKNN shrink=%g neighbours=%d: %d similarities kept in %.1fs", self.shrink,
                 self.neighbours, self.S.nnz, time.perf_counter() - t0)
        return self

    @staticmethod
    def _top_k(S: sp.csr_matrix, k: int) -> sp.csr_matrix:
        """Keep the k largest similarities in every row (item -> its neighbours)."""
        indptr, indices, data = [0], [], []
        for r in range(S.shape[0]):
            lo, hi = S.indptr[r], S.indptr[r + 1]
            if hi - lo > k:
                # stable sort: ties are broken by item index, identically on every numpy version
                keep = lo + np.sort(np.argsort(-S.data[lo:hi], kind="stable")[:k])
            else:
                keep = np.arange(lo, hi)
            indices.append(S.indices[keep])
            data.append(S.data[keep])
            indptr.append(indptr[-1] + len(keep))
        # float64: each user's score sums float32-rounded similarities, which is exact in float64,
        # so the result does not depend on the summation order of the sparse product (scipy version)
        return sp.csr_matrix((np.concatenate(data) if data else [], np.concatenate(indices) if indices else [],
                              indptr), shape=S.shape, dtype=np.float64)

    def score(self, uidx: np.ndarray) -> np.ndarray:
        # sum over the user's rated items i of sim(i, j)  ==  B_u @ S  (rows of S = item i's neighbours j)
        return np.asarray((self.B[uidx] @ self.S).todense())

    def params(self) -> dict[str, Any]:
        return {"shrink": self.shrink, "neighbours": self.neighbours, "similarity": "shrunk cosine (binary)"}
