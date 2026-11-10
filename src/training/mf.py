"""Personalised matrix-factorisation recommenders, implemented with NumPy/SciPy.

Both models use Alternating Least Squares: with one side's factors fixed, the
optimal factors of the other side are a ridge-regression closed form, so each
half-step is exact and the loss decreases monotonically. Implementing them
directly (instead of e.g. ``implicit`` / ``surprise``) keeps the project free of
compiled dependencies, so ``pip install -r requirements.txt`` works on any OS.

* :class:`ExplicitMF`  - biased MF for explicit ratings (Koren et al. 2009),
  fitted by ALS. Regularisation is weighted-lambda ALS-WR (Zhou et al. 2008,
  ``reg_mode="weighted"``), which scales lambda by each row's rating count, or
  plain ridge (``reg_mode="unweighted"``). The ``factor_term_std`` diagnostic
  shows whether the factors are actually used (0 = biases-only model).
  Predicts ratings -> RMSE / MAE; ranks by predicted rating.
* :class:`ImplicitALS` - implicit-feedback MF (Hu, Koren & Volinsky 2008).
  Every observed rating is a positive preference with confidence
  ``c = 1 + alpha * rating``; unobserved cells are weak negatives. Optimises
  ranking rather than rating error.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np
import scipy.sparse as sp

from src.common.logging_utils import get_logger
from src.training.base import FitContext, Recommender

log = get_logger("training.mf")


class ExplicitMF(Recommender):
    name = "explicit_mf"
    predicts_ratings = True

    def __init__(self, factors: int = 32, reg: float = 0.1, iterations: int = 15, seed: int = 42,
                 reg_mode: str = "weighted", bias_reg: float | None = None):
        """reg_mode "weighted" = lambda * n_ratings of the row (ALS-WR, Zhou et al. 2008);
        "unweighted" = plain lambda (classic biased MF, Koren et al. 2009), which lets users and
        items with many ratings keep larger factors. ``bias_reg`` (default = reg) regularises the
        bias terms separately."""
        self.factors, self.reg, self.iterations, self.seed = factors, reg, iterations, seed
        if reg_mode not in ("weighted", "unweighted"):
            raise ValueError(f"reg_mode must be 'weighted' or 'unweighted', got {reg_mode!r}")
        self.reg_mode, self.bias_reg = reg_mode, bias_reg
        self.history: list[float] = []

    def fit(self, ctx: FitContext) -> "ExplicitMF":
        rng = np.random.default_rng(self.seed)
        X = ctx.X.tocsr()
        Xt = X.T.tocsr()
        nu, ni, f = X.shape[0], X.shape[1], self.factors
        self.mu = float(X.data.mean())
        self.P = rng.normal(0, 0.1, (nu, f)).astype(np.float64)
        self.Q = rng.normal(0, 0.1, (ni, f)).astype(np.float64)
        self.bu = np.zeros(nu)
        self.bi = np.zeros(ni)
        eye = np.diag([1.0] * f + [(self.bias_reg if self.bias_reg is not None else self.reg) / self.reg])
        t0 = time.perf_counter()
        for it in range(self.iterations):
            self._half_step(X, self.Q, self.bi, self.P, self.bu, eye)   # users
            self._half_step(Xt, self.P, self.bu, self.Q, self.bi, eye)  # items
            self.history.append(self._train_rmse(X))
        # How much the personal (factor) term actually contributes on the training pairs:
        # ~0 means the model has collapsed to user + item biases.
        u, i = X.nonzero()
        self.factor_term_std = float(np.std(np.einsum("ij,ij->i", self.P[u], self.Q[i])))
        log.info("ExplicitMF f=%d reg=%.3g (%s): %d its in %.1fs, train RMSE %.4f, factor-term std %.4f", f,
                 self.reg, self.reg_mode, self.iterations, time.perf_counter() - t0, self.history[-1],
                 self.factor_term_std)
        return self

    def _half_step(self, R: sp.csr_matrix, Fixed: np.ndarray, fixed_bias: np.ndarray,
                   Out: np.ndarray, out_bias: np.ndarray, eye: np.ndarray) -> None:
        indptr, indices, data = R.indptr, R.indices, R.data
        for row in range(R.shape[0]):
            lo, hi = indptr[row], indptr[row + 1]
            if lo == hi:
                Out[row] = 0.0
                out_bias[row] = 0.0
                continue
            cols = indices[lo:hi]
            target = data[lo:hi] - self.mu - fixed_bias[cols]
            A = np.hstack([Fixed[cols], np.ones((hi - lo, 1))])     # [factors | bias]
            lam = self.reg * (hi - lo) if self.reg_mode == "weighted" else self.reg
            lhs = A.T @ A + lam * eye                               # ALS-WR (weighted) or plain ridge
            w = np.linalg.solve(lhs, A.T @ target)
            Out[row], out_bias[row] = w[:-1], w[-1]

    def _train_rmse(self, X: sp.csr_matrix) -> float:
        u, i = X.nonzero()
        pred = self.mu + self.bu[u] + self.bi[i] + np.einsum("ij,ij->i", self.P[u], self.Q[i])
        return float(np.sqrt(np.mean((np.asarray(X[u, i]).ravel() - pred) ** 2)))

    def score(self, uidx: np.ndarray) -> np.ndarray:
        return (self.mu + self.bu[uidx, None] + self.bi[None, :] + self.P[uidx] @ self.Q.T).astype(np.float32)

    def predict_ratings(self, uidx: np.ndarray, iidx: np.ndarray) -> np.ndarray:
        pred = self.mu + self.bu[uidx] + self.bi[iidx] + np.einsum("ij,ij->i", self.P[uidx], self.Q[iidx])
        return np.clip(pred, 1.0, 5.0)

    def params(self) -> dict[str, Any]:
        return {"factors": self.factors, "reg": self.reg, "reg_mode": self.reg_mode,
                "bias_reg": self.bias_reg if self.bias_reg is not None else self.reg,
                "iterations": self.iterations, "seed": self.seed}

    def diagnostics(self) -> dict[str, float]:
        # std of the p_u.q_i term over training pairs; ~0 = collapsed to a biases-only model
        return {"factor_term_std": round(float(getattr(self, "factor_term_std", float("nan"))), 5)}


class ImplicitALS(Recommender):
    """Implicit ALS. ``solver="cg"`` (default) runs a few conjugate-gradient steps per
    half-iteration, vectorised over all rows with sparse algebra (Takacs et al. 2011,
    as in the ``implicit`` library); ``solver="exact"`` solves each row's normal
    equations directly (slower, used as a reference in the tests)."""

    name = "implicit_als"

    def __init__(self, factors: int = 64, reg: float = 0.1, alpha: float = 10.0, iterations: int = 15,
                 seed: int = 42, solver: str = "cg", cg_steps: int = 3):
        self.factors, self.reg, self.alpha, self.iterations, self.seed = factors, reg, alpha, iterations, seed
        self.solver, self.cg_steps = solver, cg_steps

    def fit(self, ctx: FitContext) -> "ImplicitALS":
        rng = np.random.default_rng(self.seed)
        C = ctx.X.tocsr().astype(np.float32)
        C.data = 1.0 + self.alpha * C.data          # confidence of each observed interaction
        Ct = C.T.tocsr()
        nu, ni, f = C.shape[0], C.shape[1], self.factors
        self.U = rng.normal(0, 0.01, (nu, f)).astype(np.float32)
        self.V = rng.normal(0, 0.01, (ni, f)).astype(np.float32)
        step = self._cg_half_step if self.solver == "cg" else self._exact_half_step
        t0 = time.perf_counter()
        for _ in range(self.iterations):
            step(C, self.V, self.U)
            step(Ct, self.U, self.V)
        log.info("ImplicitALS f=%d reg=%.3g alpha=%.3g: %d its (%s) in %.1fs", f, self.reg, self.alpha,
                 self.iterations, self.solver, time.perf_counter() - t0)
        return self

    def _exact_half_step(self, C: sp.csr_matrix, Y: np.ndarray, Out: np.ndarray) -> None:
        # x_u = (Y^T Y + Y_u^T (C_u - I) Y_u + reg I)^-1  Y_u^T C_u p_u   (p_u = 1 on observed)
        YtY = Y.T @ Y + self.reg * np.eye(Y.shape[1], dtype=Y.dtype)
        indptr, indices, data = C.indptr, C.indices, C.data
        for row in range(C.shape[0]):
            lo, hi = indptr[row], indptr[row + 1]
            if lo == hi:
                Out[row] = 0.0
                continue
            Yu = Y[indices[lo:hi]]
            c = data[lo:hi]
            A = YtY + (Yu.T * (c - 1.0)) @ Yu
            Out[row] = np.linalg.solve(A, Yu.T @ c)

    def _cg_half_step(self, C: sp.csr_matrix, Y: np.ndarray, X: np.ndarray, block: int = 1024) -> None:
        """Warm-started CG on A_u x_u = b_u for every row u, block-vectorised.
        A_u x = (Y^T Y + reg I) x + sum_i (c_ui - 1)(y_i . x) y_i ;  b_u = sum_i c_ui y_i."""
        YtY = Y.T @ Y + self.reg * np.eye(Y.shape[1], dtype=Y.dtype)
        for s in range(0, C.shape[0], block):
            Cb = C[s:s + block]
            if Cb.nnz == 0:
                X[s:s + block] = 0.0
                continue
            rows = np.repeat(np.arange(Cb.shape[0]), np.diff(Cb.indptr))
            cols, cm1 = Cb.indices, Cb.data - 1.0
            Yc = Y[cols]

            def A_op(P: np.ndarray) -> np.ndarray:
                w = cm1 * np.einsum("nf,nf->n", P[rows], Yc)
                return P @ YtY + sp.csr_matrix((w, cols, Cb.indptr), shape=Cb.shape) @ Y

            x = X[s:s + block]
            r = (Cb @ Y) - A_op(x)
            p = r.copy()
            rs = np.einsum("uf,uf->u", r, r)
            for _ in range(self.cg_steps):
                Ap = A_op(p)
                a = rs / np.maximum(np.einsum("uf,uf->u", p, Ap), 1e-20)
                x += a[:, None] * p
                r -= a[:, None] * Ap
                rs_new = np.einsum("uf,uf->u", r, r)
                p = r + (rs_new / np.maximum(rs, 1e-20))[:, None] * p
                rs = rs_new
            X[s:s + block] = x

    def score(self, uidx: np.ndarray) -> np.ndarray:
        return (self.U[uidx] @ self.V.T).astype(np.float32)

    def params(self) -> dict[str, Any]:
        return {"factors": self.factors, "reg": self.reg, "alpha": self.alpha,
                "iterations": self.iterations, "seed": self.seed, "solver": self.solver,
                "cg_steps": self.cg_steps, "confidence": "1 + alpha * rating"}
