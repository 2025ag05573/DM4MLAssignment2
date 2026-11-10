import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from src.evaluation.metrics import rmse
from src.training.knn import ItemKNNRecommender
from src.training.mf import ExplicitMF, ImplicitALS
from src.transformation.transform import item_popularity, user_activity_summary

RATINGS = pd.DataFrame({"user_id": [1, 1, 2, 2, 3], "movie_id": [10, 20, 10, 30, 10],
                        "rating": [5, 3, 4, 2, 5], "timestamp": [100, 200, 150, 400, 500]})


def test_features_are_point_in_time():
    """A feature computed as of t must ignore everything at or after t."""
    early = user_activity_summary(RATINGS, as_of=300).set_index("user_id")
    assert early.loc[2, "n_ratings"] == 1            # the t=400 rating is in the future
    assert 3 not in early.index                      # user 3 has not appeared yet
    late = user_activity_summary(RATINGS, as_of=1000).set_index("user_id")
    assert late.loc[2, "n_ratings"] == 2


def test_item_popularity_and_bayesian_average():
    pop = item_popularity(RATINGS, as_of=1000, prior_m=2, trend_days=0).set_index("movie_id")
    assert pop.loc[10, "n_ratings"] == 3 and pop.loc[10, "popularity_rank"] == 1
    g = RATINGS["rating"].mean()
    assert pop.loc[30, "bayesian_avg"] == pytest.approx((2 + 2 * g) / (1 + 2))


class _Ctx:
    """Minimal stand-in for FitContext (models only touch .X)."""
    def __init__(self, X):
        self.X = X


def _low_rank_matrix(seed=0, n_users=60, n_items=40, density=0.3):
    rng = np.random.default_rng(seed)
    U, V = rng.normal(size=(n_users, 3)), rng.normal(size=(n_items, 3))
    R = np.clip(np.rint(3 + U @ V.T), 1, 5)
    mask = rng.random(R.shape) < density
    return sp.csr_matrix(np.where(mask, R, 0.0))


def test_explicit_mf_fits_training_data():
    X = _low_rank_matrix()
    m = ExplicitMF(factors=4, reg=0.05, iterations=10).fit(_Ctx(X))
    assert m.history[-1] < m.history[0]
    u, i = X.nonzero()
    assert rmse(np.asarray(X[u, i]).ravel(), m.predict_ratings(u, i)) < 0.6


def test_explicit_mf_regularisation_modes():
    X = _low_rank_matrix()
    weighted = ExplicitMF(factors=4, reg=5.0, reg_mode="weighted", iterations=10).fit(_Ctx(X))
    plain = ExplicitMF(factors=4, reg=0.05, reg_mode="unweighted", iterations=10).fit(_Ctx(X))
    # heavy weighted-lambda shrinkage collapses the model to biases; light ridge keeps the factors
    assert weighted.diagnostics()["factor_term_std"] < 0.05 < plain.diagnostics()["factor_term_std"]
    with pytest.raises(ValueError):
        ExplicitMF(reg_mode="other")


def test_cg_solver_matches_exact_solver():
    X = _low_rank_matrix(seed=1)
    exact = ImplicitALS(factors=4, reg=1.0, alpha=2.0, iterations=8, solver="exact").fit(_Ctx(X))
    cg = ImplicitALS(factors=4, reg=1.0, alpha=2.0, iterations=8, solver="cg", cg_steps=4).fit(_Ctx(X))
    s_exact, s_cg = exact.score(np.arange(10)), cg.score(np.arange(10))
    corr = np.mean([np.corrcoef(a, b)[0, 1] for a, b in zip(s_exact, s_cg)])
    assert corr > 0.95


def test_item_knn_recommends_co_rated_items():
    # users 0 and 1 rated movies 0 and 1 together; user 2 rated only movie 0 -> movie 1 should rank first
    X = sp.csr_matrix(np.array([[5, 4, 0, 0], [4, 5, 0, 0], [5, 0, 0, 0], [0, 0, 3, 3]], dtype=float))
    m = ItemKNNRecommender(shrink=0, neighbours=2).fit(_Ctx(X))
    s = m.score(np.array([2]))[0]
    assert np.argmax(s[1:]) + 1 == 1 and s[2] == 0 and s[3] == 0
