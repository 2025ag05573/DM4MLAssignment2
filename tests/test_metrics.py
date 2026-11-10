import numpy as np
import pytest

from src.evaluation import metrics as M


def test_precision_recall_hit():
    recs, rel = np.array([1, 2, 3, 4, 5]), {2, 5, 9}
    assert M.precision_at_k(recs, rel, 5) == pytest.approx(2 / 5)
    assert M.recall_at_k(recs, rel, 5) == pytest.approx(2 / 3)
    assert M.hit_rate_at_k(recs, rel, 1) == 0.0
    assert M.hit_rate_at_k(recs, rel, 2) == 1.0


def test_ndcg_hand_computed():
    recs, rel = np.array([7, 2, 8, 5]), {2, 5}
    dcg = 1 / np.log2(3) + 1 / np.log2(5)       # hits at ranks 2 and 4
    idcg = 1 / np.log2(2) + 1 / np.log2(3)      # ideal: hits at ranks 1 and 2
    assert M.ndcg_at_k(recs, rel, 4) == pytest.approx(dcg / idcg)
    assert M.ndcg_at_k(np.array([2, 5, 1]), rel, 3) == pytest.approx(1.0)


def test_coverage_and_diversity():
    recs = np.array([[0, 1], [0, 2]])
    assert M.catalog_coverage(recs, 10) == pytest.approx(0.3)
    V = np.eye(3)                                 # three orthogonal "genres"
    assert M.intra_list_diversity(recs, V) == pytest.approx(1.0)
    same = np.array([[0, 0]])
    assert M.intra_list_diversity(same, V) == pytest.approx(0.0)


def test_novelty_and_rating_errors():
    assert M.novelty(np.array([[0]]), np.array([0.5])) == pytest.approx(1.0)
    assert M.rmse([1, 3], [2, 5]) == pytest.approx(np.sqrt((1 + 4) / 2))
    assert M.mae([1, 3], [2, 5]) == pytest.approx(1.5)
