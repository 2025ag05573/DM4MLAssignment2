import numpy as np
import pandas as pd

from src.preparation.prepare import prepare_movies, prepare_ratings, prepare_users, standardise_title
from src.preparation.split import global_time_split


def test_title_standardisation():
    assert standardise_title("Matrix, The (1999)") == ("The Matrix", 1999)
    assert standardise_title("Apple, The (Sib) (1998)") == ("The Apple (Sib)", 1998)
    assert standardise_title("Toy Story (1995)") == ("Toy Story", 1995)
    assert standardise_title("No Year") == ("No Year", None)


def test_users_standardised_and_imputed(cfg):
    u = pd.DataFrame({"user_id": pd.array([1, 2, 3], dtype="Int64"),
                      "gender": pd.array(["F", None, "X"], dtype="string"),
                      "age": pd.array([25, 99, None], dtype="Int64"),
                      "occupation": pd.array([12, 4, None], dtype="Int64"),
                      "zip_code": pd.array(["94110-1234", "ABC", None], dtype="string")})
    out, stats = prepare_users(u, cfg)
    assert out["zip5"].tolist() == ["94110", "unknown", "unknown"]
    assert out["gender"].tolist() == ["F", "unknown", "unknown"]
    assert out["age_group"].tolist() == ["25-34", "unknown", "unknown"]
    assert out["occupation"].iloc[0] == "programmer"
    assert stats["zip_plus4_truncated"] == 1


def test_movies_genres_one_hot(cfg):
    m = pd.DataFrame({"movie_id": pd.array([1, 2], dtype="Int64"),
                      "title": pd.array(["Heat (1995)", "Odd (2000)"], dtype="string"),
                      "genres": pd.array(["Action|Crime", None], dtype="string")})
    out, _ = prepare_movies(m, cfg)
    assert out.loc[0, "genre_action"] == 1 and out.loc[0, "genre_crime"] == 1
    assert out.loc[1, "genres"] == "(no genres listed)" and out.loc[1, "n_genres"] == 0


def test_duplicate_ratings_keep_latest(cfg):
    r = pd.DataFrame({"user_id": pd.array([1, 1], dtype="Int64"), "movie_id": pd.array([5, 5], dtype="Int64"),
                      "rating": pd.array([2, 4], dtype="Int64"), "timestamp": pd.array([100, 200], dtype="Int64"),
                      "_source_line": [1, 2]})
    out, stats = prepare_ratings(r, cfg)
    assert len(out) == 1 and out["rating"].iloc[0] == 4 and stats["duplicates_removed"] == 1


def test_global_time_split_has_no_leakage():
    rng = np.random.default_rng(0)
    r = pd.DataFrame({"user_id": rng.integers(1, 50, 1000), "movie_id": rng.integers(1, 80, 1000),
                      "rating": rng.integers(1, 6, 1000), "timestamp": rng.integers(0, 300, 1000)})
    parts, cut = global_time_split(r, 0.1, 0.1)
    assert parts["train"]["timestamp"].max() < cut["t_val"] <= parts["val"]["timestamp"].min()
    assert parts["val"]["timestamp"].max() < cut["t_test"] <= parts["test"]["timestamp"].min()
    assert sum(len(p) for p in parts.values()) == len(r)
    assert abs(len(parts["test"]) / len(r) - 0.1) < 0.03       # ties may shift the cut slightly
