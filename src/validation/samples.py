"""Small hand-made tables in the *bronze* format (all strings).

Used by the unit tests and by ``scripts/validation_demo.py``; they live in the
package so the demo script does not depend on the test suite."""
from __future__ import annotations

import pandas as pd

NOW = 1_100_000_000  # 2004-11-09: fixed "now" so the future-timestamp check is deterministic
USER_COLS = ["user_id", "gender", "age", "occupation", "zip_code"]
MOVIE_COLS = ["movie_id", "title", "genres"]
RATING_COLS = ["user_id", "movie_id", "rating", "timestamp"]


def bronze(rows: list[list], columns: list[str]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=columns, dtype="string")
    df.insert(0, "_source_line", range(1, len(df) + 1))
    return df


def clean_sample() -> dict[str, pd.DataFrame]:
    return {
        "users": bronze([["1", "F", "25", "4", "10001"], ["2", "M", "35", "12", "94110-1234"],
                         ["3", "M", "18", "4", "02139"]], USER_COLS),
        "movies": bronze([["10", "Matrix, The (1999)", "Action|Sci-Fi"], ["20", "Toy Story (1995)", "Animation|Comedy"],
                          ["30", "Heat (1995)", "Action|Crime|Thriller"]], MOVIE_COLS),
        "ratings": bronze([["1", "10", "5", "978300760"], ["1", "20", "3", "978300800"],
                           ["2", "10", "4", "978301000"], ["3", "30", "2", "978302000"]], RATING_COLS),
    }


def dirty_sample() -> dict[str, pd.DataFrame]:
    """Every class of defect the validation stage is meant to catch, one per row."""
    c = clean_sample()
    users = pd.concat([c["users"], bronze(
        [["2", "M", "35", "12", "94110"],      # duplicate primary key
         ["4", "X", "99", "33", "ABCDE"],      # bad gender / age code / occupation / zip -> warnings
         ["five", "F", "25", "4", "10001"],    # non-numeric id -> bad_dtype
         [None, "F", "25", "4", "10001"]],     # missing key
        USER_COLS)], ignore_index=True)
    movies = pd.concat([c["movies"], bronze(
        [["40", "No Year Movie", "Drama|Spaghetti"]],   # no (YYYY) + unknown genre -> warnings
        MOVIE_COLS)], ignore_index=True)
    ratings = pd.concat([c["ratings"], bronze(
        [["1", "30", "7", "978300900"],        # rating out of scale
         ["2", "20", "4", "4102444800"],       # timestamp in the future (2100)
         ["3", "20", "4", "800000000"],        # timestamp before MovieLens existed (1995)
         ["999", "10", "4", "978300900"],      # orphan user
         ["1", "777", "4", "978300900"],       # orphan movie
         ["2", "30", None, "978300900"],       # missing rating
         ["3", "10", "4.5", "978300900"],      # non-integer rating -> bad_dtype
         ["1", "10", "4", "978309999"]],       # repeat of (1, 10) -> duplicate_interaction warning
        RATING_COLS)], ignore_index=True)
    for df in (users, movies, ratings):
        df["_source_line"] = range(1, len(df) + 1)
    return {"users": users, "movies": movies, "ratings": ratings}
