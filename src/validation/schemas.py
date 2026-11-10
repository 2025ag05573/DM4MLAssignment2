"""Declarative data contracts (Pandera) for the three MovieLens tables.

Each check carries a stable ``error=`` id so the validation report and the
error dataset can refer to it by name. Missing values and type coercion are
checked separately (in ``validate.py``) because they must be evaluated on the
raw string data, before the typed schema can be applied.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd
import pandera.pandas as pa

# column -> pandas dtype the validated table must have
TYPED_COLUMNS: dict[str, dict[str, str]] = {
    "users": {"user_id": "Int64", "gender": "string", "age": "Int64", "occupation": "Int64",
              "zip_code": "string"},
    "movies": {"movie_id": "Int64", "title": "string", "genres": "string"},
    "ratings": {"user_id": "Int64", "movie_id": "Int64", "rating": "Int64", "timestamp": "Int64"},
}

# Columns that identify a record: a missing value here makes the row unusable.
KEY_COLUMNS: dict[str, list[str]] = {
    "users": ["user_id"],
    "movies": ["movie_id"],
    "ratings": ["user_id", "movie_id", "rating", "timestamp"],
}

PRIMARY_KEYS: dict[str, list[str]] = {
    "users": ["user_id"],
    "movies": ["movie_id"],
    "ratings": ["user_id", "movie_id"],
}

# check id -> (severity, human description). "error" rows are quarantined,
# "warning" rows are kept and fixed downstream in preparation.
CHECK_CATALOG: dict[str, tuple[str, str]] = {
    "missing_key": ("error", "Identifier / rating / timestamp is null"),
    "missing_attribute": ("warning", "Non-key attribute is null (imputed in preparation)"),
    "bad_dtype": ("error", "Value cannot be parsed to the contract type"),
    "id_positive": ("error", "Identifier must be a positive integer"),
    "rating_in_scale": ("error", "Rating must be one of the allowed values 1-5"),
    "timestamp_in_range": ("error", "Timestamp must be after MovieLens launch and not in the future"),
    "gender_valid": ("warning", "Gender must be M or F (else set to 'unknown')"),
    "age_code_valid": ("warning", "Age must be one of the documented age-bucket codes"),
    "occupation_valid": ("warning", "Occupation code must be within 0-20"),
    "duplicate_primary_key": ("error", "Primary key appears more than once (later copies quarantined)"),
    "duplicate_interaction": ("warning", "Same (user, movie) rated more than once (latest kept)"),
    "orphan_user": ("error", "Rating references a user_id that is not in users"),
    "orphan_movie": ("error", "Rating references a movie_id that is not in movies"),
    "zip_format": ("warning", "ZIP code is not 5 digits (ZIP+4 truncated, others set unknown)"),
    "title_has_year": ("warning", "Title should end with the release year '(YYYY)'"),
    "genre_vocabulary": ("warning", "Genre token outside the documented 18-genre vocabulary"),
    "rated_before_release": ("warning", "Rating timestamp precedes the movie's release year"),
    "row_volume": ("warning", "Table row count outside the expected range"),
}


def _ts(s: str) -> int:
    return int(datetime.fromisoformat(s).replace(tzinfo=timezone.utc).timestamp())


def build_schemas(cfg: dict[str, Any], now_ts: int | None = None) -> dict[str, pa.DataFrameSchema]:
    """Typed domain contracts. Nullable=True: nulls are reported by the missing-value checks."""
    v = cfg["validation"]
    now_ts = now_ts or int(pd.Timestamp.now(tz="UTC").timestamp())
    lo, hi = v["occupation_range"]
    pos = lambda name: pa.Check.gt(0, error="id_positive")  # noqa: E731

    users = pa.DataFrameSchema({
        "user_id": pa.Column("Int64", pos("user_id"), nullable=True),
        "gender": pa.Column("string", pa.Check.isin(v["genders"], error="gender_valid"), nullable=True),
        "age": pa.Column("Int64", pa.Check.isin(v["age_codes"], error="age_code_valid"), nullable=True),
        "occupation": pa.Column("Int64", pa.Check.in_range(lo, hi, error="occupation_valid"),
                                nullable=True),
        "zip_code": pa.Column("string", pa.Check.str_matches(r"^\d{5}$", error="zip_format"),
                              nullable=True),
    }, strict=False, name="users")

    movies = pa.DataFrameSchema({
        "movie_id": pa.Column("Int64", pos("movie_id"), nullable=True),
        "title": pa.Column("string", pa.Check.str_matches(r".*\(\d{4}\)\s*$", error="title_has_year"),
                           nullable=True),
        "genres": pa.Column("string", nullable=True),
    }, strict=False, name="movies")

    ratings = pa.DataFrameSchema({
        "user_id": pa.Column("Int64", pos("user_id"), nullable=True),
        "movie_id": pa.Column("Int64", pos("movie_id"), nullable=True),
        "rating": pa.Column("Int64", pa.Check.isin(v["rating_values"], error="rating_in_scale"),
                            nullable=True),
        "timestamp": pa.Column("Int64", pa.Check.in_range(_ts(v["min_timestamp"]), now_ts,
                                                          error="timestamp_in_range"), nullable=True),
    }, strict=False, name="ratings")
    return {"users": users, "movies": movies, "ratings": ratings}
