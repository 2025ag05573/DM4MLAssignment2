"""Stage 3a - Data preparation (cleaning + standardisation).

Input  : data/processed/validated/*.parquet  (contract-conforming rows)
Output : data/processed/clean/*.parquet       (analysis-ready, standard formats)
         reports/preparation_summary.json

What happens here
-----------------
* **Duplicates** - repeated (user, movie) ratings keep the latest one.
* **Missing values** - categorical attributes become ``"unknown"``; movies
  without genres get ``"(no genres listed)"``; nothing is silently dropped.
* **Standard formats** - epoch seconds -> UTC datetime, coded columns
  (age bucket, occupation) -> readable labels, ZIP+4 -> 5-digit ZIP,
  ``"Matrix, The (1999)"`` -> title ``"The Matrix"`` + ``release_year=1999``,
  genres -> canonical pipe list + one-hot columns.
"""
from __future__ import annotations

import re
from typing import Any

import pandas as pd

from src.common.config import get_paths, load_config
from src.common.io import write_json, write_parquet
from src.common.logging_utils import get_logger

log = get_logger("preparation")

# Code books from the MovieLens 1M README
AGE_GROUPS = {1: "Under 18", 18: "18-24", 25: "25-34", 35: "35-44", 45: "45-49", 50: "50-55", 56: "56+"}
OCCUPATIONS = {
    0: "other", 1: "academic/educator", 2: "artist", 3: "clerical/admin", 4: "college/grad student",
    5: "customer service", 6: "doctor/health care", 7: "executive/managerial", 8: "farmer",
    9: "homemaker", 10: "K-12 student", 11: "lawyer", 12: "programmer", 13: "retired",
    14: "sales/marketing", 15: "scientist", 16: "self-employed", 17: "technician/engineer",
    18: "tradesman/craftsman", 19: "unemployed", 20: "writer",
}
UNKNOWN = "unknown"
NO_GENRE = "(no genres listed)"
_ARTICLE = re.compile(r"^(?P<body>.*?), (?P<art>The|A|An|Les|La|Le|L'|Il|Das|Die|Der|El)"
                      r"(?P<alt>\s+\(.*\))?$")
_YEAR = re.compile(r"^(?P<title>.*?)\s*\((?P<year>\d{4})\)\s*$")


def genre_column(genre: str) -> str:
    return "genre_" + re.sub(r"[^a-z0-9]+", "_", genre.lower()).strip("_")


def standardise_title(raw: str | None) -> tuple[str | None, int | None]:
    if raw is None or pd.isna(raw):
        return None, None
    m = _YEAR.match(raw)
    title, year = (m["title"], int(m["year"])) if m else (raw.strip(), None)
    a = _ARTICLE.match(title)
    if a:  # "Matrix, The" -> "The Matrix"; "Apple, The (Sib)" -> "The Apple (Sib)"
        sep = "" if a["art"].endswith("'") else " "
        title = f"{a['art']}{sep}{a['body']}{a['alt'] or ''}"
    return title, year


def prepare_users(users: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, dict]:
    u = users.drop_duplicates("user_id", keep="first").copy()
    stats = {"missing_imputed": {}}
    gender = u["gender"].where(u["gender"].isin(cfg["validation"]["genders"]))
    zip_raw = u["zip_code"].astype("string")
    zip5 = zip_raw.str.extract(r"^(\d{5})(?:-\d{4})?$")[0]
    stats["zip_plus4_truncated"] = int(zip_raw.str.match(r"^\d{5}-\d{4}$").fillna(False).sum())
    stats["zip_invalid_set_unknown"] = int((zip5.isna() & zip_raw.notna()).sum())
    out = pd.DataFrame({
        "user_id": u["user_id"].astype("int32"),
        "gender": gender.fillna(UNKNOWN).astype("string"),
        "age_code": u["age"].astype("Int16"),
        "age_group": u["age"].map(AGE_GROUPS).fillna(UNKNOWN).astype("string"),
        "occupation_code": u["occupation"].astype("Int16"),
        "occupation": u["occupation"].map(OCCUPATIONS).fillna(UNKNOWN).astype("string"),
        "zip5": zip5.fillna(UNKNOWN).astype("string"),
        "zip_region": zip5.str[0].fillna(UNKNOWN).astype("string"),  # first ZIP digit = US region
    })
    for col in ("gender", "age_group", "occupation", "zip5"):
        stats["missing_imputed"][col] = int((out[col] == UNKNOWN).sum())
    return out.sort_values("user_id").reset_index(drop=True), stats


def prepare_movies(movies: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, dict]:
    vocab = cfg["validation"]["genres"]
    m = movies.drop_duplicates("movie_id", keep="first").copy()
    parsed = m["title"].map(standardise_title)
    genres = (m["genres"].fillna("").str.split("|")
              .map(lambda ts: [t for t in ts if t in vocab]))
    out = pd.DataFrame({
        "movie_id": m["movie_id"].astype("int32"),
        "title": parsed.map(lambda x: x[0]).fillna(UNKNOWN).astype("string"),
        "release_year": parsed.map(lambda x: x[1]).astype("Int16"),
        "genres": genres.map(lambda g: "|".join(g) if g else NO_GENRE).astype("string"),
        "n_genres": genres.map(len).astype("int8"),
    })
    for g in vocab:
        out[genre_column(g)] = genres.map(lambda lst, g=g: g in lst).astype("int8")
    stats = {"titles_reordered_article": int(m["title"].str.contains(r", (?:The|A|An) \(", regex=True).sum()),
             "movies_without_genre": int((out["n_genres"] == 0).sum()),
             "missing_release_year": int(out["release_year"].isna().sum())}
    return out.sort_values("movie_id").reset_index(drop=True), stats


def prepare_ratings(ratings: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, dict]:
    n0 = len(ratings)
    r = (ratings.sort_values(["timestamp", "_source_line"])
         .drop_duplicates(["user_id", "movie_id"], keep="last"))  # keep_latest policy
    out = pd.DataFrame({
        "user_id": r["user_id"].astype("int32"),
        "movie_id": r["movie_id"].astype("int32"),
        "rating": r["rating"].astype("int8"),
        "timestamp": r["timestamp"].astype("int64"),
        "rated_at": pd.to_datetime(r["timestamp"].astype("int64"), unit="s", utc=True),
    })
    out = out.sort_values(["timestamp", "user_id", "movie_id"]).reset_index(drop=True)
    return out, {"duplicates_removed": int(n0 - len(out))}


def run(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = cfg or load_config()
    paths = get_paths(cfg)
    v = {t: pd.read_parquet(paths.validated / f"{t}.parquet") for t in ("users", "movies", "ratings")}
    users, su = prepare_users(v["users"], cfg)
    movies, sm = prepare_movies(v["movies"], cfg)
    ratings, sr = prepare_ratings(v["ratings"], cfg)
    for name, df in (("users", users), ("movies", movies), ("ratings", ratings)):
        write_parquet(df, paths.clean / f"{name}.parquet")
    summary = {"users": {"rows": len(users), **su}, "movies": {"rows": len(movies), **sm},
               "ratings": {"rows": len(ratings), **sr}}
    write_json(summary, paths.reports / "preparation_summary.json")
    log.info("users=%d movies=%d ratings=%d | dup ratings removed=%d | zip+4 truncated=%d | "
             "invalid zip->unknown=%d | titles with article fixed=%d", len(users), len(movies),
             len(ratings), sr["duplicates_removed"], su["zip_plus4_truncated"],
             su["zip_invalid_set_unknown"], sm["titles_reordered_article"])
    return summary


if __name__ == "__main__":
    run()
