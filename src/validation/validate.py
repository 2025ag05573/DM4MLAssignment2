"""Stage 2 - Data validation.

Runs the data contract against the bronze tables:

* type coercion and missing-value checks on the raw strings,
* the Pandera domain schema (ranges, allowed values, formats),
* duplicate and referential-integrity checks across tables,
* cross-field and volume sanity checks.

Every failing row is recorded with the id of the check it failed. Rows with an
*error*-severity failure are quarantined to ``data/errors/<table>_errors.parquet``;
rows with only *warnings* are kept (and fixed in preparation) but logged in
``data/errors/warnings.parquet``. A key-only list of every quarantined row
(table, source line, ids, failed checks) is written to
``reports/validation/quarantined_records.csv`` so the evidence can be committed
(MovieLens' licence does not allow redistributing the data itself). A quality
gate stops the pipeline when the quarantine rate of any table exceeds
``validation.max_error_rate``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from pandera.errors import SchemaErrors

from src.common.config import get_paths, load_config
from src.common.io import df_to_markdown, utc_now, write_json, write_parquet
from src.common.logging_utils import get_logger, relpath
from src.validation.schemas import (CHECK_CATALOG, KEY_COLUMNS, PRIMARY_KEYS, TYPED_COLUMNS,
                                    build_schemas)

log = get_logger("validation")
TABLE_ORDER = ("users", "movies", "ratings")  # parents first, for referential integrity


class DataQualityError(RuntimeError):
    """Raised when the quality gate fails."""


@dataclass
class Failure:
    table: str
    check: str
    column: str
    index: np.ndarray                     # row labels that failed
    examples: list = field(default_factory=list)

    @property
    def severity(self) -> str:
        return CHECK_CATALOG[self.check][0]


def _fail(table: str, check: str, column: str, mask: pd.Series, values: pd.Series | None = None) -> Failure:
    idx = mask[mask.fillna(False)].index.to_numpy()
    ex = [] if values is None else values.loc[idx[:5]].astype(str).tolist()
    return Failure(table, check, column, idx, ex)


# --------------------------------------------------------------------------- per-table checks
def coerce_and_check_missing(table: str, raw: pd.DataFrame) -> tuple[pd.DataFrame, list[Failure]]:
    """Cast raw strings to the contract dtypes; report un-parseable and missing values."""
    typed = raw.copy()
    failures: list[Failure] = []
    for col, dtype in TYPED_COLUMNS[table].items():
        s = raw[col]
        if dtype == "Int64":
            num = pd.to_numeric(s.str.strip(), errors="coerce")
            bad = s.notna() & (num.isna() | (num % 1 != 0))
            failures.append(_fail(table, "bad_dtype", col, bad, s))
            typed[col] = num.where(~bad).astype("Int64")
        else:
            typed[col] = s.astype("string").str.strip()
        missing = s.isna()
        check = "missing_key" if col in KEY_COLUMNS[table] else "missing_attribute"
        failures.append(_fail(table, check, col, missing))
    return typed, failures


def schema_checks(table: str, typed: pd.DataFrame, schema) -> list[Failure]:
    """Run the Pandera contract; return one Failure per declared check (empty if it passed)."""
    declared = [(chk.error, col) for col, c in schema.columns.items() for chk in c.checks]
    found: dict[tuple[str, str], Failure] = {}
    try:
        schema.validate(typed, lazy=True)
    except SchemaErrors as exc:
        fc = exc.failure_cases
        fc = fc[fc["index"].notna() & fc["check"].isin(CHECK_CATALOG.keys())]
        for (check, col), grp in fc.groupby(["check", "column"]):
            idx = grp["index"].astype(int).to_numpy()
            found[(check, col)] = Failure(table, check, col, idx,
                                          grp["failure_case"].astype(str).head(5).tolist())
    return [found.get(k, Failure(table, k[0], k[1], np.array([], dtype=int))) for k in declared]


def duplicate_checks(table: str, typed: pd.DataFrame) -> list[Failure]:
    pk = PRIMARY_KEYS[table]
    has_key = typed[pk].notna().all(axis=1)
    if table == "ratings":
        # Every row involved in a repeated (user, movie) pair; preparation keeps the latest.
        dup = typed.duplicated(pk, keep=False) & has_key
        return [_fail(table, "duplicate_interaction", "+".join(pk), dup)]
    dup = typed.duplicated(pk, keep="first") & has_key
    return [_fail(table, "duplicate_primary_key", pk[0], dup, typed[pk[0]])]


def genre_vocabulary_check(movies: pd.DataFrame, vocab: list[str]) -> Failure:
    tokens = movies["genres"].fillna("").str.split("|")
    bad = tokens.map(lambda ts: any(t and t not in vocab for t in ts))
    return _fail("movies", "genre_vocabulary", "genres", bad, movies["genres"])


def referential_checks(ratings: pd.DataFrame, valid_users: pd.Series, valid_movies: pd.Series) -> list[Failure]:
    ou = ratings["user_id"].notna() & ~ratings["user_id"].isin(valid_users)
    om = ratings["movie_id"].notna() & ~ratings["movie_id"].isin(valid_movies)
    return [_fail("ratings", "orphan_user", "user_id", ou, ratings["user_id"]),
            _fail("ratings", "orphan_movie", "movie_id", om, ratings["movie_id"])]


def rated_before_release_check(ratings: pd.DataFrame, movies: pd.DataFrame) -> Failure:
    year = movies.set_index("movie_id")["title"].str.extract(r"\((\d{4})\)\s*$")[0].astype("Float64")
    rel = ratings["movie_id"].map(year)
    rated_year = pd.to_datetime(ratings["timestamp"].astype("float"), unit="s", utc=True).dt.year
    bad = rel.notna() & (rated_year < rel - 1)   # 1-year grace for festival / preview screenings
    return _fail("ratings", "rated_before_release", "timestamp", bad, ratings["timestamp"])


def volume_check(table: str, n_rows: int, expected: list[int]) -> dict:
    lo, hi = expected
    return {"table": table, "check": "row_volume", "column": "*", "severity": "warning",
            "description": CHECK_CATALOG["row_volume"][1], "failed_rows": 0,
            "status": "PASS" if lo <= n_rows <= hi else "WARN",
            "examples": f"{n_rows} rows (expected {lo}-{hi})"}


# --------------------------------------------------------------------------- orchestration
def validate_tables(raw: dict[str, pd.DataFrame], cfg: dict[str, Any],
                    now_ts: int | None = None) -> dict[str, Any]:
    """Pure function (no I/O): returns validated tables, error rows and check results."""
    schemas = build_schemas(cfg, now_ts)
    typed: dict[str, pd.DataFrame] = {}
    failures: dict[str, list[Failure]] = {t: [] for t in TABLE_ORDER}

    for t in TABLE_ORDER:
        df = raw[t].reset_index(drop=True)
        typed[t], f = coerce_and_check_missing(t, df)
        failures[t] += f
        failures[t] += schema_checks(t, typed[t], schemas[t])
        failures[t] += duplicate_checks(t, typed[t])
    failures["movies"].append(genre_vocabulary_check(typed["movies"], cfg["validation"]["genres"]))

    def error_mask(t: str) -> np.ndarray:
        m = np.zeros(len(typed[t]), dtype=bool)
        for f in failures[t]:
            if f.severity == "error" and len(f.index):
                m[f.index] = True
        return m

    # Referential integrity is checked against the *valid* parent rows only.
    valid_users = typed["users"].loc[~error_mask("users"), "user_id"]
    valid_movies = typed["movies"].loc[~error_mask("movies"), "movie_id"]
    failures["ratings"] += referential_checks(typed["ratings"], valid_users, valid_movies)
    failures["ratings"].append(rated_before_release_check(typed["ratings"], typed["movies"]))

    results, validated, errors, warnings = [], {}, {}, []
    summary = {}
    for t in TABLE_ORDER:
        n = len(typed[t])
        err = error_mask(t)
        reasons = pd.Series([[] for _ in range(n)], dtype=object)
        for f in failures[t]:
            if len(f.index):
                if f.severity == "error":
                    for i in f.index:
                        reasons.iat[i].append(f"{f.check}:{f.column}")
                else:
                    warnings.append(pd.DataFrame({"table": t, "check": f.check, "column": f.column,
                                                  "_source_line": raw[t]["_source_line"].to_numpy()[f.index]}))
            results.append({"table": t, "check": f.check, "column": f.column, "severity": f.severity,
                            "description": CHECK_CATALOG[f.check][1], "failed_rows": int(len(f.index)),
                            "status": "PASS" if len(f.index) == 0 else
                            ("FAIL" if f.severity == "error" else "WARN"),
                            "examples": ", ".join(f.examples[:3])})
        results.append(volume_check(t, n, cfg["validation"]["expected_rows"][t]))
        validated[t] = typed[t].loc[~err].reset_index(drop=True)
        bad = raw[t].reset_index(drop=True).loc[err].copy()
        bad["_error_checks"] = reasons[err].map(";".join).to_numpy()
        errors[t] = bad
        summary[t] = {"rows_in": n, "rows_valid": int((~err).sum()), "rows_quarantined": int(err.sum()),
                      "error_rate": float(err.mean()) if n else 0.0,
                      "rows_with_warnings": int(sum(len(f.index) for f in failures[t] if f.severity == "warning"))}

    warn_df = (pd.concat(warnings, ignore_index=True) if warnings
               else pd.DataFrame(columns=["table", "check", "column", "_source_line"]))
    return {"validated": validated, "errors": errors, "warnings": warn_df,
            "results": pd.DataFrame(results), "summary": summary}


EVIDENCE_IDS = {"users": ["user_id"], "movies": ["movie_id"], "ratings": ["user_id", "movie_id"]}


def quarantine_evidence(out: dict[str, Any]) -> pd.DataFrame:
    """One row per quarantined source row: table, line, ids and the checks it failed."""
    cols = ["table", "source_line", "user_id", "movie_id", "failed_checks"]
    parts = []
    for t, keys in EVIDENCE_IDS.items():
        e = out["errors"][t]
        if len(e):
            parts.append(pd.DataFrame({"table": t, "source_line": e["_source_line"].to_numpy(),
                                       **{k: e[k].astype("string").to_numpy() for k in keys},
                                       "failed_checks": e["_error_checks"].to_numpy()}))
    return pd.concat(parts, ignore_index=True).reindex(columns=cols) if parts else pd.DataFrame(columns=cols)


def render_report(out: dict[str, Any], cfg: dict[str, Any], manifest_src: str) -> str:
    gate = cfg["validation"]["max_error_rate"]
    summ = pd.DataFrame([{"table": t, **s} for t, s in out["summary"].items()])
    summ["gate"] = np.where(summ["error_rate"] <= gate, "PASS", "FAIL")
    res = out["results"].copy()
    lines = [
        "# Data Validation Report", "",
        f"- Generated: {utc_now()}",
        f"- Source: `{manifest_src}`",
        f"- Quality gate: quarantine rate per table must be <= {gate:.2%}",
        f"- Checks run: {len(res)} ({(res.status == 'PASS').sum()} pass, "
        f"{(res.status == 'WARN').sum()} warn, {(res.status == 'FAIL').sum()} fail)", "",
        "## Summary by table", "", df_to_markdown(summ, ".4%"), "",
        "## Check results", "",
        "Severity **error** = row quarantined to `data/errors/`; **warning** = row kept, "
        "logged in `data/errors/warnings.parquet`, and repaired in the preparation stage. "
        "Every quarantined row is also listed (table, source line, ids, failed checks) in "
        "`reports/validation/quarantined_records.csv`.", "",
        df_to_markdown(res[["table", "check", "column", "severity", "failed_rows", "status",
                            "description", "examples"]]), ""]
    return "\n".join(lines)


def run(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = cfg or load_config()
    paths = get_paths(cfg)
    raw = {t: pd.read_parquet(paths.raw / "bronze" / f"{t}.parquet") for t in TABLE_ORDER}
    out = validate_tables(raw, cfg)

    for t in TABLE_ORDER:
        write_parquet(out["validated"][t], paths.validated / f"{t}.parquet")
        write_parquet(out["errors"][t], paths.errors / f"{t}_errors.parquet")
        out["errors"][t].to_csv(paths.errors / f"{t}_errors.csv", index=False)
    write_parquet(out["warnings"], paths.errors / "warnings.parquet")
    out["results"].to_csv(paths.reports / "validation" / "check_results.csv", index=False)
    quarantine_evidence(out).to_csv(paths.reports / "validation" / "quarantined_records.csv", index=False)

    src = "unknown"
    manifest = paths.reports / "ingestion_manifest.json"
    if manifest.exists():
        import json
        src = json.loads(manifest.read_text())["source"]
    report = render_report(out, cfg, src)
    (paths.reports / "validation" / "validation_report.md").write_text(report, encoding="utf-8")
    write_json({"summary": out["summary"], "checks": out["results"].to_dict("records")},
               paths.reports / "validation" / "validation_report.json")

    for t, s in out["summary"].items():
        log.info("%-7s in=%-8d valid=%-8d quarantined=%-5d warnings=%d", t, s["rows_in"],
                 s["rows_valid"], s["rows_quarantined"], s["rows_with_warnings"])
    for r in out["results"].query("status != 'PASS'").itertuples():
        log.warning("%s.%s [%s] %s rows: %s", r.table, r.check, r.severity, r.failed_rows, r.examples)
    log.info("Report -> %s", relpath(paths.reports / "validation" / "validation_report.md"))

    enforce_gate(out["summary"], cfg["validation"]["max_error_rate"])
    return out["summary"]


def enforce_gate(summary: dict[str, dict], max_error_rate: float) -> None:
    breached = {t: s["error_rate"] for t, s in summary.items() if s["error_rate"] > max_error_rate}
    if breached:
        raise DataQualityError(f"Quality gate failed (max_error_rate={max_error_rate}): {breached}")


if __name__ == "__main__":
    run()
