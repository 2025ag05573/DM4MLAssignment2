"""Stage 1 - Data ingestion.

Fetches the MovieLens 1M archive (or uses a local copy), records its checksum,
checks the md5 of each extracted source file against the pinned values,
parses the three ``::``-delimited source files and lands them unchanged as
*bronze* Parquet tables (every column kept as a string, so type problems are
detected by validation instead of being silently coerced at read time).

Outputs
-------
data/raw/ml-1m.zip                         original archive
data/raw/ml-1m/{users,movies,ratings}.dat  extracted source files
data/raw/bronze/{users,movies,ratings}.parquet
data/errors/malformed_lines.parquet        lines with the wrong field count
reports/ingestion_manifest.json            lineage: source, checksums, row counts
"""
from __future__ import annotations

import os
import shutil
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd

from src.common.config import get_paths, load_config
from src.common.io import md5_file, utc_now, write_json, write_parquet
from src.common.logging_utils import get_logger, relpath

log = get_logger("ingestion")

# Source layout documented in the MovieLens 1M README.
SOURCE_SCHEMAS: dict[str, list[str]] = {
    "users": ["user_id", "gender", "age", "occupation", "zip_code"],
    "movies": ["movie_id", "title", "genres"],
    "ratings": ["user_id", "movie_id", "rating", "timestamp"],
}


def _download(url: str, dest: Path, timeout: int, retries: int) -> None:
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            log.info("Downloading %s (attempt %d/%d)", url, attempt, retries)
            tmp = dest.with_suffix(".part")
            with urllib.request.urlopen(url, timeout=timeout) as resp, open(tmp, "wb") as out:
                shutil.copyfileobj(resp, out)
            tmp.replace(dest)
            return
        except Exception as exc:  # network errors are retried with backoff
            last_err = exc
            log.warning("Download failed: %s", exc)
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Could not download {url}: {last_err}")


def acquire_archive(cfg: dict[str, Any]) -> tuple[Path, str]:
    """Return the path of the ml-1m archive inside data/raw and how it was obtained."""
    paths = get_paths(cfg)
    ing = cfg["ingestion"]
    target = paths.raw / "ml-1m.zip"
    local = os.environ.get("ML1M_ZIP") or ing.get("local_zip")
    if local:
        src = Path(local).expanduser().resolve()
        if not src.exists():
            raise FileNotFoundError(f"local_zip does not exist: {src}")
        if src != target.resolve():
            shutil.copyfile(src, target)
        return target, f"local:{src.name}"
    if target.exists():
        log.info("Archive already present at %s - skipping download", relpath(target))
        return target, f"cache:{ing['source_url']}"
    _download(ing["source_url"], target, ing["download_timeout_s"], ing["download_retries"])
    return target, ing["source_url"]


def check_content(file_md5: dict[str, str], ing: dict[str, Any]) -> dict[str, Any]:
    """Compare each source file's md5 with the pinned one (``expected_file_md5``).

    The pins are on the extracted ``.dat`` files rather than the zip, so a re-packed
    archive with the same content still matches. ``checksum_policy: fail`` stops the run
    on a mismatch; ``warn`` logs it and carries on."""
    expected = ing.get("expected_file_md5") or {}
    policy = ing.get("checksum_policy", "fail")
    status = {t: ("not pinned" if not expected.get(t) else "match" if expected[t] == md5 else "MISMATCH")
              for t, md5 in file_md5.items()}
    bad = {t: (expected[t], file_md5[t]) for t, st in status.items() if st == "MISMATCH"}
    if bad:
        msg = "Source file checksum mismatch: " + ", ".join(f"{t} expected {e} got {g}" for t, (e, g) in bad.items())
        if policy == "fail":
            raise ValueError(msg + " (checksum_policy: fail)")
        log.warning("%s - continuing because checksum_policy is %r", msg, policy)
    return {"status": status, "policy": policy}


def parse_dat(path: Path, columns: list[str], encoding: str) -> tuple[pd.DataFrame, list[dict]]:
    """Parse a ``::``-delimited file as strings; collect lines with the wrong arity."""
    bad: list[dict] = []

    def on_bad(fields: list[str]):
        bad.append({"file": path.name, "raw_fields": "::".join(fields), "n_fields": len(fields),
                    "expected_fields": len(columns)})
        return None  # drop the line from the bronze table, keep it in the error set

    df = pd.read_csv(path, sep="::", engine="python", header=None, names=columns, dtype=str,
                     encoding=encoding, keep_default_na=False, on_bad_lines=on_bad)
    # Empty strings are the only "missing" marker in the source format.
    df = df.replace({"": pd.NA})
    df.insert(0, "_source_line", range(1, len(df) + 1))
    return df, bad


def run(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = cfg or load_config()
    paths = get_paths(cfg)
    ing = cfg["ingestion"]

    archive, source = acquire_archive(cfg)
    archive_md5 = md5_file(archive)
    log.info("Archive %s md5=%s (source=%s)", relpath(archive), archive_md5, source)

    extract_dir = paths.raw / "ml-1m"
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(paths.raw)
    file_md5 = {t: md5_file(extract_dir / f"{t}.dat") for t in SOURCE_SCHEMAS}
    checksum = check_content(file_md5, ing)
    log.info("Source file md5: %s (%s)", file_md5, ", ".join(f"{t} {s}" for t, s in checksum["status"].items()))
    bronze_dir = paths.raw / "bronze"
    ingested_at = utc_now()

    manifest: dict[str, Any] = {"source": source, "archive": relpath(archive),
                                "archive_md5": archive_md5, "content_checksum": checksum,
                                "ingested_at": ingested_at, "tables": {}}
    all_bad: list[dict] = []
    for table, columns in SOURCE_SCHEMAS.items():
        src_file = extract_dir / f"{table}.dat"
        df, bad = parse_dat(src_file, columns, ing["encoding"])
        df["_ingested_at"] = ingested_at
        out = write_parquet(df, bronze_dir / f"{table}.parquet")
        all_bad.extend(bad)
        manifest["tables"][table] = {"source_file": relpath(src_file), "file_md5": file_md5[table],
                                     "rows": int(len(df)), "malformed_lines": len(bad),
                                     "columns": columns, "bronze_path": relpath(out)}
        log.info("Ingested %-7s rows=%-8d malformed=%d -> %s", table, len(df), len(bad), relpath(out))

    write_parquet(pd.DataFrame(all_bad, columns=["file", "raw_fields", "n_fields", "expected_fields"]),
                  paths.errors / "malformed_lines.parquet")
    write_json(manifest, paths.reports / "ingestion_manifest.json")
    return manifest


if __name__ == "__main__":
    run()
