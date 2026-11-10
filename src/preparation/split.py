"""Stage 3b - Time-based train / validation / test split.

A single *global* time cut-off is used: everything the model is trained on
happened strictly before anything it is evaluated on, so there is no leakage
of future behaviour (a random or per-user split would let the model see
ratings made after the "present" moment it is evaluated at).

    |------------- train (80%) -------------|-- val (10%) --|-- test (10%) --|
                                         t_val           t_test

* **tune stage** : fit on ``train``, evaluate on ``val`` (hyper-parameter search)
* **final stage**: fit on ``train + val``, evaluate on ``test`` (reported numbers)

Cut-offs are timestamps, so ratings sharing a timestamp always land on the
same side.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from src.common.config import get_paths, load_config
from src.common.io import write_json, write_parquet
from src.common.logging_utils import get_logger

log = get_logger("split")


def global_time_split(ratings: pd.DataFrame, val_fraction: float, test_fraction: float
                      ) -> tuple[dict[str, pd.DataFrame], dict[str, int]]:
    r = ratings.sort_values(["timestamp", "user_id", "movie_id"]).reset_index(drop=True)
    n = len(r)
    t_val = int(r["timestamp"].iloc[int(n * (1 - val_fraction - test_fraction))])
    t_test = int(r["timestamp"].iloc[int(n * (1 - test_fraction))])
    parts = {
        "train": r[r["timestamp"] < t_val],
        "val": r[(r["timestamp"] >= t_val) & (r["timestamp"] < t_test)],
        "test": r[r["timestamp"] >= t_test],
    }
    # Leakage guard: strict temporal ordering between the parts.
    assert parts["train"]["timestamp"].max() < parts["val"]["timestamp"].min()
    assert parts["val"]["timestamp"].max() < parts["test"]["timestamp"].min()
    return {k: v.reset_index(drop=True) for k, v in parts.items()}, {"t_val": t_val, "t_test": t_test}


def describe(parts: dict[str, pd.DataFrame], cut: dict[str, int]) -> dict[str, Any]:
    iso = lambda t: pd.Timestamp(t, unit="s", tz="UTC").isoformat()  # noqa: E731
    out: dict[str, Any] = {"cutoffs": {k: iso(v) for k, v in cut.items()}, "parts": {}}
    seen_users = set(parts["train"]["user_id"])
    seen_items = set(parts["train"]["movie_id"])
    for name in ("train", "val", "test"):
        p = parts[name]
        d = {"rows": len(p), "users": int(p["user_id"].nunique()), "items": int(p["movie_id"].nunique()),
             "start": iso(p["timestamp"].min()), "end": iso(p["timestamp"].max())}
        if name != "train":
            d["warm_users"] = int(p["user_id"].drop_duplicates().isin(seen_users).sum())
            d["rows_from_warm_users"] = float(p["user_id"].isin(seen_users).mean())
            d["rows_on_warm_items"] = float(p["movie_id"].isin(seen_items).mean())
        out["parts"][name] = d
        seen_users |= set(p["user_id"])
        seen_items |= set(p["movie_id"])
    return out


def run(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = cfg or load_config()
    paths = get_paths(cfg)
    s = cfg["split"]
    if s["strategy"] != "global_time":
        raise ValueError(f"unsupported split strategy {s['strategy']}")
    ratings = pd.read_parquet(paths.clean / "ratings.parquet")
    parts, cut = global_time_split(ratings, s["val_fraction"], s["test_fraction"])
    for name, df in parts.items():
        write_parquet(df, paths.splits / f"{name}.parquet")
    summary = describe(parts, cut)
    write_json(summary, paths.reports / "split_summary.json")
    for name, d in summary["parts"].items():
        log.info("%-5s rows=%-7d users=%-5d items=%-5d %s -> %s", name, d["rows"], d["users"],
                 d["items"], d["start"][:10], d["end"][:10])
    return summary


if __name__ == "__main__":
    run()
