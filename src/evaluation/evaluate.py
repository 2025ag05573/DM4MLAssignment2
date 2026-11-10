"""Stage 7 - Model evaluation on the held-out test window.

Loads every trained model, scores it with the shared protocol (``protocol.py``),
logs the metrics into the model's own MLflow run, and writes the comparison
table, bootstrap confidence intervals, a per-segment breakdown, a chart and
sample recommendations.

Outputs (reports/evaluation/)
-----------------------------
model_comparison.{md,csv,json}   all metrics @5 and @10 (+ RMSE / MAE), 95% CIs, segments
model_comparison.png             small-multiples bar chart of the headline metrics
sample_recommendations.md        top-10 titles for a few test users, per model
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.common.config import get_paths, load_config, set_seed
from src.common.io import df_to_markdown, utc_now, write_json
from src.common.logging_utils import get_logger, relpath
from src.evaluation.protocol import EvalSet, build_eval_set, evaluate, top_k
from src.training import tracking
from src.training.base import FitContext, Recommender, load_context
from src.training.train import MODEL_NAMES, model_registry

log = get_logger("evaluation")

LABELS = {"popularity": "Popularity (baseline)", "explicit_mf": "Explicit MF (biased ALS)",
          "item_knn": "Item-kNN CF", "implicit_als": "Implicit ALS", "hybrid": "Hybrid (ALS + features)"}
# Reference categorical palette, slots 1-5 (validated; < 3:1 slots carry direct value labels).
COLORS = {"popularity": "#2a78d6", "explicit_mf": "#eb6834", "item_knn": "#1baf7a",
          "implicit_als": "#eda100", "hybrid": "#e87ba4"}
# Users by the number of ratings they made before the test window.
SEGMENTS = [(1, 49, "1-49 prior ratings"), (50, 199, "50-199 prior ratings"), (200, 10**9, "200+ prior ratings")]


def comparison_table(results: dict[str, dict[str, float]], k: int) -> pd.DataFrame:
    cols = [f"precision@{k}", f"recall@{k}", f"ndcg@{k}", f"hit_rate@{k}", f"coverage@{k}",
            f"diversity@{k}", f"novelty@{k}", "rmse", "mae"]
    return pd.DataFrame([{"model": LABELS[n], **{c: results[n].get(c, np.nan) for c in cols}}
                         for n in MODEL_NAMES if n in results])


def bootstrap(per_user: dict[str, pd.DataFrame], metric: str, baseline: str, n_boot: int, seed: int
              ) -> pd.DataFrame:
    """95% percentile-bootstrap CI of each model's mean, and of its paired difference to the
    baseline (same resampled users for both models, so user difficulty cancels out)."""
    rng = np.random.default_rng(seed)
    n = len(next(iter(per_user.values())))
    idx = rng.integers(0, n, size=(n_boot, n))
    base = per_user[baseline][metric].to_numpy()
    rows = []
    for name, pu in per_user.items():
        v = pu[metric].to_numpy()
        means = v[idx].mean(axis=1)
        diffs = (v - base)[idx].mean(axis=1)
        lo, hi = np.percentile(means, [2.5, 97.5])
        dlo, dhi = np.percentile(diffs, [2.5, 97.5])
        rows.append({"model": LABELS[name], metric: v.mean(), "ci_low": lo, "ci_high": hi,
                     f"diff_vs_{baseline}": (v - base).mean(), "diff_ci_low": dlo, "diff_ci_high": dhi,
                     "significant": "-" if name == baseline else ("yes" if dlo > 0 or dhi < 0 else "no")})
    return pd.DataFrame(rows)


def bootstrap_rating_error(models: dict, es, baseline: str, n_boot: int, seed: int) -> pd.DataFrame:
    """95% CIs for RMSE / MAE of the rating predictors and their paired difference to the
    baseline. Users are resampled (all of a user's test ratings move together), so the
    interval reflects variation across users, not just across rating pairs."""
    rng = np.random.default_rng(seed)
    users, inv = np.unique(es.pairs_u, return_inverse=True)
    idx = rng.integers(0, len(users), size=(n_boot, len(users)))
    count = np.bincount(inv, minlength=len(users))
    n_b = count[idx].sum(axis=1)

    def errors(model):
        e = model.predict_ratings(es.pairs_u, es.pairs_i) - es.pairs_r
        return np.bincount(inv, e ** 2, len(users)), np.bincount(inv, np.abs(e), len(users))

    base_sq, base_ab = errors(models[baseline])
    rows = []
    for name, model in models.items():
        if not getattr(model, "predicts_ratings", False):
            continue
        sq, ab = errors(model)
        for metric, v, b, f in (("rmse", sq, base_sq, np.sqrt), ("mae", ab, base_ab, lambda x: x)):
            point = f(v.sum() / count.sum())
            diff = point - f(b.sum() / count.sum())
            boot = f(v[idx].sum(axis=1) / n_b)
            bdiff = boot - f(b[idx].sum(axis=1) / n_b)
            lo, hi = np.percentile(boot, [2.5, 97.5])
            dlo, dhi = np.percentile(bdiff, [2.5, 97.5])
            rows.append({"model": LABELS[name], "metric": metric, "value": point, "ci_low": lo, "ci_high": hi,
                         f"diff_vs_{baseline}": diff, "diff_ci_low": dlo, "diff_ci_high": dhi,
                         "significant": "-" if name == baseline else ("yes" if dlo > 0 or dhi < 0 else "no")})
    return pd.DataFrame(rows)


def segment_table(per_user: dict[str, pd.DataFrame], history: pd.Series, k: int) -> pd.DataFrame:
    rows = []
    for lo, hi, label in SEGMENTS:
        for name, pu in per_user.items():
            h = history.reindex(pu["uidx"]).to_numpy()
            m = (h >= lo) & (h <= hi)
            rows.append({"segment": label, "users": int(m.sum()), "model": LABELS[name],
                         f"ndcg@{k}": pu.loc[m, f"ndcg@{k}"].mean(), f"hit_rate@{k}": pu.loc[m, f"hit_rate@{k}"].mean()})
    return pd.DataFrame(rows)


def plot_comparison(results: dict[str, dict[str, float]], k: int, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [(f"precision@{k}", f"Precision@{k}"), (f"recall@{k}", f"Recall@{k}"),
              (f"ndcg@{k}", f"NDCG@{k}"), (f"coverage@{k}", f"Coverage@{k}"),
              (f"diversity@{k}", f"Diversity@{k} (ILD)"), ("rmse", "RMSE (lower is better)")]
    names = [n for n in MODEL_NAMES if n in results]
    ink, muted, grid = "#0b0b0b", "#52514e", "#e4e3df"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.8), facecolor="#fcfcfb")
    for ax, (key, title) in zip(axes.ravel(), panels):
        ax.set_facecolor("#fcfcfb")
        vals = [results[n].get(key, np.nan) for n in names]
        finite = [v for v in vals if not np.isnan(v)]
        top = max(finite) if finite else 1
        for x, n, v in zip(np.arange(len(names)), names, vals):
            if np.isnan(v):
                ax.text(x, top * 0.02, "n/a", ha="center", va="bottom", color=muted, fontsize=7.5)
                continue
            ax.bar(x, v, width=0.74, color=COLORS[n], edgecolor="#fcfcfb", linewidth=2, zorder=2)
            ax.text(x, v, f"{v:.3f}" if v >= 0.01 else f"{v:.4f}", ha="center", va="bottom", color=ink,
                    fontsize=7.5)
        ax.set_title(title, color=ink, fontsize=10, loc="left", pad=8)
        ax.set_xlim(-0.5, len(names) - 0.5)
        ax.set_xticks([])
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color(grid)
        ax.tick_params(axis="y", colors=muted, labelsize=8, length=0)
        ax.grid(axis="y", color=grid, linewidth=0.8, zorder=0)
        ax.set_ylim(0, top * 1.22 if finite else 1)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[n]) for n in names]
    fig.legend(handles, [LABELS[n] for n in names], loc="lower center", ncol=len(names), frameon=False,
               fontsize=8.5, labelcolor=ink)
    fig.suptitle(f"Model comparison on the test window (K = {k})", x=0.012, ha="left", color=ink,
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    fig.savefig(path, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)


def sample_recommendations(models: dict[str, Recommender], ctx: FitContext, es: EvalSet, n_users: int,
                           k: int, seed: int) -> str:
    movies = pd.read_parquet(get_paths(ctx.cfg).clean / "movies.parquet").set_index("movie_id")["title"]
    iid = ctx.item_index["movie_id"].to_numpy()
    uid = ctx.user_index["user_id"].to_numpy()
    rng = np.random.default_rng(seed)
    pick = np.sort(rng.choice(len(es.uidx), size=min(n_users, len(es.uidx)), replace=False))
    sub = EvalSet(es.name, es.uidx[pick], [es.relevant[i] for i in pick], es.candidate_mask,
                  es.item_vectors, es.item_pop_share, es.pairs_u, es.pairs_i, es.pairs_r, es.stats)
    lines = ["# Sample recommendations (test window)", "",
             "Titles in **bold** were rated >= 4 by the user in the test window (hits).", ""]
    recs = {n: top_k(m, ctx, sub, k) for n, m in models.items()}
    for j, u in enumerate(sub.uidx):
        lines += [f"## User {uid[u]}  ({len(sub.relevant[j])} relevant test items)", "",
                  "| rank | " + " | ".join(LABELS[n] for n in recs) + " |", "|---|" + "---|" * len(recs)]
        for r in range(k):
            cells = []
            for n in recs:
                i = recs[n][j, r]
                t = str(movies.get(iid[i], iid[i])).replace("|", "/")
                cells.append(f"**{t}**" if i in sub.relevant[j] else t)
            lines.append(f"| {r + 1} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def run(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = cfg or load_config()
    seed = cfg["project"]["seed"]
    set_seed(seed)
    paths = get_paths(cfg)
    ecfg = cfg["evaluation"]
    out = paths.reports / "evaluation"
    ctx = load_context(cfg, "final")
    es = build_eval_set(ctx, pd.read_parquet(paths.splits / "test.parquet"), "test")
    log.info("test protocol: %s", es.stats)

    registry = model_registry(cfg)
    models = {n: Recommender.load(paths.root / registry[n]["path"]) for n in MODEL_NAMES if n in registry}
    results: dict[str, dict[str, float]] = {}
    per_user: dict[str, pd.DataFrame] = {}
    for name, model in models.items():
        res, pu = evaluate(model, ctx, es, ecfg["k_values"], per_user=True)
        results[name], per_user[name] = res, pu
        with tracking.start_run(cfg, f"final-{name}", run_id=registry[name]["mlflow_run_id"]):
            tracking.log_metrics(res, prefix="test_")
            tracking.log_params({f"test_{k}": v for k, v in es.stats.items()})
        log.info("%-12s P@10=%.4f R@10=%.4f NDCG@10=%.4f HR@10=%.4f Cov@10=%.3f Div@10=%.3f RMSE=%s", name,
                 res["precision@10"], res["recall@10"], res["ndcg@10"], res["hit_rate@10"], res["coverage@10"],
                 res["diversity@10"], f"{res['rmse']:.4f}" if "rmse" in res else "n/a")

    k = ecfg["k"]
    kmax = max(ecfg["k_values"])
    tables = {kk: comparison_table(results, kk) for kk in ecfg["k_values"]}
    tables[k].to_csv(out / "model_comparison.csv", index=False)
    ci = {m: bootstrap(per_user, f"{m}@{kmax}", "popularity", ecfg.get("n_bootstrap", 1000), seed)
          for m in ("ndcg", "hit_rate")}
    ci["rating"] = bootstrap_rating_error(models, es, "popularity", ecfg.get("n_bootstrap", 1000), seed)
    history = ctx.store.get_user_features(ctx.user_index["user_id"], "final")["user_rating_count"]
    history.index = ctx.user_index["uidx"].to_numpy()
    seg = segment_table(per_user, history, kmax)
    for name, df in (("bootstrap_ndcg", ci["ndcg"]), ("bootstrap_hit_rate", ci["hit_rate"]),
                     ("bootstrap_rating_error", ci["rating"]), ("segments", seg)):
        df.to_csv(out / f"{name}.csv", index=False)
    write_json({"generated": utc_now(), "protocol": es.stats, "k_values": ecfg["k_values"],
                "relevance_threshold": ecfg["relevance_threshold"], "results": results,
                "bootstrap_ndcg": ci["ndcg"].to_dict("records"), "bootstrap_hit_rate": ci["hit_rate"].to_dict("records"),
                "bootstrap_rating_error": ci["rating"].to_dict("records"),
                "segments": seg.to_dict("records")}, out / "model_comparison.json")

    asof = pd.Timestamp(ctx.store.as_of("final"), unit="s", tz="UTC")
    md = ["# Model Evaluation - test window", "",
          f"- Generated: {utc_now()}",
          f"- Fit window: train + val (ratings before {asof:%Y-%m-%d %H:%M} UTC); test window: everything after.",
          f"- Relevant = held-out rating >= {ecfg['relevance_threshold']}; candidates = items seen in the fit "
          "window, minus each user's already-rated items.",
          f"- Users evaluated: {es.stats['eval_users']} warm users with >= 1 relevant test item "
          f"(cold-start users excluded: {es.stats['cold_start_users']}; test ratings of movies new in the test "
          f"window: {es.stats['rows_on_cold_items']}); candidate items: {es.stats['candidate_items']}; "
          f"rating pairs for RMSE/MAE: {es.stats['rating_pairs']}.",
          "- Coverage = share of candidate items recommended to anyone; Diversity = mean pairwise "
          "genre-cosine distance inside a list; Novelty = mean -log2(popularity share).",
          "- RMSE / MAE only for models that predict explicit ratings (popularity uses the damped item mean).", ""]
    for kk in ecfg["k_values"]:
        md += [f"## K = {kk}", "", df_to_markdown(tables[kk]), ""]
    md += [f"## How sure are we? 95% bootstrap confidence intervals ({ecfg.get('n_bootstrap', 1000)} resamples of users)",
           "", "`significant` = the paired difference to the popularity baseline has a CI that excludes 0.", "",
           df_to_markdown(ci["ndcg"]), "", df_to_markdown(ci["hit_rate"]), "",
           "Rating error (lower is better; a negative difference = better than the damped item mean):", "",
           df_to_markdown(ci["rating"]), "",
           f"## By user history (ratings before the test window), K = {kmax}", "", df_to_markdown(seg), "",
           "![Model comparison](model_comparison.png)", ""]
    (out / "model_comparison.md").write_text("\n".join(md), encoding="utf-8")
    plot_comparison(results, k, out / "model_comparison.png")
    (out / "sample_recommendations.md").write_text(
        sample_recommendations(models, ctx, es, ecfg["n_sample_recommendations"], k, seed), encoding="utf-8")
    with tracking.start_run(cfg, "evaluation-summary", tags={"phase": "evaluation"}):
        for f in ("model_comparison.md", "model_comparison.csv", "model_comparison.png", "bootstrap_ndcg.csv",
                  "bootstrap_hit_rate.csv", "bootstrap_rating_error.csv", "segments.csv", "sample_recommendations.md"):
            tracking.log_artifact(out / f, "evaluation")
        tracking.log_artifact(paths.reports / "feature_catalog.md", "features")
        tracking.log_artifact(paths.reports / "validation" / "validation_report.md", "validation")
    log.info("results -> %s", relpath(out / "model_comparison.md"))
    return results


if __name__ == "__main__":
    run()
