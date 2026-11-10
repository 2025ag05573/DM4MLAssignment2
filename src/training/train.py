"""Stage 6 - Model training (with hyper-parameter tuning and MLflow tracking).

``tune``  : grid-search on the validation window (fit=train, score=val);
            every trial is a nested MLflow run; best params -> models/best_params.json
``train`` : fit each model on train+val with the best params; one MLflow run per
            model (params, data version, fit time, pickled model as artifact);
            run ids -> models/registry.json so evaluation logs into the same runs.
"""
from __future__ import annotations

import itertools
import time
from typing import Any, Callable

import pandas as pd

from src.common.config import get_paths, load_config, set_seed
from src.common.io import read_json, write_json
from src.common.logging_utils import get_logger
from src.evaluation.protocol import build_eval_set, evaluate
from src.training import tracking
from src.training.base import FitContext, Recommender, load_context
from src.training.hybrid import HybridRecommender
from src.training.knn import ItemKNNRecommender
from src.training.mf import ExplicitMF, ImplicitALS
from src.training.popularity import PopularityRecommender

log = get_logger("training")
MODEL_NAMES = ("popularity", "explicit_mf", "item_knn", "implicit_als", "hybrid")


def _data_tags(cfg: dict[str, Any], stage: str) -> dict[str, str]:
    paths = get_paths(cfg)
    manifest = read_json(paths.reports / "ingestion_manifest.json")
    registry = read_json(paths.features / "registry.json")
    return {"stage": stage, "data.archive_md5": manifest["archive_md5"],
            "data.ratings_md5": manifest["tables"]["ratings"]["file_md5"],
            "features.version": registry["definitions_version"],
            "features.snapshot": stage, "config": cfg["_config_path"]}


def default_params(cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    t = cfg["training"]
    return {name: dict(t[name]) for name in MODEL_NAMES}


# --------------------------------------------------------------------------- tuning
def _grid(spec: dict[str, list]) -> list[dict[str, Any]]:
    keys = list(spec)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(spec[k] for k in keys))]


def _search(cfg, ctx, es, family: str, grid: list[dict], build: Callable[[dict], Recommender],
            metric: str, higher_is_better: bool, offset: int = 0) -> tuple[dict, Recommender, pd.DataFrame]:
    rows, best = [], None
    for i, params in enumerate(grid, start=offset):
        t0 = time.perf_counter()
        model = build(params).fit(ctx)
        res = evaluate(model, ctx, es, cfg["evaluation"]["k_values"])
        with tracking.start_run(cfg, f"{family}-trial-{i:02d}", nested=True,
                                tags={"family": family, "phase": "tuning"}):
            tracking.log_params({**model.params(), **params})
            tracking.log_metrics(res, prefix="val_")
            tracking.log_metrics({"fit_seconds": time.perf_counter() - t0, **model.diagnostics()})
        rows.append({**params, **res, **model.diagnostics()})
        score = res[metric]
        if best is None or (score > best[0] if higher_is_better else score < best[0]):
            best = (score, params, model)
        log.info("tune %-12s %s -> %s=%.4f", family, params, metric, score)
    return best[1], best[2], pd.DataFrame(rows)


def _tune_als(cfg, ctx, es, g: dict[str, Any], metric: str,
              build: Callable[[dict], Recommender]) -> tuple[dict, Recommender, pd.DataFrame]:
    """Two-stage (coordinate) search for implicit ALS.

    Stage 1 grids regularisation x confidence scaling (``reg`` x ``alpha``) at a fixed,
    mid-sized ``stage1_factors``. Stage 2 varies the number of factors at the best
    reg/alpha. Larger models cost more to fit and serve, so stage 2 keeps the *smallest*
    factor count whose score is within ``size_tolerance`` (relative) of the best one.
    """
    s1 = _grid({"factors": [g["stage1_factors"]], "reg": g["reg"], "alpha": g["alpha"]})
    p1, m1, t1 = _search(cfg, ctx, es, "implicit_als", s1, build, metric, True)
    s2 = [{**p1, "factors": f} for f in sorted(g["factors"]) if f != p1["factors"]]
    _, m2, t2 = _search(cfg, ctx, es, "implicit_als", s2, build, metric, True, offset=len(s1))
    stage2 = pd.concat([t1[(t1["reg"] == p1["reg"]) & (t1["alpha"] == p1["alpha"])], t2]).sort_values("factors")
    top = stage2[metric].max()
    chosen = stage2[stage2[metric] >= top * (1 - g.get("size_tolerance", 0.0))].iloc[0]
    params = {**p1, "factors": int(chosen["factors"])}
    log.info("ALS stage 2: best %s=%.5f; smallest model within %.0f%% -> factors=%d (%s=%.5f)", metric, top,
             100 * g.get("size_tolerance", 0.0), params["factors"], metric, chosen[metric])
    if params["factors"] == p1["factors"]:
        model = m1
    elif float(chosen[metric]) == float(t2[metric].max()):
        model = m2
    else:
        model = build(params).fit(ctx)
    tab = pd.concat([t1.assign(stage=1), t2.assign(stage=2)], ignore_index=True)
    tab["selected"] = ((tab["factors"] == params["factors"]) & (tab["reg"] == params["reg"])
                       & (tab["alpha"] == params["alpha"]))
    return params, model, tab


def tune(cfg: dict[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    cfg = cfg or load_config()
    set_seed(cfg["project"]["seed"])
    paths = get_paths(cfg)
    best = default_params(cfg)
    tcfg = cfg["training"]["tuning"]
    if not tcfg["enabled"]:
        log.info("tuning disabled - using config defaults")
        write_json(best, paths.models / "best_params.json")
        return best

    ctx = load_context(cfg, "tune")
    es = build_eval_set(ctx, pd.read_parquet(paths.splits / "val.parquet"), "val")
    metric = tcfg["metric"]
    seed = cfg["project"]["seed"]
    with tracking.start_run(cfg, "hyperparameter-tuning", tags={**_data_tags(cfg, "tune"), "phase": "tuning"}):
        tracking.log_params({"selection_metric": metric, **{f"val_{k}": v for k, v in es.stats.items()}})
        base = cfg["training"]["implicit_als"]
        als_p, als_model, als_tab = _tune_als(cfg, ctx, es, tcfg["implicit_als_grid"], metric,
                                              lambda p: ImplicitALS(iterations=base["iterations"], seed=seed,
                                                                    solver=base["solver"],
                                                                    cg_steps=base["cg_steps"], **p))
        best["implicit_als"].update(als_p)

        mf = cfg["training"]["explicit_mf"]
        mf_grid = _grid(tcfg.get("explicit_mf_grid", {"factors": [mf["factors"]], "reg": [mf["reg"]]}))
        mf_grid += [dict(t) for t in tcfg.get("explicit_mf_extra_trials", [])]
        mf_p, _, mf_tab = _search(cfg, ctx, es, "explicit_mf", mf_grid,
                                  lambda p: ExplicitMF(iterations=mf["iterations"], seed=seed, **p),
                                  "rmse", False)
        best["explicit_mf"].update(mf_p)

        knn_p, _, knn_tab = _search(cfg, ctx, es, "item_knn", _grid(tcfg["item_knn_grid"]),
                                    lambda p: ItemKNNRecommender(**p), metric, True)
        best["item_knn"].update(knn_p)

        hy_p, _, hy_tab = _search(cfg, ctx, es, "hybrid", _grid(tcfg["hybrid_grid"]),
                                  lambda p: HybridRecommender(als_model, w_cf=1.0, **p), metric, True)
        best["hybrid"].update(hy_p)

        out = paths.reports / "evaluation"
        write_json(es.stats, out / "tuning_protocol.json")
        for name, tab in (("implicit_als", als_tab), ("explicit_mf", mf_tab), ("item_knn", knn_tab),
                          ("hybrid", hy_tab)):
            tab.to_csv(out / f"tuning_{name}.csv", index=False)
            tracking.log_artifact(out / f"tuning_{name}.csv", "tuning")
        tracking.log_params({f"best_{m}_{k}": v for m, p in best.items() for k, v in p.items()})
    write_json(best, paths.models / "best_params.json")
    log.info("best params: %s", best)
    return best


# --------------------------------------------------------------------------- final training
def build_model(name: str, params: dict[str, Any], seed: int, cf: ImplicitALS | None = None) -> Recommender:
    if name == "popularity":
        return PopularityRecommender(**params)
    if name == "explicit_mf":
        return ExplicitMF(seed=seed, **params)
    if name == "item_knn":
        return ItemKNNRecommender(**params)
    if name == "implicit_als":
        return ImplicitALS(seed=seed, **params)
    if name == "hybrid":
        assert cf is not None, "hybrid needs a fitted implicit_als model"
        return HybridRecommender(cf, **params)
    raise KeyError(name)


def train_model(name: str, cfg: dict[str, Any] | None = None, ctx: FitContext | None = None) -> dict[str, Any]:
    """Fit one model on train+val, save it and log the run. Returns its registry entry."""
    cfg = cfg or load_config()
    set_seed(cfg["project"]["seed"])
    paths = get_paths(cfg)
    params_file = paths.models / "best_params.json"
    params = read_json(params_file)[name] if params_file.exists() else default_params(cfg)[name]
    ctx = ctx or load_context(cfg, "final")
    cf = Recommender.load(paths.models / "implicit_als.pkl") if name == "hybrid" else None

    with tracking.start_run(cfg, f"final-{name}", tags={**_data_tags(cfg, "final"), "model": name,
                                                       "phase": "final"}) as run:
        t0 = time.perf_counter()
        model = build_model(name, params, cfg["project"]["seed"], cf).fit(ctx)
        fit_s = time.perf_counter() - t0
        path = model.save(paths.models / f"{name}.pkl")
        tracking.log_params(model.params())
        tracking.log_params({"fit_rows": int(ctx.X.nnz), "fit_users": int((ctx.X.getnnz(axis=1) > 0).sum()),
                             "fit_items": int((ctx.X.getnnz(axis=0) > 0).sum())}, prefix="data_")
        tracking.log_metrics({"fit_seconds": fit_s, **model.diagnostics()})
        tracking.log_artifact(path, "model")
        entry = {"model": name, "path": str(path.relative_to(paths.root)), "mlflow_run_id": run.info.run_id,
                 "params": model.params(), "diagnostics": model.diagnostics(), "fit_seconds": round(fit_s, 2)}
    log.info("trained %-12s in %.1fs -> %s (run %s)", name, fit_s, entry["path"], entry["mlflow_run_id"][:8])
    # One metadata file per model, so models can be trained concurrently without a shared-file race.
    write_json(entry, paths.models / f"{name}.meta.json")
    return entry


def model_registry(cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    paths = get_paths(cfg)
    return {n: read_json(paths.models / f"{n}.meta.json") for n in MODEL_NAMES
            if (paths.models / f"{n}.meta.json").exists()}


def run(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = cfg or load_config()
    tune(cfg)
    ctx = load_context(cfg, "final")
    return {name: train_model(name, cfg, ctx) for name in MODEL_NAMES}


if __name__ == "__main__":
    run()
