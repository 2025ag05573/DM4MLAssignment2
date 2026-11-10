"""Prefect orchestration of the end-to-end recommendation pipeline.

    ingest -> validate -> prepare -> split -> transform -> build_features -> tune
                                                                              |
           +----------------+----------------+-----------------+-------------+
           v                v                v                 v
   train popularity  train explicit_mf  train item_knn  train implicit_als
           |                |                |                 v
           |                |                |           train hybrid
           +----------------+-------+--------+-----------------+
                                    v
                                evaluate -> batch_recommend

Every task reads its inputs from and writes its outputs to disk (data/, models/,
reports/), so any task can be re-run in isolation and a run is reproducible
from the config + seed. Validation is a hard quality gate: if it raises, all
downstream tasks are skipped.

Run once (ephemeral Prefect, no server needed):
    python -m dags.recommendation_flow
    python -m dags.recommendation_flow --skip-tuning      # ~1-2 min instead of ~8-10
Serve on a schedule (weekly retrain, Mondays 02:00):
    python -m dags.recommendation_flow --serve
"""
from __future__ import annotations

import argparse
import os

# Route the project's stage loggers into Prefect's run logs (must precede the prefect import).
os.environ.setdefault("PREFECT_LOGGING_EXTRA_LOGGERS",
                      "ingestion,validation,preparation,split,transformation,features,training,"
                      "training.mf,training.knn,evaluation,inference")
os.environ.setdefault("PREFECT_SERVER_ANALYTICS_ENABLED", "false")   # no telemetry writes
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

from prefect import flow, task  # noqa: E402
from prefect.artifacts import create_markdown_artifact, create_table_artifact  # noqa: E402

from src.common.config import get_paths, load_config, set_seed  # noqa: E402
from src.common.io import read_json  # noqa: E402


@task(name="ingest", retries=2, retry_delay_seconds=10)
def ingest_task(cfg: dict) -> dict:
    from src.ingestion import ingest
    m = ingest.run(cfg)
    return {t: v["rows"] for t, v in m["tables"].items()} | {"archive_md5": m["archive_md5"]}


@task(name="validate")
def validate_task(cfg: dict, ingested: dict) -> dict:
    from src.validation import validate
    summary = validate.run(cfg)  # raises DataQualityError -> downstream tasks never run
    paths = get_paths(cfg)
    checks = read_json(paths.reports / "validation" / "validation_report.json")["checks"]
    create_table_artifact(key="validation-summary", description="Rows in / valid / quarantined per table",
                          table=[{"table": t, **{k: round(v, 6) if isinstance(v, float) else v
                                                 for k, v in s.items()}} for t, s in summary.items()])
    create_table_artifact(key="validation-checks", description="All data-quality checks",
                          table=[{k: c[k] for k in ("table", "check", "column", "severity", "failed_rows",
                                                    "status")} for c in checks])
    return summary


@task(name="prepare")
def prepare_task(cfg: dict, validated: dict) -> dict:
    from src.preparation import prepare
    return prepare.run(cfg)


@task(name="time-split")
def split_task(cfg: dict, prepared: dict) -> dict:
    from src.preparation import split
    return split.run(cfg)


@task(name="transform")
def transform_task(cfg: dict, splits: dict) -> dict:
    from src.transformation import transform
    s = transform.run(cfg)
    return {k: v for k, v in s.items() if k != "top5_items"}


@task(name="build-features")
def features_task(cfg: dict, derived: dict) -> dict:
    from src.features import build
    reg = build.run(cfg)
    return {"definitions_version": reg["definitions_version"],
            "snapshots": {k: v["as_of_iso"] for k, v in reg["snapshots"].items()}}


@task(name="tune-hyperparameters")
def tune_task(cfg: dict, features: dict) -> dict:
    from src.training import train
    return train.tune(cfg)


@task(name="train-model", task_run_name="train-{name}")
def train_task(cfg: dict, name: str, params: dict) -> dict:
    from src.training import train
    e = train.train_model(name, cfg)
    return {"model": name, "mlflow_run_id": e["mlflow_run_id"], "fit_seconds": e["fit_seconds"]}


@task(name="evaluate")
def evaluate_task(cfg: dict, trained: list[dict]) -> dict:
    from src.evaluation import evaluate
    results = evaluate.run(cfg)
    k = cfg["evaluation"]["k"]
    keep = [f"precision@{k}", f"recall@{k}", f"ndcg@{k}", f"coverage@{k}", f"diversity@{k}", "rmse", "mae"]
    table = [{"model": n, **{m: round(r[m], 4) for m in keep if m in r}} for n, r in results.items()]
    create_table_artifact(key="model-comparison", table=table,
                          description=f"Test-window metrics (K={k})")
    md = (get_paths(cfg).reports / "evaluation" / "model_comparison.md").read_text(encoding="utf-8")
    create_markdown_artifact(key="evaluation-report", markdown=md.split("![")[0],
                             description="Full evaluation report")
    return {n: {m: r[m] for m in keep if m in r} for n, r in results.items()}


@task(name="batch-recommend")
def recommend_task(cfg: dict, evaluated: dict, model: str = "hybrid") -> str:
    from src.inference.recommend import main
    main(["--batch", "--model", model])
    return f"data/processed/recommendations/{model}_top{cfg['evaluation']['k']}.parquet"


@flow(name="movielens-recommendation-pipeline", log_prints=True)
def recommendation_pipeline(config_path: str | None = None, skip_tuning: bool = False) -> dict:
    cfg = load_config(config_path)
    if skip_tuning:
        cfg["training"]["tuning"]["enabled"] = False
    set_seed(cfg["project"]["seed"])

    ingested = ingest_task(cfg)
    validated = validate_task(cfg, ingested)
    prepared = prepare_task(cfg, validated)
    splits = split_task(cfg, prepared)
    derived = transform_task(cfg, splits)
    features = features_task(cfg, derived)
    params = tune_task(cfg, features)

    # Independent models train concurrently; the hybrid re-uses the fitted ALS.
    pop = train_task.submit(cfg, "popularity", params)
    mf = train_task.submit(cfg, "explicit_mf", params)
    knn = train_task.submit(cfg, "item_knn", params)
    als = train_task.submit(cfg, "implicit_als", params)
    hyb = train_task.submit(cfg, "hybrid", params, wait_for=[als])
    trained = [f.result() for f in (pop, mf, knn, als, hyb)]

    results = evaluate_task(cfg, trained)
    recs = recommend_task(cfg, results)
    return {"results": results, "recommendations": recs}


def main() -> None:
    ap = argparse.ArgumentParser(description="Run or schedule the recommendation pipeline")
    ap.add_argument("--config", default=None, help="path to pipeline.yaml")
    ap.add_argument("--skip-tuning", action="store_true", help="use tuned defaults from the config")
    ap.add_argument("--serve", action="store_true", help="serve with a weekly cron schedule")
    ap.add_argument("--cron", default="0 2 * * 1", help="cron for --serve (default Mon 02:00)")
    args = ap.parse_args()
    if args.serve:
        recommendation_pipeline.serve(name="weekly-retrain", cron=args.cron,
                                      parameters={"config_path": args.config, "skip_tuning": args.skip_tuning})
    else:
        recommendation_pipeline(config_path=args.config, skip_tuning=args.skip_tuning)


if __name__ == "__main__":
    main()
