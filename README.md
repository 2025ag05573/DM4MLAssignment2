# End-to-End Data Management Pipeline for a Recommendation System

**Divya Sharma · 2025AG05573** · DM4ML (AIMLZG529) · Assignment 2 — MovieLens 1M · Pandera · custom feature store · MLflow · Prefect

A reproducible pipeline that ingests the MovieLens 1M dataset, validates it against a
data contract, prepares and transforms it, materialises point-in-time features, tunes
and trains five recommenders, evaluates them on a held-out future time window with
confidence intervals, and serves top-K recommendations. The whole workflow is a single
Prefect flow.

```
Ingestion → Validation → Preparation → Transformation → Feature store → Training → Evaluation → Serving
```

## Results at a glance

Test window = the last 10% of ratings by time (29 Dec 2000 → Feb 2003); models are fitted on
everything before it. 1,158 users evaluated, K = 10, relevant = rating ≥ 4.

| Model | Precision@10 | Recall@10 | NDCG@10 | HitRate@10 | Coverage@10 | Diversity@10 | RMSE | MAE |
|---|---|---|---|---|---|---|---|---|
| Popularity (baseline) | 0.186 | 0.052 | 0.201 | 0.624 | 0.040 | 0.661 | 0.955 | 0.757 |
| Explicit MF (biased ALS) | 0.050 | 0.017 | 0.044 | 0.309 | 0.086 | 0.660 | **0.891** | **0.697** |
| Item-kNN CF | 0.190 | 0.053 | 0.209 | 0.602 | 0.058 | 0.656 | – | – |
| Implicit ALS | 0.211 | 0.066 | 0.232 | 0.659 | **0.101** | **0.679** | – | – |
| **Hybrid (ALS + features)** | **0.215** | **0.068** | **0.233** | **0.678** | 0.085 | 0.663 | – | – |

- **Implicit ALS and the hybrid beat popularity by 15–16% NDCG@10.** The paired 95% bootstrap
  intervals of the gain (+0.022 to +0.040 for ALS) exclude zero, and ALS wins in every
  user-history segment. Their hit-rate gains are significant too.
- **The hybrid's edge over ALS (+0.6% NDCG) is within noise.**
- **Item-kNN gains 4% NDCG over popularity, which is not significant.** Its hit rate is lower.
- **Explicit MF has the best RMSE but the worst ranking.** RMSE is 0.891 vs 0.955 for the damped
  item mean (difference −0.064, 95% CI −0.081 to −0.049), but NDCG@10 is only 0.044. Optimising
  rating error is not the same as ranking well.

Full tables (K = 5 and 10, novelty, confidence intervals, user segments):
[`reports/evaluation/model_comparison.md`](reports/evaluation/model_comparison.md).

## Where each deliverable is

| Deliverable | Location |
|---|---|
| Source code, organised by stage | [`src/`](src) (one package per stage) |
| Setup & execution instructions | this README |
| Dependencies | [`requirements.txt`](requirements.txt) (tested ranges), [`requirements-lock.txt`](requirements-lock.txt) (exact versions of the reference run) |
| Architecture diagram | [`reports/figures/architecture.png`](reports/figures/architecture.png) (source: `architecture.mmd`) |
| Data validation report & logs | [`reports/validation/validation_report.md`](reports/validation/validation_report.md), [`check_results.csv`](reports/validation/check_results.csv), [`quarantined_records.csv`](reports/validation/quarantined_records.csv), [`reports/logs/reference_run.log`](reports/logs/reference_run.log); defect demo: [`validation_demo_corrupted_sample.md`](reports/validation/validation_demo_corrupted_sample.md) |
| Feature catalog (table) | [`reports/feature_catalog.md`](reports/feature_catalog.md) / [`.csv`](reports/feature_catalog.csv) |
| Model evaluation & comparison | [`reports/evaluation/model_comparison.md`](reports/evaluation/model_comparison.md), [`.png`](reports/evaluation/model_comparison.png), [`sample_recommendations.md`](reports/evaluation/sample_recommendations.md), tuning grids `tuning_*.csv` |
| Orchestration workflow file | [`dags/recommendation_flow.py`](dags/recommendation_flow.py) |
| Screenshot of a successful run | [`reports/screenshots/prefect_flow_run.png`](reports/screenshots/prefect_flow_run.png) (+ task runs, artifacts, MLflow) and one per pipeline stage in [`reports/screenshots/stages/`](reports/screenshots/stages/); see [Screenshots of the successful run](#screenshots-of-the-successful-run) |
| Final report (2–3 pages) | [`reports/DM4ML_2025AG05573_ASSIGNMENT_2.pdf`](reports/DM4ML_2025AG05573_ASSIGNMENT_2.pdf): 3-page report, plus an appendix with the run and stage screenshots |

## Architecture

![Pipeline architecture](reports/figures/architecture.png)

| # | Stage | Module | Main outputs |
|---|---|---|---|
| 1 | Ingestion | `src/ingestion/ingest.py` | `data/raw/bronze/*.parquet` (raw strings + lineage), `reports/ingestion_manifest.json` (source, checksums checked against pinned values, row counts) |
| 2 | Validation | `src/validation/{schemas,validate}.py` | `data/processed/validated/`, `data/errors/*_errors.parquet`, `data/errors/warnings.parquet`, validation report, `quarantined_records.csv`; quality gate |
| 3 | Preparation | `src/preparation/{prepare,split}.py` | `data/processed/clean/`, `data/processed/splits/{train,val,test}.parquet` |
| 4 | Transformation | `src/transformation/transform.py` | user activity summary, item popularity, genre summary, user×item matrices (`data/processed/derived/`) |
| 5 | Feature management | `src/features/{definitions,build,store,catalog}.py` | `data/features/<group>/<snapshot>.parquet`, `registry.json`, feature catalog |
| 6 | Training | `src/training/` | `models/*.pkl`, `models/*.meta.json`, `models/best_params.json`, MLflow runs |
| 7 | Evaluation | `src/evaluation/{metrics,protocol,evaluate}.py` | `reports/evaluation/*` |
| – | Serving | `src/inference/recommend.py` | `data/processed/recommendations/hybrid_top10.parquet`, CLI; popularity fallback for users the models were not trained on |
| – | Orchestration | `dags/recommendation_flow.py` | Prefect flow (14 tasks), optional weekly schedule |

## Setup

Requires **Python 3.11–3.13**. It was tested on 3.11 with the minimum versions in
`requirements.txt`, and on 3.12 with the exact versions in `requirements-lock.txt`; both give
identical metrics. No compilers or GPU are needed: the models are written in NumPy/SciPy, so
every dependency installs from wheels on macOS (Intel/Apple Silicon), Linux and Windows.

```bash
git clone <repo-url> recommendation-pipeline && cd recommendation-pipeline
python3 -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt          # tested version ranges
# or, for the exact package versions of the reference run (Python 3.12+):
pip install -r requirements-lock.txt
```

The dataset is **not** in the repository (the MovieLens licence does not allow
redistribution). The ingestion stage downloads it from GroupLens
(`https://files.grouplens.org/datasets/movielens/ml-1m.zip`, ~6 MB). Offline? Download
the zip yourself and point the pipeline at it:

```bash
export ML1M_ZIP=/path/to/ml-1m.zip         # or set ingestion.local_zip in configs/pipeline.yaml
```

Ingestion checks the MD5 of each extracted `.dat` file against `ingestion.expected_file_md5`.
The pins are on the files rather than the zip, so a re-packed archive with the same content still
matches. With the default `checksum_policy: warn`, a mismatch is logged and recorded in the
manifest. Set `checksum_policy: fail` to stop the run instead.

## Running

```bash
# Full pipeline as a Prefect flow (with hyper-parameter tuning, ~8-10 min)
python -m dags.recommendation_flow

# Faster: skip the grid search and use the tuned values stored in the config (~1-2 min)
python -m dags.recommendation_flow --skip-tuning

# The same stages as plain sequential scripts (no Prefect)
make stages            # or run each: python -m src.ingestion.ingest, python -m src.validation.validate, ...

# Unit tests (validation rules, checksums, metrics, time split, point-in-time features, models)
python -m pytest -q

# Validation report on a deliberately corrupted sample (shows every error check firing)
python -m scripts.validation_demo

# Recommendations for a user, with an explanation of the hybrid score
python -m src.inference.recommend --user-id 42
python -m src.inference.recommend --user-id 42 --model item_knn --k 5
```

A `Makefile` wraps these (`make setup`, `make run`, `make run-fast`, `make test`, `make recommend USER_ID=42`).

### Seeing the run in the Prefect and MLflow UIs

```bash
prefect server start                                   # terminal 1 -> http://127.0.0.1:4200
export PREFECT_API_URL=http://127.0.0.1:4200/api       # terminal 2
python -m dags.recommendation_flow
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --port 5000   # -> http://127.0.0.1:5000
```

Without `PREFECT_API_URL`, Prefect starts a temporary local server just for the run, so no
setup is needed. To run on a schedule, `python -m dags.recommendation_flow --serve` registers
a deployment that retrains weekly (Mondays 02:00; change with `--cron`).

`scripts/capture_screenshots.py` regenerates the screenshots in `reports/screenshots/`, including
one per pipeline stage (`--only stages` for just those), and
`scripts/build_report.py` rebuilds the report PDF (both need
`pip install playwright pypdf && playwright install chromium`).

## Screenshots of the successful run

All screenshots come from the same reference run (Prefect flow run `slim-shark`, 14 tasks,
all *Completed*). Each stage screenshot is that task's page in the Prefect UI: its state,
duration and the log lines the stage wrote.

**Whole flow:** run graph with every task, and the MLflow experiment.

![Prefect flow run](reports/screenshots/prefect_flow_run.png)
![Prefect task runs](reports/screenshots/prefect_task_runs.png)

| Stage (from the brief) | Prefect task | Screenshot |
|---|---|---|
| Data ingestion | `ingest` | [01_ingest.png](reports/screenshots/stages/01_ingest.png) |
| Data validation | `validate` | [02_validate.png](reports/screenshots/stages/02_validate.png) |
| Data preparation | `prepare`, `time-split` | [03_prepare.png](reports/screenshots/stages/03_prepare.png), [04_time-split.png](reports/screenshots/stages/04_time-split.png) |
| Data transformation | `transform` | [05_transform.png](reports/screenshots/stages/05_transform.png) |
| Feature management | `build-features` | [06_build-features.png](reports/screenshots/stages/06_build-features.png) |
| Model training | `tune-hyperparameters`, five `train-*` tasks | [07_tune-hyperparameters.png](reports/screenshots/stages/07_tune-hyperparameters.png), [08_train-popularity.png](reports/screenshots/stages/08_train-popularity.png), [09_train-explicit_mf.png](reports/screenshots/stages/09_train-explicit_mf.png), [10_train-item_knn.png](reports/screenshots/stages/10_train-item_knn.png), [11_train-implicit_als.png](reports/screenshots/stages/11_train-implicit_als.png), [12_train-hybrid.png](reports/screenshots/stages/12_train-hybrid.png), [MLflow runs](reports/screenshots/mlflow_runs.png), [MLflow charts](reports/screenshots/mlflow_charts.png) |
| Model evaluation | `evaluate` | [13_evaluate.png](reports/screenshots/stages/13_evaluate.png), [Prefect artifacts](reports/screenshots/prefect_artifacts.png) |
| Serving (batch recommendations) | `batch-recommend` | [14_batch-recommend.png](reports/screenshots/stages/14_batch-recommend.png) |
| Orchestration | the flow itself | [prefect_flow_run.png](reports/screenshots/prefect_flow_run.png), [prefect_runs.png](reports/screenshots/prefect_runs.png) |

<details>
<summary>Show every stage screenshot</summary>

**1 · Ingestion**
![Ingestion](reports/screenshots/stages/01_ingest.png)

**2 · Validation**
![Validation](reports/screenshots/stages/02_validate.png)

**3 · Preparation**
![Preparation](reports/screenshots/stages/03_prepare.png)
![Time split](reports/screenshots/stages/04_time-split.png)

**4 · Transformation**
![Transformation](reports/screenshots/stages/05_transform.png)

**5 · Feature management**
![Feature build](reports/screenshots/stages/06_build-features.png)

**6 · Model training**
![Tuning](reports/screenshots/stages/07_tune-hyperparameters.png)
![Train popularity](reports/screenshots/stages/08_train-popularity.png)
![Train explicit MF](reports/screenshots/stages/09_train-explicit_mf.png)
![Train item-kNN](reports/screenshots/stages/10_train-item_knn.png)
![Train implicit ALS](reports/screenshots/stages/11_train-implicit_als.png)
![Train hybrid](reports/screenshots/stages/12_train-hybrid.png)
![MLflow runs](reports/screenshots/mlflow_runs.png)

**7 · Evaluation**
![Evaluation](reports/screenshots/stages/13_evaluate.png)
![Prefect artifacts](reports/screenshots/prefect_artifacts.png)

**8 · Serving**
![Batch recommendations](reports/screenshots/stages/14_batch-recommend.png)

</details>

## Design summary

**Ingestion.** Source files are parsed with every column kept as a string ("bronze"), so type
problems surface in validation instead of being silently coerced on read. Lines with the
wrong number of fields go to `data/errors/malformed_lines.parquet`. The manifest records the
archive MD5 and each source file's MD5, which is checked against the pinned value.

**Validation.** A Pandera schema per table (types, ranges, allowed values, formats) plus custom
checks for missing values, un-parseable types, duplicate keys, referential integrity
(ratings → users/movies), timestamps (after MovieLens launched, not in the future, not
before the movie's release), genre vocabulary and row-count sanity — 41 checks. Each has a
severity. **Error** rows are quarantined to `data/errors/` with the failed check ids, and
[`quarantined_records.csv`](reports/validation/quarantined_records.csv) lists them by id
(table, source line, ids and checks, but no data values, because the licence does not allow
redistribution). **Warning** rows are kept, logged and repaired in preparation. The flow stops if more than 1%
of any table is quarantined. MovieLens 1M is clean, so the real report shows only 81 ZIP
code warnings; [`validation_demo_corrupted_sample.md`](reports/validation/validation_demo_corrupted_sample.md)
shows the same code catching nulls, duplicates, out-of-range ratings, future timestamps,
orphan ids and bad types (`python -m scripts.validation_demo`).

**Preparation.** Repeated (user, movie) ratings keep the latest; missing categorical
attributes become `unknown`; timestamps become UTC datetimes; coded columns become labels; ZIP+4
is truncated to 5 digits; titles are split into title + release year with trailing articles
moved to the front ("Matrix, The (1999)" → "The Matrix", 1999); genres are one-hot encoded.

**Time-based split.** One global cut-off, so nothing the model is trained on happens after
anything it is tested on: train = first 80% of ratings by time, validation = next 10% (used
only for tuning), test = last 10%. Final models are refitted on train + validation.

**Feature store.** A small file-based store (a lightweight alternative to Feast).
`definitions.py` is the single registry of 32 features (user, item and user–item
interaction features), each with entity, type, source, transformation and consumers. Features
are materialised *as of* a timestamp using only earlier interactions, as three snapshots:
`tune`, `final` and `serving`. Training and inference read the same definitions through the
same `FeatureStore` class, which avoids training/serving skew. `ui_genre_match` is an
on-demand feature computed at request time.

**Models.**
- *Popularity* — most-rated unseen items; damped (Bayesian) item mean for rating prediction.
- *Explicit MF* — biased matrix factorisation fitted with ALS and weighted-λ regularisation;
  predicts ratings → RMSE/MAE. Plain ridge regularisation is kept as an option and as a
  reference trial in the tuning table, where it is slightly worse on RMSE. A `factor_term_std`
  diagnostic, logged with every fit, shows the latent factors are actually used.
- *Item-kNN* — item-based CF: cosine similarity of co-rated movies, shrunk towards zero for
  pairs with few common raters, keeping each movie's top neighbours.
- *Implicit ALS* — implicit-feedback MF with confidence `1 + α·rating`, solved with a
  vectorised conjugate-gradient ALS.
- *Hybrid* — `z(ALS score) + w_content · genre_match + w_pop · z(log popularity)` with weights
  tuned on the validation window; content and popularity terms come from the feature store.

**Tuning.** Every grid is searched on the validation window only (NDCG@10; RMSE for explicit MF).
The grids are wide enough that each selected value lies inside its range; the tables are in
`reports/evaluation/tuning_*.csv`. Implicit ALS uses a two-stage search:
1. λ × α at 32 factors;
2. then the number of factors at the best pair, keeping the smallest model within 2% of the best
   score.

The search picks 8 factors (NDCG@10 0.144 vs the best, 0.145 at 16 factors). Strong regularisation
is what makes ALS work here: with small λ it ranks below popularity. The validation window holds
only 492 evaluable users, so differences of about 2% are noise.

**Evaluation.** Precision, Recall, NDCG and HitRate at K = 5 and 10, plus catalogue coverage,
intra-list genre diversity, novelty as mean self-information, and RMSE/MAE for the rating models. Paired 95%
bootstrap confidence intervals (users resampled) compare each model with
the baseline, for the ranking metrics and for RMSE/MAE. Results are also broken down by how many ratings a user had before
the test window.

**Tracking.** MLflow (SQLite backend) records every tuning trial as a nested run (73 trials) and
every final model as its own run. Each run holds the parameters, data checksums, feature-registry
version, fit time, fit diagnostics, pickled model and test metrics.

**Orchestration.** Prefect flow with 14 tasks: retries on ingestion, the validation quality
gate stops downstream tasks, independent models train concurrently, and the hybrid waits for
the ALS model. Validation and evaluation summaries are published as Prefect artifacts.
Every task reads and writes files, so any stage can be re-run on its own.

**Serving.** `src/inference/recommend.py` scores users with the chosen model (hybrid by
default) and never recommends a movie the user has already rated. A user is scored by the
model only if they have a rating in its training window (train + validation). Users whose first
rating came later (29 of 6,040), and unknown ids, get the popularity fallback. The batch log
reports both counts.

**Reproducibility.** One config file (`configs/pipeline.yaml`), fixed seeds, a deterministic
time split, pinned source-file checksums, a versioned feature registry and a lock file with every
package version. Results also don't depend on library internals. Sparse matrices are kept in
canonical order, and tied scores are broken by item id. A run on Python 3.11 with the minimum
versions gives the same metrics as one on Python 3.12 with the locked versions (differences
below 1e-14).

## Repository layout

```
recommendation-pipeline/
├── configs/pipeline.yaml        all parameters (paths, checks, split, features, models, grids)
├── dags/recommendation_flow.py  Prefect flow (orchestration)
├── src/
│   ├── common/                  config, logging, I/O helpers
│   ├── ingestion/               download + bronze landing + manifest
│   ├── validation/              Pandera contracts, checks, quarantine, report, quality gate
│   ├── preparation/             cleaning/standardisation, time-based split
│   ├── transformation/          derived datasets (pure, as_of-aware functions)
│   ├── features/                registry, point-in-time build, store, catalog
│   ├── training/                models, tuning, MLflow tracking
│   ├── evaluation/              metrics, protocol, bootstrap CIs, comparison report
│   └── inference/               batch / on-demand recommendations
├── tests/                       unit tests (sample tables: src/validation/samples.py)
├── scripts/                     validation demo, screenshot capture (whole run + each stage), report builder
├── data/{raw,processed,features,errors}/   generated (git-ignored)
├── models/                      generated; metadata + best params committed
└── reports/                     validation, feature catalog, evaluation, figures, screenshots, logs, final report (PDF)
```

## Dataset

F. Maxwell Harper and Joseph A. Konstan. 2015. *The MovieLens Datasets: History and Context.*
ACM TiiS 5(4). MovieLens 1M: 1,000,209 ratings (1–5) from 6,040 users on 3,883 movies,
April 2000 – February 2003, with user demographics and movie genres.
