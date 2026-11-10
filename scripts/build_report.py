"""Build the final report PDF (reports/DM4ML_2025AG05573_ASSIGNMENT_2.pdf) from the pipeline's outputs.

Every number in the report is read from the JSON/CSV files the pipeline writes, so
the report always matches the latest run. Needs Playwright (and pypdf for the PDF metadata):

    pip install playwright pypdf && playwright install chromium
    python -m scripts.build_report
"""
from __future__ import annotations

import base64
from pathlib import Path

import pandas as pd

from src.common.config import PROJECT_ROOT, load_config
from src.common.io import read_json

R = PROJECT_ROOT / "reports"
AUTHOR = "Divya Sharma"
STUDENT_ID = "2025AG05573"
COURSE = "AIMLZG529 &middot; Data Management for Machine Learning"
REPO_URL = "https://github.com/2025ag05573/DM4MLAssignment2"
PDF_NAME = f"DM4ML_{STUDENT_ID}_ASSIGNMENT_2.pdf"


def img(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def shot(path: Path, box: tuple[int, int, int, int] | None = None, width: int = 1400, panel_bottom: bool = False,
         max_height: int | None = None) -> str:
    """A screenshot as a data URI, optionally cropped and downscaled (keeps the PDF small).
    ``panel_bottom`` trims the crop just below the Prefect log panel."""
    import io

    import numpy as np
    from PIL import Image

    im = Image.open(path).convert("RGB")
    if box is not None:
        x0, y0, x1, y1 = box
        if panel_bottom:
            a = np.asarray(im).astype(int)
            panel = a[500, 1200]                                   # colour inside the log panel
            rows = np.where(np.abs(a[:, 1600] - panel).sum(axis=1) < 12)[0]
            y1 = min(int(rows.max()) + 24, a.shape[0]) if len(rows) else y1
        if max_height:
            y1 = min(y1, y0 + max_height)
        im = im.crop((x0, y0, x1, y1))
    if im.width > width:
        im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def stage_shot(path: Path, width: int = 1200, max_panel: int = 760) -> str:
    """A Prefect task-run screenshot reduced to what matters: the title bar (task name and
    state) stacked on the log panel. Navigation, tabs and the settings sidebar are dropped."""
    import io

    import numpy as np
    from PIL import Image

    im = Image.open(path).convert("RGB")
    a = np.asarray(im).astype(int)
    panel = a[500, 1200]                                           # colour inside the log panel
    rows = np.where(np.abs(a[:, 1600] - panel).sum(axis=1) < 12)[0]
    top, bottom = int(rows.min()), int(rows.max())
    x0, x1 = 540, 2640                                             # log panel's horizontal extent
    title = im.crop((x0, 30, x1, 115))
    logs = im.crop((x0, top + 40, x1, min(bottom, top + 40 + max_panel)))   # skip the date divider
    out = Image.new("RGB", (x1 - x0, title.height + logs.height + 8), tuple(int(v) for v in a[1500, 1200]))
    out.paste(title, (0, 0))
    out.paste(logs, (0, title.height + 8))
    out = out.resize((width, round(out.height * width / out.width)), Image.LANCZOS)
    buf = io.BytesIO()
    out.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


STAGES = [  # (file stem, brief stage, task)
    ("01_ingest", "Data ingestion", "ingest"),
    ("02_validate", "Data validation", "validate"),
    ("03_prepare", "Data preparation", "prepare"),
    ("04_time-split", "Data preparation: time-based split", "time-split"),
    ("05_transform", "Data transformation", "transform"),
    ("06_build-features", "Feature management", "build-features"),
    ("07_tune-hyperparameters", "Model training: tuning (first of 73 trials shown)", "tune-hyperparameters"),
    ("08_train-popularity", "Model training: popularity", "train-popularity"),
    ("09_train-explicit_mf", "Model training: explicit MF", "train-explicit_mf"),
    ("10_train-item_knn", "Model training: item-kNN", "train-item_knn"),
    ("11_train-implicit_als", "Model training: implicit ALS", "train-implicit_als"),
    ("12_train-hybrid", "Model training: hybrid", "train-hybrid"),
    ("13_evaluate", "Model evaluation", "evaluate"),
    ("14_batch-recommend", "Serving: batch recommendations", "batch-recommend"),
]


def appendix_html() -> str:
    sdir = R / "screenshots"
    cells = []
    for i, (stem, stage, task) in enumerate(STAGES, start=1):
        uri = stage_shot(sdir / "stages" / f"{stem}.png")
        cells.append(f'<figure class="shot"><img src="{uri}" alt="{stage}">'
                     f'<figcaption>A.{i + 1} {stage} &ndash; task <code>{task}</code></figcaption></figure>')
    flow = shot(sdir / "prefect_flow_run.png", box=(520, 0, 3200, 1010), width=1800)
    mlf = shot(sdir / "mlflow_runs.png", box=(470, 360, 2700, 920), width=1200)
    cells.append(f'<figure class="shot"><img src="{mlf}" alt="MLflow runs"><figcaption>A.{len(STAGES) + 2} '
                 f'Experiment tracking &ndash; the five final models and the tuning parent run in MLflow.</figcaption></figure>')
    return f"""
<section class="appendix">
<h2><span class="no">A</span> Appendix: screenshots of the successful run</h2>
<p class="small">Run evidence, not part of the 3-page report. All screenshots come from one reference run (Prefect flow
run <i>slim-shark</i>, 14 tasks, all <i>Completed</i>). Each stage screenshot is that task's page in the Prefect UI,
showing its state and the log lines it wrote. Full-size images are in <code>reports/screenshots/</code>.</p>
<figure class="wide"><img src="{flow}" alt="Prefect flow run">
<figcaption>A.1 Orchestration &ndash; the whole flow run: task graph, timeline and status.</figcaption></figure>
<div class="grid">{"".join(cells)}</div>
</section>"""


def pct(new: float, old: float) -> str:
    d = (new / old - 1) * 100
    return f"{d:+.1f}%" if abs(d) < 10 else f"{d:+.0f}%"


def popularity_concentration(top_n: int = 100) -> tuple[float, float]:
    """Share of the recommendable catalogue that the top-N fit-window titles make up,
    and the share of test-window ratings they receive."""
    sp = PROJECT_ROOT / "data" / "processed" / "splits"
    fit = pd.concat([pd.read_parquet(sp / f"{p}.parquet", columns=["movie_id"]) for p in ("train", "val")])
    test = pd.read_parquet(sp / "test.parquet", columns=["movie_id"])
    top = set(fit["movie_id"].value_counts().index[:top_n])
    return top_n / fit["movie_id"].nunique(), float(test["movie_id"].isin(top).mean())


def build_html() -> str:
    cfg = load_config()
    val = read_json(R / "validation" / "validation_report.json")
    prep = read_json(R / "preparation_summary.json")
    split = read_json(R / "split_summary.json")
    ev = read_json(R / "evaluation" / "model_comparison.json")
    best = read_json(PROJECT_ROOT / "models" / "best_params.json")
    reg = read_json(PROJECT_ROOT / "data" / "features" / "registry.json")
    tun = {m: pd.read_csv(R / "evaluation" / f"tuning_{m}.csv")
           for m in ("implicit_als", "explicit_mf", "item_knn", "hybrid")}
    n_trials = sum(len(t) for t in tun.values())
    res, proto = ev["results"], ev["protocol"]
    pop, mf, knn, als, hyb = (res[m] for m in ("popularity", "explicit_mf", "item_knn", "implicit_als", "hybrid"))
    L = {"pop": "Popularity (baseline)", "mf": "Explicit MF (biased ALS)", "knn": "Item-kNN CF",
         "als": "Implicit ALS", "hyb": "Hybrid (ALS + features)"}
    boot = {r["model"]: r for r in ev["bootstrap_ndcg"]}
    boot_hr = {r["model"]: r for r in ev["bootstrap_hit_rate"]}
    boot_rt = {(r["model"], r["metric"]): r for r in ev["bootstrap_rating_error"]}
    seg = pd.DataFrame(ev["segments"])
    seg_names = list(dict.fromkeys(seg["segment"]))
    als_wins_all = all(seg[(seg.segment == g) & (seg.model == L["als"])]["ndcg@10"].iloc[0]
                       > seg[(seg.segment == g) & (seg.model == L["pop"])]["ndcg@10"].iloc[0] for g in seg_names)
    ci = lambda name, b=boot: f"{b[name]['diff_ci_low']:+.4f} to {b[name]['diff_ci_high']:+.4f}"  # noqa: E731
    als_t = tun["implicit_als"]
    als_sel = als_t[als_t["selected"]].iloc[0]
    als_top = als_t[als_t["stage"] == 2]["ndcg@10"].max()
    mf_ref = tun["explicit_mf"][tun["explicit_mf"]["reg_mode"] == "unweighted"].iloc[0]
    mf_sel = tun["explicit_mf"]["rmse"].min()
    n_tests = sum(f.read_text().count("\ndef test_") for f in (PROJECT_ROOT / "tests").glob("test_*.py"))
    man = read_json(R / "ingestion_manifest.json")
    pinned = man["content_checksum"]["status"]
    log_txt = (R / "logs" / "reference_run.log").read_text() if (R / "logs" / "reference_run.log").exists() else ""
    import re
    fb = re.search(r"(\d+) users scored by \w+, (\d+) on the popularity fallback", log_txt)
    n_fallback = int(fb.group(2)) if fb else None
    stamps = re.findall(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)", log_txt, flags=re.M)
    RUN_MINUTES = (round((pd.Timestamp(stamps[-1]) - pd.Timestamp(stamps[0])).total_seconds() / 60)
                   if len(stamps) > 1 else "ten")
    tun_users = read_json(R / "evaluation" / "tuning_protocol.json")["eval_users"]
    checks = val["checks"]
    n_checks = len(checks)
    n_warn = sum(1 for c in checks if c["status"] == "WARN")
    quarantined = sum(s["rows_quarantined"] for s in val["summary"].values())
    sp = split["parts"]
    cat_share, test_share = popularity_concentration(100)
    a, h, m, k = best["implicit_als"], best["hybrid"], best["explicit_mf"], best["item_knn"]
    from src.features.definitions import FEATURES
    n_feat = {g: sum(1 for f in FEATURES if f.group == g)
              for g in ("user_features", "item_features", "interaction_features")}

    metric_keys = ["precision@10", "recall@10", "ndcg@10", "hit_rate@10", "coverage@10", "diversity@10",
                   "novelty@10", "rmse", "mae"]
    best_of = {key: (min if key in ("rmse", "mae") else max)(r[key] for r in res.values() if key in r)
               for key in metric_keys}

    def row(name, r):
        bold = [key for key in metric_keys if key in r and round(r[key], 3) == round(best_of[key], 3)]

        def cell(key, fmt="{:.3f}"):
            v = r.get(key)
            s = "&mdash;" if v is None else fmt.format(v)
            return f"<td class='num{' best' if key in bold else ''}'>{s}</td>"
        return (f"<tr><td>{name}</td>" + cell("precision@10") + cell("recall@10") + cell("ndcg@10")
                + cell("hit_rate@10") + cell("coverage@10") + cell("diversity@10") + cell("novelty@10", "{:.1f}")
                + cell("rmse") + cell("mae") + "</tr>")

    table = "".join([row(L["pop"], pop), row(L["mf"], mf), row(L["knn"], knn), row(L["als"], als),
                     row(L["hyb"], hyb)])
    t_val, t_test = split["cutoffs"]["t_val"][:10], split["cutoffs"]["t_test"][:10]

    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>DM4ML_{STUDENT_ID}_ASSIGNMENT_2</title>
<style>
@page {{ size: A4; margin: 15mm 16mm 16mm 16mm; }}
:root {{ --ink:#161616; --muted:#57534c; --rule:#b9b4aa; --accent:#6d1f2a; --soft:#f5f2ec; }}
* {{ box-sizing: border-box; }}
body {{ font-family: "Bitstream Charter", Charter, Lora, Georgia, serif; color: var(--ink); font-size: 8.9pt;
        line-height: 1.36; margin: 0; background:#fff; text-align: justify; hyphens: auto; }}
.titleblock {{ text-align: center; border-top: 2.2px solid var(--ink); border-bottom: 0.8px solid var(--ink);
              padding: 5px 0 6px; margin-bottom: 7px; }}
.inst {{ font-variant: small-caps; letter-spacing: 0.07em; font-size: 8.4pt; color: var(--muted); }}
.titleblock h1 {{ font-size: 14.2pt; line-height: 1.2; margin: 3px 0 4px; font-weight: 700; }}
.meta {{ display: flex; justify-content: center; flex-wrap: wrap; gap: 3px 16px; font-size: 8.3pt; }}
.meta span b {{ font-variant: small-caps; letter-spacing: 0.04em; color: var(--accent); font-weight: 700; }}
.abstract {{ margin: 0 9mm 6px; font-size: 8.7pt; }}
.abstract b.lab {{ font-variant: small-caps; letter-spacing: 0.05em; color: var(--accent); }}
h2 {{ font-family: Inter, "Helvetica Neue", Arial, sans-serif; font-size: 9.4pt; text-transform: uppercase;
      letter-spacing: 0.06em; color: var(--accent); margin: 9px 0 3px; text-align: left; }}
h2 .no {{ display: inline-block; min-width: 14px; }}
h3 {{ font-size: 9.1pt; font-style: italic; font-weight: 700; margin: 6px 0 2px; text-align: left; }}
p {{ margin: 0 0 4px; text-indent: 0; }}
ul {{ margin: 2px 0 4px 0; padding-left: 14px; }} li {{ margin-bottom: 2px; }}
code {{ font-family: "DejaVu Sans Mono", Menlo, monospace; font-size: 7.5pt; }}
table {{ border-collapse: collapse; width: 100%; font-family: Inter, "Helvetica Neue", Arial, sans-serif;
        font-size: 7.6pt; margin: 3px 0 4px; text-align: left; border-top: 1.2px solid var(--ink);
        border-bottom: 1.2px solid var(--ink); }}
th, td {{ padding: 2.2px 4px; vertical-align: top; }}
th {{ font-weight: 600; border-bottom: 0.6px solid var(--ink); }}
tr:nth-child(even) td {{ background: var(--soft); }}
td.num, th.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
td.best {{ font-weight: 700; color: var(--accent); }}
.two {{ display: grid; grid-template-columns: 1fr 57mm; gap: 5mm; align-items: start; }}
figure {{ margin: 0; }}
figcaption {{ font-size: 7.5pt; font-style: italic; color: var(--muted); margin-top: 2px; text-align: left; }}
.arch img {{ width: 100%; }}
.chart {{ text-align: center; }} .chart img {{ width: 80%; }}
.small {{ font-size: 7.6pt; color: var(--muted); text-align: left; }}
.avoid {{ break-inside: avoid; }}
.appendix {{ break-before: page; }}
.appendix figure {{ break-inside: avoid; margin-bottom: 5px; }}
.appendix img {{ width: 100%; border: 0.6px solid var(--rule); }}
.grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 4mm 5mm; margin: 4px 0; }}
</style></head><body>

<div class="titleblock">
<div class="inst">Birla Institute of Technology &amp; Science, Pilani &middot; Work Integrated Learning Programmes</div>
<div class="inst">{COURSE} &middot; Assignment 2</div>
<h1>End-to-End Data Management Pipeline for a Recommendation System</h1>
<div class="meta"><span><b>Name</b> {AUTHOR}</span><span><b>ID</b> {STUDENT_ID}</span>
<span><b>Dataset</b> MovieLens 1M</span><span><b>Repository</b> {REPO_URL.replace("https://", "")}</span></div>
</div>

<p class="abstract"><b class="lab">Abstract.</b> This report describes a reproducible pipeline that ingests MovieLens 1M,
validates it against a data contract, prepares and transforms it, serves point-in-time
features from a small feature store, tunes and trains five recommenders with MLflow tracking, evaluates them on a
future time window with bootstrap confidence intervals and runs end-to-end as one Prefect flow. On the test window
implicit ALS and the hybrid beat popularity by <b>{pct(als['ndcg@10'], pop['ndcg@10'])}&ndash;{pct(hyb['ndcg@10'], pop['ndcg@10'])[1:]} NDCG@10</b>
(95% intervals exclude zero) and cover <b>{als['coverage@10'] / pop['coverage@10']:.1f}&times;</b> more of the catalogue.</p>

<h2><span class="no">1</span> Problem, data and architecture</h2>
<div class="two">
<div>
<p>The task is to recommend movies a user will like from their past ratings and item information.
<b>MovieLens 1M</b> has all three required entities: {val['summary']['ratings']['rows_in']:,} explicit
ratings (1&ndash;5) from {val['summary']['users']['rows_in']:,} users on {val['summary']['movies']['rows_in']:,}
movies (Apr 2000&ndash;Feb 2003), with user demographics (gender, age bucket, occupation, ZIP) and movie
titles and genres. It is large enough for realistic results, yet the full pipeline, including tuning,
runs in about {RUN_MINUTES} minutes (reference run, one CPU-only machine).</p>
<table>
<tr><th>Concern</th><th>Choice</th><th>Why</th></tr>
<tr><td>Storage</td><td>Parquet, layered raw &rarr; validated &rarr; clean &rarr; derived &rarr; features</td><td>Typed, columnar, no server; every layer can be inspected</td></tr>
<tr><td>Validation</td><td>Pandera schemas + custom pandas checks</td><td>Contracts as code; lighter than Great Expectations</td></tr>
<tr><td>Features</td><td>Custom file-based feature store</td><td>Feast's core ideas (registry, entities, point-in-time snapshots) without its infrastructure</td></tr>
<tr><td>Models</td><td>NumPy / SciPy ALS implementations</td><td>No compiled dependencies, so it installs anywhere; fully transparent</td></tr>
<tr><td>Tracking</td><td>MLflow (SQLite backend)</td><td>Params, metrics, data checksum, feature version, model artifacts</td></tr>
<tr><td>Orchestration</td><td>Prefect 3</td><td>Pure Python, runs with or without a server, retries, concurrency, schedules</td></tr>
<tr><td>Quality</td><td>pytest ({n_tests} tests), one YAML config, fixed seeds, lock file</td><td>Validation rules, checksums, metrics, leakage and models tested</td></tr>
</table>
</div>
<figure class="arch"><img src="{img(R / 'figures' / 'architecture.png')}" alt="Pipeline architecture">
<figcaption>Figure 1. Pipeline architecture. Every stage is a Prefect task that reads and writes files,
so each can be re-run on its own.</figcaption></figure>
</div>

<h2><span class="no">2</span> Design decisions</h2>
<h3>2.1 Ingestion and validation</h3>
<p>Source files land as <i>bronze</i> Parquet with every column kept as a string plus lineage columns, so type
problems are caught by validation instead of being silently coerced on read. A manifest records the archive MD5
and the MD5 of each source file, which are checked against pinned values (here: {", ".join(f"{t} {st}" for t, st in pinned.items())};
a mismatch is logged, or stops the run with <code>checksum_policy: fail</code>). Validation runs <b>{n_checks} checks</b> in three groups: a Pandera contract per table
(types, ranges, allowed values, formats); integrity checks (missing keys, un-parseable values, duplicate keys,
referential integrity from ratings to users and movies); and plausibility checks (timestamps after MovieLens
launched and not in the future, ratings not before the movie's release year, genre vocabulary, row volumes).
Each check has a severity. <b>Error</b> rows are quarantined to <code>data/errors/</code>, tagged with the
checks they failed, and listed by id in the committed <code>quarantined_records.csv</code>. <b>Warning</b> rows are kept, logged, and repaired in preparation. A quality gate stops the
flow if more than {cfg['validation']['max_error_rate']:.0%} of any table is quarantined. On the real data,
{quarantined} rows were quarantined and {n_warn} check raised warnings: {val['summary']['users']['rows_with_warnings']}
users have ZIP+4 or malformed ZIP codes ({prep['users']['zip_plus4_truncated']} truncated to 5 digits,
{prep['users']['zip_invalid_set_unknown']} set to <i>unknown</i>). Because MovieLens is clean, the error path is
shown by unit tests and by a demo report in which the same code quarantines 10 corrupted rows (nulls,
duplicate keys, a rating of 7, future and pre-1997 timestamps, orphan ids, non-numeric values).</p>

<h3>2.2 Preparation and time-based split</h3>
<p>Repeated (user, movie) ratings keep the latest one. Missing categorical values become <i>unknown</i>.
Timestamps become UTC datetimes, coded age and occupation become labels, and titles are split into title and
release year with trailing articles moved to the front ("Matrix, The (1999)" &rarr; "The Matrix", 1999).
Genres are one-hot encoded. The split uses a <b>single global time cut-off</b>, so nothing used for training
happens after anything used for testing. A random or per-user split would leak future popularity trends into
training. Train is the first 80% of ratings by time (to {t_val}), validation is the next 10% (to {t_test}) and
is used only for tuning, and test is the last 10%. Final models are refitted on train + validation. MovieLens
activity is concentrated in 2000, so the test window spans two years and contains
{sp['test']['users']:,} users, {sp['test']['warm_users']:,} of whom have training history.</p>

<h3>2.3 Transformation and feature management</h3>
<p>Derived datasets (user activity summary, item popularity, genre summary, user&times;item rating matrix) are
pure functions that take an <code>as_of</code> timestamp. The feature builder reuses them, so analysis, training
and serving share one definition of, for example, "item popularity". The registry (<code>definitions.py</code>)
holds <b>{len(FEATURES)} features</b> ({n_feat['user_features']} user, {n_feat['item_features']} item,
{n_feat['interaction_features']} interaction; the two genre families expand to 18 columns each). Each feature
records its entity, type, source, transformation and consumers, and the catalog table is generated from the
registry. Features are materialised point-in-time as three snapshots: <i>tune</i> (as of the validation start),
<i>final</i> (as of the test start) and <i>serving</i> (after the last rating). The build fails if the
materialised columns drift from the registry. Training and the inference CLI both read through one
<code>FeatureStore</code> class; <code>ui_genre_match</code> is computed on demand.</p>

<h3>2.4 Models, tuning and tracking</h3>
<ul>
<li><b>Popularity</b> (baseline): the most-rated unseen items. Its rating prediction is the damped (Bayesian) item mean.</li>
<li><b>Explicit MF</b>: biased matrix factorisation fitted with ALS and weighted-&lambda;
regularisation ({m['factors']} factors, &lambda;={m['reg']}). It predicts ratings, which gives RMSE and MAE. Plain ridge is kept as a
reference trial (validation RMSE {mf_ref['rmse']:.4f} vs {mf_sel:.4f}).</li>
<li><b>Item-kNN</b>: item-based collaborative filtering, cosine similarity of co-rated movies with shrinkage (shrink {k['shrink']:g},
{k['neighbours']} neighbours per item).</li>
<li><b>Implicit ALS</b>: every rating is a positive signal with confidence 1+&alpha;&middot;rating,
solved with a vectorised conjugate-gradient ALS ({a['factors']} factors, &lambda;={a['reg']:g}, &alpha;={a['alpha']:g}).</li>
<li><b>Hybrid</b>: standardised ALS score + {h['w_content']}&middot;genre-match + {h['w_pop']}&middot;standardised
log-popularity. The content and popularity terms come from the feature store.</li>
</ul>
<p>Hyper-parameters were grid-searched on the validation window only (NDCG@10 for ranking models, RMSE for
explicit MF), with ranges wide enough that every selected value lies inside its grid. ALS uses a two-stage
search: &lambda; &times; &alpha; at {int(als_t[als_t['stage'] == 1]['factors'].iloc[0])} factors, then the number of factors at the
best pair, keeping the smallest model within 2% of the best score ({a['factors']} factors, NDCG@10
{als_sel['ndcg@10']:.4f} vs best {als_top:.4f}). Strong regularisation is what makes ALS work here: with small &lambda; it
ranks below popularity. All {n_trials} trials are nested MLflow runs. Every final model has its own run
with parameters, input checksums, feature-registry version (<code>{reg['definitions_version']}</code>), fit time and
diagnostics, the pickled model and its test metrics.</p>

<h3>2.5 Orchestration and reproducibility</h3>
<p>A Prefect flow with 14 tasks: ingest (with retries) &rarr; validate (quality gate) &rarr; prepare &rarr;
split &rarr; transform &rarr; features &rarr; tune &rarr; five training tasks (the independent models run
concurrently and the hybrid waits for ALS) &rarr; evaluate &rarr; batch recommendations. Validation and
evaluation summaries are published as Prefect artifacts, and <code>--serve</code> registers a weekly retraining
schedule. One YAML config, fixed seeds, a deterministic split, pinned input checksums and a lock file with every
package version make runs repeatable: the same metrics were reproduced on Python 3.11 (minimum dependency
versions) and 3.12 (locked versions). Serving scores users the models were trained on; the
{n_fallback if n_fallback is not None else 'few'} users whose first rating falls after the training window, and unknown ids, get the
popularity fallback.</p>

<h2><span class="no">3</span> Results</h2>
<p>Protocol: {proto['eval_users']:,} warm users with at least one relevant test item (rating &ge;
{ev['relevance_threshold']}). Candidates are the {proto['candidate_items']:,} items seen in the fit window, minus
each user's already-rated items. The {proto['cold_start_users']} cold-start users are excluded and reported.
Coverage is the share of candidates recommended to anyone;
diversity is the mean pairwise genre-cosine distance within a list; novelty is the mean
&minus;log<sub>2</sub>(popularity share). Confidence intervals use a paired bootstrap over
users. RMSE and MAE are computed on
{proto['rating_pairs']:,} held-out ratings.</p>
<table class="avoid">
<tr><th>Model (K = 10)</th><th class="num">Precision</th><th class="num">Recall</th><th class="num">NDCG</th>
<th class="num">Hit rate</th><th class="num">Coverage</th><th class="num">Diversity</th><th class="num">Novelty</th>
<th class="num">RMSE</th><th class="num">MAE</th></tr>
{table}
</table>
<figure class="chart avoid"><img src="{img(R / 'evaluation' / 'model_comparison.png')}" alt="Model comparison chart">
<figcaption>Figure 2. Test-window metrics at K = 10 (highlighted in the table = best per column). K = 5, confidence intervals and
per-segment results are in <code>reports/evaluation/model_comparison.md</code>.</figcaption></figure>
<ul>
<li><b>Personalisation helps.</b> Implicit ALS improves on popularity by {pct(als['precision@10'], pop['precision@10'])}
precision, {pct(als['recall@10'], pop['recall@10'])} recall and {pct(als['ndcg@10'], pop['ndcg@10'])} NDCG@10 (paired 95% bootstrap
interval of the gain {ci(L['als'])}){", beats it in every user-history segment" if als_wins_all else ""} and raises coverage
from {pop['coverage@10']:.1%} to {als['coverage@10']:.1%} of the catalogue. Item-kNN gains {pct(knn['ndcg@10'], pop['ndcg@10'])}
NDCG@10, which is not significant ({ci(L['knn'])}).</li>
<li><b>Popularity is a strong baseline here</b> (hit rate {pop['hit_rate@10']:.0%}): the 100 most-rated titles,
{cat_share:.1%} of the catalogue, still receive {test_share:.1%} of all test-window ratings.</li>
<li><b>Low RMSE does not mean good rankings.</b> Explicit MF has the best RMSE ({mf['rmse']:.3f} vs
{pop['rmse']:.3f}; difference {boot_rt[(L['mf'], 'rmse')]['diff_ci_low']:+.3f} to {boot_rt[(L['mf'], 'rmse')]['diff_ci_high']:+.3f},
users resampled) and MAE but the worst NDCG ({mf['ndcg@10']:.3f}). Ranking by predicted rating surfaces rarely
rated titles with high estimated means (novelty {mf['novelty@10']:.1f} vs {pop['novelty@10']:.1f} bits).</li>
<li><b>The hybrid's gain is marginal</b> ({pct(hyb['ndcg@10'], als['ndcg@10'])} NDCG over ALS, far inside the confidence
intervals; hit rate {hyb['hit_rate@10']:.3f} vs {als['hit_rate@10']:.3f}) and lowers diversity
({hyb['diversity@10']:.3f} vs {als['diversity@10']:.3f}) because it favours each user's dominant genres. Collaborative
signals already capture most of the genre information. It is kept because it shows features being reused in
training and serving, and because its score components explain each recommendation.</li>
</ul>

<h2 class="avoid"><span class="no">4</span> Limitations and future work</h2>
<ul>
<li><b>Cold start.</b> Collaborative models cannot score new users or items ({proto['cold_start_users']} test users are
excluded from evaluation). Serving falls back to popularity; a demographic or content-only model using the store's
user features would do better.</li>
<li><b>One temporal split.</b> The intervals resample users, not time; rolling-origin evaluation would show how
stable the ranking of models is across periods. The validation window is small ({tun_users} users), so tuning
choices within about 2% NDCG are noise.</li>
<li><b>Evaluation bias.</b> Unrated items count as irrelevant (missing-not-at-random), so absolute precision is
understated. Offline metrics are not a substitute for an online A/B test.</li>
<li><b>Data.</b> The data is old (2000&ndash;2003) and clean, so the error path of the validation stage was exercised
only with synthetic defects. There is no drift monitoring between runs yet.</li>
<li><b>Infrastructure.</b> The feature store is batch-only and file-based (no online low-latency store), and
MLflow and Prefect run locally without a model registry or CI. Scaling to MovieLens 25M would call for
Polars/Spark, the <code>implicit</code> library and a database-backed store.</li>
</ul>
<p class="small">Reproduce: <code>pip install -r requirements-lock.txt &amp;&amp; python -m dags.recommendation_flow</code>
(see README.md). Evidence: <code>reports/validation/</code>, <code>reports/feature_catalog.md</code>,
<code>reports/evaluation/</code>, <code>reports/screenshots/</code>.</p>

{appendix_html()}
</body></html>"""


def main() -> None:
    import tempfile

    from playwright.sync_api import sync_playwright
    from pypdf import PdfReader, PdfWriter

    out_pdf = R / PDF_NAME
    footer = (f'<div style="width:100%;font-size:7pt;color:#57534c;padding:0 16mm;display:flex;'
              f'justify-content:space-between;font-family:Georgia,serif;">'
              f'<span>{AUTHOR} &middot; {STUDENT_ID} &middot; DM4ML Assignment 2</span>'
              f'<span>Page <span class="pageNumber"></span> of <span class="totalPages"></span></span></div>')
    with tempfile.TemporaryDirectory() as tmp:          # the HTML is only an intermediate
        src = Path(tmp) / "report.html"
        src.write_text(build_html(), encoding="utf-8")
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            page.goto(src.as_uri())
            page.wait_for_timeout(500)
            page.pdf(path=str(out_pdf), format="A4", print_background=True, prefer_css_page_size=True,
                     display_header_footer=True, header_template="<span></span>", footer_template=footer)
            browser.close()
    reader, writer = PdfReader(out_pdf), PdfWriter()
    writer.append(reader)
    writer.add_metadata({"/Title": f"DM4ML Assignment 2 - {AUTHOR} ({STUDENT_ID})", "/Author": AUTHOR})
    with open(out_pdf, "wb") as fh:
        writer.write(fh)
    print(f"wrote {out_pdf.relative_to(PROJECT_ROOT)} ({len(reader.pages)} pages)")


if __name__ == "__main__":
    main()
