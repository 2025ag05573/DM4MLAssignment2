"""Capture screenshots of the latest Prefect flow run, each of its pipeline stages, and the MLflow experiment.

Optional helper (needs ``pip install playwright && playwright install chromium``).
Start both UIs first:

    prefect server start                       # http://127.0.0.1:4200
    mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --port 5000
    python -m scripts.capture_screenshots
    python -m scripts.capture_screenshots --only stages    # just the per-stage task-run pages
"""
from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request

from playwright.sync_api import sync_playwright

from src.common.config import PROJECT_ROOT, load_config

PREFECT = "http://127.0.0.1:4200"
MLFLOW = "http://127.0.0.1:5000"
OUT = PROJECT_ROOT / "reports" / "screenshots"


def _post(url: str, body: dict) -> list:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


def latest_flow_run_id() -> str:
    runs = _post(f"{PREFECT}/api/flow_runs/filter",
                 {"sort": "START_TIME_DESC", "limit": 1,
                  "flows": {"name": {"any_": ["movielens-recommendation-pipeline"]}},
                  "flow_runs": {"state": {"type": {"any_": ["COMPLETED"]}}}})
    return runs[0]["id"]


def task_runs(flow_run_id: str) -> list[dict]:
    """Task runs of a flow run, in the order they started."""
    return _post(f"{PREFECT}/api/task_runs/filter",
                 {"flow_runs": {"id": {"any_": [flow_run_id]}}, "sort": "EXPECTED_START_TIME_ASC", "limit": 100})


def capture_stages(page, run_id: str) -> None:
    """One screenshot per pipeline stage: the task run's page with its state, duration and logs."""
    out = OUT / "stages"
    out.mkdir(parents=True, exist_ok=True)
    # Prefect shows a one-off community sign-up dialog a few seconds after the first page load;
    # dismiss it before taking any screenshot.
    page.goto(f"{PREFECT}/runs/flow-run/{run_id}")
    _settle(page, 2)
    try:
        skip = page.get_by_role("button", name="Skip")
        skip.wait_for(state="visible", timeout=8000)
        skip.click()
        time.sleep(1)
    except Exception:
        pass
    for n, t in enumerate(task_runs(run_id), start=1):
        stage = re.sub(r"-[0-9a-f]{3}$", "", t["name"])          # drop Prefect's run-name suffix
        page.goto(f"{PREFECT}/runs/task-run/{t['id']}")
        _settle(page, 3)
        _click_if_visible(page.get_by_role("button", name="Skip"))
        page.screenshot(path=str(out / f"{n:02d}_{stage}.png"))


def experiment_id() -> str:
    name = load_config()["mlflow"]["experiment_name"]
    with urllib.request.urlopen(f"{MLFLOW}/api/2.0/mlflow/experiments/get-by-name?experiment_name={name}") as r:
        return json.loads(r.read())["experiment"]["experiment_id"]


def _settle(page, seconds: float) -> None:
    """Wait for the SPA to load; some UIs keep background requests open, so cap the wait."""
    try:
        page.wait_for_load_state("networkidle", timeout=10_000)
    except Exception:
        pass
    time.sleep(seconds)


def _click_if_visible(locator) -> None:
    try:
        if locator.is_visible(timeout=3000):
            locator.click()
    except Exception:
        pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["stages"], help="capture only the per-stage task-run pages")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    run_id = latest_flow_run_id()
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1600, "height": 1000}, device_scale_factor=2)
        capture_stages(page, run_id)
        if args.only == "stages":
            browser.close()
            print(f"stage screenshots -> {(OUT / 'stages').relative_to(PROJECT_ROOT)}")
            return
        exp_id = experiment_id()

        # ---- Prefect: run graph + logs, task runs, artifacts, runs list
        page.goto(f"{PREFECT}/runs/flow-run/{run_id}")
        _settle(page, 3)
        _click_if_visible(page.get_by_role("button", name="Skip"))      # community sign-up modal
        time.sleep(1)
        page.screenshot(path=str(OUT / "prefect_flow_run.png"))
        for tab, fname in (("Task Runs", "prefect_task_runs.png"), ("Artifacts", "prefect_artifacts.png")):
            page.get_by_text(tab, exact=True).first.click()
            time.sleep(3)
            page.screenshot(path=str(OUT / fname), full_page=True)
        page.goto(f"{PREFECT}/runs")
        _settle(page, 3)
        page.screenshot(path=str(OUT / "prefect_runs.png"))

        # ---- MLflow: runs table + metric charts
        page.goto(f"{MLFLOW}/#/experiments/{exp_id}/runs")
        _settle(page, 4)
        _click_if_visible(page.get_by_role("button", name="Close").last)  # assistant side panel
        time.sleep(1)
        page.screenshot(path=str(OUT / "mlflow_runs.png"))
        page.goto(f"{MLFLOW}/#/experiments/{exp_id}/runs?compareRunsMode=CHART")
        _settle(page, 5)
        page.screenshot(path=str(OUT / "mlflow_charts.png"))
        browser.close()
    print(f"screenshots -> {OUT.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
