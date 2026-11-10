PY ?= python3
VENV ?= .venv
BIN := $(VENV)/bin

.PHONY: setup setup-locked run run-fast stages test recommend prefect-ui mlflow-ui validation-demo clean

setup:                ## create a virtualenv and install dependencies
	$(PY) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -r requirements.txt

setup-locked:         ## same, with the exact package versions of the reference run (Python 3.12+)
	$(PY) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -r requirements-lock.txt

run:                  ## full pipeline via Prefect (with hyper-parameter tuning, ~8-10 min)
	$(BIN)/python -m dags.recommendation_flow

run-fast:             ## full pipeline, tuned defaults from the config (~1-2 min)
	$(BIN)/python -m dags.recommendation_flow --skip-tuning

stages:               ## same pipeline as plain sequential scripts (no Prefect)
	$(BIN)/python -m src.ingestion.ingest
	$(BIN)/python -m src.validation.validate
	$(BIN)/python -m src.preparation.prepare
	$(BIN)/python -m src.preparation.split
	$(BIN)/python -m src.transformation.transform
	$(BIN)/python -m src.features.build
	$(BIN)/python -m src.training.train
	$(BIN)/python -m src.evaluation.evaluate

test:                 ## unit tests (validation rules, metrics, split, features, models)
	$(BIN)/python -m pytest -q

validation-demo:      ## validation report on a deliberately corrupted sample
	$(BIN)/python -m scripts.validation_demo

recommend:            ## top-10 for one user, e.g. make recommend USER_ID=42
	$(BIN)/python -m src.inference.recommend --user-id $(or $(USER_ID),42)

prefect-ui:           ## Prefect UI on http://127.0.0.1:4200 (then: export PREFECT_API_URL=http://127.0.0.1:4200/api)
	$(BIN)/prefect server start

mlflow-ui:            ## MLflow UI on http://127.0.0.1:5000
	$(BIN)/mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --port 5000

clean:                ## delete generated data, models and the tracking store
	find data -type f ! -name .gitkeep -delete
	rm -f models/*.pkl models/*.json
	rm -rf mlruns
