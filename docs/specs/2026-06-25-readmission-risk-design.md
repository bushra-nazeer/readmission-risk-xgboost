# readmission-risk-xgboost — Design Spec

- **Date:** 2026-06-25
- **Status:** Approved
- **Portfolio role:** Data Scientist flagship (repo 1 of 6)

## 1. Overview

A production-grade machine-learning repository that predicts **30-day hospital
readmission risk** from patient encounter data, with full explainability,
calibration, fairness analysis, experiment tracking, and a deployable scoring
API. It is a self-contained, reproducible reference implementation built on
public data — it demonstrates the technique end-to-end; it does not reproduce
any proprietary employer system.

## 2. Resume claims this proves

| Claim (across resume variants) | How this repo demonstrates it |
|---|---|
| Patient risk prediction (XGBoost + SHAP), ~0.89 AUC | XGBoost/LightGBM models with SHAP; README reports the *real* achieved AUC |
| Explainability (SHAP, LIME) | SHAP global + local plots; one LIME example |
| Model evaluation & optimization | ROC-AUC, PR-AUC, Brier, calibration; Optuna tuning |
| MLflow, experiment tracking, model versioning | MLflow tracking of params/metrics/artifacts |
| RESTful ML APIs (FastAPI), productionized inference | FastAPI `/predict` + `/health`, Dockerized |
| HIPAA-compliant pipelines, PII handling | No real PHI; synthetic/public data; documented PII-safe design |

## 3. Dataset

- **UCI Diabetes 130-US Hospitals (1999–2008)**, ~101,766 encounters, 50 features.
- Fetched reproducibly via the `ucimlrepo` package (`fetch_ucirepo(id=296)`).
- **Target:** binary — `readmitted == "<30"` (positive, ~11%) vs. everything else.
- **Known realistic ceiling:** ROC-AUC ≈ 0.68–0.72 for the <30-day target. The
  README reports the actual run value, not the resume's aspirational number.

## 4. Functional requirements (pipeline stages)

1. **Data acquisition** — download via `ucimlrepo`, cache to `data/raw/`
   (gitignored). Deterministic; re-runnable offline once cached.
2. **Feature engineering** — map ICD-9 `diag_1/2/3` to clinical categories
   (circulatory, respiratory, diabetes, injury, …); age brackets → ordinals;
   service-utilization aggregates (inpatient/outpatient/emergency); medication
   change/count features; principled handling of `?`/missing (drop
   `weight`, `payer_code` if too sparse; impute/flag the rest).
3. **Train/validate/test split** — stratified; fixed seed for reproducibility.
4. **Modeling** — Logistic-Regression baseline → XGBoost (primary) + LightGBM.
   Imbalance via `scale_pos_weight` + operating-threshold selection (not naive
   oversampling). Optuna hyperparameter search with MLflow logging.
5. **Evaluation** — ROC-AUC, PR-AUC, Brier score, calibration curve, confusion
   matrix at chosen threshold, recall@fixed-precision.
6. **Explainability** — SHAP global (beeswarm + bar) and local (waterfall for
   sample patients); one LIME example for contrast.
7. **Fairness** — sliced metrics (AUC, recall, FPR) by `race`, `gender`, age
   group; flag disparities.
8. **Model card** — generate `reports/MODEL_CARD.md` (intended use, data,
   metrics, limitations, fairness, ethics).
9. **Serving** — FastAPI service loads the trained artifact; `/predict` returns
   risk score + top SHAP contributors; `/health` for liveness.

## 5. Architecture & modules

```
src/readmission/
  config.py      # load config/config.yaml (paths, params, feature lists)
  data.py        # download + load + cache; train/test split
  features.py    # ICD grouping, encoding, transforms (pure, deterministic, tested)
  train.py       # build pipeline, Optuna tuning, fit, MLflow log, persist artifact
  evaluate.py    # metrics, calibration, fairness slices -> reports/
  explain.py     # SHAP + LIME -> reports/figures/
  model_card.py  # render reports/MODEL_CARD.md from metrics
  api/
    main.py      # FastAPI app
    schemas.py   # Pydantic request/response models
```

Each module has one responsibility and a clear interface. `features.py` is pure
and independently testable (input frame → output frame), so the same transforms
run identically in training and in the API.

## 6. Data flow

`ucimlrepo` → `data.py` (raw cache) → `features.py` (feature matrix) →
`train.py` (Optuna + fit → `models/model.pkl`, MLflow run) →
`evaluate.py` + `explain.py` (→ `reports/`) → `model_card.py` (→ MODEL_CARD.md).
At serve time: request → `schemas.py` validation → `features.py` transform →
`models/model.pkl` → risk score + SHAP drivers.

## 7. Tech stack & environment

- **Language:** Python 3.12 (canonical, via Docker `python:3.12-slim`). Local
  3.14 is too new for some ML wheels; `uv` provisions a 3.12 dev env locally.
- **Core:** pandas, numpy, scikit-learn, xgboost, lightgbm, optuna, shap, lime,
  mlflow, fastapi, uvicorn, pydantic, ucimlrepo.
- **Dev/quality:** ruff (lint+format), pytest, pytest-cov.
- **Deps & lock:** `pyproject.toml` (+ `uv.lock` or pinned `requirements.txt`).
- **Workflows:** `Makefile` — `make data | train | evaluate | serve | test`.

## 8. Repo structure

```
readmission-risk-xgboost/
├── README.md                  # architecture diagram, real results, how-to-run
├── pyproject.toml
├── Makefile
├── Dockerfile                 # python:3.12-slim, runs API
├── docker-compose.yml         # api (+ optional mlflow ui)
├── .github/workflows/ci.yml   # ruff + pytest on push/PR
├── .gitignore · .env.example · .dockerignore
├── config/config.yaml
├── notebooks/01_eda.ipynb
├── src/readmission/           # (see §5)
├── models/.gitkeep            # trained artifact committed (small) for demo
├── reports/                   # SHAP/calibration/fairness PNGs + MODEL_CARD.md (committed)
├── tests/                     # test_features.py, test_data.py, test_api.py
└── docs/                      # architecture.md + this spec
```

## 9. Testing strategy

- `test_features.py` — deterministic transforms: known input → expected output,
  ICD grouping correctness, no leakage of target into features.
- `test_data.py` — split is stratified and seed-stable; schema sanity.
- `test_api.py` — FastAPI `TestClient`: `/health` ok, `/predict` validates input
  and returns a score in [0,1] + SHAP drivers; bad payload → 422.
- Coverage reported; CI fails on test failure or lint error.

## 10. CI

GitHub Actions (`ci.yml`): set up Python 3.12 → install (uv) → `ruff check` →
`pytest`. Runs on push and PR. Badge in README.

## 11. Reproducibility & deliverable

- Docker is the source of truth; `make` targets give one-command local runs.
- **Committed for GitHub presentation:** generated figures, `MODEL_CARD.md`, and
  a small trained model — so the repo renders richly without anyone running it.
- **Gitignored:** `data/raw/` (re-downloadable), `mlruns/`, large/derived files.
- **Account-agnostic:** README/CI badges use a `<your-username>` placeholder.
- **Deliverable:** git-initialized folder + `readmission-risk-xgboost.zip` ready
  to `git remote add` + `push`. Verified to run (data → train → serve → tests
  green) before completion.

## 12. Integrity notes

- README frames this as a **working reference implementation on public data**,
  not the proprietary Healthstream system; reports real metrics only.
- No real PHI is used or required; the PII-safe design is documented, not faked.

## 13. Out of scope (YAGNI)

- Cloud deployment (covered conceptually; repo 5 owns the MLOps lifecycle).
- Orchestration/scheduling (Airflow) — that's repo 1's lakehouse / repo 5's job.
- Deep-learning models — XGBoost/LightGBM are the right tool for this tabular task.
- Front-end UI — the API + notebook + reports are the interface.

## 14. Success criteria

- `make data && make train` produces a model + MLflow run + populated `reports/`.
- `make serve` exposes a working `/predict` returning a calibrated risk score
  with SHAP drivers; `make test` is green; CI passes.
- README clearly states real metrics, the dataset, and one-command repro steps.
