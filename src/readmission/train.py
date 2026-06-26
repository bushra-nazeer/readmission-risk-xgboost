"""Model training with Optuna tuning and MLflow tracking.

Builds a ``Pipeline(preprocessor, classifier)`` so the persisted artifact takes
an engineered feature frame directly — the exact transforms used at serve time.
A logistic-regression baseline is trained for comparison.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import mlflow
import optuna
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline

from .config import Config, load_config
from .data import load_raw, make_target, split
from .features import build_preprocessor, engineer_features

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _build_classifier(model_type: str, params: dict, random_state: int):
    if model_type == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            **params,
            random_state=random_state,
            n_jobs=-1,
            eval_metric="logloss",
            tree_method="hist",
        )
    if model_type == "lightgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(**params, random_state=random_state, n_jobs=-1, verbose=-1)
    if model_type == "logreg":
        return LogisticRegression(max_iter=1000, random_state=random_state, **params)
    raise ValueError(f"unknown model_type: {model_type}")


def _suggest_params(trial: optuna.Trial, model_type: str, scale_pos_weight: float) -> dict:
    if model_type == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 600),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "scale_pos_weight": scale_pos_weight,
        }
    if model_type == "lightgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 600),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "scale_pos_weight": scale_pos_weight,
        }
    return {}


def _raw_defaults(raw_features: pd.DataFrame) -> dict:
    defaults: dict = {}
    for col in raw_features.columns:
        series = raw_features[col]
        if pd.api.types.is_numeric_dtype(series):
            defaults[col] = float(series.median())
        else:
            mode = series.mode(dropna=True)
            defaults[col] = mode.iloc[0] if len(mode) else None
    return defaults


def train(
    cfg: Config,
    model_type: str = "xgboost",
    n_trials: int | None = None,
    sample_n: int | None = None,
) -> dict:
    """Train, tune, evaluate, log to MLflow, and persist the model. Returns metrics."""
    raw = load_raw(cfg)
    if sample_n:
        raw = raw.sample(n=min(sample_n, len(raw)), random_state=cfg.random_state).reset_index(
            drop=True
        )

    y = make_target(raw, positive=cfg.target_positive_label, column=cfg.target.column)
    X = engineer_features(raw, drop_columns=cfg.drop_columns)
    X_train, X_val, X_test, y_train, y_val, y_test = split(X, y, cfg)

    positives = int(y_train.sum())
    negatives = int(len(y_train) - positives)
    scale_pos_weight = float(negatives / max(positives, 1))
    n_trials = n_trials if n_trials is not None else cfg.optuna.n_trials

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_params(trial, model_type, scale_pos_weight)
        pipe = Pipeline(
            [
                ("pre", build_preprocessor(X_train)),
                ("clf", _build_classifier(model_type, params, cfg.random_state)),
            ]
        )
        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_val)[:, 1]
        return average_precision_score(y_val, proba)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, timeout=cfg.optuna.timeout_seconds)
    best_params = dict(study.best_params)
    best_params["scale_pos_weight"] = scale_pos_weight

    X_fit = pd.concat([X_train, X_val])
    y_fit = pd.concat([y_train, y_val])
    model = Pipeline(
        [
            ("pre", build_preprocessor(X_fit)),
            ("clf", _build_classifier(model_type, best_params, cfg.random_state)),
        ]
    )
    model.fit(X_fit, y_fit)

    test_proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        "model_type": model_type,
        "roc_auc": float(roc_auc_score(y_test, test_proba)),
        "pr_auc": float(average_precision_score(y_test, test_proba)),
        "n_train": int(len(X_fit)),
        "n_test": int(len(X_test)),
        "positive_rate": float(y.mean()),
    }

    baseline = Pipeline(
        [
            ("pre", build_preprocessor(X_fit)),
            ("clf", _build_classifier("logreg", {"class_weight": "balanced"}, cfg.random_state)),
        ]
    )
    baseline.fit(X_fit, y_fit)
    metrics["baseline_roc_auc"] = float(
        roc_auc_score(y_test, baseline.predict_proba(X_test)[:, 1])
    )

    mlflow.set_experiment("readmission-risk")
    with mlflow.start_run(run_name=f"{model_type}-optuna"):
        mlflow.log_params(best_params)
        mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})

    model_path = Path(cfg.paths.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

    raw_features = raw.drop(columns=[cfg.target.column], errors="ignore")
    metadata = {
        "model_type": model_type,
        "best_params": best_params,
        "metrics": metrics,
        "feature_columns": X_fit.columns.tolist(),
        "raw_feature_columns": raw_features.columns.tolist(),
        "raw_defaults": _raw_defaults(raw_features),
        "target_positive_label": cfg.target_positive_label,
        "dataset": "UCI Diabetes 130-US Hospitals (1999-2008), id=296",
    }
    Path(cfg.paths.model_metadata).write_text(json.dumps(metadata, indent=2))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the readmission risk model.")
    parser.add_argument("--model", default="xgboost", choices=["xgboost", "lightgbm"])
    parser.add_argument("--trials", type=int, default=None, help="Optuna trials (default: config).")
    parser.add_argument("--sample", type=int, default=None, help="Subsample N rows (smoke runs).")
    args = parser.parse_args()
    cfg = load_config()
    metrics = train(cfg, model_type=args.model, n_trials=args.trials, sample_n=args.sample)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
