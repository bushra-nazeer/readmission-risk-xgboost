"""Evaluation: metrics, threshold selection, calibration, and fairness slices."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .config import Config, load_config
from .data import load_raw, make_target, split
from .features import engineer_features

# Use a non-interactive backend so figures render in headless containers/CI.
matplotlib.use("Agg")


def compute_metrics(y_true, y_proba, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "brier": float(brier_score_loss(y_true, y_proba)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "threshold": float(threshold),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def pick_threshold(y_true, y_proba, target_precision: float = 0.3) -> float:
    """Smallest threshold meeting ``target_precision`` with the best recall; else 0.5."""
    prec, rec, thr = precision_recall_curve(y_true, y_proba)
    best_threshold, best_recall = 0.5, -1.0
    for p, r, t in zip(prec[:-1], rec[:-1], thr, strict=False):
        if p >= target_precision and r > best_recall:
            best_recall, best_threshold = r, float(t)
    return best_threshold


def fairness_report(y_true, y_proba, sensitive_df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Per-subgroup AUC, recall, and false-positive rate."""
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    y_pred = (y_proba >= threshold).astype(int)
    rows = []
    for col in sensitive_df.columns:
        for group in sorted(sensitive_df[col].dropna().unique(), key=str):
            mask = (sensitive_df[col] == group).to_numpy()
            yt, yp, pr = y_true[mask], y_pred[mask], y_proba[mask]
            if len(yt) == 0:
                continue
            auc = (
                float(roc_auc_score(yt, pr))
                if 0 < yt.sum() < len(yt)
                else None
            )
            tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
            rows.append(
                {
                    "attribute": col,
                    "group": str(group),
                    "count": int(len(yt)),
                    "positives": int(yt.sum()),
                    "auc": auc,
                    "recall": float(tp / (tp + fn)) if (tp + fn) else None,
                    "fpr": float(fp / (fp + tn)) if (fp + tn) else None,
                }
            )
    return pd.DataFrame(rows)


def evaluate(cfg: Config):
    """Score the held-out test set; write metrics, fairness, and figures."""
    model = joblib.load(cfg.paths.model_path)
    raw = load_raw(cfg)
    y = make_target(raw, positive=cfg.target_positive_label, column=cfg.target.column)
    X = engineer_features(raw, drop_columns=cfg.drop_columns)
    _, _, X_test, _, _, y_test = split(X, y, cfg)

    proba = model.predict_proba(X_test)[:, 1]
    threshold = pick_threshold(y_test.to_numpy(), proba, cfg.threshold.target_precision)
    metrics = compute_metrics(y_test.to_numpy(), proba, threshold)
    metrics["positive_rate"] = float(y.mean())

    sensitive = raw.loc[X_test.index, cfg.sensitive_features].reset_index(drop=True)
    fairness = fairness_report(y_test.to_numpy(), proba, sensitive, threshold)

    reports = Path(cfg.paths.reports_dir)
    figures = Path(cfg.paths.figures_dir)
    reports.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    (reports / "metrics.json").write_text(json.dumps(metrics, indent=2))
    fairness.to_csv(reports / "fairness.csv", index=False)

    frac_pos, mean_pred = calibration_curve(y_test, proba, n_bins=10, strategy="quantile")
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], "--", color="grey", label="Perfectly calibrated")
    plt.plot(mean_pred, frac_pos, marker="o", label="Model")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed readmission frequency")
    plt.title("Calibration curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures / "calibration.png", dpi=120)
    plt.close()

    plot_df = fairness.dropna(subset=["recall"])
    if len(plot_df):
        labels = (plot_df["attribute"] + ": " + plot_df["group"]).tolist()
        plt.figure(figsize=(8, max(3, 0.4 * len(plot_df))))
        plt.barh(labels, plot_df["recall"])
        plt.xlabel("Recall")
        plt.title(f"Recall by subgroup (threshold={threshold:.2f})")
        plt.tight_layout()
        plt.savefig(figures / "fairness_recall.png", dpi=120)
        plt.close()

    return metrics, fairness


def main() -> None:
    cfg = load_config()
    metrics, _ = evaluate(cfg)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
