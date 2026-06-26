import numpy as np
import pandas as pd

from readmission.evaluate import compute_metrics, fairness_report, pick_threshold


def test_compute_metrics_known_values():
    y_true = [0, 0, 1, 1]
    y_proba = [0.1, 0.4, 0.35, 0.8]
    m = compute_metrics(y_true, y_proba, threshold=0.5)
    assert 0.0 <= m["roc_auc"] <= 1.0
    assert m["confusion"]["tp"] == 1
    assert m["confusion"]["tn"] == 2
    assert "brier" in m and "pr_auc" in m


def test_fairness_report_one_row_per_group():
    y_true = np.array([0, 1, 0, 1])
    y_proba = np.array([0.2, 0.9, 0.3, 0.6])
    sensitive = pd.DataFrame({"gender": ["M", "M", "F", "F"]})
    report = fairness_report(y_true, y_proba, sensitive, threshold=0.5)
    assert set(report["group"]) == {"M", "F"}
    assert len(report) == 2
    assert {"attribute", "group", "count", "recall", "fpr"} <= set(report.columns)


def test_pick_threshold_in_range():
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_proba = np.array([0.1, 0.2, 0.6, 0.9, 0.4, 0.55])
    t = pick_threshold(y_true, y_proba, target_precision=0.3)
    assert 0.0 <= t <= 1.0
