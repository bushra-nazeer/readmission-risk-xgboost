"""Explainability: SHAP global/local plots, a LIME example, and a helper the
API reuses to surface per-prediction drivers."""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from .config import Config, load_config
from .data import load_raw, make_target, split
from .features import engineer_features

# Use a non-interactive backend so figures render in headless containers/CI.
matplotlib.use("Agg")


def _transform(model, X: pd.DataFrame):
    """Run the pipeline's preprocessor; return (matrix, clean feature_names)."""
    pre = model.named_steps["pre"]
    matrix = pre.transform(X)
    # Strip ColumnTransformer's "num__"/"cat__" prefixes for readable labels.
    names = [name.split("__", 1)[-1] for name in pre.get_feature_names_out()]
    return np.asarray(matrix), names


def top_shap_for_instance(model, x_row: pd.DataFrame, k: int = 8):
    """Return the top-k (feature, signed_shap_value) drivers for one row."""
    matrix, names = _transform(model, x_row)
    explainer = shap.TreeExplainer(model.named_steps["clf"])
    values = np.asarray(explainer.shap_values(matrix))
    if values.ndim == 3:  # (n, features, classes) in some versions
        values = values[..., -1]
    row = values[0]
    order = np.argsort(np.abs(row))[::-1][:k]
    return [(str(names[i]), float(row[i])) for i in order]


def explain(cfg: Config, sample_size: int = 1000, max_display: int = 20) -> Path:
    """Generate SHAP beeswarm/bar/waterfall and a LIME example into figures_dir."""
    model = joblib.load(cfg.paths.model_path)
    raw = load_raw(cfg)
    y = make_target(raw, positive=cfg.target_positive_label, column=cfg.target.column)
    X = engineer_features(raw, drop_columns=cfg.drop_columns)
    _, _, X_test, _, _, _ = split(X, y, cfg)

    sample = X_test.sample(n=min(sample_size, len(X_test)), random_state=cfg.random_state)
    matrix, names = _transform(model, sample)
    explainer = shap.TreeExplainer(model.named_steps["clf"])
    explanation = explainer(matrix)

    figures = Path(cfg.paths.figures_dir)
    figures.mkdir(parents=True, exist_ok=True)

    shap.summary_plot(
        explanation.values, features=matrix, feature_names=names, show=False, max_display=max_display
    )
    plt.tight_layout()
    plt.savefig(figures / "shap_beeswarm.png", dpi=120, bbox_inches="tight")
    plt.close()

    shap.summary_plot(
        explanation.values,
        features=matrix,
        feature_names=names,
        plot_type="bar",
        show=False,
        max_display=max_display,
    )
    plt.tight_layout()
    plt.savefig(figures / "shap_bar.png", dpi=120, bbox_inches="tight")
    plt.close()

    try:
        shap.plots.waterfall(explanation[0], show=False, max_display=12)
        plt.tight_layout()
        plt.savefig(figures / "shap_waterfall.png", dpi=120, bbox_inches="tight")
        plt.close()
    except Exception as exc:  # waterfall API varies across shap versions
        print(f"SHAP waterfall skipped: {exc}")

    try:
        from lime.lime_tabular import LimeTabularExplainer

        lime_explainer = LimeTabularExplainer(
            matrix,
            feature_names=names,
            class_names=["no_readmit", "readmit"],
            discretize_continuous=True,
            random_state=cfg.random_state,
        )
        local = lime_explainer.explain_instance(
            matrix[0], model.named_steps["clf"].predict_proba, num_features=10
        )
        local.save_to_file(str(figures / "lime_example.html"))
    except Exception as exc:
        print(f"LIME example skipped: {exc}")

    return figures


def main() -> None:
    cfg = load_config()
    figures = explain(cfg)
    print(f"Explainability artifacts written to: {figures}")


if __name__ == "__main__":
    main()
