import pandas as pd

from readmission.config import load_config
from readmission.model_card import render_model_card


def test_render_model_card_has_sections_and_values():
    cfg = load_config()
    metrics = {
        "roc_auc": 0.712,
        "pr_auc": 0.331,
        "brier": 0.090,
        "precision": 0.31,
        "recall": 0.42,
        "threshold": 0.28,
        "positive_rate": 0.11,
    }
    fairness = pd.DataFrame(
        [{"attribute": "gender", "group": "Male", "count": 100, "auc": 0.70, "recall": 0.40, "fpr": 0.2}]
    )
    card = render_model_card(metrics, fairness, cfg, {"model_type": "xgboost"})

    for section in ["## Intended Use", "## Metrics", "## Fairness", "## Limitations", "## Ethical"]:
        assert section in card
    assert "0.712" in card
    assert "xgboost" in card
