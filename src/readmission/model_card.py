"""Generate a Markdown model card from computed metrics and fairness results."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

from .config import Config, load_config


def _fmt(value) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _df_to_md(df: pd.DataFrame) -> str:
    if df is None or len(df) == 0:
        return "_Not available._"
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    separator = "| " + " | ".join("---" for _ in cols) + " |"
    rows = ["| " + " | ".join(_fmt(r[c]) for c in cols) + " |" for _, r in df.iterrows()]
    return "\n".join([header, separator, *rows])


def render_model_card(
    metrics: dict, fairness: pd.DataFrame, cfg: Config, metadata: dict | None = None
) -> str:
    metadata = metadata or {}
    model_type = metadata.get("model_type", "gradient-boosted trees")
    positive_pct = metrics.get("positive_rate", 0) * 100
    return f"""# Model Card, 30-Day Hospital Readmission Risk

## Model Details
- **Type:** {model_type}
- **Task:** Binary classification, probability of inpatient readmission within 30 days.
- **Dataset:** {metadata.get("dataset", "UCI Diabetes 130-US Hospitals (1999-2008)")}.
- **Positive label:** `readmitted == "{cfg.target_positive_label}"`.

## Intended Use
Decision *support* for care-management teams to prioritize post-discharge
follow-up. This is **not** a diagnostic device and not a substitute for
clinical judgment.

## Metrics (held-out test set)
| Metric | Value |
|---|---|
| ROC-AUC | {_fmt(metrics.get("roc_auc"))} |
| PR-AUC | {_fmt(metrics.get("pr_auc"))} |
| Brier score | {_fmt(metrics.get("brier"))} |
| Precision @ threshold | {_fmt(metrics.get("precision"))} |
| Recall @ threshold | {_fmt(metrics.get("recall"))} |
| Operating threshold | {_fmt(metrics.get("threshold"))} |

A logistic-regression baseline is trained alongside for reference
(baseline ROC-AUC: {_fmt(metadata.get("metrics", {}).get("baseline_roc_auc"))}).

## Fairness (subgroup performance at the operating threshold)
{_df_to_md(fairness)}

## Limitations
- Trained on 1999-2008 US hospital encounters; not representative of current
  populations or non-US care settings.
- Class imbalance (~{positive_pct:.0f}% positive) makes PR-AUC and calibration
  more informative than raw accuracy.
- Demographic labels are coarse and self-reported; the fairness table is
  indicative, not exhaustive.

## Ethical Considerations
- Subgroup metrics are reported and should be monitored before any real use.
- This is a **public-data reference implementation**, not a deployed clinical
  system, and contains no protected health information.
"""


def write_model_card(cfg: Config) -> Path:
    reports = Path(cfg.paths.reports_dir)
    metrics = json.loads((reports / "metrics.json").read_text())
    fairness = (
        pd.read_csv(reports / "fairness.csv")
        if (reports / "fairness.csv").exists()
        else pd.DataFrame()
    )
    metadata_path = Path(cfg.paths.model_metadata)
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    if "positive_rate" not in metrics:
        metrics["positive_rate"] = metadata.get("metrics", {}).get("positive_rate", 0)

    card = render_model_card(metrics, fairness, cfg, metadata)
    output = reports / "MODEL_CARD.md"
    output.write_text(card)
    return output


def main() -> None:
    cfg = load_config()
    path = write_model_card(cfg)
    print(f"Model card written to: {path}")


if __name__ == "__main__":
    main()
