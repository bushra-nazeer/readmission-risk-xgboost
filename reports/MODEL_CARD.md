# Model Card, 30-Day Hospital Readmission Risk

## Model Details
- **Type:** xgboost
- **Task:** Binary classification, probability of inpatient readmission within 30 days.
- **Dataset:** UCI Diabetes 130-US Hospitals (1999-2008), id=296.
- **Positive label:** `readmitted == "<30"`.

## Intended Use
Decision *support* for care-management teams to prioritize post-discharge
follow-up. This is **not** a diagnostic device and not a substitute for
clinical judgment.

## Metrics (held-out test set)
| Metric | Value |
|---|---|
| ROC-AUC | 0.688 |
| PR-AUC | 0.235 |
| Brier score | 0.206 |
| Precision @ threshold | 0.300 |
| Recall @ threshold | 0.200 |
| Operating threshold | 0.641 |

A logistic-regression baseline is trained alongside for reference
(baseline ROC-AUC: 0.648).

## Fairness (subgroup performance at the operating threshold)
| attribute | group | count | positives | auc | recall | fpr |
| --- | --- | --- | --- | --- | --- | --- |
| race | AfricanAmerican | 3866 | 432 | 0.683 | 0.192 | 0.059 |
| race | Asian | 123 | 10 | 0.576 | 0.100 | 0.053 |
| race | Caucasian | 15223 | 1717 | 0.687 | 0.206 | 0.061 |
| race | Hispanic | 404 | 50 | 0.768 | 0.240 | 0.056 |
| race | Other | 276 | 20 | 0.690 | 0.150 | 0.035 |
| gender | Female | 10924 | 1255 | 0.691 | 0.206 | 0.059 |
| gender | Male | 9430 | 1016 | 0.684 | 0.192 | 0.058 |
| age | [0-10) | 34 | 0 |, |, | 0.000 |
| age | [10-20) | 130 | 6 | 0.804 | 0.000 | 0.008 |
| age | [20-30) | 324 | 43 | 0.836 | 0.535 | 0.121 |
| age | [30-40) | 725 | 78 | 0.708 | 0.282 | 0.083 |
| age | [40-50) | 1913 | 188 | 0.745 | 0.261 | 0.066 |
| age | [50-60) | 3457 | 301 | 0.718 | 0.256 | 0.054 |
| age | [60-70) | 4547 | 529 | 0.663 | 0.168 | 0.056 |
| age | [70-80) | 5234 | 628 | 0.657 | 0.197 | 0.066 |
| age | [80-90) | 3414 | 424 | 0.660 | 0.144 | 0.050 |
| age | [90-100) | 576 | 74 | 0.647 | 0.122 | 0.022 |

## Limitations
- Trained on 1999-2008 US hospital encounters; not representative of current
  populations or non-US care settings.
- Class imbalance (~11% positive) makes PR-AUC and calibration
  more informative than raw accuracy.
- Demographic labels are coarse and self-reported; the fairness table is
  indicative, not exhaustive.

## Ethical Considerations
- Subgroup metrics are reported and should be monitored before any real use.
- This is a **public-data reference implementation**, not a deployed clinical
  system, and contains no protected health information.
