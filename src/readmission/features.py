"""Feature engineering for the diabetes readmission dataset.

All transforms here are pure and deterministic so the exact same logic runs in
training and at serve time. The headline transform groups raw ICD-9 diagnosis
codes into clinical categories (the standard mapping from Strack et al., 2014).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

AGE_MIDPOINTS = {
    "[0-10)": 5,
    "[10-20)": 15,
    "[20-30)": 25,
    "[30-40)": 35,
    "[40-50)": 45,
    "[50-60)": 55,
    "[60-70)": 65,
    "[70-80)": 75,
    "[80-90)": 85,
    "[90-100)": 95,
}

MED_COLUMNS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
    "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
    "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
    "examide", "citoglipton", "insulin", "glyburide-metformin",
    "glipizide-metformin", "glimepiride-pioglitazone",
    "metformin-rosiglitazone", "metformin-pioglitazone",
]

DIAGNOSIS_COLUMNS = ["diag_1", "diag_2", "diag_3"]
UTILIZATION_COLUMNS = ["number_inpatient", "number_outpatient", "number_emergency"]


def group_icd9(code) -> str:
    """Map a raw ICD-9 code to a clinical category."""
    if code is None or (isinstance(code, float) and np.isnan(code)):
        return "missing"
    text = str(code).strip()
    if text in ("", "?", "nan", "None"):
        return "missing"
    if text.startswith(("V", "E")):
        return "other"
    try:
        value = float(text)
    except ValueError:
        return "other"
    whole = int(value)
    if 250 <= value < 251:
        return "diabetes"
    if (390 <= whole <= 459) or whole == 785:
        return "circulatory"
    if (460 <= whole <= 519) or whole == 786:
        return "respiratory"
    if (520 <= whole <= 579) or whole == 787:
        return "digestive"
    if 800 <= whole <= 999:
        return "injury"
    if 710 <= whole <= 739:
        return "musculoskeletal"
    if (580 <= whole <= 629) or whole == 788:
        return "genitourinary"
    if 140 <= whole <= 239:
        return "neoplasms"
    return "other"


def engineer_features(df: pd.DataFrame, drop_columns: list[str] | None = None) -> pd.DataFrame:
    """Transform the raw encounter frame into a model-ready feature frame.

    Drops identifiers/target, normalizes missing markers, converts age brackets
    to midpoints, groups diagnoses, and derives utilization/medication features.
    The DataFrame index is preserved so callers can re-join other columns.
    """
    drop_columns = drop_columns or []
    out = df.replace("?", np.nan)
    out = out.drop(columns=[c for c in ["readmitted"] if c in out.columns], errors="ignore")
    out = out.drop(columns=[c for c in drop_columns if c in out.columns], errors="ignore")

    if "age" in out.columns:
        out["age_midpoint"] = out["age"].map(AGE_MIDPOINTS)
        out = out.drop(columns=["age"])

    for col in DIAGNOSIS_COLUMNS:
        if col in out.columns:
            out[col + "_group"] = out[col].map(group_icd9)
            out = out.drop(columns=[col])

    util_cols = [c for c in UTILIZATION_COLUMNS if c in out.columns]
    if util_cols:
        out["total_visits"] = out[util_cols].sum(axis=1)

    present_meds = [c for c in MED_COLUMNS if c in out.columns]
    if present_meds:
        out["n_active_meds"] = (out[present_meds] != "No").sum(axis=1)
        out["n_med_changes"] = out[present_meds].isin(["Up", "Down"]).sum(axis=1)

    return out


def build_preprocessor(feature_frame: pd.DataFrame) -> ColumnTransformer:
    """Impute + scale numerics and impute + one-hot categoricals."""
    numeric = feature_frame.select_dtypes(include=["number"]).columns.tolist()
    categorical = [c for c in feature_frame.columns if c not in numeric]

    numeric_pipe = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    categorical_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [("num", numeric_pipe, numeric), ("cat", categorical_pipe, categorical)],
        remainder="drop",
    )
