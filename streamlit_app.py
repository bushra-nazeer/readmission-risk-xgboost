"""Streamlit demo for the 30-day readmission risk model.

Run locally with `streamlit run streamlit_app.py`, or deploy free on Streamlit
Community Cloud by pointing it at this file.
"""

from __future__ import annotations

import json

import joblib
import pandas as pd
import streamlit as st

from readmission.config import load_config
from readmission.explain import top_shap_for_instance
from readmission.features import engineer_features

st.set_page_config(page_title="Readmission Risk", page_icon="🏥", layout="centered")

AGE_BANDS = [
    "[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)",
    "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)",
]


@st.cache_resource
def _load():
    cfg = load_config()
    model = joblib.load(cfg.paths.model_path)
    with open(cfg.paths.model_metadata) as fh:
        meta = json.load(fh)
    return cfg, model, meta


cfg, model, meta = _load()

st.title("30-Day Hospital Readmission Risk")
st.caption(
    "Estimates the probability of inpatient readmission within 30 days. "
    "Public UCI Diabetes data; educational reference implementation."
)

left, right = st.columns(2)
age = left.selectbox("Age band", AGE_BANDS, index=7)
time_in_hospital = left.slider("Days in hospital", 1, 14, 5)
number_diagnoses = left.slider("Number of diagnoses", 1, 16, 9)
num_medications = right.slider("Number of medications", 1, 50, 16)
number_inpatient = right.slider("Prior inpatient visits", 0, 12, 1)
diag_1 = right.text_input("Primary diagnosis (ICD-9)", "428")

if st.button("Estimate risk", type="primary"):
    row = {col: meta["raw_defaults"].get(col) for col in meta["raw_feature_columns"]}
    row.update(
        {
            "age": age,
            "time_in_hospital": time_in_hospital,
            "number_diagnoses": number_diagnoses,
            "num_medications": num_medications,
            "number_inpatient": number_inpatient,
            "diag_1": diag_1,
        }
    )
    frame = engineer_features(pd.DataFrame([row]), drop_columns=cfg.drop_columns)
    proba = float(model.predict_proba(frame)[:, 1][0])

    st.metric("Readmission risk", f"{proba * 100:.1f}%")
    st.progress(min(proba, 1.0))

    st.subheader("Top contributing factors")
    for feature, value in top_shap_for_instance(model, frame, k=6):
        arrow = "increases risk" if value > 0 else "lowers risk"
        st.write(f"- `{feature}`: {arrow}  ({value:+.3f})")
