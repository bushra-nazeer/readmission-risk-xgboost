"""FastAPI scoring service for readmission risk.

Loads the trained pipeline at startup and exposes ``/health`` and ``/predict``.
Each prediction returns a calibrated risk score, a coarse band, and the top
SHAP drivers for that specific patient.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from ..config import load_config
from ..explain import top_shap_for_instance
from ..features import engineer_features
from .schemas import FactorContribution, PatientEncounter, PredictionResponse

cfg = load_config()
_state: dict = {"model": None, "metadata": {}}


def _load_model() -> None:
    model_path = Path(cfg.paths.model_path)
    if model_path.exists():
        _state["model"] = joblib.load(model_path)
    metadata_path = Path(cfg.paths.model_metadata)
    if metadata_path.exists():
        _state["metadata"] = json.loads(metadata_path.read_text())


@asynccontextmanager
async def lifespan(_: FastAPI):
    _load_model()
    yield


app = FastAPI(title="Readmission Risk API", version="0.1.0", lifespan=lifespan)


def _risk_band(score: float) -> str:
    if score >= 0.5:
        return "High"
    if score >= 0.2:
        return "Medium"
    return "Low"


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": _state["model"] is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PatientEncounter) -> PredictionResponse:
    model = _state["model"]
    if model is None:
        raise HTTPException(
            status_code=503, detail="Model not loaded. Train a model first (`make train`)."
        )

    metadata = _state["metadata"]
    raw_columns = metadata.get("raw_feature_columns", [])
    raw_defaults = metadata.get("raw_defaults", {})

    row = {col: raw_defaults.get(col) for col in raw_columns}
    for key, value in payload.model_dump().items():
        if key in row and value is not None:
            row[key] = value

    frame = engineer_features(pd.DataFrame([row]), drop_columns=cfg.drop_columns)
    score = float(model.predict_proba(frame)[:, 1][0])

    try:
        factors = [
            FactorContribution(feature=name, contribution=value)
            for name, value in top_shap_for_instance(model, frame, k=6)
        ]
    except Exception:
        factors = []

    return PredictionResponse(
        risk_score=score,
        risk_band=_risk_band(score),
        top_factors=factors,
        model_type=metadata.get("model_type", "unknown"),
    )
