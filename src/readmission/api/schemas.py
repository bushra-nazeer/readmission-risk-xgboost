"""Pydantic request/response models for the scoring API.

The request mirrors the most informative raw encounter fields. All are optional
and extra fields are accepted, so any subset of the original dataset columns can
be posted; unspecified fields fall back to training-set defaults server-side.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class PatientEncounter(BaseModel):
    model_config = ConfigDict(extra="allow")

    race: str | None = None
    gender: str | None = None
    age: str | None = Field(default=None, description="Age bracket, e.g. '[70-80)'.")
    admission_type_id: int | None = None
    discharge_disposition_id: int | None = None
    admission_source_id: int | None = None
    time_in_hospital: int | None = None
    num_lab_procedures: int | None = None
    num_procedures: int | None = None
    num_medications: int | None = None
    number_outpatient: int | None = None
    number_emergency: int | None = None
    number_inpatient: int | None = None
    number_diagnoses: int | None = None
    diag_1: str | None = None
    diag_2: str | None = None
    diag_3: str | None = None
    max_glu_serum: str | None = None
    A1Cresult: str | None = None
    insulin: str | None = None
    metformin: str | None = None
    change: str | None = None
    diabetesMed: str | None = None


class FactorContribution(BaseModel):
    feature: str
    contribution: float


class PredictionResponse(BaseModel):
    risk_score: float = Field(..., ge=0.0, le=1.0)
    risk_band: str
    top_factors: list[FactorContribution]
    model_type: str
