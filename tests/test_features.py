import numpy as np
import pandas as pd

from readmission.features import build_preprocessor, engineer_features, group_icd9


def test_group_icd9_categories():
    assert group_icd9("428") == "circulatory"
    assert group_icd9("250.83") == "diabetes"
    assert group_icd9("486") == "respiratory"
    assert group_icd9("V27") == "other"
    assert group_icd9("E909") == "other"
    assert group_icd9(np.nan) == "missing"
    assert group_icd9("?") == "missing"


def _fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "encounter_id": [1, 2],
            "patient_nbr": [10, 20],
            "age": ["[70-80)", "[60-70)"],
            "diag_1": ["428", "250.83"],
            "diag_2": ["?", "486"],
            "diag_3": ["V27", "800"],
            "number_inpatient": [1, 0],
            "number_outpatient": [0, 2],
            "number_emergency": [0, 1],
            "metformin": ["No", "Up"],
            "insulin": ["Steady", "No"],
            "readmitted": ["<30", "NO"],
        }
    )


def test_engineer_features_transforms():
    out = engineer_features(_fixture(), drop_columns=["encounter_id", "patient_nbr"])

    assert "readmitted" not in out.columns
    assert "encounter_id" not in out.columns and "patient_nbr" not in out.columns
    assert out["age_midpoint"].tolist() == [75, 65]
    assert out["diag_1_group"].tolist() == ["circulatory", "diabetes"]
    assert out["diag_2_group"].tolist() == ["missing", "respiratory"]
    assert out["total_visits"].tolist() == [1, 3]
    assert out["n_active_meds"].tolist() == [1, 1]
    assert out["n_med_changes"].tolist() == [0, 1]


def test_build_preprocessor_fit_transform_shape():
    out = engineer_features(_fixture(), drop_columns=["encounter_id", "patient_nbr"])
    pre = build_preprocessor(out)
    matrix = pre.fit_transform(out)
    assert matrix.shape[0] == 2
    assert matrix.shape[1] > 0
