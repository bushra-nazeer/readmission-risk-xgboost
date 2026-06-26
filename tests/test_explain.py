import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from readmission.explain import top_shap_for_instance
from readmission.features import build_preprocessor


@pytest.fixture
def tiny_model():
    from xgboost import XGBClassifier

    rng = np.random.RandomState(0)
    X = pd.DataFrame(
        {
            "num1": rng.randn(80),
            "num2": rng.randn(80),
            "cat1": rng.choice(["a", "b"], size=80),
        }
    )
    y = ((X["num1"] + (X["cat1"] == "a") * 0.7 + rng.randn(80) * 0.1) > 0).astype(int)
    pipe = Pipeline(
        [
            ("pre", build_preprocessor(X)),
            ("clf", XGBClassifier(n_estimators=25, max_depth=3, eval_metric="logloss", tree_method="hist")),
        ]
    )
    pipe.fit(X, y)
    return pipe, X


def test_top_shap_for_instance_returns_ranked_factors(tiny_model):
    model, X = tiny_model
    factors = top_shap_for_instance(model, X.iloc[[0]], k=3)
    assert 1 <= len(factors) <= 3
    assert all(isinstance(name, str) and isinstance(value, float) for name, value in factors)
