import numpy as np
import pandas as pd

from readmission.config import load_config
from readmission.data import make_target, split


def _synthetic(n: int = 300) -> pd.DataFrame:
    rng = np.random.RandomState(0)
    return pd.DataFrame(
        {
            "f1": rng.randn(n),
            "f2": rng.randint(0, 5, size=n),
            "readmitted": rng.choice(["<30", ">30", "NO"], size=n, p=[0.2, 0.3, 0.5]),
        }
    )


def test_make_target_maps_labels():
    df = pd.DataFrame({"readmitted": ["<30", ">30", "NO", "<30"]})
    assert make_target(df).tolist() == [1, 0, 0, 1]


def test_split_is_seed_stable_and_stratified():
    cfg = load_config()
    df = _synthetic(400)
    y = make_target(df)
    X = df.drop(columns=["readmitted"])

    first = split(X, y, cfg)
    second = split(X, y, cfg)

    # Same seed -> identical test partition.
    assert list(first[2].index) == list(second[2].index)

    # Stratification: positive rate preserved in the train fold.
    full_rate = y.mean()
    train_rate = first[3].mean()
    assert abs(full_rate - train_rate) < 0.08
