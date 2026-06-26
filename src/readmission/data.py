"""Data acquisition and splitting.

The raw dataset is the UCI *Diabetes 130-US Hospitals (1999-2008)* set
(``ucimlrepo`` id 296). It is fetched once and cached as parquet so subsequent
runs are offline and deterministic.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import Config, load_config

UCI_DATASET_ID = 296


def load_raw(cfg: Config, force_download: bool = False) -> pd.DataFrame:
    """Return the raw encounter frame (features + ``readmitted`` target).

    Cached to ``cfg.paths.data_raw``; re-downloaded only when missing or when
    ``force_download`` is True.
    """
    cache = Path(cfg.paths.data_raw)
    if cache.exists() and not force_download:
        return pd.read_parquet(cache)

    from ucimlrepo import fetch_ucirepo

    dataset = fetch_ucirepo(id=UCI_DATASET_ID)
    df = pd.concat([dataset.data.features, dataset.data.targets], axis=1)
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache)
    return df


def make_target(df: pd.DataFrame, positive: str = "<30", column: str = "readmitted") -> pd.Series:
    """Binary target: 1 when readmitted within 30 days, else 0."""
    return (df[column].astype(str) == positive).astype(int)


def split(X: pd.DataFrame, y: pd.Series, cfg: Config):
    """Stratified, seeded train/val/test split.

    Returns ``(X_train, X_val, X_test, y_train, y_val, y_test)``.
    """
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=cfg.test_size, stratify=y, random_state=cfg.random_state
    )
    val_relative = cfg.val_size / (1.0 - cfg.test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=val_relative, stratify=y_tmp, random_state=cfg.random_state
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and cache the raw dataset.")
    parser.add_argument("--force", action="store_true", help="Re-download even if cached.")
    args = parser.parse_args()
    cfg = load_config()
    df = load_raw(cfg, force_download=args.force)
    print(f"Loaded raw encounters: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"Cached to: {cfg.paths.data_raw}")


if __name__ == "__main__":
    main()
