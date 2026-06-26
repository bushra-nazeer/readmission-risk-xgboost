"""Typed configuration loaded from ``config/config.yaml``.

Every module reads settings through :func:`load_config` so paths, seeds, and
hyperparameters live in exactly one place.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel

DEFAULT_CONFIG_PATH = "config/config.yaml"


class Paths(BaseModel):
    data_raw: str
    data_processed: str
    model_path: str
    model_metadata: str
    reports_dir: str
    figures_dir: str


class TargetCfg(BaseModel):
    column: str = "readmitted"
    positive_label: str = "<30"


class OptunaCfg(BaseModel):
    n_trials: int = 25
    timeout_seconds: int = 900


class ThresholdCfg(BaseModel):
    target_precision: float = 0.30


class Config(BaseModel):
    paths: Paths
    random_state: int = 42
    test_size: float = 0.2
    val_size: float = 0.1
    target: TargetCfg = TargetCfg()
    drop_columns: list[str] = []
    sensitive_features: list[str] = []
    optuna: OptunaCfg = OptunaCfg()
    threshold: ThresholdCfg = ThresholdCfg()

    @property
    def target_positive_label(self) -> str:
        return self.target.positive_label


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> Config:
    """Read the YAML config file into a validated :class:`Config`."""
    with open(path) as fh:
        data = yaml.safe_load(fh)
    return Config(**data)
