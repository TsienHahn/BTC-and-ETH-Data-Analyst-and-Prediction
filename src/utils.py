
from pathlib import Path
import random

import numpy as np
import tensorflow as tf
import yaml

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def set_random_seed(seed: int = 42) -> None:
    """Set random seeds for reproducible experiments."""

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_config(config_path: str) -> dict:
    """Load YAML project configuration."""

    path = PROJECT_ROOT / config_path

    if not path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:
        config = yaml.safe_load(file)

    if config is None:
        raise ValueError(
            f"Configuration file is empty: {path}"
        )

    return config


def create_project_directories() -> None:
    """Create directories required by the pipeline."""

    directories = [
        PROJECT_ROOT / "data" / "raw",
        PROJECT_ROOT / "data" / "processed",
        PROJECT_ROOT / "models",
        PROJECT_ROOT / "results",
        PROJECT_ROOT / "results" / "figures"
    ]

    for directory in directories:
        directory.mkdir(
            parents=True,
            exist_ok=True
        )
