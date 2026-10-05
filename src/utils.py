"""Shared project utility functions."""

from pathlib import Path
import random
import numpy as np
import tensorflow as tf
import yaml


def set_random_seed(seed: int = 42) -> None:
    """Set random seeds for reproducible experiments."""

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_config(config_path: str) -> dict:
    """Load YAML project configuration."""

    with open(
        config_path,
        "r",
        encoding="utf-8"
    ) as file:
        return yaml.safe_load(file)


def create_project_directories() -> None:
    """Create directories required by the pipeline."""

    directories = [
        "data/raw",
        "data/processed",
        "models",
        "results",
        "results/figures"
    ]

    for directory in directories:
        Path(directory).mkdir(
            parents=True,
            exist_ok=True
        )
