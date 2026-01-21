"""Deterministic runtime helpers."""

import random
from typing import Optional

import numpy as np


def set_seeds(seed: Optional[int] = 42) -> None:
    """
    Set random seeds for deterministic execution.

    Args:
        seed: Random seed value (None to skip seeding)
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)


def ensure_stable_sorting(data, **kwargs):
    """
    Ensure stable sorting for pandas DataFrames/Series.

    Args:
        data: Data to sort
        **kwargs: Additional sorting parameters

    Returns:
        Sorted data with stable algorithm
    """
    if hasattr(data, "sort_values"):
        # Pandas DataFrame or Series
        kwargs.setdefault("kind", "stable")
        return data.sort_values(**kwargs)
    return data


def configure_matplotlib_backend() -> None:
    """Configure matplotlib for deterministic, non-interactive rendering."""
    import matplotlib

    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt

    plt.ioff()  # Turn off interactive mode


def get_deterministic_style() -> dict:
    """
    Get deterministic matplotlib style configuration.

    Returns:
        Dictionary of matplotlib rcParams
    """
    return {
        "figure.figsize": (10, 6),
        "figure.dpi": 100,
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "lines.linewidth": 1.5,
        "lines.markersize": 6,
        "savefig.dpi": 100,
        "savefig.bbox": "tight",
    }
