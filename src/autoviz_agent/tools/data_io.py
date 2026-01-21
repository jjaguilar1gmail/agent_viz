"""Data I/O tools."""

from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd

from autoviz_agent.utils.logging import get_logger

logger = get_logger(__name__)


def load_dataset(path: Union[str, Path], **kwargs) -> pd.DataFrame:
    """
    Load dataset from file.

    Args:
        path: Path to dataset file (CSV)
        **kwargs: Additional arguments for pandas read_csv

    Returns:
        Loaded DataFrame

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is unsupported
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path, **kwargs)
        logger.info(f"Loaded CSV dataset: {path} ({len(df)} rows, {len(df.columns)} columns)")
        return df
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")


def sample_rows(df: pd.DataFrame, n: int = 5, random_state: Optional[int] = 42) -> pd.DataFrame:
    """
    Sample rows from DataFrame deterministically.

    Args:
        df: Input DataFrame
        n: Number of rows to sample
        random_state: Random seed for reproducibility

    Returns:
        Sampled DataFrame
    """
    if len(df) <= n:
        return df.copy()

    if random_state is not None:
        return df.sample(n=n, random_state=random_state).sort_index()
    else:
        return df.head(n)


def save_dataframe(df: pd.DataFrame, path: Union[str, Path], **kwargs) -> None:
    """
    Save DataFrame to file.

    Args:
        path: Output path
        df: DataFrame to save
        **kwargs: Additional arguments for pandas to_csv
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(path, index=False, **kwargs)
    logger.info(f"Saved DataFrame to: {path}")
