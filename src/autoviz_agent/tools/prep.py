"""Data preparation tools."""

from typing import Any, Dict, List, Optional

import pandas as pd

from autoviz_agent.registry.tools import tool
from autoviz_agent.utils.logging import get_logger

logger = get_logger(__name__)


@tool(
    description="Handle missing values",
    param_overrides={"columns": {"role": "any"}},
)
def handle_missing(
    df: pd.DataFrame, strategy: str = "drop", columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Handle missing values.

    Args:
        df: Input DataFrame
        strategy: Strategy (drop, fill_mean, fill_median, fill_mode, fill_zero)
        columns: Specific columns to apply strategy to (None for all)

    Returns:
        DataFrame with missing values handled
    """
    result = df.copy()

    if columns is None:
        columns = result.columns.tolist()

    if strategy == "drop":
        result = result.dropna(subset=columns)
        logger.info(f"Dropped rows with missing values in {len(columns)} columns")
    elif strategy == "fill_mean":
        for col in columns:
            if pd.api.types.is_numeric_dtype(result[col]):
                result[col] = result[col].fillna(result[col].mean())
    elif strategy == "fill_median":
        for col in columns:
            if pd.api.types.is_numeric_dtype(result[col]):
                result[col] = result[col].fillna(result[col].median())
    elif strategy == "fill_mode":
        for col in columns:
            mode_val = result[col].mode()
            if len(mode_val) > 0:
                result[col] = result[col].fillna(mode_val[0])
    elif strategy == "fill_zero":
        result[columns] = result[columns].fillna(0)
    else:
        logger.warning(f"Unknown strategy: {strategy}, no action taken")

    return result


@tool(
    description="Parse datetime columns",
    param_overrides={"columns": {"role": "temporal"}},
)
def parse_datetime(
    df: pd.DataFrame, columns: List[str], format: Optional[str] = None
) -> pd.DataFrame:
    """
    Parse datetime columns.

    Args:
        df: Input DataFrame
        columns: Columns to parse as datetime
        format: Optional datetime format string

    Returns:
        DataFrame with parsed datetime columns
    """
    result = df.copy()

    for col in columns:
        if col in result.columns:
            result[col] = pd.to_datetime(result[col], format=format, errors="coerce")
            logger.info(f"Parsed datetime column: {col}")

    return result


@tool(description="Convert column types")
def cast_types(df: pd.DataFrame, type_map: Dict[str, str]) -> pd.DataFrame:
    """
    Cast column types.

    Args:
        df: Input DataFrame
        type_map: Dictionary of column_name -> dtype

    Returns:
        DataFrame with casted types
    """
    result = df.copy()

    for col, dtype in type_map.items():
        if col in result.columns:
            try:
                if dtype == "category":
                    result[col] = result[col].astype("category")
                elif dtype in ["int", "int64"]:
                    result[col] = pd.to_numeric(result[col], errors="coerce").astype("Int64")
                elif dtype in ["float", "float64"]:
                    result[col] = pd.to_numeric(result[col], errors="coerce").astype("float64")
                elif dtype == "string":
                    result[col] = result[col].astype(str)
                else:
                    result[col] = result[col].astype(dtype)

                logger.info(f"Cast column {col} to {dtype}")
            except Exception as e:
                logger.warning(f"Failed to cast {col} to {dtype}: {e}")

    return result


@tool(description="Clean column names")
def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names (lowercase, replace spaces with underscores).

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with normalized column names
    """
    result = df.copy()
    result.columns = [col.lower().replace(" ", "_").replace("-", "_") for col in result.columns]
    logger.info("Normalized column names")
    return result
