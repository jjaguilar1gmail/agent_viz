"""Schema inference tools."""

from typing import Any, Dict, List

import pandas as pd

from autoviz_agent.models.state import ColumnProfile, SchemaProfile
from autoviz_agent.utils.logging import get_logger

logger = get_logger(__name__)


def infer_schema(df: pd.DataFrame) -> SchemaProfile:
    """
    Infer dataset schema and column profiles.

    Args:
        df: Input DataFrame

    Returns:
        SchemaProfile with column details
    """
    columns = []

    for col in df.columns:
        profile = _infer_column_profile(df, col)
        columns.append(profile)

    data_shape = _detect_data_shape(df)

    schema = SchemaProfile(columns=columns, row_count=len(df), data_shape=data_shape)

    logger.info(f"Inferred schema: {len(columns)} columns, {len(df)} rows, shape={data_shape}")
    return schema


def _infer_column_profile(df: pd.DataFrame, col: str) -> ColumnProfile:
    """Infer profile for a single column."""
    series = df[col]

    # Data type
    dtype = str(series.dtype)

    # Missing rate
    missing_rate = series.isna().sum() / len(series)

    # Cardinality
    cardinality = series.nunique()

    # Infer roles
    roles = _infer_column_roles(series, col)

    return ColumnProfile(
        name=col, dtype=dtype, missing_rate=missing_rate, cardinality=cardinality, roles=roles
    )


def _infer_column_roles(series: pd.Series, col_name: str) -> List[str]:
    """Infer semantic roles for a column."""
    roles = []

    col_lower = col_name.lower()

    # ID columns
    if any(kw in col_lower for kw in ["id", "_id", "key"]):
        roles.append("id")

    # Temporal columns
    if pd.api.types.is_datetime64_any_dtype(series):
        roles.append("datetime")
    elif any(kw in col_lower for kw in ["date", "time", "timestamp", "year", "month"]):
        roles.append("temporal")

    # Categorical
    if pd.api.types.is_categorical_dtype(series) or pd.api.types.is_object_dtype(series):
        if series.nunique() < len(series) * 0.5:  # Less than 50% unique
            roles.append("categorical")

    # Numeric
    if pd.api.types.is_numeric_dtype(series):
        roles.append("numeric")

        # Metrics vs dimensions
        if series.nunique() > len(series) * 0.5:  # High cardinality
            roles.append("metric")
        else:
            roles.append("dimension")

    return roles


def _detect_data_shape(df: pd.DataFrame) -> str:
    """
    Detect dataset shape/structure.

    Returns:
        Shape label: wide, long, time_series, or unknown
    """
    # Time series: has datetime column
    datetime_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
    if datetime_cols:
        return "time_series"

    # Wide vs long heuristics
    n_cols = len(df.columns)
    n_rows = len(df)

    if n_cols > n_rows * 0.1:  # Many columns relative to rows
        return "wide"
    elif n_rows > n_cols * 10:  # Many rows relative to columns
        return "long"

    return "unknown"


def get_schema_summary(schema: SchemaProfile) -> str:
    """
    Get a text summary of the schema.

    Args:
        schema: Schema profile

    Returns:
        Summary string
    """
    lines = [
        f"Dataset: {schema.row_count} rows, {len(schema.columns)} columns",
        f"Shape: {schema.data_shape}",
        "",
        "Columns:",
    ]

    for col in schema.columns:
        roles_str = ", ".join(col.roles) if col.roles else "none"
        lines.append(f"  - {col.name}: {col.dtype} (missing={col.missing_rate:.1%}, roles={roles_str})")

    return "\n".join(lines)
