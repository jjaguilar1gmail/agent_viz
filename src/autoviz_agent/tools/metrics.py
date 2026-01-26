"""Metrics and statistical computation tools."""

from typing import Any, Dict, List, Optional

import pandas as pd

from autoviz_agent.registry.tools import tool
from autoviz_agent.utils.logging import get_logger

logger = get_logger(__name__)


@tool(
    description="Aggregate data by groups",
    param_overrides={
        "group_by": {"required": False, "default": "auto"},
        "agg_map": {"required": False, "default": "auto"},
    },
)
def aggregate(
    df: pd.DataFrame, group_by: List[str], agg_map: Dict[str, str]
) -> pd.DataFrame:
    """
    Aggregate data by groups.

    Args:
        df: Input DataFrame
        group_by: Columns to group by
        agg_map: Dictionary of column -> aggregation function (sum, mean, count, etc.)

    Returns:
        Aggregated DataFrame
    """
    result = df.groupby(group_by, as_index=False).agg(agg_map)
    logger.info(f"Aggregated data by {group_by}")
    return result


@tool(
    description="Compute summary statistics",
    param_overrides={"columns": {"role": "numeric"}},
)
def compute_summary_stats(df: pd.DataFrame, columns: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Compute summary statistics.

    Args:
        df: Input DataFrame
        columns: Columns to compute stats for (None for all numeric)

    Returns:
        Dictionary of statistics
    """
    if columns is None:
        columns = df.select_dtypes(include=["number"]).columns.tolist()

    stats = {}
    for col in columns:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            stats[col] = {
                "count": int(df[col].count()),
                "mean": float(df[col].mean()),
                "std": float(df[col].std()),
                "min": float(df[col].min()),
                "25%": float(df[col].quantile(0.25)),
                "50%": float(df[col].quantile(0.50)),
                "75%": float(df[col].quantile(0.75)),
                "max": float(df[col].max()),
            }

    logger.info(f"Computed summary stats for {len(stats)} columns")
    return stats


@tool(description="Compute correlation matrix")
def compute_correlations(df: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    """
    Compute correlation matrix.

    Args:
        df: Input DataFrame
        method: Correlation method (pearson, spearman, kendall)

    Returns:
        Correlation matrix
    """
    numeric_df = df.select_dtypes(include=["number"])
    corr = numeric_df.corr(method=method)
    logger.info(f"Computed {method} correlations for {len(corr)} columns")
    return corr


@tool(
    description="Count unique values",
    param_overrides={"column": {"role": "categorical"}},
)
def compute_value_counts(
    df: pd.DataFrame, column: str, top_n: Optional[int] = 10, normalize: bool = False
) -> pd.Series:
    """
    Compute value counts for a column.

    Args:
        df: Input DataFrame
        column: Column name
        top_n: Number of top values to return (None for all)
        normalize: Whether to return proportions

    Returns:
        Value counts series
    """
    counts = df[column].value_counts(normalize=normalize)

    if top_n is not None:
        counts = counts.head(top_n)

    logger.info(f"Computed value counts for {column}")
    return counts


@tool(
    description="Compute percentiles",
    param_overrides={"column": {"role": "numeric"}},
)
def compute_percentiles(
    df: pd.DataFrame, column: str, percentiles: List[float]
) -> Dict[float, float]:
    """
    Compute percentiles for a column.

    Args:
        df: Input DataFrame
        column: Column name
        percentiles: List of percentiles (0-1)

    Returns:
        Dictionary of percentile -> value
    """
    result = {}
    for p in percentiles:
        result[p] = float(df[column].quantile(p))

    logger.info(f"Computed {len(percentiles)} percentiles for {column}")
    return result
