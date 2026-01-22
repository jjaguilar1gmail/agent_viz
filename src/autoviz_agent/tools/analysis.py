"""Analysis tools."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from autoviz_agent.registry.tools import tool
from autoviz_agent.utils.logging import get_logger

logger = get_logger(__name__)


@tool(
    description="Detect anomalies",
    param_overrides={"column": {"role": "numeric"}},
)
def detect_anomalies(
    df: pd.DataFrame, column: str, method: str = "iqr", threshold: float = 1.5
) -> pd.DataFrame:
    """
    Detect anomalies in a column.

    Args:
        df: Input DataFrame
        column: Column to analyze
        method: Detection method (iqr, zscore)
        threshold: Threshold for anomaly detection

    Returns:
        DataFrame with is_anomaly boolean column
    """
    result = df.copy()
    result["is_anomaly"] = False

    if method == "iqr":
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        result["is_anomaly"] = (df[column] < lower_bound) | (df[column] > upper_bound)

    elif method == "zscore":
        mean = df[column].mean()
        std = df[column].std()
        z_scores = np.abs((df[column] - mean) / std)
        result["is_anomaly"] = z_scores > threshold

    anomaly_count = result["is_anomaly"].sum()
    logger.info(f"Detected {anomaly_count} anomalies in {column} using {method} method")

    return result


@tool(
    description="Segment metric by category",
    param_overrides={
        "segment_by": {"role": "categorical"},
        "metric": {"role": "numeric"},
    },
)
def segment_metric(
    df: pd.DataFrame, segment_by: str, metric: str, agg: str = "mean"
) -> pd.DataFrame:
    """
    Segment a metric by a categorical column.

    Args:
        df: Input DataFrame
        segment_by: Column to segment by
        metric: Metric column
        agg: Aggregation function (mean, sum, median, etc.)

    Returns:
        Segmented DataFrame
    """
    if agg == "mean":
        result = df.groupby(segment_by)[metric].mean().reset_index()
    elif agg == "sum":
        result = df.groupby(segment_by)[metric].sum().reset_index()
    elif agg == "median":
        result = df.groupby(segment_by)[metric].median().reset_index()
    elif agg == "count":
        result = df.groupby(segment_by)[metric].count().reset_index()
    else:
        result = df.groupby(segment_by)[metric].agg(agg).reset_index()

    # Sort for determinism
    result = result.sort_values(by=segment_by).reset_index(drop=True)

    logger.info(f"Segmented {metric} by {segment_by} using {agg}")
    return result


@tool(
    description="Compute distribution statistics",
    param_overrides={"column": {"role": "numeric"}},
)
def compute_distributions(df: pd.DataFrame, column: str, bins: int = 10) -> Dict[str, Any]:
    """
    Compute distribution statistics.

    Args:
        df: Input DataFrame
        column: Column to analyze
        bins: Number of bins for histogram

    Returns:
        Distribution statistics
    """
    series = df[column].dropna()

    counts, bin_edges = np.histogram(series, bins=bins)

    dist = {
        "mean": float(series.mean()),
        "median": float(series.median()),
        "std": float(series.std()),
        "skew": float(series.skew()),
        "kurtosis": float(series.kurtosis()),
        "min": float(series.min()),
        "max": float(series.max()),
        "histogram": {
            "counts": counts.tolist(),
            "bin_edges": bin_edges.tolist(),
        },
    }

    logger.info(f"Computed distribution for {column}")
    return dist


@tool(
    description="Compare metrics across groups",
    param_overrides={
        "group_col": {"role": "categorical"},
        "metric_col": {"role": "numeric"},
    },
)
def compare_groups(
    df: pd.DataFrame, group_col: str, metric_col: str, groups: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Compare metric across groups.

    Args:
        df: Input DataFrame
        group_col: Grouping column
        metric_col: Metric to compare
        groups: Specific groups to compare (None for all)

    Returns:
        Comparison statistics
    """
    if groups is not None:
        df_filtered = df[df[group_col].isin(groups)]
    else:
        df_filtered = df

    comparison = {}
    for group in sorted(df_filtered[group_col].unique()):
        group_data = df_filtered[df_filtered[group_col] == group][metric_col]
        comparison[str(group)] = {
            "count": int(group_data.count()),
            "mean": float(group_data.mean()),
            "median": float(group_data.median()),
            "std": float(group_data.std()),
        }

    logger.info(f"Compared {metric_col} across {len(comparison)} groups in {group_col}")
    return comparison


@tool(
    description="Extract time series features",
    param_overrides={
        "date_col": {"role": "temporal"},
        "value_col": {"role": "numeric"},
    },
)
def compute_time_series_features(
    df: pd.DataFrame, date_col: str, value_col: str
) -> Dict[str, Any]:
    """
    Compute time series features.

    Args:
        df: Input DataFrame
        date_col: Date/time column
        value_col: Value column

    Returns:
        Time series features
    """
    df_sorted = df.sort_values(date_col).copy()
    values = df_sorted[value_col].values

    features = {
        "trend": "increasing" if values[-1] > values[0] else "decreasing",
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "range": float(np.max(values) - np.min(values)),
    }

    # Simple moving statistics
    if len(values) > 3:
        features["recent_mean"] = float(np.mean(values[-3:]))
        features["overall_mean"] = float(np.mean(values))

    logger.info(f"Computed time series features for {value_col}")
    return features
