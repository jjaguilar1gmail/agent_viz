"""Schema-derived tags and data shape detection."""

from typing import List, Set

from autoviz_agent.models.state import SchemaProfile
from autoviz_agent.utils.logging import get_logger

logger = get_logger(__name__)


def extract_schema_tags(schema: SchemaProfile) -> Set[str]:
    """
    Extract tags from schema profile for template matching.

    Args:
        schema: Schema profile

    Returns:
        Set of tags describing the schema
    """
    tags = set()

    # Data shape tags
    tags.add(f"shape:{schema.data_shape}")

    # Size tags
    if schema.row_count < 100:
        tags.add("size:small")
    elif schema.row_count < 10000:
        tags.add("size:medium")
    else:
        tags.add("size:large")

    column_count = len(schema.columns)
    if column_count < 5:
        tags.add("columns:few")
    elif column_count < 20:
        tags.add("columns:moderate")
    else:
        tags.add("columns:many")

    # Column type tags
    role_counts = {}
    for col in schema.columns:
        for role in col.roles:
            role_counts[role] = role_counts.get(role, 0) + 1

    if role_counts.get("datetime", 0) > 0:
        tags.add("has:datetime")
    if role_counts.get("categorical", 0) > 0:
        tags.add("has:categorical")
    if role_counts.get("numeric", 0) > 0:
        tags.add("has:numeric")
    if role_counts.get("metric", 0) > 0:
        tags.add("has:metric")
    if role_counts.get("id", 0) > 0:
        tags.add("has:id")

    # Data quality tags
    high_missing_cols = sum(1 for col in schema.columns if col.missing_rate > 0.1)
    if high_missing_cols > 0:
        tags.add("quality:missing_values")

    # Cardinality tags
    high_cardinality_cols = sum(1 for col in schema.columns if col.cardinality > 100)
    if high_cardinality_cols > 0:
        tags.add("cardinality:high")

    logger.info(f"Extracted {len(tags)} schema tags")
    return tags


def compute_tag_overlap(schema_tags: Set[str], template_tags: Set[str]) -> float:
    """
    Compute overlap score between schema and template tags.

    Args:
        schema_tags: Tags from schema
        template_tags: Tags from template requirements

    Returns:
        Overlap score (0-1)
    """
    if not template_tags:
        return 1.0  # No requirements means always matches

    intersection = schema_tags & template_tags
    overlap = len(intersection) / len(template_tags)

    return overlap


def infer_analysis_hints(schema: SchemaProfile) -> List[str]:
    """
    Infer analysis hints from schema.

    Args:
        schema: Schema profile

    Returns:
        List of suggested analysis types
    """
    hints = []

    # Datetime column suggests time series analysis
    has_datetime = any("datetime" in col.roles for col in schema.columns)
    if has_datetime:
        hints.append("time_series_investigation")

    # Many categorical columns suggest segmentation
    categorical_count = sum(1 for col in schema.columns if "categorical" in col.roles)
    if categorical_count >= 2:
        hints.append("segmentation_drivers")
        hints.append("comparative_analysis")

    # Numeric columns with high variance might have anomalies
    numeric_count = sum(1 for col in schema.columns if "numeric" in col.roles)
    if numeric_count > 0:
        hints.append("anomaly_detection")

    # Default to general EDA
    if not hints or len(hints) == 0:
        hints.append("general_eda")

    logger.info(f"Inferred analysis hints: {hints}")
    return hints
