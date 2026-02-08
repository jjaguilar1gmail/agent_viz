"""Schema-derived tags and data shape detection."""

from typing import Dict, List, Set

from autoviz_agent.models.state import SchemaProfile
from autoviz_agent.utils.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# Requirement-to-Capability Mapping (Deterministic)
# =============================================================================

# This is the source of truth for coverage validation.
# Each requirement type maps to a set of tool capabilities that can satisfy it.

REQUIREMENT_TO_CAPABILITY_MAP: Dict[str, List[str]] = {
    # Analysis types
    "total": ["aggregate", "summary_stats", "compute"],
    "compare": ["aggregate", "segment", "group_by", "compare"],
    "trend": ["time_series", "plot", "trend", "time_series_plot", "time_series_features"],
    "distribution": ["distribution_plot", "distribution_stats", "plot", "frequency"],
    "anomaly": ["anomaly_detection", "outlier_detection"],
    "correlation": ["correlation", "relationship", "plot"],
    
    # Output types
    "chart": ["plot", "visualization"],
    "table": ["aggregate", "summary_stats", "compute"],
    
    # Special requirements
    "group_by": ["aggregate", "segment", "group_by"],
    "time": ["time_series", "parse_datetime", "temporal"],
}

# Capability aliases for backwards compatibility
CAPABILITY_ALIASES: Dict[str, str] = {
    "summarize": "summary_stats",
    "group": "group_by",
    "temporal": "time_series",
    "viz": "plot",
    "chart": "plot",
}

# Core capabilities that should always be available
CORE_CAPABILITIES: Set[str] = {
    "aggregate",
    "plot",
    "summary_stats",
}


# =============================================================================
# Validation Functions
# =============================================================================

def get_required_capabilities(requirement_label: str) -> List[str]:
    """
    Get capabilities that can satisfy a requirement.
    
    Args:
        requirement_label: Requirement type (e.g., "total", "compare")
    
    Returns:
        List of capability names
    
    Raises:
        ValueError: If requirement label is unknown
    """
    if requirement_label not in REQUIREMENT_TO_CAPABILITY_MAP:
        # Check if it's a special case (outputs, group_by, time)
        if requirement_label in ["chart", "table", "group_by", "time"]:
            return REQUIREMENT_TO_CAPABILITY_MAP[requirement_label]
        
        raise ValueError(
            f"Unknown requirement label: '{requirement_label}'. "
            f"Allowed: {list(REQUIREMENT_TO_CAPABILITY_MAP.keys())}"
        )
    
    return REQUIREMENT_TO_CAPABILITY_MAP[requirement_label]


def normalize_capability(capability: str) -> str:
    """
    Normalize capability name using aliases.
    
    Args:
        capability: Capability name
    
    Returns:
        Normalized capability name
    """
    return CAPABILITY_ALIASES.get(capability, capability)


def infer_time_grain(
    data_span_days: int,
    num_points: int,
    missing_dates_pct: float
) -> str:
    """
    Infer appropriate time grain for plotting.
    
    Args:
        data_span_days: Total days spanned by data
        num_points: Number of data points
        missing_dates_pct: Percentage of missing dates/irregular intervals
    
    Returns:
        Time grain: "daily", "weekly", "monthly", "yearly"
    """
    # If missing dates exceed 20%, prefer weekly
    if missing_dates_pct > 0.2:
        return "weekly"
    
    # If span is < 90 days or > 60 points: daily
    if data_span_days < 90 or num_points > 60:
        return "daily"
    
    # If span is 90-365 days: weekly
    if data_span_days <= 365:
        return "weekly"
    
    # Otherwise: monthly
    return "monthly"


# =============================================================================
# Schema Tag Extraction
# =============================================================================


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
