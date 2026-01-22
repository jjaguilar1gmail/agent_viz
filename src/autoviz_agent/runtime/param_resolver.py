"""Parameter resolution for tool calls."""

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from autoviz_agent.io.artifacts import ArtifactManager
from autoviz_agent.models.state import SchemaProfile
from autoviz_agent.registry.tools import TOOL_REGISTRY
from autoviz_agent.utils.logging import get_logger

logger = get_logger(__name__)


class ParamResolver:
    """Resolve tool call parameters with schema-derived defaults."""

    def __init__(
        self,
        schema: SchemaProfile,
        artifact_manager: Optional[ArtifactManager] = None,
        user_question: Optional[str] = None,
    ):
        """
        Initialize parameter resolver.

        Args:
            schema: Dataset schema profile
            artifact_manager: Artifact manager for generating output paths
            user_question: User's original question for extracting mentioned columns
        """
        self.schema = schema
        self.artifact_manager = artifact_manager
        self.user_question = user_question or ""
        self._temporal_cols = [c.name for c in schema.columns if 'temporal' in c.roles]
        self._numeric_cols = [c.name for c in schema.columns if c.dtype in ['int64', 'float64', 'int32', 'float32']]
        # Get categorical columns, excluding temporal ones for better grouping defaults
        self._categorical_cols = [
            c.name for c in schema.columns 
            if 'categorical' in c.roles and 'temporal' not in c.roles
        ]
        # Keep all categorical including temporal as backup
        self._all_categorical_cols = [c.name for c in schema.columns if 'categorical' in c.roles]
        # Extract columns mentioned in user question
        self._mentioned_cols = self._extract_mentioned_columns()
        # Extract columns mentioned in user question
        self._mentioned_cols = self._extract_mentioned_columns()

    def _extract_mentioned_columns(self) -> Dict[str, List[str]]:
        """
        Extract column names mentioned in user question using keyword matching.
        
        Returns:
            Dict with 'categorical' and 'numeric' lists of mentioned columns
        """
        if not self.user_question:
            return {'categorical': [], 'numeric': []}
        
        question_lower = self.user_question.lower()
        mentioned_categorical = []
        mentioned_numeric = []
        
        # Check each column name to see if it's mentioned in the question
        # Handle variations like "product type" matching "product_category"
        for col in self._categorical_cols + self._all_categorical_cols:
            col_variants = [
                col.lower(),
                col.lower().replace('_', ' '),  # product_category -> product category
                col.lower().replace('_', ''),   # product_category -> productcategory
            ]
            # Also check for partial matches (e.g., "product" in "product type" matching "product_category")
            col_parts = col.lower().split('_')
            
            # Check exact and underscore variants
            if any(variant in question_lower for variant in col_variants):
                if col not in mentioned_categorical:
                    mentioned_categorical.append(col)
            # Check if any significant part of column name appears in question
            # (e.g., "product" from "product_category" matching "product type")
            elif any(len(part) > 3 and part in question_lower for part in col_parts):
                if col not in mentioned_categorical:
                    mentioned_categorical.append(col)
        
        for col in self._numeric_cols:
            col_variants = [
                col.lower(),
                col.lower().replace('_', ' '),
                col.lower().replace('_', ''),
            ]
            if any(variant in question_lower for variant in col_variants):
                if col not in mentioned_numeric:
                    mentioned_numeric.append(col)
        
        if mentioned_categorical or mentioned_numeric:
            logger.info(f"Extracted from question - categorical: {mentioned_categorical}, numeric: {mentioned_numeric}")
        
        return {'categorical': mentioned_categorical, 'numeric': mentioned_numeric}

    def resolve(self, tool_name: str, params: Dict[str, Any], sequence: int = 1) -> Dict[str, Any]:
        """
        Resolve parameters for a tool call, filling in missing required params with defaults.

        Args:
            tool_name: Name of the tool
            params: Provided parameters
            sequence: Sequence number for unique naming

        Returns:
            Resolved parameters
        """
        resolved = params.copy()

        # Apply tool-specific resolution strategies
        if tool_name == "parse_datetime":
            resolved = self._resolve_parse_datetime(resolved)
        elif tool_name.startswith("plot_"):
            resolved = self._resolve_plot_params(tool_name, resolved, sequence)
        elif tool_name == "aggregate":
            resolved = self._resolve_aggregate(resolved)
        elif tool_name == "segment_metric":
            resolved = self._resolve_segment_metric(resolved)
        elif tool_name == "compute_distributions":
            resolved = self._resolve_compute_distributions(resolved)
        elif tool_name == "detect_anomalies":
            resolved = self._resolve_detect_anomalies(resolved)

        return resolved

    def _resolve_parse_datetime(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve parameters for parse_datetime tool."""
        if "columns" not in params or params["columns"] is None:
            # Use temporal columns from schema or detect date-like column names
            if self._temporal_cols:
                params["columns"] = self._temporal_cols
            else:
                date_cols = [
                    c.name for c in self.schema.columns
                    if 'date' in c.name.lower() or 'time' in c.name.lower()
                ]
                if date_cols:
                    params["columns"] = date_cols

        # Remove invalid parameters
        params.pop("auto_detect", None)
        return params

    def _resolve_plot_params(
        self, tool_name: str, params: Dict[str, Any], sequence: int
    ) -> Dict[str, Any]:
        """Resolve common plotting parameters."""
        # Generate output_path if missing
        if "output_path" not in params or params["output_path"] is None:
            plot_type = tool_name.replace("plot_", "")
            if self.artifact_manager:
                params["output_path"] = str(
                    self.artifact_manager.get_path("chart", f"{plot_type}_{sequence}.png")
                )
            else:
                params["output_path"] = f"{plot_type}_{sequence}.png"

        # Tool-specific parameter resolution
        if tool_name == "plot_line":
            params = self._resolve_plot_line(params)
        elif tool_name == "plot_bar":
            params = self._resolve_plot_bar(params)
        elif tool_name == "plot_scatter":
            params = self._resolve_plot_scatter(params)
        elif tool_name == "plot_histogram":
            params = self._resolve_plot_histogram(params)
        elif tool_name == "plot_heatmap":
            params = self._resolve_plot_heatmap(params)
        elif tool_name == "plot_boxplot":
            params = self._resolve_plot_boxplot(params)

        return params

    def _resolve_plot_line(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve parameters for line plot."""
        if "x" not in params and self._temporal_cols:
            params["x"] = self._temporal_cols[0]
        if "y" not in params and self._numeric_cols:
            params["y"] = self._numeric_cols[0]
        # Remove invalid parameters
        params.pop("show_trend", None)
        return params

    def _resolve_plot_bar(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve parameters for bar plot."""
        if "x" not in params or params.get("x") == "auto":
            # Priority 1: Categorical mentioned in question
            if self._mentioned_cols['categorical']:
                params["x"] = self._mentioned_cols['categorical'][0]
            # Priority 2: Non-temporal categorical
            elif self._categorical_cols:
                params["x"] = self._categorical_cols[0]
            # Priority 3: All categorical
            elif self._all_categorical_cols:
                params["x"] = self._all_categorical_cols[0]
        if "y" not in params or params.get("y") == "auto":
            # Priority 1: Numeric mentioned in question
            if self._mentioned_cols['numeric']:
                params["y"] = self._mentioned_cols['numeric'][0]
            # Priority 2: First numeric column
            elif self._numeric_cols:
                params["y"] = self._numeric_cols[0]
        # Add hue for second categorical dimension
        if "hue" not in params:
            # If user mentioned 2+ categoricals, use second one for hue
            if len(self._mentioned_cols['categorical']) >= 2:
                params["hue"] = self._mentioned_cols['categorical'][1]
                logger.info(f"Using second mentioned column as hue: {params['hue']}")
        return params

    def _resolve_plot_scatter(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve parameters for scatter plot."""
        if "x" not in params and self._numeric_cols:
            params["x"] = self._numeric_cols[0] if len(self._numeric_cols) > 0 else "index"
        if "y" not in params and self._numeric_cols:
            params["y"] = self._numeric_cols[1] if len(self._numeric_cols) > 1 else self._numeric_cols[0]
        # Remove invalid parameters
        params.pop("highlight_anomalies", None)
        return params

    def _resolve_plot_histogram(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve parameters for histogram."""
        if "column" not in params and self._numeric_cols:
            params["column"] = self._numeric_cols[0]
        # Remove invalid parameters
        params.pop("max_columns", None)
        return params

    def _resolve_plot_heatmap(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve parameters for heatmap."""
        # Rename 'df' to 'data' if present
        if "df" in params:
            params["data"] = params.pop("df")
        # Rename 'annotation' to 'annot' if present
        if "annotation" in params:
            params["annot"] = params.pop("annotation")
        return params

    def _resolve_plot_boxplot(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve parameters for boxplot."""
        if "column" not in params and self._numeric_cols:
            params["column"] = self._numeric_cols[0]
        return params

    def _resolve_aggregate(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve parameters for aggregate tool."""
        # Handle group_by parameter
        if "group_by" not in params or params.get("group_by") == "auto" or params.get("group_by") == ["auto"]:
            # Priority 1: Use columns mentioned in user question
            if self._mentioned_cols['categorical']:
                params["group_by"] = self._mentioned_cols['categorical'][:2]
                logger.info(f"Using mentioned columns for grouping: {params['group_by']}")
            # Priority 2: Non-temporal categorical columns
            elif self._categorical_cols:
                params["group_by"] = self._categorical_cols[:2]  # Use up to 2 categorical columns
            # Priority 3: All categorical (including temporal) as last resort
            elif self._all_categorical_cols:
                params["group_by"] = self._all_categorical_cols[:2]
            else:
                params["group_by"] = []
        
        # Handle agg_map parameter
        if "agg_map" not in params or params.get("agg_map") == "auto":
            # Priority 1: Use numeric column mentioned in question
            if self._mentioned_cols['numeric']:
                params["agg_map"] = {self._mentioned_cols['numeric'][0]: "sum"}
                logger.info(f"Using mentioned metric: {self._mentioned_cols['numeric'][0]}")
            # Priority 2: All numeric columns
            elif self._numeric_cols:
                params["agg_map"] = {col: "sum" for col in self._numeric_cols}
            else:
                params["agg_map"] = {}
        
        return params

    def _resolve_segment_metric(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve parameters for segment_metric tool."""
        if "segment_by" not in params or params.get("segment_by") == "auto":
            # Priority 1: Categorical mentioned in question
            if self._mentioned_cols['categorical']:
                params["segment_by"] = self._mentioned_cols['categorical'][0]
            # Priority 2: Non-temporal categorical
            elif self._categorical_cols:
                params["segment_by"] = self._categorical_cols[0]
            # Priority 3: All categorical
            elif self._all_categorical_cols:
                params["segment_by"] = self._all_categorical_cols[0]
        if "metric" not in params or params.get("metric") == "auto":
            # Priority 1: Numeric mentioned in question
            if self._mentioned_cols['numeric']:
                params["metric"] = self._mentioned_cols['numeric'][0]
            # Priority 2: First numeric column
            elif self._numeric_cols:
                params["metric"] = self._numeric_cols[0]
        if "agg" not in params:
            params["agg"] = "mean"
        return params

    def _resolve_compute_distributions(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve parameters for compute_distributions tool."""
        if "column" not in params and self._numeric_cols:
            params["column"] = self._numeric_cols[0]
        return params

    def _resolve_detect_anomalies(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve parameters for detect_anomalies tool."""
        if "column" not in params and self._numeric_cols:
            params["column"] = self._numeric_cols[0]
        return params
