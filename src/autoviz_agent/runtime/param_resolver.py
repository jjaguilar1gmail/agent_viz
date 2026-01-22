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
    ):
        """
        Initialize parameter resolver.

        Args:
            schema: Dataset schema profile
            artifact_manager: Artifact manager for generating output paths
        """
        self.schema = schema
        self.artifact_manager = artifact_manager
        self._temporal_cols = [c.name for c in schema.columns if 'temporal' in c.roles]
        self._numeric_cols = [c.name for c in schema.columns if c.dtype in ['int64', 'float64', 'int32', 'float32']]
        self._categorical_cols = [c.name for c in schema.columns if 'categorical' in c.roles]

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
            if self._categorical_cols:
                params["x"] = self._categorical_cols[0]
        if "y" not in params or params.get("y") == "auto":
            if self._numeric_cols:
                params["y"] = self._numeric_cols[0]
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
            if self._categorical_cols:
                params["group_by"] = self._categorical_cols[:2]  # Use up to 2 categorical columns
            else:
                params["group_by"] = []
        
        # Handle agg_map parameter
        if "agg_map" not in params or params.get("agg_map") == "auto":
            if self._numeric_cols:
                # Default to sum aggregation of all numeric columns
                params["agg_map"] = {col: "sum" for col in self._numeric_cols}
            else:
                params["agg_map"] = {}
        
        return params

    def _resolve_segment_metric(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve parameters for segment_metric tool."""
        if "segment_by" not in params or params.get("segment_by") == "auto":
            if self._categorical_cols:
                params["segment_by"] = self._categorical_cols[0]
        if "metric" not in params or params.get("metric") == "auto":
            if self._numeric_cols:
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
