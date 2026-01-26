"""Parameter resolution for tool calls."""

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from autoviz_agent.io.artifacts import ArtifactManager
from autoviz_agent.models.state import SchemaProfile
from autoviz_agent.registry.tools import TOOL_REGISTRY, ensure_default_tools_registered
from autoviz_agent.runtime.column_selectors import ColumnSelector
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

        ensure_default_tools_registered()
        
        # Use ColumnSelector for all column selections
        self.column_selector = ColumnSelector(schema, user_question)

    def resolve(self, tool_name: str, params: Dict[str, Any], sequence: int = 1) -> Dict[str, Any]:
        """
        Resolve parameters for a tool call using registry schema and column selectors.

        Args:
            tool_name: Name of the tool
            params: Provided parameters
            sequence: Sequence number for unique naming

        Returns:
            Resolved parameters
        """
        resolved = params.copy()
        
        # Get tool schema from registry
        schema = TOOL_REGISTRY.get_schema(tool_name)
        if not schema:
            logger.warning(f"No schema found for tool: {tool_name}")
            return resolved
        
        # Generate output_path for plot tools
        if tool_name.startswith("plot_"):
            if "output_path" not in resolved or resolved.get("output_path") is None:
                plot_type = tool_name.replace("plot_", "")
                if self.artifact_manager:
                    resolved["output_path"] = str(
                        self.artifact_manager.get_path("chart", f"{plot_type}_{sequence}.png")
                    )
                else:
                    resolved["output_path"] = f"{plot_type}_{sequence}.png"
        
        # Special handling for hue (before schema resolution) - use second mentioned categorical
        if tool_name == "plot_bar" and "hue" not in resolved:
            mentioned = self.column_selector.get_mentioned_cols()
            if len(mentioned['categorical']) >= 2:
                resolved["hue"] = mentioned['categorical'][1]
                logger.info(f"Using second mentioned column as hue: {resolved['hue']}")
        
        # Resolve parameters using schema-driven defaults and role-based column selection
        for param in schema.parameters:
            param_name = param.name
            
            # Skip if parameter already provided and not "auto"
            if param_name in resolved and resolved[param_name] != "auto":
                continue
            
            # Handle "auto" values or missing parameters
            if param_name not in resolved or resolved[param_name] == "auto":
                # Use role-based column selection if role is specified
                if param.role:
                    if param.role in ["temporal", "numeric", "categorical"]:
                        exclude = []
                        if tool_name == "plot_scatter" and param_name == "y" and "x" in resolved:
                            exclude = [resolved["x"]]
                        selected = self.column_selector.select(param.role, count=1, exclude=exclude)
                        if selected:
                            resolved[param_name] = selected[0]
                    elif param.role == "any":
                        selected = self.column_selector.select("any", count=1)
                        if selected:
                            resolved[param_name] = selected[0]
                # Use default value if provided in schema
                elif param.default is not None:
                    resolved[param_name] = param.default
                # Fallback: infer role from parameter name patterns
                else:
                    inferred_role = self._infer_param_role(param_name, tool_name)
                    if inferred_role:
                        selected = self.column_selector.select(inferred_role, count=1)
                        if selected:
                            resolved[param_name] = selected[0]
        
        # Special handling for multi-column parameters (group_by, columns)
        if "group_by" in resolved and (resolved["group_by"] == "auto" or resolved["group_by"] == ["auto"]):
            selected = self.column_selector.select("categorical", count=2)
            resolved["group_by"] = selected
            if selected:
                logger.info(f"Selected group_by columns: {selected}")
        
        # Special handling for agg_map
        if "agg_map" in resolved and resolved["agg_map"] == "auto":
            numeric_cols = self.column_selector.select("numeric", count=1)
            if numeric_cols:
                resolved["agg_map"] = {numeric_cols[0]: "sum"}
                logger.info(f"Selected agg_map: {resolved['agg_map']}")
        
        # Apply known parameter aliases (normalize)
        resolved = self._normalize_aliases(resolved, tool_name)
        
        # Remove known invalid parameters
        resolved = self._remove_invalid_params(resolved, tool_name)
        
        return resolved
    
    def _infer_param_role(self, param_name: str, tool_name: str) -> Optional[str]:
        """
        Infer parameter role from common naming patterns.
        
        Args:
            param_name: Parameter name
            tool_name: Tool name for context
            
        Returns:
            Inferred role or None
        """
        # Common patterns for temporal parameters
        if param_name in ["x"] and tool_name == "plot_line":
            return "temporal"
        
        # Common patterns for numeric parameters
        if param_name in ["y", "metric", "column"] or "value" in param_name.lower():
            return "numeric"
        
        # Common patterns for categorical parameters
        if param_name in ["x", "hue", "segment_by"] and tool_name != "plot_line":
            return "categorical"
        
        # For scatter plots, x and y are both numeric
        if tool_name == "plot_scatter" and param_name in ["x", "y"]:
            return "numeric"
        
        return None
    
    def _normalize_aliases(self, params: Dict[str, Any], tool_name: str) -> Dict[str, Any]:
        """Normalize known parameter aliases."""
        # For plot_heatmap, df parameter should be data for seaborn
        if tool_name == "plot_heatmap" and "df" in params and "data" not in params:
            params["data"] = params.pop("df")
        
        # annotation -> annot for heatmap
        if "annotation" in params and "annot" not in params:
            params["annot"] = params.pop("annotation")
        return params
    
    def _remove_invalid_params(self, params: Dict[str, Any], tool_name: str) -> Dict[str, Any]:
        """Remove known invalid parameters for specific tools."""
        # Known invalid parameters by tool
        invalid_params = {
            "plot_line": ["show_trend"],
            "plot_scatter": ["highlight_anomalies"],
            "plot_histogram": ["max_columns"],
            "parse_datetime": ["auto_detect"],
        }
        
        if tool_name in invalid_params:
            for invalid in invalid_params[tool_name]:
                params.pop(invalid, None)
        
        return params
