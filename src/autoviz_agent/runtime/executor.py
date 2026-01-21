"""Tool executor and registry dispatch."""

import time
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from autoviz_agent.io.artifacts import ArtifactManager
from autoviz_agent.registry.schemas import ToolCallResult
from autoviz_agent.registry.tools import TOOL_REGISTRY
from autoviz_agent.reporting.execution_log import ExecutionLog
from autoviz_agent.tools import analysis, data_io, metrics, prep, schema, visualization
from autoviz_agent.utils.logging import get_logger

logger = get_logger(__name__)


class ToolExecutor:
    """Execute tool calls with registry dispatch."""

    def __init__(self, artifact_manager: ArtifactManager):
        """
        Initialize tool executor.

        Args:
            artifact_manager: Artifact manager for saving outputs
        """
        self.artifact_manager = artifact_manager
        self.execution_log = ExecutionLog()
        self._register_tools()

    def _register_tools(self) -> None:
        """Register all available tools."""
        # Data I/O tools
        from autoviz_agent.registry.tools import ToolSchema

        TOOL_REGISTRY.register(
            ToolSchema(name="load_dataset", description="Load dataset from file", returns="DataFrame"),
            data_io.load_dataset,
        )
        TOOL_REGISTRY.register(
            ToolSchema(name="sample_rows", description="Sample rows from DataFrame", returns="DataFrame"),
            data_io.sample_rows,
        )

        # Schema tools
        TOOL_REGISTRY.register(
            ToolSchema(name="infer_schema", description="Infer dataset schema", returns="SchemaProfile"),
            schema.infer_schema,
        )

        # Preparation tools
        TOOL_REGISTRY.register(
            ToolSchema(name="handle_missing", description="Handle missing values", returns="DataFrame"),
            prep.handle_missing,
        )
        TOOL_REGISTRY.register(
            ToolSchema(name="parse_datetime", description="Parse datetime columns", returns="DataFrame"),
            prep.parse_datetime,
        )

        # Metrics tools
        TOOL_REGISTRY.register(
            ToolSchema(name="compute_summary_stats", description="Compute summary statistics", returns="Dict"),
            metrics.compute_summary_stats,
        )
        TOOL_REGISTRY.register(
            ToolSchema(name="compute_correlations", description="Compute correlation matrix", returns="DataFrame"),
            metrics.compute_correlations,
        )

        # Analysis tools
        TOOL_REGISTRY.register(
            ToolSchema(name="detect_anomalies", description="Detect anomalies", returns="DataFrame"),
            analysis.detect_anomalies,
        )
        TOOL_REGISTRY.register(
            ToolSchema(name="segment_metric", description="Segment metric by category", returns="DataFrame"),
            analysis.segment_metric,
        )
        TOOL_REGISTRY.register(
            ToolSchema(name="compute_distributions", description="Compute distribution statistics", returns="Dict"),
            analysis.compute_distributions,
        )

        # Visualization tools
        TOOL_REGISTRY.register(
            ToolSchema(name="plot_line", description="Create line plot", returns="Path"),
            visualization.plot_line,
        )
        TOOL_REGISTRY.register(
            ToolSchema(name="plot_bar", description="Create bar plot", returns="Path"),
            visualization.plot_bar,
        )
        TOOL_REGISTRY.register(
            ToolSchema(name="plot_scatter", description="Create scatter plot", returns="Path"),
            visualization.plot_scatter,
        )
        TOOL_REGISTRY.register(
            ToolSchema(name="plot_histogram", description="Create histogram", returns="Path"),
            visualization.plot_histogram,
        )
        TOOL_REGISTRY.register(
            ToolSchema(name="plot_heatmap", description="Create heatmap", returns="Path"),
            visualization.plot_heatmap,
        )

        logger.info(f"Registered {len(TOOL_REGISTRY.list_tools())} tools")

    def execute(self, tool_call: Dict[str, Any], context: Dict[str, Any]) -> ToolCallResult:
        """
        Execute a single tool call.

        Args:
            tool_call: Tool call specification
            context: Execution context (dataset, etc.)

        Returns:
            Tool call result
        """
        tool_name = tool_call["tool"]
        args = tool_call.get("args", {})
        sequence = tool_call["sequence"]

        start_time = time.time()

        try:
            tool_func = TOOL_REGISTRY.get_tool(tool_name)
            if not tool_func:
                raise ValueError(f"Tool not found: {tool_name}")

            # Inject context into args if needed
            resolved_args = self._resolve_args(args, context)

            # Execute tool
            result = tool_func(**resolved_args)

            # Handle result (save to context or artifact)
            outputs = self._handle_result(tool_name, result, context)

            duration_ms = (time.time() - start_time) * 1000

            tool_result = ToolCallResult(
                tool=tool_name,
                sequence=sequence,
                success=True,
                outputs=outputs,
                duration_ms=duration_ms,
                warnings=[],
                errors=[],
            )

            self.execution_log.add_entry(
                sequence=sequence,
                tool=tool_name,
                args=resolved_args,
                result=outputs,
                duration_ms=duration_ms,
                status="success",
            )

            logger.info(f"Executed tool: {tool_name} (seq={sequence}, duration={duration_ms:.2f}ms)")
            return tool_result

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000

            tool_result = ToolCallResult(
                tool=tool_name,
                sequence=sequence,
                success=False,
                outputs={},
                duration_ms=duration_ms,
                warnings=[],
                errors=[str(e)],
            )

            self.execution_log.add_entry(
                sequence=sequence,
                tool=tool_name,
                args=args,
                result={"error": str(e)},
                duration_ms=duration_ms,
                status="error",
            )

            logger.error(f"Tool execution failed: {tool_name} - {e}")
            return tool_result

    def _resolve_args(self, args: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve arguments with context references."""
        resolved = {}
        for key, value in args.items():
            if isinstance(value, str) and value.startswith("$"):
                # Context reference like "$dataset"
                context_key = value[1:]
                resolved[key] = context.get(context_key)
            else:
                resolved[key] = value
        return resolved

    def _handle_result(self, tool_name: str, result: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tool result and update context."""
        outputs = {}

        if isinstance(result, pd.DataFrame):
            # Store DataFrame in context
            outputs["type"] = "dataframe"
            outputs["shape"] = result.shape
            outputs["columns"] = result.columns.tolist()
            context[f"{tool_name}_result"] = result

        elif isinstance(result, Path):
            # Chart or file output
            outputs["type"] = "file"
            outputs["path"] = str(result)

        elif isinstance(result, dict):
            outputs = result

        else:
            outputs["value"] = result

        return outputs
