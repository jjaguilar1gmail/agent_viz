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
        # Tools are auto-registered via @tool decorator when modules are imported
        logger.info(f"Initialized executor with {len(TOOL_REGISTRY.list_tools())} registered tools")

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
