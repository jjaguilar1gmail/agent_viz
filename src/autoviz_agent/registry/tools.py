"""Tool registry and schema management."""

from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field


class ToolParameter(BaseModel):
    """Tool parameter definition."""

    name: str = Field(..., description="Parameter name")
    type: str = Field(..., description="Parameter type")
    description: str = Field(..., description="Parameter description")
    required: bool = Field(default=True, description="Whether parameter is required")
    default: Optional[Any] = Field(None, description="Default value")


class ToolSchema(BaseModel):
    """Tool schema definition."""

    name: str = Field(..., description="Tool name")
    description: str = Field(..., description="Tool description")
    parameters: List[ToolParameter] = Field(default_factory=list, description="Tool parameters")
    returns: str = Field(..., description="Return type description")
    version: str = Field(default="1.0.0", description="Tool version")


class ToolRegistry:
    """Registry for available tools."""

    def __init__(self):
        self._tools: Dict[str, Callable] = {}
        self._schemas: Dict[str, ToolSchema] = {}

    def register(self, schema: ToolSchema, func: Callable) -> None:
        """
        Register a tool.

        Args:
            schema: Tool schema definition
            func: Tool implementation function
        """
        self._tools[schema.name] = func
        self._schemas[schema.name] = schema

    def get_tool(self, name: str) -> Optional[Callable]:
        """
        Get tool by name.

        Args:
            name: Tool name

        Returns:
            Tool function or None if not found
        """
        return self._tools.get(name)

    def get_schema(self, name: str) -> Optional[ToolSchema]:
        """
        Get tool schema by name.

        Args:
            name: Tool name

        Returns:
            Tool schema or None if not found
        """
        return self._schemas.get(name)

    def list_tools(self) -> List[str]:
        """
        List all registered tool names.

        Returns:
            List of tool names
        """
        return list(self._tools.keys())

    def get_all_schemas(self) -> Dict[str, ToolSchema]:
        """
        Get all tool schemas.

        Returns:
            Dictionary of tool name to schema
        """
        return self._schemas.copy()


# Global tool registry instance
TOOL_REGISTRY = ToolRegistry()
