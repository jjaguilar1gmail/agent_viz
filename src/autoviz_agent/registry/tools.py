"""Tool registry and schema management."""

import inspect
from typing import Any, Callable, Dict, List, Optional, get_type_hints

from pydantic import BaseModel, Field


class ToolParameter(BaseModel):
    """Tool parameter definition."""

    name: str = Field(..., description="Parameter name")
    type: str = Field(..., description="Parameter type")
    description: str = Field(..., description="Parameter description")
    required: bool = Field(default=True, description="Whether parameter is required")
    default: Optional[Any] = Field(None, description="Default value")
    role: Optional[str] = Field(None, description="Column role hint (temporal/numeric/categorical/any)")


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

    def clear(self) -> None:
        """Clear all registered tools and schemas (useful for testing)."""
        self._tools.clear()
        self._schemas.clear()

    def export_schema(self) -> Dict[str, Any]:
        """
        Export tool schemas in JSON format suitable for LLM consumption.

        Returns:
            Dictionary with tool schemas in JSON format
        """
        tools_json = []
        for name, schema in self._schemas.items():
            tool_def = {
                "name": schema.name,
                "description": schema.description,
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
            
            for param in schema.parameters:
                tool_def["parameters"]["properties"][param.name] = {
                    "type": param.type,
                    "description": param.description
                }
                if param.role:
                    tool_def["parameters"]["properties"][param.name]["x-role"] = param.role
                if param.default is not None:
                    tool_def["parameters"]["properties"][param.name]["default"] = param.default
                    
                if param.required:
                    tool_def["parameters"]["required"].append(param.name)
            
            tools_json.append(tool_def)
        
        return {"tools": tools_json}


def register_tools_from_modules(modules: List[Any]) -> None:
    """
    Register tools from already-imported modules by scanning for tool metadata.

    Args:
        modules: List of imported modules to scan
    """
    for module in modules:
        for _, obj in inspect.getmembers(module, inspect.isfunction):
            schema = getattr(obj, "__tool_schema__", None)
            if schema:
                TOOL_REGISTRY.register(schema, obj)


def ensure_default_tools_registered() -> None:
    """
    Ensure default tools are registered, even if registry was cleared.
    """
    if TOOL_REGISTRY.list_tools():
        return
    from autoviz_agent.tools import analysis, data_io, metrics, prep, schema, visualization

    register_tools_from_modules(
        [analysis, data_io, metrics, prep, schema, visualization]
    )


# Global tool registry instance
TOOL_REGISTRY = ToolRegistry()


def tool(
    name: Optional[str] = None,
    description: Optional[str] = None,
    version: str = "1.0.0",
    param_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Callable:
    """
    Decorator to register a function as a tool with automatic schema extraction.
    
    Args:
        name: Tool name (defaults to function name)
        description: Tool description (defaults to function docstring)
        version: Tool version
    
    Returns:
        Decorated function
    
    Example:
        @tool(description="Load a dataset from CSV file")
        def load_dataset(path: str, encoding: str = "utf-8") -> pd.DataFrame:
            '''Load dataset from file.'''
            ...
    """
    def decorator(func: Callable) -> Callable:
        # Extract metadata
        tool_name = name or func.__name__
        tool_description = description or (func.__doc__ or "").strip().split("\n")[0]
        overrides = param_overrides or {}
        
        # Extract parameters from function signature
        sig = inspect.signature(func)
        type_hints = get_type_hints(func) if hasattr(func, '__annotations__') else {}
        
        parameters = []
        for param_name, param in sig.parameters.items():
            # Skip self, cls, *args, **kwargs
            if param_name in ('self', 'cls') or param.kind in (
                inspect.Parameter.VAR_POSITIONAL, 
                inspect.Parameter.VAR_KEYWORD
            ):
                continue
            
            # Get type annotation
            param_type = type_hints.get(param_name, Any)
            type_str = _get_type_string(param_type)
            
            # Check if required (no default value)
            is_required = param.default == inspect.Parameter.empty
            default_value = None if is_required else param.default
            
            # Extract description from docstring if available
            param_description = f"Parameter {param_name}"
            if func.__doc__:
                # Simple docstring parsing for Args section
                doc_lines = func.__doc__.split("\n")
                in_args = False
                for line in doc_lines:
                    if "Args:" in line:
                        in_args = True
                        continue
                    if in_args and param_name in line and ":" in line:
                        parts = line.split(":", 1)
                        if len(parts) > 1:
                            param_description = parts[1].strip()
                        break
                    if in_args and (line.strip().startswith("Returns:") or 
                                   line.strip().startswith("Raises:")):
                        break
            
            # Apply per-parameter overrides if provided
            override = overrides.get(param_name, {})
            if "type" in override:
                type_str = override["type"]
            if "description" in override:
                param_description = override["description"]
            if "required" in override:
                is_required = bool(override["required"])
                if not is_required and "default" not in override:
                    default_value = None
            if "default" in override:
                default_value = override["default"]
                is_required = False

            parameters.append(
                ToolParameter(
                    name=param_name,
                    type=type_str,
                    description=param_description,
                    required=is_required,
                    default=default_value,
                    role=override.get("role"),
                )
            )
        
        # Extract return type
        return_type = type_hints.get('return', Any)
        returns_str = _get_type_string(return_type)
        
        # Create schema
        schema = ToolSchema(
            name=tool_name,
            description=tool_description,
            parameters=parameters,
            returns=returns_str,
            version=version
        )
        
        # Register tool
        TOOL_REGISTRY.register(schema, func)
        func.__tool_schema__ = schema

        # Return original function (unchanged)
        return func
    
    return decorator


def _get_type_string(type_hint: Any) -> str:
    """
    Convert Python type hint to string representation.
    
    Args:
        type_hint: Python type hint
    
    Returns:
        String representation of type
    """
    if type_hint == Any or type_hint == inspect.Parameter.empty:
        return "any"
    
    # Handle string type hints
    if isinstance(type_hint, str):
        return type_hint.lower()
    
    # Get type name
    if hasattr(type_hint, '__name__'):
        type_name = type_hint.__name__
    elif hasattr(type_hint, '__origin__'):
        # Handle generic types like List[str], Dict[str, int]
        origin = type_hint.__origin__
        if hasattr(origin, '__name__'):
            type_name = origin.__name__
        else:
            type_name = str(origin).replace('typing.', '')
    else:
        type_name = str(type_hint).replace('typing.', '')
    
    # Map Python types to simple names
    type_map = {
        'int': 'integer',
        'float': 'number',
        'str': 'string',
        'bool': 'boolean',
        'list': 'array',
        'dict': 'object',
        'DataFrame': 'dataframe',
        'Path': 'string',
        'NoneType': 'null'
    }
    
    return type_map.get(type_name, type_name.lower())
