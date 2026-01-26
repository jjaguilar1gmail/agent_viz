"""Unit tests for tool decorator and registry."""

import pytest
import pandas as pd
from pathlib import Path

from autoviz_agent.registry.tools import tool, TOOL_REGISTRY, ToolSchema, ToolParameter


@pytest.fixture(autouse=True)
def reset_registry():
    """Clear registry before each test."""
    TOOL_REGISTRY.clear()
    yield
    TOOL_REGISTRY.clear()


def test_tool_decorator_basic():
    """Test basic tool decorator functionality."""
    @tool(description="Test tool")
    def test_func(x: int, y: int = 5) -> int:
        """Add two numbers."""
        return x + y
    
    # Verify registration
    assert "test_func" in TOOL_REGISTRY.list_tools()
    
    # Verify schema
    schema = TOOL_REGISTRY.get_schema("test_func")
    assert schema is not None
    assert schema.name == "test_func"
    assert schema.description == "Test tool"
    assert len(schema.parameters) == 2
    
    # Verify parameters
    param_map = {p.name: p for p in schema.parameters}
    assert "x" in param_map
    assert "y" in param_map
    assert param_map["x"].required is True
    assert param_map["y"].required is False
    assert param_map["y"].default == 5


def test_tool_decorator_with_types():
    """Test decorator extracts correct types."""
    @tool()
    def typed_func(
        text: str,
        count: int,
        ratio: float,
        flag: bool,
        data: pd.DataFrame
    ) -> dict:
        """Test function with various types."""
        return {}
    
    schema = TOOL_REGISTRY.get_schema("typed_func")
    param_map = {p.name: p for p in schema.parameters}
    
    assert param_map["text"].type == "string"
    assert param_map["count"].type == "integer"
    assert param_map["ratio"].type == "number"
    assert param_map["flag"].type == "boolean"
    assert param_map["data"].type == "dataframe"


def test_tool_decorator_custom_name():
    """Test decorator with custom name."""
    @tool(name="custom_name", description="Custom tool")
    def original_name() -> None:
        """Original function."""
        pass
    
    # Should be registered under custom name
    assert "custom_name" in TOOL_REGISTRY.list_tools()
    assert "original_name" not in TOOL_REGISTRY.list_tools()
    
    schema = TOOL_REGISTRY.get_schema("custom_name")
    assert schema.name == "custom_name"


def test_tool_function_remains_callable():
    """Test that decorated function still works."""
    @tool(description="Add numbers")
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b
    
    # Function should still be callable
    result = add(3, 5)
    assert result == 8


def test_registry_get_tool():
    """Test getting tool from registry."""
    @tool()
    def sample_tool(x: int) -> int:
        """Sample tool."""
        return x * 2
    
    # Get tool function
    func = TOOL_REGISTRY.get_tool("sample_tool")
    assert func is not None
    assert func(5) == 10


def test_registry_get_nonexistent_tool():
    """Test getting non-existent tool returns None."""
    func = TOOL_REGISTRY.get_tool("nonexistent")
    assert func is None


def test_registry_clear():
    """Test clearing registry."""
    @tool()
    def tool1() -> None:
        pass
    
    @tool()
    def tool2() -> None:
        pass
    
    assert len(TOOL_REGISTRY.list_tools()) == 2
    
    TOOL_REGISTRY.clear()
    
    assert len(TOOL_REGISTRY.list_tools()) == 0


def test_export_schema():
    """Test exporting schema for LLM."""
    @tool(description="Load data")
    def load_data(path: str, encoding: str = "utf-8") -> pd.DataFrame:
        """Load dataset from file."""
        return pd.DataFrame()
    
    schema_json = TOOL_REGISTRY.export_schema()
    
    assert "tools" in schema_json
    assert len(schema_json["tools"]) == 1
    
    tool_def = schema_json["tools"][0]
    assert tool_def["name"] == "load_data"
    assert tool_def["description"] == "Load data"
    assert "parameters" in tool_def
    assert "properties" in tool_def["parameters"]
    assert "path" in tool_def["parameters"]["properties"]
    assert "encoding" in tool_def["parameters"]["properties"]
    assert "path" in tool_def["parameters"]["required"]
    assert "encoding" not in tool_def["parameters"]["required"]
    assert tool_def["parameters"]["properties"]["encoding"]["default"] == "utf-8"


def test_multiple_tools_registered():
    """Test multiple tools can be registered."""
    @tool()
    def tool_a() -> None:
        pass
    
    @tool()
    def tool_b() -> None:
        pass
    
    @tool()
    def tool_c() -> None:
        pass
    
    tools = TOOL_REGISTRY.list_tools()
    assert len(tools) == 3
    assert "tool_a" in tools
    assert "tool_b" in tools
    assert "tool_c" in tools


def test_tool_with_optional_params():
    """Test tool with all optional parameters."""
    @tool()
    def optional_tool(a: int = 1, b: int = 2, c: int = 3) -> int:
        """Tool with all optional params."""
        return a + b + c
    
    schema = TOOL_REGISTRY.get_schema("optional_tool")
    for param in schema.parameters:
        assert param.required is False
        assert param.default is not None


def test_tool_with_no_params():
    """Test tool with no parameters."""
    @tool()
    def no_params() -> str:
        """Tool with no params."""
        return "result"
    
    schema = TOOL_REGISTRY.get_schema("no_params")
    assert len(schema.parameters) == 0


def test_get_all_schemas():
    """Test getting all schemas."""
    @tool()
    def tool1() -> None:
        pass
    
    @tool()
    def tool2() -> None:
        pass
    
    schemas = TOOL_REGISTRY.get_all_schemas()
    assert len(schemas) == 2
    assert "tool1" in schemas
    assert "tool2" in schemas
    assert isinstance(schemas["tool1"], ToolSchema)
