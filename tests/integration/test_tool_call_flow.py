"""Integration tests for end-to-end tool call flow."""

import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import Mock

from autoviz_agent.models.state import SchemaProfile, ColumnProfile
from autoviz_agent.registry.tools import tool, TOOL_REGISTRY
from autoviz_agent.registry.validation import validate_tool_call
from autoviz_agent.runtime.param_resolver import ParamResolver


@pytest.fixture(autouse=True)
def reset_registry():
    """Clear registry before each test."""
    TOOL_REGISTRY.clear()
    yield
    TOOL_REGISTRY.clear()


@pytest.fixture
def sample_schema():
    """Create a sample schema for testing."""
    columns = [
        ColumnProfile(
            name="date",
            dtype="datetime64[ns]",
            roles=["temporal"],
            missing_rate=0.0,
            cardinality=100
        ),
        ColumnProfile(
            name="value",
            dtype="float64",
            roles=["numeric"],
            missing_rate=0.0,
            cardinality=50
        ),
    ]
    return SchemaProfile(
        columns=columns,
        row_count=100,
        data_shape="long"
    )


def test_end_to_end_tool_registration_and_execution():
    """Test full flow: register, validate, execute."""
    # Step 1: Register a tool
    @tool(description="Add two numbers")
    def add_numbers(a: int, b: int) -> int:
        """Add two numbers together."""
        return a + b
    
    # Step 2: Create tool call
    tool_call = {
        "tool": "add_numbers",
        "sequence": 1,
        "args": {"a": 5, "b": 3}
    }
    
    # Step 3: Validate
    validation = validate_tool_call(tool_call)
    assert validation.is_valid
    assert len(validation.errors) == 0
    
    # Step 4: Execute
    func = TOOL_REGISTRY.get_tool("add_numbers")
    assert func is not None
    
    result = func(**tool_call["args"])
    assert result == 8


def test_end_to_end_with_validation_failure():
    """Test validation catches missing required parameters."""
    @tool()
    def requires_params(x: int, y: int) -> int:
        """Requires x and y."""
        return x + y
    
    # Missing required parameter
    tool_call = {
        "tool": "requires_params",
        "sequence": 1,
        "args": {"x": 5}  # Missing 'y'
    }
    
    validation = validate_tool_call(tool_call)
    assert not validation.is_valid
    assert len(validation.errors) > 0
    assert any("y" in err.lower() for err in validation.errors)


def test_end_to_end_with_param_resolution(sample_schema):
    """Test parameter resolution in full flow."""
    # Register a visualization tool
    @tool(description="Create line plot")
    def plot_line(x: str, y: str, output_path: Path) -> Path:
        """Create a line plot."""
        return output_path
    
    # Create resolver
    resolver = ParamResolver(sample_schema)
    
    # Resolve parameters
    params = resolver.resolve("plot_line", {}, sequence=1)
    
    # Create tool call
    tool_call = {
        "tool": "plot_line",
        "sequence": 1,
        "args": params
    }
    
    # Validate
    validation = validate_tool_call(tool_call)
    assert validation.is_valid
    
    # Execute
    func = TOOL_REGISTRY.get_tool("plot_line")
    result = func(**tool_call["args"])
    
    # Verify output
    assert isinstance(result, Path) or isinstance(result, str)


def test_end_to_end_with_dataframe_tool():
    """Test tool that works with DataFrames."""
    @tool(description="Filter DataFrame")
    def filter_df(df: pd.DataFrame, column: str, threshold: float) -> pd.DataFrame:
        """Filter DataFrame by column threshold."""
        return df[df[column] > threshold]
    
    # Create test data
    test_df = pd.DataFrame({
        "value": [1, 2, 3, 4, 5]
    })
    
    # Create tool call
    tool_call = {
        "tool": "filter_df",
        "sequence": 1,
        "args": {
            "df": test_df,
            "column": "value",
            "threshold": 2.5
        }
    }
    
    # Validate
    validation = validate_tool_call(tool_call)
    assert validation.is_valid
    
    # Execute
    func = TOOL_REGISTRY.get_tool("filter_df")
    result = func(**tool_call["args"])
    
    # Verify result
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3  # Values > 2.5: [3, 4, 5]
    assert result["value"].tolist() == [3, 4, 5]


def test_end_to_end_with_optional_params():
    """Test tool with optional parameters."""
    @tool()
    def process_data(data: pd.DataFrame, multiplier: int = 2) -> pd.DataFrame:
        """Process data with optional multiplier."""
        return data * multiplier
    
    test_df = pd.DataFrame({"value": [1, 2, 3]})
    
    # Call without optional parameter
    tool_call1 = {
        "tool": "process_data",
        "sequence": 1,
        "args": {"data": test_df}
    }
    
    validation = validate_tool_call(tool_call1)
    assert validation.is_valid
    
    func = TOOL_REGISTRY.get_tool("process_data")
    result1 = func(**tool_call1["args"])
    assert result1["value"].tolist() == [2, 4, 6]
    
    # Call with optional parameter
    tool_call2 = {
        "tool": "process_data",
        "sequence": 2,
        "args": {"data": test_df, "multiplier": 3}
    }
    
    result2 = func(**tool_call2["args"])
    assert result2["value"].tolist() == [3, 6, 9]


def test_end_to_end_multiple_tools_sequence():
    """Test sequence of multiple tool calls."""
    @tool()
    def load_data() -> pd.DataFrame:
        """Load data."""
        return pd.DataFrame({"value": [1, 2, 3]})
    
    @tool()
    def transform_data(df: pd.DataFrame) -> pd.DataFrame:
        """Transform data."""
        return df * 2
    
    @tool()
    def summarize_data(df: pd.DataFrame) -> dict:
        """Summarize data."""
        return {"sum": df["value"].sum()}
    
    # Execute sequence
    func1 = TOOL_REGISTRY.get_tool("load_data")
    df = func1()
    
    func2 = TOOL_REGISTRY.get_tool("transform_data")
    df = func2(df=df)
    
    func3 = TOOL_REGISTRY.get_tool("summarize_data")
    summary = func3(df=df)
    
    assert summary["sum"] == 12  # (1+2+3)*2 = 12


def test_end_to_end_validation_rejects_unknown_params():
    """Test validation rejects unknown parameters."""
    @tool()
    def strict_tool(x: int) -> int:
        """Tool with strict parameter checking."""
        return x * 2
    
    # Include an unknown parameter
    tool_call = {
        "tool": "strict_tool",
        "sequence": 1,
        "args": {
            "x": 5,
            "unknown_param": "value"  # Should be rejected
        }
    }
    
    validation = validate_tool_call(tool_call)
    assert not validation.is_valid
    assert any("unknown_param" in err.lower() for err in validation.errors)


def test_end_to_end_export_and_validate_schema():
    """Test exporting schema and validating against it."""
    @tool(description="Test tool")
    def test_tool(a: int, b: str = "default") -> dict:
        """Test tool with mixed params."""
        return {"a": a, "b": b}
    
    # Export schema
    schema_json = TOOL_REGISTRY.export_schema()
    
    assert len(schema_json["tools"]) == 1
    tool_def = schema_json["tools"][0]
    
    # Verify schema structure
    assert tool_def["name"] == "test_tool"
    assert "a" in tool_def["parameters"]["properties"]
    assert "b" in tool_def["parameters"]["properties"]
    assert "a" in tool_def["parameters"]["required"]
    assert "b" not in tool_def["parameters"]["required"]
    
    # Validate a correct call
    tool_call = {
        "tool": "test_tool",
        "sequence": 1,
        "args": {"a": 10}
    }
    
    validation = validate_tool_call(tool_call)
    assert validation.is_valid
