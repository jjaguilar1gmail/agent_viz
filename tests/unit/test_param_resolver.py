"""Unit tests for parameter resolver."""

import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import Mock

from autoviz_agent.models.state import SchemaProfile, ColumnProfile
from autoviz_agent.runtime.param_resolver import ParamResolver
from autoviz_agent.registry.tools import TOOL_REGISTRY


@pytest.fixture(autouse=True)
def reset_and_load_tools():
    """Clear registry and import tools before each test."""
    TOOL_REGISTRY.clear()
    # Import tools to trigger registration
    from autoviz_agent.tools import data_io, schema, prep, metrics, analysis, visualization
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
            name="revenue",
            dtype="float64",
            roles=["numeric"],
            missing_rate=0.0,
            cardinality=50
        ),
        ColumnProfile(
            name="category",
            dtype="object",
            roles=["categorical"],
            missing_rate=0.0,
            cardinality=5
        ),
        ColumnProfile(
            name="quantity",
            dtype="int64",
            roles=["numeric"],
            missing_rate=0.0,
            cardinality=30
        ),
    ]
    return SchemaProfile(
        columns=columns,
        row_count=100,
        data_shape="long"
    )


@pytest.fixture
def mock_artifact_manager():
    """Create a mock artifact manager."""
    manager = Mock()
    manager.get_path = Mock(side_effect=lambda type, name: Path(f"outputs/{type}/{name}"))
    return manager


def test_resolver_initialization(sample_schema):
    """Test resolver initializes correctly."""
    resolver = ParamResolver(sample_schema)
    
    assert resolver.column_selector is not None
    assert len(resolver.column_selector.get_temporal_cols()) == 1
    assert "date" in resolver.column_selector.get_temporal_cols()
    assert len(resolver.column_selector.get_numeric_cols()) == 2
    assert "revenue" in resolver.column_selector.get_numeric_cols()
    assert "quantity" in resolver.column_selector.get_numeric_cols()
    assert len(resolver.column_selector.get_categorical_cols()) == 1
    assert "category" in resolver.column_selector.get_categorical_cols()


def test_resolve_plot_line(sample_schema, mock_artifact_manager):
    """Test resolving plot_line parameters."""
    resolver = ParamResolver(sample_schema, mock_artifact_manager)
    
    # Empty params should be filled
    resolved = resolver.resolve("plot_line", {}, sequence=1)
    
    assert "x" in resolved
    assert resolved["x"] == "date"  # temporal column
    assert "y" in resolved
    assert resolved["y"] == "revenue"  # first numeric column
    assert "output_path" in resolved
    assert "line_1.png" in str(resolved["output_path"])


def test_resolve_plot_line_partial(sample_schema, mock_artifact_manager):
    """Test resolving with partial parameters."""
    resolver = ParamResolver(sample_schema, mock_artifact_manager)
    
    # Provide y, should fill x and output_path
    resolved = resolver.resolve("plot_line", {"y": "quantity"}, sequence=2)
    
    assert resolved["x"] == "date"
    assert resolved["y"] == "quantity"  # Preserved
    assert "line_2.png" in str(resolved["output_path"])


def test_resolve_plot_bar(sample_schema, mock_artifact_manager):
    """Test resolving plot_bar parameters."""
    resolver = ParamResolver(sample_schema, mock_artifact_manager)
    
    resolved = resolver.resolve("plot_bar", {}, sequence=1)
    
    assert resolved["x"] == "category"  # categorical column
    assert resolved["y"] == "revenue"  # numeric column
    assert "bar_1.png" in str(resolved["output_path"])


def test_resolve_plot_histogram(sample_schema, mock_artifact_manager):
    """Test resolving plot_histogram parameters."""
    resolver = ParamResolver(sample_schema, mock_artifact_manager)
    
    resolved = resolver.resolve("plot_histogram", {}, sequence=1)
    
    assert "column" in resolved
    assert resolved["column"] == "revenue"  # first numeric
    assert "histogram_1.png" in str(resolved["output_path"])


def test_resolve_plot_scatter(sample_schema, mock_artifact_manager):
    """Test resolving plot_scatter parameters."""
    resolver = ParamResolver(sample_schema, mock_artifact_manager)
    
    resolved = resolver.resolve("plot_scatter", {}, sequence=1)
    
    assert "x" in resolved
    assert resolved["x"] == "revenue"  # first numeric
    assert "y" in resolved
    assert resolved["y"] == "quantity"  # second numeric
    assert "scatter_1.png" in str(resolved["output_path"])


def test_resolve_plot_heatmap(sample_schema, mock_artifact_manager):
    """Test resolving plot_heatmap parameters."""
    resolver = ParamResolver(sample_schema, mock_artifact_manager)
    
    # Test renaming df -> data
    resolved = resolver.resolve("plot_heatmap", {"df": "dataframe"}, sequence=1)
    
    assert "data" in resolved
    assert resolved["data"] == "dataframe"
    assert "df" not in resolved
    assert "heatmap_1.png" in str(resolved["output_path"])


def test_resolve_parse_datetime(sample_schema):
    """Test resolving parse_datetime parameters."""
    resolver = ParamResolver(sample_schema)
    
    resolved = resolver.resolve("parse_datetime", {}, sequence=1)
    
    assert "columns" in resolved
    assert "date" in resolved["columns"]


def test_resolve_aggregate(sample_schema):
    """Test resolving aggregate parameters."""
    resolver = ParamResolver(sample_schema)
    
    resolved = resolver.resolve("aggregate", {}, sequence=1)
    
    assert "group_by" in resolved
    assert resolved["group_by"] == ["category"]  # categorical
    assert "agg_map" in resolved
    assert "revenue" in resolved["agg_map"]


def test_resolve_segment_metric(sample_schema):
    """Test resolving segment_metric parameters."""
    resolver = ParamResolver(sample_schema)
    
    resolved = resolver.resolve("segment_metric", {}, sequence=1)
    
    assert "segment_by" in resolved
    assert resolved["segment_by"] == "category"
    assert "metric" in resolved
    assert resolved["metric"] == "revenue"


def test_resolve_removes_invalid_params(sample_schema):
    """Test that resolver removes invalid parameters."""
    resolver = ParamResolver(sample_schema)
    
    # plot_line should remove show_trend
    resolved = resolver.resolve("plot_line", {"x": "date", "show_trend": True}, sequence=1)
    assert "show_trend" not in resolved
    
    # plot_histogram should remove max_columns
    resolved = resolver.resolve("plot_histogram", {"column": "revenue", "max_columns": 10}, sequence=1)
    assert "max_columns" not in resolved


def test_resolve_preserves_provided_params(sample_schema, mock_artifact_manager):
    """Test that provided parameters are preserved."""
    resolver = ParamResolver(sample_schema, mock_artifact_manager)
    
    custom_params = {
        "x": "custom_x",
        "y": "custom_y",
        "title": "Custom Title"
    }
    
    resolved = resolver.resolve("plot_line", custom_params, sequence=1)
    
    assert resolved["x"] == "custom_x"
    assert resolved["y"] == "custom_y"
    assert resolved["title"] == "Custom Title"


def test_resolve_without_artifact_manager(sample_schema):
    """Test resolver works without artifact manager."""
    resolver = ParamResolver(sample_schema, artifact_manager=None)
    
    resolved = resolver.resolve("plot_line", {}, sequence=1)
    
    # Should still generate output_path, just simpler
    assert "output_path" in resolved
    assert "line_1.png" in str(resolved["output_path"])


def test_resolve_unknown_tool(sample_schema):
    """Test resolving unknown tool returns params unchanged."""
    resolver = ParamResolver(sample_schema)
    
    params = {"some": "param"}
    resolved = resolver.resolve("unknown_tool", params, sequence=1)
    
    # Should return params unchanged (no error)
    assert resolved == params
