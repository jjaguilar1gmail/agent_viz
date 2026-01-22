"""Test that all tool functions are properly registered."""

import inspect
import pytest
from pathlib import Path
from autoviz_agent.tools import analysis, data_io, metrics, prep, schema, visualization
from autoviz_agent.runtime.executor import ToolExecutor
from autoviz_agent.registry.tools import TOOL_REGISTRY
from autoviz_agent.io.artifacts import ArtifactManager


def get_public_functions(module):
    """Get all public functions (not starting with _) from a module."""
    return [
        name
        for name, obj in inspect.getmembers(module, inspect.isfunction)
        if not name.startswith("_") and obj.__module__ == module.__name__
    ]


def test_all_visualization_tools_registered():
    """Ensure all visualization tools are registered."""
    # Create executor to trigger registration (need artifact_manager)
    artifact_manager = ArtifactManager("test_run", Path("outputs/test"))
    executor = ToolExecutor(artifact_manager)
    registered_tools = set(TOOL_REGISTRY.list_tools())
    
    # Get all public functions from visualization module
    viz_functions = get_public_functions(visualization)
    
    # Filter to only plot_* functions (the ones we want to register)
    plot_functions = [f for f in viz_functions if f.startswith("plot_")]
    
    missing = [f for f in plot_functions if f not in registered_tools]
    
    assert not missing, f"Visualization tools not registered: {missing}"


def test_all_analysis_tools_registered():
    """Ensure all analysis tools are registered."""
    artifact_manager = ArtifactManager("test_run", Path("outputs/test"))
    executor = ToolExecutor(artifact_manager)
    registered_tools = set(TOOL_REGISTRY.list_tools())
    
    analysis_functions = get_public_functions(analysis)
    
    # These are the main analysis tools that should be registered
    expected_tools = [
        "detect_anomalies",
        "segment_metric",
        "compute_distributions",
        "compare_groups",
        "compute_time_series_features",
    ]
    
    missing = [f for f in expected_tools if f not in registered_tools]
    
    if missing:
        pytest.fail(
            f"Analysis tools not registered: {missing}\n"
            f"Available in module: {analysis_functions}\n"
            f"Registered: {sorted(registered_tools)}"
        )


def test_all_metrics_tools_registered():
    """Ensure all metrics tools are registered."""
    artifact_manager = ArtifactManager("test_run", Path("outputs/test"))
    executor = ToolExecutor(artifact_manager)
    registered_tools = set(TOOL_REGISTRY.list_tools())
    
    metrics_functions = get_public_functions(metrics)
    
    # These are the main metrics tools that should be registered
    expected_tools = [
        "compute_summary_stats",
        "compute_correlations",
        "compute_value_counts",
        "compute_percentiles",
        "aggregate",
    ]
    
    missing = [f for f in expected_tools if f not in registered_tools]
    
    if missing:
        pytest.fail(
            f"Metrics tools not registered: {missing}\n"
            f"Available in module: {metrics_functions}\n"
            f"Registered: {sorted(registered_tools)}"
        )


def test_all_prep_tools_registered():
    """Ensure all prep tools are registered."""
    artifact_manager = ArtifactManager("test_run", Path("outputs/test"))
    executor = ToolExecutor(artifact_manager)
    registered_tools = set(TOOL_REGISTRY.list_tools())
    
    prep_functions = get_public_functions(prep)
    
    # These are the main prep tools that should be registered
    expected_tools = [
        "handle_missing",
        "parse_datetime",
        "cast_types",
        "normalize_column_names",
    ]
    
    missing = [f for f in expected_tools if f not in registered_tools]
    
    if missing:
        pytest.fail(
            f"Prep tools not registered: {missing}\n"
            f"Available in module: {prep_functions}\n"
            f"Registered: {sorted(registered_tools)}"
        )


def test_all_data_io_tools_registered():
    """Ensure all data I/O tools are registered."""
    artifact_manager = ArtifactManager("test_run", Path("outputs/test"))
    executor = ToolExecutor(artifact_manager)
    registered_tools = set(TOOL_REGISTRY.list_tools())
    
    data_io_functions = get_public_functions(data_io)
    
    # These are the main data I/O tools that should be registered
    expected_tools = [
        "load_dataset",
        "sample_rows",
        "save_dataframe",
    ]
    
    missing = [f for f in expected_tools if f not in registered_tools]
    
    if missing:
        pytest.fail(
            f"Data I/O tools not registered: {missing}\n"
            f"Available in module: {data_io_functions}\n"
            f"Registered: {sorted(registered_tools)}"
        )


def test_no_duplicate_registrations():
    """Ensure no tools are registered multiple times."""
    artifact_manager = ArtifactManager("test_run", Path("outputs/test"))
    executor = ToolExecutor(artifact_manager)
    registered_tools = TOOL_REGISTRY.list_tools()
    
    # If there are duplicates, len(set) < len(list)
    assert len(set(registered_tools)) == len(registered_tools), \
        f"Duplicate tool registrations detected: {registered_tools}"


if __name__ == "__main__":
    # Run tests and print results
    print("Running tool registration tests...\n")
    
    tests = [
        test_all_visualization_tools_registered,
        test_all_analysis_tools_registered,
        test_all_metrics_tools_registered,
        test_all_prep_tools_registered,
        test_all_data_io_tools_registered,
        test_no_duplicate_registrations,
    ]
    
    for test in tests:
        try:
            # Clear registry before each test
            TOOL_REGISTRY._tools.clear()
            test()
            print(f"✅ {test.__name__}")
        except (AssertionError, Exception) as e:
            print(f"❌ {test.__name__}")
            print(f"   {str(e)}\n")
