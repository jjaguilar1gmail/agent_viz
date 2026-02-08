"""Integration tests for requirement extraction flow."""

import pytest
from unittest.mock import Mock, patch, MagicMock

from autoviz_agent.models.state import SchemaProfile, ColumnProfile
from autoviz_agent.llm.llm_contracts import RequirementExtractionOutput, TimeRequirement


@pytest.fixture
def mock_schema():
    """Create mock schema profile."""
    return SchemaProfile(
        row_count=1000,
        columns=[
            ColumnProfile(
                name="date",
                dtype="datetime64",
                roles=["temporal"],
                cardinality=365,
                missing_rate=0.0,
            ),
            ColumnProfile(
                name="region",
                dtype="object",
                roles=["categorical"],
                cardinality=5,
                missing_rate=0.0,
            ),
            ColumnProfile(
                name="revenue",
                dtype="float64",
                roles=["numeric", "metric"],
                cardinality=1000,
                missing_rate=0.0,
            ),
        ],
        data_shape="time_series",
    )


@pytest.fixture
def mock_dataframe():
    """Create mock dataframe."""
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    
    dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(100)]
    data = {
        "date": dates,
        "region": np.random.choice(["East", "West", "North", "South", "Central"], 100),
        "revenue": np.random.uniform(1000, 5000, 100),
    }
    return pd.DataFrame(data)


class TestRequirementExtractionLLM:
    """Integration tests for requirement extraction with LLM."""

    def test_requirement_extraction_contract(self, mock_schema):
        """Test that requirement extraction schema works with actual prompt building."""
        from autoviz_agent.llm.prompts import PromptBuilder
        from autoviz_agent.llm.llm_contracts import get_requirement_extraction_schema
        
        builder = PromptBuilder()
        
        # Build prompt
        prompt = builder.build_requirement_extraction_prompt(
            "show revenue trends by region",
            mock_schema
        )
        
        # Verify prompt contains key elements
        assert "revenue" in prompt.lower()
        assert "region" in prompt.lower()
        assert "trend" in prompt.lower()
        
        # Get schema
        schema = get_requirement_extraction_schema()
        
        # Verify schema has required structure for xgrammar2
        assert "type" in schema
        assert schema["type"] == "object"
        assert "properties" in schema
        assert "metrics" in schema["properties"]
        assert "analysis" in schema["properties"]
        assert "outputs" in schema["properties"]
        assert "additionalProperties" in schema
        assert schema["additionalProperties"] == False  # Required for xgrammar2


class TestCoverageValidationIntegration:
    """Integration tests for coverage validation."""

    def test_coverage_validation_with_actual_tools(self, mock_schema):
        """Test coverage validation with actual tool registry."""
        from autoviz_agent.planning.diff import validate_plan_coverage
        
        requirements = RequirementExtractionOutput(
            metrics=["revenue"],
            group_by=["region"],
            time=TimeRequirement(column="", grain="unknown"),
            analysis=["compare"],
            outputs=["chart"],
            constraints=[],
        )
        
        # Plan with tools from actual registry
        plan = {
            "steps": [
                {
                    "step_id": "1",
                    "action": "aggregate",
                    "tool": "aggregate",
                    "inputs": {"df": "df", "group_by": ["region"], "metrics": ["revenue"]},
                    "outputs": ["agg_df"],
                    "description": "Group by region",
                    "satisfies": ["compare", "group_by:region"],
                },
                {
                    "step_id": "2",
                    "action": "plot",
                    "tool": "plot_bar",
                    "inputs": {"df": "agg_df", "x": "region", "y": "revenue"},
                    "outputs": ["chart"],
                    "description": "Visualize",
                    "satisfies": ["output:chart"],
                },
            ]
        }
        
        is_valid, report = validate_plan_coverage(plan, requirements)
        
        # Should validate successfully with real tools
        assert "valid" in report
        assert "missing_coverage" in report
        assert "plan_capabilities" in report
        # aggregate and plot_bar are real tools with capabilities
        assert len(report["plan_capabilities"]) > 0

    def test_coverage_with_missing_tool(self, mock_schema):
        """Test coverage validation detects unregistered tools."""
        from autoviz_agent.planning.diff import validate_plan_coverage
        
        requirements = RequirementExtractionOutput(
            metrics=["revenue"],
            group_by=[],
            time=TimeRequirement(column="", grain="unknown"),
            analysis=["total"],
            outputs=["chart"],
            constraints=[],
        )
        
        # Plan with non-existent tool
        plan = {
            "steps": [
                {
                    "step_id": "1",
                    "action": "compute",
                    "tool": "nonexistent_tool",
                    "inputs": {"df": "df"},
                    "outputs": ["result"],
                    "description": "Do something",
                    "satisfies": ["total"],
                },
            ]
        }
        
        is_valid, report = validate_plan_coverage(plan, requirements)
        
        # Should handle missing tools gracefully
        assert "valid" in report
        assert "plan_capabilities" in report


        # Should handle missing tools gracefully
        assert "valid" in report
        assert "plan_capabilities" in report


class TestToolRetrieverIntegration:
    """Integration tests for tool retriever."""

    def test_retriever_with_real_registry(self):
        """Test retriever with actual tool registry."""
        from autoviz_agent.planning.retrieval import get_tool_retriever, HAS_EMBEDDINGS
        
        retriever = get_tool_retriever()
        
        requirements = RequirementExtractionOutput(
            metrics=["revenue"],
            group_by=["region", "product"],
            time=TimeRequirement(column="", grain="unknown"),
            analysis=["compare", "total"],
            outputs=["chart"],
            constraints=[],
        )
        
        template_tools = ["plot_bar"]
        
        narrowed_tools = retriever.retrieve_tools(requirements, template_tools)
        
        # Should have template tool
        assert "plot_bar" in narrowed_tools
        
        # Should include aggregate for grouping (if embeddings work)
        if HAS_EMBEDDINGS and retriever._index is not None:
            assert "aggregate" in narrowed_tools
        
        # Check cap
        assert len(narrowed_tools) <= 12

    def test_retriever_query_building(self):
        """Test that retriever builds appropriate queries from requirements."""
        from autoviz_agent.planning.schema_tags import get_required_capabilities
        
        # Test that different analysis types map to capabilities
        compare_caps = get_required_capabilities("compare")
        assert len(compare_caps) > 0
        assert any(cap in ["aggregate", "group_by", "segment"] for cap in compare_caps)
        
        trend_caps = get_required_capabilities("trend")
        assert len(trend_caps) > 0
        assert any(cap in ["plot", "time_series", "trend"] for cap in trend_caps)
        
        # Verify all allowed analysis types have mappings
        from autoviz_agent.llm.llm_contracts import ALLOWED_ANALYSIS_TYPES
        for analysis_type in ALLOWED_ANALYSIS_TYPES:
            caps = get_required_capabilities(analysis_type)
            assert len(caps) > 0, f"Analysis type {analysis_type} has no capability mapping"

