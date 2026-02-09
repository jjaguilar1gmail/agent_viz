"""Integration tests for requirement extraction to plan adaptation flow."""

import pytest
from pathlib import Path
from autoviz_agent.llm.llm_contracts import RequirementExtractionOutput
from autoviz_agent.planning.retrieval import get_tool_retriever
from autoviz_agent.planning.diff import validate_plan_coverage


class TestRequirementToAdaptationFlow:
    """Test the full pipeline from requirements to adaptation."""

    @pytest.fixture
    def retriever(self):
        """Get tool retriever."""
        return get_tool_retriever()

    @pytest.fixture
    def sample_template_plan(self):
        """Sample template plan."""
        return {
            "template_id": "test_template",
            "curated_tools": ["aggregate", "plot_bar"],
            "steps": [
                {
                    "step_id": "agg",
                    "tool": "aggregate",
                    "description": "Aggregate data"
                }
            ]
        }

    def test_requirements_to_tool_narrowing(self, retriever):
        """Requirements should flow into tool narrowing."""
        requirements = RequirementExtractionOutput(
            metrics=["revenue"],
            group_by=["region"],
            analysis=["compare", "trend"]
        )
        
        template_tools = ["aggregate"]
        narrowed_tools = retriever.retrieve_tools(
            requirements,
            template_tools,
            top_k=5,
            cap=12
        )
        
        assert isinstance(narrowed_tools, list)
        assert len(narrowed_tools) > len(template_tools), \
            "Should expand beyond template tools based on requirements"
        assert "aggregate" in narrowed_tools, "Template tools should be preserved"

    def test_tool_narrowing_respects_requirements(self, retriever):
        """Narrowed tools should be relevant to requirements."""
        # Requirements asking for trends and comparison
        requirements = RequirementExtractionOutput(
            metrics=["revenue"],
            analysis=["trend", "compare"]
        )
        
        template_tools = []
        narrowed_tools = retriever.retrieve_tools(
            requirements,
            template_tools,
            top_k=8,
            cap=12
        )
        
        # Should include tools for plotting/trends
        relevant_tools = ["plot_line", "plot_bar", "aggregate", "compare_groups"]
        has_relevant = any(tool in narrowed_tools for tool in relevant_tools)
        assert has_relevant, f"Should include relevant tools. Got: {narrowed_tools}"

    def test_narrowed_tools_to_coverage_validation(self, retriever):
        """Narrowed tools should be used in coverage validation."""
        requirements = RequirementExtractionOutput(
            metrics=["revenue"],
            analysis=["compare"]
        )
        
        # Narrow tools
        template_tools = []
        narrowed_tools = retriever.retrieve_tools(
            requirements,
            template_tools,
            top_k=5,
            cap=10
        )
        
        # Create plan using narrowed tools
        plan = {
            "steps": [
                {
                    "step_id": "agg",
                    "tool": narrowed_tools[0] if narrowed_tools else "aggregate",
                    "satisfies": ["compare"]
                }
            ]
        }
        
        # Validate coverage
        is_valid, report = validate_plan_coverage(plan, requirements)
        
        assert isinstance(is_valid, bool)
        assert "plan_capabilities" in report
        assert "missing_coverage" in report

    def test_coverage_failure_triggers_expansion(self, retriever):
        """Coverage failure should allow tool expansion retry."""
        requirements = RequirementExtractionOutput(
            metrics=["revenue"],
            analysis=["compare", "trend", "distribution", "anomaly"]
        )
        
        # First attempt with small cap
        template_tools = []
        narrowed_tools_small = retriever.retrieve_tools(
            requirements,
            template_tools,
            top_k=3,
            cap=6
        )
        
        # Second attempt with expanded parameters
        narrowed_tools_expanded = retriever.retrieve_tools(
            requirements,
            template_tools,
            top_k=15,
            cap=None  # Remove cap
        )
        
        assert len(narrowed_tools_expanded) > len(narrowed_tools_small), \
            "Expansion should retrieve more tools"

    def test_empty_requirements_graceful_handling(self, retriever):
        """Empty requirements should be handled gracefully."""
        requirements = RequirementExtractionOutput()  # Empty
        
        template_tools = ["aggregate"]
        narrowed_tools = retriever.retrieve_tools(
            requirements,
            template_tools,
            top_k=5,
            cap=10
        )
        
        # Should still return tools (template + safety)
        assert len(narrowed_tools) > 0
        assert "aggregate" in narrowed_tools


class TestAdaptationWithNarrowedTools:
    """Test that adaptation receives and uses narrowed tools."""

    def test_narrowed_tools_parameter_structure(self):
        """Narrowed tools should be simple list of strings."""
        narrowed_tools = ["aggregate", "plot_bar", "compute_summary_stats"]
        
        assert isinstance(narrowed_tools, list)
        assert all(isinstance(tool, str) for tool in narrowed_tools)
        assert len(narrowed_tools) > 0

    def test_adaptation_prompt_filtering(self):
        """Adaptation should only see narrowed tools, not all 24 tools."""
        # This is integration point - in actual code, the prompt builder
        # receives narrowed_tools and filters TOOL_REGISTRY
        
        all_tools = 24  # Total tools in registry
        narrowed_tools = ["aggregate", "plot_bar", "plot_line", "compute_summary_stats"]
        
        # Simulating what _build_tool_catalog does
        visible_tools = narrowed_tools if narrowed_tools else list(range(all_tools))
        
        assert len(visible_tools) == len(narrowed_tools)
        assert len(visible_tools) < all_tools, \
            "Narrowed tools should be much smaller than full registry"


class TestRequirementToAdaptationIntegration:
    """Integration tests for the complete flow."""

    def test_multi_analysis_requirement_flow(self):
        """Multiple analysis types should flow through entire pipeline."""
        requirements = RequirementExtractionOutput(
            metrics=["revenue", "cost"],
            group_by=["region", "product"],
            analysis=["compare", "trend", "distribution"]
        )
        
        # Step 1: Tool narrowing
        retriever = get_tool_retriever()
        narrowed_tools = retriever.retrieve_tools(
            requirements,
            template_tools=[],
            top_k=10,
            cap=15
        )
        
        assert len(narrowed_tools) > 5, "Should retrieve sufficient tools"
        
        # Step 2: Create plan with narrowed tools
        plan = {
            "steps": [
                {
                    "step_id": "compare_step",
                    "tool": "compare_groups",
                    "satisfies": ["compare"]
                },
                {
                    "step_id": "trend_step",
                    "tool": "plot_line",
                    "satisfies": ["trend"]
                },
                {
                    "step_id": "dist_step",
                    "tool": "plot_histogram",
                    "satisfies": ["distribution"]
                }
            ]
        }
        
        # Step 3: Validate coverage
        is_valid, report = validate_plan_coverage(plan, requirements)
        
        # Should have good coverage
        assert len(report["plan_capabilities"]) > 0
        assert len(report["step_to_requirements"]) >= 3

    def test_time_series_requirement_flow(self):
        """Time series requirements should flow through pipeline."""
        requirements = RequirementExtractionOutput(
            metrics=["revenue"],
            time={"column": "date", "grain": "daily"},
            analysis=["trend"]
        )
        
        retriever = get_tool_retriever()
        narrowed_tools = retriever.retrieve_tools(
            requirements,
            template_tools=[],
            top_k=8,
            cap=12
        )
        
        # Should include time-related tools
        time_tools = ["parse_datetime", "plot_line", "compute_time_series_features"]
        has_time_tool = any(tool in narrowed_tools for tool in time_tools)
        assert has_time_tool, f"Should include time-related tools. Got: {narrowed_tools}"

    def test_anomaly_detection_requirement_flow(self):
        """Anomaly detection requirements should retrieve appropriate tools."""
        requirements = RequirementExtractionOutput(
            metrics=["revenue"],
            analysis=["anomaly"]
        )
        
        retriever = get_tool_retriever()
        narrowed_tools = retriever.retrieve_tools(
            requirements,
            template_tools=[],
            top_k=8,
            cap=12
        )
        
        # Should include anomaly detection tools
        anomaly_tools = ["detect_anomalies", "compute_summary_stats"]
        has_anomaly_tool = any(tool in narrowed_tools for tool in anomaly_tools)
        assert has_anomaly_tool, f"Should include anomaly tools. Got: {narrowed_tools}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
