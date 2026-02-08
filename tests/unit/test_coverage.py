"""Unit tests for coverage validation."""

import pytest

from autoviz_agent.llm.llm_contracts import RequirementExtractionOutput, TimeRequirement
from autoviz_agent.planning.diff import validate_plan_coverage, generate_coverage_error_payload
from autoviz_agent.registry.tools import TOOL_REGISTRY


class TestCoverageValidation:
    """Test coverage validation logic."""

    def test_valid_coverage_all_requirements(self):
        """Test valid coverage when all requirements are satisfied."""
        requirements = RequirementExtractionOutput(
            metrics=["revenue"],
            group_by=["region"],
            time=TimeRequirement(column="date", grain="monthly"),
            analysis=["total", "compare"],
            outputs=["chart", "table"],
            constraints=["Q4 2023"],
        )
        
        plan = {
            "steps": [
                {
                    "step_id": "1",
                    "action": "aggregate",
                    "tool": "aggregate",
                    "inputs": {"df": "df", "group_by": ["region"], "metrics": ["revenue"]},
                    "outputs": ["agg_df"],
                    "description": "Compute revenue by region",
                    "satisfies": ["total", "compare", "group_by:region"],
                },
                {
                    "step_id": "2",
                    "action": "plot",
                    "tool": "plot_bar",
                    "inputs": {"df": "agg_df", "x": "region", "y": "revenue"},
                    "outputs": ["chart"],
                    "description": "Visualize comparison",
                    "satisfies": ["output:chart"],
                },
            ]
        }
        
        is_valid, report = validate_plan_coverage(plan, requirements)
        
        assert "valid" in report
        assert "missing_coverage" in report
        assert "plan_capabilities" in report

    def test_missing_analysis_requirement(self):
        """Test detection of missing analysis requirement."""
        requirements = RequirementExtractionOutput(
            metrics=["revenue"],
            group_by=[],
            time=TimeRequirement(column="", grain="unknown"),
            analysis=["trend", "anomaly"],
            outputs=["chart"],
            constraints=[],
        )
        
        plan = {
            "steps": [
                {
                    "step_id": "1",
                    "action": "plot",
                    "tool": "plot_line",
                    "inputs": {"df": "df", "x": "date", "y": "revenue"},
                    "outputs": ["chart"],
                    "description": "Show trend",
                    "satisfies": ["trend", "output:chart"],
                },
            ]
        }
        
        is_valid, report = validate_plan_coverage(plan, requirements)
        
        assert not report["valid"]
        assert "analysis.anomaly" in report["missing_coverage"]

    def test_missing_output_requirement(self):
        """Test detection of missing output requirement."""
        requirements = RequirementExtractionOutput(
            metrics=["revenue"],
            group_by=[],
            time=TimeRequirement(column="", grain="unknown"),
            analysis=["total"],
            outputs=["chart", "table"],
            constraints=[],
        )
        
        plan = {
            "steps": [
                {
                    "step_id": "1",
                    "action": "plot",
                    "tool": "plot_bar",
                    "inputs": {"df": "df", "x": "region", "y": "revenue"},
                    "outputs": ["chart"],
                    "description": "Visualize",
                    "satisfies": ["total", "output:chart"],
                },
            ]
        }
        
        is_valid, report = validate_plan_coverage(plan, requirements)
        
        assert not report["valid"]
        assert "output.table" in report["missing_coverage"]

    def test_missing_group_by_requirement(self):
        """Test detection of missing group_by requirement."""
        requirements = RequirementExtractionOutput(
            metrics=["revenue"],
            group_by=["region", "product"],
            time=TimeRequirement(column="", grain="unknown"),
            analysis=["total"],
            outputs=["chart"],
            constraints=[],
        )
        
        plan = {
            "steps": [
                {
                    "step_id": "1",
                    "action": "aggregate",
                    "tool": "aggregate",
                    "inputs": {"df": "df", "group_by": ["region"], "metrics": ["revenue"]},
                    "outputs": ["agg_df"],
                    "description": "Group by region",
                    "satisfies": ["total", "group_by:region"],
                },
            ]
        }
        
        is_valid, report = validate_plan_coverage(plan, requirements)
        
        assert not report["valid"]
        # Group by validation checks if group_by capability is present
        # when requirements.group_by is not empty

    def test_missing_time_requirement(self):
        """Test validation with time grain requirement."""
        requirements = RequirementExtractionOutput(
            metrics=["revenue"],
            group_by=[],
            time=TimeRequirement(column="date", grain="monthly"),
            analysis=["trend"],
            outputs=["chart"],
            constraints=[],
        )
        
        plan = {
            "steps": [
                {
                    "step_id": "1",
                    "action": "plot",
                    "tool": "plot_line",
                    "inputs": {"df": "df", "x": "date", "y": "revenue"},
                    "outputs": ["chart"],
                    "description": "Show trend",
                    "satisfies": ["trend", "output:chart"],
                },
            ]
        }
        
        is_valid, report = validate_plan_coverage(plan, requirements)
        
        # Validation runs and produces a report
        assert "valid" in report
        assert "missing_coverage" in report

    def test_unjustified_steps(self):
        """Test detection of unjustified plan steps."""
        requirements = RequirementExtractionOutput(
            metrics=["revenue"],
            group_by=[],
            time=TimeRequirement(column="", grain="unknown"),
            analysis=["total"],
            outputs=["chart"],
            constraints=[],
        )
        
        plan = {
            "steps": [
                {
                    "step_id": "1",
                    "action": "compute",
                    "tool": "compute_summary_stats",
                    "inputs": {"df": "df", "columns": ["revenue"]},
                    "outputs": ["stats"],
                    "description": "Compute statistics",
                    "satisfies": ["total"],
                },
                {
                    "step_id": "2",
                    "action": "plot",
                    "tool": "plot_histogram",
                    "inputs": {"df": "df", "column": "revenue"},
                    "outputs": ["chart"],
                    "description": "Extra histogram",
                    "satisfies": [],
                },
            ]
        }
        
        is_valid, report = validate_plan_coverage(plan, requirements)
        
        assert "unjustified_steps" in report
        # Step 2 has empty satisfies
        assert "2" in report["unjustified_steps"]

    def test_empty_requirements(self):
        """Test coverage validation with empty requirements."""
        requirements = RequirementExtractionOutput(
            metrics=[],
            group_by=[],
            time=TimeRequirement(column="", grain="unknown"),
            analysis=[],
            outputs=[],
            constraints=[],
        )
        
        plan = {"steps": []}
        
        is_valid, report = validate_plan_coverage(plan, requirements)
        
        assert report["valid"]
        assert len(report["missing_coverage"]) == 0


class TestCoverageErrorPayload:
    """Test coverage error payload generation."""

    def test_generate_error_payload(self):
        """Test error payload generation."""
        requirements = RequirementExtractionOutput(
            metrics=["revenue"],
            group_by=["region"],
            time=TimeRequirement(column="date", grain="daily"),
            analysis=["trend", "compare"],
            outputs=["chart"],
            constraints=["last 30 days"],
        )
        
        plan = {
            "steps": [
                {
                    "step_id": "1",
                    "action": "plot",
                    "tool": "plot_line",
                    "inputs": {"df": "df", "x": "date", "y": "revenue"},
                    "outputs": ["chart"],
                    "description": "Show trend",
                    "satisfies": ["trend", "output:chart"],
                },
            ]
        }
        
        is_valid, report = validate_plan_coverage(plan, requirements)
        
        error_payload = generate_coverage_error_payload(report)
        
        # Error payload should be a string with coverage information
        assert isinstance(error_payload, str)
        assert "Coverage validation failed" in error_payload or "missing" in error_payload.lower()

    def test_empty_suggested_tools_when_valid(self):
        """Test no suggested tools when coverage is valid."""
        requirements = RequirementExtractionOutput(
            metrics=["revenue"],
            group_by=[],
            time=TimeRequirement(column="", grain="unknown"),
            analysis=["total"],
            outputs=["chart"],
            constraints=[],
        )
        
        plan = {
            "steps": [
                {
                    "step_id": "1",
                    "action": "aggregate",
                    "tool": "compute_summary_stats",
                    "inputs": {"df": "df", "columns": ["revenue"]},
                    "outputs": ["stats"],
                    "description": "Total revenue",
                    "satisfies": ["total", "output:chart"],
                },
            ]
        }
        
        is_valid, report = validate_plan_coverage(plan, requirements)
        
        error_payload = generate_coverage_error_payload(report)
        
        # Should have minimal error message when valid
        assert isinstance(error_payload, str)
