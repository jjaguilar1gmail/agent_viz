"""Contract tests for coverage validation."""

import pytest
from autoviz_agent.llm.llm_contracts import RequirementExtractionOutput
from autoviz_agent.planning.diff import validate_plan_coverage, generate_coverage_error_payload
from autoviz_agent.planning.schema_tags import (
    REQUIREMENT_TO_CAPABILITY_MAP,
    get_required_capabilities,
)


class TestRequirementToCapabilityMapping:
    """Test the requirement-to-capability mapping registry."""

    def test_mapping_exists(self):
        """Mapping registry should be defined."""
        assert REQUIREMENT_TO_CAPABILITY_MAP is not None
        assert isinstance(REQUIREMENT_TO_CAPABILITY_MAP, dict)

    def test_mapping_has_analysis_types(self):
        """Mapping should include all analysis types."""
        expected_analysis = ["total", "compare", "trend", "distribution", "anomaly", "correlation"]
        
        for analysis_type in expected_analysis:
            assert analysis_type in REQUIREMENT_TO_CAPABILITY_MAP, \
                f"Missing mapping for analysis type: {analysis_type}"

    def test_mapping_has_output_types(self):
        """Mapping should include output types."""
        assert "chart" in REQUIREMENT_TO_CAPABILITY_MAP
        assert "table" in REQUIREMENT_TO_CAPABILITY_MAP

    def test_mapping_has_special_requirements(self):
        """Mapping should include special requirements."""
        assert "group_by" in REQUIREMENT_TO_CAPABILITY_MAP
        assert "time" in REQUIREMENT_TO_CAPABILITY_MAP

    def test_get_required_capabilities_works(self):
        """get_required_capabilities should return capability list."""
        caps = get_required_capabilities("trend")
        assert isinstance(caps, list)
        assert len(caps) > 0
        assert "time_series" in caps or "plot" in caps or "trend" in caps

    def test_get_required_capabilities_raises_on_unknown(self):
        """get_required_capabilities should raise on unknown requirement."""
        with pytest.raises(ValueError, match="Unknown requirement"):
            get_required_capabilities("unknown_requirement_type")


class TestCoverageValidation:
    """Test plan coverage validation logic."""

    @pytest.fixture
    def sample_requirements(self):
        """Sample requirements for testing."""
        return RequirementExtractionOutput(
            metrics=["revenue"],
            group_by=["region"],
            analysis=["compare", "trend"]
        )

    def test_validate_plan_coverage_returns_tuple(self, sample_requirements):
        """validate_plan_coverage should return (bool, dict) tuple."""
        plan = {"steps": []}
        result = validate_plan_coverage(plan, sample_requirements)
        
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)
        assert isinstance(result[1], dict)

    def test_empty_plan_fails_validation(self, sample_requirements):
        """Empty plan should fail validation with requirements."""
        plan = {"steps": []}
        is_valid, report = validate_plan_coverage(plan, sample_requirements)
        
        assert not is_valid, "Empty plan should not satisfy requirements"
        assert "missing_coverage" in report
        assert len(report["missing_coverage"]) > 0

    def test_complete_plan_passes_validation(self):
        """Plan with all required capabilities should pass."""
        requirements = RequirementExtractionOutput(
            metrics=["revenue"],
            analysis=["total"]
        )
        
        plan = {
            "steps": [
                {
                    "step_id": "agg",
                    "tool": "aggregate",
                    "satisfies": ["total"]
                }
            ]
        }
        
        is_valid, report = validate_plan_coverage(plan, requirements)
        
        assert is_valid or len(report["missing_coverage"]) == 0, \
            f"Plan should pass validation. Report: {report}"

    def test_unjustified_steps_detected(self):
        """Steps without 'satisfies' field should be flagged as unjustified."""
        requirements = RequirementExtractionOutput(
            metrics=["revenue"],
            analysis=["total"]
        )
        
        plan = {
            "steps": [
                {
                    "step_id": "mysterious_step",
                    "tool": "aggregate"
                    # Missing 'satisfies' field
                }
            ]
        }
        
        is_valid, report = validate_plan_coverage(plan, requirements)
        
        assert "unjustified_steps" in report
        assert "mysterious_step" in report["unjustified_steps"]

    def test_missing_coverage_detected(self):
        """Missing capabilities should be detected."""
        requirements = RequirementExtractionOutput(
            metrics=["revenue"],
            analysis=["compare", "trend", "distribution"]
        )
        
        plan = {
            "steps": [
                {
                    "step_id": "agg",
                    "tool": "aggregate",
                    "satisfies": ["compare"]
                }
            ]
        }
        
        is_valid, report = validate_plan_coverage(plan, requirements)
        
        assert not is_valid
        assert len(report["missing_coverage"]) > 0
        # Should be missing trend and distribution
        assert any("trend" in key or "distribution" in key 
                   for key in report["missing_coverage"].keys())

    def test_group_by_coverage(self):
        """group_by requirement should be validated."""
        requirements = RequirementExtractionOutput(
            metrics=["revenue"],
            group_by=["region"]
        )
        
        plan = {
            "steps": [
                {
                    "step_id": "group",
                    "tool": "aggregate",
                    "satisfies": ["group_by"]
                }
            ]
        }
        
        is_valid, report = validate_plan_coverage(plan, requirements)
        
        # Should pass or have no group_by in missing
        assert is_valid or "group_by" not in report["missing_coverage"]

    def test_time_coverage(self):
        """time requirement should be validated."""
        requirements = RequirementExtractionOutput(
            metrics=["revenue"],
            time={"column": "date", "grain": "daily"}
        )
        
        plan = {
            "steps": [
                {
                    "step_id": "parse",
                    "tool": "parse_datetime",
                    "satisfies": ["time"]
                }
            ]
        }
        
        is_valid, report = validate_plan_coverage(plan, requirements)
        
        # Test should verify validation runs, may or may not pass
        # depending on tool capabilities
        assert "missing_coverage" in report
        assert "unjustified_steps" in report


class TestCoverageValidationReport:
    """Test coverage validation report structure."""

    def test_report_has_required_fields(self):
        """Report should have all required fields."""
        requirements = RequirementExtractionOutput(analysis=["total"])
        plan = {"steps": []}
        
        is_valid, report = validate_plan_coverage(plan, requirements)
        
        required_fields = ["valid", "missing_coverage", "unjustified_steps", 
                          "plan_capabilities", "step_to_requirements"]
        for field in required_fields:
            assert field in report, f"Missing report field: {field}"

    def test_plan_capabilities_extracted(self):
        """Report should list plan's capabilities."""
        requirements = RequirementExtractionOutput(analysis=["total"])
        plan = {
            "steps": [
                {
                    "step_id": "agg",
                    "tool": "aggregate",
                    "satisfies": ["total"]
                },
                {
                    "step_id": "plot",
                    "tool": "plot_bar",
                    "satisfies": []
                }
            ]
        }
        
        is_valid, report = validate_plan_coverage(plan, requirements)
        
        assert "plan_capabilities" in report
        assert isinstance(report["plan_capabilities"], list)
        # Should include capabilities from tools used
        assert len(report["plan_capabilities"]) > 0

    def test_step_to_requirements_mapping(self):
        """Report should map steps to requirements they satisfy."""
        requirements = RequirementExtractionOutput(analysis=["compare", "trend"])
        plan = {
            "steps": [
                {
                    "step_id": "agg",
                    "tool": "aggregate",
                    "satisfies": ["compare"]
                },
                {
                    "step_id": "plot",
                    "tool": "plot_line",
                    "satisfies": ["trend"]
                }
            ]
        }
        
        is_valid, report = validate_plan_coverage(plan, requirements)
        
        assert "step_to_requirements" in report
        mapping = report["step_to_requirements"]
        assert "agg" in mapping
        assert "compare" in mapping["agg"]
        assert "plot" in mapping
        assert "trend" in mapping["plot"]


class TestCoverageErrorPayload:
    """Test coverage error message generation."""

    def test_error_payload_generated(self):
        """Should generate actionable error message."""
        report = {
            "valid": False,
            "missing_coverage": {"analysis.trend": ["time_series", "plot"]},
            "unjustified_steps": ["mystery_step"],
            "plan_capabilities": ["aggregate"],
            "step_to_requirements": {}
        }
        
        payload = generate_coverage_error_payload(report)
        
        assert isinstance(payload, str)
        assert len(payload) > 0
        assert "trend" in payload.lower()
        assert "mystery_step" in payload

    def test_error_payload_mentions_missing_capabilities(self):
        """Error should list missing capabilities."""
        report = {
            "valid": False,
            "missing_coverage": {
                "analysis.distribution": ["distribution_plot", "frequency"]
            },
            "unjustified_steps": [],
            "plan_capabilities": [],
            "step_to_requirements": {}
        }
        
        payload = generate_coverage_error_payload(report)
        
        assert "distribution" in payload.lower()
        assert "capabilities" in payload.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
