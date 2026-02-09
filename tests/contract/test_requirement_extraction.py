"""Contract tests for requirement extraction."""

import pytest
from autoviz_agent.llm.llm_contracts import (
    RequirementExtractionOutput,
    get_requirement_extraction_schema,
)


class TestRequirementExtractionSchema:
    """Test requirement extraction schema contract."""

    def test_schema_exists(self):
        """Schema should be defined and retrievable."""
        schema = get_requirement_extraction_schema()
        assert schema is not None
        assert isinstance(schema, dict)

    def test_schema_has_required_fields(self):
        """Schema should have all required fields."""
        schema = get_requirement_extraction_schema()
        properties = schema.get("properties", {})
        
        required_fields = ["metrics", "group_by", "time", "analysis", "outputs", "constraints"]
        for field in required_fields:
            assert field in properties, f"Missing required field: {field}"

    def test_schema_rejects_additional_properties(self):
        """Schema should reject additional properties for strict validation."""
        schema = get_requirement_extraction_schema()
        assert schema.get("additionalProperties") is False

    def test_analysis_has_closed_labels(self):
        """Analysis field should have enum constraint for closed label set."""
        schema = get_requirement_extraction_schema()
        analysis_schema = schema.get("properties", {}).get("analysis", {})
        
        # Should be array of enums
        assert "items" in analysis_schema
        items = analysis_schema["items"]
        assert "enum" in items, "analysis should have closed enum labels"
        
        # Verify expected analysis types
        expected_types = ["total", "compare", "trend", "distribution", "anomaly", "correlation"]
        enum_values = items["enum"]
        for expected in expected_types:
            assert expected in enum_values, f"Missing analysis type: {expected}"


class TestRequirementExtractionOutput:
    """Test RequirementExtractionOutput model."""

    def test_empty_requirements_valid(self):
        """Empty requirements should be valid."""
        output = RequirementExtractionOutput()
        assert output.metrics == []
        assert output.group_by == []
        assert output.analysis == []

    def test_basic_requirements(self):
        """Basic requirements should validate."""
        output = RequirementExtractionOutput(
            metrics=["revenue", "cost"],
            group_by=["region"],
            analysis=["total", "compare"]
        )
        assert len(output.metrics) == 2
        assert len(output.group_by) == 1
        assert len(output.analysis) == 2

    def test_time_requirement(self):
        """Time requirement should accept column and grain."""
        output = RequirementExtractionOutput(
            time={"column": "date", "grain": "daily"}
        )
        assert output.time is not None
        assert output.time.column == "date"
        assert output.time.grain == "daily"

    def test_constraints_optional(self):
        """Constraints should be optional."""
        output = RequirementExtractionOutput(
            metrics=["revenue"],
            constraints=["exclude_nulls"]
        )
        assert len(output.constraints) == 1

    def test_serialization(self):
        """Requirements should serialize to dict."""
        output = RequirementExtractionOutput(
            metrics=["revenue"],
            group_by=["region"],
            analysis=["compare", "trend"]
        )
        data = output.model_dump()
        
        assert isinstance(data, dict)
        assert data["metrics"] == ["revenue"]
        assert data["group_by"] == ["region"]
        assert data["analysis"] == ["compare", "trend"]


class TestRequirementExtractionContract:
    """Test contract between LLM and requirement extraction."""

    def test_closed_analysis_labels(self):
        """Only predefined analysis labels should be accepted."""
        valid_labels = ["total", "compare", "trend", "distribution", "anomaly", "correlation"]
        
        for label in valid_labels:
            output = RequirementExtractionOutput(analysis=[label])
            assert label in output.analysis

    def test_multiple_metrics_supported(self):
        """Multiple metrics should be supported."""
        output = RequirementExtractionOutput(
            metrics=["revenue", "cost", "profit", "margin"]
        )
        assert len(output.metrics) == 4

    def test_multiple_group_by_supported(self):
        """Multiple group_by dimensions should be supported."""
        output = RequirementExtractionOutput(
            group_by=["region", "product_type", "customer_segment"]
        )
        assert len(output.group_by) == 3

    def test_outputs_field_exists(self):
        """Outputs field should be part of schema."""
        output = RequirementExtractionOutput(
            outputs=["chart", "table"]
        )
        assert len(output.outputs) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
