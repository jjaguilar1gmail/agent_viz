"""Unit tests for requirement extraction."""

import pytest
from pydantic import ValidationError

from autoviz_agent.llm.llm_contracts import (
    RequirementExtractionOutput,
    TimeRequirement,
    validate_requirement_extraction_output,
    ALLOWED_ANALYSIS_TYPES,
)


class TestRequirementExtractionSchema:
    """Test requirement extraction schema validation."""

    def test_valid_requirement_extraction(self):
        """Test valid requirement extraction output."""
        data = {
            "metrics": ["revenue", "sales"],
            "group_by": ["region", "product_category"],
            "time": {"column": "date", "grain": "daily"},
            "analysis": ["total", "compare", "trend"],
            "outputs": ["chart", "table"],
            "constraints": ["Q4 2023"],
        }
        
        result = validate_requirement_extraction_output(data)
        
        assert result.metrics == ["revenue", "sales"]
        assert result.group_by == ["region", "product_category"]
        assert result.time.column == "date"
        assert result.time.grain == "daily"
        assert result.analysis == ["total", "compare", "trend"]
        assert result.outputs == ["chart", "table"]
        assert result.constraints == ["Q4 2023"]

    def test_empty_requirements(self):
        """Test empty requirement extraction."""
        data = {
            "metrics": [],
            "group_by": [],
            "time": {"column": "", "grain": "unknown"},
            "analysis": [],
            "outputs": [],
            "constraints": [],
        }
        
        result = validate_requirement_extraction_output(data)
        
        assert result.metrics == []
        assert result.group_by == []
        assert result.time.column == ""
        assert result.analysis == []

    def test_invalid_analysis_type(self):
        """Test that invalid analysis types are rejected."""
        data = {
            "metrics": ["revenue"],
            "group_by": [],
            "time": {"column": "", "grain": "unknown"},
            "analysis": ["invalid_type"],
            "outputs": ["chart"],
            "constraints": [],
        }
        
        with pytest.raises(ValidationError):
            validate_requirement_extraction_output(data)

    def test_allowed_analysis_types(self):
        """Test all allowed analysis types are accepted."""
        for analysis_type in ALLOWED_ANALYSIS_TYPES:
            data = {
                "metrics": ["revenue"],
                "group_by": [],
                "time": {"column": "", "grain": "unknown"},
                "analysis": [analysis_type],
                "outputs": ["chart"],
                "constraints": [],
            }
            
            result = validate_requirement_extraction_output(data)
            assert analysis_type in result.analysis

    def test_time_requirement_defaults(self):
        """Test time requirement with defaults."""
        time_req = TimeRequirement()
        
        assert time_req.column == ""
        assert time_req.grain == "unknown"

    def test_minimal_valid_requirements(self):
        """Test minimal valid requirements."""
        data = {
            "metrics": ["revenue"],
            "group_by": [],
            "time": {"column": "", "grain": "unknown"},
            "analysis": ["total"],
            "outputs": ["chart"],
            "constraints": [],
        }
        
        result = validate_requirement_extraction_output(data)
        
        assert result.metrics == ["revenue"]
        assert result.analysis == ["total"]


class TestRequirementExtractionPrompt:
    """Test requirement extraction prompt building."""

    def test_prompt_contains_allowed_types(self):
        """Test that prompt includes allowed analysis types."""
        from autoviz_agent.llm.prompts import PromptBuilder
        from autoviz_agent.models.state import SchemaProfile, ColumnProfile
        
        builder = PromptBuilder()
        
        # Create mock schema
        columns = [
            ColumnProfile(
                name="date",
                dtype="datetime64",
                roles=["temporal"],
                cardinality=365,
                missing_rate=0.0,
            ),
            ColumnProfile(
                name="revenue",
                dtype="float64",
                roles=["numeric", "metric"],
                cardinality=1000,
                missing_rate=0.0,
            ),
        ]
        schema = SchemaProfile(
            row_count=1000,
            columns=columns,
            data_shape="time_series",
        )
        
        prompt = builder.build_requirement_extraction_prompt(
            "show me revenue trends",
            schema
        )
        
        # Check that allowed types are in prompt
        for analysis_type in ALLOWED_ANALYSIS_TYPES:
            assert analysis_type in prompt.lower()
        
        # Check schema info is included
        assert "date" in prompt
        assert "revenue" in prompt
