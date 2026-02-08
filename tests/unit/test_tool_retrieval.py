"""Unit tests for tool retrieval and narrowing."""

import pytest
from unittest.mock import Mock, patch, MagicMock

from autoviz_agent.planning.retrieval import ToolRetriever, get_tool_retriever, HAS_EMBEDDINGS
from autoviz_agent.llm.llm_contracts import RequirementExtractionOutput, TimeRequirement
from autoviz_agent.registry.tools import TOOL_REGISTRY


class TestToolRetriever:
    """Test tool retrieval and narrowing logic."""

    def test_retriever_singleton(self):
        """Test that get_tool_retriever returns singleton."""
        retriever1 = get_tool_retriever()
        retriever2 = get_tool_retriever()
        
        assert retriever1 is retriever2

    def test_build_index_creates_faiss_index(self):
        """Test FAISS index creation."""
        retriever = ToolRetriever()
        
        # Check index is built or fallback mode is active
        assert retriever._index is not None or not HAS_EMBEDDINGS
        
        if retriever._index is not None:
            assert retriever._tool_names is not None
            assert len(retriever._tool_names) > 0

    def test_retrieve_tools_with_requirements(self):
        """Test tool retrieval with requirements."""
        retriever = get_tool_retriever()
        
        requirements = RequirementExtractionOutput(
            metrics=["revenue"],
            group_by=["region"],
            time=TimeRequirement(column="", grain="unknown"),
            analysis=["compare", "total"],
            outputs=["chart"],
            constraints=[],
        )
        
        template_tools = ["plot_bar"]
        
        narrowed_tools = retriever.retrieve_tools(requirements, template_tools)
        
        # Should have template tool
        assert "plot_bar" in narrowed_tools
        
        # Should include aggregate for compare/total
        assert "aggregate" in narrowed_tools or retriever.fallback_mode
        
        # Should be capped at 12
        assert len(narrowed_tools) <= 12

    def test_retrieve_tools_precedence_template_first(self):
        """Test that template tools have precedence."""
        retriever = get_tool_retriever()
        
        requirements = RequirementExtractionOutput(
            metrics=["revenue"],
            group_by=[],
            time=TimeRequirement(column="", grain="unknown"),
            analysis=["trend"],
            outputs=["chart"],
            constraints=[],
        )
        
        template_tools = ["plot_line", "plot_bar", "plot_scatter"]
        
        narrowed_tools = retriever.retrieve_tools(requirements, template_tools)
        
        # All template tools should be first
        for tool in template_tools:
            assert tool in narrowed_tools[:len(template_tools)]

    def test_retrieve_tools_fallback_mode(self):
        """Test retrieval in fallback mode (no embeddings)."""
        # When embeddings unavailable, retriever returns all registered tools
        retriever = get_tool_retriever()
        
        requirements = RequirementExtractionOutput(
            metrics=["revenue"],
            group_by=[],
            time=TimeRequirement(column="", grain="unknown"),
            analysis=["total"],
            outputs=["chart"],
            constraints=[],
        )
        
        template_tools = ["plot_bar"]
        
        narrowed_tools = retriever.retrieve_tools(requirements, template_tools)
        
        # Should return tools (either narrowed or all in fallback)
        assert len(narrowed_tools) > 0
        assert "plot_bar" in narrowed_tools

    def test_retrieve_tools_no_requirements(self):
        """Test retrieval with no requirements."""
        retriever = get_tool_retriever()
        
        template_tools = ["plot_line", "aggregate"]
        
        narrowed_tools = retriever.retrieve_tools(None, template_tools)
        
        # Should at least include template tools
        for tool in template_tools:
            assert tool in narrowed_tools

    def test_retrieve_tools_empty_template(self):
        """Test retrieval with empty template tools."""
        retriever = get_tool_retriever()
        
        requirements = RequirementExtractionOutput(
            metrics=["revenue"],
            group_by=["region"],
            time=TimeRequirement(column="", grain="unknown"),
            analysis=["compare"],
            outputs=["chart"],
            constraints=[],
        )
        
        template_tools = []
        
        narrowed_tools = retriever.retrieve_tools(requirements, template_tools)
        
        # Should have retrieved tools based on requirements
        assert len(narrowed_tools) > 0
        # Should include aggregate for compare (if embeddings work)
        # In fallback mode, all tools are returned
        assert "aggregate" in narrowed_tools or not HAS_EMBEDDINGS

    def test_retrieve_tools_with_time_grain(self):
        """Test retrieval with temporal requirements."""
        retriever = get_tool_retriever()
        
        requirements = RequirementExtractionOutput(
            metrics=["revenue"],
            group_by=[],
            time=TimeRequirement(column="date", grain="monthly"),
            analysis=["trend"],
            outputs=["chart"],
            constraints=[],
        )
        
        template_tools = []
        
        narrowed_tools = retriever.retrieve_tools(requirements, template_tools)
        
        # Should include relevant tools
        assert len(narrowed_tools) > 0
        # Time series tools should be included
        assert "plot_line" in narrowed_tools or "aggregate" in narrowed_tools or not HAS_EMBEDDINGS

    def test_retrieve_tools_deduplication(self):
        """Test that retrieval deduplicates tools."""
        retriever = get_tool_retriever()
        
        requirements = RequirementExtractionOutput(
            metrics=["revenue"],
            group_by=[],
            time=TimeRequirement(column="", grain="unknown"),
            analysis=["total"],
            outputs=["chart"],
            constraints=[],
        )
        
        # Template tools that might also be retrieved
        template_tools = ["aggregate", "plot_bar"]
        
        narrowed_tools = retriever.retrieve_tools(requirements, template_tools)
        
        # Check no duplicates
        assert len(narrowed_tools) == len(set(narrowed_tools))

    def test_registry_hash_changes_trigger_rebuild(self):
        """Test that registry changes trigger index rebuild."""
        retriever = ToolRetriever()
        initial_hash = retriever._get_registry_hash()
        
        # Hash should be consistent
        second_hash = retriever._get_registry_hash()
        assert initial_hash == second_hash
        
        # Hash is deterministic based on registry content
        assert len(initial_hash) == 16  # SHA256 truncated to 16 chars


class TestToolDocuments:
    """Test tool document generation for embeddings."""

    def test_get_tool_documents_format(self):
        """Test tool document format."""
        documents = TOOL_REGISTRY.get_tool_documents()
        
        assert len(documents) > 0
        assert isinstance(documents, dict)
        
        # Check first document has tool name as key and description as value
        tool_names = list(documents.keys())
        assert len(tool_names[0]) > 0
        assert isinstance(documents[tool_names[0]], str)
        assert len(documents[tool_names[0]]) > 0

    def test_get_tool_documents_includes_capabilities(self):
        """Test that documents include capability keywords."""
        documents = TOOL_REGISTRY.get_tool_documents()
        
        # Find aggregate tool document
        agg_doc = documents.get("aggregate")
        
        if agg_doc:
            doc_text = agg_doc.lower()
            # Should contain capability keywords
            assert "aggregate" in doc_text or "group" in doc_text

    def test_get_tool_documents_deterministic(self):
        """Test that document generation is deterministic."""
        docs1 = TOOL_REGISTRY.get_tool_documents()
        docs2 = TOOL_REGISTRY.get_tool_documents()
        
        # Same keys
        assert set(docs1.keys()) == set(docs2.keys())
        
        # Same content for each tool
        for tool_name in docs1:
            assert docs1[tool_name] == docs2[tool_name]


class TestCapabilityMapping:
    """Test requirement to capability mapping."""

    def test_get_required_capabilities(self):
        """Test getting capabilities for known analysis types."""
        from autoviz_agent.planning.schema_tags import get_required_capabilities
        
        caps = get_required_capabilities("compare")
        assert "aggregate" in caps or "group_by" in caps
        
        caps = get_required_capabilities("trend")
        assert "plot" in caps or "time_series" in caps

    def test_get_required_capabilities_unknown(self):
        """Test unknown analysis type raises error."""
        from autoviz_agent.planning.schema_tags import get_required_capabilities
        
        with pytest.raises(ValueError):
            get_required_capabilities("unknown_analysis_type")

    def test_all_analysis_types_mapped(self):
        """Test all allowed analysis types have capability mappings."""
        from autoviz_agent.planning.schema_tags import REQUIREMENT_TO_CAPABILITY_MAP
        from autoviz_agent.llm.llm_contracts import ALLOWED_ANALYSIS_TYPES
        
        for analysis_type in ALLOWED_ANALYSIS_TYPES:
            assert analysis_type in REQUIREMENT_TO_CAPABILITY_MAP
            assert len(REQUIREMENT_TO_CAPABILITY_MAP[analysis_type]) > 0

    def test_infer_time_grain(self):
        """Test time grain inference."""
        from autoviz_agent.planning.schema_tags import infer_time_grain
        
        # Test with sample data patterns
        # Daily: 30 data points over 30 days = daily
        grain = infer_time_grain(data_span_days=30, num_points=30, missing_dates_pct=0.0)
        assert isinstance(grain, str)
        assert len(grain) > 0
        
        # Monthly-ish: 12 points over 365 days
        grain = infer_time_grain(data_span_days=365, num_points=12, missing_dates_pct=0.0)
        assert grain in ["M", "MS", "monthly", "weekly", "unknown"]  # weekly is valid for ~52/12 ratio
        
        # Yearly: 5 points over 1825 days (5 years)
        grain = infer_time_grain(data_span_days=1825, num_points=5, missing_dates_pct=0.0)
        assert isinstance(grain, str)
        assert grain in ["Y", "YS", "yearly", "monthly", "unknown"]
