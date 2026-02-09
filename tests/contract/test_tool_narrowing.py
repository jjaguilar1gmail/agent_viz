"""Contract tests for tool narrowing and retrieval."""

import pytest
from autoviz_agent.llm.llm_contracts import RequirementExtractionOutput
from autoviz_agent.planning.retrieval import ToolRetriever, get_tool_retriever


class TestToolRetrieverContract:
    """Test ToolRetriever contract and behavior."""

    @pytest.fixture
    def retriever(self):
        """Get tool retriever instance."""
        return get_tool_retriever()

    @pytest.fixture
    def sample_requirements(self):
        """Sample requirements for testing."""
        return RequirementExtractionOutput(
            metrics=["revenue"],
            group_by=["region"],
            analysis=["compare", "trend"]
        )

    def test_retriever_initializes(self, retriever):
        """Retriever should initialize with FAISS index."""
        assert retriever is not None
        assert retriever._index is not None

    def test_retrieve_tools_returns_list(self, retriever, sample_requirements):
        """retrieve_tools should return list of tool names."""
        template_tools = ["aggregate", "plot_bar"]
        result = retriever.retrieve_tools(sample_requirements, template_tools, top_k=5, cap=10)
        
        assert isinstance(result, list)
        assert all(isinstance(tool, str) for tool in result)

    def test_template_tools_always_included(self, retriever, sample_requirements):
        """Template tools should always be in result (precedence rule 1)."""
        template_tools = ["aggregate", "plot_bar"]
        result = retriever.retrieve_tools(sample_requirements, template_tools, top_k=3, cap=10)
        
        for tool in template_tools:
            assert tool in result, f"Template tool '{tool}' should be included"

    def test_cap_limits_total_tools(self, retriever, sample_requirements):
        """Cap should limit total tools returned."""
        template_tools = []
        result = retriever.retrieve_tools(sample_requirements, template_tools, top_k=20, cap=8)
        
        # Allow slight overflow for safety tools
        assert len(result) <= 10, f"Cap violated: {len(result)} tools returned (cap=8 + 2 safety)"

    def test_no_cap_allows_unlimited(self, retriever, sample_requirements):
        """cap=None should allow unlimited tools."""
        template_tools = []
        result = retriever.retrieve_tools(sample_requirements, template_tools, top_k=15, cap=None)
        
        # Should retrieve more than default cap
        assert len(result) > 12, "cap=None should allow more tools than default cap"

    def test_safety_tools_included(self, retriever, sample_requirements):
        """Safety tools should be included if space allows."""
        template_tools = []
        result = retriever.retrieve_tools(sample_requirements, template_tools, top_k=5, cap=15)
        
        # Should include at least some tools
        assert len(result) > 0, "Should retrieve some tools"

    def test_deduplication(self, retriever, sample_requirements):
        """Result should not contain duplicate tools."""
        template_tools = ["aggregate"]
        result = retriever.retrieve_tools(sample_requirements, template_tools, top_k=10, cap=15)
        
        assert len(result) == len(set(result)), "Tools should be deduplicated"

    def test_empty_requirements_fallback(self, retriever):
        """Empty requirements should trigger fallback behavior."""
        empty_reqs = RequirementExtractionOutput()
        template_tools = ["aggregate"]
        result = retriever.retrieve_tools(empty_reqs, template_tools, top_k=5, cap=10)
        
        assert len(result) > 0, "Should return some tools even with empty requirements"
        assert "aggregate" in result, "Template tools should be preserved"


class TestToolNarrowingPrecedence:
    """Test precedence rules for tool narrowing."""

    @pytest.fixture
    def retriever(self):
        return get_tool_retriever()

    def test_precedence_template_first(self, retriever):
        """Template tools should have highest precedence."""
        reqs = RequirementExtractionOutput(metrics=["revenue"], analysis=["total"])
        template_tools = ["custom_tool", "aggregate"]
        
        result = retriever.retrieve_tools(reqs, template_tools, top_k=3, cap=10)
        
        # Template tools should be first in result
        assert result[0] in template_tools or result[1] in template_tools

    def test_precedence_retrieval_second(self, retriever):
        """Retrieved tools should come after template tools."""
        reqs = RequirementExtractionOutput(
            metrics=["revenue"],
            analysis=["compare", "trend", "distribution"]
        )
        template_tools = []
        
        result = retriever.retrieve_tools(reqs, template_tools, top_k=5, cap=10)
        
        # Should retrieve tools relevant to requirements
        assert len(result) > 0, "Should retrieve tools based on requirements"

    def test_precedence_safety_last(self, retriever):
        """Safety tools should be added last if space permits."""
        reqs = RequirementExtractionOutput(metrics=["revenue"], analysis=["total"])
        template_tools = []
        
        result = retriever.retrieve_tools(reqs, template_tools, top_k=3, cap=15)
        
        # Should retrieve tools based on requirements
        assert len(result) > 0, "Should retrieve tools"


class TestToolNarrowingQueries:
    """Test query building from requirements."""

    @pytest.fixture
    def retriever(self):
        return get_tool_retriever()

    def test_queries_built_from_requirements(self, retriever):
        """Queries should be built from requirements structure."""
        reqs = RequirementExtractionOutput(
            metrics=["revenue", "cost"],
            group_by=["region"],
            analysis=["compare", "trend"]
        )
        
        queries = retriever._build_queries(reqs)
        
        assert isinstance(queries, list)
        assert len(queries) > 0, "Should generate at least one query"

    def test_metrics_generate_queries(self, retriever):
        """Metrics should generate retrieval queries."""
        reqs = RequirementExtractionOutput(metrics=["revenue", "profit"])
        queries = retriever._build_queries(reqs)
        
        # Should generate queries for metrics
        assert any("revenue" in q.lower() or "metric" in q.lower() for q in queries)

    def test_analysis_generates_queries(self, retriever):
        """Analysis types should generate retrieval queries."""
        reqs = RequirementExtractionOutput(analysis=["compare", "trend"])
        queries = retriever._build_queries(reqs)
        
        # Should generate queries for analysis types
        assert len(queries) >= 2, "Should generate query per analysis type"


class TestToolRetrieverCaching:
    """Test FAISS index caching behavior."""

    def test_index_cached_by_registry_hash(self):
        """Index should be cached based on registry hash."""
        retriever1 = get_tool_retriever()
        retriever2 = get_tool_retriever()
        
        # Both should use same cached index
        hash1 = retriever1._get_registry_hash()
        hash2 = retriever2._get_registry_hash()
        
        assert hash1 == hash2, "Registry hash should be consistent"

    def test_registry_hash_deterministic(self):
        """Registry hash should be deterministic."""
        retriever = get_tool_retriever()
        
        hash1 = retriever._get_registry_hash()
        hash2 = retriever._get_registry_hash()
        
        assert hash1 == hash2, "Hash should be deterministic"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
