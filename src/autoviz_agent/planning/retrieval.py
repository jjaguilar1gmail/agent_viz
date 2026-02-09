"""Plan retrieval algorithm and scoring + tool retrieval with embeddings."""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from autoviz_agent.models.state import Intent, SchemaProfile
from autoviz_agent.utils.logging import get_logger
from autoviz_agent.registry.tools import TOOL_REGISTRY, ensure_default_tools_registered
from autoviz_agent.llm.llm_contracts import RequirementExtractionOutput

logger = get_logger(__name__)


# Optional dependencies for embeddings
try:
    import faiss
    from sentence_transformers import SentenceTransformer
    HAS_EMBEDDINGS = True
except ImportError:
    HAS_EMBEDDINGS = False
    logger.warning(
        "faiss-cpu or sentence-transformers not installed. "
        "Tool retrieval will use fallback mode. "
        "Install with: pip install faiss-cpu sentence-transformers"
    )


class PlanRetrieval:
    """Plan retrieval with deterministic scoring."""

    def __init__(self, templates: Dict[str, Dict[str, Any]]):
        """
        Initialize plan retrieval.

        Args:
            templates: Dictionary of template_id to template data
        """
        self.templates = templates

    def retrieve(
        self, intent: Intent, schema: SchemaProfile, top_k: int = 3
    ) -> List[Tuple[str, float]]:
        """
        Retrieve best matching templates.

        Args:
            intent: Classified user intent
            schema: Inferred dataset schema
            top_k: Number of top candidates to return

        Returns:
            List of (template_id, score) tuples sorted by score descending
        """
        candidates = []

        for template_id, template in self.templates.items():
            score = self._score_template(template, intent, schema)
            if score > 0:
                candidates.append((template_id, score))

        # Sort by score descending, then by template_id for determinism
        candidates.sort(key=lambda x: (-x[1], x[0]))

        return candidates[:top_k]

    def _score_template(
        self, template: Dict[str, Any], intent: Intent, schema: SchemaProfile
    ) -> float:
        """
        Score a template against intent and schema.

        Args:
            template: Template data
            intent: User intent
            schema: Dataset schema

        Returns:
            Score (0-1 range, higher is better)
        """
        score = 0.0

        # Intent match (hard filter + scoring)
        if intent.label.value not in template.get("intents", []):
            return 0.0  # Hard filter: intent must match

        score += 0.5  # Base score for intent match

        # Data shape match
        if schema.data_shape in template.get("data_shape", []):
            score += 0.2

        # Requirements check (hard filters)
        requires = template.get("requires", {})
        if requires.get("min_rows", 0) > schema.row_count:
            return 0.0
        if requires.get("min_columns", 0) > len(schema.columns):
            return 0.0

        # Preferences (soft scoring)
        prefers = template.get("prefers", {})
        if prefers.get("has_datetime", False):
            if any(col.dtype in ["datetime64", "datetime"] for col in schema.columns):
                score += 0.1

        if prefers.get("has_categorical", False):
            if any(col.dtype in ["object", "category"] for col in schema.columns):
                score += 0.1

        if prefers.get("has_numeric", False):
            if any(
                col.dtype in ["int64", "float64", "int32", "float32"] for col in schema.columns
            ):
                score += 0.1

        return min(score, 1.0)  # Cap at 1.0

    def select_best(self, intent: Intent, schema: SchemaProfile) -> str:
        """
        Select the single best template.

        Args:
            intent: User intent
            schema: Dataset schema

        Returns:
            Template ID

        Raises:
            ValueError: If no matching templates found
        """
        candidates = self.retrieve(intent, schema, top_k=1)
        
        # Fallback: If no exact intent match, try with general_eda
        if not candidates:
            logger.warning(f"No templates found for intent={intent.label}, falling back to general_eda")
            
            # Try to find a general_eda template
            for template_id, template in self.templates.items():
                if "general_eda" in template.get("intents", []):
                    logger.info(f"Selected fallback template: {template_id}")
                    return template_id
            
            # If still no match, just pick the first available template
            if self.templates:
                fallback_id = next(iter(self.templates.keys()))
                logger.warning(f"Using first available template as last resort: {fallback_id}")
                return fallback_id
            
            # Truly no templates available
            raise ValueError(f"No templates available in the system")

        template_id, score = candidates[0]
        logger.info(f"Selected template: {template_id} (score={score:.2f})")
        return template_id


# =============================================================================
# Tool Retrieval with Embeddings
# =============================================================================

class ToolRetriever:
    """Semantic tool retrieval with FAISS indexing."""
    
    DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
    CACHE_DIR = Path(".cache/tool_index")
    SAFETY_TOOLS = ["aggregate", "plot_line", "compute_summary_stats"]
    
    def __init__(
        self, 
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        cache_dir: Optional[Path] = None,
    ):
        """
        Initialize tool retriever.
        
        Args:
            embedding_model: Sentence transformer model name
            cache_dir: Directory for caching embeddings
        """
        self.embedding_model_name = embedding_model
        self.cache_dir = cache_dir or self.CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model and index
        self._model: Optional[Any] = None
        self._index: Optional[Any] = None
        self._tool_names: List[str] = []
        self._tool_documents: Dict[str, str] = {}
        
        if not HAS_EMBEDDINGS:
            logger.warning("Embeddings not available, tool retrieval will use fallback")
            return
        
        # Load or build index
        self._initialize_index()
    
    def _get_registry_hash(self) -> str:
        """
        Compute hash of tool registry for cache invalidation.
        
        Returns:
            Registry hash string
        """
        ensure_default_tools_registered()
        schemas = TOOL_REGISTRY.get_all_schemas()
        
        # Create deterministic representation
        schema_data = {
            name: {
                "description": s.description,
                "capabilities": sorted(s.capabilities),
                "params": [p.name for p in s.parameters],
                "returns": s.returns,
            }
            for name, s in sorted(schemas.items())
        }
        
        registry_str = json.dumps(schema_data, sort_keys=True)
        return hashlib.sha256(registry_str.encode()).hexdigest()[:16]
    
    def _initialize_index(self) -> None:
        """Load or build embedding index."""
        if not HAS_EMBEDDINGS:
            return
        
        registry_hash = self._get_registry_hash()
        index_path = self.cache_dir / f"index_{registry_hash}.faiss"
        meta_path = self.cache_dir / f"meta_{registry_hash}.json"
        
        # Check if cached index exists
        if index_path.exists() and meta_path.exists():
            try:
                self._load_index(index_path, meta_path)
                logger.info(f"Loaded tool index from cache (hash={registry_hash})")
                return
            except Exception as e:
                logger.warning(f"Failed to load cached index: {e}. Rebuilding...")
        
        # Build new index
        logger.info("Building tool embedding index...")
        self._build_index()
        
        # Save to cache
        try:
            self._save_index(index_path, meta_path)
            logger.info(f"Saved tool index to cache (hash={registry_hash})")
        except Exception as e:
            logger.warning(f"Failed to save index to cache: {e}")
    
    def _build_index(self) -> None:
        """Build FAISS index from tool documents."""
        if not HAS_EMBEDDINGS:
            return
        
        # Ensure tools are registered
        ensure_default_tools_registered()
        
        # Get tool documents
        self._tool_documents = TOOL_REGISTRY.get_tool_documents()
        self._tool_names = list(self._tool_documents.keys())
        
        if not self._tool_names:
            logger.warning("No tools found in registry")
            return
        
        # Initialize embedding model
        logger.info(f"Loading embedding model: {self.embedding_model_name}")
        self._model = SentenceTransformer(self.embedding_model_name)
        
        # Embed tool documents
        documents = [self._tool_documents[name] for name in self._tool_names]
        logger.info(f"Embedding {len(documents)} tool documents...")
        embeddings = self._model.encode(documents, show_progress_bar=False)
        
        # Build FAISS index
        dimension = embeddings.shape[1]
        self._index = faiss.IndexFlatL2(dimension)
        self._index.add(embeddings.astype('float32'))
        
        logger.info(f"Built FAISS index with {len(self._tool_names)} tools")
    
    def _load_index(self, index_path: Path, meta_path: Path) -> None:
        """Load index from cache."""
        if not HAS_EMBEDDINGS:
            return
        
        # Load metadata
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        
        self._tool_names = meta["tool_names"]
        self._tool_documents = meta["tool_documents"]
        
        # Load FAISS index
        self._index = faiss.read_index(str(index_path))
        
        # Load embedding model
        self._model = SentenceTransformer(self.embedding_model_name)
    
    def _save_index(self, index_path: Path, meta_path: Path) -> None:
        """Save index to cache."""
        if not HAS_EMBEDDINGS or self._index is None:
            return
        
        # Save metadata
        meta = {
            "tool_names": self._tool_names,
            "tool_documents": self._tool_documents,
        }
        with open(meta_path, 'w') as f:
            json.dump(meta, f)
        
        # Save FAISS index
        faiss.write_index(self._index, str(index_path))
    
    def retrieve_tools(
        self,
        requirements: RequirementExtractionOutput,
        template_tools: List[str],
        top_k: int = 5,
        cap: Optional[int] = 12,
    ) -> List[str]:
        """
        Retrieve relevant tools using precedence rules.
        
        Precedence:
        1. Template-curated tools (always included)
        2. Retrieved tools by embedding similarity
        3. Safety tools (if missing)
        
        Args:
            requirements: Extracted requirements
            template_tools: Template-curated tool list
            top_k: Number of tools to retrieve per query
            cap: Maximum total tools to return (None for no limit)
        
        Returns:
            List of tool names (deduplicated and capped)
        """
        if not HAS_EMBEDDINGS or self._index is None:
            # Fallback: return template tools + safety tools
            logger.warning("Embeddings not available, using fallback tool selection")
            return self._fallback_tool_selection(template_tools)
        
        # Start with template tools
        selected_tools = list(template_tools)
        
        # Build retrieval queries from requirements
        queries = self._build_queries(requirements)
        
        # Retrieve tools for each query
        retrieved_tools = []
        for query in queries:
            tools = self._search(query, top_k)
            retrieved_tools.extend(tools)
        
        # Deduplicate and add retrieved tools
        for tool in retrieved_tools:
            if tool not in selected_tools:
                if cap is None or len(selected_tools) < cap:
                    selected_tools.append(tool)
        
        # Add safety tools if missing and space allows
        for tool in self.SAFETY_TOOLS:
            if tool not in selected_tools:
                if cap is None or len(selected_tools) < cap + 2:  # Allow slight overflow for safety
                    selected_tools.append(tool)
        
        logger.info(f"Selected {len(selected_tools)} tools: {selected_tools[:5]}...")
        return selected_tools
    
    def _build_queries(self, requirements: RequirementExtractionOutput) -> List[str]:
        """
        Build retrieval queries from requirements.
        
        Args:
            requirements: Extracted requirements
        
        Returns:
            List of query strings
        """
        queries = []
        
        # Query from analysis types
        if requirements.analysis:
            analysis_str = " ".join(requirements.analysis)
            queries.append(f"Analysis: {analysis_str}")
        
        # Query from metrics and grouping
        if requirements.metrics:
            metric_str = " ".join(requirements.metrics)
            if requirements.group_by:
                group_str = " ".join(requirements.group_by)
                queries.append(f"Aggregate {metric_str} by {group_str}")
            else:
                queries.append(f"Compute {metric_str}")
        
        # Query from time requirements
        if requirements.time and requirements.time.column:
            queries.append(f"Time series analysis with {requirements.time.grain} grain")
        
        # Combined query
        combined_parts = []
        if requirements.analysis:
            combined_parts.extend(requirements.analysis)
        if requirements.metrics:
            combined_parts.extend(requirements.metrics)
        if requirements.group_by:
            combined_parts.append("grouped")
        if requirements.time and requirements.time.column:
            combined_parts.append("temporal")
        
        if combined_parts:
            queries.append(" ".join(combined_parts))
        
        return queries if queries else ["general analysis"]
    
    def _search(self, query: str, top_k: int) -> List[str]:
        """
        Search for tools by query.
        
        Args:
            query: Query string
            top_k: Number of results to return
        
        Returns:
            List of tool names
        """
        if not HAS_EMBEDDINGS or self._model is None or self._index is None:
            return []
        
        # Embed query
        query_embedding = self._model.encode([query], show_progress_bar=False)
        
        # Search index
        distances, indices = self._index.search(
            query_embedding.astype('float32'), 
            min(top_k, len(self._tool_names))
        )
        
        # Get tool names
        tools = [self._tool_names[idx] for idx in indices[0]]
        return tools
    
    def _fallback_tool_selection(self, template_tools: List[str]) -> List[str]:
        """
        Fallback tool selection when embeddings not available.
        
        Args:
            template_tools: Template-curated tools
        
        Returns:
            List of tool names
        """
        selected = list(template_tools)
        
        # Add safety tools if missing
        for tool in self.SAFETY_TOOLS:
            if tool not in selected:
                selected.append(tool)
        
        return selected


# Global retriever instance (lazy initialization)
_retriever: Optional[ToolRetriever] = None


def get_tool_retriever() -> ToolRetriever:
    """
    Get global tool retriever instance.
    
    Returns:
        ToolRetriever instance
    """
    global _retriever
    if _retriever is None:
        _retriever = ToolRetriever()
    return _retriever
