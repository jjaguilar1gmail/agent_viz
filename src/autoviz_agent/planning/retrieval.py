"""Plan retrieval algorithm and scoring."""

from typing import Any, Dict, List, Tuple

from autoviz_agent.models.state import Intent, SchemaProfile
from autoviz_agent.utils.logging import get_logger

logger = get_logger(__name__)


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
