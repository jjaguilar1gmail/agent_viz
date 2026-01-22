"""Generic column selection by role."""

from typing import Dict, List, Optional

from autoviz_agent.models.state import SchemaProfile
from autoviz_agent.utils.logging import get_logger

logger = get_logger(__name__)


class ColumnSelector:
    """Select columns by role with user mention prioritization."""

    def __init__(
        self,
        schema: SchemaProfile,
        user_question: Optional[str] = None,
    ):
        """
        Initialize column selector.

        Args:
            schema: Dataset schema profile
            user_question: User's original question for extracting mentioned columns
        """
        self.schema = schema
        self.user_question = user_question or ""
        
        # Cache column lists by role
        self._temporal_cols = [c.name for c in schema.columns if 'temporal' in c.roles]
        self._numeric_cols = [c.name for c in schema.columns if c.dtype in ['int64', 'float64', 'int32', 'float32']]
        self._categorical_cols = [
            c.name for c in schema.columns 
            if 'categorical' in c.roles and 'temporal' not in c.roles
        ]
        self._all_categorical_cols = [c.name for c in schema.columns if 'categorical' in c.roles]
        
        # Extract columns mentioned in user question
        self._mentioned_cols = self._extract_mentioned_columns()

    def _extract_mentioned_columns(self) -> Dict[str, List[str]]:
        """
        Extract column names mentioned in user question using keyword matching.
        
        Returns:
            Dict with 'categorical' and 'numeric' lists of mentioned columns
        """
        if not self.user_question:
            return {'categorical': [], 'numeric': []}
        
        question_lower = self.user_question.lower()
        mentioned_categorical = []
        mentioned_numeric = []
        
        # Check each column name to see if it's mentioned in the question
        # Handle variations like "product type" matching "product_category"
        for col in self._categorical_cols + self._all_categorical_cols:
            col_variants = [
                col.lower(),
                col.lower().replace('_', ' '),  # product_category -> product category
                col.lower().replace('_', ''),   # product_category -> productcategory
            ]
            # Also check for partial matches (e.g., "product" in "product type" matching "product_category")
            col_parts = col.lower().split('_')
            
            # Check exact and underscore variants
            if any(variant in question_lower for variant in col_variants):
                if col not in mentioned_categorical:
                    mentioned_categorical.append(col)
            # Check if any significant part of column name appears in question
            # (e.g., "product" from "product_category" matching "product type")
            elif any(len(part) > 3 and part in question_lower for part in col_parts):
                if col not in mentioned_categorical:
                    mentioned_categorical.append(col)
        
        for col in self._numeric_cols:
            col_variants = [
                col.lower(),
                col.lower().replace('_', ' '),
                col.lower().replace('_', ''),
            ]
            if any(variant in question_lower for variant in col_variants):
                if col not in mentioned_numeric:
                    mentioned_numeric.append(col)
        
        if mentioned_categorical or mentioned_numeric:
            logger.info(f"Extracted from question - categorical: {mentioned_categorical}, numeric: {mentioned_numeric}")
        
        return {'categorical': mentioned_categorical, 'numeric': mentioned_numeric}

    def select(self, role: str, count: int = 1, exclude: Optional[List[str]] = None) -> List[str]:
        """
        Select columns by role with prioritization.

        Args:
            role: Column role (temporal/numeric/categorical/any)
            count: Number of columns to select
            exclude: Columns to exclude from selection

        Returns:
            List of selected column names
        """
        exclude = exclude or []
        
        if role == "temporal":
            candidates = [c for c in self._temporal_cols if c not in exclude]
        elif role == "numeric":
            # Priority 1: Numeric columns mentioned in question
            mentioned = [c for c in self._mentioned_cols['numeric'] if c not in exclude]
            # Priority 2: All numeric columns
            all_numeric = [c for c in self._numeric_cols if c not in exclude]
            candidates = mentioned + [c for c in all_numeric if c not in mentioned]
        elif role == "categorical":
            # Priority 1: Categorical columns mentioned in question
            mentioned = [c for c in self._mentioned_cols['categorical'] if c not in exclude]
            # Priority 2: Non-temporal categorical
            non_temporal = [c for c in self._categorical_cols if c not in exclude]
            # Priority 3: All categorical
            all_categorical = [c for c in self._all_categorical_cols if c not in exclude]
            candidates = mentioned + [c for c in non_temporal if c not in mentioned] + [c for c in all_categorical if c not in mentioned and c not in non_temporal]
        elif role == "any":
            # Return any available columns
            candidates = [c.name for c in self.schema.columns if c.name not in exclude]
        else:
            logger.warning(f"Unknown role: {role}, returning empty list")
            return []
        
        selected = candidates[:count] if candidates else []
        
        if selected:
            logger.debug(f"Selected {len(selected)} columns for role '{role}': {selected}")
        
        return selected

    def get_temporal_cols(self) -> List[str]:
        """Get all temporal columns."""
        return self._temporal_cols.copy()

    def get_numeric_cols(self) -> List[str]:
        """Get all numeric columns."""
        return self._numeric_cols.copy()

    def get_categorical_cols(self) -> List[str]:
        """Get non-temporal categorical columns."""
        return self._categorical_cols.copy()

    def get_all_categorical_cols(self) -> List[str]:
        """Get all categorical columns including temporal."""
        return self._all_categorical_cols.copy()

    def get_mentioned_cols(self) -> Dict[str, List[str]]:
        """Get columns mentioned in user question."""
        return self._mentioned_cols.copy()
