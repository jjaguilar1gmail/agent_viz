"""Centralized prompt templates for LLM operations."""

import json
from pathlib import Path
from typing import Any, Dict, List
from autoviz_agent.models.state import Intent, SchemaProfile


# JSON Schemas for LLM outputs
INTENT_SCHEMA = {
    "type": "object",
    "properties": {
        "primary": {
            "type": "string",
            "enum": ["general_eda", "time_series_investigation", "anomaly_detection", "comparative_analysis"]
        },
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "reasoning": {"type": "string"}
    },
    "required": ["primary", "confidence", "reasoning"]
}

ADAPTATION_SCHEMA = {
    "type": "object",
    "properties": {
        "changes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["add", "remove", "modify"]},
                    "step_id": {"type": "string"},
                    "tool": {"type": "string"},
                    "description": {"type": "string"},
                    "params": {"type": "object"}
                },
                "required": ["action", "step_id"]
            }
        },
        "rationale": {"type": "string"}
    },
    "required": ["changes", "rationale"]
}


class PromptBuilder:
    """Builder for LLM prompts with template support."""

    def __init__(self, template_dir: Path | None = None):
        """
        Initialize prompt builder.

        Args:
            template_dir: Optional directory containing template files (.md, .j2).
                         If None, uses embedded templates.
        """
        self.template_dir = template_dir
        self._templates_cache: Dict[str, str] = {}

    def _load_template(self, name: str) -> str | None:
        """Load template from file if template_dir is set."""
        if not self.template_dir:
            return None
        
        template_path = self.template_dir / f"{name}.md"
        if not template_path.exists():
            return None
        
        # Cache template
        if name not in self._templates_cache:
            self._templates_cache[name] = template_path.read_text(encoding='utf-8')
        
        return self._templates_cache[name]

    def build_intent_prompt(self, question: str, schema: SchemaProfile) -> str:
        """
        Build prompt for intent classification.

        Args:
            question: User's question
            schema: Dataset schema profile

        Returns:
            Formatted prompt string
        """
        # Try loading from template file first
        template = self._load_template("intent")
        if template:
            return self._format_intent_template(template, question, schema)
        
        # Fall back to embedded prompt
        return self._build_embedded_intent_prompt(question, schema)

    def _format_intent_template(self, template: str, question: str, schema: SchemaProfile) -> str:
        """Format intent template with variables."""
        column_info = ', '.join(f"{c.name}({c.dtype})" for c in schema.columns[:5])
        if len(schema.columns) > 5:
            column_info += '...'
        
        temporal_cols = [c.name for c in schema.columns if 'temporal' in c.roles]
        numeric_cols = [c.name for c in schema.columns if c.dtype in ['int64', 'float64']]
        
        return template.format(
            question=question,
            row_count=schema.row_count,
            column_count=len(schema.columns),
            column_info=column_info,
            temporal_cols=', '.join(temporal_cols) if temporal_cols else 'none',
            numeric_col_count=len(numeric_cols),
            schema_json=json.dumps(INTENT_SCHEMA, indent=2)
        )

    def _build_embedded_intent_prompt(self, question: str, schema: SchemaProfile) -> str:
        """Build intent prompt using embedded template."""
        column_info = ', '.join(f"{c.name}({c.dtype})" for c in schema.columns[:5])
        if len(schema.columns) > 5:
            column_info += '...'
        
        temporal_cols = [c.name for c in schema.columns if 'temporal' in c.roles]
        numeric_cols = [c.name for c in schema.columns if c.dtype in ['int64', 'float64']]
        
        return f"""Classify the user's analytical intent based on their question and the dataset structure.

AVAILABLE INTENTS (with templates):
1. general_eda - Broad exploration (e.g., "summarize this data", "what's in here?")
2. time_series_investigation - Temporal patterns (e.g., "trends over time", "seasonal patterns")
3. anomaly_detection - Outliers and unusual values (e.g., "find anomalies", "detect outliers")
4. comparative_analysis - Compare groups/categories (e.g., "compare by region", "revenue by product")

INTENT SELECTION RULES:
- "compare", "by", "across", "versus", "difference between" → comparative_analysis
- "time", "trend", "over time", "temporal" → time_series_investigation
- "anomaly", "outlier", "unusual", "abnormal" → anomaly_detection
- General questions → general_eda

USER QUESTION: "{question}"

DATASET SCHEMA:
- Rows: {schema.row_count}, Columns: {len(schema.columns)}
- Column names: {column_info}
- Temporal columns: {', '.join(temporal_cols) if temporal_cols else 'none'}
- Numeric columns: {len(numeric_cols)}

EXAMPLES:
Question: "Compare revenue by region and product" → {{"primary": "comparative_analysis", "confidence": 0.95, "reasoning": "Keywords 'compare' and 'by' indicate comparison across categories"}}
Question: "Analyze revenue trends over time" → {{"primary": "time_series_investigation", "confidence": 0.95, "reasoning": "Keywords 'trends' and 'over time' indicate temporal analysis"}}
Question: "Find unusual sales patterns" → {{"primary": "anomaly_detection", "confidence": 0.90, "reasoning": "User explicitly asks for 'unusual' patterns"}}

Your response (JSON only, no other text):
{{"primary": "<intent>", "confidence": 0.0-1.0, "reasoning": "<why you chose this intent>"}}

Response:"""

    def build_adaptation_prompt(
        self, 
        template_plan: Dict[str, Any],
        schema: SchemaProfile,
        intent: Intent,
        user_question: str
    ) -> str:
        """
        Build prompt for plan adaptation.

        Args:
            template_plan: Original plan template
            schema: Dataset schema profile
            intent: Classified intent
            user_question: User's question

        Returns:
            Formatted prompt string
        """
        # Try loading from template file first
        template = self._load_template("adapt_plan")
        if template:
            return self._format_adaptation_template(template, template_plan, schema, intent, user_question)
        
        # Fall back to embedded prompt
        return self._build_embedded_adaptation_prompt(template_plan, schema, intent, user_question)

    def _format_adaptation_template(
        self,
        template: str,
        plan: Dict[str, Any],
        schema: SchemaProfile,
        intent: Intent,
        question: str
    ) -> str:
        """Format adaptation template with variables."""
        template_steps = plan.get('steps', [])
        step_summary = "\n".join([
            f"  - {s.get('step_id')}: {s.get('tool')} - {s.get('description', '')}" 
            for s in template_steps[:5]
        ])
        
        temporal_cols = [c.name for c in schema.columns if 'temporal' in c.roles]
        numeric_cols = [c.name for c in schema.columns if c.dtype in ['int64', 'float64', 'float', 'int']]
        categorical_cols = [c.name for c in schema.columns if c.dtype in ['object', 'string', 'category']]
        
        # Build detailed column list
        column_details = '\n'.join([
            f"  - {c.name} ({c.dtype}, cardinality: {c.cardinality})"
            for c in schema.columns
        ])
        
        return template.format(
            question=question,
            intent_label=intent.label,
            template_id=plan.get('template_id'),
            step_count=len(template_steps),
            step_summary=step_summary,
            row_count=schema.row_count,
            column_count=len(schema.columns),
            column_details=column_details,
            temporal_cols=', '.join(temporal_cols) if temporal_cols else 'none',
            numeric_cols=', '.join(numeric_cols) if numeric_cols else 'none',
            categorical_cols=', '.join(categorical_cols) if categorical_cols else 'none',
            data_shape=schema.data_shape,
            schema_json=json.dumps(ADAPTATION_SCHEMA, indent=2)
        )

    def _build_embedded_adaptation_prompt(
        self,
        template: Dict[str, Any],
        schema: SchemaProfile,
        intent: Intent,
        question: str
    ) -> str:
        """Build adaptation prompt using embedded template."""
        template_steps = template.get('steps', [])
        step_summary = "\n".join([
            f"  - {s.get('step_id')}: {s.get('tool')} - {s.get('description', '')}" 
            for s in template_steps[:5]
        ])
        
        temporal_cols = [c.name for c in schema.columns if 'temporal' in c.roles]
        numeric_cols = [c.name for c in schema.columns if c.dtype in ['int64', 'float64', 'float', 'int']]
        categorical_cols = [c.name for c in schema.columns if c.dtype in ['object', 'string', 'category']]
        
        # Build detailed column list
        column_details = '\n'.join([
            f"  - {c.name} ({c.dtype}, cardinality: {c.cardinality})"
            for c in schema.columns
        ])
        
        return f"""Review this analysis plan template and suggest modifications based on the user's specific question.

USER QUESTION: "{question}"
INTENT: {intent.label}

TEMPLATE: {template.get('template_id')} ({len(template_steps)} steps)
Current steps:
{step_summary}

DATASET CONTEXT:
- Rows: {schema.row_count}, Columns: {len(schema.columns)}
- Shape: {schema.data_shape}

AVAILABLE COLUMNS (use EXACT names):
{column_details}
- Temporal columns: {', '.join(temporal_cols) if temporal_cols else 'none'}
- Numeric columns: {', '.join(numeric_cols) if numeric_cols else 'none'}
- Categorical columns: {', '.join(categorical_cols) if categorical_cols else 'none'}

AVAILABLE TOOLS (use EXACT names):
- aggregate: Group data and compute aggregations (use for "compare by", "group by")
- segment_metric: Segment metric by category
- detect_anomalies: Detect outliers or unusual values
- compute_summary_stats: Basic statistics (mean, std, etc.)
- compute_correlations: Correlation matrix
- compute_distributions: Distribution analysis
- plot_line, plot_bar, plot_scatter, plot_histogram, plot_heatmap, plot_boxplot

YOUR TASK:
1. Check if the user question mentions specific requirements not in the template
2. Look for keywords like "anomaly", "outlier", "unusual", "compare", "segment", "group by"
3. Suggest adding, removing, or modifying steps to better match the question

EXAMPLES:
- User asks "find outliers" → add step with tool="detect_anomalies"
- User asks "compare revenue by region" → add step with tool="aggregate", params={{"group_by": ["region"], "agg_func": "sum"}}
- User asks "compare by region and product" → add step with tool="aggregate", params={{"group_by": ["region", "product_category"], "agg_func": "sum"}}
- Template fits perfectly → return empty changes array

CRITICAL RULES:
1. ONLY use column names that appear in "AVAILABLE COLUMNS" above
2. If the user mentions a concept (like "product type") but the actual column is named differently (like "product_category"), use the ACTUAL column name
3. Never invent or guess column names - only use what you see in the schema

RESPONSE FORMAT (JSON only, no preamble):
{{"changes": [{{"action": "add|remove|modify", "step_id": "<id>", "tool": "<exact_tool_name>", "description": "<reason>", "params": {{"df": "$dataframe"}}}}], "rationale": "<overall explanation>"}}

IMPORTANT: 
- tool MUST be an exact tool name from the list above (e.g., "aggregate", NOT "aggregate by region")
- For grouping/comparing, use tool="aggregate" with group_by parameter
- step_id must be unique (e.g., "compare_by_region_product")
- Column names in params MUST exactly match the names from "AVAILABLE COLUMNS"

Your JSON response:"""

    def get_schema(self, prompt_type: str) -> Dict[str, Any] | None:
        """
        Get JSON schema for prompt output validation.

        Args:
            prompt_type: Type of prompt ("intent" or "adaptation")

        Returns:
            JSON schema dict or None if not found
        """
        schemas = {
            "intent": INTENT_SCHEMA,
            "adaptation": ADAPTATION_SCHEMA
        }
        return schemas.get(prompt_type)
