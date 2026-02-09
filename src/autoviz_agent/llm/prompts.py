"""Centralized prompt templates for LLM operations."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from autoviz_agent.models.state import Intent, SchemaProfile
from autoviz_agent.registry.intents import (
    get_intent_labels,
    render_intent_catalog,
    render_intent_examples,
    render_intent_rules,
)
from autoviz_agent.registry.tools import TOOL_REGISTRY, ensure_default_tools_registered
from autoviz_agent.llm.llm_contracts import (
    get_intent_schema,
    get_adaptation_schema,
    get_requirement_extraction_schema,
    get_param_fill_schema,
    get_tool_selection_schema,
    ALLOWED_ANALYSIS_TYPES,
)


# JSON Schemas for LLM outputs (legacy, use llm_contracts for new code)
INTENT_SCHEMA = {
    "type": "object",
    "properties": {
        "primary": {
            "type": "string",
            "enum": get_intent_labels(exposed_only=True),
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
                    "params": {"type": "object"},
                    "satisfies": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["action", "step_id", "satisfies"]
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
            intent_catalog=render_intent_catalog(),
            intent_rules=render_intent_rules(),
            intent_examples=render_intent_examples(),
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
{render_intent_catalog()}

INTENT SELECTION RULES:
{render_intent_rules()}

USER QUESTION: "{question}"

DATASET SCHEMA:
- Rows: {schema.row_count}, Columns: {len(schema.columns)}
- Column names: {column_info}
- Temporal columns: {', '.join(temporal_cols) if temporal_cols else 'none'}
- Numeric columns: {len(numeric_cols)}

EXAMPLES:
{render_intent_examples()}

Your response (JSON only, no other text):
{{"primary": "<intent>", "confidence": 0.0-1.0, "reasoning": "<why you chose this intent>"}}

Response:"""

    def build_adaptation_prompt(
        self, 
        template_plan: Dict[str, Any],
        schema: SchemaProfile,
        intent: Intent,
        user_question: str,
        narrowed_tools: Optional[List[str]] = None,
        requirements: Optional[Any] = None,
        validation_errors: Optional[str] = None,
    ) -> str:
        """
        Build prompt for plan adaptation.

        Args:
            template_plan: Original plan template
            schema: Dataset schema profile
            intent: Classified intent
            user_question: User's question
            narrowed_tools: Optional narrowed tool list from retrieval

        Returns:
            Formatted prompt string
        """
        # Try loading from template file first
        template = self._load_template("adapt_plan")
        if template:
            return self._format_adaptation_template(
                template,
                template_plan,
                schema,
                intent,
                user_question,
                narrowed_tools,
                requirements,
                validation_errors,
            )
        
        # Fall back to embedded prompt
        return self._build_embedded_adaptation_prompt(
            template_plan,
            schema,
            intent,
            user_question,
            narrowed_tools,
            requirements,
            validation_errors,
        )

    def _format_adaptation_template(
        self,
        template: str,
        plan: Dict[str, Any],
        schema: SchemaProfile,
        intent: Intent,
        question: str,
        narrowed_tools: Optional[List[str]] = None,
        requirements: Optional[Any] = None,
        validation_errors: Optional[str] = None,
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
            tool_catalog=self._build_tool_catalog(narrowed_tools),
            requirements_summary=self._format_requirements_summary(requirements),
            validation_errors=validation_errors or "(none)",
            schema_json=json.dumps(ADAPTATION_SCHEMA, indent=2)
        )

    def _build_embedded_adaptation_prompt(
        self,
        template: Dict[str, Any],
        schema: SchemaProfile,
        intent: Intent,
        question: str,
        narrowed_tools: Optional[List[str]] = None,
        requirements: Optional[Any] = None,
        validation_errors: Optional[str] = None,
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

REQUIREMENTS (must be satisfied):
{self._format_requirements_summary(requirements)}

VALIDATION ERRORS (if any):
{validation_errors or "(none)"}

AVAILABLE TOOLS (use EXACT names):
{self._build_tool_catalog(narrowed_tools)}

YOUR TASK:
1. Use the Requirements section as the contract. Every requirement must be satisfied by at least one step.
2. If `group_by` is non-empty, add or modify an `aggregate` step with the required group_by columns.
3. If `time.column` is set, ensure a time-aware plot uses that column (and aggregated data if needed).
4. Remove steps that do not satisfy any requirement.
5. If Validation Errors are present, fix them explicitly.
6. Prefer tools whose capabilities directly match the requirements (e.g., use compare tools for compare needs).
7. Choose the minimal set of steps that fully covers the requirements.

EXAMPLES:
- Requirements include group_by=["region"] -> add `aggregate` with group_by=["region"], agg_func="sum"
- Requirements include analysis=["trend"] and time.column="date" -> ensure `plot_line` uses x="date"
- Requirements do NOT include "anomaly" -> remove or do not add `detect_anomalies`

CRITICAL RULES:
1. ONLY use column names that appear in "AVAILABLE COLUMNS" above
2. If the user mentions a concept (like "product type") but the actual column is named differently (like "product_category"), use the ACTUAL column name
3. Never invent or guess column names - only use what you see in the schema

RESPONSE FORMAT (JSON only, no preamble):
{{"changes": [{{"action": "add|remove|modify", "step_id": "<id>", "tool": "<exact_tool_name>", "description": "<reason>", "params": {{"df": "$dataframe"}}, "satisfies": ["analysis.total"]}}], "rationale": "<overall explanation>"}}

IMPORTANT:
- tool MUST be an exact tool name from the list above (e.g., "aggregate", NOT "aggregate by region")
- For grouping/comparing, use tool="aggregate" with group_by parameter
- step_id must be unique (e.g., "compare_by_region_product")
- Column names in params MUST exactly match the names from "AVAILABLE COLUMNS"
- Include a non-empty "satisfies" list for each added or modified step.

Your JSON response:"""

    def _build_tool_catalog(self, narrowed_tools: Optional[List[str]] = None) -> str:
        """Build tool catalog, optionally filtering to narrowed tool list."""
        ensure_default_tools_registered()
        schemas = TOOL_REGISTRY.get_all_schemas()
        if not schemas:
            return "- (no tools registered)"
        
        # Filter to narrowed tools if provided
        if narrowed_tools:
            schemas = {name: schemas[name] for name in narrowed_tools if name in schemas}
            if not schemas:
                # Fallback to all tools if narrowing resulted in empty list
                schemas = TOOL_REGISTRY.get_all_schemas()
        
        lines = []
        for name in sorted(schemas.keys()):
            description = (schemas[name].description or "").strip()
            if description:
                lines.append(f"- **{name}**: {description}")
            else:
                lines.append(f"- **{name}**")
        return "\n".join(lines)

    def _format_requirements_summary(self, requirements: Optional[Any]) -> str:
        """Format requirements summary for prompts."""
        if not requirements:
            return "- (none)"
        metrics = getattr(requirements, "metrics", []) or []
        group_by = getattr(requirements, "group_by", []) or []
        analysis = getattr(requirements, "analysis", []) or []
        outputs = getattr(requirements, "outputs", []) or []
        constraints = getattr(requirements, "constraints", []) or []
        time = getattr(requirements, "time", None)
        time_col = getattr(time, "column", "") if time else ""
        time_grain = getattr(time, "grain", "unknown") if time else "unknown"

        lines = [
            f"- metrics: {', '.join(metrics) if metrics else '[]'}",
            f"- group_by: {', '.join(group_by) if group_by else '[]'}",
            f"- time: column={time_col or 'none'}, grain={time_grain or 'unknown'}",
            f"- analysis: {', '.join(analysis) if analysis else '[]'}",
            f"- outputs: {', '.join(outputs) if outputs else '[]'}",
            f"- constraints: {', '.join(constraints) if constraints else '[]'}",
        ]
        return "\n".join(lines)

    def get_schema(self, prompt_type: str) -> Dict[str, Any] | None:
        """
        Get JSON schema for prompt output validation.

        Args:
            prompt_type: Type of prompt ("intent", "adaptation", or "requirements")

        Returns:
            JSON schema dict or None if not found
        """
        schemas = {
            "intent": get_intent_schema(),
            "adaptation": get_adaptation_schema(),
            "requirements": get_requirement_extraction_schema(),
            "param_fill": get_param_fill_schema(),
            "tool_selection": get_tool_selection_schema(),
        }
        return schemas.get(prompt_type)

    def build_tool_selection_prompt(
        self,
        capability_targets: List[str],
        candidate_tools: List[str],
        tool_catalog: str,
    ) -> str:
        """
        Build prompt for selecting tools from candidates.
        """
        template = self._load_template("select_tools")
        if template:
            return template.format(
                capability_targets=", ".join(capability_targets),
                candidate_tools=", ".join(candidate_tools) if candidate_tools else "none",
                tool_catalog=tool_catalog,
            )

        return f"""Select the minimal set of tools that covers all capability targets.

CAPABILITY TARGETS:
{", ".join(capability_targets)}

CANDIDATE TOOLS (choose only from this list):
{", ".join(candidate_tools) if candidate_tools else "none"}

TOOL CATALOG:
{tool_catalog}

RULES:
1. Only choose tools from the candidate list.
2. Choose the smallest set that covers all targets.
3. Prefer tools whose capabilities directly match the targets.

RESPONSE FORMAT (JSON only):
{{"selected_tools": ["tool_a", "tool_b"], "rationale": "<why>"}}
"""

    def build_param_fill_prompt(
        self,
        plan: Dict[str, Any],
        schema: SchemaProfile,
        requirements: Optional[Any] = None,
        validation_errors: Optional[str] = None,
    ) -> str:
        """
        Build prompt for filling tool parameters.
        """
        template = self._load_template("fill_params")
        if template:
            return self._format_param_fill_template(
                template, plan, schema, requirements, validation_errors
            )

        return self._build_embedded_param_fill_prompt(
            plan, schema, requirements, validation_errors
        )

    def _format_param_fill_template(
        self,
        template: str,
        plan: Dict[str, Any],
        schema: SchemaProfile,
        requirements: Optional[Any],
        validation_errors: Optional[str],
    ) -> str:
        step_summary = "\n".join(
            [
                f"  - {s.get('step_id')}: {s.get('tool')} ({s.get('description', '')})"
                for s in plan.get("steps", [])
            ]
        )
        temporal_cols = [c.name for c in schema.columns if 'temporal' in c.roles]
        numeric_cols = [c.name for c in schema.columns if c.dtype in ['int64', 'float64', 'float', 'int']]
        categorical_cols = [c.name for c in schema.columns if c.dtype in ['object', 'string', 'category']]
        column_details = '\n'.join(
            [f"  - {c.name} ({c.dtype}, cardinality: {c.cardinality})" for c in schema.columns]
        )

        param_constraints = self._format_param_constraints(plan)

        return template.format(
            step_summary=step_summary,
            row_count=schema.row_count,
            column_count=len(schema.columns),
            column_details=column_details,
            temporal_cols=', '.join(temporal_cols) if temporal_cols else 'none',
            numeric_cols=', '.join(numeric_cols) if numeric_cols else 'none',
            categorical_cols=', '.join(categorical_cols) if categorical_cols else 'none',
            requirements_summary=self._format_requirements_summary(requirements),
            validation_errors=validation_errors or "(none)",
            param_constraints=param_constraints,
        )

    def _build_embedded_param_fill_prompt(
        self,
        plan: Dict[str, Any],
        schema: SchemaProfile,
        requirements: Optional[Any],
        validation_errors: Optional[str],
    ) -> str:
        step_summary = "\n".join(
            [
                f"  - {s.get('step_id')}: {s.get('tool')} ({s.get('description', '')})"
                for s in plan.get("steps", [])
            ]
        )
        temporal_cols = [c.name for c in schema.columns if 'temporal' in c.roles]
        numeric_cols = [c.name for c in schema.columns if c.dtype in ['int64', 'float64', 'float', 'int']]
        categorical_cols = [c.name for c in schema.columns if c.dtype in ['object', 'string', 'category']]
        column_details = '\n'.join(
            [f"  - {c.name} ({c.dtype}, cardinality: {c.cardinality})" for c in schema.columns]
        )

        param_constraints = self._format_param_constraints(plan)

        return f"""Fill tool parameters for each step. Use only valid column names.

STEPS:
{step_summary}

DATASET:
- Rows: {schema.row_count}, Columns: {len(schema.columns)}

AVAILABLE COLUMNS (use EXACT names):
{column_details}
- Numeric columns: {', '.join(numeric_cols) if numeric_cols else 'none'}
- Categorical columns: {', '.join(categorical_cols) if categorical_cols else 'none'}
- Temporal columns: {', '.join(temporal_cols) if temporal_cols else 'none'}

REQUIREMENTS:
{self._format_requirements_summary(requirements)}

VALIDATION ERRORS (if any):
{validation_errors or "(none)"}

FIX VALIDATION ERRORS (highest priority):
If validation errors are present, you MUST correct them. Do not repeat the same invalid
values. Use the errors as ground truth and adjust only the offending params.
Common fixes:
- agg_map must be a dict mapping real column names to agg functions (e.g., {"revenue": "sum"}).
- Replace invalid column names with exact names from AVAILABLE COLUMNS.

RULES:
1. Only use column names from the list above.
2. Only fill parameters required by each tool.
3. Prefer requirements-aligned columns (metrics/group_by/time).
4. Do not change tools; only fill params for the existing step_id.
5. Do not set output_path; it is auto-filled during tool call generation.
6. For agg_map, keys must be real column names from the schema.

ALLOWED PARAMS BY STEP:
{param_constraints}

RESPONSE FORMAT (JSON only):
{{"steps": [{{"step_id": "<id>", "params": {{"df": "$dataframe"}}}}], "rationale": "<why>"}}
"""

    def _format_param_constraints(self, plan: Dict[str, Any]) -> str:
        ensure_default_tools_registered()
        lines = []
        for step in plan.get("steps", []):
            tool_name = step.get("tool")
            step_id = step.get("step_id")
            schema = TOOL_REGISTRY.get_schema(tool_name)
            if not schema:
                continue
            params = ", ".join(
                p.name for p in schema.parameters if p.name != "output_path"
            ) or "none"
            lines.append(f"- {step_id} ({tool_name}): {params}")
        return "\n".join(lines) if lines else "- (none)"

    def build_requirement_extraction_prompt(
        self,
        question: str,
        schema: SchemaProfile,
        validation_errors: Optional[str] = None,
    ) -> str:
        """
        Build prompt for requirement extraction.

        Args:
            question: User's question
            schema: Dataset schema profile

        Returns:
            Formatted prompt string
        """
        # Try loading from template file first
        template = self._load_template("requirements")
        if template:
            return self._format_requirement_template(
                template, question, schema, validation_errors
            )
        
        # Fall back to embedded prompt
        return self._build_embedded_requirement_prompt(
            question, schema, validation_errors
        )

    def _format_requirement_template(
        self,
        template: str,
        question: str,
        schema: SchemaProfile,
        validation_errors: Optional[str] = None,
    ) -> str:
        """Format requirement extraction template with variables."""
        column_info = ', '.join(f"{c.name}({c.dtype})" for c in schema.columns[:10])
        if len(schema.columns) > 10:
            column_info += '...'
        
        temporal_cols = [c.name for c in schema.columns if 'temporal' in c.roles]
        numeric_cols = [c.name for c in schema.columns if c.dtype in ['int64', 'float64', 'float', 'int']]
        categorical_cols = [c.name for c in schema.columns if c.dtype in ['object', 'string', 'category']]
        
        return template.format(
            question=question,
            row_count=schema.row_count,
            column_count=len(schema.columns),
            column_info=column_info,
            temporal_cols=', '.join(temporal_cols) if temporal_cols else 'none',
            numeric_cols=', '.join(numeric_cols) if numeric_cols else 'none',
            categorical_cols=', '.join(categorical_cols) if categorical_cols else 'none',
            validation_errors=validation_errors or "(none)",
        )

    def _build_embedded_requirement_prompt(
        self,
        question: str,
        schema: SchemaProfile,
        validation_errors: Optional[str] = None,
    ) -> str:
        """Build requirement extraction prompt using embedded template."""
        column_info = ', '.join(f"{c.name}({c.dtype})" for c in schema.columns[:10])
        if len(schema.columns) > 10:
            column_info += '...'
        
        temporal_cols = [c.name for c in schema.columns if 'temporal' in c.roles]
        numeric_cols = [c.name for c in schema.columns if c.dtype in ['int64', 'float64', 'float', 'int']]
        categorical_cols = [c.name for c in schema.columns if c.dtype in ['object', 'string', 'category']]
        
        return f"""Extract structured requirements from the user's question. Use a closed set of analysis types.

USER QUESTION: "{question}"

DATASET SCHEMA:
- Rows: {schema.row_count}, Columns: {len(schema.columns)}
- Column names: {column_info}
- Temporal columns: {', '.join(temporal_cols) if temporal_cols else 'none'}
- Numeric columns: {', '.join(numeric_cols) if numeric_cols else 'none'}
- Categorical columns: {', '.join(categorical_cols) if categorical_cols else 'none'}

IMPORTANT: Only choose column names from the lists above. Do not invent or rename columns.

VALIDATION ERRORS (if any):
{validation_errors or "(none)"}

ALLOWED ANALYSIS TYPES (select all that apply):
{', '.join(f'"{t}"' for t in ALLOWED_ANALYSIS_TYPES)}

EXTRACTION RULES:
1. Metrics: List numeric columns to analyze
2. Group By: List categorical columns for grouping
3. Time: Specify time column and grain (daily, weekly, monthly, yearly, or "unknown")
4. Analysis: Select ONLY the types from allowed list that match the question
5. Outputs: Specify "chart", "table", or both
6. Constraints: List any filters or special conditions

EXAMPLES:
Q: "Show me revenue totals by region"
{{"metrics": ["revenue"], "group_by": ["region"], "time": {{"column": "", "grain": "unknown"}}, "analysis": ["total", "compare"], "outputs": ["chart", "table"], "constraints": []}}

Q: "Get revenue by region and product over time"
{{"metrics": ["revenue"], "group_by": ["region", "product_type"], "time": {{"column": "date", "grain": "unknown"}}, "analysis": ["total", "compare", "trend"], "outputs": ["chart", "table"], "constraints": []}}

Your response (JSON only, no other text):
{{"metrics": [...], "group_by": [...], "time": {{"column": "...", "grain": "..."}}, "analysis": [...], "outputs": [...], "constraints": []}}

Response:"""
