# Plan Adaptation Prompt

Review this analysis plan template and suggest modifications based on the user's specific question.

## User Context

**USER QUESTION:** "{question}"  
**INTENT:** {intent_label}

## Template Overview

**TEMPLATE:** {template_id} ({step_count} steps)

Current steps:
{step_summary}

## Dataset Context

- Rows: {row_count}, Columns: {column_count}
- Temporal columns: {temporal_cols}
- Shape: {data_shape}

## Available Columns (use EXACT names)

{column_details}
- Numeric columns: {numeric_cols}
- Categorical columns: {categorical_cols}
- Temporal columns: {temporal_cols}

## Requirements (must be satisfied)

{requirements_summary}

Use the requirements above to guide changes; do not add steps that do not satisfy them.

## Validation Errors (if any)

{validation_errors}

## Available Tools (use EXACT names)

{tool_catalog}

## Your Task

1. Use the Requirements section as the contract. Every requirement must be satisfied by at least one step.
2. If `group_by` is non-empty, add or modify an `aggregate` step with the required group_by columns.
3. If `time.column` is set, ensure a time-aware plot uses that column (and aggregated data if needed).
4. Remove steps that do not satisfy any requirement.
5. If Validation Errors are present, fix them explicitly.
6. Only use column names from the "Available Columns" list. Do not invent new names.
7. Prefer tools whose capabilities directly match the requirements (e.g., use compare tools for compare needs).
8. Choose the minimal set of steps that fully covers the requirements.

## Examples

- Requirements include group_by=["region"] -> add `aggregate` with group_by=["region"], agg_func="sum"
- Requirements include analysis=["trend"] and time.column="date" -> ensure `plot_line` uses x="date"
- Requirements do NOT include "anomaly" -> remove or do not add `detect_anomalies`

## Response Format

RESPONSE FORMAT (JSON only, no preamble):
{{"changes": [{{"action": "add|remove|modify", "step_id": "<id>", "tool": "<exact_tool_name>", "description": "<reason>", "params": {{"df": "$dataframe"}}, "satisfies": ["analysis.total"]}}], "rationale": "<overall explanation>"}}

## Important Notes

- tool MUST be an exact tool name from the list above (e.g., "aggregate", NOT "aggregate by region")
- For grouping/comparing, use tool="aggregate" with group_by parameter
- step_id must be unique (e.g., "compare_by_region_product")
- Include a non-empty "satisfies" list for each added or modified step.

Your JSON response:
