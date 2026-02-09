# Plan Parameter Fill Prompt

Fill tool parameters for each step. Use only valid column names and satisfy the requirements.

## Steps

{step_summary}

## Dataset

- Rows: {row_count}, Columns: {column_count}

## Available Columns (use EXACT names)

{column_details}
- Numeric columns: {numeric_cols}
- Categorical columns: {categorical_cols}
- Temporal columns: {temporal_cols}

## Requirements

{requirements_summary}

## Validation Errors (if any)

{validation_errors}

## Fix Validation Errors (highest priority)

If validation errors are provided, you MUST correct them. Do not repeat the same invalid
values. Use the errors as ground truth and adjust only the offending params.

Common fixes:
- If agg_map is invalid, it MUST be a dict mapping real column names to agg functions.
  Example: {{"revenue": "sum"}} (not ["sum", "mean"] and not {{"total": "sum"}}).
- If a column name is invalid, replace it with an EXACT column from the list.

## Rules

1. Only use column names from the list above.
2. Only fill parameters that exist for the tool.
3. Prefer requirements-aligned columns (metrics/group_by/time).
4. Do not change tools; only fill params for the existing step_id.
5. Do not set output_path; it is auto-filled during tool call generation.
6. For agg_map, keys must be real column names from the schema.

## Allowed Params by Step

{param_constraints}

## Response Format

RESPONSE FORMAT (JSON only, no preamble):
{{"steps": [{{"step_id": "<id>", "params": {{"df": "$dataframe"}}}}], "rationale": "<overall explanation>"}}
