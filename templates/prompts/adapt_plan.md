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

## Available Tools (use EXACT names)

- **aggregate**: Group data and compute aggregations (use for "compare by", "group by")
- **segment_metric**: Segment metric by category
- **detect_anomalies**: Detect outliers or unusual values
- **compute_summary_stats**: Basic statistics (mean, std, etc.)
- **compute_correlations**: Correlation matrix
- **compute_distributions**: Distribution analysis
- **plot_line**, **plot_bar**, **plot_scatter**, **plot_histogram**, **plot_heatmap**, **plot_boxplot**

## Your Task

1. Check if the user question mentions specific requirements not in the template
2. Look for keywords like "anomaly", "outlier", "unusual", "compare", "segment", "group by"
3. Suggest adding, removing, or modifying steps to better match the question

## Examples

- User asks "find outliers" → add step with tool="detect_anomalies"
- User asks "compare revenue by region" → add step with tool="aggregate", params={{"group_by": ["region"], "agg_func": "sum"}}
- User asks "compare by region and product" → add step with tool="aggregate", params={{"group_by": ["region", "product_category"], "agg_func": "sum"}}
- Template fits perfectly → return empty changes array

## Response Format

RESPONSE FORMAT (JSON only, no preamble):
{{"changes": [{{"action": "add|remove|modify", "step_id": "<id>", "tool": "<exact_tool_name>", "description": "<reason>", "params": {{"df": "$dataframe"}}}}], "rationale": "<overall explanation>"}}

## Important Notes

- tool MUST be an exact tool name from the list above (e.g., "aggregate", NOT "aggregate by region")
- For grouping/comparing, use tool="aggregate" with group_by parameter
- step_id must be unique (e.g., "compare_by_region_product")

Your JSON response:
