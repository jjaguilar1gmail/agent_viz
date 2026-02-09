# Requirement Extraction Prompt

Extract structured requirements from the user's question. Use a closed set of analysis types and leave fields empty or "unknown" if not present in the question. DO NOT infer extra analysis beyond what the user explicitly requested.

## User Question

"{question}"

## Dataset Schema

- Rows: {row_count}, Columns: {column_count}
- Column names: {column_info}
- Temporal columns: {temporal_cols}
- Numeric columns: {numeric_cols}
- Categorical columns: {categorical_cols}

## Allowed Analysis Types

Use ONLY these labels (select all that apply from the question):
- "total": Computing sums, counts, or overall aggregates
- "compare": Comparing groups, segments, or categories
- "trend": Analyzing changes over time, patterns, or temporal behavior
- "distribution": Examining spread, frequency, or distribution of values
- "anomaly": Detecting outliers, unusual patterns, or anomalies
- "correlation": Finding relationships or correlations between variables

## Time Grain Rules

ONLY specify a time grain if the user explicitly mentions it (e.g., "daily", "weekly", "monthly", "yearly").
If no grain is mentioned, use "unknown" to defer to execution.

## Extraction Rules

1. **Metrics**: List numeric columns the user wants to analyze (e.g., revenue, sales, count)
2. **Group By**: List categorical columns for grouping or segmentation (e.g., region, product_category)
3. **Time**: Specify time column if temporal analysis is requested, and grain only if explicitly mentioned
4. **Analysis**: Select ONLY the analysis types that match the user's request from the allowed list
5. **Outputs**: Specify "chart", "table", or both based on what the user asks for
6. **Constraints**: List any special conditions, filters, or requirements (e.g., "only Q4 2023", "top 10")

## Examples

### Example 1
Question: "Show me revenue totals by region"
```json
{{
  "metrics": ["revenue"],
  "group_by": ["region"],
  "time": {{"column": "", "grain": "unknown"}},
  "analysis": ["total", "compare"],
  "outputs": ["chart", "table"],
  "constraints": []
}}
```

### Example 2
Question: "Get revenue totals by region and product type over time"
```json
{{
  "metrics": ["revenue"],
  "group_by": ["region", "product_type"],
  "time": {{"column": "date", "grain": "unknown"}},
  "analysis": ["total", "compare", "trend"],
  "outputs": ["chart", "table"],
  "constraints": []
}}
```

### Example 3
Question: "Plot daily sales trends for Q4 2023"
```json
{{
  "metrics": ["sales"],
  "group_by": [],
  "time": {{"column": "date", "grain": "daily"}},
  "analysis": ["trend"],
  "outputs": ["chart"],
  "constraints": ["Q4 2023"]
}}
```

### Example 4
Question: "Find anomalies in transaction amounts by merchant"
```json
{{
  "metrics": ["transaction_amount"],
  "group_by": ["merchant"],
  "time": {{"column": "", "grain": "unknown"}},
  "analysis": ["anomaly", "distribution"],
  "outputs": ["chart", "table"],
  "constraints": []
}}
```

## Response Format

Your response (JSON only, no other text):
{{"metrics": [...], "group_by": [...], "time": {{"column": "...", "grain": "..."}}, "analysis": [...], "outputs": [...], "constraints": [...]}}

Response:
