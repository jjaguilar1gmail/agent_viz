# Intent Classification Prompt

Classify the user's analytical intent based on their question and the dataset structure.

## Available Intents (with templates)

1. **general_eda** - Broad exploration (e.g., "summarize this data", "what's in here?")
2. **time_series_investigation** - Temporal patterns (e.g., "trends over time", "seasonal patterns")
3. **anomaly_detection** - Outliers and unusual values (e.g., "find anomalies", "detect outliers")
4. **comparative_analysis** - Compare groups/categories (e.g., "compare by region", "revenue by product")

## Intent Selection Rules

- "compare", "by", "across", "versus", "difference between" → comparative_analysis
- "time", "trend", "over time", "temporal" → time_series_investigation
- "anomaly", "outlier", "unusual", "abnormal" → anomaly_detection
- General questions → general_eda

## User Question

"{question}"

## Dataset Schema

- Rows: {row_count}, Columns: {column_count}
- Column names: {column_info}
- Temporal columns: {temporal_cols}
- Numeric columns: {numeric_col_count}

## Examples

Question: "Compare revenue by region and product" → {{"primary": "comparative_analysis", "confidence": 0.95, "reasoning": "Keywords 'compare' and 'by' indicate comparison across categories"}}

Question: "Analyze revenue trends over time" → {{"primary": "time_series_investigation", "confidence": 0.95, "reasoning": "Keywords 'trends' and 'over time' indicate temporal analysis"}}

Question: "Find unusual sales patterns" → {{"primary": "anomaly_detection", "confidence": 0.90, "reasoning": "User explicitly asks for 'unusual' patterns"}}

## Response Format

Your response (JSON only, no other text):
{{"primary": "<intent>", "confidence": 0.0-1.0, "reasoning": "<why you chose this intent>"}}

Response:
