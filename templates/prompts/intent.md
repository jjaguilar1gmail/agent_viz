# Intent Classification Prompt

Classify the user's analytical intent based on their question and the dataset structure.

## Available Intents (with templates)

{intent_catalog}

## Intent Selection Rules

{intent_rules}

## User Question

"{question}"

## Dataset Schema

- Rows: {row_count}, Columns: {column_count}
- Column names: {column_info}
- Temporal columns: {temporal_cols}
- Numeric columns: {numeric_col_count}

## Examples

{intent_examples}

## Response Format

Your response (JSON only, no other text):
{{"primary": "<intent>", "confidence": 0.0-1.0, "reasoning": "<why you chose this intent>"}}

Response:
