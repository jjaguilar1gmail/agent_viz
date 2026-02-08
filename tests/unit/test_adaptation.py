"""Test plan adaptation with real LLM"""
from gpt4all import GPT4All
import json

# Load model
model = GPT4All(
    model_name="Phi-3.5-mini-instruct-Q4_K_M.gguf",
    model_path="models",
    allow_download=False
)

# Sample adaptation prompt
prompt = """You are an expert data analyst. Adapt this analysis plan based on the user question and dataset schema.

**User Question**: Show me the revenue trends and identify any outliers

**Dataset Schema**:
- Columns: date (datetime64[ns]), revenue (float64), region (object), product (object)
- Rows: 20
- Shape: wide

**Current Plan**:
{
  "metadata": {
    "template_id": "time_series_01",
    "intent": "TIME_SERIES_INVESTIGATION"
  },
  "steps": [
    {
      "tool": "parse_datetime",
      "params": {
        "column": "date"
      }
    },
    {
      "tool": "compute_summary_stats",
      "params": {
        "columns": ["revenue"]
      }
    }
  ]
}

**Instructions**: Return ONLY a JSON object with:
- "modifications": array of changes made
- "reasoning": string explaining why
- "adapted_plan": the modified plan JSON

Respond with JSON only, no other text."""

print("Sending prompt to LLM...")
response = model.generate(
    prompt=prompt,
    max_tokens=512,
    temp=0.1,
    top_p=0.9
)

print(f"\nLLM RESPONSE ({len(response)} chars):")
print(repr(response[:500]))
print("\n" + "="*80)
print(response)
