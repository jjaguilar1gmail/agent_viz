"""Test the new adaptation prompt."""
from gpt4all import GPT4All

model = GPT4All("Phi-3.5-mini-instruct-Q4_K_M.gguf", model_path="models", allow_download=False)

prompt = """Review this analysis plan template and suggest modifications based on the user's specific question.

USER QUESTION: "Show revenue trends over time and identify any unusual outliers or anomalies"
INTENT: time_series_investigation

TEMPLATE: time_series_01 (5 steps)
Current steps:
  - parse_datetime: parse_datetime - Parse and validate datetime column
  - summary_stats: compute_summary_stats - Compute summary statistics for metrics
  - plot_time_series: plot_line - Plot metrics over time
  - plot_distributions: plot_histogram - Distribution of key metrics
  - detect_anomalies: detect_anomalies - Detect temporal anomalies

DATASET CONTEXT:
- Rows: 20, Columns: 4
- Temporal columns: date
- Shape: wide

YOUR TASK:
1. Check if the user question mentions specific requirements not in the template
2. Look for keywords like "anomaly", "outlier", "unusual", "compare", "segment", "correlate"
3. Suggest adding, removing, or modifying steps to better match the question

EXAMPLES:
- If user asks for "outliers" but template lacks anomaly detection → add detect_anomalies step
- If user asks for "trends" and template has histogram → change to line plot
- If template fits perfectly → return empty changes array

RESPONSE FORMAT (JSON only, no preamble):
{"changes": [{"action": "add|remove|modify", "step_id": "<id>", "reason": "<specific reason>"}], "rationale": "<overall explanation of changes or 'Template fits user question well'>"}"""

print("Sending prompt...")
response = model.generate(prompt, max_tokens=300, temp=0.1, top_p=0.9)

print(f"\nRAW RESPONSE ({len(response)} chars):")
print(response)
print("\n" + "="*80)

# Try to extract JSON
json_start = response.find('{')
if json_start >= 0:
    response = response[json_start:]
    brace_count = 0
    for i, char in enumerate(response):
        if char == '{':
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0:
                extracted = response[:i+1]
                print("\nEXTRACTED JSON:")
                print(extracted)
                
                # Try to parse
                import json
                try:
                    parsed = json.loads(extracted)
                    print("\n✅ VALID JSON")
                    print(json.dumps(parsed, indent=2))
                except Exception as e:
                    print(f"\n❌ PARSE ERROR: {e}")
                break
