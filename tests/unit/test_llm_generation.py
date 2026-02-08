"""Test LLM generation to see raw output."""

from gpt4all import GPT4All
from pathlib import Path

model_path = Path("models")
model_name = "Phi-3.5-mini-instruct-Q4_K_M.gguf"

print("Loading model...")
llm = GPT4All(model_name=model_name, model_path=str(model_path), allow_download=False)
print("✓ Model loaded\n")

# Test with a simple intent classification prompt
prompt = """You are an analytical intent classifier. Given a user question and dataset schema, classify the analytical intent.

Available intent labels:
- general_eda: Broad exploratory data analysis
- time_series_investigation: Time-based trends and patterns
- anomaly_detection: Finding unusual patterns or outliers

User Question: Analyze revenue trends over time

Dataset Schema:
- Rows: 20
- Columns: 4
- Shape: wide
- Columns: date, revenue, region, product_category

Respond with JSON only:
{"primary": "<intent_label>", "secondary": ["<label1>"], "confidence": 0.7, "reasoning": "<brief explanation>"}

Response:"""

print("=" * 70)
print("PROMPT:")
print("=" * 70)
print(prompt)
print("\n" + "=" * 70)
print("LLM RESPONSE:")
print("=" * 70)

response = llm.generate(prompt, max_tokens=256, temp=0.1)
print(response)
print("\n" + "=" * 70)
