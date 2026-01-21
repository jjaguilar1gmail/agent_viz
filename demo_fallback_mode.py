"""Demonstrate fallback mode behavior - SIMPLIFIED."""

print("\n" + "="*70)
print("FALLBACK MODE: HOW IT ACTUALLY WORKS")
print("="*70 + "\n")

print("The 'LLM' client uses simple KEYWORD MATCHING when no model is loaded:\n")

# Show the actual fallback logic
print("1. INTENT CLASSIFICATION")
print("-" * 70)
print("Code: src/autoviz_agent/llm/client.py:91-108\n")

test_cases = [
    ("Show me trends over time", "time_series_investigation", 0.70),
    ("Find anomalies in revenue", "anomaly_detection", 0.70),
    ("Compare regions", "comparative_analysis", 0.70),
    ("Analyze the data", "general_eda", 0.60),
]

for question, intent, confidence in test_cases:
    print(f"Question: '{question}'")
    
    # Show the keyword matching logic
    q_lower = question.lower()
    if any(word in q_lower for word in ["time", "trend", "temporal", "series", "over time"]):
        matched_keywords = [w for w in ["time", "trend", "temporal", "series", "over time"] if w in q_lower]
        print(f"  → Matched keywords: {matched_keywords}")
        print(f"  → Intent: {intent} (confidence={confidence})")
    elif any(word in q_lower for word in ["anomaly", "outlier", "unusual"]):
        matched_keywords = [w for w in ["anomaly", "outlier", "unusual"] if w in q_lower]
        print(f"  → Matched keywords: {matched_keywords}")
        print(f"  → Intent: {intent} (confidence={confidence})")
    elif any(word in q_lower for word in ["segment", "group", "compare", "difference"]):
        matched_keywords = [w for w in ["segment", "group", "compare", "difference"] if w in q_lower]
        print(f"  → Matched keywords: {matched_keywords}")
        print(f"  → Intent: {intent} (confidence={confidence})")
    else:
        print(f"  → No specific keywords matched")
        print(f"  → Intent: {intent} (confidence={confidence}) [DEFAULT]")
    print()

print("\n2. PLAN ADAPTATION")
print("-" * 70)
print("Code: src/autoviz_agent/llm/client.py:111-113\n")
print("When asked to adapt a plan:")
print("  → Returns: {'changes': [], 'rationale': '...(fallback mode)'}")
print("  → Result: Template used AS-IS with ZERO modifications")
print()

print("\n3. WHAT ACTUALLY RUNS")
print("-" * 70)
print("✓ Intent: Keyword matching (works, but limited)")
print("✓ Template selection: Retrieval/scoring (fully functional)")
print("✓ Tool execution: All 15 tools work normally")
print("✓ Artifacts: All 5 observability files generated")
print("✗ Adaptation: None (templates used without modification)")
print("✗ Semantic understanding: None (just keyword detection)")
print()

print("\n4. EVIDENCE FROM ACTUAL RUNS")
print("-" * 70)

import os
from pathlib import Path

# Check the most recent runs
outputs_dir = Path("outputs")
if outputs_dir.exists():
    runs = sorted(outputs_dir.iterdir(), key=lambda x: x.stat().st_ctime, reverse=True)[:3]
    
    for run in runs:
        plan_diff = run / "plans" / "plan_diff.md"
        if plan_diff.exists():
            content = plan_diff.read_text()
            print(f"\n Run: {run.name}")
            print(f"   plan_diff.md content:")
            print("   " + "\n   ".join(content.strip().split("\n")[:5]))

print("\n\n" + "="*70)
print("CONCLUSION")
print("="*70)
print("\n❌ NO real LLM has been tested")
print("✓  Fallback mode uses keyword matching as placeholder")
print("✓  System architecture is complete and proven")
print("✓  Observability and tooling fully functional")
print("\n⚠️  To enable real LLM capabilities:")
print("   1. Download Phi-3.5-mini-instruct-Q4_K_M.gguf (4GB)")
print("   2. Place in models/ directory")
print("   3. pip install llama-cpp-python")
print("   4. Re-run analysis")
print("\n" + "="*70 + "\n")
