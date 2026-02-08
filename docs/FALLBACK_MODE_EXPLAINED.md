# Fallback Mode (Deterministic Mode) - How It Works

## ❌ Real LLM Status: NOT TESTED

**Correct** - I have NOT tested with an actual LLM model. The system runs entirely in **fallback/deterministic mode** right now.

## 🔍 How Fallback Mode Works

The fallback mode uses **simple keyword matching** instead of LLM inference. Here's exactly what happens:

### 1. Intent Classification (Without LLM)

**Location:** `src/autoviz_agent/llm/client.py:91-108`

**Method:** Simple keyword matching on the user's question

```python
def _fallback_generate(self, prompt: str) -> str:
    """Keyword-based fallback when LLM unavailable."""
    
    # Extract user question from prompt
    if "user question:" in prompt_lower:
        question_part = prompt_lower.split("user question:")[1].split("\n")[0]
    
    # Keyword-based classification
    if any(word in question_part for word in ["time", "trend", "temporal", "series", "over time"]):
        return '{"primary": "time_series_investigation", "confidence": 0.7, ...}'
    
    elif any(word in question_part for word in ["anomaly", "outlier", "unusual"]):
        return '{"primary": "anomaly_detection", "confidence": 0.7, ...}'
    
    elif any(word in question_part for word in ["segment", "group", "compare"]):
        return '{"primary": "comparative_analysis", "confidence": 0.7, ...}'
    
    else:
        return '{"primary": "general_eda", "confidence": 0.6, ...}'
```

**Example Results:**
- "Show me trends over time" → `time_series_investigation` (confidence: 0.70)
- "Find anomalies" → `anomaly_detection` (confidence: 0.70)
- "Analyze the data" → `general_eda` (confidence: 0.60)

### 2. Plan Adaptation (Without LLM)

**Location:** `src/autoviz_agent/llm/client.py:111-113`

**Method:** Returns empty changes list (no adaptation)

```python
# Plan adaptation fallback - return minimal changes
if "adapt" in prompt_lower or "modify" in prompt_lower:
    return '{"changes": [], "rationale": "No adaptation needed - using template as-is (fallback mode)"}'
```

**Result:** Templates are used **exactly as written** with zero modifications.

### 3. What Still Works (Without LLM)

✅ **Intent classification** - Keyword-based (limited but functional)
✅ **Template selection** - Retrieval/scoring based on intent match
✅ **Tool execution** - Fully functional (all 15 tools work)
✅ **Report generation** - Complete with provenance
✅ **Artifact tracking** - All 5 observability artifacts generated

### 4. What Doesn't Work (Without LLM)

❌ **Smart intent classification** - Can't understand complex/nuanced questions
❌ **Plan adaptation** - Templates used rigidly without schema-aware modifications
❌ **Context-aware reasoning** - No understanding of data semantics

## 📊 Current Test Results

All test runs are using **keyword-based fallback**, not actual LLM inference:

| Question | Fallback Intent | Template Selected | Adaptation |
|----------|----------------|-------------------|------------|
| "Analyze the data" | general_eda (0.60) | simple_eda_01 | None (0 changes) |
| "Show me summary statistics" | general_eda (0.60) | simple_eda_01 | None (0 changes) |
| "Find anomalies in revenue" | general_eda (0.60) | general_eda_01 | None (0 changes) |
| "Show me trends over time" | time_series_investigation (0.70) | time_series_01 | None (0 changes) |

## 🎯 Why Fallback Mode Exists

The design philosophy is **graceful degradation**:

1. **Development/Testing** - Can work on the system without needing a 4GB model downloaded
2. **Reliability** - System doesn't crash if model is missing or corrupted
3. **Demonstration** - Shows the architecture even without LLM inference
4. **Bounded Scope** - LLM is only used for 2 tasks (intent + adaptation), rest is deterministic

## 🚀 To Enable Real LLM Mode

**Required Steps:**

1. **Download Model** (4GB):
   ```bash
   # Download Phi-3.5-mini-instruct Q4_K_M quantized
   mkdir models
   # Download from HuggingFace: microsoft/Phi-3.5-mini-instruct-gguf
   # Place in: models/Phi-3.5-mini-instruct-Q4_K_M.gguf
   ```

2. **Install llama-cpp-python**:
   ```bash
   pip install llama-cpp-python>=0.2.0
   ```

3. **Run Again**:
   ```bash
   python -m autoviz_agent.cli.main run examples/december_revenue/december_revenue.csv "Analyze revenue trends"
   ```

**Expected Changes with Real LLM:**

- **Intent classification**: Semantic understanding instead of keyword matching
- **Plan adaptation**: Schema-aware modifications (e.g., "Dataset has temporal column → add time_series_features step")
- **Reasoning**: `plan_diff.md` will show actual adaptations with rationale
- **Confidence scores**: More accurate based on semantic similarity

## 📝 Current Architecture Value

Even in fallback mode, the system demonstrates:

✅ **Graph orchestration** - 11-node LangGraph pipeline works correctly
✅ **Template system** - Retrieval, scoring, and selection functional
✅ **Tool execution** - 15 tools with context resolution
✅ **Observability** - Complete artifact tracking and provenance
✅ **Error handling** - Graceful degradation when LLM unavailable

**Bottom Line:** The system is architecturally complete and functional, just operating with simplified decision-making (keywords instead of semantic understanding).

## 🔬 What Real Testing Would Show

With a real LLM, you'd expect to see:

```markdown
# plan_diff.md (with real LLM)

## Changes

### Change 1: Add temporal feature extraction
- **Action:** Insert after step "parse_datetime"
- **Step ID:** time_features
- **Tool:** compute_time_series_features
- **Reason:** Dataset has date column with daily granularity. Extract day-of-week 
  and month features to identify temporal patterns in revenue.

### Change 2: Modify summary stats to focus on revenue
- **Action:** Modify step "summary_stats"
- **Parameter Change:** columns: ["revenue"]
- **Reason:** User question specifically asks about "revenue" - focus analysis 
  on this metric rather than all numeric columns.

## Rationale
User is asking about revenue trends over time. The time_series_01 template is 
appropriate, but should be enhanced with temporal feature extraction and focused 
on the revenue column specifically to directly address the question.
```

**Current fallback mode produces:**
```markdown
# plan_diff.md (fallback mode)

## Changes

No changes between template and adapted plan.
```

## 🎓 Summary

**Question:** "How does fallback mode perform LLM tasks?"  
**Answer:** It doesn't. It uses **simple keyword heuristics** as a placeholder:

- Intent classification → Keyword matching
- Plan adaptation → Zero modifications (use template as-is)

This is **by design** for graceful degradation, but is **not a substitute** for real LLM inference.

**Status:** System architecture is complete and proven. LLM integration point exists but is using simplified deterministic logic until a model is loaded.
