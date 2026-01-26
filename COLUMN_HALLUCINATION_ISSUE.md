# Column Name Hallucination Issue

## Problem Summary

During the plan adaptation step, the LLM (Qwen 2.5 1.5B) hallucinates column names based on user question terminology instead of using actual column names from the dataset schema, even when provided with complete schema information in the prompt.

## Issue Status

⚠️ **ACTIVE** - Steps with hallucinated column names fail validation and get dropped from execution

## Concrete Example

### Test Case
- **Dataset**: `examples/december_revenue/december_revenue.csv`
- **Actual Columns**: `date`, `revenue`, `region`, `product_category`
- **User Query**: "get revenue totals by region and product type"

### Expected Behavior
LLM should use actual column names from schema:
```json
{
  "tool": "plot_line",
  "params": {
    "x": "date",
    "y": "revenue",
    "hue": "product_category"
  }
}
```

### Actual Behavior
LLM invents column names based on user's wording:
```json
{
  "tool": "plot_line",
  "params": {
    "x": "date",
    "y": "revenue_total",        // ❌ Column doesn't exist (from "revenue totals")
    "groups": ["region", "product_type"]  // ❌ "product_type" doesn't exist, should be "product_category"
                                          // ❌ "groups" parameter doesn't exist, should be "hue"
  }
}
```

### Validation Result
```
WARNING: Tool call validation failed for plot_line:
  - Missing required parameter: df
  - Unknown parameter: groups
ERROR: Cannot repair tool call for plot_line, dropping it
WARNING: Dropped 1 invalid tool calls: ['plot_line']
```

**Outcome**: 6 of 7 planned steps executed (plot_line dropped)

## Root Cause Analysis

### 1. Model Capacity Limitation
- **Model**: Qwen 2.5 1.5B Instruct (1.5B parameters, 4-bit quantized)
- **Constraint**: Small model struggles to follow complex schema constraints
- **Behavior**: Interprets user intent semantically rather than following strict schema rules

### 2. User Terminology vs Schema Mismatch
The LLM maps user language directly to parameter values:
- User says "revenue totals" → LLM uses `revenue_total`
- User says "product type" → LLM uses `product_type`
- Actual column: `product_category` (semantically similar but different name)

### 3. xgrammar2 Limitations
- **What it enforces**: JSON structure and types (string, number, array)
- **What it doesn't enforce**: Semantic correctness of string values
- **Result**: Structurally valid JSON with semantically invalid column names

## What Was Tried

### Attempt 1: Enhanced Prompts ❌ INSUFFICIENT

**Changes Made** (`src/autoviz_agent/llm/prompts.py` lines 180-280):
1. Added complete column schema to adaptation prompt:
   ```python
   column_details = '\n'.join([
       f"  - {c.name} ({c.dtype}, cardinality: {c.cardinality})"
       for c in schema.columns
   ])
   ```
2. Added "AVAILABLE COLUMNS" section listing all columns with types
3. Added "CRITICAL RULES" section:
   - "ONLY use column names that appear in AVAILABLE COLUMNS"
   - "Never invent or guess column names"
   - "Column names in params MUST exactly match"

**Result**: LLM still hallucinates column names despite explicit schema and rules

**Conclusion**: Prompt engineering alone insufficient for 1.5B parameter model

## Impact

### Functional Impact
- ✅ Intent classification: Works correctly
- ✅ Plan structure: Valid JSON, correct steps
- ❌ Parameter values: Column names incorrect
- ⚠️ Execution: Steps with invalid columns dropped
- ⚠️ User experience: Missing visualizations/analyses

### Production Readiness
- **Development**: Acceptable (most steps work)
- **Production**: Not recommended (unpredictable failures)

## Potential Solutions

### Option 1: Parameter Repair (Recommended for Current Setup)
**Approach**: Implement fuzzy matching and column name mapping
- Map user terminology to actual columns
- Use similarity metrics (Levenshtein, semantic)
- Apply repairs before validation

**Implementation Location**: `VLLMClient._apply_adaptations()` or new `ColumnNameRepair` class

**Pros**:
- Works with existing Qwen 1.5B model
- No hardware changes needed
- Can handle common synonyms

**Cons**:
- Adds complexity
- May not catch all cases
- Requires maintenance of synonym mappings

### Option 2: Stricter JSON Schema (Worth Testing)
**Approach**: Add column name enums to xgrammar2 schema
```json
{
  "properties": {
    "y": {
      "enum": ["date", "revenue", "region", "product_category"]
    }
  }
}
```

**Pros**:
- Forces LLM to choose from valid columns
- No post-processing needed
- Leverages xgrammar2 constraints

**Cons**:
- Schema becomes dataset-specific
- Need to dynamically generate enums
- May not work for all parameter types

### Option 3: Larger Model (Hardware Dependent)
**Approach**: Upgrade to 3B+ parameter model
- Qwen 2.5 3B 4-bit (~2.3GB) - might fit with optimizations
- Phi-3.5-mini (3.8B) - requires 6GB+ VRAM

**Pros**:
- Better instruction following
- More reliable schema adherence
- Overall quality improvement

**Cons**:
- Current hardware: 4GB VRAM insufficient
- Requires GPU upgrade
- Higher latency

### Option 4: Hybrid Approach (Best Long-term)
**Approach**: Combine multiple strategies
1. Stricter JSON schema with column enums
2. Larger model if/when VRAM permits
3. Fallback parameter repair for edge cases

## Workarounds (Temporary)

### For Users
1. Use exact column names in questions:
   - ✅ "Show revenue by product_category"
   - ❌ "Show revenue by product type"

2. Accept that some suggested steps may be dropped

### For Developers
1. Monitor `execution_log.json` for dropped steps
2. Check validation warnings in logs
3. Consider manual review of adapted plans

## Related Files

- **Issue Manifestation**: `outputs/*/plans/plan_adapted.json` (LLM-generated plans)
- **Validation Logs**: `outputs/*/logs/execution_log.json` (dropped steps)
- **Prompt Code**: `src/autoviz_agent/llm/prompts.py` (lines 180-280)
- **vLLM Client**: `src/autoviz_agent/llm/vllm_client.py` (`_apply_adaptations()`)
- **Tool Definitions**: `src/autoviz_agent/tools/visualization.py` (parameter signatures)

## Next Steps

1. **Immediate**: Implement Option 1 (parameter repair with fuzzy matching)
2. **Short-term**: Test Option 2 (enum constraints in JSON schema)
3. **Long-term**: Evaluate Option 3 (larger model when hardware permits)

## References

- Main implementation doc: [VLLM_IMPLEMENTATION.md](VLLM_IMPLEMENTATION.md)
- Test dataset: `examples/december_revenue/december_revenue.csv`
- Example output: `outputs/3640bf4d/plans/plan_adapted.json`
