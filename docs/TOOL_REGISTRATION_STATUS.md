# Tool Registration Status & Prevention Strategy

## Current Status

### ✅ All Visualization Tools Registered (6/6)
- plot_line
- plot_bar
- plot_scatter
- plot_histogram
- plot_heatmap
- plot_boxplot

### ❌ Missing Registrations

#### Analysis Tools (2 missing out of 5)
- ✅ detect_anomalies
- ✅ segment_metric
- ✅ compute_distributions
- ❌ **compare_groups** - Compare metrics across groups
- ❌ **compute_time_series_features** - Extract time series features

#### Metrics Tools (3 missing out of 5)
- ✅ compute_summary_stats
- ✅ compute_correlations
- ❌ **compute_value_counts** - Count unique values
- ❌ **compute_percentiles** - Compute percentiles
- ❌ **aggregate** - Aggregate data by groups

#### Prep Tools (2 missing out of 4)
- ✅ handle_missing
- ✅ parse_datetime
- ❌ **cast_types** - Convert column types
- ❌ **normalize_column_names** - Clean column names

#### Data I/O Tools (1 missing out of 3)
- ✅ load_dataset
- ✅ sample_rows
- ❌ **save_dataframe** - Save DataFrame to file

## How to Fix This Issue

### Option 1: Register Missing Tools Now
Add these registrations to `src/autoviz_agent/runtime/executor.py` in the `_register_tools()` method:

```python
# Analysis tools (add these)
TOOL_REGISTRY.register(
    ToolSchema(name="compare_groups", description="Compare metrics across groups", returns="DataFrame"),
    analysis.compare_groups,
)
TOOL_REGISTRY.register(
    ToolSchema(name="compute_time_series_features", description="Extract time series features", returns="Dict"),
    analysis.compute_time_series_features,
)

# Metrics tools (add these)
TOOL_REGISTRY.register(
    ToolSchema(name="compute_value_counts", description="Count unique values", returns="DataFrame"),
    metrics.compute_value_counts,
)
TOOL_REGISTRY.register(
    ToolSchema(name="compute_percentiles", description="Compute percentiles", returns="Dict"),
    metrics.compute_percentiles,
)
TOOL_REGISTRY.register(
    ToolSchema(name="aggregate", description="Aggregate data by groups", returns="DataFrame"),
    metrics.aggregate,
)

# Prep tools (add these)
TOOL_REGISTRY.register(
    ToolSchema(name="cast_types", description="Convert column types", returns="DataFrame"),
    prep.cast_types,
)
TOOL_REGISTRY.register(
    ToolSchema(name="normalize_column_names", description="Clean column names", returns="DataFrame"),
    prep.normalize_column_names,
)

# Data I/O tools (add this)
TOOL_REGISTRY.register(
    ToolSchema(name="save_dataframe", description="Save DataFrame to file", returns="None"),
    data_io.save_dataframe,
)
```

## How to Prevent This in the Future

### 1. Automated Test (Already Created)
Run `python tests/test_tool_registration.py` to validate all tools are registered.

**Add to CI/CD pipeline:**
```bash
pytest tests/test_tool_registration.py -v
```

### 2. Pre-commit Hook
Create `.git/hooks/pre-commit`:
```bash
#!/bin/sh
python tests/test_tool_registration.py
if [ $? -ne 0 ]; then
    echo "❌ Tool registration validation failed!"
    echo "Run 'python tests/test_tool_registration.py' to see missing tools"
    exit 1
fi
```

### 3. Documentation Convention
When adding new tools:
1. Create function in `tools/<module>.py`
2. **Immediately** add registration in `runtime/executor.py`
3. Run test to verify: `python tests/test_tool_registration.py`

### 4. Code Review Checklist
- [ ] New tool function has docstring with Args/Returns
- [ ] Tool is registered in `executor.py` with correct `ToolSchema`
- [ ] `test_tool_registration.py` passes
- [ ] Tool is added to appropriate template if needed

### 5. Auto-Registration Pattern (Future Enhancement)
Consider using decorators to auto-register tools:

```python
@register_tool("detect_anomalies", "Detect anomalies", returns="DataFrame")
def detect_anomalies(...):
    ...
```

This would eliminate the need for manual registration in `executor.py`.

## Testing the Fix

After adding missing registrations, verify with:
```bash
python tests/test_tool_registration.py
```

Expected output:
```
Running tool registration tests...

✅ test_all_visualization_tools_registered
✅ test_all_analysis_tools_registered
✅ test_all_metrics_tools_registered
✅ test_all_prep_tools_registered
✅ test_all_data_io_tools_registered
✅ test_no_duplicate_registrations
```
