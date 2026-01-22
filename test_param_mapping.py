"""Quick test of parameter mapping logic."""

# Simulate the parameter fixes
params_cases = [
    {"tool": "plot_histogram", "params": {"df": "$dataframe", "max_columns": 6}},
    {"tool": "plot_heatmap", "params": {"df": "$dataframe", "annotation": True}},
    {"tool": "compute_distributions", "params": {"df": "$dataframe"}},
]

numeric_cols = ["revenue"]

for case in params_cases:
    tool_name = case["tool"]
    params = case["params"].copy()
    
    print(f"\n{tool_name}:")
    print(f"  Before: {params}")
    
    # Apply fixes
    if tool_name == "plot_histogram":
        if "column" not in params and numeric_cols:
            params["column"] = numeric_cols[0]
        params.pop("max_columns", None)
    
    elif tool_name == "plot_heatmap":
        if "df" in params:
            params["data"] = params.pop("df")
        if "annotation" in params:
            params["annot"] = params.pop("annotation")
    
    elif tool_name == "compute_distributions":
        if "column" not in params and numeric_cols:
            params["column"] = numeric_cols[0]
    
    print(f"  After:  {params}")
    
    # Validate
    if tool_name == "plot_histogram":
        assert "max_columns" not in params, "❌ max_columns should be removed"
        assert "column" in params, "❌ column should be added"
        print("  ✅ All checks passed")
    
    elif tool_name == "plot_heatmap":
        assert "df" not in params, "❌ df should be renamed to data"
        assert "data" in params, "❌ data parameter should exist"
        assert "annotation" not in params, "❌ annotation should be renamed to annot"
        assert "annot" in params, "❌ annot parameter should exist"
        print("  ✅ All checks passed")
    
    elif tool_name == "compute_distributions":
        assert "column" in params, "❌ column should be added"
        print("  ✅ All checks passed")

print("\n✅ All parameter mappings are correct!")
