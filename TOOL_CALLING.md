# Tool Calling Architecture

## Overview

The AutoViz Agent uses a **registry-driven, decorator-based** tool calling system that automatically discovers and registers tools, validates parameters, and resolves defaults based on dataset schema. This architecture decouples tool definitions from the LLM layer and makes the system backend-agnostic.

## Key Components

### 1. Tool Decorator (`@tool`)

Tools are registered automatically using the `@tool` decorator, which extracts metadata from function signatures and docstrings:

```python
from autoviz_agent.registry.tools import tool

@tool(description="Load dataset from CSV file")
def load_dataset(path: str, encoding: str = "utf-8") -> pd.DataFrame:
    """
    Load dataset from file.
    
    Args:
        path: Path to dataset file
        encoding: File encoding
    
    Returns:
        Loaded DataFrame
    """
    # Implementation
    ...
```

The decorator automatically:
- Extracts parameter names, types, defaults, and descriptions
- Determines required vs optional parameters
- Registers the tool with the global `TOOL_REGISTRY`
- Infers return type from type hints

### 2. Tool Registry (`TOOL_REGISTRY`)

The global registry serves as the single source of truth for all tools:

```python
from autoviz_agent.registry.tools import TOOL_REGISTRY

# List all registered tools
tools = TOOL_REGISTRY.list_tools()

# Get a specific tool
tool_func = TOOL_REGISTRY.get_tool("load_dataset")

# Get tool schema
schema = TOOL_REGISTRY.get_schema("load_dataset")

# Export schemas for LLM
json_schema = TOOL_REGISTRY.export_schema()
```

**Key Methods:**
- `register(schema, func)`: Register a tool (called by decorator)
- `get_tool(name)`: Retrieve tool function by name
- `get_schema(name)`: Get tool schema definition
- `list_tools()`: List all registered tool names
- `export_schema()`: Export JSON schema for LLM consumption
- `clear()`: Clear registry (useful for testing)

### 3. Parameter Resolution (`ParamResolver`)

The `ParamResolver` fills in missing parameters based on dataset schema:

```python
from autoviz_agent.runtime.param_resolver import ParamResolver

resolver = ParamResolver(schema, artifact_manager)
resolved_params = resolver.resolve("plot_line", {"y": "revenue"}, sequence=1)
# Adds: x=<temporal_column>, output_path=<generated_path>
```

**Resolution Strategies:**
- **Visualization tools**: Auto-generate output paths, select appropriate columns
- **Analysis tools**: Select default columns based on types (numeric, categorical, temporal)
- **Preparation tools**: Detect date columns, handle missing patterns
- **Validation**: Remove invalid parameters before execution

### 4. Validation (`validate_tool_call`)

Tool calls are validated against registered schemas before execution:

```python
from autoviz_agent.registry.validation import validate_tool_call

result = validate_tool_call({
    "tool": "plot_line",
    "sequence": 1,
    "args": {"x": "date", "y": "value", "output_path": "chart.png"}
})

if not result.is_valid:
    print(f"Errors: {result.errors}")
    print(f"Warnings: {result.warnings}")
```

**Validation Checks:**
- Tool exists in registry
- All required parameters provided
- No unknown parameters
- Parameter types match schema (when possible)

### 5. Schema Export for LLMs

The registry can export tool schemas in JSON format for LLM consumption:

```json
{
  "tools": [
    {
      "name": "load_dataset",
      "description": "Load dataset from CSV file",
      "parameters": {
        "type": "object",
        "properties": {
          "path": {
            "type": "string",
            "description": "Path to dataset file"
          },
          "encoding": {
            "type": "string",
            "description": "File encoding",
            "default": "utf-8"
          }
        },
        "required": ["path"]
      }
    }
  ]
}
```

This format can be passed to:
- vLLM's native tool calling API
- OpenAI-compatible APIs
- Custom LLM backends with tool support

## Execution Flow

```
1. LLM generates plan with tool calls
   ↓
2. ParamResolver fills missing parameters
   ↓
3. Validator checks tool calls
   ↓
4. ToolExecutor dispatches to registry
   ↓
5. Tool function executes
   ↓
6. Results logged and returned
```

## Adding a New Tool

To add a new tool, simply decorate your function:

```python
# src/autoviz_agent/tools/my_module.py

from autoviz_agent.registry.tools import tool
import pandas as pd

@tool(description="Calculate rolling average")
def rolling_average(
    df: pd.DataFrame,
    column: str,
    window: int = 3
) -> pd.DataFrame:
    """
    Calculate rolling average for a column.
    
    Args:
        df: Input DataFrame
        column: Column to compute rolling average
        window: Window size for rolling calculation
    
    Returns:
        DataFrame with new rolling average column
    """
    result = df.copy()
    result[f"{column}_rolling"] = df[column].rolling(window=window).mean()
    return result
```

**That's it!** The tool is automatically:
- Registered when the module is imported
- Available in the tool registry
- Validated against its schema
- Discoverable by the LLM

## Backend Agnosticism

The architecture supports multiple LLM backends:

### Current: gpt4all
- Uses JSON-in-text parsing
- Fallback mode for keyword-based intent classification
- Works with local GGUF models

### Future: vLLM
- Native tool calling API
- Uses `export_schema()` to pass tool definitions
- Structured output parsing
- See `src/autoviz_agent/llm/vllm_adapter.py` for adapter stub

### Switching Backends

To switch backends, implement a new adapter that:
1. Uses `TOOL_REGISTRY.export_schema()` to get tool definitions
2. Calls the backend with tool schema
3. Parses tool call responses
4. Returns standardized tool call format

The tool execution layer remains unchanged.

## Testing

### Unit Tests
```python
from autoviz_agent.registry.tools import TOOL_REGISTRY, tool

def test_tool_decorator():
    # Clear registry for isolation
    TOOL_REGISTRY.clear()
    
    @tool(description="Test tool")
    def test_func(x: int, y: int = 5) -> int:
        return x + y
    
    # Verify registration
    assert "test_func" in TOOL_REGISTRY.list_tools()
    
    # Verify schema
    schema = TOOL_REGISTRY.get_schema("test_func")
    assert schema.name == "test_func"
    assert len(schema.parameters) == 2
```

### Integration Tests
```python
def test_end_to_end_tool_call():
    # Generate tool call
    tool_call = {
        "tool": "load_dataset",
        "sequence": 1,
        "args": {"path": "data.csv"}
    }
    
    # Validate
    result = validate_tool_call(tool_call)
    assert result.is_valid
    
    # Execute
    tool_func = TOOL_REGISTRY.get_tool("load_dataset")
    df = tool_func(**tool_call["args"])
    assert isinstance(df, pd.DataFrame)
```

## Benefits

1. **No Manual Registration**: Tools register themselves via decorator
2. **Type Safety**: Parameter types captured from function signatures
3. **Validation**: Catch errors before execution
4. **Schema-Driven Defaults**: Automatically fill missing parameters
5. **Backend Agnostic**: Works with any LLM that supports structured output
6. **Testable**: Easy to mock, validate, and test tools in isolation
7. **Self-Documenting**: Docstrings become part of the schema

## Migration from Old System

The old manual registration in `executor.py` looked like:

```python
# OLD - Manual registration
TOOL_REGISTRY.register(
    ToolSchema(name="load_dataset", description="Load dataset", returns="DataFrame"),
    data_io.load_dataset
)
```

New decorator-based registration:

```python
# NEW - Automatic registration
@tool(description="Load dataset from file")
def load_dataset(path: str) -> pd.DataFrame:
    ...
```

**No changes needed in:**
- Tool function implementations
- Execution logic
- Validation rules
- Reporting/logging

**Only changes:**
- Add `@tool` decorator to each tool function
- Remove manual `_register_tools()` method
- Import tool modules to trigger registration
