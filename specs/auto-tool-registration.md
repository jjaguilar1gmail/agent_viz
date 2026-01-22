# Future Feature: Automatic Tool Registration

**Status**: Proposed  
**Priority**: Medium  
**Effort**: Small (~2-4 hours)  
**Related**: [Tool Registration Status](../TOOL_REGISTRATION_STATUS.md)

## Problem Statement

Currently, adding a new tool function requires manual synchronization between the tool implementation and the executor registration:

1. Create tool function in `tools/<module>.py`
2. Manually add registration in `runtime/executor.py`
3. Easy to forget step 2, leading to "tool not found" errors

**Recent Example**: `plot_boxplot` existed in `visualization.py` but wasn't registered, causing template execution failures.

## Proposed Solution

Implement a decorator-based auto-registration pattern that eliminates manual registration.

### Core Concept

Use a `@register_tool` decorator on tool functions that automatically registers them when the module is imported.

## Implementation Design

### 1. Create Tool Registration Decorator

**File**: `src/autoviz_agent/registry/decorators.py`

```python
from functools import wraps
from typing import Callable, Optional
from .tools import TOOL_REGISTRY, ToolSchema

def register_tool(
    name: Optional[str] = None,
    description: Optional[str] = None,
    returns: Optional[str] = None,
    category: Optional[str] = None
):
    """
    Decorator to automatically register a tool function.
    
    Args:
        name: Tool name (defaults to function name)
        description: Tool description (extracted from docstring if not provided)
        returns: Return type annotation
        category: Tool category (data_io, prep, metrics, analysis, visualization)
    
    Example:
        @register_tool(description="Detect anomalies", returns="DataFrame", category="analysis")
        def detect_anomalies(df, method="zscore", threshold=3.0):
            ...
    """
    def decorator(func: Callable) -> Callable:
        tool_name = name or func.__name__
        tool_desc = description or _extract_description_from_docstring(func)
        tool_returns = returns or _infer_return_type(func)
        
        # Register the tool immediately
        TOOL_REGISTRY.register(
            ToolSchema(
                name=tool_name,
                description=tool_desc,
                returns=tool_returns,
                category=category
            ),
            func
        )
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        
        return wrapper
    return decorator

def _extract_description_from_docstring(func: Callable) -> str:
    """Extract first line of docstring as description."""
    if func.__doc__:
        return func.__doc__.strip().split('\n')[0]
    return "No description"

def _infer_return_type(func: Callable) -> str:
    """Infer return type from type annotations."""
    import inspect
    sig = inspect.signature(func)
    if sig.return_annotation != inspect.Parameter.empty:
        return sig.return_annotation.__name__
    return "Unknown"
```

### 2. Apply Decorators to Tool Functions

**Example - analysis.py:**
```python
from autoviz_agent.registry.decorators import register_tool

@register_tool(returns="DataFrame", category="analysis")
def detect_anomalies(
    df: pd.DataFrame,
    column: str,
    method: str = "zscore",
    threshold: float = 3.0
) -> pd.DataFrame:
    """Detect anomalies using statistical methods."""
    # ... implementation

@register_tool(returns="DataFrame", category="analysis")
def compare_groups(
    df: pd.DataFrame,
    metric_col: str,
    group_col: str,
    agg_func: str = "mean"
) -> pd.DataFrame:
    """Compare metrics across groups."""
    # ... implementation
```

**Example - visualization.py:**
```python
from autoviz_agent.registry.decorators import register_tool

@register_tool(returns="Path", category="visualization")
def plot_line(
    df: pd.DataFrame,
    x: str,
    y: str,
    output_path: Path,
    title: Optional[str] = None
) -> Path:
    """Create a line plot."""
    # ... implementation

@register_tool(returns="Path", category="visualization")
def plot_boxplot(
    df: pd.DataFrame,
    column: str,
    output_path: Path,
    by: Optional[str] = None,
    title: Optional[str] = None
) -> Path:
    """Create a box plot."""
    # ... implementation
```

### 3. Simplify Executor Registration

**Current** (50+ lines of manual registration):
```python
def _register_tools(self) -> None:
    """Register all available tools."""
    from autoviz_agent.registry.tools import ToolSchema

    TOOL_REGISTRY.register(
        ToolSchema(name="load_dataset", description="Load dataset from file", returns="DataFrame"),
        data_io.load_dataset,
    )
    TOOL_REGISTRY.register(
        ToolSchema(name="sample_rows", description="Sample rows from DataFrame", returns="DataFrame"),
        data_io.sample_rows,
    )
    # ... 20+ more registrations
```

**After** (5 lines - just import modules):
```python
def _register_tools(self) -> None:
    """Register all available tools."""
    # Import modules - decorators auto-register tools
    from autoviz_agent.tools import (
        analysis, data_io, metrics, prep, visualization
    )
    
    logger.info(f"Registered {len(TOOL_REGISTRY.list_tools())} tools")
```

### 4. Enhanced Tool Schema (Optional)

```python
@dataclass
class ToolSchema:
    name: str
    description: str
    returns: str
    category: Optional[str] = None  # NEW: track tool category
    parameters: Optional[Dict] = None  # NEW: auto-extract from signature
    examples: Optional[List[str]] = None  # NEW: usage examples
```

### 5. Auto-Discovery Pattern (Optional Enhancement)

For maximum automation:

```python
def _auto_discover_tools(self):
    """Automatically discover and import all tool modules."""
    import importlib
    import pkgutil
    from autoviz_agent import tools
    
    # Discover all modules in tools package
    for _, module_name, _ in pkgutil.iter_modules(tools.__path__):
        # Import module - decorators register tools automatically
        importlib.import_module(f'autoviz_agent.tools.{module_name}')
```

## Benefits

1. **Single Source of Truth**: Tool metadata lives with the function definition
2. **Impossible to Forget**: Can't add a tool without decorator (enforced by linter/tests)
3. **Easier Maintenance**: No synchronization between files needed
4. **Better Documentation**: Decorator can extract docstrings automatically
5. **Type Safety**: Can validate parameter types from annotations
6. **Categorization**: Automatically group tools by module/category
7. **Reduced Code**: Executor shrinks from ~50 lines to ~5 lines

## Testing Strategy

### Updated Test
```python
def test_all_tool_functions_decorated():
    """Ensure all public tool functions use @register_tool."""
    for module in [analysis, metrics, prep, data_io, visualization]:
        functions = get_public_functions(module)
        for func_name in functions:
            func = getattr(module, func_name)
            # Check if function was registered (has decorator applied)
            assert func_name in TOOL_REGISTRY.list_tools(), \
                f"{module.__name__}.{func_name} missing @register_tool decorator"
```

### Linter Rule (Optional)
```python
# Custom pylint checker
# Ensures all functions in tools/ have @register_tool decorator
```

## Migration Strategy

**Phased approach to minimize risk:**

### Phase 1: Infrastructure
- [ ] Create `registry/decorators.py` with `@register_tool`
- [ ] Update `ToolSchema` to support optional fields
- [ ] Add tests for decorator functionality

### Phase 2: Pilot Module
- [ ] Apply decorator to `visualization.py` (smallest, most stable)
- [ ] Remove visualization registrations from `executor.py`
- [ ] Test that all visualization tools work correctly
- [ ] Verify `test_tool_registration.py` still passes

### Phase 3: Remaining Modules
- [ ] Apply to `analysis.py`
- [ ] Apply to `metrics.py`
- [ ] Apply to `prep.py`
- [ ] Apply to `data_io.py`
- [ ] Remove all manual registrations from `executor.py`

### Phase 4: Simplification
- [ ] Simplify `executor._register_tools()` to just import modules
- [ ] Optional: Implement auto-discovery pattern
- [ ] Update documentation

### Phase 5: Enforcement
- [ ] Update test to check for decorator presence
- [ ] Add pre-commit hook to validate
- [ ] Document new pattern in contributor guide

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Import-time side effects | Medium | Keep decorators simple, only register |
| Circular imports | Low | Decorators don't import tool modules |
| Performance overhead | Low | Registration happens once at import |
| Breaking existing code | High | Use phased migration, test each step |

## Success Criteria

- [ ] All 24 tools auto-register via decorators
- [ ] `executor.py` reduced from 50+ lines to ~5 lines
- [ ] All existing tests pass
- [ ] `test_tool_registration.py` validates decorator presence
- [ ] Zero "tool not found" errors in CI/CD
- [ ] Documentation updated with new pattern

## Related Issues

- **Root Cause**: Manual registration led to `plot_boxplot` not being registered
- **Test Created**: `tests/test_tool_registration.py` catches missing registrations
- **Temporary Fix**: All 24 tools now manually registered

## Future Enhancements

1. **Auto-generate CLI commands** from registered tools
2. **Auto-generate API documentation** from tool schemas
3. **Tool versioning** - track schema changes over time
4. **Tool dependencies** - declare when tools require other tools
5. **Tool permissions** - control which tools are available in different contexts

## References

- Similar patterns: Flask's `@app.route`, pytest's `@pytest.fixture`
- Python decorators: [PEP 318](https://www.python.org/dev/peps/pep-0318/)
- Registry pattern: [Martin Fowler - Registry](https://martinfowler.com/eaaCatalog/registry.html)
