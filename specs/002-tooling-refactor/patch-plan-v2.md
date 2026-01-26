# Tooling Refactor Patch Plan (v2)

## Purpose
Targeted fixes to reduce brittleness in tool-call handling without a full rewrite. Focus is on strict validation, schema-driven defaults, and removal of per-tool heuristics that duplicate logic across layers.

## Scope
- Enforce tool-call validation before execution.
- Move defaults to registry-driven schemas and generic column selectors.
- Minimize tool-specific logic in `ParamResolver`.
- Keep current tool implementations intact.

## Planned Patches (Ordered)

### Patch 1: Strict Validation + Repair Gate
- **Change**: Make tool-call validation blocking.
- **Behavior**: If a tool call fails validation, try a minimal repair (fill missing required params from schema defaults). If still invalid, drop it and log.
- **Files**:
  - `src/autoviz_agent/llm/client.py`
  - `src/autoviz_agent/registry/validation.py`
- **Notes**: This prevents late runtime failures from bad args.

### Patch 2: Registry-Driven Defaults
- **Change**: Add default values and roles to `ToolParameter` schema and use them at runtime.
- **Behavior**: Resolver looks up required params from registry schema and fills from defaults without tool-specific branching.
- **Files**:
  - `src/autoviz_agent/registry/tools.py`
  - `src/autoviz_agent/runtime/param_resolver.py`
- **Notes**: This centralizes defaults in tool metadata instead of LLM logic.

### Patch 3: Generic Column Selector
- **Change**: Add a small helper to select columns by role (temporal/numeric/categorical), optionally honoring user mentions.
- **Behavior**: Any parameter tagged with role uses the same selector; no per-tool logic.
- **Files**:
  - `src/autoviz_agent/runtime/param_resolver.py` (or new `src/autoviz_agent/runtime/column_selectors.py`)
- **Notes**: Eliminates repeated heuristics for `plot_*`, `aggregate`, `segment_metric`, etc.

### Patch 4: Remove Tool-Specific Branches
- **Change**: Reduce `ParamResolver.resolve()` to a generic path that:
  - pulls the tool schema from registry,
  - fills required defaults,
  - resolves role-based columns,
  - normalizes known aliases (df -> data, annotation -> annot) only if declared.
- **Files**:
  - `src/autoviz_agent/runtime/param_resolver.py`
- **Notes**: Tool-specific methods are deleted or minimized.

### Patch 5: Clean Up Known Issues
- **Change**: Remove duplicate `_extract_mentioned_columns()` call and unused registry import.
- **Files**:
  - `src/autoviz_agent/runtime/param_resolver.py`

## Tests

### Unit Tests
- **Strict validation**
  - Invalid args are rejected or repaired, not executed.
  - New test: `tests/unit/test_tool_validation_strict.py`
- **Schema defaults**
  - Default values are injected from registry definitions.
  - New test: `tests/unit/test_tool_defaults.py`
- **Column selection**
  - Role-based selection returns correct column for temporal/numeric/categorical.
  - New test: `tests/unit/test_column_selectors.py`

### Integration Tests
- **Tool call flow**
  - Simulated tool-call JSON runs through validation + resolver + executor with no per-tool branches.
  - New test: `tests/integration/test_tool_call_flow_strict.py`

## Acceptance Criteria
- Any invalid tool call is blocked or repaired before execution.
- `ParamResolver` uses registry schema + generic selectors, not per-tool logic.
- Tool defaults and role hints live in a single source of truth.
- Tests pass and cover failure cases.

## Notes
- This plan keeps current tool functions unchanged.
- It lays groundwork for vLLM integration but does not require it.
