# Tooling Refactor Plan: Registry-Driven Tool Calls

## Goal
Replace the current manual tool registration and brittle tool-call generation with a registry-driven, schema-first pipeline that is backend-agnostic (gpt4all today, vLLM later). This reduces per-tool heuristics, centralizes defaults/validation, and makes tool calls deterministic and testable.

## Why This Change
- Manual `_register_tools()` is error-prone and easy to desync from `tools/*`.
- `LLMClient.generate_tool_calls()` has special-case logic that grows without bounds and encodes tool-specific knowledge in the LLM layer.
- Validation exists but is not part of the execution flow, so bad tool calls fail late.
- We want to swap LLM backends later (vLLM, llama.cpp) with minimal code churn.

## Non-Goals
- No changes to analytics logic or tool function behavior yet.
- No new UI/CLI behavior changes in this phase.
- No change to output artifacts format unless required by tool-call validation.

## Current Pain Points (Short)
- Tool list and schemas are duplicated in multiple places.
- Tool-call generation is fragile (implicit defaults, name fixes, etc.).
- No structured source of truth for tool definitions.

## Proposed Architecture
- **Registry is the source of truth** for tool metadata (name, description, params, defaults).
- **Tools self-register** via a decorator (no manual central registry list).
- **LLM produces tool calls in a validated schema**; local validation/repair happens before execution.
- **Defaults live with tools** or a shared resolver, not in the LLM layer.

## Phase Plan (Incremental)

### Phase 1: Tool Decorator + Schema Generation
1. Add a `@tool` decorator in `src/autoviz_agent/registry/tools.py`.
2. Decorator captures:
   - Name, description, params (from signature/docstring), types, required/defaults.
3. Add a `registry.export_schema()` method to produce a JSON schema for LLM use.

### Phase 2: Auto-Registration
1. Remove manual `_register_tools()` body in `src/autoviz_agent/runtime/executor.py`.
2. Import `autoviz_agent.tools.*` modules on executor init so decorators run and register tools.
3. Keep `TOOL_REGISTRY` as singleton, but add `registry.clear()` for tests.

### Phase 3: LLM Tool-Call Contract
1. Update `LLMClient` to request tool calls in a strict JSON schema.
2. Parse tool-call JSON, validate with `registry/validation.py`.
3. Add a repair step to adjust known minor issues (e.g., missing required arg) or reject.

### Phase 4: Move Defaults Out of LLM Layer
1. Remove per-tool heuristics from `LLMClient.generate_tool_calls()`.
2. Replace with a `ParamResolver` (new module) or per-tool default logic.
3. LLM provides intent-level parameters; resolver fills in from schema profile.

### Phase 5: vLLM Readiness
1. Add vLLM adapter to `LLMClient` (new transport layer).
2. Use `export_schema()` to pass tool definitions to vLLM `tools` API or embed in prompt.
3. Maintain a text-only fallback for gpt4all.

### Phase 6: update readme
1. update readme based on these changes.
2. Add a new explainer .md explaining how this new tool calling works.

## Proposed File Touches
- `src/autoviz_agent/registry/tools.py` (decorator, schema export, registry reset)
- `src/autoviz_agent/runtime/executor.py` (auto-import tools)
- `src/autoviz_agent/tools/*.py` (annotate with decorator)
- `src/autoviz_agent/llm/client.py` (tool-call generation to schema-driven output)
- `src/autoviz_agent/registry/validation.py` (wire validation into runtime flow)
- `src/autoviz_agent/runtime/executor.py` or new `param_resolver.py` (defaults)

## Test Plan

### Unit Tests
1. **Tool decorator registration**
   - Verify decorated functions appear in registry with schema.
   - File: `tests/unit/test_tool_registry.py`

2. **Schema export**
   - Validate exported schema matches expected JSON structure.
   - File: `tests/unit/test_tool_schema_export.py`

3. **Validation behavior**
   - Missing required arg -> validation error.
   - Unknown arg -> validation error.
   - File: `tests/unit/test_tool_validation.py`

4. **Param resolver**
   - Given schema profile, defaults are filled for missing args.
   - File: `tests/unit/test_param_resolver.py`

### Integration Tests
1. **End-to-end tool call path**
   - Simulate LLM returning tool-call JSON and execute via executor.
   - Validate execution log output.
   - File: `tests/integration/test_tool_call_flow.py`

2. **LLM fallback (text-only)**
   - Ensure gpt4all fallback still produces valid JSON tool calls.
   - File: `tests/integration/test_llm_tool_calls_fallback.py`

### Regression Tests
- Update or replace:
  - `tests/test_tool_registration.py` (to new registration model)
  - `tests/test_param_mapping.py` (to new resolver)

## Acceptance Criteria
- All tools registered without manual lists.
- LLM tool calls validated before execution.
- No per-tool heuristic logic in `LLMClient.generate_tool_calls()`.
- vLLM switch requires only transport + prompt/tool schema wiring.
- Tests above pass locally.

## Notes on LLM Backend Dependencies
- Tool registry, schema, and validation are backend-agnostic.
- vLLM provides native tool calling; gpt4all will remain JSON-in-text.
- The plan intentionally isolates backend transport from tool execution.
