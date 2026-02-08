# Implementation Plan: Planning Pipeline Upgrade

This document translates `docs/PLANNING_PIPELINE.md` into an implementation outline and concrete tasks. It is intended for an AI agent to execute.

## Implementation Outline (Phased)

1) Contracts and prompts for new LLM steps
2) Tool metadata registry integration and embedding index
3) Tool narrowing + retrieval integration
4) Requirement extraction + plan adaptation wiring
5) Coverage check + mapping registry
6) Repair policy alignment + replan triggers
7) Tests, CI gates, and documentation updates

## Task List (Actionable)

### Phase 1: Contracts and Prompt Specs

- Add requirement extraction prompt template. Files: `templates/prompts/requirements.md`.
- Add requirement extraction JSON schema. Files: `src/autoviz_agent/llm/llm_contracts.py`.
- Update prompt builder to build requirement extraction prompt and expose the schema. Files: `src/autoviz_agent/llm/prompts.py`.
- Wire vLLM client to call requirement extraction with xgrammar2 `response_format`. Files: `src/autoviz_agent/llm/vllm_client.py`.
- Wire gpt4all client to call requirement extraction with JSON parsing. Files: `src/autoviz_agent/llm/client.py`.
- Ensure JSON-only outputs with `additionalProperties: false` for xgrammar2 compliance. Files: `src/autoviz_agent/llm/llm_contracts.py`, `templates/prompts/requirements.md`.
- Preserve existing intent/adaptation prompt fields and JSON schemas. Files: `templates/prompts/intent.md`, `templates/prompts/adapt_plan.md`, `src/autoviz_agent/llm/llm_contracts.py`.

### Phase 2: Tool Metadata Registry and Embedding Index

- Define a deterministic tool document format sourced from `TOOL_REGISTRY`. Files: `src/autoviz_agent/registry/tools.py`, `src/autoviz_agent/planning/retrieval.py`.
- Add capability metadata to tool schemas if missing. Files: `src/autoviz_agent/registry/tools.py` and tool definitions under `src/autoviz_agent/tools/`.
- Implement tool embedding index builder using `sentence-transformers` + `faiss-cpu`. Files: `src/autoviz_agent/planning/retrieval.py` (or new module), plus config in `config.yaml`.
- Add cache directory and registry hashing for re-index. Files: `src/autoviz_agent/planning/retrieval.py`.
- Expose retrieval API returning top-N tools. Files: `src/autoviz_agent/planning/retrieval.py`.
- Treat runtime registry as source of truth; no static lists. Files: `src/autoviz_agent/registry/tools.py`, `src/autoviz_agent/llm/prompts.py`.

### Phase 3: Tool Narrowing Integration

- Build tool narrowing queries from the requirement schema. Files: `src/autoviz_agent/planning/retrieval.py`.
- Apply precedence rules (template, retrieval, safety, cap). Files: `src/autoviz_agent/planning/retrieval.py`, `src/autoviz_agent/planning/template_loader.py`.
- Cap tool list and drop lowest-scoring retrieval tools. Files: `src/autoviz_agent/planning/retrieval.py`.
- Pass only narrowed tool catalog to plan adaptation prompt. Files: `src/autoviz_agent/llm/prompts.py`, `src/autoviz_agent/planning/tool_calls.py` (if needed).
- Ensure narrowed list derived from `TOOL_REGISTRY`. Files: `src/autoviz_agent/registry/tools.py`, `src/autoviz_agent/llm/prompts.py`.
- Add retrieval fallback (expand N or discovery query) when coverage fails. Files: `src/autoviz_agent/planning/retrieval.py`, `src/autoviz_agent/graph/nodes.py`.
- Define template-curated tool subsets per intent and load them. Files: `templates/*.json`, `src/autoviz_agent/planning/template_loader.py`.

### Phase 4: Requirement Extraction + Plan Adaptation Wiring

- Insert requirement extraction after intent/template selection. Files: `src/autoviz_agent/graph/nodes.py`, `src/autoviz_agent/graph/graph_builder.py`.
- Feed requirements into tool narrowing and plan adaptation. Files: `src/autoviz_agent/planning/tool_calls.py`, `src/autoviz_agent/llm/prompts.py`.
- Add intent override logic on coverage failure/contradiction. Files: `src/autoviz_agent/graph/nodes.py`, `src/autoviz_agent/planning/template_loader.py`.
- Update adaptation prompt to require requirement-to-step mapping. Files: `templates/prompts/adapt_plan.md`, `src/autoviz_agent/llm/prompts.py`.
- Ensure requirement extraction schema enforces closed labels and time grain rules. Files: `src/autoviz_agent/llm/llm_contracts.py`.

### Phase 5: Coverage Check and Mapping Registry

- Implement deterministic requirement-to-capability mapping (table-driven, versioned). Files: `src/autoviz_agent/planning/schema_tags.py` or new module.
- Ensure each plan step declares which requirement(s) it satisfies. Files: `src/autoviz_agent/planning/diff.py`, `src/autoviz_agent/planning/template_schema.py`.
- Add a coverage validator that:
  - Verifies all requirements have at least one matching step.
  - Flags unjustified steps (no mapped requirement).
  - Enforces group_by/time requirements with aggregate/segment and time-aware tools.
- Add replan triggers when coverage fails, with error payload. Files: `src/autoviz_agent/graph/nodes.py`, `src/autoviz_agent/planning/diff.py`.
- Add label validation and unknown requirement policy. Files: `src/autoviz_agent/llm/llm_contracts.py`, `src/autoviz_agent/planning/schema_tags.py`.

### Phase 6: Repair Policy Alignment

- Classify repairs as safe vs semantic in runtime repair path. Files: `src/autoviz_agent/registry/validation.py`, `src/autoviz_agent/runtime/param_resolver.py`.
- Ensure semantic repairs trigger replan instead of auto-fix. Files: `src/autoviz_agent/graph/nodes.py`, `src/autoviz_agent/runtime/executor.py`.
- Audit existing repair logic. Files: `src/autoviz_agent/registry/validation.py`, `src/autoviz_agent/runtime/executor.py`.
- Record repair provenance and log changes consistently. Files: `src/autoviz_agent/reporting/execution_log.py`.

### Phase 7: Tests, CI Gates, and Docs

- Add unit tests for:
  - requirement schema validation and JSON-only parsing
  - mapping registry completeness (requirement labels must map to capabilities)
  - tool registry capability consistency
  - prompt/schema parity for xgrammar2 (schemas match prompt response specs)
- Add integration tests for the end-to-end flow with xgrammar2.
- Add regression questions to detect extraction/coverage drift.
- Update docs to reflect new prompts, schemas, and workflow guarantees.
- Add CI gates for registry hash changes (force re-index) and mapping completeness.
  Files: `tests/unit/test_llm_generation.py`, `tests/unit/test_adaptation.py`, `tests/integration/test_vllm_integration.py`, `docs/PLANNING_PIPELINE.md`, `docs/IMPLEMENTATION_PLAN.md`.

## Cross-Check Coverage Against `docs/PLANNING_PIPELINE.md`

Included items:
- Intent classification + template selection preserved.
- Requirement extraction step with xgrammar2 schema and prompt.
- Tool metadata registry as source of truth; deterministic tool document format.
- FAISS + sentence-transformers (`bge-small-en`) index, cached by registry hash.
- Tool narrowing precedence, safety set, and cap behavior.
- Retrieval fallback on coverage failure.
- Template strategy with curated tool subsets.
- Deterministic requirement-to-capability mapping + coverage check.
- Time grain handling with irregular interval heuristic.
- Repair policy (safe vs semantic) + replan triggers.
- Metrics and evaluation tracking hooks.
- High-level weaknesses and mitigations addressed via tests and CI gates.
- Example walkthrough support (no changes required, but maintained for future updates).
- Compatibility with existing xgrammar2 prompt/schema contracts.

If any of these items are missing in implementation, update this plan before coding.

## Traceability Matrix (Planning Pipeline -> Implementation Tasks)

- Context and motivation -> Phase 7 docs update + regression tests for drift.
- New dependencies (faiss-cpu, sentence-transformers, bge-small-en) -> Phase 2 embedding index + retrieval API.
- High-level flow (intent -> requirements -> narrowing -> adaptation -> coverage -> execution) -> Phases 1-6.
- Intent classification compatibility (xgrammar2) -> Phase 1 prompt/schema preservation + Phase 7 parity tests.
- Requirement extraction schema + prompt -> Phase 1 tasks + Phase 4 wiring.
- Requirement-to-capability mapping + mitigations -> Phase 5 mapping + Phase 7 CI gates.
- Tool metadata source-of-truth + embedding format -> Phase 2 tool document + registry enforcement.
- Tool narrowing precedence + safety set -> Phase 3 integration tasks.
- Template strategy (curated tool subsets) -> Phase 3 template loader + template updates.
- Time grain handling -> Phase 5 coverage + Phase 4 wiring (execution policy).
- Repair policy (safe vs semantic) -> Phase 6 alignment tasks.
- High-level weaknesses/mitigations -> Phase 7 tests/CI + Phase 4 intent override.
- Metrics and evaluation -> Phase 7 tests + reporting hooks.

## Implementation Checklist

| Status | Task | Files |
| --- | --- | --- |
| [ ] | Requirement extraction prompt template | `templates/prompts/requirements.md` |
| [ ] | Requirement extraction schema + xgrammar2 contract | `src/autoviz_agent/llm/llm_contracts.py` |
| [ ] | PromptBuilder supports requirement extraction | `src/autoviz_agent/llm/prompts.py` |
| [ ] | vLLM client uses requirement extraction schema | `src/autoviz_agent/llm/vllm_client.py` |
| [ ] | gpt4all client supports requirement extraction | `src/autoviz_agent/llm/client.py` |
| [ ] | Preserve existing intent/adaptation schemas | `templates/prompts/intent.md`, `templates/prompts/adapt_plan.md`, `src/autoviz_agent/llm/llm_contracts.py` |
| [ ] | Tool document format + registry source-of-truth | `src/autoviz_agent/registry/tools.py`, `src/autoviz_agent/planning/retrieval.py` |
| [ ] | Capability metadata added to tools | `src/autoviz_agent/registry/tools.py`, `src/autoviz_agent/tools/` |
| [ ] | Capability aliases + core capability set | `src/autoviz_agent/planning/schema_tags.py` |
| [ ] | FAISS index build + cache hashing | `src/autoviz_agent/planning/retrieval.py` |
| [ ] | Retrieval API for top-N tools | `src/autoviz_agent/planning/retrieval.py` |
| [ ] | Registry hash logged on index rebuild | `src/autoviz_agent/planning/retrieval.py`, `src/autoviz_agent/utils/logging.py` |
| [ ] | Tool narrowing precedence + cap logic | `src/autoviz_agent/planning/retrieval.py`, `src/autoviz_agent/planning/template_loader.py` |
| [ ] | Narrowed tool catalog fed to prompts | `src/autoviz_agent/llm/prompts.py` |
| [ ] | Retrieval fallback on coverage failure | `src/autoviz_agent/planning/retrieval.py`, `src/autoviz_agent/graph/nodes.py` |
| [ ] | Template-curated tool subsets per intent | `templates/*.json`, `src/autoviz_agent/planning/template_loader.py` |
| [ ] | Requirement extraction inserted after intent | `src/autoviz_agent/graph/nodes.py`, `src/autoviz_agent/graph/graph_builder.py` |
| [ ] | Re-extraction retry on invalid/unknown labels | `src/autoviz_agent/graph/nodes.py` |
| [ ] | Requirements flow into adaptation | `src/autoviz_agent/planning/tool_calls.py`, `src/autoviz_agent/llm/prompts.py` |
| [ ] | Intent override on coverage failure | `src/autoviz_agent/graph/nodes.py`, `src/autoviz_agent/planning/template_loader.py` |
| [ ] | Adaptation prompt requires requirement-to-step mapping | `templates/prompts/adapt_plan.md`, `src/autoviz_agent/llm/prompts.py` |
| [ ] | Requirement extraction closed labels + time grain rules | `src/autoviz_agent/llm/llm_contracts.py` |
| [ ] | Requirement-to-capability mapping | `src/autoviz_agent/planning/schema_tags.py` |
| [ ] | Plan steps declare satisfied requirements | `src/autoviz_agent/planning/diff.py`, `src/autoviz_agent/planning/template_schema.py` |
| [ ] | Coverage validator + replan trigger | `src/autoviz_agent/planning/diff.py`, `src/autoviz_agent/graph/nodes.py` |
| [ ] | Unknown requirement label policy | `src/autoviz_agent/llm/llm_contracts.py`, `src/autoviz_agent/planning/schema_tags.py` |
| [ ] | Time-grain auto policy for plots | `src/autoviz_agent/runtime/param_resolver.py`, `src/autoviz_agent/tools/visualization.py` |
| [ ] | Repair classification (safe vs semantic) | `src/autoviz_agent/registry/validation.py`, `src/autoviz_agent/runtime/param_resolver.py` |
| [ ] | Semantic repairs trigger replan | `src/autoviz_agent/graph/nodes.py`, `src/autoviz_agent/runtime/executor.py` |
| [ ] | Repair provenance logging | `src/autoviz_agent/reporting/execution_log.py` |
| [ ] | Unit tests for schemas + mapping | `tests/unit/test_schema.py`, `tests/unit/test_adaptation.py` |
| [ ] | Integration tests with xgrammar2 | `tests/integration/test_vllm_integration.py` |
| [ ] | Regression prompts suite | `tests/` (new or existing) |
| [ ] | Metrics tracking hooks (coverage, repair rate) | `src/autoviz_agent/reporting/execution_log.py` |
| [ ] | Dependencies added (faiss-cpu, sentence-transformers) | `pyproject.toml`, `config.yaml` |
| [ ] | Docs updated | `docs/PLANNING_PIPELINE.md`, `docs/IMPLEMENTATION_PLAN.md` |
| [ ] | CI gates (registry hash + mapping) | `tests/`, CI config if present |
