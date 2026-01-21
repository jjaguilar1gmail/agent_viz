# Phase 0 Research: AutoViz Agent

## Decision 1: LangGraph for deterministic orchestration
**Decision**: Use LangGraph with explicit, deterministic routing and typed state.
**Rationale**: Supports explicit node boundaries, structured state, and auditable
control flow required by the constitution while remaining offline-capable.
**Alternatives considered**: Custom DAG runner (more bespoke effort, higher risk
of missing observability guarantees), Airflow (overkill for local runs).

## Decision 2: Tool call validation with Pydantic + JSON Schema
**Decision**: Define tool schemas with Pydantic models and emit JSON Schema for
validation; enforce `extra=forbid` and reject unknown tools before execution.
**Rationale**: Strong typing and strict validation protect the bounded LLM role
and enable deterministic, inspectable tool calls with clear repair messages.
**Alternatives considered**: Marshmallow (less strict by default), ad-hoc
validation (higher risk of drift and silent failures).

## Decision 3: Deterministic analytics and plotting (pandas + Matplotlib)
**Decision**: Use pandas for deterministic aggregation and Matplotlib (Agg
backend) for chart rendering with fixed seeds, sorting, and stable styling.
**Rationale**: Mature offline tooling with controllable determinism and file
artifact outputs aligned to provenance requirements.
**Alternatives considered**: Plotly (interactive but less deterministic),
Altair/Vega (adds rendering dependencies and potential runtime variability).

## Decision 4: Offline model runtime strategy
**Decision**: Standardize on llama.cpp as the mandatory offline runtime and
configure models via local YAML with quantized weights.
**Rationale**: Satisfies offline and hardware constraints while keeping model
selection explicit and reproducible.
**Alternatives considered**: vLLM-only (optional per requirements) or cloud APIs
(violates offline constraints).
