# Planning Pipeline for Scalable Intent to Plan Adaptation

This document defines a scalable, LLM-robust planning pipeline that supports diverse user language, rapid tool/workflow growth, and small-model constraints.

## Context and Motivation

Current flow (simplified):
- Intent classification selects a template.
- A single plan adaptation prompt edits the template using a full tool list.
- Execution includes repairs when tool params are invalid.

Observed issues:
- Small LLMs overfit to the template/tool list and add unjustified steps (e.g., anomaly and histogram when not requested).
- Large tool lists increase cognitive load and encourage spurious tool use.
- There is no hard link between user requirements and the steps that the plan claims to satisfy.

The changes below formalize requirements, narrow tools, and add coverage checks so the plan must reflect the user's request, even with smaller models.

## New Dependencies (Required)

These are required for tool retrieval and embeddings.

- Vector index: `faiss-cpu`.
- Embeddings: `sentence-transformers` with `bge-small-en` as the default model for robust short-text retrieval.

## Goals

- Robust to varied phrasing without brittle keyword rules.
- Scales as tools and workflows increase.
- Keeps prompts short and stable.
- Works with small LLMs; reduces hallucinated steps.
- Enables deterministic checks before execution.

## High-Level Flow

1) Classify intent and select the base template.
2) Extract structured requirements from the user question.
3) Narrow tool candidates using metadata and retrieval.
4) Adapt the template plan using requirements and candidate tools.
5) Validate coverage and remove unjustified steps.
6) Execute with runtime repairs if needed.

## 1) Intent Classification and Template Selection

Intent classification is unchanged from today: it selects a base template that matches the
primary analysis type (time series, comparative, anomaly, etc.). This step must remain
compatible with existing prompt specs and xgrammar2 contracts.

## 2) Structured Requirements Extraction

LLM produces a small schema. The schema is the contract for planning.

Schema (example):

```
{
  "metrics": ["revenue"],
  "group_by": ["region", "product_category"],
  "time": {"column": "date", "grain": "unknown"},
  "analysis": ["total", "compare", "trend"],
  "outputs": ["chart", "table"],
  "constraints": []
}
```

Rules:
- Use a closed set for analysis types (total, compare, trend, distribution, anomaly, correlation, etc.).
- Do not infer extra analysis beyond the question.
- Leave fields empty or unknown if not present.

### Requirement-to-Capability Mapping (Deterministic)

The coverage check should be deterministic and rely on a fixed mapping from requirements to tool capabilities.
This avoids LLM disagreement and makes missing coverage actionable.

Suggested mapping:
- analysis: total -> requires capability: aggregate
- analysis: compare -> requires capability: aggregate or segment
- analysis: trend -> requires capability: time_series_plot or time_series_features
- analysis: distribution -> requires capability: distribution_plot or distribution_stats
- analysis: anomaly -> requires capability: anomaly_detection
- outputs: chart -> requires at least one plot capability
- outputs: table -> requires aggregate or summary_stats
- group_by: non-empty -> requires aggregate or segment with group_by params
- time: column set -> requires parse_datetime (unless already typed) and time-aware plot or features

Each planned step must declare which requirement(s) it satisfies.

#### Risk Mitigations

- Version the mapping registry and require updates in CI when new requirement labels or tool capabilities are added.
- Allow many-to-many mappings (one requirement satisfied by multiple capabilities) to reduce false negatives.
- Add an "unknown requirement" policy: warn and trigger re-extraction with the allowed label set instead of hard failing.
- Maintain capability aliases to avoid breakage from naming drift.
- Enforce label validation: if extraction emits labels outside the allowed set, block and re-run extraction.
- Keep a minimal core capability set to avoid total failure when the registry is incomplete.

## 3) Tool Metadata and Catalog

Every tool has compact metadata in a registry file (YAML or JSON).

Example metadata:

```
- name: aggregate
  capabilities: ["aggregate", "group_by", "summarize"]
  inputs: ["df"]
  params: ["group_by", "agg_func", "metrics"]
  outputs: ["df"]

- name: plot_line
  capabilities: ["plot", "time_series"]
  inputs: ["df"]
  params: ["x", "y", "color"]
  outputs: ["chart"]
```

This avoids embedding full tool lists in prompts.

Embedding note:
- Precompute and embed a compact tool document that includes name + description + capabilities + key params + outputs.
- At runtime, only embed the query derived from requirements (not the tools).
- Compare query embeddings against the precomputed tool embeddings in FAISS.

## 4) Tool Narrowing Strategy

Goal: provide the LLM a small, relevant tool subset.

Pipeline:
- Build retrieval queries from requirements (each requirement + combined).
- Retrieve top-N tools by embedding similarity against tool metadata.
- Add template-curated tools (intent-specific minimal set).
- Add a tiny safety set (aggregate, plot_line, compute_summary_stats) if not present.
- Cap total tools (8-12) to keep the prompt compact.

Precedence and conflict resolution:
- Template-curated tools are always included.
- Retrieval tools are appended until the cap is reached.
- Safety set is appended only if missing, even if it exceeds the cap by 1-2 tools.
- If the cap is exceeded, drop the lowest-scoring retrieval tools first, never drop template or safety tools.

If you want pure RAG, add a fallback step if coverage validation fails:
- Expand N or run a tool discovery query focused on missing requirements.

## 5) Plan Adaptation

Input:
- Requirements schema
- Candidate tool subset
- Template plan

Output:
- JSON modifications: add/remove/modify steps with tool, params, and rationale.

Hard constraints:
- Each step must cite which requirement it satisfies.
- Remove steps with no matching requirement.
- For group_by or time requirements, ensure aggregate or segmentation precedes plotting.

### Time Grain Handling

Recommendation: infer time grain during extraction with a closed enum, but only select a grain if explicitly mentioned.
If grain is unknown, defer to execution with an auto-grain policy based on data span and density.

Example policy:
- < 90 days span or > 60 points: daily
- 90-365 days: weekly
- > 365 days: monthly

This keeps extraction conservative while ensuring plots remain readable.

## 6) Coverage and Consistency Check

A deterministic checker validates:

- Every requirement has at least one plan step mapping.
- Each step maps to at least one requirement.
- Group_by requirements are fulfilled by aggregate/segment steps.
- Trend/time requirements are fulfilled by time-based plots or features.

If invalid, retry adaptation with an explicit error payload:

```
Missing coverage: group_by=[region, product_category]
Remove unjustified steps: detect_anomalies
```

## 7) Execution and Repair

During tool execution:
- Validate parameters against tool schemas.
- Repair known issues (missing df, invalid params) and log changes.
- Record provenance for each repair.

Repair policy:
- Safe repairs (missing df, column name casing, default params) are allowed with explicit logging.
- Semantic repairs (changing group_by, metric, or time grain) should fail fast and trigger re-planning.
- Any repair that changes analysis intent must be rejected.

## Prompt Set (Minimal)

### A) Intent Classification Prompt (LLM)

- Input: question + dataset schema
- Output: strict JSON schema

### B) Requirement Extraction Prompt (LLM)

- Input: question + dataset schema
- Output: strict JSON schema

### C) Tool Narrowing (System)

- Input: requirements
- Output: tool subset

### D) Plan Adaptation Prompt (LLM)

- Input: requirements + tool subset + template
- Output: JSON changes

### E) Coverage Check (Deterministic + LLM Optional)

- If missing coverage, return a structured error and re-run adaptation.

### Compatibility With Existing LLM Contracts (xgrammar2)

This plan must preserve the current structured-output pipeline and prompt specs:
- Prompts live in `templates/prompts/intent.md` and `templates/prompts/adapt_plan.md`.
- JSON schemas for xgrammar2 live in `src/autoviz_agent/llm/llm_contracts.py`.
- The vLLM client enforces grammar via `response_format` using those schemas.

Any new LLM step (e.g., requirement extraction) must:
- Add a prompt template under `templates/prompts/`.
- Add a JSON schema contract in `src/autoviz_agent/llm/llm_contracts.py`.
- Use JSON-only responses with `additionalProperties: false` to keep xgrammar2 strict.
- Be wired through `PromptBuilder` and the vLLM client so the grammar path stays intact.

Do not change existing intent/adaptation schemas or prompt fields in a way that breaks
xgrammar2 validation.

## Template Strategy

Maintain intent-specific templates (time_series, comparative, anomaly).
Each template references a curated tool subset.

Example for time_series_grouped:
- parse_datetime
- aggregate (group_by time + categories)
- plot_line (x=time, y=metric, color=category)
- summary_stats (optional)

No anomalies or distributions unless required by the schema.

## Scalability Notes

- Adding a tool: update metadata only.
- Adding a workflow: create a new template + minimal tool subset.
- Diverse user language: handled in requirement extraction, not in tool mapping.
- Small LLMs: short prompts + validation loops reduce drift.

## Metrics and Evaluation

Track:
- Requirement coverage rate
- Unjustified step rate
- Tool retrieval miss rate
- Plan repair rate
- User satisfaction with outputs

Use a small test suite of questions to regression test extraction and planning.

## Implementation Steps

1) Define the requirement schema and allowed analysis labels.
2) Build tool metadata registry.
3) Implement retrieval-based narrowing with template safety set.
4) Update plan adaptation prompt to require requirement-to-step mapping.
5) Implement coverage validator and retry loop.
6) Add regression tests for intent extraction and plan adaptation.

## Indexing and Embedding Implementation

Recommended design:
- Tool metadata is embedded once at startup and cached to disk.
- Re-embed only when the tool registry changes (hash the registry file).
- Use FAISS for local similarity search over tool metadata strings.
- Store embeddings and metadata in a small local cache directory (e.g., `.cache/tool_index`).

This keeps retrieval fast and deterministic while avoiding repeated embedding costs.

## Example Walkthrough

User question:
\"get revenue totals by region and product type over time\"

Dataset schema:
- date (temporal), revenue (numeric), region (categorical), product_category (categorical)

Note: The outputs below are conceptual. Actual LLM responses should be strict JSON per the prompt specs.

### 1) Intent Classification and Template Selection

Intent output:
- primary: time_series_investigation
- confidence: 0.90

Template selected:
- time_series_grouped

### 2) Requirement Extraction Output

```
{
  \"metrics\": [\"revenue\"],
  \"group_by\": [\"region\", \"product_category\"],
  \"time\": {\"column\": \"date\", \"grain\": \"unknown\"},
  \"analysis\": [\"total\", \"compare\", \"trend\"],
  \"outputs\": [\"chart\", \"table\"],
  \"constraints\": []
}
```

### 3) Tool Narrowing (Example Result)

Template tools (time_series_grouped):
- parse_datetime
- aggregate
- plot_line
- compute_summary_stats

Retrieved tools (top-N):
- segment_metric
- plot_bar

Safety tools added:
- aggregate (already present)
- plot_line (already present)
- compute_summary_stats (already present)

Final tool subset (cap 8):
- parse_datetime, aggregate, plot_line, compute_summary_stats, segment_metric, plot_bar

### 4) Plan Adaptation (Changes)

- Add aggregate step:
  - tool: aggregate
  - params: group_by=[\"date\", \"region\", \"product_category\"], agg_func=\"sum\", metrics=[\"revenue\"]
  - satisfies: analysis.total, group_by, time

- Modify plot_line step:
  - tool: plot_line
  - params: x=\"date\", y=\"revenue\", color=\"product_category\"
  - satisfies: analysis.trend, outputs.chart

- Add table output:
  - tool: compute_summary_stats or save_dataframe
  - satisfies: outputs.table

- Remove detect_anomalies or plot_histogram if present:
  - no matching requirement

### 5) Coverage Check

- analysis.total -> aggregate (OK)
- analysis.compare -> aggregate + grouping (OK)
- analysis.trend -> plot_line (OK)
- outputs.chart -> plot_line (OK)
- outputs.table -> summary_stats/save_dataframe (OK)
- group_by -> aggregate with group_by (OK)
- time -> parse_datetime + plot_line (OK)

### 6) Execution

- parse_datetime: casts date column
- aggregate: computes revenue totals per date/region/product_category
- plot_line: time series plot by product_category (optionally faceted by region)
- summary_stats/save_dataframe: table output

Result: plan includes only user-requested analysis steps and avoids anomaly/distribution extras.

## Recommendations on Open Questions

- Coverage check: deterministic mapping is best. LLM-assisted checks can be additive, but the deterministic map should be the gate.
- Time grain inference: conservative in extraction, automatic in execution. This avoids overconfident grain guesses while still producing readable outputs.
