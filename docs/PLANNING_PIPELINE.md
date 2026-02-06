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

## New Dependencies (Optional)

These are for tool retrieval and embeddings if you adopt the narrowing strategy.

- Vector index: `faiss-cpu` (fast and common) or `hnswlib` (lightweight, simple install).
- Embeddings: `sentence-transformers` with `all-MiniLM-L6-v2` (small and fast) or `bge-small-en` (strong recall).
- Hosted embeddings alternative: OpenAI `text-embedding-3-small` via API if you already use OpenAI.

## Goals

- Robust to varied phrasing without brittle keyword rules.
- Scales as tools and workflows increase.
- Keeps prompts short and stable.
- Works with small LLMs; reduces hallucinated steps.
- Enables deterministic checks before execution.

## High-Level Flow

1) Extract structured requirements from the user question.
2) Narrow tool candidates using metadata and retrieval.
3) Adapt a template plan using the requirements and candidate tools.
4) Validate coverage and remove unjustified steps.
5) Execute with runtime repairs if needed.

## 1) Structured Requirements Extraction

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

## 2) Tool Metadata and Catalog

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

## 3) Tool Narrowing Strategy

Goal: provide the LLM a small, relevant tool subset.

Pipeline:
- Build retrieval queries from requirements (each requirement + combined).
- Retrieve top-N tools by embedding similarity against tool metadata.
- Add template-curated tools (intent-specific minimal set).
- Add a tiny safety set (aggregate, plot_line, compute_summary_stats) if not present.
- Cap total tools (8-12) to keep the prompt compact.

If you want pure RAG, add a fallback step if coverage validation fails:
- Expand N or run a tool discovery query focused on missing requirements.

## 4) Plan Adaptation

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

## 5) Coverage and Consistency Check

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

## 6) Execution and Repair

During tool execution:
- Validate parameters against tool schemas.
- Repair known issues (missing df, invalid params) and log changes.
- Record provenance for each repair.

## Prompt Set (Minimal)

### A) Requirement Extraction Prompt (LLM)

- Input: question + dataset schema
- Output: strict JSON schema

### B) Tool Narrowing (System)

- Input: requirements
- Output: tool subset

### C) Plan Adaptation Prompt (LLM)

- Input: requirements + tool subset + template
- Output: JSON changes

### D) Coverage Check (Deterministic + LLM Optional)

- If missing coverage, return a structured error and re-run adaptation.

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
