# Implementation Plan: AutoViz Agent

**Branch**: `001-autoviz-agent` | **Date**: 2026-01-21 | **Spec**: [specs/001-autoviz-agent/spec.md](specs/001-autoviz-agent/spec.md)
**Input**: Feature specification from `/specs/001-autoviz-agent/spec.md`

**Note**: This template is filled in by the `/speckit.plan` command. Ensure the Constitution Check section reflects .specify/memory/constitution.md.

## Summary

Build AutoViz Agent as an offline, deterministic analysis pipeline that accepts
datasets and questions, infers schema and intent, retrieves and adapts curated
plan templates, and executes registered tools to produce 5–10 charts, a report,
and full provenance artifacts. The design enforces bounded LLM responsibilities,
schema-validated tool calls, and persistent planning evidence that makes plan
selection and mutation inspectable.

## Technical Context

<!--
  ACTION REQUIRED: Replace the content in this section with the technical details
  for the project. The structure here is presented in advisory capacity to guide
  the iteration process.
-->

**Language/Version**: Python 3.10+ (target 3.11)  
**Primary Dependencies**: LangGraph, pandas, numpy, pydantic, jsonschema, matplotlib, seaborn, pyyaml  
**Storage**: Local files (JSON, Markdown, PNG/SVG)  
**Testing**: pytest  
**Target Platform**: Offline desktop (Windows/macOS/Linux)  
**Project Type**: single  
**Performance Goals**: 95% of datasets up to 1M rows x 100 columns complete within 5 minutes  
**Constraints**: Fully offline; deterministic execution; quantized local model runtime; llama.cpp required; 4GB GPU dev / 12GB GPU target  
**Scale/Scope**: Single-user runs; local artifacts per run

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- [x] Plan originates from curated plan library (no exact keyword matching)
- [x] Plan retrieval uses bounded intent labels, hard filters, deterministic scoring
- [x] Plan mutation recorded as diff with rationale
- [x] LLM role limited to intent, template selection, plan adaptation, tool calls
- [x] Analysis/aggregation/visualization executed as deterministic code
- [x] Tool calls schema-validated; unknown tools rejected; repair loop defined
- [x] Two-pass generation planned (optional explanation + schema tool calls)
- [x] Offline execution and quantization requirements addressed (llama.cpp mandatory)
- [x] Demo evidence planned (template, adapted plan, diff; mutation proof)
- [x] README updates planned for architecture and extension paths

**Post-Design Re-check (Phase 1)**: PASS

## Project Structure

### Documentation (this feature)

```text
specs/[###-feature]/
├── plan.md              # This file (/speckit.plan command output)
├── research.md          # Phase 0 output (/speckit.plan command)
├── data-model.md        # Phase 1 output (/speckit.plan command)
├── quickstart.md        # Phase 1 output (/speckit.plan command)
├── contracts/           # Phase 1 output (/speckit.plan command)
└── tasks.md             # Phase 2 output (/speckit.tasks command - NOT created by /speckit.plan)
```

### Source Code (repository root)
<!--
  ACTION REQUIRED: Replace the placeholder tree below with the concrete layout
  for this feature. Delete unused options and expand the chosen structure with
  real paths (e.g., apps/admin, packages/something). The delivered plan must
  not include Option labels.
-->

```text
src/
└── autoviz_agent/
  ├── cli/
  ├── config/
  ├── graph/
  ├── io/
  ├── llm/
  ├── models/
  ├── planning/
  ├── registry/
  ├── reporting/
  ├── runtime/
  ├── templates/
  ├── tools/
  └── utils/

tests/
├── contract/
├── integration/
└── unit/
```

**Structure Decision**: Single project layout for a local Python application.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

No violations identified.
