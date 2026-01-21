<!--
Sync Impact Report
- Version change: N/A → 1.0.0
- Modified principles: N/A (initial adoption)
- Added sections: Core Principles (7), Auditability Requirements, Deviation Handling
- Removed sections: N/A
- Templates requiring updates:
	- .specify/templates/plan-template.md ✅ updated
	- .specify/templates/spec-template.md ✅ updated
	- .specify/templates/tasks-template.md ✅ updated
- Follow-up TODOs:
	- TODO(RATIFICATION_DATE): original adoption date not provided
-->
# AutoViz Agent Constitution

## Core Principles

### I. Agentic Programming, Deterministic Execution
AutoViz Agent MUST demonstrate agentic programming, not Auto-EDA or prompt-driven
scripting. Planning, plan selection, plan adaptation, and execution boundaries
MUST be explicit, inspectable, and persisted. All analysis, statistics,
aggregation, and visualization logic MUST be deterministic code.

### II. Bounded LLM Capabilities
The LLM is a bounded reasoning component, never a source of truth. It MAY only:
- classify intent
- select among predefined plan templates
- adapt plans by pruning, reordering, or parameterizing steps
- emit structured tool calls
It MUST NEVER:
- write analysis code
- compute statistics
- generate plots directly
- invent tools, metrics, or workflows

### III. Curated Planning Discipline
All plans MUST originate from a curated plan library. Exact keyword matching
against user text is forbidden. Plan retrieval MUST use bounded intent labels,
schema-derived hard filters, and deterministic scoring. Plan mutation MUST be
logged as a diff with rationale.

### IV. Reliability and Safety Guarantees
Tool calls MUST be schema-validated and unknown tools MUST be rejected. Invalid
outputs MUST trigger repair or clarification loops. Two-pass generation is
mandatory: (1) optional unconstrained explanation and (2) strict schema-
constrained tool calls.

### V. Offline Model and Infrastructure Constraints
The system MUST run fully offline. It MUST support 4GB GPU laptops for
development and 12GB GPUs as the primary target. Quantization is a first-class
requirement. llama.cpp support is mandatory; vLLM support is optional but
preferred.

### VI. Demonstration Anti-Drift Evidence
The system MUST visibly show the original template, the adapted plan, and the
plan diff. At least one demo MUST prove plan step removal, plan step insertion,
and parameter specialization. The system MUST include a canonical demo (e.g.,
December revenue anomaly).

### VII. Human-Centered Documentation
A clear, README is mandatory. The README MUST explain
architecture, graph flow, templates, tools, configs, and extension paths.
Assume the codebase is partially AI-generated and prioritize clarity.

## Auditability Requirements

Every run MUST persist evidence that makes planning and execution inspectable,
including the chosen template, the adapted plan, the plan diff with rationale,
the tool call list, and the execution log. Evidence artifacts MUST be retained
with the run outputs and referenced in the report.

## Deviation Handling

Any proposed deviation from this constitution MUST be documented with a risk
assessment, an explicit mitigation plan, and a time-bound expiration. If a
deviation becomes permanent, the constitution MUST be amended before release.

## Governance

- This constitution supersedes conflicting project documentation.
- Amendments require a documented proposal, an updated Sync Impact Report, and
	a semantic version bump.
- All specifications and plans MUST include a constitution compliance check.
- Reviews MUST verify that LLM boundaries, deterministic execution, offline
	operation, and demonstration evidence remain intact.

**Version**: 1.0.0 | **Ratified**: TODO(RATIFICATION_DATE): original adoption date not provided | **Last Amended**: 2026-01-21
