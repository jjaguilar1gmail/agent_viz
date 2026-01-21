# Feature Specification: AutoViz Agent

**Feature Branch**: `001-autoviz-agent`  
**Created**: 2026-01-21  
**Status**: Draft  
**Input**: User description: "Build AutoViz Agent to accept datasets and analytical questions, infer schema and intent, select and adapt plans, execute deterministically, and produce charts, a report, and full provenance logs under strict safety constraints and a defined execution graph." 

## User Scenarios & Testing *(mandatory)*

<!--
  IMPORTANT: User stories should be PRIORITIZED as user journeys ordered by importance.
  Each user story/journey must be INDEPENDENTLY TESTABLE - meaning if you implement just ONE of them,
  you should still have a viable MVP (Minimum Viable Product) that delivers value.
  
  Assign priorities (P1, P2, P3, etc.) to each story, where P1 is the most critical.
  Think of each story as a standalone slice of functionality that can be:
  - Developed independently
  - Tested independently
  - Deployed independently
  - Demonstrated to users independently
-->

### User Story 1 - Run deterministic analysis (Priority: P1)

As an analyst, I want to submit a dataset and a natural-language question so
the system can infer schema, choose and adapt a plan, and deliver charts,
report, and provenance artifacts without relying on free-form LLM analysis.

**Why this priority**: This is the primary value of AutoViz Agent and the
minimum viable experience.

**Independent Test**: Can be fully tested by providing a CSV or DataFrame and
confirming deterministic outputs (charts, report, execution log) are produced.

**Acceptance Scenarios**:

1. **Given** a valid CSV and a question, **When** the run completes, **Then**
  the system outputs 5–10 charts, a markdown report, and an execution
  provenance log.
2. **Given** the same dataset and question, **When** the run is repeated,
  **Then** the outputs are identical in content and ordering.

---

### User Story 2 - Inspect plan provenance (Priority: P2)

As a reviewer, I want to see the original plan template, the adapted plan, and
the plan diff with rationale so I can understand why the system chose and
modified a plan.

**Why this priority**: Transparency and inspectability are core to the
agentic programming goals and risk mitigation.

**Independent Test**: Can be tested by running any analysis and verifying the
presence and readability of plan artifacts and tool call order.

**Acceptance Scenarios**:

1. **Given** a completed run, **When** I review the provenance artifacts,
  **Then** I can see the template, adapted plan, and plan diff with rationale.
2. **Given** a completed run, **When** I review the tool call list, **Then** I
  can determine the order of execution and inputs for each tool.

---

### User Story 3 - Handle invalid tool calls safely (Priority: P3)

As a user, I want the system to reject invalid or unknown tool calls and guide
me through repair or clarification so runs remain safe and trustworthy.

**Why this priority**: Reliability and safety are required for bounded LLM
operation.

**Independent Test**: Can be tested by injecting an invalid tool call and
confirming a safe repair or clarification loop occurs.

**Acceptance Scenarios**:

1. **Given** a tool call that violates schema, **When** the system validates it,
  **Then** it is rejected and a repair or clarification loop is initiated.
2. **Given** an unknown tool name, **When** validation occurs, **Then** the
  tool is rejected and the run records the rejection in the execution log.

---

### Edge Cases

- Dataset is empty or contains only headers.
- Dataset has unsupported or mixed data types that prevent schema inference.
- The question conflicts with available columns (e.g., missing referenced field).
- The intent classifier cannot confidently map to the bounded intent set.
- The plan library has no compatible template after hard filters.
- Tool output is malformed or missing expected artifacts.

## Requirements *(mandatory)*

All requirements MUST comply with the constitution (bounded LLM role,
deterministic analysis code, curated plan library, schema-validated tools,
offline execution, demo evidence, and documentation clarity).

### Functional Requirements

- **FR-001**: System MUST accept a dataset as CSV or DataFrame and a
  natural-language analytical question.
- **FR-002**: System MUST infer schema attributes including types, missingness,
  and cardinality for all dataset fields.
- **FR-003**: System MUST classify analytical intent strictly within the
  bounded set: general_eda, time_series_investigation, segmentation_drivers,
  anomaly_detection, comparative_analysis.
- **FR-004**: System MUST retrieve plans exclusively from a curated plan
  template library and MUST NOT use exact keyword matching against user text.
- **FR-005**: System MUST adapt the chosen plan based on schema features and
  user goals, and MUST record a diff with rationale.
- **FR-006**: System MUST execute the adapted plan deterministically and MUST
  NOT use the LLM for free-form analysis, statistics, aggregation, or
  visualization.
- **FR-007**: System MUST produce 5–10 opinionated charts, a markdown report,
  and a complete execution provenance log for each run.
- **FR-008**: System MUST validate tool calls against schemas, reject unknown
  tools, and initiate repair or clarification when outputs are invalid.
- **FR-009**: System MUST operate fully offline and MUST NOT use cloud APIs or
  AutoML capabilities.
- **FR-010**: System MUST expose an explicit graph-based execution flow with
  stages named: classify_intent, infer_schema, retrieve_template, adapt_plan,
  explain_plan (optional), compile_tool_calls, execute_tools, summarize_results,
  repair_or_clarify.
- **FR-011**: System MUST persist all planning and execution artifacts needed
  to answer why the plan was chosen, what changed, and which tools ran.

### Key Entities *(include if feature involves data)*

- **UserRequest**: The analytical question and context provided by the user.
- **Dataset**: The input table provided as CSV or DataFrame.
- **SchemaProfile**: Types, missingness, and cardinality per field.
- **Intent**: One of the bounded analytical intent labels.
- **PlanTemplate**: A curated plan selected from the library.
- **AdaptedPlan**: The plan after schema- and goal-driven adjustments.
- **PlanDiff**: The recorded changes and rationale from template to adapted plan.
- **ToolCall**: A validated, schema-constrained instruction to a deterministic tool.
- **ExecutionLog**: Ordered record of tool execution, inputs, outputs, and errors.
- **Artifact**: Charts, report, and provenance outputs for a run.

## Assumptions

- The system is used by a single analyst per run and stores outputs locally.
- Users provide datasets that fit into memory on the target hardware.
- The plan template library is curated and available before runs begin.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: For a valid dataset and question, users receive 5–10 charts, a
  markdown report, and an execution log in 100% of successful runs.
- **SC-002**: Re-running the same dataset and question produces identical
  outputs (charts, report text, and log ordering) 100% of the time.
- **SC-003**: 95% of target datasets (up to 1M rows and 100 columns) complete
  within 5 minutes on the primary target hardware.
- **SC-004**: 100% of tool calls in a run are schema-validated and unknown
  tools are rejected without execution.
- **SC-005**: Reviewers can determine plan selection rationale and tool order
  within 5 minutes using the persisted artifacts.
