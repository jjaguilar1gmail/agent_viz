---

description: "Task list for AutoViz Agent implementation"
---

# Tasks: AutoViz Agent

**Input**: Design documents from `/specs/001-autoviz-agent/`
**Prerequisites**: plan.md (required), spec.md (required), research.md, data-model.md, contracts/, quickstart.md

**Tests**: Not requested; no test tasks included.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and basic structure

- [X] T001 Create package skeleton under src/autoviz_agent/ and __init__.py files
- [X] T002 Initialize pyproject.toml with runtime dependencies (LangGraph, pandas, numpy, pydantic, jsonschema, matplotlib, seaborn, pyyaml) and dev deps in pyproject.toml
- [X] T003 [P] Add logging configuration helper in src/autoviz_agent/utils/logging.py
- [X] T004 [P] Add settings/config defaults in src/autoviz_agent/config/settings.py
- [X] T005 [P] Add CLI entry scaffold in src/autoviz_agent/cli/main.py
- [X] T006 [P] Create tests/contract/, tests/integration/, and tests/unit/ directories with placeholder __init__.py files

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

- [X] T007 Define core state models (RunState, UserRequest, SchemaProfile, Artifact) in src/autoviz_agent/models/state.py
- [X] T008 Define tool registry and schema models in src/autoviz_agent/registry/tools.py and src/autoviz_agent/registry/schemas.py
- [X] T009 Implement plan template JSON schema in src/autoviz_agent/planning/template_schema.py
- [X] T010 Implement template loader + validator in src/autoviz_agent/planning/template_loader.py
- [X] T011 Implement plan retrieval algorithm and scoring in src/autoviz_agent/planning/retrieval.py
- [X] T012 Implement plan diff generation with rationale hooks in src/autoviz_agent/planning/diff.py
- [X] T013 Implement deterministic runtime helpers (seeds, ordering, matplotlib backend) in src/autoviz_agent/runtime/determinism.py
- [X] T014 Implement artifact path manager and persistence helpers in src/autoviz_agent/io/artifacts.py
- [X] T015 Implement execution log writer and structured entries in src/autoviz_agent/reporting/execution_log.py
- [X] T016 Implement bounded LLM client wrapper (intent + plan adaptation only) in src/autoviz_agent/llm/client.py
- [X] T017 Define graph state schema and routing enums in src/autoviz_agent/graph/state.py
- [X] T018 Implement graph node scaffolding and error routing in src/autoviz_agent/graph/nodes.py

---

## Phase 3: User Story 1 - Run deterministic analysis (Priority: P1) 🎯 MVP

**Goal**: Accept input, infer schema and intent, execute a deterministic plan, and output charts, report, and log

**Independent Test**: Provide a CSV and a question, then verify 5–10 charts, a report, and an execution log are produced and deterministic across runs.

### Implementation for User Story 1

- [X] T019 [P] [US1] Implement load_dataset and sample_rows in src/autoviz_agent/tools/data_io.py
- [X] T020 [P] [US1] Implement infer_schema in src/autoviz_agent/tools/schema.py
- [X] T021 [P] [US1] Implement handle_missing, parse_datetime, cast_types in src/autoviz_agent/tools/prep.py
- [X] T022 [P] [US1] Implement aggregate, compute_summary_stats, compute_correlations in src/autoviz_agent/tools/metrics.py
- [X] T023 [P] [US1] Implement detect_anomalies, segment_metric, compute_distributions in src/autoviz_agent/tools/analysis.py
- [X] T024 [P] [US1] Implement plotting tools and fixed styles in src/autoviz_agent/tools/visualization.py
- [X] T025 [US1] Implement report writer to produce report.md in src/autoviz_agent/reporting/report_writer.py
- [X] T026 [US1] Implement tool call compiler in src/autoviz_agent/planning/tool_calls.py
- [X] T027 [US1] Implement tool executor and registry dispatch in src/autoviz_agent/runtime/executor.py
- [X] T028 [US1] Build LangGraph pipeline wiring nodes in src/autoviz_agent/graph/graph_builder.py
- [X] T029 [US1] Implement CLI run command (CSV path + question) in src/autoviz_agent/cli/main.py
- [X] T030 [US1] Create baseline plan templates in src/autoviz_agent/templates/general_eda.json and src/autoviz_agent/templates/time_series_investigation.json
- [X] T031 [US1] Add schema-derived tags and data_shape detection in src/autoviz_agent/planning/schema_tags.py

**Checkpoint**: User Story 1 runs end-to-end and produces deterministic artifacts.

---

## Phase 4: User Story 2 - Inspect plan provenance (Priority: P2)

**Goal**: Make plan selection and mutation artifacts fully inspectable with clear references

**Independent Test**: Run analysis and verify plan_template.json, plan_adapted.json, plan_diff.md, and tool_calls.json are present and linked in report.

### Implementation for User Story 2

- [X] T032 [US2] Persist plan_template.json, plan_adapted.json, and plan_diff.md in src/autoviz_agent/io/artifacts.py
- [X] T033 [US2] Export tool_calls.json and execution_log.json in src/autoviz_agent/reporting/export.py
- [X] T034 [US2] Add provenance section linking artifacts in src/autoviz_agent/reporting/report_writer.py
- [X] T035 [US2] Implement run runner entrypoint to expose run metadata in src/autoviz_agent/runtime/runner.py

**Checkpoint**: A reviewer can answer why the plan was chosen, what changed, and which tools executed.

---

## Phase 5: User Story 3 - Handle invalid tool calls safely (Priority: P3)

**Goal**: Reject invalid tool calls and trigger repair or clarification loops

**Independent Test**: Inject an invalid tool call and verify the system rejects it and records a repair attempt.

### Implementation for User Story 3

- [X] T036 [US3] Implement tool call schema validation and unknown tool rejection in src/autoviz_agent/registry/validation.py
- [X] T037 [US3] Implement repair_or_clarify node logic in src/autoviz_agent/graph/nodes.py
- [X] T038 [US3] Add validation error models and logging in src/autoviz_agent/models/errors.py and src/autoviz_agent/reporting/execution_log.py

**Checkpoint**: Invalid tool calls are safely rejected with a logged repair/clarification path.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [X] T039 [P] Add README with architecture, graph flow, templates, tools, configs, and extension paths in README.md
- [X] T040 [P] Add canonical demo assets and instructions in examples/december_revenue/README.md
- [X] T041 [P] Add deterministic run verification script in scripts/verify_determinism.py and reference it in specs/001-autoviz-agent/quickstart.md
- [X] T042 Update quickstart usage examples in specs/001-autoviz-agent/quickstart.md

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3+)**: All depend on Foundational phase completion
- **Polish (Final Phase)**: Depends on all desired user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational (Phase 2)
- **User Story 2 (P2)**: Can start after Foundational (Phase 2)
- **User Story 3 (P3)**: Can start after Foundational (Phase 2)

### Parallel Opportunities

- T003, T004, T005, T006 can run in parallel after T001–T002
- In US1, T019–T024 can run in parallel (different files)

---

## Parallel Example: User Story 1

Task: "Implement load_dataset and sample_rows in src/autoviz_agent/tools/data_io.py"
Task: "Implement infer_schema in src/autoviz_agent/tools/schema.py"
Task: "Implement handle_missing, parse_datetime, cast_types in src/autoviz_agent/tools/prep.py"
Task: "Implement aggregate, compute_summary_stats, compute_correlations in src/autoviz_agent/tools/metrics.py"
Task: "Implement detect_anomalies, segment_metric, compute_distributions in src/autoviz_agent/tools/analysis.py"
Task: "Implement plotting tools and fixed styles in src/autoviz_agent/tools/visualization.py"

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational
3. Complete Phase 3: User Story 1
4. Stop and validate deterministic outputs and artifact persistence

### Incremental Delivery

1. Setup + Foundational → foundation ready
2. Add User Story 1 → validate artifacts and determinism (MVP)
3. Add User Story 2 → validate provenance inspection
4. Add User Story 3 → validate repair/clarification handling
5. Polish and demo assets
