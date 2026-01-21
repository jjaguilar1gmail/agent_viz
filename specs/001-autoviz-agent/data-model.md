# Data Model: AutoViz Agent

## Entities

### RunState
- **Fields**: run_id, user_request, dataset_source, inferred_schema, intent,
  template_id, template_plan, adapted_plan, plan_diff, tool_calls,
  execution_log, artifacts, status, created_at, updated_at
- **Relationships**: Owns `ToolCall`, `ExecutionLog`, `Artifact`

### UserRequest
- **Fields**: question, output_goal, constraints
- **Relationships**: referenced by `RunState`

### DatasetSource
- **Fields**: source_type (csv|dataframe), path (optional), checksum (optional)
- **Relationships**: referenced by `RunState`

### SchemaProfile
- **Fields**: columns[{name, dtype, missing_rate, cardinality, roles}]
- **Relationships**: referenced by `RunState`

### Intent
- **Fields**: label (enum), confidence (0–1), top_intents
- **Relationships**: referenced by `RunState`

### PlanTemplate
- **Fields**: template_id, version, intents, data_shape, requires, prefers,
  supports, steps[]
- **Relationships**: selected by `RunState`

### PlanStep
- **Fields**: step_id, tool, description, params
- **Relationships**: part of `PlanTemplate` or `AdaptedPlan`

### AdaptedPlan
- **Fields**: plan_id, base_template_id, steps[], adaptations
- **Relationships**: produced from `PlanTemplate`

### PlanDiff
- **Fields**: added_steps, removed_steps, modified_steps, rationale
- **Relationships**: compares `PlanTemplate` and `AdaptedPlan`

### ToolCall
- **Fields**: tool_name, args, schema_version, sequence, status
- **Relationships**: executed in `ExecutionLog`

### ToolResult
- **Fields**: tool_name, outputs, duration_ms, warnings, errors
- **Relationships**: recorded by `ExecutionLog`

### ExecutionLog
- **Fields**: entries[{sequence, tool_call, tool_result, timestamp}]
- **Relationships**: belongs to `RunState`

### Artifact
- **Fields**: artifact_type (chart|report|plan|log), path, mime_type
- **Relationships**: belongs to `RunState`

### Report
- **Fields**: sections, summary, references
- **Relationships**: produced by `ExecutionLog` and `Artifact`

## Validation Rules

- `Intent.label` MUST be one of: general_eda, time_series_investigation,
  segmentation_drivers, anomaly_detection, comparative_analysis.
- `PlanTemplate` MUST conform to the required JSON schema.
- `PlanStep.tool` MUST exist in the tool registry.
- `ToolCall` MUST validate against the registered schema with no extra fields.
- `RunState` MUST include plan artifacts and execution logs before completion.

## State Transitions

- `initialized` → `schema_inferred` → `intent_classified` → `template_selected`
  → `plan_adapted` → `tool_calls_compiled` → `tools_executed`
  → `summarized` → `completed`
- Failure transitions: any state → `repair_or_clarify` → (back to prior step) or
  → `failed`
