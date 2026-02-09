# Tool Selection Prompt

Select the minimal set of tools that covers all capability targets.

## Capability Targets

{capability_targets}

## Candidate Tools (choose only from this list)

{candidate_tools}

## Tool Catalog

{tool_catalog}

## Rules

1. Only choose tools from the candidate list.
2. Choose the smallest set that covers all targets.
3. Prefer tools whose capabilities directly match the targets.

## Response Format

RESPONSE FORMAT (JSON only, no preamble):
{{"selected_tools": ["tool_a", "tool_b"], "rationale": "<overall explanation>"}}
