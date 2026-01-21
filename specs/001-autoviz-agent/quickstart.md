# Quickstart: AutoViz Agent

## Prerequisites

- Python 3.10+
- Local model weights (quantized) configured for llama.cpp
- Offline environment (no external network access required)

## Install

1. Create a virtual environment.
2. Install dependencies (pandas, numpy, matplotlib, seaborn, pydantic,
   jsonschema, langgraph, pyyaml).

## Configure

1. Set the model configuration YAML to point at local quantized weights.
2. Verify the plan template library is available on disk.
3. Configure output directory for run artifacts.

## Run

1. Provide a CSV path (or DataFrame in Python) and a natural-language question.
2. Start a run and observe the execution log for progress.
3. Review artifacts: plan_template.json, plan_adapted.json, plan_diff.md,
   tool_calls.json, execution_log.json, report.md, and charts/.

## Verify

- Re-run the same input and confirm deterministic outputs.
- Confirm tool calls are schema-validated and unknown tools are rejected.
- Inspect plan diff and rationale for clear mutation evidence.
