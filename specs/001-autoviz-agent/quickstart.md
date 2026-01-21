# Quickstart: AutoViz Agent

## Prerequisites

- Python 3.10+
- Virtual environment recommended
- Local model weights (quantized) configured for llama.cpp (optional - placeholder LLM included)
- Offline environment (no external network access required)

## Install

```bash
# Clone repository
git clone <repository-url>
cd agent_viz

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install package
pip install -e .

# Verify installation
autoviz --version
```

## Configure

The default configuration works out of the box. To customize:

1. **Model Configuration**: Edit `src/autoviz_agent/config/settings.py` to point at local quantized weights
2. **Templates**: Plan templates are in `src/autoviz_agent/templates/`
3. **Output Directory**: Default is `outputs/`, configure via CLI `--output-dir`

## Run

### Basic Usage

```bash
# Run analysis on a CSV file
autoviz run data.csv "What are the trends over time?"
```

### Advanced Options

```bash
# Specify output directory
autoviz run data.csv "Compare sales by region" --output-dir ./results

# Set random seed for reproducibility
autoviz run data.csv "Detect anomalies" --seed 42

# Set log level
autoviz run data.csv "Question" --log-level DEBUG
```

### Example: December Revenue Analysis

```bash
# Run the demo
autoviz run examples/december_revenue/december_revenue.csv \
  "What are the revenue trends in December?" \
  --output-dir examples/december_revenue/output \
  --seed 42
```

## Review Artifacts

After running, check the output directory for:

1. **report.md**: Main analysis report with insights and chart references
2. **charts/**: Visualization files (PNG/SVG)
3. **plans/plan_template.json**: Original template plan selected
4. **plans/plan_adapted.json**: Plan adapted to your dataset
5. **plans/plan_diff.md**: Diff showing what changed and why
6. **logs/tool_calls.json**: Complete tool call sequence
7. **logs/execution_log.json**: Detailed execution timeline

### Example Report Structure

```markdown
# Analysis Report

## Summary
Key insights from the analysis...

## Visualizations
![Revenue Trend](charts/revenue_trend.png)
![Revenue by Region](charts/revenue_by_region.png)

## Plan Provenance
- **Template Plan**: [plan_template.json](plans/plan_template.json)
- **Adapted Plan**: [plan_adapted.json](plans/plan_adapted.json)
- **Plan Diff**: [plan_diff.md](plans/plan_diff.md)
- **Tool Calls**: [tool_calls.json](logs/tool_calls.json)
- **Execution Log**: [execution_log.json](logs/execution_log.json)
```

## Verify Determinism

Run the same analysis twice and verify outputs are identical:

```bash
# Run 1
autoviz run data.csv "Question" --seed 42 --output-dir run1

# Run 2
autoviz run data.csv "Question" --seed 42 --output-dir run2

# Verify with script
python scripts/verify_determinism.py run1/<run_id> run2/<run_id>
```

Expected output:
```
✅ Runs are identical (deterministic)
```

## Python API

Use the Python API for programmatic access:

```python
from pathlib import Path
from autoviz_agent.config.settings import DEFAULT_SETTINGS
from autoviz_agent.runtime.runner import RunRunner

# Create runner
runner = RunRunner(DEFAULT_SETTINGS)

# Execute analysis
result = runner.run(
    dataset_path=Path("data.csv"),
    question="What are the key insights?",
    output_dir=Path("outputs"),
    seed=42
)

# Check results
print(f"Run ID: {result.run_id}")
print(f"Status: {result.status}")
print(f"Artifacts: {len(result.artifacts)}")

# Access metadata
metadata = runner.get_run_metadata(result.run_id)
print(f"Question: {metadata.get('question')}")
```

## Understanding Intents

The system classifies questions into intents:

- **general_eda**: Exploratory data analysis
- **time_series_investigation**: Temporal trends and patterns
- **segmentation_drivers**: What drives differences between groups
- **anomaly_detection**: Unusual values or patterns
- **comparative_analysis**: Compare groups or categories

The intent determines which plan template is selected.

## Troubleshooting

### No templates found

If you see "Templates directory not found", the CLI will auto-create placeholder templates. For production, add templates to `src/autoviz_agent/templates/`.

### Dataset not found

Ensure the CSV path is correct and the file exists.

### Validation errors

If tool calls fail validation, check the execution log for details. The system will reject unknown tools and invalid parameters.

## Next Steps

1. **Create Custom Templates**: Add new plan templates to `src/autoviz_agent/templates/`
2. **Add Tools**: Register new analysis tools in `src/autoviz_agent/runtime/executor.py`
3. **Customize LLM**: Update `src/autoviz_agent/llm/client.py` with your model
4. **Run Tests**: `pytest` to verify functionality

See [README.md](../../README.md) for full architecture documentation.
