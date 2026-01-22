# AutoViz Agent

**Deterministic data visualization and analysis pipeline**

## Overview

AutoViz Agent is an offline, deterministic analysis pipeline that accepts datasets and questions, infers schema and intent, retrieves and adapts curated plan templates, and executes registered tools to produce charts, reports, and full provenance artifacts.

## Key Features

- **Deterministic Execution**: Same input always produces same output (fixed seeds, stable sorting, reproducible charts)
- **Offline Operation**: Runs completely offline with quantized local models (llama.cpp)
- **Plan Provenance**: Full traceability with template plan, adapted plan, diff, tool calls, and execution log
- **Bounded LLM Role**: LLM limited to intent classification and plan adaptation; all analysis is deterministic code
- **Schema-Validated Tool Calls**: Strict validation with automatic rejection of unknown tools and invalid parameters

## Architecture

### Core Components

```
src/autoviz_agent/
├── cli/              # Command-line interface
├── config/           # Configuration and settings
├── graph/            # LangGraph pipeline orchestration
├── io/               # I/O operations and artifact management
├── llm/              # Bounded LLM client (intent + plan adaptation)
├── models/           # Core data models (state, errors)
├── planning/         # Plan retrieval, adaptation, diffing
├── registry/         # Tool registry and validation
├── reporting/        # Report generation and execution logs
├── runtime/          # Execution and determinism helpers
├── templates/        # Plan templates (JSON)
├── tools/            # Analysis tools (data I/O, schema, prep, metrics, analysis, viz)
└── utils/            # Utilities (logging, etc.)
```

### Execution Flow

1. **Initialize**: Load dataset and user question
2. **Infer Schema**: Detect column types, roles, data shape
3. **Classify Intent**: Use LLM to classify intent (general_eda, time_series, etc.)
4. **Select Template**: Retrieve best matching plan template using deterministic scoring
5. **Adapt Plan**: Use LLM to adapt template to specific dataset
6. **Compile Tool Calls**: Generate tool call sequence from plan
7. **Execute Tools**: Run tools with schema validation
8. **Summarize**: Generate report with charts and provenance artifacts

### Graph Pipeline

The execution pipeline is built with LangGraph and uses explicit node boundaries:

```
initialize → infer_schema → classify_intent → select_template
    → adapt_plan → compile_tool_calls → execute_tools → summarize → complete
```

Error handling routes to `repair_or_clarify` or `error` nodes with full logging.

### Plan Templates

Templates are JSON files in `src/autoviz_agent/templates/` that define:

- **Intents**: Which user intents this template supports
- **Data Shape**: Required data shapes (wide, long, time_series)
- **Requirements**: Hard filters (min rows, columns, types)
- **Preferences**: Soft scoring factors (has datetime, categorical, numeric)
- **Steps**: Ordered list of tool calls with parameters

Example template structure:

```json
{
  "template_id": "general_eda_v1",
  "version": "1.0.0",
  "intents": ["general_eda"],
  "data_shape": ["wide", "long", "unknown"],
  "requires": {"min_rows": 10, "min_columns": 2},
  "prefers": {"has_numeric": true},
  "steps": [
    {
      "step_id": "load_data",
      "tool": "load_dataset",
      "description": "Load dataset",
      "params": {"path": "$dataset_path"}
    }
  ]
}
```

### Tool Registry

**New in v2.0**: The tool registry now uses a **decorator-based auto-registration** system.

All tools are automatically registered using the `@tool` decorator:

```python
from autoviz_agent.registry.tools import tool

@tool(description="Load dataset from CSV file")
def load_dataset(path: str, encoding: str = "utf-8") -> pd.DataFrame:
    """Load dataset from file."""
    ...
```

The decorator extracts metadata from function signatures and docstrings:
- Parameter names, types, defaults, and descriptions
- Return type from type hints
- Required vs optional parameters

Tool calls are validated against registered schemas before execution. The registry provides:
- `TOOL_REGISTRY.list_tools()`: List all registered tools
- `TOOL_REGISTRY.get_tool(name)`: Get tool function
- `TOOL_REGISTRY.get_schema(name)`: Get tool schema
- `TOOL_REGISTRY.export_schema()`: Export JSON schema for LLM

**Parameter Resolution**: Missing parameters are automatically filled using `ParamResolver`, which uses dataset schema to select appropriate columns and generate output paths.

For more details, see [TOOL_CALLING.md](TOOL_CALLING.md).

### Configuration

Settings are defined in `src/autoviz_agent/config/settings.py` with defaults:

- **Model**: Provider (llama.cpp), path, context size, temperature, seed
- **Runtime**: Output directory, templates directory, execution timeout, caching
- **Logging**: Log level

### Extension Points

1. **Add New Tools**: Register in `runtime/executor.py` with schema
2. **Add New Templates**: Create JSON file in `templates/` directory
3. **Customize LLM**: Update `llm/client.py` with your model provider
4. **Add Analysis Types**: Extend tools in `tools/` directory

## Installation

### Prerequisites

- Python 3.10+
- Virtual environment recommended

### Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install package
pip install -e .

# Install dev dependencies
pip install -e ".[dev]"
```

## Usage

### Basic Usage

```bash
# Run analysis
autoviz run data.csv "What are the trends over time?"

# Specify output directory
autoviz run data.csv "Compare sales by region" --output-dir ./my_results

# Set random seed
autoviz run data.csv "Detect anomalies" --seed 123
```

### Output Artifacts

Each run produces:

- `report.md`: Analysis report with charts and provenance
- `charts/`: PNG/SVG charts
- `plans/plan_template.json`: Original template plan
- `plans/plan_adapted.json`: Adapted plan
- `plans/plan_diff.md`: Diff with rationale
- `logs/tool_calls.json`: Tool call sequence
- `logs/execution_log.json`: Detailed execution log

### Python API

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

print(f"Run ID: {result.run_id}")
print(f"Status: {result.status}")
```

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=autoviz_agent

# Run specific test file
pytest tests/unit/test_schema.py
```

### Code Quality

```bash
# Format code
black src/ tests/

# Lint
ruff src/ tests/

# Type checking
mypy src/
```

## Determinism

To ensure deterministic execution:

1. **Fixed Seeds**: Random seeds set for Python, NumPy
2. **Stable Sorting**: All sorts use `kind='stable'`
3. **Matplotlib Agg Backend**: Non-interactive, deterministic rendering
4. **Ordered Operations**: Templates, tools, results always processed in sorted order

Verify determinism:

```bash
# Run twice with same input
autoviz run data.csv "Question" --seed 42 --output-dir run1
autoviz run data.csv "Question" --seed 42 --output-dir run2

# Compare artifacts (should be identical)
diff -r run1 run2
```

## Constitution Compliance

This implementation follows the AutoViz Agent constitution:

- ✅ Plans originate from curated library (no keyword matching)
- ✅ Plan retrieval uses bounded intent, hard filters, deterministic scoring
- ✅ Plan mutation recorded as diff with rationale
- ✅ LLM role limited to intent, template selection, plan adaptation
- ✅ Analysis executed as deterministic code
- ✅ Tool calls schema-validated; unknown tools rejected
- ✅ Offline execution with llama.cpp
- ✅ Full provenance artifacts (template, adapted plan, diff)

## License

MIT

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## Support

For issues or questions, please open an issue on GitHub.
