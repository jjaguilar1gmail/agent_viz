"""Test general_eda parameter fixes."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from autoviz_agent.llm.client import LLMClient
from autoviz_agent.tools.schema import infer_schema
from autoviz_agent.tools.data_io import load_dataset
from autoviz_agent.io.artifacts import ArtifactManager
import json

# Load the general_eda template
with open("templates/general_eda.json") as f:
    template = json.load(f)

# Load dataset and infer schema
df = load_dataset("examples/december_revenue/december_revenue.csv")
schema = infer_schema(df)

# Create artifact manager
artifact_mgr = ArtifactManager("test_run", Path("outputs"))

# Create LLM client (will use fallback)
config = {"model_path": "fake", "use_llm": False}
client = LLMClient(config)

# Generate tool calls
tool_calls = client.generate_tool_calls(template, schema, artifact_mgr)

print("Generated tool calls:")
for tc in tool_calls:
    print(f"\n{tc['sequence']}. {tc['tool']}")
    print(f"   Args: {tc['args']}")
    
    # Check for issues
    if tc['tool'] == 'compute_distributions':
        if 'column' not in tc['args']:
            print("   ❌ ERROR: Missing 'column' parameter")
        else:
            print(f"   ✅ Has 'column': {tc['args']['column']}")
    
    if tc['tool'] == 'plot_histogram':
        if 'max_columns' in tc['args']:
            print("   ❌ ERROR: Has invalid 'max_columns' parameter")
        else:
            print("   ✅ No 'max_columns' parameter")
        if 'column' not in tc['args']:
            print("   ❌ ERROR: Missing 'column' parameter")
        else:
            print(f"   ✅ Has 'column': {tc['args']['column']}")
    
    if tc['tool'] == 'plot_heatmap':
        if 'df' in tc['args']:
            print("   ❌ ERROR: Has 'df' instead of 'data' parameter")
        elif 'data' not in tc['args']:
            print("   ❌ ERROR: Missing 'data' parameter")
        else:
            print("   ✅ Has 'data' parameter")
        
        if 'annotation' in tc['args']:
            print("   ❌ ERROR: Has 'annotation' instead of 'annot' parameter")
        elif 'annot' not in tc['args']:
            print("   ⚠️  WARNING: Missing 'annot' parameter (optional)")
        else:
            print(f"   ✅ Has 'annot': {tc['args']['annot']}")
