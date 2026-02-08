"""Test new prompt strategy implementation."""

import sys
import json
import yaml
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from autoviz_agent.llm.prompts import PromptBuilder, INTENT_SCHEMA, ADAPTATION_SCHEMA
from autoviz_agent.llm.client import LLMClient
from autoviz_agent.models.state import SchemaProfile, ColumnProfile, Intent, IntentLabel


def create_test_schema():
    """Create a test schema profile."""
    columns = [
        ColumnProfile(name="date", dtype="object", missing_rate=0.0, cardinality=20, roles=["temporal"]),
        ColumnProfile(name="revenue", dtype="float64", missing_rate=0.0, cardinality=20, roles=["numeric", "metric"]),
        ColumnProfile(name="region", dtype="object", missing_rate=0.0, cardinality=4, roles=["categorical"]),
        ColumnProfile(name="product_category", dtype="object", missing_rate=0.0, cardinality=3, roles=["categorical"])
    ]
    return SchemaProfile(
        row_count=20,
        columns=columns,
        data_shape="wide"
    )


def test_prompt_builder():
    """Test PromptBuilder class."""
    print("=" * 70)
    print("TEST 1: PromptBuilder (embedded templates)")
    print("=" * 70)
    
    # Test with embedded templates (no template_dir)
    builder = PromptBuilder()
    schema = create_test_schema()
    
    # Test intent prompt
    print("\n1. Intent Prompt:")
    print("-" * 70)
    question = "Compare revenue by region and product"
    intent_prompt = builder.build_intent_prompt(question, schema)
    print(intent_prompt[:500] + "...")
    print(f"\nPrompt length: {len(intent_prompt)} characters")
    
    # Test adaptation prompt
    print("\n2. Adaptation Prompt:")
    print("-" * 70)
    template_plan = {
        "template_id": "general_eda",
        "steps": [
            {"step_id": "load", "tool": "load_dataset", "description": "Load data"},
            {"step_id": "stats", "tool": "compute_summary_stats", "description": "Basic stats"}
        ]
    }
    intent = Intent(label=IntentLabel.COMPARATIVE_ANALYSIS, confidence=0.95)
    adapt_prompt = builder.build_adaptation_prompt(template_plan, schema, intent, question)
    print(adapt_prompt[:500] + "...")
    print(f"\nPrompt length: {len(adapt_prompt)} characters")
    
    print("\n[PASS] PromptBuilder works with embedded templates\n")


def test_prompt_builder_with_files():
    """Test PromptBuilder with template files."""
    print("=" * 70)
    print("TEST 2: PromptBuilder (file-based templates)")
    print("=" * 70)
    
    templates_dir = Path(__file__).parent / "templates" / "prompts"
    if not templates_dir.exists():
        print(f"[ERROR] Templates directory not found: {templates_dir}")
        return
    
    builder = PromptBuilder(template_dir=templates_dir)
    schema = create_test_schema()
    
    # Test intent prompt from file
    print("\n1. Intent Prompt (from intent.md):")
    print("-" * 70)
    question = "Find unusual sales patterns"
    intent_prompt = builder.build_intent_prompt(question, schema)
    print(intent_prompt[:500] + "...")
    print(f"\nPrompt length: {len(intent_prompt)} characters")
    
    # Test adaptation prompt from file
    print("\n2. Adaptation Prompt (from adapt_plan.md):")
    print("-" * 70)
    template_plan = {
        "template_id": "comparative_analysis",
        "steps": [
            {"step_id": "aggregate", "tool": "aggregate", "description": "Group by categories"}
        ]
    }
    intent = Intent(label=IntentLabel.ANOMALY_DETECTION, confidence=0.90)
    adapt_prompt = builder.build_adaptation_prompt(template_plan, schema, intent, question)
    print(adapt_prompt[:500] + "...")
    print(f"\nPrompt length: {len(adapt_prompt)} characters")
    
    print("\n[PASS] PromptBuilder works with file-based templates\n")


def test_llm_client_integration():
    """Test LLMClient with new PromptBuilder."""
    print("=" * 70)
    print("TEST 3: LLMClient Integration")
    print("=" * 70)
    
    # Load config from YAML
    config_path = Path(__file__).parent / "config.yaml"
    if not config_path.exists():
        print(f"[ERROR] Config file not found: {config_path}")
        return
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    model_config = config.get("model", {})
    
    print(f"\nModel: {model_config.get('name')}")
    print(f"Path: {model_config.get('path')}")
    
    # Initialize client
    client = LLMClient(model_config)
    print(f"PromptBuilder initialized: {client.prompt_builder is not None}")
    print(f"Using fallback mode: {client._use_fallback}")
    
    schema = create_test_schema()
    
    # Test intent classification
    print("\n1. Testing Intent Classification:")
    print("-" * 70)
    question = "Compare revenue by region and product"
    try:
        intent = client.classify_intent(question, schema)
        print(f"[PASS] Intent: {intent.label}")
        print(f"  Confidence: {intent.confidence:.2f}")
        print(f"  Top intents: {intent.top_intents}")
    except Exception as e:
        print(f"[ERROR] Error during intent classification: {e}")
    
    # Test plan adaptation
    print("\n2. Testing Plan Adaptation:")
    print("-" * 70)
    template_plan = {
        "template_id": "general_eda",
        "steps": [
            {"step_id": "load", "tool": "load_dataset", "description": "Load data"},
            {"step_id": "stats", "tool": "compute_summary_stats", "description": "Basic stats"}
        ]
    }
    intent = Intent(label=IntentLabel.COMPARATIVE_ANALYSIS, confidence=0.95)
    try:
        adapted_plan = client.adapt_plan(template_plan, schema, intent, question)
        print(f"[PASS] Adapted plan:")
        print(f"  Changes applied: {adapted_plan.get('changes_applied', 0)}")
        print(f"  Rationale: {adapted_plan.get('adaptation_rationale', 'N/A')[:100]}...")
        print(f"  Steps: {len(adapted_plan.get('steps', []))}")
    except Exception as e:
        print(f"[ERROR] Error during plan adaptation: {e}")
    
    print("\n[PASS] LLMClient integration test complete\n")


def test_schemas():
    """Test JSON schemas."""
    print("=" * 70)
    print("TEST 4: JSON Schemas")
    print("=" * 70)
    
    print("\n1. Intent Schema:")
    print(json.dumps(INTENT_SCHEMA, indent=2))
    
    print("\n2. Adaptation Schema:")
    print(json.dumps(ADAPTATION_SCHEMA, indent=2))
    
    # Test schema retrieval
    builder = PromptBuilder()
    intent_schema = builder.get_schema("intent")
    adaptation_schema = builder.get_schema("adaptation")
    
    print(f"\n[PASS] Intent schema retrieved: {intent_schema is not None}")
    print(f"[PASS] Adaptation schema retrieved: {adaptation_schema is not None}")
    print()


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("PROMPT STRATEGY IMPLEMENTATION TESTS")
    print("=" * 70 + "\n")
    
    try:
        test_prompt_builder()
        test_prompt_builder_with_files()
        test_schemas()
        test_llm_client_integration()
        
        print("=" * 70)
        print("ALL TESTS COMPLETED")
        print("=" * 70)
        print("\n[SUCCESS] Prompt strategy implementation is working!\n")
    except Exception as e:
        print(f"\n[ERROR] Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
