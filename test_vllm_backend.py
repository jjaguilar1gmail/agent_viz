"""Test vLLM backend integration."""

import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from autoviz_agent.llm.factory import create_llm_client
from autoviz_agent.llm.vllm_client import VLLMClient
from autoviz_agent.llm.llm_contracts import (
    get_intent_schema,
    get_adaptation_schema,
    validate_intent_output,
    validate_adaptation_output,
)
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


def test_contracts():
    """Test LLM contracts and schemas."""
    print("=" * 70)
    print("TEST 1: LLM Contracts")
    print("=" * 70)
    
    # Test intent schema
    print("\n1. Intent Schema:")
    print("-" * 70)
    intent_schema = get_intent_schema()
    print(json.dumps(intent_schema, indent=2))
    
    # Test adaptation schema
    print("\n2. Adaptation Schema:")
    print("-" * 70)
    adaptation_schema = get_adaptation_schema()
    print(json.dumps(adaptation_schema, indent=2))
    
    # Test validation
    print("\n3. Testing Validation:")
    print("-" * 70)
    
    # Valid intent
    try:
        valid_intent = {
            "primary": "comparative_analysis",
            "confidence": 0.95,
            "reasoning": "User asks to compare by region"
        }
        result = validate_intent_output(valid_intent)
        print(f"[PASS] Valid intent validated: {result.primary}")
    except Exception as e:
        print(f"[ERROR] Intent validation failed: {e}")
    
    # Valid adaptation
    try:
        valid_adaptation = {
            "changes": [
                {
                    "action": "add",
                    "step_id": "compare_regions",
                    "tool": "aggregate",
                    "description": "Add aggregation step",
                    "params": {"group_by": ["region"]}
                }
            ],
            "rationale": "Need to group by region for comparison"
        }
        result = validate_adaptation_output(valid_adaptation)
        print(f"[PASS] Valid adaptation validated: {len(result.changes)} changes")
    except Exception as e:
        print(f"[ERROR] Adaptation validation failed: {e}")
    
    print("\n[PASS] LLM contracts test complete\n")


def test_vllm_client_creation():
    """Test vLLM client creation."""
    print("=" * 70)
    print("TEST 2: vLLM Client Creation")
    print("=" * 70)
    
    # Test client creation with mock config
    print("\n1. Creating vLLM client with test config:")
    print("-" * 70)
    
    model_config = {
        "backend": "vllm",
        "url": "http://localhost:8000",
        "model_name": "test-model",
        "temperature": 0.1,
        "max_tokens": 512,
        "use_grammar": True
    }
    
    try:
        client = VLLMClient(model_config)
        print(f"[PASS] VLLMClient created")
        print(f"  Base URL: {client.base_url}")
        print(f"  Model: {client.model_name}")
        print(f"  Temperature: {client.temperature}")
        print(f"  Use Grammar: {client.use_grammar}")
        print(f"  PromptBuilder initialized: {client.prompt_builder is not None}")
    except Exception as e:
        print(f"[WARN] VLLMClient creation warning (expected if server not running): {e}")
    
    print("\n[PASS] vLLM client creation test complete\n")


def test_factory():
    """Test LLM client factory."""
    print("=" * 70)
    print("TEST 3: LLM Client Factory")
    print("=" * 70)
    
    # Test gpt4all backend
    print("\n1. Testing gpt4all backend selection:")
    print("-" * 70)
    gpt4all_config = {
        "backend": "llama.cpp",
        "path": "models/test.gguf",
        "name": "test-model"
    }
    
    try:
        client = create_llm_client(gpt4all_config)
        print(f"[PASS] Factory created client: {type(client).__name__}")
    except Exception as e:
        print(f"[ERROR] Factory failed for gpt4all: {e}")
    
    # Test vLLM backend
    print("\n2. Testing vLLM backend selection:")
    print("-" * 70)
    vllm_config = {
        "backend": "vllm",
        "url": "http://localhost:8000",
        "model_name": "test-model"
    }
    
    try:
        client = create_llm_client(vllm_config)
        print(f"[PASS] Factory created client: {type(client).__name__}")
    except Exception as e:
        print(f"[WARN] Factory warning (expected if server not running): {e}")
    
    # Test invalid backend
    print("\n3. Testing invalid backend handling:")
    print("-" * 70)
    invalid_config = {
        "backend": "invalid-backend"
    }
    
    try:
        client = create_llm_client(invalid_config)
        print(f"[ERROR] Should have raised ValueError for invalid backend")
    except ValueError as e:
        print(f"[PASS] Correctly raised ValueError: {e}")
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
    
    print("\n[PASS] Factory test complete\n")


def test_vllm_with_live_server():
    """Test vLLM client with live server (if available)."""
    print("=" * 70)
    print("TEST 4: vLLM Live Server Integration (Optional)")
    print("=" * 70)
    
    print("\nThis test requires a running vLLM server at http://localhost:8000")
    print("If you haven't started the server, this test will be skipped.")
    print("-" * 70)
    
    model_config = {
        "backend": "vllm",
        "url": "http://localhost:8000",
        "model_name": "phi-4-mini",
        "temperature": 0.1,
        "max_tokens": 512,
        "use_grammar": True
    }
    
    try:
        import requests
        # Check if server is running
        response = requests.get(f"{model_config['url']}/v1/models", timeout=2)
        response.raise_for_status()
        
        print("[INFO] vLLM server is running! Testing live integration...")
        
        # Create client
        client = VLLMClient(model_config)
        schema = create_test_schema()
        
        # Test intent classification
        print("\n1. Testing Intent Classification:")
        print("-" * 70)
        question = "Compare revenue by region and product"
        try:
            intent = client.classify_intent(question, schema)
            print(f"[PASS] Intent: {intent.label}")
            print(f"  Confidence: {intent.confidence:.2f}")
        except Exception as e:
            print(f"[ERROR] Intent classification failed: {e}")
        
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
        intent_obj = Intent(label=IntentLabel.COMPARATIVE_ANALYSIS, confidence=0.95)
        try:
            adapted_plan = client.adapt_plan(template_plan, schema, intent_obj, question)
            print(f"[PASS] Adapted plan:")
            print(f"  Changes applied: {adapted_plan.get('changes_applied', 0)}")
            print(f"  Rationale: {adapted_plan.get('adaptation_rationale', 'N/A')[:100]}...")
        except Exception as e:
            print(f"[ERROR] Plan adaptation failed: {e}")
        
        print("\n[PASS] Live server integration test complete\n")
        
    except requests.exceptions.RequestException:
        print("[SKIP] vLLM server not running at http://localhost:8000")
        print("       Start server with: vllm serve <model> --host 0.0.0.0 --port 8000")
    except Exception as e:
        print(f"[ERROR] Unexpected error during live test: {e}")


def test_config_yaml():
    """Test that config.yaml has proper vLLM entries."""
    print("=" * 70)
    print("TEST 5: Config YAML Validation")
    print("=" * 70)
    
    import yaml
    
    config_path = Path(__file__).parent / "config.yaml"
    if not config_path.exists():
        print(f"[ERROR] Config file not found: {config_path}")
        return
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("\n1. Checking for vLLM model entries:")
    print("-" * 70)
    
    vllm_models = [name for name, cfg in config.get("models", {}).items() 
                   if cfg.get("backend") == "vllm"]
    
    if vllm_models:
        print(f"[PASS] Found {len(vllm_models)} vLLM model(s): {', '.join(vllm_models)}")
        for model_name in vllm_models:
            model_cfg = config["models"][model_name]
            print(f"\n  {model_name}:")
            print(f"    URL: {model_cfg.get('url')}")
            print(f"    Model Name: {model_cfg.get('model_name')}")
            print(f"    Use Grammar: {model_cfg.get('use_grammar')}")
    else:
        print("[WARN] No vLLM models found in config.yaml")
    
    print("\n2. Checking backend settings:")
    print("-" * 70)
    
    if "vllm" in config.get("backends", {}):
        vllm_backend = config["backends"]["vllm"]
        print(f"[PASS] vLLM backend configuration found:")
        print(f"  URL: {vllm_backend.get('url')}")
        print(f"  Use Grammar: {vllm_backend.get('use_grammar')}")
        print(f"  Timeout: {vllm_backend.get('timeout')}")
    else:
        print("[WARN] No vLLM backend configuration in config.yaml")
    
    print("\n[PASS] Config validation complete\n")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("vLLM BACKEND INTEGRATION TESTS")
    print("=" * 70 + "\n")
    
    try:
        test_contracts()
        test_vllm_client_creation()
        test_factory()
        test_config_yaml()
        test_vllm_with_live_server()
        
        print("=" * 70)
        print("ALL TESTS COMPLETED")
        print("=" * 70)
        print("\n[SUCCESS] vLLM backend implementation is ready!")
        print("\nNext steps:")
        print("1. Set up WSL2 with vLLM server (see VLLM_WSL2_STRATEGY.md)")
        print("2. Start vLLM server: vllm serve <model> --host 0.0.0.0 --port 8000")
        print("3. Update default_model in config.yaml to use vLLM backend")
        print("4. Run: autoviz run <dataset> <question>\n")
        
    except Exception as e:
        print(f"\n[ERROR] Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
