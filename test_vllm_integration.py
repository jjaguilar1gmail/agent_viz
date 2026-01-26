"""Test vLLM integration with xgrammar2 structured outputs."""

import yaml
from pathlib import Path
from src.autoviz_agent.llm.factory import create_llm_client
from src.autoviz_agent.models.state import SchemaProfile, ColumnProfile

def test_vllm_connection():
    """Test basic vLLM server connectivity."""
    print("=" * 60)
    print("TEST 1: vLLM Server Connectivity")
    print("=" * 60)
    
    # Load config
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    
    model_config = config["models"]["qwen-1.5b-vllm"]
    model_config["backend"] = "vllm"
    
    # Create client
    try:
        client = create_llm_client(model_config)
        print(f"✓ Connected to vLLM server at {model_config['url']}")
        print(f"  Model: {model_config['model_name']}")
        print(f"  Grammar enabled: {model_config.get('use_grammar', True)}")
        return client
    except Exception as e:
        print(f"✗ Failed to connect: {e}")
        return None

def test_intent_classification(client):
    """Test intent classification with xgrammar2."""
    print("\n" + "=" * 60)
    print("TEST 2: Intent Classification (xgrammar2 structured output)")
    print("=" * 60)
    
    # Create mock schema
    schema = SchemaProfile(
        columns=[
            ColumnProfile(
                name="date",
                dtype="datetime",
                missing_rate=0.0,
                cardinality=100,
                roles=["temporal"]
            ),
            ColumnProfile(
                name="revenue",
                dtype="float",
                missing_rate=0.0,
                cardinality=95,
                roles=["metric"]
            ),
            ColumnProfile(
                name="region",
                dtype="string",
                missing_rate=0.0,
                cardinality=4,
                roles=["dimension"]
            )
        ],
        row_count=100,
        data_shape="time_series"
    )
    
    question = "analyze revenue trends over time by region"
    
    try:
        print(f"Question: {question}")
        print("Sending request to vLLM with JSON schema constraint...")
        
        intent = client.classify_intent(question, schema)
        
        print(f"\n✓ Intent classified successfully!")
        print(f"  Label: {intent.label.value}")
        print(f"  Confidence: {intent.confidence:.2f}")
        print(f"  Top intents: {intent.top_intents}")
        
        # Verify it's valid structured output
        assert hasattr(intent, 'label'), "Intent missing 'label' field"
        assert hasattr(intent, 'confidence'), "Intent missing 'confidence' field"
        print("\n✓ Structured output validation passed!")
        
        return True
    except Exception as e:
        print(f"\n✗ Intent classification failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_plan_adaptation(client):
    """Test plan adaptation with xgrammar2."""
    print("\n" + "=" * 60)
    print("TEST 3: Plan Adaptation (xgrammar2 structured output)")
    print("=" * 60)
    
    # Create mock schema and intent
    from src.autoviz_agent.models.state import Intent, IntentLabel
    
    schema = SchemaProfile(
        columns=[
            ColumnProfile(
                name="timestamp",
                dtype="datetime",
                missing_rate=0.0,
                cardinality=100,
                roles=["temporal"]
            ),
            ColumnProfile(
                name="sales",
                dtype="float",
                missing_rate=0.0,
                cardinality=95,
                roles=["metric"]
            )
        ],
        row_count=100,
        data_shape="time_series"
    )
    
    intent = Intent(
        label=IntentLabel.TIME_SERIES_INVESTIGATION,
        confidence=0.9,
        top_intents=[{"time_series_investigation": 0.9}]
    )
    
    template_plan = {
        "intent": "general_eda",
        "steps": [
            {
                "step_id": "load",
                "tool": "load_dataset",
                "description": "Load data",
                "params": {"path": "$input"}
            },
            {
                "step_id": "stats",
                "tool": "compute_summary_stats",
                "description": "Compute statistics",
                "params": {"df": "$dataframe"}
            }
        ]
    }
    
    question = "show me sales trends over time"
    
    try:
        print(f"Question: {question}")
        print(f"Template steps: {len(template_plan['steps'])}")
        print("Sending adaptation request to vLLM with JSON schema constraint...")
        
        adapted = client.adapt_plan(template_plan, schema, intent, question)
        
        print(f"\n✓ Plan adapted successfully!")
        print(f"  Rationale: {adapted.get('adaptation_rationale', 'N/A')}")
        print(f"  Changes applied: {adapted.get('changes_applied', 0)}")
        print(f"  Final steps: {len(adapted.get('steps', []))}")
        
        # Verify structured output
        assert 'steps' in adapted, "Adapted plan missing 'steps' field"
        print("\n✓ Structured output validation passed!")
        
        return True
    except Exception as e:
        print(f"\n✗ Plan adaptation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("vLLM + xgrammar2 Integration Tests")
    print("=" * 60 + "\n")
    
    # Test 1: Connection
    client = test_vllm_connection()
    if not client:
        print("\n✗ Cannot proceed without vLLM connection")
        return False
    
    # Test 2: Intent classification
    test2_passed = test_intent_classification(client)
    
    # Test 3: Plan adaptation
    test3_passed = test_plan_adaptation(client)
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"✓ Server connection: PASS")
    print(f"{'✓' if test2_passed else '✗'} Intent classification: {'PASS' if test2_passed else 'FAIL'}")
    print(f"{'✓' if test3_passed else '✗'} Plan adaptation: {'PASS' if test3_passed else 'FAIL'}")
    
    all_passed = test2_passed and test3_passed
    print(f"\n{'='*60}")
    print(f"{'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
    print(f"{'='*60}\n")
    
    return all_passed

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
