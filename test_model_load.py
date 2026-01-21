"""Test model loading with ctransformers."""

from ctransformers import AutoModelForCausalLM
from pathlib import Path

model_path = Path("models/Phi-3.5-mini-instruct-Q4_K_M.gguf")

print(f"Model file exists: {model_path.exists()}")
print(f"Model size: {model_path.stat().st_size / (1024**3):.2f} GB")
print()

# Try different model types
model_types = ['llama', 'phi', 'gpt2', 'gptj', 'gptneox']

for model_type in model_types:
    print(f"Trying model_type='{model_type}'...")
    try:
        llm = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            model_type=model_type,
            context_length=2048,
        )
        print(f"✓ SUCCESS with model_type='{model_type}'")
        
        # Test generation
        response = llm("Hello, this is a test.", max_new_tokens=20)
        print(f"  Test response: {response[:100]}...")
        break
    except Exception as e:
        print(f"✗ Failed: {str(e)[:100]}")
        continue

print("\nIf all failed, the GGUF file may be incompatible with ctransformers.")
print("Consider using llama-cpp-python or transformers library instead.")
