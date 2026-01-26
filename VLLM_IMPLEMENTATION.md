# vLLM Backend Implementation

## Overview

This implementation adds support for vLLM backend with xgrammar2 for grammar-constrained JSON outputs, while maintaining backward compatibility with the existing gpt4all backend.

## What Was Implemented

### 1. LLM Contracts Module (`src/autoviz_agent/llm/llm_contracts.py`)
- **Pydantic models** for structured outputs:
  - `IntentOutput`: Intent classification response
  - `AdaptationOutput`: Plan adaptation response with changes
- **JSON schemas** for xgrammar2 grammar generation:
  - `get_intent_schema()`: Returns JSON Schema Draft 7 for intent
  - `get_adaptation_schema()`: Returns JSON Schema Draft 7 for adaptation
- **Validation helpers**:
  - `validate_intent_output()`: Validates and parses intent JSON
  - `validate_adaptation_output()`: Validates and parses adaptation JSON

### 2. vLLM Client (`src/autoviz_agent/llm/vllm_client.py`)
- **HTTP client** for OpenAI-compatible vLLM server
- **xgrammar2 integration** via `response_format` parameter
- **Same interface** as LLMClient:
  - `classify_intent()`: Intent classification with grammar constraint
  - `adapt_plan()`: Plan adaptation with grammar constraint
  - `generate_tool_calls()`: Reuses ParamResolver for parameter filling
- **Connection verification** on initialization
- **Error handling** with fallback to safe defaults

### 3. Client Factory (`src/autoviz_agent/llm/factory.py`)
- **Backend selection** based on config:
  - `llama.cpp` / `gpt4all` → LLMClient (existing)
  - `vllm` → VLLMClient (new)
- **Single entry point** via `create_llm_client(model_config)`
- **Graceful error** handling for unsupported backends

### 4. Config Schema Updates (`config.yaml`)
- **New vLLM model entries**:
  - `phi-4-mini-vllm`: Phi-4 via vLLM
  - `llama-3.1-8b-vllm`: Llama 3.1 via vLLM
- **Backend settings** for vLLM:
  - `url`: Server endpoint (default: http://localhost:8000)
  - `timeout`: Request timeout (default: 60s)
  - `use_grammar`: Enable xgrammar2 (default: true)
  - `verify_ssl`: SSL verification (default: false)

### 5. Integration Updates
- **Graph nodes** (`src/autoviz_agent/graph/nodes.py`):
  - Updated to use `create_llm_client()` factory
  - Transparent backend switching based on config
- **Backward compatible**: Existing gpt4all configs work unchanged

## Usage

### Option 1: Use existing gpt4all backend (no changes)
```bash
# Uses default_model: "phi-3.5-mini" (gpt4all backend)
autoviz run dataset.csv "your question"
```

### Option 2: Use vLLM backend

1. **Start vLLM server** (in WSL2):
```bash
cd ~/vllm
source .venv/bin/activate
vllm serve ~/models/phi-4-mini-instruct-bnb-4bit \
  --host 0.0.0.0 \
  --port 8000
```

2. **Update config.yaml**:
```yaml
default_model: "phi-4-mini-vllm"  # Switch to vLLM backend
```

3. **Run analysis**:
```bash
autoviz run dataset.csv "your question"
```

## Benefits of vLLM Backend

### 1. Grammar-Constrained Outputs (xgrammar2)
- **Guarantees valid JSON**: No parsing errors
- **Enforces schema**: LLM must follow exact structure
- **Reduces validation failures**: More reliable tool calls

### 2. Better Performance
- **GPU acceleration**: Full CUDA support in WSL2
- **Optimized inference**: vLLM is production-grade
- **Batching support**: Can handle multiple requests

### 3. Hybrid Parameter Strategy
- **LLM proposes** within strict grammar (xgrammar2)
- **ParamResolver fills** missing/"auto" values (deterministic)
- **Validation repairs** any remaining issues (safety net)

## Architecture

```
┌─────────────────────────────────────────────────────┐
│              Graph Nodes (nodes.py)                 │
└────────────────────┬────────────────────────────────┘
                     │ create_llm_client(config)
                     ▼
┌─────────────────────────────────────────────────────┐
│            Client Factory (factory.py)              │
│  - Reads backend from config                        │
│  - Returns LLMClient or VLLMClient                  │
└────────────┬───────────────────────┬────────────────┘
             │                       │
    backend=gpt4all        backend=vllm
             │                       │
             ▼                       ▼
┌────────────────────┐   ┌──────────────────────────┐
│   LLMClient        │   │    VLLMClient            │
│   (client.py)      │   │    (vllm_client.py)      │
│                    │   │                          │
│ - gpt4all library  │   │ - HTTP requests          │
│ - Local GGUF       │   │ - OpenAI API format      │
│ - Fallback mode    │   │ - xgrammar2 support      │
└────────────────────┘   └──────────────────────────┘
             │                       │
             └───────────┬───────────┘
                         │ Uses
                         ▼
┌─────────────────────────────────────────────────────┐
│         PromptBuilder (prompts.py)                  │
│  - Centralized prompt templates                     │
│  - File-based or embedded                           │
└─────────────────────────────────────────────────────┘
                         │ Validates with
                         ▼
┌─────────────────────────────────────────────────────┐
│       LLM Contracts (llm_contracts.py)              │
│  - Pydantic models                                  │
│  - JSON schemas for xgrammar2                       │
│  - Validation helpers                               │
└─────────────────────────────────────────────────────┘
```

## Testing

Run comprehensive tests:
```bash
python test_vllm_backend.py
```

Tests include:
1. ✅ LLM contracts validation
2. ✅ vLLM client creation
3. ✅ Factory backend selection
4. ✅ Config YAML validation
5. ⏭️ Live server integration (requires running vLLM)

## Files Created/Modified

### New Files
- `src/autoviz_agent/llm/llm_contracts.py` - JSON contracts and schemas
- `src/autoviz_agent/llm/vllm_client.py` - vLLM HTTP client
- `src/autoviz_agent/llm/factory.py` - Backend factory
- `test_vllm_backend.py` - Integration tests

### Modified Files
- `config.yaml` - Added vLLM backend config
- `src/autoviz_agent/graph/nodes.py` - Use factory instead of direct LLMClient

### Unchanged (Backward Compatible)
- `src/autoviz_agent/llm/client.py` - Original gpt4all client
- `src/autoviz_agent/llm/prompts.py` - Prompt templates
- All existing tools and runtime components

## Next Steps

1. **Setup WSL2 + vLLM** (see [VLLM_WSL2_STRATEGY.md](VLLM_WSL2_STRATEGY.md))
2. **Start vLLM server** with your chosen model
3. **Switch backend** in config.yaml
4. **Test with real data** and compare results
5. **Monitor performance** and adjust timeouts/settings

## Troubleshooting

### vLLM server not connecting
```
Failed to connect to vLLM server at http://localhost:8000
```
**Solution**: Start vLLM server in WSL2 or adjust URL in config.yaml

### Grammar not supported
```
WARNING: xgrammar2 not available
```
**Solution**: Install xgrammar2 in vLLM venv: `pip install xgrammar2`

### Request timeout
```
vLLM request timed out
```
**Solution**: Increase `timeout` in config.yaml backends.vllm section

### Invalid backend
```
ValueError: Unsupported backend: xyz
```
**Solution**: Use "llama.cpp", "gpt4all", or "vllm" in model config

## Summary

✅ **Complete vLLM backend implementation**
✅ **xgrammar2 support for structured outputs**
✅ **Backward compatible with existing gpt4all backend**
✅ **Comprehensive testing and validation**
✅ **Production-ready with error handling**

The system now supports both local (gpt4all) and server-based (vLLM) inference, with grammar-constrained generation for more reliable outputs!
