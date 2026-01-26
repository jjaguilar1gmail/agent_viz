# vLLM Backend Implementation

## Overview

This document details the complete journey of implementing vLLM with xgrammar2 support for grammar-constrained JSON outputs. The implementation maintains backward compatibility with the existing gpt4all backend while providing production-grade GPU-accelerated inference.

**Status**: ✅ Fully operational with Qwen 2.5 1.5B on RTX 3050 4GB VRAM

## Hardware Context

- **GPU**: NVIDIA RTX 3050 (4GB VRAM)
- **Environment**: WSL2 2.6.3.0 with Ubuntu, CUDA 13.0
- **Windows**: Python 3.11 (development)
- **WSL2**: Python 3.12 (vLLM server)

## Implementation Journey

### Phase 1: Initial vLLM Setup

#### PyTorch 2.9.1 Regex Bug Fix
**Issue**: vLLM failed with `ModuleNotFoundError: No module named 're._compiler'`
- PyTorch 2.9.1 broke regex module imports
- **Solution**: Downgraded to PyTorch 2.9.0
```bash
cd ~/vllm
source .venv/bin/activate
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0
```

### Phase 2: Model Selection for 4GB VRAM

#### Attempt 1: Phi-4 (FAILED - Too Large)
- **Model**: microsoft/phi-4 (14B parameters)
- **Issue**: 28GB model size, impossible for 4GB VRAM
- **Result**: Abandoned immediately

#### Attempt 2: Phi-4-mini (FAILED - Still Too Large)
- **Model**: unsloth/Phi-4-mini-instruct-bnb-4bit (3.8B parameters)
- **Model Size**: 2.87 GiB
- **Issue**: OOM errors even at 50% GPU utilization
  - Model: 2.87 GiB
  - KV cache needed: >1 GiB
  - Total: >4 GB (exceeds VRAM)
- **Detailed Error**:
  ```
  ValueError: No available memory for the cache blocks. 
  Try increasing `gpu_memory_utilization` when initializing the engine.
  ```
- **Result**: 4GB VRAM insufficient for 3.8B parameter models

#### Attempt 3: Qwen 2.5 1.5B (SUCCESS ✅)
- **Model**: unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit
- **Parameters**: 1.5B (4-bit quantized)
- **Model Size**: 1.15 GiB loaded
- **Download Size**: 1.14 GB (11 files)
- **vLLM Configuration**:
  ```bash
  vllm serve /home/effguilar/models/qwen-1.5b-4bit \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 1024 \
    --gpu-memory-utilization 0.8 \
    --enable-chunked-prefill \
    --guided-decoding-backend xgrammar
  ```
- **Memory Usage**: ~3.2 GiB total (model + KV cache)
- **Result**: ✅ Server starts successfully, CUDA graphs captured

### Phase 3: Code Implementation

#### 1. LLM Contracts Module (`src/autoviz_agent/llm/llm_contracts.py`)
- **Pydantic models** for structured outputs:
  - `IntentOutput`: Intent classification response
  - `AdaptationOutput`: Plan adaptation response with changes
- **JSON schemas** for xgrammar2 grammar generation:
  - `get_intent_schema()`: Returns JSON Schema Draft 7 for intent
  - `get_adaptation_schema()`: Returns JSON Schema Draft 7 for adaptation
- **Validation helpers**:
  - `validate_intent_output()`: Validates and parses intent JSON
  - `validate_adaptation_output()`: Validates and parses adaptation JSON

#### 2. vLLM Client (`src/autoviz_agent/llm/vllm_client.py`)
- **HTTP client** for OpenAI-compatible vLLM server (186 lines)
- **xgrammar2 integration** via `response_format` parameter
- **Same interface** as LLMClient:
  - `classify_intent()`: Intent classification with grammar constraint
  - `adapt_plan()`: Plan adaptation with grammar constraint
  - `generate_tool_calls()`: Reuses ParamResolver for parameter filling
- **Connection verification** on initialization
- **Error handling** with structured validation and repair

#### 3. Client Factory (`src/autoviz_agent/llm/factory.py`)
- **Backend selection** based on config (45 lines):
  - `llama.cpp` / `gpt4all` → LLMClient (existing)
  - `vllm` → VLLMClient (new)
- **Single entry point** via `create_llm_client(model_config)`
- **Graceful error** handling for unsupported backends

#### 4. Config Schema Updates (`config.yaml`)
- **Active vLLM model**:
  ```yaml
  qwen-1.5b-vllm:
    name: "Qwen 2.5 1.5B Instruct"
    backend: "vllm"
    url: "http://localhost:8000"
    model_name: "/home/effguilar/models/qwen-1.5b-4bit"
    temperature: 0.1
    max_tokens: 1024
    use_grammar: true
  ```
- **Backend settings**:
  - `url`: Server endpoint (http://localhost:8000)
  - `timeout`: Request timeout (60s default)
  - `use_grammar`: Enable xgrammar2 (true)
  - `temperature`: Low for deterministic outputs (0.1)

#### 5. Integration Updates
- **Graph nodes** (`src/autoviz_agent/graph/nodes.py`):
  - Updated to use `create_llm_client()` factory
  - Transparent backend switching based on config
- **Backward compatible**: Existing gpt4all configs work unchanged

#### 6. Prompt Enhancements (`src/autoviz_agent/llm/prompts.py`)
**Problem**: LLM was hallucinating column names during plan adaptation
- User question: "revenue totals by product type"
- LLM suggested: `revenue_total`, `product_type` columns
- Actual columns: `revenue`, `product_category`

**Solution**: Enhanced adaptation prompts to include full schema
- Added `column_details` with name, dtype, cardinality
- Added `numeric_cols`, `categorical_cols`, `temporal_cols` lists
- Added "AVAILABLE COLUMNS" section in prompt
- Added "CRITICAL RULES" section:
  - "ONLY use column names that appear in AVAILABLE COLUMNS"
  - "Never invent or guess column names"
  - "Column names in params MUST exactly match"

### Phase 4: Testing & Validation

#### Integration Tests (`test_vllm_integration.py`)
Created comprehensive test suite:
```bash
python test_vllm_integration.py
```

**Test Results**: ✅ ALL TESTS PASSED
1. ✅ Server connectivity verification
2. ✅ Intent classification with structured JSON output
3. ✅ Plan adaptation with structured JSON output
4. ✅ JSON schema validation (xgrammar2)
5. ✅ Error handling and fallbacks

#### Production Testing
**Test Query**: "get revenue totals by region and product type"
- **Dataset**: examples/december_revenue/december_revenue.csv
- **Columns**: date, revenue, region, product_category

**Results**:
- ✅ Intent classification: comparative_analysis (95% confidence)
- ✅ Plan adaptation: Suggested 2 changes (valid JSON structure)
- ⚠️ Execution: 6 of 7 steps executed (1 dropped due to invalid parameters)

## Current Limitations

### 1. Small Model Constraints (Qwen 1.5B)
**Issue**: LLM hallucinates column names and parameters despite schema information
- User: "revenue totals by product type"
- LLM suggests: `revenue_total`, `product_type` (hallucinated)
- Actual columns: `revenue`, `product_category`
- LLM also suggests: `groups` parameter (doesn't exist in plot_line)

**Why**: Qwen 1.5B (1.5B parameters) may be too small to reliably follow complex schema constraints

**Mitigation Options**:
1. Implement parameter repair/mapping in ParamResolver
2. Use larger model (Qwen 2.5 3B if VRAM permits)
3. Add column name enums to JSON schema for stricter validation
4. Implement fuzzy matching for column name correction

### 2. VRAM Constraints
**Maximum model size for RTX 3050 4GB**: ~1.5B parameters (4-bit quantized)
- Larger models (3B+) would require:
  - Better GPU (6GB+ VRAM recommended)
  - CPU offloading (significantly slower)
  - Model quantization below 4-bit (accuracy loss)

## Usage

### Starting vLLM Server (WSL2)
```bash
cd ~/vllm
source .venv/bin/activate

# Start Qwen 1.5B server
vllm serve /home/effguilar/models/qwen-1.5b-4bit \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 1024 \
  --gpu-memory-utilization 0.8 \
  --enable-chunked-prefill \
  --guided-decoding-backend xgrammar
```

**Server Startup Output**:
```
INFO: Loading model weights...
INFO: Initializing KV cache...
INFO: Capturing CUDA graphs...
INFO: Server running at http://0.0.0.0:8000
```

### Using vLLM Backend

1. **Verify server is running**:
```bash
curl http://localhost:8000/health
# Expected: {"status": "ok"}
```

2. **Update config.yaml** (if not already set):
```yaml
default_model: "qwen-1.5b-vllm"
```

3. **Run analysis**:
```bash
autoviz run dataset.csv "your question"
```

### Switching Back to gpt4all
```yaml
default_model: "phi-3.5-mini"  # Uses gpt4all backend
```

## Benefits of vLLM Backend

### 1. Grammar-Constrained Outputs (xgrammar2)
- **Guarantees valid JSON structure**: No parsing errors
- **Enforces schema format**: LLM must follow exact structure
- **Reduces validation failures**: More reliable than text parsing
- **Note**: Enforces structure, not semantic correctness (column names still need validation)

### 2. GPU Acceleration
- **CUDA support**: Full GPU utilization in WSL2
- **Optimized inference**: Production-grade performance
- **Batching support**: Can handle multiple requests efficiently
- **Memory management**: Efficient KV cache handling

### 3. Hybrid Architecture
- **xgrammar2**: Structural constraints via JSON schema
- **ParamResolver**: Fills missing/"auto" values deterministically
- **Validation layer**: Safety net for semantic errors

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────┐
│         autoviz CLI (src/autoviz_agent/cli/)        │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│              Graph Nodes (nodes.py)                 │
│  - classify_intent_node()                           │
│  - adapt_plan_node()                                │
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
│ - gpt4all library  │   │ - HTTP POST requests     │
│ - Local GGUF       │   │ - OpenAI API format      │
│ - Fallback mode    │   │ - xgrammar2 via          │
│ - Text parsing     │   │   response_format        │
└────────────────────┘   └──────────────────────────┘
             │                       │
             └───────────┬───────────┘
                         │ Uses
                         ▼
┌─────────────────────────────────────────────────────┐
│         PromptBuilder (prompts.py)                  │
│  - Intent classification template                   │
│  - Adaptation template with full schema             │
│  - Column details (name, dtype, cardinality)        │
│  - Available columns list                           │
│  - Critical rules for column matching               │
└─────────────────────────────────────────────────────┘
                         │ Validates with
                         ▼
┌─────────────────────────────────────────────────────┐
│       LLM Contracts (llm_contracts.py)              │
│  - Pydantic models (IntentOutput, AdaptationOutput) │
│  - JSON schemas for xgrammar2                       │
│  - Validation helpers                               │
└─────────────────────────────────────────────────────┘
```

## Files Created/Modified

### New Files
- `src/autoviz_agent/llm/llm_contracts.py` (135 lines)
  - JSON contracts and schemas for xgrammar2
- `src/autoviz_agent/llm/vllm_client.py` (186 lines)
  - vLLM HTTP client with xgrammar2 support
- `src/autoviz_agent/llm/factory.py` (45 lines)
  - Backend factory for client selection
- `test_vllm_integration.py` (120 lines)
  - End-to-end integration tests

### Modified Files
- `config.yaml`
  - Added qwen-1.5b-vllm backend configuration
- `src/autoviz_agent/graph/nodes.py`
  - Use factory instead of direct LLMClient instantiation
- `src/autoviz_agent/llm/prompts.py`
  - Enhanced with full schema information (lines 180-280)
  - Added column details, available columns section
  - Added critical rules for column name matching

### Unchanged (Backward Compatible)
- `src/autoviz_agent/llm/client.py` - Original gpt4all client
- All existing tools in `src/autoviz_agent/tools/`
- Runtime execution components
- Reporting and output generation

## Installation & Setup

### Prerequisites
- **WSL2** with Ubuntu (Windows 11)
- **NVIDIA GPU** with CUDA support
- **CUDA 12.0+** installed in WSL2
- **Python 3.12** in WSL2

### Step 1: Install vLLM in WSL2
```bash
# Create virtual environment
cd ~
mkdir vllm
cd vllm
python3.12 -m venv .venv
source .venv/bin/activate

# Install PyTorch 2.9.0 (critical - 2.9.1 has regex bug)
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0

# Install vLLM with xgrammar2
pip install vllm==0.14.1 xgrammar

# Install bitsandbytes for 4-bit quantization
pip install bitsandbytes>=0.46.1
```

### Step 2: Download Qwen 1.5B Model
```bash
# Install HuggingFace CLI
pip install huggingface-hub[cli]

# Download model (1.14 GB)
mkdir -p ~/models
huggingface-cli download unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit \
  --local-dir ~/models/qwen-1.5b-4bit \
  --local-dir-use-symlinks False
```

### Step 3: Verify Installation
```bash
# Test vLLM installation
python -c "import vllm; print(f'vLLM version: {vllm.__version__}')"
# Expected: vLLM version: 0.14.1

# Test xgrammar
python -c "import xgrammar; print('xgrammar installed')"
# Expected: xgrammar installed

# Check model files
ls -lh ~/models/qwen-1.5b-4bit/
# Expected: 11 files, ~1.14 GB total
```

### Step 4: Start Server
```bash
cd ~/vllm
source .venv/bin/activate

vllm serve /home/effguilar/models/qwen-1.5b-4bit \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 1024 \
  --gpu-memory-utilization 0.8 \
  --enable-chunked-prefill \
  --guided-decoding-backend xgrammar
```

Wait for: `INFO: Server running at http://0.0.0.0:8000`

### Step 5: Test from Windows
```powershell
# Test server health
curl http://localhost:8000/health

# Run integration tests
cd C:\Users\JeffAguilar\Code\agent_viz
python test_vllm_integration.py
```

## Troubleshooting

### Server Connection Issues

**Problem**: `Failed to connect to vLLM server at http://localhost:8000`

**Solutions**:
1. Verify server is running in WSL2:
   ```bash
   curl http://localhost:8000/health
   ```
2. Check WSL2 networking:
   ```powershell
   # From Windows
   wsl hostname -I
   # Use the IP address in config.yaml if localhost doesn't work
   ```
3. Restart server with proper flags:
   ```bash
   vllm serve <model> --host 0.0.0.0 --port 8000
   ```

### PyTorch Regex Bug

**Problem**: `ModuleNotFoundError: No module named 're._compiler'`

**Solution**: Use PyTorch 2.9.0 (not 2.9.1)
```bash
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0
```

### Out of Memory (OOM) Errors

**Problem**: `No available memory for the cache blocks`

**Solutions**:
1. Use smaller model (Qwen 1.5B recommended for 4GB VRAM)
2. Reduce max-model-len:
   ```bash
   vllm serve <model> --max-model-len 512  # Reduce from 1024
   ```
3. Lower gpu-memory-utilization:
   ```bash
   vllm serve <model> --gpu-memory-utilization 0.7  # Reduce from 0.8
   ```
4. Check current VRAM usage:
   ```bash
   nvidia-smi
   ```

### xgrammar2 Not Available

**Problem**: `WARNING: xgrammar2 not available, using fallback`

**Solution**: Install xgrammar in vLLM environment
```bash
cd ~/vllm
source .venv/bin/activate
pip install xgrammar
```

### Request Timeouts

**Problem**: `vLLM request timed out after 60 seconds`

**Solutions**:
1. Increase timeout in config.yaml:
   ```yaml
   backends:
     vllm:
       timeout: 120  # Increase from 60
   ```
2. Reduce max_tokens:
   ```yaml
   qwen-1.5b-vllm:
     max_tokens: 512  # Reduce from 1024
   ```
3. Use faster model or better hardware

### Invalid Backend Error

**Problem**: `ValueError: Unsupported backend: xyz`

**Solution**: Use valid backend names in config.yaml:
- `llama.cpp` (gpt4all)
- `gpt4all` (gpt4all)
- `vllm` (vLLM server)

### Column Name Hallucination

**Problem**: LLM suggests wrong column names (e.g., `revenue_total` instead of `revenue`)

**Why**: Small models (1.5B params) struggle with complex schema constraints

**Current Status**: ⚠️ Known limitation - steps with wrong columns get dropped

**Workarounds**:
1. Use more explicit column names in your dataset
2. Phrase questions using exact column names when possible
3. Future: Implement parameter repair with fuzzy matching

## Performance Benchmarks

### Qwen 2.5 1.5B on RTX 3050 4GB

**Hardware**:
- GPU: NVIDIA RTX 3050 (4GB VRAM)
- CPU: Available for preprocessing
- RAM: 16GB system memory

**Memory Usage**:
- Model: 1.15 GiB
- KV cache: ~2.0 GiB
- Total VRAM: ~3.2 GiB / 4.0 GiB (80%)

**Inference Speed** (approximate):
- Intent classification: ~2-3 seconds
- Plan adaptation: ~3-5 seconds
- Token throughput: ~30-50 tokens/second

**Quality**:
- ✅ Intent classification: Reliable (95%+ confidence)
- ⚠️ Plan adaptation: Structure correct, semantics need validation
- ⚠️ Column names: Prone to hallucination with small model

## Comparison: vLLM vs gpt4all

| Feature | gpt4all (Phi-3.5-mini) | vLLM (Qwen 1.5B) |
|---------|------------------------|------------------|
| **Backend** | Local GGUF | HTTP server |
| **Model Size** | 2.4GB (3.8B params) | 1.14GB (1.5B params) |
| **VRAM Usage** | CPU-only (no CUDA) | GPU: 3.2GB |
| **Speed** | Slower (CPU) | Faster (GPU) |
| **Structured Output** | Text parsing | xgrammar2 JSON |
| **JSON Validity** | Can fail | Guaranteed |
| **Setup** | Simple (pip install) | Complex (WSL2+CUDA) |
| **Memory** | System RAM | GPU VRAM |
| **Reliability** | Fallback mode | Server dependent |
| **Column Names** | Issues vary | Known hallucination |

**Recommendation**:
- **Development**: gpt4all (easier setup, good enough)
- **Production**: vLLM (faster, structured outputs, scalable)
- **4GB VRAM**: Qwen 1.5B only option
- **6GB+ VRAM**: Try Qwen 3B or Phi-3.5-mini via vLLM

## Future Improvements

### Short Term (Implementation Ready)
1. **Parameter Repair Logic**
   - Fuzzy match column names (e.g., "revenue_total" → "revenue")
   - Map common synonyms (e.g., "product_type" → "product_category")
   - Implement in VLLMClient._apply_adaptations()

2. **Enhanced JSON Schema**
   - Add column name enums to force valid choices
   - Include parameter type enums
   - Stricter validation in xgrammar2

3. **Better Error Messages**
   - Log which columns were hallucinated
   - Suggest closest matches
   - Provide repair hints

### Medium Term (Requires Research)
1. **Larger Model Support**
   - Test Qwen 2.5 3B (if fits in VRAM)
   - Evaluate Phi-3.5-mini via vLLM
   - CPU offloading for hybrid execution

2. **Multi-Model Strategy**
   - Use Qwen 1.5B for intent classification (fast)
   - Use larger model for plan adaptation (accurate)
   - Hybrid approach for speed + quality

3. **Prompt Engineering**
   - Few-shot examples with correct column usage
   - Stronger emphasis on schema matching
   - Chain-of-thought for column selection

### Long Term (Infrastructure)
1. **Model Fine-Tuning**
   - Fine-tune on autoviz-specific tasks
   - Train on correct column name usage
   - Improve schema constraint following

2. **Active Learning**
   - Collect user corrections
   - Retrain on validated outputs
   - Iteratively improve accuracy

3. **Hardware Upgrade**
   - 8GB+ VRAM GPU for larger models
   - Better performance and quality
   - Support for more sophisticated models

## Technical Details

### vLLM Server Configuration

**Recommended Flags**:
```bash
vllm serve <model> \
  --host 0.0.0.0 \                      # Listen on all interfaces
  --port 8000 \                         # Default port
  --max-model-len 1024 \                # Context window (adjust for VRAM)
  --gpu-memory-utilization 0.8 \        # Use 80% of VRAM
  --enable-chunked-prefill \            # Memory optimization
  --guided-decoding-backend xgrammar \  # Enable xgrammar2
  --dtype float16 \                     # Half precision (default)
  --tensor-parallel-size 1              # Single GPU
```

**Optional Optimizations**:
```bash
  --disable-log-requests \              # Reduce logging overhead
  --max-num-seqs 8 \                    # Batch size
  --max-num-batched-tokens 1024 \       # Max tokens per batch
  --swap-space 4                        # CPU swap space (GB)
```

### xgrammar2 JSON Schema Format

Example schema passed to vLLM:
```json
{
  "type": "object",
  "properties": {
    "intent": {
      "type": "string",
      "enum": ["general_eda", "comparative_analysis", "time_series", "anomaly_detection"]
    },
    "confidence": {
      "type": "number",
      "minimum": 0.0,
      "maximum": 1.0
    }
  },
  "required": ["intent", "confidence"],
  "additionalProperties": false
}
```

This schema is automatically generated from Pydantic models in `llm_contracts.py`.

### Environment Variables

**WSL2 vLLM Environment**:
```bash
export VLLM_USE_TRITON=0              # Disable Triton (compatibility)
export PYTORCH_ENABLE_MPS_FALLBACK=1  # PyTorch fallback
export CUDA_VISIBLE_DEVICES=0         # Use GPU 0
```

**Windows Development Environment**:
```powershell
# No special variables needed
# Connects via HTTP to WSL2 server
```

## Summary

### ✅ Completed
- vLLM 0.14.1 installed in WSL2 with xgrammar2 support
- PyTorch 2.9.0 (fixed regex bug in 2.9.1)
- Qwen 2.5 1.5B 4-bit model working on RTX 3050 4GB VRAM
- VLLMClient integration with OpenAI-compatible API
- Factory pattern for backend switching
- Comprehensive integration tests (all passing)
- Backward compatibility with gpt4all
- Enhanced prompts with full schema information

### ⚠️ Known Issues
- Small model (1.5B) hallucinates column names despite schema
- Steps with invalid parameters get dropped during validation
- Need parameter repair logic for production use

### 🎯 Achievements
- **Successfully running vLLM on 4GB VRAM** (Qwen 1.5B)
- **xgrammar2 guarantees valid JSON structure**
- **GPU acceleration working** in WSL2
- **Production-ready architecture** with error handling
- **Comprehensive documentation** for setup and troubleshooting

The system now supports both local (gpt4all) and server-based (vLLM) inference with grammar-constrained generation for more reliable structured outputs!
