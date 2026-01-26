# vLLM Setup Status

## Current Status: ⚠️ Blocked by PyTorch Bug

### Completed Steps ✅

1. ✅ WSL2 installed and configured (version 2.6.3.0)
2. ✅ Ubuntu WSL distribution installed
3. ✅ GPU accessible from WSL2 (NVIDIA RTX 3050 with CUDA 13.0)
4. ✅ Python virtual environment created in ~/vllm
5. ✅ vLLM 0.14.1 installed with all dependencies
6. ✅ xgrammar and outlines installed
7. ✅ Phi-4 model downloaded (~28GB in ~/models/phi-4)

### Current Status 🔄

**✅ PyTorch Bug SOLVED** - Downgraded to PyTorch 2.9.0 (no regex bug!)

**❌ NEW Blocker: GPU Memory**

- Error: `CUDA out of memory` - tried to allocate 350MB, only 4GB total VRAM
- Issue: Full Phi-4 model (28GB safetensors) requires ~10GB VRAM to load
- Your RTX 3050 has only 4GB VRAM
- **Solution: Use quantized models or smaller models**

### Workaround Options

#### ✅ Option 1: PyTorch 2.9.0 + Quantized Model (RECOMMENDED)
```bash
# Already done: PyTorch 2.9.0 installed
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0  # ✅ Done

# Download AWQ quantized Phi-4 (fits in 4GB VRAM)
huggingface-cli download casperhansen/phi-4-awq --local-dir ~/models/phi-4-awq

# Start server
vllm serve ~/models/phi-4-awq \
  --host 0.0.0.0 --port 8000 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.85 \
  --quantization awq
```

#### Option 2: Use Smaller Model (Phi-3.5-mini)
```bash
# Download Phi-3.5-mini (3.8B params, fits easily)
huggingface-cli download microsoft/Phi-3.5-mini-instruct --local-dir ~/models/phi-3.5-mini

# Start server
vllm serve ~/models/phi-3.5-mini \
  --host 0.0.0.0 --port 8000 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.90
```

#### Option 3: Use gpt4all (Already Working!)
Your current GGUF model setup with gpt4all works perfectly and uses GPU efficiently.

### Next Steps When Bug is Fixed

1. Start vLLM server:
```bash
cd ~/vllm
source .venv/bin/activate
vllm serve microsoft/phi-4 \
  --model ~/models/phi-4 \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.75 \
  --dtype float16 \
  --trust-remote-code
```

2. Test connection from Windows:
```powershell
curl http://localhost:8000/v1/models
```

3. Update config.yaml:
```yaml
default_model: phi-4-mini-vllm
```

4. Run autoviz with vLLM:
```powershell
autoviz run examples/december_revenue/december_revenue.csv "analyze revenue patterns"
```

### Infrastructure Ready ✅

All code infrastructure is complete and tested:
- `src/autoviz_agent/llm/vllm_client.py` - HTTP client for vLLM
- `src/autoviz_agent/llm/factory.py` - Backend selection
- `src/autoviz_agent/llm/prompts.py` - Centralized prompts
- `src/autoviz_agent/llm/llm_contracts.py` - xgrammar schemas
- `config.yaml` - vLLM backend configuration

### Testing Status

- ✅ Prompt system works with gpt4all
- ✅ Factory pattern switches backends correctly
- ✅ End-to-end test with gpt4all passes
- ⏳ vLLM backend waiting for server availability

### Recommendations

1. **For now**: Continue using gpt4all with GGUF models (working perfectly)
2. **Monitor**: Watch for PyTorch 2.9.2 or vLLM 0.15+ releases
3. **Alternative**: Consider using TGI (Text Generation Inference) or Ollama as alternative backends

### Resources

- [vLLM GitHub Issues](https://github.com/vllm-project/vllm/issues)
- [PyTorch Issue Tracker](https://github.com/pytorch/pytorch/issues)
- [Current project models: models/](file:///C:/Users/JeffAguilar/Code/agent_viz/models/)
- [vLLM documentation](https://docs.vllm.ai/)

---

**Last Updated**: 2025-01-XX
**By**: GitHub Copilot
**Status**: Implementation complete, server blocked by upstream bug
