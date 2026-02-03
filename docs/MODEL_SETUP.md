# AutoViz Agent - Model Setup

## Quick Start (Without LLM)

The system works in **fallback mode** without downloading models. It uses keyword-based intent classification and runs the deterministic analysis pipeline.

Try it now:
```bash
autoviz run examples/december_revenue/december_revenue.csv "What are the trends?"
```

## Full LLM Setup (Optional)

To enable actual LLM-driven plan adaptation, download a model:

### Phi-3.5-mini (Recommended - 2.7GB)

```bash
# Create models directory
mkdir models

# Download from HuggingFace (requires git-lfs)
cd models
wget https://huggingface.co/microsoft/Phi-3.5-mini-instruct-gguf/resolve/main/Phi-3.5-mini-instruct-Q4_K_M.gguf
```

### Alternative Models

See [config.yaml](config.yaml) for other supported models:
- Llama 3.1 8B Instruct
- Mistral 7B Instruct v0.3
- Gemma 2 9B Instruct

## Configuration

Edit `config.yaml` to:
- Change default model
- Adjust temperature/sampling parameters
- Configure GPU layers
- Set output paths

## Observability Artifacts

Every run creates:
- `plan_template.json` - Original template selected
- `plan_adapted.json` - LLM-adapted plan
- `plan_diff.md` - Visual diff showing changes
- `tool_calls.json` - Tool execution sequence
- `execution_log.json` - Complete execution trace
- `report.md` - Final analysis report
- `charts/` - Generated visualizations

These prove **this is not a rules engine** - the LLM actively adapts plans!
