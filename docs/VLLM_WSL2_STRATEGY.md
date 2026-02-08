# vLLM + xgrammar2 Strategy on WSL2 (Windows host)

## Purpose
Use a local vLLM server with grammar-constrained outputs (xgrammar2) to improve
LLM reliability, while keeping this repo on Windows for development.

## Decision Summary (Why this approach)
- WSL2 provides Linux + NVIDIA CUDA support, which vLLM expects.
- Keeping the repo on Windows avoids forcing all dev tooling into Linux.
- xgrammar2 reduces JSON parsing failures and invalid tool params.
- A hybrid param strategy keeps safety while allowing more LLM control.

## When this is the right choice
- You want local inference with NVIDIA GPU acceleration.
- You want better structured outputs than "JSON in text."
- You want to keep the existing deterministic tool execution pipeline.

## Tradeoffs
- vLLM is not officially supported on native Windows.
- Quantized model support can be version- and backend-specific.
- Grammar support varies by vLLM version (test first).

## Setup Steps (WSL2 + vLLM)
### 0) Prereqs
- Windows 11 or Windows 10 22H2
- NVIDIA GPU
- Internet access in WSL2

### 1) Install NVIDIA Windows driver (WSL2-capable)
1. Download the latest NVIDIA driver that includes WSL2 support.
2. Install on Windows and reboot. - this is done, but you can still verify below.

Verify from Windows PowerShell:
```powershell
nvidia-smi
```

### 2) Install WSL2 + Ubuntu
Run in Windows PowerShell (Admin):
```powershell
wsl --install -d Ubuntu
```

Reboot if prompted, then launch "Ubuntu" from the Start menu and create a Linux user.

Verify:
```powershell
wsl -l -v
```
You should see Ubuntu with version 2.

### 3) Verify GPU inside WSL2
Open Ubuntu (WSL) and run:
```bash
nvidia-smi
```
You should see your GPU details.

### 4) Install system deps + Python
In WSL (Ubuntu):
```bash
sudo apt-get update
sudo apt-get install -y python3 python3-venv python3-pip build-essential
python3 --version
```

If your Python is older than 3.10, install a newer one via apt or pyenv.

### 5) Create a vLLM venv and install vLLM
In WSL:
```bash
mkdir -p ~/vllm
cd ~/vllm
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install vllm
```

### 6) Install xgrammar2 (if supported by your vLLM version)
If your vLLM build supports xgrammar2, install it in the same venv:
```bash
pip install xgrammar2
```

If vLLM complains about grammar support, check the vLLM docs for the correct
version/flag. Grammar support is version-specific.

### 7) Get Phi-4-mini-instruct (bit-quant, Unsloth-style)
Use a Hugging Face repo that provides a bit-quant version of Phi-4-mini-instruct
(for example, an Unsloth 4-bit variant). Replace the model ID below with the exact
repo you want.

Example download:
```bash
mkdir -p ~/models
cd ~/models
git lfs install
export HF_MODEL="unsloth/Phi-4-mini-instruct-bnb-4bit"
git clone https://huggingface.co/$HF_MODEL
```

If you already have the model on Windows, copy it into WSL for performance:
```bash
cp -r /mnt/c/path/to/$HF_MODEL ~/models/
```

### 8) Run the vLLM OpenAI-compatible server
In WSL (inside the venv):
```bash
vllm serve ~/models/$HF_MODEL \
  --host 0.0.0.0 \
  --port 8000
```

If your vLLM version does not have `vllm serve`, use:
```bash
python -m vllm.entrypoints.openai.api_server \
  --model ~/models/$HF_MODEL \
  --host 0.0.0.0 \
  --port 8000
```

If your bit-quant repo requires an explicit quantization flag, add it here
(example, verify with vLLM docs):
```bash
vllm serve ~/models/$HF_MODEL \
  --quantization bitsandbytes \
  --host 0.0.0.0 \
  --port 8000
```

### 9) Smoke test the server
From Windows PowerShell:
```powershell
curl http://localhost:8000/v1/models
```

You should see the model listed.

### 10) Wire this repo to the vLLM server
This repo currently uses gpt4all in `src/autoviz_agent/llm/client.py`.
To use vLLM, you will need an HTTP client that calls the OpenAI-compatible
endpoint in step 8.

Minimal change idea:
- Add a `backend: vllm` model entry in `config.yaml`
- In `LLMClient`, if backend is vllm, call `POST /v1/chat/completions`
  with the prompt and parse the JSON response.
- Add a `src/autoviz_agent/llm/llm_contracts.py` file for JSON output contracts
  used by xgrammar2.

If you want, I can implement this and add xgrammar2 config options.

## Strategy Change (Why and How)
### Why change the parameter strategy?
The current ParamResolver is deterministic but can be brittle when:
- Tools or params are new or domain-specific.
- The user specifies precise columns or values not captured by heuristics.
- JSON parsing is unreliable (free-form model output).

### Proposed hybrid strategy
1) LLM proposes params within a strict grammar (xgrammar2).
2) ParamResolver fills missing or "auto" values and validates.
3) Validation/repair remains the safety net.

### Where xgrammar2 helps most
- Intent classification JSON
- Plan adaptation JSON
- Any future direct tool-call emission

## What changes for config.json?
This repo does not have a `config.json`. The only config you change here is
`config.yaml` for the backend and model selection.

If you meant the model's Hugging Face `config.json`:
- You usually do not edit it. vLLM reads it to determine architecture, dtype,
  and quantization metadata.
- If your 4-bit repo ships a compatible `config.json`, just point vLLM at the
  repo directory and pass the correct `--quantization` flag (if required).

## Notes
- WSL sees your Windows C: drive at `/mnt/c`.
- For best performance, keep models inside the WSL filesystem, not `/mnt/c`.
