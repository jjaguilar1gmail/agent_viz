#!/bin/bash
# Workaround for PyTorch 2.9.1 regex bug
cd ~/vllm
source .venv/bin/activate

# Patch the regex issue by limiting recursion
export PYTHONRECURSIONLIMIT=2000

# Start vLLM server
vllm serve microsoft/phi-4 \
  --model ~/models/phi-4 \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.70 \
  --dtype auto \
  --disable-log-requests \
  --trust-remote-code