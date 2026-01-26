#!/bin/bash
# vLLM Setup Script for WSL2 Ubuntu

set -e  # Exit on error

echo "=========================================="
echo "vLLM Setup for WSL2 - Step by Step"
echo "=========================================="

# Step 2: Update system and install dependencies
echo ""
echo "Step 2: Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y python3 python3-venv python3-pip build-essential git git-lfs

echo ""
echo "Python version:"
python3 --version

# Step 3: Create vLLM directory and virtual environment
echo ""
echo "Step 3: Creating vLLM environment..."
mkdir -p ~/vllm
cd ~/vllm

if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo "Virtual environment created"
else
    echo "Virtual environment already exists"
fi

source .venv/bin/activate

# Step 4: Install vLLM
echo ""
echo "Step 4: Installing vLLM (this may take a few minutes)..."
pip install --upgrade pip
pip install vllm

# Step 5: Install xgrammar (optional but recommended)
echo ""
echo "Step 5: Installing xgrammar2..."
pip install outlines || echo "Note: outlines installation failed, xgrammar may not be available"

echo ""
echo "=========================================="
echo "Installation complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Download a model (we'll do this next)"
echo "2. Start vLLM server"
echo ""
echo "To activate vLLM environment in the future:"
echo "  cd ~/vllm && source .venv/bin/activate"
