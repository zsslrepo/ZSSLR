#!/bin/bash
# One-shot environment setup for ZSSLR-AzSL.
set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "==> Working directory: $ROOT"

if [ ! -d ".venv" ]; then
    echo "==> Creating virtual environment in .venv"
    python -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

echo "==> Upgrading pip"
pip install --upgrade pip wheel setuptools

echo "==> Installing PyTorch (CPU build by default; override TORCH_INDEX_URL for CUDA)"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-}"
if [ -n "$TORCH_INDEX_URL" ]; then
    pip install torch torchvision --index-url "$TORCH_INDEX_URL"
else
    pip install torch torchvision
fi

echo "==> Installing remaining requirements"
pip install -r requirements.txt

echo "==> Installing this package in editable mode"
pip install -e .

echo "==> Creating expected directories"
mkdir -p checkpoints data outputs/{logs,checkpoints,evaluation,figures}

echo "==> Done. Activate with: source .venv/bin/activate"
