#!/bin/bash

echo "Setting up GRPO Training Environment..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.cargo/env
fi

echo "Creating virtual environment and installing dependencies..."
uv sync

echo "Checking CUDA availability..."
uv run python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA devices: {torch.cuda.device_count()}') if torch.cuda.is_available() else None"

echo ""
echo "Setup complete! To run the training script:"
echo "  uv run python grpo_demo.py"
echo ""
echo "To activate the environment manually:"
echo "  source .venv/bin/activate"
echo ""
echo "Make sure to set up your Weights & Biases account if you haven't already:"
echo "  uv run wandb login" 