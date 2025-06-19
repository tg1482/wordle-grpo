# GRPO Training Environment

This project implements Granular Format Rewards for Eliciting Mathematical Reasoning Capabilities in Small Language Models using GRPO (Generalized Reward-based Policy Optimization).

## Setup

### Prerequisites

- Python 3.9+
- CUDA-compatible GPU (recommended)
- `uv` package manager

### Installation

1. Install uv (if not already installed):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Create and activate the environment:

```bash
uv sync
```

3. Activate the virtual environment:

```bash
source .venv/bin/activate
```

## Usage

### Running the Training Script

```bash
uv run python grpo_demo.py
```

### Configuration

The script trains a model on GSM8K mathematical reasoning tasks with multiple reward functions:

- XML format compliance rewards
- Correctness rewards
- Integer format rewards

Key parameters you can modify in `grpo_demo.py`:

- `model_name`: Choose between "meta-llama/Llama-3.2-1B-Instruct" or "Qwen/Qwen2.5-1.5B-Instruct"
- Training hyperparameters in `GRPOConfig`
- Reward function weights and combinations

### Requirements

- GPU memory: ~8-16GB VRAM (depending on model size and batch size)
- Weights & Biases account (for logging)
- Internet connection (for downloading models and datasets)

## Troubleshooting

If you encounter CUDA/GPU issues:

1. Ensure CUDA drivers are properly installed
2. Check PyTorch CUDA compatibility: `python -c "import torch; print(torch.cuda.is_available())"`
3. Consider reducing `per_device_train_batch_size` if running out of memory

For flash attention issues, you may need to install flash-attn separately:

```bash
uv add flash-attn --no-build-isolation
```
