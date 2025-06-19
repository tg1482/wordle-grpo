# Can LLMs outperform humans in Wordle (WIP)

This project is a small attempt to leverage GRPO to see whether LLMs can strategize on which words to guess next and achieve a lower sample rate.

## Setup

### Prerequisites

- Python 3.9+
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
uv run python wordle_grpo.py
```
