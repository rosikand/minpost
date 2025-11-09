# `minpost`: A minimal Python package for post-training language models.

`minpost` is a lightweight research-focused, toolkit for post-training and reinforcement learning on language models. It packages common building blocks—training loops, data utilities, reward functions, and optimization helpers—into a composable Python API.

## Installation

```bash
pip install -e .

# Optional: install developer tools (formatting, linting, tests)
pip install -e .[dev]
```

This project uses `pyproject.toml` (PEP 621) with `setuptools` as the build backend.

## Quickstart

```python
from minpost import Trainer, DataLoader
from minpost.rewards import RewardFunctionRegistry
from minpost.utils import setup_logging

# Configure logging
setup_logging()

# Prepare training configuration
config = {
    "learning_rate": 2e-5,
    "batch_size": 8,
    "epochs": 3,
    "max_length": 512,
}

# Instantiate model (placeholder)
# model = ...

# Load data (placeholder)
# data_loader = DataLoader("/path/to/data", tokenizer)
# train_dataset = data_loader.load()

# Optionally create rewards via the registry
format_reward_cls = RewardFunctionRegistry.get("format_reward")
format_reward = format_reward_cls()

# Initialize trainer (placeholder)
# trainer = Trainer(model, config)
# trainer.train(train_dataset, rewards=[format_reward])
```

## Documentation & Examples

- API overview: see `api_usage.md` for detailed descriptions and extended examples.
- Notebook walkthrough: `notebooks/getting_started.ipynb` provides an interactive scaffold.
- Example scripts: explore `examples/` for CLI workflows.

## Project Structure

```
minpost/
├── minpost/                  # Main package
│   ├── __init__.py
│   ├── data.py               # Data loading and preprocessing utilities
│   ├── trainer.py            # Core training loop scaffolding
│   ├── utils.py              # Configuration and logging helpers
│   ├── rewards.py            # Reward interfaces and registry
│   └── grpo.py               # GRPO optimization helper
├── notebooks/                # Jupyter notebooks (e.g., getting_started.ipynb)
├── examples/                 # Example scripts
├── tests/                    # Test suite
├── api_usage.md              # API reference and usage guide
├── pyproject.toml            # Package configuration
└── README.md
```

