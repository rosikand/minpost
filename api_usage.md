# `minpost` API Usage Guide

This document walks through the primary modules exposed by `minpost` and illustrates how to combine them to build post-training workflows.

## Table of Contents

1. [Core Imports](#core-imports)
2. [Trainer Interfaces](#trainer-interfaces)
3. [Data Utilities](#data-utilities)
4. [Reward Functions](#reward-functions)
5. [Optimization Helpers (GRPO)](#optimization-helpers-grpo)
6. [Utilities](#utilities)
7. [Putting It All Together](#putting-it-all-together)

---

## Core Imports

```python
from minpost import Trainer, DataLoader
from minpost.rewards import RewardFunction, RewardFunctionRegistry
from minpost.grpo import GRPO
from minpost.utils import setup_logging, load_config, save_config
```

Call `setup_logging()` early in your application or notebook to capture logs from other components.

---

## Trainer Interfaces

### `TrainExperiment`

Located in `minpost.trainer`, `TrainExperiment` is a base class for custom experiments. Subclass it when you need to orchestrate setup, run, and teardown hooks around training.

```python
from minpost.trainer import TrainExperiment

class MyExperiment(TrainExperiment):
    def setup(self):
        # Allocate resources (optimizers, schedulers, etc.)
        ...

    def run(self):
        # Implement the main training / evaluation loop
        ...

    def teardown(self):
        # Clean up state (close files, free memory, etc.)
        ...
```

### `RLVRTrainer`

`RLVRTrainer` extends `TrainExperiment` with reinforcement-learning-from-feedback (RLHF/RLVR) conventions. It expects `self.rl_algo` to be set to an algorithm instance (e.g., `GRPO`).

```python
from minpost.trainer import RLVRTrainer
from minpost.grpo import GRPO

class MyRLHFTrainer(RLVRTrainer):
    def setup(self):
        super().setup()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config["learning_rate"])
        self.rl_algo = GRPO(
            model=self.model,
            optimizer=self.optimizer,
            kl_coeff=self.config.get("kl_coeff", 0.1),
        )

    def run(self):
        for batch in self._build_batches():
            loss_dict = self.rl_algo.step(**batch)
            self._log_metrics(loss_dict)
```

### `Trainer`

`Trainer` is a light-weight placeholder class that you can extend or replace with your preferred supervised fine-tuning loop.

```python
trainer = Trainer(model, config)
trainer.train(train_dataset, eval_dataset=eval_dataset)
```

Override `.train`, `.evaluate`, and `.save` to suit your environment.

---

## Data Utilities

`DataLoader` (from `minpost.data`) is a simple scaffold for dataset preparation.

```python
from minpost import DataLoader

loader = DataLoader(
    dataset_path="/path/to/data.jsonl",
    tokenizer=my_tokenizer,
    max_length=512,
)

# Implement `.load` and `.preprocess` to return processed samples
dataset = loader.load()
```

Subclass or monkey-patch the placeholder methods to integrate with Hugging Face `datasets`, PyTorch `DataLoader`, or custom pipelines.

---

## Reward Functions

Reward logic lives in `minpost.rewards`.

### Base Interface

```python
from minpost.rewards import RewardFunction

class LengthReward(RewardFunction):
    def __call__(self, completions, **kwargs):
        return [float(len(item) < 200) for item in completions]
```

### Registry

`RewardFunctionRegistry` lets you register and instantiate reward functions by string name.

```python
from minpost.rewards import RewardFunctionRegistry, RewardFunction

@RewardFunctionRegistry.register("length_reward")
class LengthReward(RewardFunction):
    def __call__(self, completions, **kwargs):
        return [float(len(text) < 200) for text in completions]

# Later on…
reward_cls = RewardFunctionRegistry.get("length_reward")
reward_fn = reward_cls()
scores = reward_fn(["short completion", "very long completion ..."])
```

### Built-in Rewards

- `format_reward`: checks whether a completion matches the `<think>...</think><answer>...</answer>` template. Use it via the registry:

```python
format_reward_cls = RewardFunctionRegistry.get("format_reward")
format_reward = format_reward_cls()
scores = format_reward(completions)
```

---

## Optimization Helpers (GRPO)

`GRPO` in `minpost.grpo` implements Generalized Regularized Policy Optimization for RLHF-style loops.

```python
from minpost.grpo import GRPO

grpo = GRPO(
    model=model,
    optimizer=optimizer,
    kl_coeff=0.1,
    clip_eps=0.2,
    value_coef=0.5,
    entropy_coef=0.01,
)

loss_dict = grpo.step(
    logprobs=batch["logprobs"],
    old_logprobs=batch["old_logprobs"],
    advantages=batch["advantages"],
    values=batch["values"],
    old_values=batch["old_values"],
    returns=batch["returns"],
    ref_logprobs=batch.get("ref_logprobs"),
)
```

The returned dictionary exposes scalar tensors such as `loss`, `policy_loss`, `value_loss`, `entropy_loss`, and `kl_loss` for logging or debugging.

---

## Utilities

Utility helpers in `minpost.utils` support configuration management and model inspection.

- `setup_logging(level=logging.INFO)`: initialize package-wide logging.
- `load_config(path)`: load JSON configurations into dictionaries.
- `save_config(config, path)`: persist configuration dictionaries.
- `compare_model_structure_loose(model_a, model_b)`: compare parameter shapes and module order between models.
- `compare_model_tree(model_a, model_b, max_depth=None)`: recursively diff module trees.
- `summarize_model_diff(model_a, model_b, max_depth=None)`: high-level summary of structural differences.

---

## Putting It All Together

Here is a sketch of how the pieces can be orchestrated in a training script:

```python
import torch
from minpost import Trainer, DataLoader
from minpost.grpo import GRPO
from minpost.rewards import RewardFunctionRegistry
from minpost.utils import setup_logging, load_config

def main(config_path: str):
    setup_logging()
    config = load_config(config_path)

    # Prepare model, optimizer, and tokenizer (placeholders)
    model = ...  # torch.nn.Module
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    tokenizer = ...  # your tokenizer

    # Data
    data_loader = DataLoader(config["dataset_path"], tokenizer, max_length=config["max_length"])
    train_dataset = data_loader.load()

    # Reward(s)
    format_reward_cls = RewardFunctionRegistry.get("format_reward")
    format_reward = format_reward_cls()

    # RL optimization
    grpo = GRPO(model=model, optimizer=optimizer, kl_coeff=config.get("kl_coeff", 0.1))

    # Trainer scaffold
    trainer = Trainer(model, config)
    trainer.train(train_dataset)

    # Example usage inside your custom loop
    # loss_dict = grpo.step(**batch)
    # rewards = format_reward(completions)

if __name__ == "__main__":
    main("configs/train.json")
```

Customize the placeholders with your model, tokenizer, dataset, and training logic to build a complete post-training system.


