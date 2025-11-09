"""
File: rewards.py
------------------
Provides reward functions for the package.
"""


from abc import ABC, abstractmethod
import re
from typing import Any, Mapping, Sequence

class RewardFunction(ABC):
    """
    Abstract base class for reward functions.
    All custom reward functions should inherit from this class and override __call__.
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def __call__(self, *args, **kwargs):
        """
        Compute the reward for a given set of arguments.

        Returns:
            reward(s): The computed reward(s), format and type defined by subclass.
        """
        pass


class RewardFunctionRegistry:
    """
    Registry for reward functions.
    Allows registration and retrieval of reward function classes by name.
    """
    _registry = {}

    @classmethod
    def register(cls, name):
        """
        Decorator to register a reward function class with a given name.
        
        Usage:
            @RewardFunctionRegistry.register("my_reward")
            class MyReward(RewardFunction):
                ...
        """
        def decorator(reward_cls):
            if name in cls._registry:
                raise ValueError(f"Reward function '{name}' already registered.")
            cls._registry[name] = reward_cls
            return reward_cls
        return decorator

    @classmethod
    def get(cls, name):
        """
        Retrieve a reward function class by name.
        
        Args:
            name (str): Name of the registered reward function.
        Returns:
            RewardFunction subclass.
        Raises:
            KeyError if not found.
        """
        if name not in cls._registry:
            raise KeyError(f"Reward function '{name}' is not registered.")
        return cls._registry[name]

    @classmethod
    def create(cls, name, *args, **kwargs):
        """
        Instantiate a reward function by name.
        
        Args:
            name (str): Name of the registered reward function.
            *args, **kwargs: Arguments to the constructor.
        Returns:
            An instance of the RewardFunction subclass.
        """
        reward_cls = cls.get(name)
        return reward_cls(*args, **kwargs)

    @classmethod
    def list_registered(cls):
        """
        List the names of all registered reward functions.
        Returns:
            List of str.
        """
        return list(cls._registry.keys())


@RewardFunctionRegistry.register("format_reward")
class FormatReward(RewardFunction):
    """
    Reward function that gives a reward if the completion follows a <think>...</think><answer>...</answer> format.
    """
    _pattern = re.compile(r"^<think>.*?</think>\s*<answer>.*?</answer>$", re.DOTALL)

    def __call__(self, completions, **kwargs):
        """
        Args:
            completions: List[List[dict]], each inner dict has a "content" field (str).
        Returns:
            List[float]: 1.0 if format matches, 0.0 otherwise, per completion.
        """
        completion_contents = [self._extract_content(completion) for completion in completions]
        return [1.0 if content and self._pattern.match(content) else 0.0 for content in completion_contents]

    @staticmethod
    def _extract_content(completion: Any) -> str:
        """
        Attempt to extract the textual content from a completion structure.
        Supports strings, mappings with a "content" key, or sequences of such mappings.
        """
        if isinstance(completion, str):
            return completion

        if isinstance(completion, Mapping):
            return completion.get("content", "")

        if isinstance(completion, Sequence) and not isinstance(completion, (str, bytes)):
            for item in completion:
                if isinstance(item, Mapping) and "content" in item:
                    return item["content"]
                if isinstance(item, str):
                    return item

        return ""

