"""Utility functions."""

import json
import logging


def setup_logging(level=logging.INFO):
    """Configure logging for the package."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def load_config(config_path):
    """
    Load configuration from JSON file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        dict: Configuration dictionary
    """
    with open(config_path, 'r') as f:
        return json.load(f)


def save_config(config, path):
    """
    Save configuration to JSON file.
    
    Args:
        config: Configuration dict
        path: Output path
    """
    with open(path, 'w') as f:
        json.dump(config, f, indent=2)
