"""
File: utils.py
------------------
Provides utility functions for the package. 
"""


import json
import logging
import torch
import torch.nn as nn
from collections import OrderedDict



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




def compare_model_structure_loose(model_a: nn.Module, model_b: nn.Module, verbose=True):
    """
    Compare structure of two models ignoring parameter names.
    Checks that the models have the same sequence of parameter shapes
    and the same sequence of module types.

    Example usage:
    ------------------------------------------------------------
    from transformers import AutoModelForCausalLM
    from your_qwen2 import Qwen2ForCausalLM

    hf_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct", torch_dtype=torch.float32)
    your_model = Qwen2ForCausalLM()

    compare_model_structure_loose(your_model, hf_model)
    ------------------------------------------------------------
    """
    structure_match = True
    shape_mismatches = []
    type_mismatches = []

    # Compare parameter shapes by order
    params_a = list(model_a.parameters())
    params_b = list(model_b.parameters())

    if len(params_a) != len(params_b):
        if verbose:
            print(f"⚠️ Different number of parameters: {len(params_a)} vs {len(params_b)}")
        structure_match = False

    for i, (p_a, p_b) in enumerate(zip(params_a, params_b)):
        if p_a.shape != p_b.shape:
            shape_mismatches.append((i, p_a.shape, p_b.shape))
            structure_match = False

    # Compare module types by order
    mods_a = [type(m).__name__ for m in model_a.modules()]
    mods_b = [type(m).__name__ for m in model_b.modules()]

    min_len = min(len(mods_a), len(mods_b))
    for i in range(min_len):
        if mods_a[i] != mods_b[i]:
            type_mismatches.append((i, mods_a[i], mods_b[i]))
            structure_match = False

    if verbose:
        print("\n🔍 Name-Independent Structure Comparison")
        print(f"✅ Same param count: {len(params_a) == len(params_b)}")
        print(f"✅ Matching shapes: {len(shape_mismatches) == 0}")
        print(f"✅ Matching module types: {len(type_mismatches) == 0}")
        if not structure_match:
            if shape_mismatches:
                print("⚠️ Shape mismatches:")
                for idx, s1, s2 in shape_mismatches[:5]:
                    print(f"  - #{idx}: {s1} vs {s2}")
            if type_mismatches:
                print("⚠️ Type mismatches:")
                for idx, t1, t2 in type_mismatches[:5]:
                    print(f"  - #{idx}: {t1} vs {t2}")
        else:
            print("🎯 Architectures are structurally equivalent (ignoring names).")

    return {
        "match": structure_match,
        "shape_mismatches": shape_mismatches,
        "type_mismatches": type_mismatches,
    }


def compare_model_tree(model_a: nn.Module, model_b: nn.Module, name_a="Model A", name_b="Model B", indent=0, max_depth=None):
    """
    Recursively compare two PyTorch model trees and print a diff of structure, types, and parameter shapes.
    Works even if module names differ (matches by order).
    """

    indent_str = "  " * indent
    tree_match = True

    # Get children (submodules)
    children_a = list(model_a.named_children())
    children_b = list(model_b.named_children())

    if max_depth is not None and indent >= max_depth:
        return True

    # Compare module type
    if type(model_a) != type(model_b):
        print(f"{indent_str}⚠️ Type mismatch: {type(model_a).__name__} vs {type(model_b).__name__}")
        tree_match = False

    # Compare parameter shapes
    params_a = list(model_a.parameters(recurse=False))
    params_b = list(model_b.parameters(recurse=False))

    if len(params_a) != len(params_b):
        print(f"{indent_str}⚠️ Param count mismatch: {len(params_a)} vs {len(params_b)}")
        tree_match = False
    else:
        for i, (p_a, p_b) in enumerate(zip(params_a, params_b)):
            if p_a.shape != p_b.shape:
                print(f"{indent_str}⚠️ Param shape mismatch #{i}: {tuple(p_a.shape)} vs {tuple(p_b.shape)}")
                tree_match = False

    # Compare children recursively
    if len(children_a) != len(children_b):
        print(f"{indent_str}⚠️ Child count mismatch: {len(children_a)} vs {len(children_b)}")
        tree_match = False

    for i, (child_a, child_b) in enumerate(zip(children_a, children_b)):
        name1, module1 = child_a
        name2, module2 = child_b
        print(f"{indent_str}↳ Layer {i}: {type(module1).__name__} vs {type(module2).__name__}")
        if type(module1) != type(module2):
            print(f"{indent_str}  ⚠️ Type mismatch: {type(module1).__name__} vs {type(module2).__name__}")
            tree_match = False

        ok = compare_model_tree(module1, module2, name1, name2, indent + 1, max_depth)
        if not ok:
            tree_match = False

    return tree_match


def summarize_model_diff(model_a, model_b, max_depth=None):
    """
    Summarize tree comparison between two models.

    Example usage:
    ------------------------------------------------------------
    from transformers import AutoModelForCausalLM
    from your_qwen2 import Qwen2ForCausalLM

    hf_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-0.5B-Instruct",
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    your_model = Qwen2ForCausalLM()

    summarize_model_diff(your_model, hf_model, max_depth=3)
    ------------------------------------------------------------
    """
    print("🔍 Comparing model trees...\n")
    match = compare_model_tree(model_a, model_b, max_depth=max_depth)
    if match:
        print("\n🎯 Models are structurally identical (within inspected depth).")
    else:
        print("\n⚠️ Differences found in structure or parameter shapes.")
