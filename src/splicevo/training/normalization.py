"""Normalization utilities for usage values in splice site prediction."""

import numpy as np
import torch
from typing import Dict, Union, Tuple, Optional
import json
from pathlib import Path

# SSE values are already in [0,1] range and need no normalization
# This module is kept for backwards compatibility with old code
# but all functions are no-ops for SSE

def normalize_usage_arrays(
    usage_arrays: Dict[str, np.ndarray],
    method: str = 'none'
) -> Tuple[Dict[str, np.ndarray], Dict]:
    """
    SSE needs no normalization (already in [0,1] range).
    This function is kept for backwards compatibility.
    
    Args:
        usage_arrays: Dict with 'sse' array
        method: Ignored (SSE needs no normalization)
    
    Returns:
        Same arrays, identity stats
    """
    if 'sse' in usage_arrays:
        return {'sse': usage_arrays['sse'].astype(np.float32)}, {'method': 'identity', 'sse': {'transform': 'identity'}}
    return {}, {'method': 'identity'}


def denormalize_usage(
    normalized_values: Union[np.ndarray, torch.Tensor],
    stats: Dict,
    key: str = 'sse',
    sample_idx: int = None
) -> Union[np.ndarray, torch.Tensor]:
    """
    SSE needs no denormalization (identity transform).
    
    Returns:
        Same values unchanged
    """
    return normalized_values


def save_normalization_stats(stats: Dict, filepath: Union[str, Path]):
    """Save normalization statistics to JSON file."""
    with open(filepath, 'w') as f:
        json.dump(stats, f, indent=2)


def load_normalization_stats(filepath: Union[str, Path]) -> Dict:
    """Load normalization statistics from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)