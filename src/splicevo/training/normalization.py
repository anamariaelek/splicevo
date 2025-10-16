"""Normalization utilities for usage values in splice site prediction."""

import numpy as np
import torch
from typing import Dict, Union, Tuple
import json
from pathlib import Path


def normalize_usage_arrays(
    usage_arrays: Dict[str, np.ndarray],
    method: str = 'per_sample_cpm'
) -> Tuple[Dict[str, np.ndarray], Dict]:
    """
    Normalize usage arrays to remove library size effects and stabilize variance.
    
    This is the standard preprocessing step for training. It:
    1. Normalizes per-sample (like CPM) to remove library size effects
    2. Applies log1p transform to stabilize variance
    3. Standardizes globally using ALL data statistics
    
    Args:
        usage_arrays: Dict with 'alpha', 'beta', 'sse' arrays
                     Shape: (n_samples, seq_len, n_conditions)
        method: Normalization method ('per_sample_cpm' or 'global')
    
    Returns:
        normalized_arrays: Dict with normalized arrays (same keys/shapes)
        stats: Dict with normalization statistics (for denormalization)
    """
    alpha = usage_arrays['alpha']
    beta = usage_arrays['beta']
    sse = usage_arrays['sse']
    
    if method == 'per_sample_cpm':
        # Per-sample normalization (removes library size effects)
        normalized_alpha, alpha_stats = _normalize_per_sample_cpm(alpha)
        normalized_beta, beta_stats = _normalize_per_sample_cpm(beta)
        
        normalized_arrays = {
            'alpha': normalized_alpha,
            'beta': normalized_beta,
            'sse': sse.astype(np.float32)  # Keep as-is (already in [0,1])
        }
        
        stats = {
            'alpha': alpha_stats,
            'beta': beta_stats,
            'sse': {'transform': 'identity'},
            'method': method
        }
        
    elif method == 'global':
        # Simple global normalization (preserves sample differences)
        normalized_alpha, alpha_stats = _normalize_global(alpha)
        normalized_beta, beta_stats = _normalize_global(beta)
        
        normalized_arrays = {
            'alpha': normalized_alpha,
            'beta': normalized_beta,
            'sse': sse.astype(np.float32)
        }
        
        stats = {
            'alpha': alpha_stats,
            'beta': beta_stats,
            'sse': {'transform': 'identity'},
            'method': method
        }
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return normalized_arrays, stats


def _normalize_per_sample_cpm(arr: np.ndarray) -> Tuple[np.ndarray, Dict]:
    """Normalize per-sample using CPM-like approach."""
    n_samples, seq_len, n_conditions = arr.shape
    
    normalized = np.full_like(arr, np.nan, dtype=np.float32)  # Initialize with NaN
    sample_scales = np.zeros((n_samples, n_conditions), dtype=np.float32)
    
    scale_factor = 1e6  # CPM scaling
    
    print(f"  Step 1: Per-sample CPM normalization...")
    # Step 1: Per-sample normalization
    for i in range(n_samples):
        for c in range(n_conditions):
            valid_mask = ~np.isnan(arr[i, :, c])
            
            if valid_mask.sum() > 0:
                total = np.sum(arr[i, :, c][valid_mask])
                sample_scales[i, c] = total
                
                # Only normalize valid (non-NaN) values
                if total > 0:
                    normalized[i, valid_mask, c] = (arr[i, valid_mask, c] / total) * scale_factor
                else:
                    # If total is 0, keep as 0 (not NaN)
                    normalized[i, valid_mask, c] = 0.0
            else:
                # No valid values in this sample/condition
                sample_scales[i, c] = 1.0
                # normalized remains NaN for this slice
    
    # Step 2: Log transform + global standardization (using ALL data)
    print(f"  Step 2: Log1p + standardization (using all data)...")
    valid = ~np.isnan(normalized)
    
    if valid.sum() == 0:
        raise ValueError("No valid values to normalize!")
    
    log_values = np.log1p(normalized[valid])
    
    log_mean = float(np.mean(log_values))
    log_std = float(np.std(log_values))
    
    if log_std < 1e-10:
        log_std = 1.0
        print(f"    Warning: log_std is very small, setting to 1.0")
    
    # Apply to all valid data
    final = np.full_like(normalized, np.nan, dtype=np.float32)
    final[valid] = (np.log1p(normalized[valid]) - log_mean) / (log_std + 1e-8)
    
    stats = {
        'transform': 'per_sample_cpm_log1p_zscore',
        'scale_factor': float(scale_factor),
        'log_mean': log_mean,
        'log_std': log_std,
        'sample_scales': sample_scales.tolist()
    }
    
    print(f"    → log_mean={log_mean:.3f}, log_std={log_std:.3f}")
    print(f"    → Valid values: {valid.sum():,} / {arr.size:,} ({100*valid.sum()/arr.size:.1f}%)")
    
    return final, stats


def _normalize_global(arr: np.ndarray) -> Tuple[np.ndarray, Dict]:
    """Simple global log1p + standardization (preserves sample variation)."""
    valid = ~np.isnan(arr)
    
    if valid.sum() == 0:
        raise ValueError("No valid values to normalize!")
    
    log_values = np.log1p(arr[valid])
    
    log_mean = float(np.mean(log_values))
    log_std = float(np.std(log_values))
    
    if log_std < 1e-10:
        log_std = 1.0
        print(f"  Warning: log_std is very small, setting to 1.0")
    
    normalized = np.full_like(arr, np.nan, dtype=np.float32)
    normalized[valid] = (np.log1p(arr[valid]) - log_mean) / (log_std + 1e-8)
    
    stats = {
        'transform': 'log1p_zscore',
        'log_mean': log_mean,
        'log_std': log_std,
        'original_min': float(np.min(arr[valid])),
        'original_max': float(np.max(arr[valid]))
    }
    
    print(f"  log_mean={log_mean:.3f}, log_std={log_std:.3f}")
    print(f"  Valid values: {valid.sum():,} / {arr.size:,} ({100*valid.sum()/arr.size:.1f}%)")
    
    return normalized, stats


def denormalize_usage(
    normalized_values: Union[np.ndarray, torch.Tensor],
    stats: Dict,
    key: str,
    sample_idx: int = None
) -> Union[np.ndarray, torch.Tensor]:
    """
    Denormalize predictions back to original scale.
    
    Args:
        normalized_values: Normalized predictions (seq_len, n_conditions) or batched
        stats: Full stats dict from normalize_usage_arrays
        key: 'alpha', 'beta', or 'sse'
        sample_idx: Sample index (required for per_sample_cpm method)
    
    Returns:
        Denormalized values (original read counts)
    """
    transform = stats[key]['transform']
    
    if transform == 'identity':
        return normalized_values
    
    elif transform == 'per_sample_cpm_log1p_zscore':
        if sample_idx is None:
            raise ValueError("sample_idx required for per_sample_cpm denormalization")
        
        is_torch = isinstance(normalized_values, torch.Tensor)
        
        log_mean = stats[key]['log_mean']
        log_std = stats[key]['log_std']
        scale_factor = stats[key]['scale_factor']
        sample_scales = np.array(stats[key]['sample_scales'])
        
        if is_torch:
            # Reverse z-score
            log_vals = normalized_values * log_std + log_mean
            # Reverse log1p
            cpm_vals = torch.expm1(log_vals)
            # Reverse CPM
            scales = torch.tensor(
                sample_scales[sample_idx],
                device=normalized_values.device,
                dtype=normalized_values.dtype
            )
            original = (cpm_vals / scale_factor) * scales
            original = torch.clamp(original, min=0.0)
        else:
            log_vals = normalized_values * log_std + log_mean
            cpm_vals = np.expm1(log_vals)
            scales = sample_scales[sample_idx]
            original = (cpm_vals / scale_factor) * scales
            original = np.maximum(original, 0.0)
        
        return original
    
    elif transform == 'log1p_zscore':
        is_torch = isinstance(normalized_values, torch.Tensor)
        
        log_mean = stats[key]['log_mean']
        log_std = stats[key]['log_std']
        
        if is_torch:
            log_vals = normalized_values * log_std + log_mean
            original = torch.expm1(log_vals)
            original = torch.clamp(original, min=0.0)
        else:
            log_vals = normalized_values * log_std + log_mean
            original = np.expm1(log_vals)
            original = np.maximum(original, 0.0)
        
        return original
    
    else:
        raise ValueError(f"Unknown transform: {transform}")


def save_normalization_stats(stats: Dict, filepath: Union[str, Path]):
    """Save normalization statistics to JSON file."""
    with open(filepath, 'w') as f:
        json.dump(stats, f, indent=2)


def load_normalization_stats(filepath: Union[str, Path]) -> Dict:
    """Load normalization statistics from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)