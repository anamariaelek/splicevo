"""Utility functions for data handling."""

import numpy as np
import os
from pathlib import Path
import json
from typing import Tuple, Optional, Union, Dict

def load_processed_data(fn):

    if fn.endswith(".npz"):
        # Load npz file
        test_data = np.load(fn)
        # Check that sequences and labels exist
        if 'sequences' not in test_data or 'labels' not in test_data:
            raise ValueError("NPZ file must contain at least 'sequences' and 'labels' arrays.")
        sequences = test_data['sequences']
        print(f"Sequences shape: {sequences.shape}")
        labels = test_data['labels']
        print(f"Labels shape: {labels.shape}")
        # Check which usage files exist and load them
        if 'usage_sse' in test_data.keys():
            sse = test_data['usage_sse']
            print(f"SSE shape: {sse.shape}")
        else:
            sse = None
        if 'usage_alpha' in test_data.keys():
            alpha = test_data['usage_alpha']
            print(f"Alpha shape: {alpha.shape}")
        else:
            alpha = None
        if 'usage_beta' in test_data.keys():
            beta = test_data['usage_beta']
            print(f"Beta shape: {beta.shape}")
        else:
            beta = None

    elif os.path.isdir(fn):
        # Load mmap files
        mmap_dir = os.path.abspath(fn)
        seq_file = os.path.join(mmap_dir, 'sequences.mmap')
        lbl_file = os.path.join(mmap_dir, 'labels.mmap')
        # Check which usage files exist
        usage_files = {
            'usage_alpha': os.path.join(mmap_dir, 'usage_alpha.mmap'),
            'usage_beta': os.path.join(mmap_dir, 'usage_beta.mmap'),
            'usage_sse': os.path.join(mmap_dir, 'usage_sse.mmap')
        }
        available_usage = {k: v for k, v in usage_files.items() if os.path.exists(v)}

        # Load memmap files
        if os.path.exists(seq_file) and os.path.exists(lbl_file):

            # Parse dtypes from metadata
            meta_path = os.path.join(mmap_dir, 'metadata.json')
            if not os.path.exists(meta_path):
                raise FileNotFoundError(f"Missing metadata.json in {mmap_dir}")
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            seq_dtype = np.dtype(meta.get('sequences_dtype', 'float32'))
            lbl_dtype = np.dtype(meta.get('labels_dtype', 'int8'))
            usage_dtype = np.dtype(meta.get('usage_dtype', 'float32'))

            # Load sequences and labels
            sequences = np.memmap(seq_file, dtype=seq_dtype, mode='r', shape=tuple(meta['sequences_shape']))
            print(f"Sequences shape: {sequences.shape}")
            labels = np.memmap(lbl_file, dtype=lbl_dtype, mode='r', shape=tuple(meta['labels_shape']))
            print(f"Labels shape: {labels.shape}")

            # Load available usage arrays
            if 'usage_sse' in available_usage:
                sse = np.memmap(
                    available_usage['usage_sse'],
                    dtype=usage_dtype, 
                    mode='r',
                    shape=tuple(meta['usage_shape'])
                )
                print(f"SSE shape: {sse.shape}")
            else:
                sse = None
            if 'usage_alpha' in available_usage:
                alpha = np.memmap(
                    available_usage['usage_alpha'],
                    dtype=usage_dtype, 
                    mode='r',
                    shape=tuple(meta['usage_shape'])
                )
                print(f"Alpha shape: {alpha.shape}")
            else:
                alpha = None
            if 'usage_beta' in available_usage:
                beta = np.memmap(
                    available_usage['usage_beta'],
                    dtype=usage_dtype, 
                    mode='r',
                    shape=tuple(meta['usage_shape'])
                )
                print(f"Beta shape: {beta.shape}")
            else:
                beta = None
            
        else:
            raise FileNotFoundError(f"Memmap files not found in {mmap_dir}.")
        
    else:
        raise ValueError("Unknown file format")
    
    return sequences, labels, alpha, beta, sse

def load_predictions(fn, keys=None):
    
    # If npz file
    if fn.endswith('.npz'):
        
        # Load npz file
        pred = np.load(fn)
        
        # If keys is None, set to all available keys
        if keys is None:
            keys = pred.keys()
        
        # Print shapes of loaded arrays
        for key in keys:
            print(f"{key}: {pred[key].shape}")

        # Load metadata
        with open(fn.replace('.npz', '.metadata.json'), 'r') as f:
            meta = json.load(f)

        # Load selected arrays
        pred_preds = pred['splice_predictions'] if 'splice_predictions' in keys else None
        pred_probs = pred['splice_probs'] if 'splice_probs' in keys else None
        pred_sse = pred['usage_sse'] if 'usage_sse' in keys else None
        true_labels = pred['labels_true'] if 'labels_true' in keys else None
        true_sse = pred['usage_sse_true'] if 'usage_sse_true' in keys else None

    # If a directory with memmap files
    elif os.path.isdir(fn):
        pred_dir = Path(fn)
        pred_preds_file = pred_dir / 'splice_predictions.mmap'
        pred_probs_file = pred_dir / 'splice_probs.mmap'
        pred_sse_file = pred_dir / 'usage_sse.mmap'
        true_labels_file = pred_dir / 'labels_true.mmap'
        true_sse_file = pred_dir / 'usage_sse_true.mmap'

        # If keys is None, set to all available keys
        if keys is None:
            keys = [
                'splice_predictions', 'splice_probs', 'usage_alpha', 
                'usage_beta', 'usage_sse', 'labels_true', 
                'usage_alpha_true', 'usage_beta_true', 'usage_sse_true'
            ]

        # Load metadata
        meta_path = pred_dir / 'metadata.json'
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing metadata.json in {pred_dir}")
        with open(meta_path, 'r') as f:
            meta = json.load(f)

        # Load selected arrays
        if pred_preds_file.exists() and 'splice_predictions' in (keys or []):
            pred_preds = np.memmap(
                pred_preds_file,
                dtype=np.dtype(meta['splice_predictions']['dtype']),
                mode='r',
                shape=tuple(meta['splice_predictions']['shape'])
            )
            print(f"Loaded splice predictions shape: {pred_preds.shape}")
        else:
            pred_preds = None
        if pred_probs_file.exists() and 'splice_probs' in (keys or []):
            pred_probs = np.memmap(
                pred_probs_file,
                dtype=np.dtype(meta['splice_probs']['dtype']),
                mode='r',
                shape=tuple(meta['splice_probs']['shape'])
            )
            print(f"Loaded splice probs shape: {pred_probs.shape}")
        else:
            pred_probs = None
        if pred_sse_file.exists() and 'usage_sse' in (keys or []):
            pred_sse = np.memmap(
                pred_sse_file,
                dtype=np.dtype(meta['usage_sse']['dtype']),
                mode='r',
                shape=tuple(meta['usage_sse']['shape'])
            )
            print(f"Loaded splice sse shape: {pred_sse.shape}")
        else:
            pred_sse = None
        if true_labels_file.exists() and 'labels_true' in (keys or []):
            true_labels = np.memmap(
                true_labels_file,
                dtype=np.dtype(meta['labels_true']['dtype']),
                mode='r',
                shape=tuple(meta['labels_true']['shape'])
            )
            print(f"Loaded true labels shape: {true_labels.shape}")
        else:
            true_labels = None
        if true_sse_file.exists() and 'usage_sse_true' in (keys or []):
            true_sse = np.memmap(
                true_sse_file,
                dtype=np.dtype(meta['usage_sse_true']['dtype']),
                mode='r',
                shape=tuple(meta['usage_sse_true']['shape'])
            )
            print(f"Loaded true sse shape: {true_sse.shape}")
        else:
            true_sse = None

    return pred_preds, pred_probs, pred_sse, meta, true_labels, true_sse

def save_to_memmap(
    data: np.ndarray,
    filepath: Path,
    dtype: Optional[np.dtype] = None
) -> np.memmap:
    """
    Save array to memory-mapped file.
    
    Args:
        data: Array to save
        filepath: Path to save memmap file
        dtype: Data type (defaults to data.dtype)
        
    Returns:
        Memory-mapped array
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    dtype = dtype or data.dtype
    memmap_array = np.memmap(
        filepath,
        dtype=dtype,
        mode='w+',
        shape=data.shape
    )
    memmap_array[:] = data[:]
    memmap_array.flush()
    
    return memmap_array


def create_memmap_dataset(
    sequences: np.ndarray,
    splice_labels: np.ndarray,
    usage_targets: Union[np.ndarray, Dict[str, np.ndarray]],
    output_dir: Path,
    prefix: str = ""
) -> Tuple[Path, Path, Union[Path, Dict[str, Path]], dict]:
    """
    Create memory-mapped files for a dataset.
    
    Args:
        sequences: DNA sequences array
        splice_labels: Splice site labels array
        usage_targets: Usage values array or dict of arrays {'alpha': ..., 'beta': ..., 'sse': ...}
        output_dir: Directory to save memmap files
        prefix: Prefix for filenames (e.g., 'train_', 'val_')
        
    Returns:
        Tuple of (sequences_path, labels_path, usage_path(s), metadata_dict)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save arrays as memmap
    seq_path = output_dir / f"{prefix}sequences.mmap"
    labels_path = output_dir / f"{prefix}splice_labels.mmap"
    
    save_to_memmap(sequences, seq_path, dtype=np.float32)
    save_to_memmap(splice_labels, labels_path, dtype=np.int64)
    
    # Handle usage targets as dict or single array
    if isinstance(usage_targets, dict):
        usage_paths = {}
        usage_metadata = {}
        for key in ['alpha', 'beta', 'sse']:
            if key in usage_targets:
                usage_path = output_dir / f"{prefix}usage_{key}.mmap"
                save_to_memmap(usage_targets[key], usage_path, dtype=np.float32)
                usage_paths[key] = usage_path
                usage_metadata[f'usage_{key}_shape'] = usage_targets[key].shape
                usage_metadata[f'usage_{key}_dtype'] = str(usage_targets[key].dtype)
    else:
        # Legacy single array support
        usage_paths = output_dir / f"{prefix}usage_targets.mmap"
        save_to_memmap(usage_targets, usage_paths, dtype=np.float32)
        usage_metadata = {
            'usage_targets_shape': usage_targets.shape,
            'usage_targets_dtype': str(usage_targets.dtype)
        }
    
    # Save metadata
    metadata = {
        'sequences_shape': sequences.shape,
        'splice_labels_shape': splice_labels.shape,
        'sequences_dtype': str(sequences.dtype),
        'splice_labels_dtype': str(splice_labels.dtype),
        **usage_metadata
    }
    
    return seq_path, labels_path, usage_paths, metadata
