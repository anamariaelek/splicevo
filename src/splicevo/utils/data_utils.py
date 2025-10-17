"""Utility functions for data handling."""

import numpy as np
from pathlib import Path
from typing import Tuple, Optional


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
    usage_targets: np.ndarray,
    output_dir: Path,
    prefix: str = ""
) -> Tuple[Path, Path, Path, dict]:
    """
    Create memory-mapped files for a dataset.
    
    Args:
        sequences: DNA sequences array
        splice_labels: Splice site labels array
        usage_targets: Usage values array
        output_dir: Directory to save memmap files
        prefix: Prefix for filenames (e.g., 'train_', 'val_')
        
    Returns:
        Tuple of (sequences_path, labels_path, usage_path, metadata_dict)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save arrays as memmap
    seq_path = output_dir / f"{prefix}sequences.mmap"
    labels_path = output_dir / f"{prefix}splice_labels.mmap"
    usage_path = output_dir / f"{prefix}usage_targets.mmap"
    
    save_to_memmap(sequences, seq_path, dtype=np.float32)
    save_to_memmap(splice_labels, labels_path, dtype=np.int64)
    save_to_memmap(usage_targets, usage_path, dtype=np.float32)
    
    # Save metadata
    metadata = {
        'sequences_shape': sequences.shape,
        'splice_labels_shape': splice_labels.shape,
        'usage_targets_shape': usage_targets.shape,
        'sequences_dtype': str(sequences.dtype),
        'splice_labels_dtype': str(splice_labels.dtype),
        'usage_targets_dtype': str(usage_targets.dtype)
    }
    
    return seq_path, labels_path, usage_path, metadata
