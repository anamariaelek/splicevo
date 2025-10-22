"""Dataset classes for splice site prediction."""

import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Union


class SpliceDataset(Dataset):
    """
    Dataset for splice site prediction with memory-mapped support.
    
    Supports both in-memory and memory-mapped arrays for efficient
    handling of large datasets.
    """
    
    def __init__(
        self,
        sequences: Union[np.ndarray, np.memmap],
        splice_labels: Union[np.ndarray, np.memmap],
        usage_targets: Union[Dict[str, np.ndarray], Dict[str, np.memmap]],
        use_memmap: bool = False
    ):
        """
        Initialize dataset.
        
        Args:
            sequences: DNA sequences (B, L, 4) or path to memmap
            splice_labels: Splice site labels (B, L)
            usage_targets: Dict of usage values {'alpha': (B, L, C), 'beta': ..., 'sse': ...}
            use_memmap: Whether arrays are memory-mapped
        """
        self.sequences = sequences
        self.splice_labels = splice_labels
        self.usage_targets = usage_targets
        self.use_memmap = use_memmap
        
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample."""
        # For memmap, convert to regular array for this sample
        # Use .copy() to ensure arrays are writable
        seq = np.array(self.sequences[idx], copy=True) if self.use_memmap else self.sequences[idx].copy()
        labels = np.array(self.splice_labels[idx], copy=True) if self.use_memmap else self.splice_labels[idx].copy()
        
        # Handle dictionary of usage arrays
        if isinstance(self.usage_targets, dict):
            # Stack the usage arrays along the last dimension
            usage_list = []
            for key in ['alpha', 'beta', 'sse']:
                usage_arr = self.usage_targets[key][idx]
                if self.use_memmap:
                    usage_arr = np.array(usage_arr, copy=True)
                else:
                    usage_arr = usage_arr.copy()
                usage_list.append(usage_arr)
            usage = np.stack(usage_list, axis=-1)  # (L, C) -> stack -> (L, 3, C) or similar
        else:
            # Legacy single array support
            usage = np.array(self.usage_targets[idx], copy=True) if self.use_memmap else self.usage_targets[idx].copy()
        
        return {
            'sequences': torch.from_numpy(seq).float(),
            'splice_labels': torch.from_numpy(labels).long(),
            'usage_targets': torch.from_numpy(usage).float()
        }
    
    @classmethod
    def from_memmap(
        cls,
        sequences_path: Union[str, Path],
        splice_labels_path: Union[str, Path],
        usage_targets_paths: Dict[str, Union[str, Path]],
        sequences_shape: tuple,
        splice_labels_shape: tuple,
        usage_targets_shape: tuple,
        dtype: np.dtype = np.float32
    ):
        """
        Create dataset from memory-mapped files.
        
        Args:
            sequences_path: Path to sequences memmap file
            splice_labels_path: Path to splice labels memmap file
            usage_targets_paths: Dict of paths to usage memmap files {'alpha': path, 'beta': path, 'sse': path}
            sequences_shape: Shape of sequences array
            splice_labels_shape: Shape of splice labels array
            usage_targets_shape: Shape of usage targets array
            dtype: Data type for arrays
            
        Returns:
            SpliceDataset with memory-mapped arrays
        """
        sequences = np.memmap(sequences_path, dtype=dtype, mode='r', shape=sequences_shape)
        splice_labels = np.memmap(splice_labels_path, dtype=np.int64, mode='r', shape=splice_labels_shape)
        
        usage_targets = {}
        for key, path in usage_targets_paths.items():
            usage_targets[key] = np.memmap(path, dtype=dtype, mode='r', shape=usage_targets_shape)
        
        return cls(sequences, splice_labels, usage_targets, use_memmap=True)
