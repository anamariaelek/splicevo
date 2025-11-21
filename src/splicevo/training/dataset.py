"""Dataset classes for splice site prediction."""

import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from typing import Optional, Union


class SpliceDataset(Dataset):
    """Dataset for splice site prediction with usage statistics."""
    
    def __init__(
        self,
        sequences: Union[np.ndarray, np.memmap],
        splice_labels: Union[np.ndarray, np.memmap],
        usage_sse: Optional[Union[np.ndarray, np.memmap]] = None
    ):
        """
        Initialize dataset.
        
        Args:
            sequences: One-hot encoded sequences (n_samples, seq_len, 4)
            splice_labels: Splice site labels (n_samples, seq_len)
            usage_sse: SSE values (n_samples, seq_len, n_conditions) or None
        """
        self.sequences = sequences
        self.splice_labels = splice_labels
        self.usage_sse = usage_sse
        
        # Validate shapes
        assert len(self.sequences) == len(self.splice_labels), \
            "Sequences and labels must have same length"
        
        if self.usage_sse is not None:
            assert len(self.sequences) == len(self.usage_sse), \
                "Sequences and usage must have same length"
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        """Get a single example."""
        # Get sequences and labels (make copies to avoid memmap issues)
        sequences = np.array(self.sequences[idx], dtype=np.float32)
        splice_labels = np.array(self.splice_labels[idx], dtype=np.int64)
        
        # Convert to tensors
        sequences = torch.from_numpy(sequences)
        splice_labels = torch.from_numpy(splice_labels)
        
        # Get usage targets if available
        if self.usage_sse is not None:
            usage_targets = np.array(self.usage_sse[idx], dtype=np.float32)
            usage_targets = torch.from_numpy(usage_targets)
        else:
            # Create dummy tensor if no usage data
            usage_targets = torch.zeros((sequences.shape[0], 1), dtype=torch.float32)
        
        return {
            'sequences': sequences,
            'splice_labels': splice_labels,
            'usage_targets': usage_targets
        }
    
    @classmethod
    def from_memmap(
        cls,
        sequences_path: Union[str, Path],
        splice_labels_path: Union[str, Path],
        usage_sse_path: Union[str, Path],
        sequences_shape: tuple,
        splice_labels_shape: tuple,
        usage_sse_shape: tuple,
        dtype: np.dtype = np.float32
    ):
        """
        Create dataset from memory-mapped files.
        
        Args:
            sequences_path: Path to sequences memmap file
            splice_labels_path: Path to splice labels memmap file
            usage_sse_path: Path to SSE memmap file
            sequences_shape: Shape of sequences array
            splice_labels_shape: Shape of splice labels array
            usage_sse_shape: Shape of usage SSE array
            dtype: Data type for arrays
            
        Returns:
            SpliceDataset with memory-mapped arrays
        """
        sequences = np.memmap(sequences_path, dtype=dtype, mode='r', shape=sequences_shape)
        splice_labels = np.memmap(splice_labels_path, dtype=np.int64, mode='r', shape=splice_labels_shape)
        usage_sse = np.memmap(usage_sse_path, dtype=dtype, mode='r', shape=usage_sse_shape)
        
        return cls(sequences, splice_labels, usage_sse, use_memmap=True)
