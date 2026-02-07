"""Dataset classes for splice site prediction."""

import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Union

from ..data.data_loader import sparse_labels_to_dense_batch, sparse_to_dense_batch


class SpliceDataset(Dataset):
    """Dataset for splice site prediction with usage statistics.
    
    Supports index-based access for memory-efficient streaming from memmap files.
    """
    
    def __init__(
        self,
        sequences: Union[np.ndarray, np.memmap],
        labels_sparse_df: pd.DataFrame,
        window_size: int,
        usage_sparse_df: Optional[pd.DataFrame] = None,
        n_conditions: int = 0,
        species_ids: Optional[Union[np.ndarray, np.memmap]] = None,
        indices: Optional[np.ndarray] = None
    ):
        """
        Initialize dataset.
        
        Args:
            sequences: One-hot encoded sequences (n_samples, seq_len, 4)
            labels_sparse_df: Sparse labels DataFrame (sample_idx, position, label)
            window_size: Window size for reconstructing labels from sparse format
            usage_sparse_df: Sparse usage DataFrame (sample_idx, position, condition_idx, alpha, beta, sse)
            n_conditions: Number of usage conditions (0 if no usage data)
            species_ids: Species IDs (n_samples,) or None
            indices: Optional indices for subset access (enables streaming without slicing)
        """
        self.sequences = sequences
        self.labels_sparse_df = labels_sparse_df
        self.window_size = window_size
        self.usage_sparse_df = usage_sparse_df
        self.n_conditions = n_conditions
        self.species_ids = species_ids
        self.indices = indices
        
        # If indices provided, use them for length; otherwise use full array
        if self.indices is not None:
            self.length = len(self.indices)
        else:
            self.length = len(self.sequences)
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        """Get a single example with memory-efficient streaming."""
        # Map to actual index if using subset
        actual_idx = self.indices[idx] if self.indices is not None else idx
        
        # Load single sample from memmap (no full array loading)
        sequences = np.array(self.sequences[actual_idx], dtype=np.float32)
        
        # Reconstruct dense labels from sparse format for this sample
        # Use module-level function for efficiency
        splice_labels = sparse_labels_to_dense_batch(
            self.labels_sparse_df,
            np.array([actual_idx]),
            self.window_size
        )[0]  # Get first (and only) sample from batch
        
        # Convert to tensors
        sequences = torch.from_numpy(sequences)
        splice_labels = torch.from_numpy(splice_labels)
        
        # Get usage targets if available (reconstruct from sparse)
        if self.usage_sparse_df is not None and self.n_conditions > 0:
            # Reconstruct dense usage for this single sample
            usage_arrays = sparse_to_dense_batch(
                self.usage_sparse_df,
                np.array([actual_idx]),
                self.window_size,
                self.n_conditions
            )
            # Get SSE values (shape: (1, window_size, n_conditions))
            usage_targets = torch.from_numpy(usage_arrays['sse'][0])  # Remove batch dimension
        else:
            # Create dummy tensor if no usage data
            usage_targets = torch.zeros((sequences.shape[0], 1), dtype=torch.float32)
        
        # Add species ID
        if self.species_ids is not None:
            species_id = torch.tensor(self.species_ids[actual_idx], dtype=torch.long)
        else:
            species_id = torch.tensor(0, dtype=torch.long)
        
        return {
            'sequences': sequences,
            'splice_labels': splice_labels,
            'usage_targets': usage_targets,
            'species_id': species_id
        }
    
    @classmethod
    def from_memmap(
        cls,
        sequences_path: Union[str, Path],
        labels_sparse_path: Union[str, Path],
        window_size: int,
        sequences_shape: tuple,
        usage_sparse_path: Optional[Union[str, Path]] = None,
        n_conditions: int = 0,
        dtype: np.dtype = np.float32
    ):
        """
        Create dataset from memory-mapped files with sparse labels and usage.
        
        Alternatively, use the module-level load_memmap_data() function to load
        all data at once.
        
        Args:
            sequences_path: Path to sequences memmap file
            labels_sparse_path: Path to sparse labels parquet file
            window_size: Window size for label reconstruction
            sequences_shape: Shape of sequences array
            usage_sparse_path: Path to sparse usage parquet file (optional)
            n_conditions: Number of usage conditions (0 if no usage data)
            dtype: Data type for arrays
            
        Returns:
            SpliceDataset with memory-mapped arrays and sparse labels/usage
        """
        sequences = np.memmap(sequences_path, dtype=dtype, mode='r', shape=sequences_shape)
        labels_sparse_df = pd.read_parquet(labels_sparse_path)
        
        # Load usage if provided
        usage_sparse_df = None
        if usage_sparse_path is not None:
            usage_sparse_df = pd.read_parquet(usage_sparse_path)
        
        return cls(
            sequences=sequences,
            labels_sparse_df=labels_sparse_df,
            window_size=window_size,
            usage_sparse_df=usage_sparse_df,
            n_conditions=n_conditions
        )
