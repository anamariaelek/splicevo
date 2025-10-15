"""PyTorch Dataset for splice site data."""

import torch
from torch.utils.data import Dataset
import numpy as np


class SpliceDataset(Dataset):
    """
    PyTorch Dataset for splice site sequences with labels and usage data.
    
    Loads data from numpy arrays created by MultiGenomeDataLoader.to_arrays()
    """
    
    def __init__(
        self,
        sequences: np.ndarray,
        labels: np.ndarray,
        usage_arrays: dict,
        transform=None
    ):
        """
        Initialize dataset.
        
        Args:
            sequences: One-hot encoded sequences (n_samples, seq_len, 4)
            labels: Splice site labels (n_samples, seq_len) with values 0, 1, 2
            usage_arrays: Dictionary with keys 'alpha', 'beta', 'sse'
                         Each has shape (n_samples, seq_len, n_conditions)
            transform: Optional transforms to apply to sequences
        """
        self.sequences = torch.from_numpy(sequences).float()
        self.labels = torch.from_numpy(labels).long()
        
        # Stack usage arrays into single tensor
        self.usage = torch.stack([
            torch.from_numpy(usage_arrays['alpha']).float(),
            torch.from_numpy(usage_arrays['beta']).float(),
            torch.from_numpy(usage_arrays['sse']).float()
        ], dim=-1)  # Shape: (n_samples, seq_len, n_conditions, 3)
        
        self.transform = transform
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        usage = self.usage[idx]
        
        if self.transform:
            sequence = self.transform(sequence)
        
        return {
            'sequences': sequence,
            'splice_labels': label,
            'usage_targets': usage
        }
