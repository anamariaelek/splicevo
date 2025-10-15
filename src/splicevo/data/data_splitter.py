"""Data splitting utilities for splice site model training."""

import numpy as np
import pandas as pd
from typing import List, Optional, Dict
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
import warnings

from .data_loader import MultiGenomeDataLoader
from ..io.splice_sites import SpliceSite


@dataclass
class DataSplit:
    """Container for train/validation/test data splits."""
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    usage_arrays_train: Optional[Dict[str, np.ndarray]]
    usage_arrays_val: Optional[Dict[str, np.ndarray]]
    usage_arrays_test: Optional[Dict[str, np.ndarray]]
    train_metadata: pd.DataFrame
    val_metadata: pd.DataFrame
    test_metadata: pd.DataFrame
    split_info: Dict


class StratifiedGCSplitter:
    """
    Advanced data splitter for splice site data that handles:
    1. Class imbalance between positive/negative examples
    2. GC content stratification for balanced representation
    3. Chromosome splitting with ortholog exclusion to avoid data leakage.
    """
    
    def __init__(self, 
                 test_size: float = 0.2,
                 val_size: float = 0.2,
                 gc_bins: int = 10,
                 random_state: int = 42):
        """
        Initialize the splitter.
        
        Args:
            test_size: Fraction of data for testing
            val_size: Fraction of remaining data for validation  
            gc_bins: Number of GC content bins for stratification
            random_state: Random seed for reproducibility
        """
        self.test_size = test_size
        self.val_size = val_size
        self.gc_bins = gc_bins
        self.random_state = random_state
        