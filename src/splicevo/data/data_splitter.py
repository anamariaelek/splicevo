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
        
    def _create_gc_bins(self, gc_contents: np.ndarray) -> np.ndarray:
        """Create GC content bins for stratification."""
        gc_percentiles = np.linspace(0, 100, self.gc_bins + 1)
        gc_thresholds = np.percentile(gc_contents, gc_percentiles)
        gc_bins = np.digitize(gc_contents, gc_thresholds) - 1
        # Ensure bins are in valid range
        gc_bins = np.clip(gc_bins, 0, self.gc_bins - 1)
        return gc_bins
        
    def _create_stratification_labels(self, 
                                    y: np.ndarray, 
                                    metadata: pd.DataFrame,
                                    strategy: str = 'gc_class') -> np.ndarray:
        """
        Create stratification labels based on the chosen strategy.
        
        Args:
            y: Target labels
            metadata: Metadata DataFrame
            strategy: Stratification strategy ('gc_class', 'genome_class', 'gc_genome_class')
        """
        # Make sure that y and metadata are aligned
        assert len(y) == len(metadata), "Length of y and metadata must match"

        # Create stratification labels
        if strategy == 'gc_class':
            # Stratify by GC content bins and class
            gc_bins = self._create_gc_bins(metadata['gc_content'].values)
            strat_labels = [f"{gc_bin}_{label}" for gc_bin, label in zip(gc_bins, y)]
            
        elif strategy == 'genome_class':
            # Stratify by genome and class
            strat_labels = [f"{genome}_{label}" for genome, label in 
                          zip(metadata['genome_id'], y)]
            
        elif strategy == 'gc_genome_class':
            # Stratify by GC content bins, genome, and class
            gc_bins = self._create_gc_bins(metadata['gc_content'].values)
            strat_labels = [f"{genome}_{gc_bin}_{label}" for genome, gc_bin, label in 
                          zip(metadata['genome_id'], gc_bins, y)]
        else:
            raise ValueError(f"Unknown stratification strategy: {strategy}")
            
        return np.array(strat_labels)
        
    def _filter_rare_strata(self, 
                           strat_labels: np.ndarray, 
                           min_samples: int = 2) -> np.ndarray:
        """Filter out strata with too few samples for splitting."""
        from collections import Counter
        
        strat_counts = Counter(strat_labels)
        valid_strata = {label for label, count in strat_counts.items() if count >= min_samples}
        
        if len(valid_strata) < len(strat_counts):
            n_filtered = len(strat_counts) - len(valid_strata)
            warnings.warn(f"Filtered out {n_filtered} strata with < {min_samples} samples")
            
        # Replace rare strata with generic labels
        filtered_labels = []
        for label in strat_labels:
            if label in valid_strata:
                filtered_labels.append(label)
            else:
                # Use just the class label for rare strata
                class_label = label.split('_')[-1]
                filtered_labels.append(f"rare_{class_label}")
                
        return np.array(filtered_labels)

    def _identify_orthologous_genes(self, 
                                   metadata: pd.DataFrame,
                                   test_chromosomes: Dict[str, List[str]]) -> np.ndarray:
        """
        Identify genes that are orthologous to those on test chromosomes.
        
        Args:
            metadata: Metadata DataFrame containing gene and genome information
            test_chromosomes: Dict mapping genome_id to list of chromosome names
            
        Returns:
            Boolean mask indicating which genes should be excluded from training
        """
        ortholog_mask = np.zeros(len(metadata), dtype=bool)
        
        # Get gene IDs from test chromosomes
        test_gene_ids = set()
        for genome_id, chromosomes in test_chromosomes.items():
            for chromosome in chromosomes:
                # Find genes on test chromosomes
                test_chr_mask = (
                    (metadata['genome_id'] == genome_id) & 
                    (metadata['chromosome'] == chromosome)
                )
                if 'gene_id' in metadata.columns:
                    test_chr_genes = metadata.loc[test_chr_mask, 'gene_id'].unique()
                    test_gene_ids.update(test_chr_genes)
        
        # Priority order for ortholog detection methods
        ortholog_methods = ['ortholog_group', 'gene_family', 'gene_name']
        
        for method in ortholog_methods:
            if method in metadata.columns:
                print(f"Using {method} for ortholog detection")
                
                if method == 'ortholog_group':
                    # Use ortholog group information (preferred method)
                    test_ortholog_groups = set()
                    test_mask = metadata['gene_id'].isin(test_gene_ids)
                    if test_mask.any():
                        test_groups = metadata.loc[test_mask, 'ortholog_group'].unique()
                        # Filter out singleton groups (these are genes without orthologs)
                        test_ortholog_groups = {
                            group for group in test_groups 
                            if not str(group).startswith('singleton_')
                        }
                        
                        if test_ortholog_groups:
                            # Mark orthologous genes in other locations
                            for genome_id, chromosomes in test_chromosomes.items():
                                for chromosome in chromosomes:
                                    same_chr_mask = (
                                        (metadata['genome_id'] == genome_id) & 
                                        (metadata['chromosome'] == chromosome)
                                    )
                                    ortholog_mask_temp = metadata['ortholog_group'].isin(test_ortholog_groups)
                                    ortholog_mask |= (ortholog_mask_temp & ~same_chr_mask)
                    break  # Use first available method
                    
                elif method == 'gene_family':
                    # Use gene family information
                    test_families = set()
                    test_mask = metadata['gene_id'].isin(test_gene_ids)
                    if test_mask.any():
                        test_families = set(metadata.loc[test_mask, 'gene_family'].unique())
                        # Mark genes from same families in OTHER genomes as orthologous
                        for genome_id, chromosomes in test_chromosomes.items():
                            for chromosome in chromosomes:
                                same_chr_mask = (
                                    (metadata['genome_id'] == genome_id) & 
                                    (metadata['chromosome'] == chromosome)
                                )
                                family_mask = metadata['gene_family'].isin(test_families)
                                ortholog_mask |= (family_mask & ~same_chr_mask)
                    break
                    
                elif method == 'gene_name':
                    # Use gene name similarity (assumes standardized naming across genomes)
                    test_gene_names = set()
                    test_mask = metadata['gene_id'].isin(test_gene_ids)
                    if test_mask.any():
                        test_gene_names = set(metadata.loc[test_mask, 'gene_name'].unique())
                        # Mark genes with same names in other locations
                        for genome_id, chromosomes in test_chromosomes.items():
                            for chromosome in chromosomes:
                                same_chr_mask = (
                                    (metadata['genome_id'] == genome_id) & 
                                    (metadata['chromosome'] == chromosome)
                                )
                                name_mask = metadata['gene_name'].isin(test_gene_names)
                                ortholog_mask |= (name_mask & ~same_chr_mask)
                    break
        else:
            # No ortholog information found
            warnings.warn(
                "No ortholog_group, gene_family, or gene_name information found. "
                "Cannot identify orthologous genes - proceeding with chromosome-only splitting."
            )
        
        return ortholog_mask
             
    def stratified_split(self, 
                        X: np.ndarray, 
                        y: np.ndarray, 
                        metadata: pd.DataFrame,
                        stratify_by: str = 'gc_class') -> DataSplit:
        """
        Split data using stratified sampling based on GC content and/or class.
        
        Args:
            X: Feature matrix (sequences)
            y: Target labels  
            metadata: Metadata DataFrame
            stratify_by: Stratification strategy ('gc_class', 'genome_class', 'gc_genome_class')
        """
        # Create stratification labels
        strat_labels = self._create_stratification_labels(y, metadata, stratify_by)
        strat_labels = self._filter_rare_strata(strat_labels)
        
        # First split: train+val vs test
        X_train_val, X_test, y_train_val, y_test, train_val_idx, test_idx = train_test_split(
            X, y, range(len(y)),
            test_size=self.test_size,
            stratify=strat_labels,
            random_state=self.random_state
        )
        
        train_val_metadata = metadata.iloc[train_val_idx].reset_index(drop=True)
        test_metadata = metadata.iloc[test_idx].reset_index(drop=True)
        
        # Second split: train vs val
        if self.val_size > 0:
            train_val_strat = strat_labels[train_val_idx]
            train_val_strat = self._filter_rare_strata(train_val_strat)
            
            X_train, X_val, y_train, y_val, train_idx, val_idx = train_test_split(
                X_train_val, y_train_val, range(len(y_train_val)),
                test_size=self.val_size,
                stratify=train_val_strat,
                random_state=self.random_state
            )
            
            train_metadata = train_val_metadata.iloc[train_idx].reset_index(drop=True)
            val_metadata = train_val_metadata.iloc[val_idx].reset_index(drop=True)
        else:
            X_train, y_train = X_train_val, y_train_val
            train_metadata = train_val_metadata
            X_val = np.array([])
            y_val = np.array([])
            val_metadata = pd.DataFrame()
        
        split_info = {
            'strategy': 'stratified',
            'stratify_by': stratify_by,
            'n_train': len(y_train),
            'n_val': len(y_val),
            'n_test': len(y_test)
        }
        
        return DataSplit(
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val,
            X_test=X_test, y_test=y_test,
            train_metadata=train_metadata,
            val_metadata=val_metadata,
            test_metadata=test_metadata,
            split_info=split_info
        )
              
    def chromosome_aware_split(self, 
                              X: np.ndarray, 
                              y: np.ndarray, 
                              metadata: pd.DataFrame,
                              test_chromosomes: Optional[Dict[str, List[str]]] = None,
                              val_chromosomes: Optional[Dict[str, List[str]]] = None,
                              exclude_orthologs: bool = True) -> DataSplit:
        """
        Split data ensuring specific chromosomes are used for test/validation and 
        optionally excluding orthologous genes from training.
        
        Args:
            X: Feature matrix (sequences)
            y: Target labels
            metadata: Metadata DataFrame (must contain 'genome_id' and 'chromosome' columns)
            test_chromosomes: Dict mapping genome_id to list of chromosome names for testing
            val_chromosomes: Dict mapping genome_id to list of chromosome names for validation  
            exclude_orthologs: Whether to exclude orthologous genes from training set
            
        Returns:
            DataSplit object with chromosome-aware splits
        """
        if 'chromosome' not in metadata.columns:
            raise ValueError("Metadata must contain 'chromosome' column for chromosome-aware splitting")
        if 'genome_id' not in metadata.columns:
            raise ValueError("Metadata must contain 'genome_id' column for chromosome-aware splitting")
        
        # Initialize masks
        test_mask = np.zeros(len(metadata), dtype=bool)
        val_mask = np.zeros(len(metadata), dtype=bool)
        
        # Create test set from specified chromosomes
        if test_chromosomes:
            for genome_id, chromosomes in test_chromosomes.items():
                for chromosome in chromosomes:
                    chr_mask = (
                        (metadata['genome_id'] == genome_id) & 
                        (metadata['chromosome'] == chromosome)
                    )
                    test_mask |= chr_mask
        
        # Create validation set from specified chromosomes
        if val_chromosomes:
            for genome_id, chromosomes in val_chromosomes.items():
                for chromosome in chromosomes:
                    chr_mask = (
                        (metadata['genome_id'] == genome_id) & 
                        (metadata['chromosome'] == chromosome)
                    )
                    val_mask |= chr_mask
        
        # Ensure no overlap between test and validation
        if np.any(test_mask & val_mask):
            overlapping_indices = np.where(test_mask & val_mask)[0]
            warnings.warn(
                f"Found {len(overlapping_indices)} overlapping samples between "
                "test and validation chromosomes. Removing from validation set."
            )
            val_mask &= ~test_mask
        
        # Identify orthologous genes if requested
        ortholog_mask = np.zeros(len(metadata), dtype=bool)
        if exclude_orthologs and (test_chromosomes or val_chromosomes):
            all_test_val_chromosomes = {}
            if test_chromosomes:
                all_test_val_chromosomes.update(test_chromosomes)
            if val_chromosomes:
                for genome_id, chroms in val_chromosomes.items():
                    if genome_id in all_test_val_chromosomes:
                        all_test_val_chromosomes[genome_id].extend(chroms)
                    else:
                        all_test_val_chromosomes[genome_id] = chroms
            
            ortholog_mask = self._identify_orthologous_genes(metadata, all_test_val_chromosomes)
        
        # Create training mask (everything not in test, val, or orthologous)
        train_mask = ~(test_mask | val_mask | ortholog_mask)
        
        # Extract data splits
        X_train = X[train_mask]
        y_train = y[train_mask]
        train_metadata = metadata[train_mask].reset_index(drop=True)
        
        X_test = X[test_mask]
        y_test = y[test_mask]
        test_metadata = metadata[test_mask].reset_index(drop=True)
        
        X_val = X[val_mask]
        y_val = y[val_mask]
        val_metadata = metadata[val_mask].reset_index(drop=True)
        
        # If no validation chromosomes specified and val_size > 0, 
        # create validation set from training data
        if not val_chromosomes and self.val_size > 0 and len(y_train) > 0:
            # Use stratified split on training data
            strat_labels = self._create_stratification_labels(
                y_train, train_metadata, 'gc_class'
            )
            strat_labels = self._filter_rare_strata(strat_labels)
            
            X_train, X_val_new, y_train, y_val_new, train_idx, val_idx = train_test_split(
                X_train, y_train, range(len(y_train)),
                test_size=self.val_size,
                stratify=strat_labels,
                random_state=self.random_state
            )
            
            train_metadata_new = train_metadata.iloc[train_idx].reset_index(drop=True)
            val_metadata_new = train_metadata.iloc[val_idx].reset_index(drop=True)
            
            # Update metadata
            train_metadata = train_metadata_new
            
            # Combine chromosome-based validation with stratified validation
            if len(y_val) > 0:
                X_val = np.vstack([X_val, X_val_new])
                y_val = np.concatenate([y_val, y_val_new])
                val_metadata = pd.concat([val_metadata, val_metadata_new], ignore_index=True)
            else:
                X_val = X_val_new
                y_val = y_val_new
                val_metadata = val_metadata_new
        
        split_info = {
            'strategy': 'chromosome_aware',
            'test_chromosomes': test_chromosomes,
            'val_chromosomes': val_chromosomes,
            'exclude_orthologs': exclude_orthologs,
            'n_train': len(y_train),
            'n_val': len(y_val),
            'n_test': len(y_test),
            'n_excluded_orthologs': np.sum(ortholog_mask) if exclude_orthologs else 0
        }
        
        return DataSplit(
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val,
            X_test=X_test, y_test=y_test,
            train_metadata=train_metadata,
            val_metadata=val_metadata,
            test_metadata=test_metadata,
            split_info=split_info
        )
        
    def balanced_class_split(self, 
                           X: np.ndarray, 
                           y: np.ndarray, 
                           metadata: pd.DataFrame,
                           balance_method: str = 'undersample',
                           stratify_by: str = 'gc_class') -> DataSplit:
        """
        Split data with class balancing to handle positive/negative imbalance.
        
        Args:
            X: Feature matrix (sequences)
            y: Target labels
            metadata: Metadata DataFrame  
            balance_method: Method for balancing ('undersample', 'oversample', 'smote')
            stratify_by: Stratification strategy for final split ('gc_class', etc.)
        Returns:
            DataSplit object with balanced class splits
        """
        from collections import Counter
        
        class_counts = Counter(y)
        print(f"Original class distribution: {class_counts}")
        
        if balance_method == 'undersample':
            # Undersample majority classes
            min_count = min(class_counts.values())
            
            balanced_indices = []
            for class_label in class_counts.keys():
                class_indices = np.where(y == class_label)[0]
                if len(class_indices) > min_count:
                    np.random.seed(self.random_state)
                    selected_indices = np.random.choice(class_indices, min_count, replace=False)
                else:
                    selected_indices = class_indices
                balanced_indices.extend(selected_indices)
            
            balanced_indices = np.array(balanced_indices)
            np.random.shuffle(balanced_indices)
            
            X_balanced = X[balanced_indices]
            y_balanced = y[balanced_indices]
            metadata_balanced = metadata.iloc[balanced_indices].reset_index(drop=True)
            
        elif balance_method == 'oversample':
            # Simple random oversampling
            max_count = max(class_counts.values())
            
            balanced_indices = []
            for class_label in class_counts.keys():
                class_indices = np.where(y == class_label)[0]
                if len(class_indices) < max_count:
                    np.random.seed(self.random_state)
                    additional_indices = np.random.choice(
                        class_indices, 
                        max_count - len(class_indices), 
                        replace=True
                    )
                    selected_indices = np.concatenate([class_indices, additional_indices])
                else:
                    selected_indices = class_indices
                balanced_indices.extend(selected_indices)
            
            balanced_indices = np.array(balanced_indices)
            np.random.shuffle(balanced_indices)
            
            X_balanced = X[balanced_indices]
            y_balanced = y[balanced_indices]
            metadata_balanced = metadata.iloc[balanced_indices].reset_index(drop=True)
            
        else:
            # No balancing, use original data
            X_balanced = X
            y_balanced = y
            metadata_balanced = metadata
        
        # Now perform stratified split on balanced data
        return self.stratified_split(X_balanced, y_balanced, metadata_balanced, stratify_by)
        
    def get_split_statistics(self, data_split: DataSplit) -> pd.DataFrame:
        """Generate comprehensive statistics for data splits."""
        stats = []
        
        splits = [
            ('train', data_split.y_train, data_split.train_metadata),
            ('val', data_split.y_val, data_split.val_metadata),
            ('test', data_split.y_test, data_split.test_metadata)
        ]
        
        for split_name, y_split, meta_split in splits:
            if len(y_split) == 0:
                continue
                
            # Class distribution
            unique_classes, class_counts = np.unique(y_split, return_counts=True)
            
            for class_label, count in zip(unique_classes, class_counts):
                # GC content statistics for this class
                class_mask = y_split == class_label
                class_gc = meta_split[class_mask]['gc_content']
                
                # Genome distribution for this class
                class_genomes = meta_split[class_mask]['genome_id'].value_counts()
                
                stats.append({
                    'split': split_name,
                    'class': class_label,
                    'count': count,
                    'fraction': count / len(y_split),
                    'gc_mean': class_gc.mean(),
                    'gc_std': class_gc.std(),
                    'gc_min': class_gc.min(),
                    'gc_max': class_gc.max(),
                    'n_genomes': len(class_genomes),
                    'dominant_genome': class_genomes.index[0] if len(class_genomes) > 0 else None,
                    'dominant_genome_frac': class_genomes.iloc[0] / count if len(class_genomes) > 0 else 0
                })
        
        return pd.DataFrame(stats).round(4)
