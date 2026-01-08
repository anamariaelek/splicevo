"""Flexible API for computing attributions for splice site and usage predictions.

This module provides a unified interface for calculating attributions for either:
- Explicit sequence indices (window_indices)
- Genomic coordinates (genomic_coords)

The API automatically handles the mapping between genomic coordinates and sequence indices.
"""

import numpy as np
import pandas as pd
import torch
from typing import Optional, List, Tuple, Dict, Union

from splicevo.model import SplicevoModel
from splicevo.utils.window_utils import (
    resolve_window_indices,
    build_genomic_coords_dict,
    filter_splice_sites_by_genomic_coords
)


class AttributionCalculator:
    """
    Flexible calculator for sequence attributions using gradients.
    
    Supports computing attributions for either window indices or genomic coordinates.
    Handles both splice site classification and usage prediction tasks.
    """
    
    def __init__(self, model: SplicevoModel, device: str = 'cuda', verbose: bool = False):
        """
        Initialize the attribution calculator.
        
        Args:
            model: SplicevoModel instance
            device: Device to use ('cuda' or 'cpu')
            verbose: Whether to print debug information
        """
        self.model = model
        self.device = device
        self.verbose = verbose
        model.eval()
    
    def compute_splice_attributions(
        self,
        sequences: np.ndarray,
        labels: np.ndarray,
        meta_df: pd.DataFrame,
        window_indices: Optional[np.ndarray] = None,
        genomic_coords: Optional[List[Tuple[str, str, int, int, str]]] = None,
        positions: Optional[List[Tuple[int, int]]] = None,
        predictions: Optional[np.ndarray] = None,
        filter_by_correct: bool = False,
        return_per_site: bool = True
    ) -> Dict:
        """
        Compute input gradient attributions for splice site classification.
        
        Positive gradients indicate inputs that help make correct predictions.
        Negative gradients indicate inputs that hurt predictions.
        
        Args:
            sequences: One-hot encoded sequences (n_windows, seq_len, 4)
            labels: Splice site labels (n_windows, seq_len) with values 0/1/2
            meta_df: Metadata DataFrame with window information
            window_indices: Optional explicit indices of windows to process
            genomic_coords: Optional list of (genome_id, chromosome, start, end, strand) tuples
                           to find overlapping windows
            positions: Optional list of (seq_idx, position) tuples specifying exact splice sites to compute.
                      Each tuple is (seq_idx, position) where seq_idx identifies the sequence window
                      and position is the genomic position within that window.
                      Must be paired with window_indices - each position pairs with corresponding window_index.
                      If provided, only compute attributions for these specific sites.
            predictions: Optional model predictions (n_windows, seq_len) to filter for correct sites
            filter_by_correct: If True and predictions provided, only compute for correctly predicted sites
            return_per_site: If True, return attributions aggregated per site; if False, raw per nucleotide
            
        Returns:
            Dictionary with:
            - 'attributions': Dict mapping 'seq_idx_pos' to attribution data
            - 'metadata': Dict with computation metadata
            - 'summary': Dict with statistics about computed attributions
            
        Examples:
            >>> # Compute attributions for all splice sites in specific windows
            >>> result = calc.compute_splice_attributions(
            ...     sequences, labels, meta_df,
            ...     window_indices=np.array([0, 5, 10])
            ... )
            
            >>> # Compute attributions for specific positions in specific windows
            >>> # Each position value pairs with the corresponding window_index
            >>> result = calc.compute_splice_attributions(
            ...     sequences, labels, meta_df,
            ...     window_indices=np.array([0, 5, 10]),
            ...     positions=[100, 95, 120]  # seq 0 pos 100, seq 5 pos 95, seq 10 pos 120
            ... )
            
            >>> # Compute attributions for genomic region
            >>> coords = [('human_GRCh37', '3', 142740160, 142740259, '+')]
            >>> result = calc.compute_splice_attributions(
            ...     sequences, labels, meta_df,
            ...     genomic_coords=coords
            ... )
            
            >>> # Only compute for correctly predicted sites
            >>> result = calc.compute_splice_attributions(
            ...     sequences, labels, meta_df,
            ...     window_indices=np.array([0, 5, 10]),
            ...     predictions=model_preds,
            ...     filter_by_correct=True
            ... )
        """
        # Validate positions parameter usage
        if positions is not None and window_indices is None:
            raise ValueError(
                "positions parameter requires window_indices to be provided. "
                "Use: compute_attributions_splice(..., window_indices=..., positions=...)"
            )
        
        # Process positions parameter
        positions_set = None
        if positions is not None:
            # positions can be:
            # 1. List of integers: [106, 281, 961, ...] -> paired with window_indices in order (1:1 pairing)
            # 2. Mixed list: [(30, 106), 281, 961, ...] -> tuples mean (seq_idx, pos), ints pair with window_indices
            # 3. Must have same length as window_indices
            
            if len(positions) != len(window_indices):
                raise ValueError(
                    f"positions ({len(positions)} items) and window_indices ({len(window_indices)} items) "
                    "must have the same length."
                )
            
            positions_set = set()
            for i, pos in enumerate(positions):
                seq_idx = window_indices[i]
                if isinstance(pos, (tuple, list)) and len(pos) == 2:
                    # Explicit (pos1, pos2) tuple - both are position integers
                    # Add both positions for this window_index
                    for p in pos:
                        positions_set.add((seq_idx, p))
                else:
                    # Single position integer - pair with corresponding window_index
                    positions_set.add((seq_idx, pos))
        
        window_indices = resolve_window_indices(
            meta_df, 
            window_indices=window_indices,
            genomic_coords=genomic_coords
        )
        
        # Build genomic coordinate dictionary for filtering if provided
        genomic_coords_dict = None
        if genomic_coords is not None:
            genomic_coords_dict = build_genomic_coords_dict(genomic_coords)
        
        attributions = {}
        metadata = {
            'task': 'splice_classification',
            'device': self.device,
            'model_type': type(self.model).__name__,
            'window_indices': window_indices.tolist(),
            'genomic_coords_provided': genomic_coords is not None
        }
        
        count_processed = 0
        count_skipped = 0
        
        for seq_idx in window_indices:
            if seq_idx >= len(sequences):
                continue
            
            sequence = sequences[seq_idx]  # (seq_len, 4)
            target = labels[seq_idx]  # (seq_len,)
            
            # Find splice sites in this window
            site_positions = np.where(target != 0)[0]
            
            # Filter by specific positions if provided
            if positions_set is not None:
                site_positions = np.array([p for p in site_positions if (seq_idx, p) in positions_set])
            
            # Filter by genomic coordinates if provided
            if genomic_coords_dict is not None:
                row = meta_df.iloc[seq_idx]
                donor_mask = target == 1
                acceptor_mask = target == 2
                donor_pos = np.where(donor_mask)[0]
                acceptor_pos = np.where(acceptor_mask)[0]
                
                # If positions filter was applied, use that subset
                if positions_set is not None:
                    donor_pos = np.array([p for p in donor_pos if (seq_idx, p) in positions_set])
                    acceptor_pos = np.array([p for p in acceptor_pos if (seq_idx, p) in positions_set])
                
                donor_pos_filtered, acceptor_pos_filtered = filter_splice_sites_by_genomic_coords(
                    donor_pos, acceptor_pos,
                    window_start=int(row['window_start']),
                    window_strand=str(row['strand']),
                    genomic_coords_dict=genomic_coords_dict,
                    genome_id=str(row.get('genome_id', '')),
                    chromosome=str(row['chromosome'])
                )
                
                site_positions = np.concatenate([donor_pos_filtered, acceptor_pos_filtered])
            
            for position in site_positions:
                site_class = target[position]
                
                # Skip if predictions filter is enabled
                if filter_by_correct and predictions is not None:
                    if predictions[seq_idx, position] != site_class:
                        count_skipped += 1
                        continue
                
                attr = self._compute_splice_attribution(
                    sequence, target, position
                )
                
                site_id = f"{seq_idx}_{position}"
                
                # Calculate genomic coordinates
                window_start = int(meta_df.iloc[seq_idx]['window_start'])
                window_end = int(meta_df.iloc[seq_idx]['window_end'])
                strand = str(meta_df.iloc[seq_idx]['strand'])
                
                # Position within window maps to genomic coordinate
                # Always add position to window_start, regardless of strand
                genomic_coord = window_start + position
                
                attributions[site_id] = {
                    'id': site_id,
                    'seq_idx': int(seq_idx),
                    'position': int(position),
                    'site_class': int(site_class),
                    'site_type': 'donor' if site_class == 1 else 'acceptor',
                    'sequence': sequence,
                    'attribution': attr,
                    'metadata': {
                        'genome_id': meta_df.iloc[seq_idx].get('genome_id'),
                        'chromosome': str(meta_df.iloc[seq_idx]['chromosome']),
                        'window_start': window_start,
                        'window_end': window_end,
                        'strand': strand,
                        'genomic_coord': int(genomic_coord),
                    }
                }
                
                count_processed += 1
        
        return {
            'attributions': attributions,
            'metadata': metadata,
            'summary': {
                'total_processed': count_processed,
                'total_skipped': count_skipped,
                'n_windows': len(window_indices)
            }
        }
    
    def compute_usage_attributions(
        self,
        sequences: np.ndarray,
        labels: np.ndarray,
        usage: np.ndarray,
        meta_df: pd.DataFrame,
        window_indices: Optional[np.ndarray] = None,
        genomic_coords: Optional[List[Tuple[str, str, int, int, str]]] = None,
        positions: Optional[List[Tuple[int, int]]] = None,
        predictions: Optional[np.ndarray] = None,
        filter_by_correct: bool = False,
        condition_names: Optional[List[str]] = None,
        share_attributions_across_conditions: bool = False
    ) -> Dict:
        """
        Compute input gradient attributions for usage (condition) predictions.
        
        Args:
            sequences: One-hot encoded sequences (n_windows, seq_len, 4)
            labels: Splice site labels (n_windows, seq_len) with values 0/1/2
            usage: Usage values (n_windows, seq_len, n_conditions)
            meta_df: Metadata DataFrame with window information
            window_indices: Optional explicit indices of windows to process
            genomic_coords: Optional list of genomic coordinate tuples to find overlapping windows
            positions: Optional list of (seq_idx, position) tuples specifying exact splice sites to compute
            predictions: Optional model predictions (n_windows, seq_len, n_conditions) to filter
            filter_by_correct: If True, only process sites with correct predictions
            condition_names: Optional list of condition names for labeling
            share_attributions_across_conditions: If True, compute all conditions in a single forward pass
                                                 using shared input gradients (faster, ~N_CONDITIONS speedup)
            
        Returns:
            Dictionary with attributions, metadata, and summary statistics
        """
        # Validate positions parameter usage
        if positions is not None and window_indices is None:
            raise ValueError(
                "positions parameter requires window_indices to be provided. "
                "Use: compute_attributions_usage(..., window_indices=..., positions=...)"
            )
        
        # Process positions parameter
        positions_set = None
        if positions is not None:
            # positions can be:
            # 1. List of integers: [106, 281, 961, ...] -> paired with window_indices in order (1:1 pairing)
            # 2. Mixed list: [(30, 106), 281, 961, ...] -> tuples mean (pos1, pos2), ints pair with window_indices
            # 3. Must have same length as window_indices
            
            if len(positions) != len(window_indices):
                raise ValueError(
                    f"positions ({len(positions)} items) and window_indices ({len(window_indices)} items) "
                    "must have the same length."
                )
            
            positions_set = set()
            for i, pos in enumerate(positions):
                seq_idx = window_indices[i]
                if isinstance(pos, (tuple, list)) and len(pos) == 2:
                    # Explicit (pos1, pos2) tuple - both are position integers
                    # Add both positions for this window_index
                    for p in pos:
                        positions_set.add((seq_idx, p))
                else:
                    # Single position integer - pair with corresponding window_index
                    positions_set.add((seq_idx, pos))
        
        window_indices = resolve_window_indices(
            meta_df,
            window_indices=window_indices,
            genomic_coords=genomic_coords
        )
        
        # Build genomic coordinate dictionary for filtering if provided
        genomic_coords_dict = None
        if genomic_coords is not None:
            genomic_coords_dict = build_genomic_coords_dict(genomic_coords)
        
        attributions = {}
        metadata = {
            'task': 'usage_prediction',
            'device': self.device,
            'model_type': type(self.model).__name__,
            'window_indices': window_indices.tolist(),
            'n_conditions': usage.shape[2] if usage.ndim == 3 else 1,
            'genomic_coords_provided': genomic_coords is not None
        }
        
        if condition_names is not None:
            metadata['condition_names'] = condition_names
        
        count_processed = 0
        count_skipped = 0
        
        for seq_idx in window_indices:
            if seq_idx >= len(sequences):
                continue
            
            sequence = sequences[seq_idx]  # (seq_len, 4)
            target = usage[seq_idx]  # (seq_len, n_conditions)
            seq_labels = labels[seq_idx]  # (seq_len,) for finding sites
            
            # Find splice sites in this window
            site_positions = np.where(seq_labels != 0)[0]
            
            # Filter by specific positions if provided
            if positions_set is not None:
                site_positions = np.array([p for p in site_positions if (seq_idx, p) in positions_set])
            
            # Filter by genomic coordinates if provided
            if genomic_coords_dict is not None:
                row = meta_df.iloc[seq_idx]
                donor_mask = seq_labels == 1
                acceptor_mask = seq_labels == 2
                donor_pos = np.where(donor_mask)[0]
                acceptor_pos = np.where(acceptor_mask)[0]
                
                # If positions filter was applied, use that subset
                if positions_set is not None:
                    donor_pos = np.array([p for p in donor_pos if (seq_idx, p) in positions_set])
                    acceptor_pos = np.array([p for p in acceptor_pos if (seq_idx, p) in positions_set])
                
                donor_pos_filtered, acceptor_pos_filtered = filter_splice_sites_by_genomic_coords(
                    donor_pos, acceptor_pos,
                    window_start=int(row['window_start']),
                    window_strand=str(row['strand']),
                    genomic_coords_dict=genomic_coords_dict,
                    genome_id=str(row.get('genome_id', '')),
                    chromosome=str(row['chromosome'])
                )
                
                site_positions = np.concatenate([donor_pos_filtered, acceptor_pos_filtered])
            
            for position in site_positions:
                site_class = seq_labels[position]
                
                # Skip if predictions filter is enabled
                if filter_by_correct and predictions is not None:
                    # For usage, filter by splice site classification correctness (same as splice)
                    if predictions[seq_idx, position] != site_class:
                        count_skipped += 1
                        continue
                
                attr = self._compute_usage_attribution(
                    sequence, target, position,
                    share_attributions_across_conditions=share_attributions_across_conditions
                )
                
                site_id = f"{seq_idx}_{position}"
                
                # Calculate genomic coordinates
                window_start = int(meta_df.iloc[seq_idx]['window_start'])
                window_end = int(meta_df.iloc[seq_idx]['window_end'])
                strand = str(meta_df.iloc[seq_idx]['strand'])
                
                # Position within window maps to genomic coordinate
                # Always add position to window_start, regardless of strand
                genomic_coord = window_start + position
                
                attributions[site_id] = {
                    'id': site_id,
                    'seq_idx': int(seq_idx),
                    'position': int(position),
                    'site_class': int(site_class),
                    'site_type': 'donor' if site_class == 1 else 'acceptor',
                    'sequence': sequence,
                    'attribution': attr,  # (seq_len, 4, n_conditions)
                    'metadata': {
                        'genome_id': meta_df.iloc[seq_idx].get('genome_id'),
                        'chromosome': str(meta_df.iloc[seq_idx]['chromosome']),
                        'window_start': window_start,
                        'window_end': window_end,
                        'strand': strand,
                        'genomic_coord': int(genomic_coord),
                    }
                }
                
                count_processed += 1
        
        return {
            'attributions': attributions,
            'metadata': metadata,
            'summary': {
                'total_processed': count_processed,
                'total_skipped': count_skipped,
                'n_windows': len(window_indices)
            }
        }
    
    def _compute_splice_attribution(
        self,
        sequence: np.ndarray,
        target: np.ndarray,
        position: int
    ) -> np.ndarray:
        """
        Compute input gradient attribution for a single splice site.
        
        Args:
            sequence: One-hot encoded sequence (seq_len, 4)
            target: Target labels (seq_len,)
            position: Position to compute attributions for
            
        Returns:
            Attribution array (seq_len, 4)
        """
        self.model.set_output_type('splice')
        
        seq_tensor = torch.from_numpy(sequence.copy()).float().to(self.device).unsqueeze(0)
        target_tensor = torch.from_numpy(target.copy()).long().to(self.device)
        
        seq_tensor.requires_grad = True
        
        with torch.enable_grad():
            output = self.model(seq_tensor)
            splice_logits = output['splice_logits']  # (1, seq_len, num_classes)
            
            pos_logits = splice_logits[:, position, :]
            target_class = target_tensor[position]
            
            loss = torch.nn.functional.cross_entropy(
                pos_logits,
                target_class.unsqueeze(0),
                reduction='mean'
            )
            
            loss.backward()
        
        attr = seq_tensor.grad.detach().cpu().numpy()[0, :, :]
        return attr
    
    def _compute_usage_attribution(
        self,
        sequence: np.ndarray,
        target: np.ndarray,
        position: int,
        share_attributions_across_conditions: bool = False
    ) -> np.ndarray:
        """
        Compute input gradient attribution for usage prediction at a single site.
        
        Args:
            sequence: One-hot encoded sequence (seq_len, 4)
            target: Target values (seq_len, n_conditions)
            position: Position to compute attributions for
            share_attributions_across_conditions: If True, compute all conditions in a single forward pass.
                                                  Faster but assumes shared input gradients across conditions.
            
        Returns:
            Attribution array (seq_len, 4, n_conditions)
        """
        self.model.set_output_type('usage')
        
        n_conditions = target.shape[1]
        seq_len = sequence.shape[0]
        attributions = np.zeros((seq_len, 4, n_conditions), dtype=np.float32)
        
        seq_tensor = torch.from_numpy(sequence.copy()).float().to(self.device).unsqueeze(0)
        target_tensor = torch.from_numpy(target.copy()).float().to(self.device)
        
        if share_attributions_across_conditions:
            # Optimized: single forward pass for all conditions
            seq_tensor.requires_grad = True
            self.model.zero_grad()
            
            with torch.enable_grad():
                output = self.model(seq_tensor)
                usage_pred = output['usage_predictions']  # (1, seq_len, n_conditions)
                pos_pred = usage_pred[:, position, :]  # (1, n_conditions)
                target_vals = target_tensor[position, :]  # (n_conditions,)
                
                # Compute loss for all conditions at once
                loss = torch.nn.functional.mse_loss(
                    pos_pred,
                    target_vals.unsqueeze(0),
                    reduction='mean'
                )
                
                loss.backward()
            
            # Extract gradients once for all conditions
            grad_all = seq_tensor.grad.detach().cpu().numpy()[0, :, :]  # (seq_len, 4)
            
            # Replicate gradient across all conditions (all conditions share same input gradients)
            for col in range(n_conditions):
                attributions[:, :, col] = grad_all
        else:
            # Original: separate forward pass per condition
            for col in range(n_conditions):
                seq_tensor.requires_grad = True
                self.model.zero_grad()
                
                with torch.enable_grad():
                    output = self.model(seq_tensor)
                    usage_pred = output['usage_predictions']  # (1, seq_len, n_conditions)
                    pos_pred = usage_pred[:, position, col]  # (1,)
                    target_val = target_tensor[position, col]
                    
                    loss = torch.nn.functional.mse_loss(
                        pos_pred,
                        target_val.unsqueeze(0),
                        reduction='mean'
                    )
                    
                    loss.backward()
                
                attributions[:, :, col] = seq_tensor.grad.detach().cpu().numpy()[0, :, :]
                seq_tensor.grad.zero_()
                seq_tensor.requires_grad = False
        
        return attributions
