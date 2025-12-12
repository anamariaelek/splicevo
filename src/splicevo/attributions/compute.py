"""
Attribution computation for splice site and usage predictions.

Computes input gradients (attributions) for single sequences using PyTorch.
Positive gradients indicate inputs that help the model make correct predictions.
Negative gradients indicate inputs that hurt predictions.
"""

import torch
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, List

from splicevo.model import SplicevoModel
from splicevo.attributions.attributions import AttributionCalculator


def compute_attribution_splice(
    model: SplicevoModel,
    sequence: np.ndarray,
    target: np.ndarray,
    position: int,
    device: str = 'cuda',
    verbose: bool = False
) -> np.ndarray:
    """
    Compute input gradient for a single sequence at a specific position (splice site classification).
    
    Positive gradients indicate inputs that help the model make correct predictions.
    Negative gradients indicate inputs that hurt predictions.
    
    Args:
        model: SplicevoModel instance
        sequence: One-hot encoded sequence (seq_len, 4)
        target: Target labels (seq_len,) with values 0/1/2
        position: Position to compute gradients for
        device: Device to use (e.g., 'cuda' or 'cpu')
        verbose: Print debug information
        
    Returns:
        Gradient/attribution array (seq_len, 4) - positive values help correct predictions
    """
    model.eval()
    
    # Convert to tensors
    seq_tensor = torch.from_numpy(sequence.copy()).float().to(device).unsqueeze(0)  # (1, seq_len, 4)
    target_tensor = torch.from_numpy(target.copy()).long().to(device)  # (seq_len,)

    if verbose:
        print(f"Input shapes:")
        print(f"  Sequence: {seq_tensor.shape}")
        print(f"  Targets: {target_tensor.shape}")
        print(f"  Computing for position: {position}")
        print(f"  Target class at position {position}: {target_tensor[position].item()}")
        
    # Enable gradient computation for input
    seq_tensor.requires_grad = True
    
    # Forward pass
    with torch.enable_grad():
        output = model(seq_tensor)
        splice_logits = output['splice_logits']  # (batch=1, seq_len, num_classes)
        
        # Get logits for the target position
        pos_logits = splice_logits[:, position, :]  # (1, num_classes)
        
        # Get target class at this position
        target_class = target_tensor[position]
        
        # Compute cross-entropy loss
        loss = torch.nn.functional.cross_entropy(
            pos_logits, 
            target_class.unsqueeze(0),  # Make it (1,)
            reduction='mean'
        )
        
        if verbose:
            print(f"\nLoss computation:")
            print(f"  Logits: {pos_logits.detach().cpu().numpy()}")
            print(f"  Target: {target_class.item()}")
            print(f"  Loss: {loss.item():.6f}")
        
        # Compute gradients of loss
        loss.backward()
    
    # Extract gradients
    attr_np = seq_tensor.grad.detach().cpu().numpy()[0, :, :]  # (seq_len, 4)
    
    if verbose:
        print(f"\nAttribution shape: {attr_np.shape}")
        print(f"Attribution stats:")
        print(f"  Min: {attr_np.min():.6f}")
        print(f"  Max: {attr_np.max():.6f}")
        print(f"  Mean: {attr_np.mean():.6f}")
        print(f"  Std: {attr_np.std():.6f}")
    
    return attr_np


def compute_attribution_usage(
    model: SplicevoModel,
    sequence: np.ndarray,
    target: np.ndarray,
    position: int,
    device: str = 'cuda',
    verbose: bool = False
) -> np.ndarray:
    """
    Compute input gradient for a single sequence at a specific position for all usage columns (conditions).
    
    Returns:
        Gradient/attribution array (seq_len, 4, n_conditions) - positive values help accurate predictions
    """
    model.eval()
    n_conditions = target.shape[1]
    seq_len = sequence.shape[0]
    attributions = np.zeros((seq_len, 4, n_conditions), dtype=np.float32)
    losses = np.zeros(n_conditions, dtype=np.float32)
    
    seq_tensor = torch.from_numpy(sequence.copy()).float().to(device).unsqueeze(0)  # (1, seq_len, 4)
    target_tensor = torch.from_numpy(target.copy()).float().to(device)  # (seq_len, n_conditions)
    
    for col in range(n_conditions):
        seq_tensor.requires_grad = True
        model.zero_grad()
        with torch.enable_grad():
            output = model(seq_tensor)
            usage_pred = output['usage_predictions']  # (batch=1, seq_len, n_conditions)
            pos_pred = usage_pred[:, position, col]  # (1,)
            target_val = target_tensor[position, col]
            loss = torch.nn.functional.mse_loss(
                pos_pred,
                target_val.unsqueeze(0),
                reduction='mean'
            )
            losses[col] = loss.item()
            loss.backward()
        attributions[:, :, col] = seq_tensor.grad.detach().cpu().numpy()[0, :, :]
        seq_tensor.grad.zero_()
        seq_tensor.requires_grad = False
        if verbose:
            print(f"Condition {col}: Loss={loss.item():.6f}")
    if verbose:
        print(f"Attributions shape: {attributions.shape}")
        print(f"Losses: {losses}")
    return attributions


def compute_attributions_for_sequence(
    model: SplicevoModel,
    sequences: np.ndarray,
    labels: np.ndarray,
    seq_idx: int,
    usage: Optional[np.ndarray] = None,
    output_type: str = 'splice',
    predictions: Optional[np.ndarray] = None,
    metadata: Optional[Dict] = None,
    device: str = 'cuda',
    verbose: bool = False
) -> Tuple[Dict, int, int]:
    """
    Compute attributions for all splice sites in a sequence. If predictions are provided, only compute for correctly predicted sites.
    
    Args:
        model: SplicevoModel instance
        sequences: Array of sequences (num_sequences, seq_len, 4)
        labels: Array of splice site labels (num_sequences, seq_len)
        seq_idx: Index of sequence to process
        usage: Usage/condition data (num_sequences, seq_len, n_conditions) - required if output_type='usage'
        output_type: 'splice' or 'usage'
        predictions: Model predictions for sequences (num_sequences, seq_len) - required to filter correct predictions
        metadata: Metadata DataFrame with 'strand' column indexed by seq_idx
        device: Device to use
        verbose: Print debug information
        
    Returns:
        Tuple of:
        - attrs_dict: Dictionary with attribution info for each splice site
        - count_skipped: Number of splice sites skipped (incorrect predictions)
        - count_calculated: Number of attributions successfully calculated
    """
    if output_type not in ['splice', 'usage']:
        raise ValueError(f"Unknown output_type: {output_type}, must be 'splice' or 'usage'")
    
    sequence = sequences[seq_idx]  # (seq_len, 4)
    
    if output_type == 'splice':
        target = labels[seq_idx]
        model.set_output_type('splice')
    elif output_type == 'usage':
        if usage is None:
            raise ValueError("usage array required for output_type='usage'")
        target = usage[seq_idx]
        model.set_output_type('usage')
    
    attrs_dict = {}
    count_skipped = 0
    count_calculated = 0
    
    # Find all splice sites in the sequence
    positions = np.where(labels[seq_idx] != 0)[0]
    
    for position in positions:
        site_class = labels[seq_idx, position]
        
        # Skip if predictions don't match (unless predictions not provided)
        if predictions is not None:
            pred_class = predictions[seq_idx, position]
            if site_class != pred_class:
                count_skipped += 1
                continue
        
        count_calculated += 1
        
        if verbose:
            print("-" * 65)
            print(f"Computing attributions for sequence {seq_idx}, position {position}, site_class {site_class}")
        
        # Determine site type
        site_type = "not a splice site"
        if site_class == 1:
            site_type = 'donor'
        elif site_class == 2:
            site_type = 'acceptor'
        
        # Get strand information if metadata provided
        strand = None
        if metadata is not None:
            strand = metadata.iloc[seq_idx]['strand']
        
        # Compute attributions
        if output_type == "splice":
            attr = compute_attribution_splice(
                model, 
                sequence, 
                target,
                position=position,
                device=device,
                verbose=verbose
            )
        elif output_type == "usage":
            attr = compute_attribution_usage(
                model, 
                sequence, 
                target,
                position=position,
                device=device,
                verbose=verbose
            )
        
        id = f"{seq_idx}_{position}"
        attrs_dict[id] = {
            'id': id,
            'seq_idx': seq_idx,
            'position': position,
            'sequence': sequence,
            'attr': attr,
            'site_class': site_class,
            'site_type': site_type,
            'strand': strand
        }
    
    return attrs_dict, count_skipped, count_calculated


def compute_attributions_batch(
    model: SplicevoModel,
    sequences: np.ndarray,
    labels: np.ndarray,
    seq_indices: list,
    usage: Optional[np.ndarray] = None,
    output_type: str = 'splice',
    predictions: Optional[np.ndarray] = None,
    metadata: Optional[Dict] = None,
    device: str = 'cuda',
    verbose: bool = False
) -> Tuple[Dict, int, int]:
    """
    Compute attributions for multiple sequences.
    
    Args:
        model: SplicevoModel instance
        sequences: Array of sequences
        labels: Array of splice site labels
        seq_indices: List of sequence indices to process
        usage: Usage/condition data
        output_type: 'splice' or 'usage'
        predictions: Model predictions
        metadata: Metadata with sequence information
        device: Device to use
        verbose: Print debug information
        
    Returns:
        Tuple of:
        - combined_attrs_dict: Dictionary with all attributions from all sequences
        - total_skipped: Total number of splice sites skipped
        - total_calculated: Total number of attributions calculated
    """
    combined_attrs_dict = {}
    total_skipped = 0
    total_calculated = 0
    
    for seq_idx in seq_indices:
        attrs_dict, count_skipped, count_calculated = compute_attributions_for_sequence(
            model,
            sequences,
            labels,
            seq_idx,
            usage=usage,
            output_type=output_type,
            predictions=predictions,
            metadata=metadata,
            device=device,
            verbose=verbose
        )
        combined_attrs_dict.update(attrs_dict)
        total_skipped += count_skipped
        total_calculated += count_calculated
    
    return combined_attrs_dict, total_skipped, total_calculated


# High-level convenience API for flexible attributions calculation


def compute_attributions_splice(
    model: SplicevoModel,
    sequences: np.ndarray,
    labels: np.ndarray,
    meta_df: pd.DataFrame,
    window_indices: Optional[np.ndarray] = None,
    positions: Optional[List[int]] = None,
    genomic_coords: Optional[List[Tuple[str, str, int, int, str]]] = None,
    predictions: Optional[np.ndarray] = None,
    filter_by_correct: bool = False,
    device: str = 'cuda',
    verbose: bool = False
) -> Dict:
    """
    Compute splice site attributions using flexible window/coordinate specification.
    
    High-level convenience function for the AttributionCalculator. Supports either:
    - Explicit window indices
    - Specific window and splice site positions
    - Genomic coordinates (automatically finds overlapping windows)
    
    Args:
        model: SplicevoModel instance
        sequences: One-hot encoded sequences (n_windows, seq_len, 4)
        labels: Splice site labels (n_windows, seq_len) with values 0/1/2
        meta_df: Metadata DataFrame with window information
        window_indices: Optional explicit window indices to process
        genomic_coords: Optional list of (genome_id, chromosome, start, end, strand) tuples
        positions: Optional list of position integers paired with window_indices (same length required)
        predictions: Optional model predictions for filtering correct predictions
        filter_by_correct: If True, only compute for correctly predicted sites
        device: Device to use ('cuda' or 'cpu')
        verbose: Whether to print debug information
        
    Returns:
        Dictionary with 'attributions', 'metadata', and 'summary' keys
        
    Examples:
        >>> # Compute for specific windows
        >>> result = compute_attributions_splice(
        ...     model, sequences, labels, meta_df,
        ...     window_indices=np.array([0, 5, 10])
        ... )
        
        >>> # Compute for specific splice sites in specific windows
        >>> result = compute_attributions_splice(
        ...     model, sequences, labels, meta_df,
        ...     window_indices=np.array([0, 5, 10]),
        ...     positions=[100, 95, 120]  # Each position pairs with corresponding window_index
        ... )
        
        >>> # Compute for genomic region
        >>> coords = [('human_GRCh37', '3', 142740160, 142740259, '+')]
        >>> result = compute_attributions_splice(
        ...     model, sequences, labels, meta_df,
        ...     genomic_coords=coords
        ... )
    """
    calc = AttributionCalculator(model, device=device, verbose=verbose)
    return calc.compute_splice_attributions(
        sequences, labels, meta_df,
        window_indices=window_indices,
        positions=positions,
        genomic_coords=genomic_coords,
        predictions=predictions,
        filter_by_correct=filter_by_correct
    )


def compute_attributions_usage(
    model: SplicevoModel,
    sequences: np.ndarray,
    labels: np.ndarray,
    usage: np.ndarray,
    meta_df: pd.DataFrame,
    window_indices: Optional[np.ndarray] = None,
    genomic_coords: Optional[List[Tuple[str, str, int, int, str]]] = None,
    positions: Optional[List[int]] = None,
    predictions: Optional[np.ndarray] = None,
    filter_by_correct: bool = False,
    condition_names: Optional[List[str]] = None,
    device: str = 'cuda',
    verbose: bool = False
) -> Dict:
    """
    Compute usage (condition) attributions using flexible window/coordinate specification.
    
    High-level convenience function for the AttributionCalculator. Supports either:
    - Explicit window indices
    - Specific window and splice site positions
    - Genomic coordinates (automatically finds overlapping windows)
    - Default evenly-spaced windows
    
    Args:
        model: SplicevoModel instance
        sequences: One-hot encoded sequences (n_windows, seq_len, 4)
        labels: Splice site labels (n_windows, seq_len) with values 0/1/2
        usage: Usage values (n_windows, seq_len, n_conditions)
        meta_df: Metadata DataFrame with window information
        window_indices: Optional explicit window indices to process
        positions: Optional list of position integers paired with window_indices (same length required)
        genomic_coords: Optional list of (genome_id, chromosome, start, end, strand) tuples
        predictions: Optional model predictions for filtering
        filter_by_correct: If True, only compute for correctly predicted sites
        condition_names: Optional list of condition names for labeling
        device: Device to use ('cuda' or 'cpu')
        verbose: Whether to print debug information
        
    Returns:
        Dictionary with 'attributions', 'metadata', and 'summary' keys
        
    Examples:
        >>> # Compute for specific windows
        >>> result = compute_attributions_usage(
        ...     model, sequences, labels, usage, meta_df,
        ...     window_indices=np.array([0, 5, 10])
        ... )
        
        >>> # Compute for specific splice sites in specific windows
        >>> result = compute_attributions_usage(
        ...     model, sequences, labels, usage, meta_df,
        ...     window_indices=np.array([0, 5, 10]),
        ...     positions=[100, 95, 120]  # Each position pairs with corresponding window_index
        ... )
        
        >>> # Compute for genomic region
        >>> coords = [('human_GRCh37', '3', 142740160, 142740259, '+')]
        >>> result = compute_attributions_usage(
        ...     model, sequences, labels, usage, meta_df,
        ...     genomic_coords=coords
        ... )
    """
    calc = AttributionCalculator(model, device=device, verbose=verbose)
    return calc.compute_usage_attributions(
        sequences, labels, usage, meta_df,
        window_indices=window_indices,
        positions=positions,
        genomic_coords=genomic_coords,
        predictions=predictions,
        filter_by_correct=filter_by_correct,
        condition_names=condition_names
    )
