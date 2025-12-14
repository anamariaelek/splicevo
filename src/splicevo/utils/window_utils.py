"""Utilities for window-based sequence operations and genomic coordinate resolution."""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict


def find_overlapping_windows(
    meta_df: pd.DataFrame,
    genomic_coords: List[Tuple[str, str, int, int, str]]
) -> np.ndarray:
    """
    Find all windows overlapping with specified genomic coordinates.
    
    Args:
        meta_df: DataFrame with window metadata containing 'genome_id', 'chromosome', 
                 'window_start', 'window_end', 'strand'
        genomic_coords: List of (genome_id, chromosome, start, end, strand) tuples.
                       Strand can be '+', '-', or '*' (match any strand)
        
    Returns:
        Array of window indices that overlap with any of the specified coordinates
        
    Examples:
        >>> genomic_targets = [
        ...     ('human_GRCh37', '3', 142740160, 142740259, '+'),
        ...     ('mouse_GRCm38', '5', 124543487, 124543507, '*'),
        ... ]
        >>> indices = find_overlapping_windows(meta_df, genomic_targets)
    """
    overlapping_indices = []

    for genome_id, chrom, coord_start, coord_end, strand in genomic_coords:
        if strand in ['+', '-']:          
            # Find windows on the same chromosome and strand
            matches = meta_df[
                (meta_df['genome_id'].astype(str) == str(genome_id)) &
                (meta_df['chromosome'].astype(str) == str(chrom)) &
                (meta_df['strand'].astype(str) == str(strand))
            ]
        else:
            # Strand not specified, match any strand
            matches = meta_df[
                (meta_df['genome_id'].astype(str) == str(genome_id)) &
                (meta_df['chromosome'].astype(str) == str(chrom))
            ]
        # Find overlapping windows
        for idx, row in matches.iterrows():
            window_start = int(row['window_start'])
            window_end = int(row['window_end'])
            
            # Check for overlap
            if window_start < coord_end and window_end > coord_start:
                overlapping_indices.append(idx)
    
    return np.array(sorted(set(overlapping_indices)))


def resolve_window_indices(
    meta_df: pd.DataFrame,
    window_indices: Optional[np.ndarray] = None,
    genomic_coords: Optional[List[Tuple[str, str, int, int, str]]] = None
) -> np.ndarray:
    """
    Resolve window indices from either explicit indices or genomic coordinates.
    
    This function provides a flexible interface to specify which windows to analyze:
    - If window_indices are provided, use them directly
    - If genomic_coords are provided, find overlapping windows
    - If neither, return evenly spaced windows across the dataset
    
    Args:
        meta_df: DataFrame with window metadata
        window_indices: Optional array of explicit window indices to use
        genomic_coords: Optional list of genomic coordinate tuples (genome_id, chrom, start, end, strand)
        default_n: Default number of evenly spaced windows if neither indices nor coords provided
        
    Returns:
        Array of window indices to process
        
    Raises:
        ValueError: If no windows are found from genomic coordinates
        
    Examples:
        >>> # Use explicit indices
        >>> indices = resolve_window_indices(meta_df, window_indices=np.array([0, 5, 10]))
        
        >>> # Use genomic coordinates
        >>> coords = [('human_GRCh37', '3', 142740160, 142740259, '+')]
        >>> indices = resolve_window_indices(meta_df, genomic_coords=coords)
        
        >>> # Use defaults
        >>> indices = resolve_window_indices(meta_df, default_n=10)
    """
    if window_indices is not None:
        return np.asarray(window_indices, dtype=int)
    
    if genomic_coords is not None:
        indices = find_overlapping_windows(meta_df, genomic_coords)
        if len(indices) == 0:
            raise ValueError("No windows found overlapping the specified genomic coordinates.")
        return indices
    
    # Default: use all windows
    return np.asarray(range(meta_df.shape[0]))


def build_genomic_coords_dict(
    genomic_coords: List[Tuple[str, str, int, int, str]]
) -> Dict[Tuple[str, str, str], List[Tuple[int, int]]]:
    """
    Build a dictionary mapping (genome_id, chromosome, strand) to list of coordinate ranges.
    
    This is useful for filtering splice sites by genomic location within a window.
    
    Args:
        genomic_coords: List of (genome_id, chromosome, start, end, strand) tuples
        
    Returns:
        Dictionary with (genome_id, chromosome, strand) tuples as keys and
        lists of (start, end) coordinate tuples as values
        
    Note:
        If strand is '*', expands to both '+' and '-'
    """
    coords_dict = {}
    
    for genome_id, chrom, coord_start, coord_end, strand in genomic_coords:
        if strand not in ['+', '-']:
            # Strand not specified, add to both
            for s in ['+', '-']:
                key = (str(genome_id), str(chrom), str(s))
                if key not in coords_dict:
                    coords_dict[key] = []
                coords_dict[key].append((coord_start, coord_end))
        else:
            key = (str(genome_id), str(chrom), str(strand))
            if key not in coords_dict:
                coords_dict[key] = []
            coords_dict[key].append((coord_start, coord_end))
    
    return coords_dict


def filter_splice_sites_by_genomic_coords(
    donor_positions: np.ndarray,
    acceptor_positions: np.ndarray,
    window_start: int,
    window_strand: str,
    genomic_coords_dict: Dict[Tuple[str, str, str], List[Tuple[int, int]]],
    genome_id: str,
    chromosome: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filter splice site positions to only include those within specified genomic coordinate ranges.
    
    For each site, computes its genomic position and checks if it falls within
    any of the specified coordinate ranges for the window's genome/chromosome/strand.
    
    Args:
        donor_positions: Array of donor site positions in the sequence window
        acceptor_positions: Array of acceptor site positions in the sequence window
        window_start: Start genomic coordinate of the window
        window_strand: Strand of the window ('+' or '-')
        genomic_coords_dict: Dictionary mapping (genome_id, chrom, strand) to coordinate ranges
        genome_id: Genome ID of the window
        chromosome: Chromosome of the window
        
    Returns:
        Tuple of (filtered_donor_positions, filtered_acceptor_positions) as numpy arrays
        
    Note:
        For '+' strand: genomic_pos = window_start + sequence_pos
        For '-' strand: genomic_pos = window_start - sequence_pos
        Donor sites are at GT (pos, pos+1) on '+' strand
        Acceptor sites are at AG (pos, pos+1) on '+' strand
    """
    coord_key = (str(genome_id), str(chromosome), str(window_strand))
    
    if coord_key not in genomic_coords_dict:
        # No matching coordinates for this window, exclude all sites
        return np.array([], dtype=int), np.array([], dtype=int)
    
    coord_ranges = genomic_coords_dict[coord_key]
    filtered_donors = []
    filtered_acceptors = []
    
    # Filter donor sites
    for pos in donor_positions:
        if window_strand == '+':
            genomic_pos = window_start + pos - 1
        else:
            genomic_pos = window_start - pos + 1
        
        # Check if this position overlaps with any provided coordinate range
        for coord_start, coord_end in coord_ranges:
            if coord_start <= genomic_pos <= coord_end:
                filtered_donors.append(pos)
                break
    
    # Filter acceptor sites
    for pos in acceptor_positions:
        if window_strand == '+':
            genomic_pos = window_start + pos + 1
        else:
            genomic_pos = window_start - pos - 1
        
        # Check if this position overlaps with any provided coordinate range
        for coord_start, coord_end in coord_ranges:
            if coord_start <= genomic_pos <= coord_end:
                filtered_acceptors.append(pos)
                break
    
    return np.array(filtered_donors, dtype=int), np.array(filtered_acceptors, dtype=int)
