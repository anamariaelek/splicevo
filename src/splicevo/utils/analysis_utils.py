"""Analysis utilities for splice site inspection and visualization."""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, List, Dict, Union
from .sequence_utils import one_hot_decode, complement_sequence
from .window_utils import (
    resolve_window_indices,
    build_genomic_coords_dict,
    filter_splice_sites_by_genomic_coords
)


def analyze_splice_sites(
    test_seq: np.ndarray,
    test_labels: np.ndarray,
    test_meta_df: pd.DataFrame,
    test_alpha: Optional[np.ndarray] = None,
    test_beta: Optional[np.ndarray] = None,
    test_sse: Optional[np.ndarray] = None,
    genome=None,
    usage_conditions: Optional[List[Dict[str, str]]] = None,
    window_indices: Optional[np.ndarray] = None,
    genomic_coords: Optional[List[Tuple[str, int, int, str]]] = None,
    context_size: int = 4500,
    show_usage: bool = True,
    return_data: bool = False,
    verbose: bool = True
) -> Union[None, List[Dict]]:
    """
    Analyze splice sites in selected windows, either by index or genomic coordinates.
    
    Args:
        test_seq: One-hot encoded sequences array [n_windows, seq_length, 4]
        test_labels: Splice site labels array [n_windows, window_size]
        test_meta_df: DataFrame with window metadata
        test_alpha: Optional usage alpha values [n_windows, window_size, n_conditions]
        test_beta: Optional usage beta values [n_windows, window_size, n_conditions]
        test_sse: Optional splice site efficiency values [n_windows, window_size, n_conditions]
        genome: Optional GenomeData object for retrieving FASTA sequences
        usage_conditions: List of condition dicts with 'condition_key', 'tissue', 'timepoint', 'display_name'
        window_indices: Array of row indices to analyze from test_meta_df
        genomic_coords: List of (genome_id, chromosome, start, end, strand) tuples to find overlapping windows
        context_size: Size of context used in the data
        show_usage: Whether to display usage statistics
        return_data: If True, return analysis data instead of printing
        verbose: If True, print analysis details
        
    Returns:
        List of analysis data dicts if return_data=True, else None
    """
    
    # Resolve window indices from either explicit indices or genomic coordinates
    try:
        window_indices = resolve_window_indices(
            test_meta_df,
            window_indices=window_indices,
            genomic_coords=genomic_coords,
            default_n=5
        )
    except ValueError as e:
        print(f"Error: {e}")
        return [] if return_data else None
    
    if verbose and genomic_coords is not None:
        print(f"Found {len(window_indices)} windows overlapping specified coordinates:")
        for idx in window_indices:
            row = test_meta_df.iloc[idx]
            print(f"  Window {idx}: {row['genome_id']}:{row['chromosome']}:{row['window_start']}-{row['window_end']} ({row['strand']})")
        print()
    
    # Build genomic coordinate dictionary for filtering splice sites if provided
    genomic_coords_dict = None
    if genomic_coords is not None:
        genomic_coords_dict = build_genomic_coords_dict(genomic_coords)
    
    if verbose:
        print(f"{'='*80}")
        print("SPLICE SITE ANALYSIS")
        print(f"{'='*80}\n")
    
    analysis_results = []
    
    for idx, row_idx in enumerate(window_indices):
        if row_idx >= len(test_meta_df):
            continue
            
        row = test_meta_df.iloc[row_idx]
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"Window {idx + 1} (index {row_idx}): {row.get('gene_id', 'unknown')} on chromosome {row['chromosome']} ({row['strand']} strand)")
            print(f"{'='*80}")
            print(f"Genome ID: {row.get('genome_id', 'unknown')}")
            print(f"Genomic location: {row['chromosome']}:{row['window_start']}-{row['window_end']}")
            print(f"Donor sites: {row.get('n_donor_sites', '?')}, Acceptor sites: {row.get('n_acceptor_sites', '?')}")
        
        chrom = str(row['chromosome'])
        start = int(row['window_start'])
        end = int(row['window_end'])
        strand = str(row['strand'])
        
        # Get sequence from FASTA if genome is available
        fasta_seq = None
        if genome is not None:
            try:
                context_start = max(1, start - context_size)
                context_end = end + context_size
                fasta_seq = genome.get_seq(chrom, context_start+1, context_end, rc=False)
                if not isinstance(fasta_seq, str):
                    fasta_seq = str(fasta_seq)
                if strand == "-":
                    fasta_seq = complement_sequence(fasta_seq)
                if verbose:
                    print(f"\nSequence from FASTA (length={len(fasta_seq)}):")
                    print(f"{fasta_seq[:100]}..." if len(fasta_seq) > 100 else fasta_seq)
            except Exception as e:
                if verbose:
                    print(f"\nWarning: Could not retrieve sequence from FASTA: {e}")
        
        # Get sequence from processed data
        one_hot_seq = test_seq[row_idx]
        seq_string = one_hot_decode(one_hot_seq)
        
        if verbose:
            print(f"\nSequence from processed data (length={len(seq_string)}):")
            print(seq_string[:100] + "..." if len(seq_string) > 100 else seq_string)
        
        # Get labels and find splice sites - ensure labels is 1D
        labels_row = np.atleast_1d(test_labels[row_idx])
        donor_positions = np.where(labels_row == 1)[0]
        acceptor_positions = np.where(labels_row == 2)[0]
        
        # Filter splice sites by genomic coordinates if provided
        if genomic_coords_dict is not None:
            donor_pos_filtered, acceptor_pos_filtered = filter_splice_sites_by_genomic_coords(
                donor_positions, acceptor_positions,
                window_start=start,
                window_strand=strand,
                genomic_coords_dict=genomic_coords_dict,
                genome_id=row.get('genome_id', ''),
                chromosome=chrom
            )
            
            if verbose:
                print(f"\nFiltering splice sites using genomic coordinates...")
                print(f"Filtered donor sites: {donor_pos_filtered}")
                print(f"Filtered acceptor sites: {acceptor_pos_filtered}")
            
            donor_positions = donor_pos_filtered
            acceptor_positions = acceptor_pos_filtered
        
        if verbose:
            print(f"\nDonor site positions in sequence: {donor_positions}")
            if len(donor_positions) > 0:
                print("Sequences around donor sites (±5bp):")
                for pos in donor_positions[:3]:
                    pos = pos + context_size  # Adjust for context
                    start_seq = max(0, pos - 5)
                    end_seq = min(len(seq_string), pos + 7)
                    context_seq = seq_string[start_seq:end_seq]
                    marker = " "*(pos-start_seq) + "^^"
                    print(f"  {context_seq}")
                    print(f"  {marker}")
            
            print(f"\nAcceptor site positions in sequence: {acceptor_positions}")
            if len(acceptor_positions) > 0:
                print("Sequences around acceptor sites (±5bp):")
                for pos in acceptor_positions[:3]:
                    pos = pos + context_size  # Adjust for context
                    start_seq = max(0, pos - 5)
                    end_seq = min(len(seq_string), pos + 7)
                    context_seq = seq_string[start_seq:end_seq]
                    marker = " "*(pos-start_seq) + "^^"
                    print(f"  {context_seq}")
                    print(f"  {marker}")
        
        # Collect usage data for plotting
        usage_data = None
        if show_usage and test_sse is not None:
            usage_data = _extract_usage_data(
                row_idx, test_sse, test_alpha, test_beta, labels_row, 
                usage_conditions, donor_positions, acceptor_positions
            )
            
            if verbose:
                _print_usage_stats(usage_data)
        
        # Collect all analysis data
        window_analysis = {
            'window_index': row_idx,
            'genome_id': row.get('genome_id', 'unknown'),
            'chromosome': chrom,
            'start': start,
            'end': end,
            'strand': strand,
            'gene_id': row.get('gene_id', 'unknown'),
            'sequence': seq_string,
            'fasta_sequence': fasta_seq,
            'labels': labels_row,
            'donor_positions': donor_positions,
            'acceptor_positions': acceptor_positions,
            'usage_data': usage_data
        }
        
        analysis_results.append(window_analysis)
    
    if return_data:
        return analysis_results
    
    return None


def _extract_usage_data(
    row_idx: int,
    test_sse: np.ndarray,
    test_alpha: Optional[np.ndarray],
    test_beta: Optional[np.ndarray],
    labels: np.ndarray,
    usage_conditions: Optional[List],
    donor_positions: np.ndarray,
    acceptor_positions: np.ndarray
) -> Dict[str, List[Dict]]:
    """
    Extract usage data for splice sites in a window.
    
    Returns:
        Dictionary with 'donors' and 'acceptors' keys, each containing list of position data dicts
    """
    sse_usage = test_sse[row_idx]
    
    usage_data = {'donors': [], 'acceptors': []}
    
    # Process donor sites
    for pos in donor_positions:
        if pos < len(sse_usage):
            site_data = _get_site_usage_data(pos, sse_usage, test_alpha, test_beta, row_idx, usage_conditions)
            usage_data['donors'].append(site_data)
    
    # Process acceptor sites
    for pos in acceptor_positions:
        if pos < len(sse_usage):
            site_data = _get_site_usage_data(pos, sse_usage, test_alpha, test_beta, row_idx, usage_conditions)
            usage_data['acceptors'].append(site_data)
    
    return usage_data


def _get_site_usage_data(
    pos: int,
    test_sse: np.ndarray,
    test_alpha: Optional[np.ndarray],
    test_beta: Optional[np.ndarray],
    row_idx: int,
    usage_conditions: Optional[List]
) -> Dict:
    """
    Extract usage data for a single splice site across conditions.
    
    Returns:
        Dictionary with position and usage values organized by tissue and timepoint
    """
    sse_vals = test_sse[pos]
    alpha_vals = test_alpha[row_idx, pos] if test_alpha is not None else None
    beta_vals = test_beta[row_idx, pos] if test_beta is not None else None
    
    site_data = {
        'position': pos,
        'conditions': {}
    }
    
    if usage_conditions is not None and len(usage_conditions) > 0:
        for cond_idx, cond in enumerate(usage_conditions):
            if cond_idx < len(sse_vals) and not np.isnan(sse_vals[cond_idx]):
                # Handle both dict and string condition formats
                if isinstance(cond, dict):
                    condition_key = cond.get('condition_key', f'cond_{cond_idx}')
                    tissue = cond.get('tissue', 'unknown')
                    timepoint = cond.get('timepoint', 'NA')
                    display_name = cond.get('display_name', condition_key)
                else:
                    # cond is a string
                    condition_key = str(cond)
                    tissue = str(condition_key.split('_')[0]) if '_' in condition_key else cond.get('tissue', 'unknown')
                    timepoint = int(condition_key.split('_')[1]) if '_' in condition_key else cond.get('timepoint', 'NA')
                    display_name = condition_key
                
                site_data['conditions'][condition_key] = {
                    'tissue': tissue,
                    'timepoint': timepoint,
                    'display_name': display_name,
                    'sse': float(sse_vals[cond_idx]),
                    'alpha': float(alpha_vals[cond_idx]) if alpha_vals is not None and cond_idx < len(alpha_vals) else None,
                    'beta': float(beta_vals[cond_idx]) if beta_vals is not None and cond_idx < len(beta_vals) else None
                }
    
    return site_data


def _print_usage_stats(usage_data: Dict[str, List[Dict]]) -> None:
    """
    Print usage statistics for splice sites.
    
    Args:
        usage_data: Dictionary with 'donors' and 'acceptors' keys
    """
    if not usage_data or (len(usage_data.get('donors', [])) == 0 and len(usage_data.get('acceptors', [])) == 0):
        return
    
    print(f"\n  Usage patterns at splice sites:")
    
    if len(usage_data.get('donors', [])) > 0:
        print(f"    Donor sites:")
        for site_data in usage_data['donors'][:3]:
            pos = site_data['position']
            print(f"      Position {pos}:")
            for cond_key, cond_data in site_data.get('conditions', {}).items():
                print(f"        {cond_data['display_name']}: SSE={cond_data['sse']:.4f}")
    
    if len(usage_data.get('acceptors', [])) > 0:
        print(f"    Acceptor sites:")
        for site_data in usage_data['acceptors'][:3]:
            pos = site_data['position']
            print(f"      Position {pos}:")
            for cond_key, cond_data in site_data.get('conditions', {}).items():
                print(f"        {cond_data['display_name']}: SSE={cond_data['sse']:.4f}")
