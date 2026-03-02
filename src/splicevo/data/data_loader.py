"""Data loading utilities for multi-genome splice site model training."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import json
import gc

from ..io.genome import GenomeData
from ..io.gene_annotation import GTFProcessor, Transcript
from ..io.splice_sites import SpliceSite
from ..utils.sequence_utils import complement_sequence

# Nucleotide to index mapping for one-hot encoding
nuc_to_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}


def load_sparse_usage(memmap_dir: Union[str, Path]) -> pd.DataFrame:
    """
    Load sparse usage data from a saved directory.
    
    Args:
        memmap_dir: Directory containing usage.parquet
        
    Returns:
        DataFrame with columns: sample_idx, position, condition_idx, alpha, beta, sse
    """
    memmap_dir = Path(memmap_dir)
    usage_file = memmap_dir / 'usage.parquet'
    
    if not usage_file.exists():
        raise FileNotFoundError(f"Sparse usage file not found: {usage_file}")
    
    return pd.read_parquet(usage_file)


def sparse_to_dense_batch(usage_df: pd.DataFrame,
                         sample_indices: np.ndarray,
                         window_size: int,
                         n_conditions: int) -> Dict[str, np.ndarray]:
    """
    Convert sparse usage data to dense arrays for a specific batch of samples.
    
    This is designed for efficient batch loading during training - converts only
    the requested samples to dense format on-the-fly.
    
    Args:
        usage_df: Sparse usage DataFrame (from load_sparse_usage)
        sample_indices: Array of sample indices to load
        window_size: Size of each window
        n_conditions: Number of conditions
        
    Returns:
        Dictionary with keys 'alpha', 'beta', 'sse' containing dense arrays of shape
        (batch_size, window_size, n_conditions) with NaN for missing values
    """
    batch_size = len(sample_indices)
    
    # Filter to only the requested samples
    sample_set = set(sample_indices)
    batch_data = usage_df[usage_df['sample_idx'].isin(sample_set)]
    
    # Initialize dense arrays with NaN
    usage_arrays = {
        'alpha': np.full((batch_size, window_size, n_conditions), np.nan, dtype=np.float32),
        'beta': np.full((batch_size, window_size, n_conditions), np.nan, dtype=np.float32),
        'sse': np.full((batch_size, window_size, n_conditions), np.nan, dtype=np.float32)
    }
    
    # Create mapping from original sample_idx to batch position
    sample_idx_map = {orig_idx: batch_pos for batch_pos, orig_idx in enumerate(sample_indices)}
    
    # Fill in values efficiently using vectorized operations where possible
    for row in batch_data.itertuples(index=False):
        batch_idx = sample_idx_map[row.sample_idx]
        pos = row.position
        cond = row.condition_idx
        usage_arrays['alpha'][batch_idx, pos, cond] = row.alpha
        usage_arrays['beta'][batch_idx, pos, cond] = row.beta
        usage_arrays['sse'][batch_idx, pos, cond] = row.sse
    
    return usage_arrays


def sparse_labels_to_dense_batch(labels_df: pd.DataFrame,
                                sample_indices: np.ndarray,
                                window_size: int) -> np.ndarray:
    """
    Convert sparse label data to dense arrays for a specific batch of samples.
    
    This is designed for efficient batch loading during training - converts only
    the requested samples to dense format on-the-fly.
    
    Args:
        labels_df: Sparse labels DataFrame with columns [sample_idx, position, label]
        sample_indices: Array of sample indices to load
        window_size: Size of each window
        
    Returns:
        Dense labels array of shape (batch_size, window_size) with 0 for unlabeled positions
    """
    batch_size = len(sample_indices)
    
    # Filter to only the requested samples
    sample_set = set(sample_indices)
    batch_data = labels_df[labels_df['sample_idx'].isin(sample_set)]
    
    # Initialize dense array with zeros (no splice site)
    labels = np.zeros((batch_size, window_size), dtype=np.int8)
    
    # Create mapping from original sample_idx to batch position
    sample_idx_map = {orig_idx: batch_pos for batch_pos, orig_idx in enumerate(sample_indices)}
    
    # Fill in label values
    for row in batch_data.itertuples(index=False):
        batch_idx = sample_idx_map[row.sample_idx]
        pos = row.position
        labels[batch_idx, pos] = row.label
    
    return labels


def load_memmap_data(memmap_dir: Union[str, Path],
                    load_labels: bool = False,
                    load_usage: bool = False) -> Tuple[np.ndarray, Optional[pd.DataFrame], Optional[pd.DataFrame], pd.DataFrame]:
    """
    Load data from memory-mapped files created by MultiGenomeDataLoader.to_arrays().
    
    Args:
        memmap_dir: Directory containing the memmap files
        load_labels: If True, loads sparse labels as a DataFrame. If False, returns None for labels.
        load_usage: If True, loads sparse usage as a DataFrame. If False, returns None for usage.
        
    Returns:
        Tuple of (sequences, labels_df_or_none, usage_df_or_none, metadata)
        - sequences: memmap array of shape (n_samples, seq_length, 4)
        - labels: DataFrame with sparse labels data if load_labels=True, else None
        - usage: DataFrame with sparse usage data if load_usage=True, else None
        - metadata: DataFrame with sample metadata
    """
    memmap_dir = Path(memmap_dir)
    
    # Load metadata
    with open(memmap_dir / 'metadata.json', 'r') as f:
        meta = json.load(f)
    
    # Load sequences as memmap
    sequences = np.memmap(
        memmap_dir / 'sequences.mmap',
        dtype=meta['sequences_dtype'],
        mode='r',
        shape=tuple(meta['sequences_shape'])
    )
    
    # Load sparse labels if requested
    labels = None
    if load_labels:
        labels_file = memmap_dir / 'labels.parquet'
        if labels_file.exists():
            labels = pd.read_parquet(labels_file)
    
    # Load sparse usage if requested
    usage = None
    if load_usage:
        usage_file = memmap_dir / 'usage.parquet'
        if usage_file.exists():
            usage = pd.read_parquet(usage_file)
    
    # Load metadata
    metadata = pd.DataFrame()
    metadata_file = memmap_dir / 'metadata.csv'
    if metadata_file.exists():
        metadata = pd.read_csv(metadata_file)
    
    return sequences, labels, usage, metadata


class MultiGenomeDataLoader:
    """
    Data loader for multi-genome splice site prediction.
    
    This class handles loading and processing splice site data from multiple genomes,
    keeping track of genome, chromosome, and transcript origin for each example.
    Supports loading splice site usage statistics from tissue/cell-type specific files.
    """
    
    def __init__(self):
        """
        Initialize the data loader.
        """
        self.genomes: Dict[str, GenomeData] = {}
        self.loaded_data: List[SpliceSite] = []
        self.usage_data: Dict[str, Dict[str, pd.DataFrame]] = {}  # {genome_id: {condition_key: usage_df}}
        self.usage_conditions: List[Dict[str, str]] = []  # List of condition metadata
        self.condition_to_key: Dict[str, str] = {}  # Maps (tissue, timepoint) to condition_key
    
    def add_genome(self, 
                   genome_id: str, 
                   genome_path: Union[str, Path], 
                   gtf_path: Union[str, Path],
                   chromosomes: Optional[List[str]] = None,
                   metadata: Optional[Dict] = None) -> None:
        """
        Add a genome to the data loader.
        
        Args:
            genome_id: Unique identifier for the genome
            genome_path: Path to the genome FASTA file
            gtf_path: Path to the GTF annotation file
            chromosomes: List of chromosomes to include (None for all)
            metadata: Additional metadata for the genome
        """
        if metadata is None:
            metadata = {}
            
        self.genomes[genome_id] = GenomeData(
            genome_id=genome_id,
            genome_path=Path(genome_path),
            gtf_path=Path(gtf_path),
            chromosomes=chromosomes,
            metadata=metadata
        )

    def _encode_ohe_window(self, seq: str, start: int, length: int) -> np.ndarray:
        """
        Encode a window of a DNA sequence into one-hot representation.
        
        This is more memory efficient than encoding the entire sequence and then slicing,
        as it only allocates memory for the requested window.
        
        Args:
            seq: DNA sequence string
            start: Start position in the sequence
            length: Length of window to encode
            
        Returns:
            One-hot encoded array of shape (length, 4)
        """
        ohe = np.zeros((length, 4), dtype=np.float32)
        for i in range(length):
            nuc = seq[start + i]
            idx = nuc_to_idx.get(nuc, 4)
            if idx < 4:
                ohe[i, idx] = 1.0
        return ohe
        
    def _collect_all_positions(self, 
                             transcripts: List[Transcript],
                             genome_id: str) -> Tuple[List[Dict], int]:
        """
        Collect all splice site positions for batch processing using optimized bulk operations.
        
        Args:
            transcripts: List of Transcript objects
            genome_id: Genome identifier
            
        Returns:
            Tuple of (positions_data, positive_count)
        """
        print(f"Collecting positions from {len(transcripts)} transcripts...")
        
        # Pre-allocate lists for bulk operations
        all_positions = []
        positive_count = 0
        
        # Process transcripts in batches for better memory usage
        batch_size = 1000
        
        # Use tqdm for progress bar
        for batch_start in tqdm(range(0, len(transcripts), batch_size), 
                               desc="Processing transcripts", 
                               unit="batch"):
            batch_end = min(batch_start + batch_size, len(transcripts))
            batch_transcripts = transcripts[batch_start:batch_end]
            
            # Collect all positive positions for this batch
            batch_positives = []
            
            for transcript in batch_transcripts:
                # Skip transcripts without splice sites
                if len(transcript.splice_donor_sites) == 0 and len(transcript.splice_acceptor_sites) == 0:
                    continue
                    
                chrom = transcript.exons.iloc[0]['chrom']
                
                # Collect donor sites in bulk
                for donor_pos in transcript.splice_donor_sites:
                    # Don't populate usage_stats here to save memory (99 conditions × 500k sites)
                    # Usage will be looked up on-demand during array extraction
                    batch_positives.append({
                        'genome_id': genome_id,
                        'chromosome': chrom,
                        'transcript_id': transcript.transcript_id,
                        'gene_id': transcript.gene_id,
                        'position': donor_pos,
                        'site_type': 1,  # donor
                        'strand': transcript.strand,
                        'site_usage': {}  # Empty dict - will look up during array extraction
                    })
                    
                # Collect acceptor sites in bulk
                for acceptor_pos in transcript.splice_acceptor_sites:
                    # Don't populate usage_stats here to save memory (99 conditions × 500k sites)
                    # Usage will be looked up on-demand during array extraction
                    batch_positives.append({
                        'genome_id': genome_id,
                        'chromosome': chrom,
                        'transcript_id': transcript.transcript_id,
                        'gene_id': transcript.gene_id,
                        'position': acceptor_pos,
                        'site_type': 2,  # acceptor
                        'strand': transcript.strand,
                        'site_usage': {}  # Empty dict - will look up during array extraction
                    })
            
            # Bulk extend instead of individual appends
            all_positions.extend(batch_positives)
            positive_count += len(batch_positives)
        
        print(f"Collected {len(all_positions)} positions ({positive_count} positive)")
        return all_positions
    
    def load_genome_data(self, 
                        genome_id: str, 
                        max_transcripts: Optional[int] = None) -> List[SpliceSite]:
        """
        Load splice site data from a specific genome using optimized batch processing.
        
        Args:
            genome_id: ID of the genome to load
            max_transcripts: Maximum number of transcripts to process (None for all)
            
        Returns:
            List of splice sites
        """
        if genome_id not in self.genomes:
            raise ValueError(f"Genome {genome_id} not found. Add it first with add_genome().")
            
        genome_data = self.genomes[genome_id]
        
        # Load genome and annotations
        #print(f"Loading genome {genome_id}...")
        #genome = genome_data.load_genome()
        
        print(f"Processing GTF annotations for {genome_id}...")
        gtf_processor = GTFProcessor(str(genome_data.gtf_path))
        transcripts = gtf_processor.process_gtf(chromosomes=genome_data.chromosomes)
        
        if max_transcripts is not None:
            transcripts = transcripts[:max_transcripts]
        
        # Collect all positions first (fast)
        positions_data = self._collect_all_positions(transcripts, genome_id)
        
        if not positions_data:
            print("No splice sites found")
            return []
        
        # Process all splice sites
        print(f"Processing {len(positions_data)} sequences...")
        examples = SpliceSite.from_positions_batch(positions_data)
        print(f"Loaded {len(examples)} examples")
        return examples

    def load_all_genomes_data(self,
                              max_transcripts_per_genome: Optional[int] = None) -> None:
        """
        Load data from all added genomes using optimized parallel batch processing.
        
        Args:
            batch_size: Number of sequences to process in each batch
            max_transcripts_per_genome: Maximum transcripts per genome
            n_jobs: Number of parallel jobs (None for auto-detection)
        """
        self.loaded_data = []
        
        for genome_id in self.genomes:
            # Load all splice sites
            genome_examples = self.load_genome_data(
                genome_id, 
                max_transcripts=max_transcripts_per_genome
            )
            self.loaded_data.extend(genome_examples)
            
        print(f"Total loaded examples: {len(self.loaded_data)}")
        
    def get_dataframe(self) -> pd.DataFrame:
        """Get loaded data as DataFrame"""
        if not self.loaded_data:
            return pd.DataFrame()
        
        loaded_data = []

        for example in self.loaded_data:
            loaded_data.append({
                'position': example.position,
                'genome_id': example.genome_id,
                'chromosome': example.chromosome,
                'gene_id': example.gene_id,
                'transcript_id': example.transcript_id,
                'site_type': example.site_type,
                'strand': example.strand
            })
        df = pd.DataFrame(loaded_data)
        return df
    
    def get_summary(self) -> pd.DataFrame:
        """Get summary statistics of loaded data."""
        df = self.get_dataframe()
        summary = df[['genome_id', 'chromosome', 'site_type', 'position']].drop_duplicates().groupby(
            ['genome_id', 'chromosome', 'site_type']
        ).agg({'position': 'count'}).reset_index().rename(columns={'position': 'n_sites'})
        return summary

    def add_usage_file(self, 
                      genome_id: str,
                      usage_file: Union[str, Path],
                      tissue: str,
                      timepoint: Optional[str] = None) -> None:
        """
        Add a usage statistics file for a specific genome, tissue, and optionally timepoint.
        
        Args:
            genome_id: Genome identifier
            usage_file: Path to usage statistics file
            tissue: Tissue or cell type name
            timepoint: Optional timepoint identifier
            
        Expected file format: TSV with columns: Sample, Region, Site, Strand, Gene, SSE, alpha_count, beta1_count, beta2_count
        """
        usage_path = Path(usage_file)
        
        if not usage_path.exists():
            raise FileNotFoundError(f"Usage file not found: {usage_path}")
            
        # Create condition key
        if timepoint is not None:
            condition_key = f"{tissue}_{timepoint}"
            condition_display = f"{tissue} {timepoint}"
        else:
            condition_key = tissue
            condition_display = tissue
            
        print(f"Loading usage file for {genome_id} - {condition_display}: {usage_path}")
        
        try:
            # Load usage data
            df = pd.read_csv(usage_path, sep='\t', header=0, dtype={'Region': str})
            
            # Check required columns
            required_columns = {'Sample', 'Region', 'Site', 'Strand', 'Gene', 'SSE', 'alpha_count', 'beta1_count', 'beta2_count'}
            if not required_columns.issubset(set(df.columns)):
                missing = required_columns - set(df.columns)
                raise ValueError(f"Usage file missing required columns: {missing}")
            
            # Clean and validate data
            df = df.dropna(subset=['Region', 'Site', 'Strand', 'SSE', 'alpha_count', 'beta1_count'])
            df['Site'] = df['Site'].astype(int)
            df['SSE'] = df['SSE'].astype(float)
            df['alpha_count'] = df['alpha_count'].astype(float)
            df['beta1_count'] = df['beta1_count'].astype(float)
            df['beta2_count'] = df['beta2_count'].astype(float)
            df['Strand'] = df['Strand'].astype(str)
            df['Region'] = df['Region'].astype(str)
            
            # Create standardized columns for compatibility with existing code
            df['chromosome'] = df['Region']
            df['position'] = df['Site']
            df['strand'] = df['Strand']
            df['sse'] = df['SSE']
            df['alpha'] = df['alpha_count']
            df['beta'] = df['beta1_count'] + df['beta2_count']  # Combine beta counts
            
            # Create lookup key for fast access
            df['lookup_key'] = df['chromosome'] + ':' + df['position'].astype(str) + ':' + df['strand']
            
            # Handle duplicates by taking the first occurrence
            n_before = len(df)
            df = df.drop_duplicates(subset=['lookup_key'], keep='first')
            n_after = len(df)
            if n_before != n_after:
                print(f"Removed {n_before - n_after} duplicate entries")
            
            df = df.set_index('lookup_key')
            
            # Store in nested dictionary
            if genome_id not in self.usage_data:
                self.usage_data[genome_id] = {}
                
            self.usage_data[genome_id][condition_key] = df
            
            # Store condition metadata if not already present
            condition_metadata = {
                'condition_key': condition_key,
                'tissue': tissue,
                'timepoint': timepoint if timepoint is not None else 'NA',
                'display_name': condition_display
            }
            
            # Check if this condition already exists
            existing_condition = None
            for cond in self.usage_conditions:
                if cond['condition_key'] == condition_key:
                    existing_condition = cond
                    break
                    
            if existing_condition is None:
                self.usage_conditions.append(condition_metadata)
                self.condition_to_key[f"{tissue}_{timepoint if timepoint else 'NA'}"] = condition_key
            
            print(f"Loaded {len(df)} usage entries for {condition_display}")
            
        except Exception as e:
            raise ValueError(f"Failed to load usage file {usage_path}: {e}")
    
    def get_available_conditions(self) -> pd.DataFrame:
        """
        Get a DataFrame of all available usage conditions.
        
        Returns:
            DataFrame with columns: condition_key, tissue, timepoint, display_name
        """
        return pd.DataFrame(self.usage_conditions)
    
    def get_usage_stats(self, 
                       genome_id: str, 
                       chromosome: str, 
                       position: int, 
                       strand: str) -> Dict[str, Dict[str, float]]:
        """
        Get usage statistics for a splice site across all conditions.
        
        Args:
            genome_id: Genome identifier
            chromosome: Chromosome name
            position: Genomic position
            strand: Strand (+ or -)
            
        Returns:
            Dictionary mapping condition_key -> {alpha, beta, sse}
        """
        usage_stats = {}
        
        if genome_id not in self.usage_data:
            return usage_stats
            
        lookup_key = f"{chromosome}:{position}:{strand}"
        
        for condition_key, usage_df in self.usage_data[genome_id].items():
            if lookup_key in usage_df.index:
                row = usage_df.loc[lookup_key]
                # Handle case where row might be a Series (single match) or DataFrame (multiple matches)
                if isinstance(row, pd.DataFrame):
                    # Take the first row if there are duplicates (shouldn't happen after deduplication)
                    row = row.iloc[0]
                
                usage_stats[condition_key] = {
                    'alpha': float(row['alpha']),
                    'beta': float(row['beta']),
                    'sse': float(row['sse'])
                }
                
        return usage_stats

    def get_usage_array_info(self,
                            usage_arrays: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, any]:
        """
        Get information about usage arrays structure.
        
        Returns:
            Dictionary with usage array metadata including condition details
        """
        if not self.loaded_data:
            return {}

        if usage_arrays is None:
            # This method needs usage arrays, which are no longer returned by to_arrays()
            # Users should load sparse usage and reconstruct as needed
            raise ValueError("usage_arrays must be provided. Use the module-level functions load_sparse_usage() and sparse_to_dense_batch() to reconstruct usage arrays.")

        info = {
            'n_samples': usage_arrays['alpha'].shape[0],
            'n_conditions': usage_arrays['alpha'].shape[2],
            'conditions': self.usage_conditions,
            'condition_coverage': {}
        }
        
        # Calculate coverage per condition
        for i, cond in enumerate(self.usage_conditions):
            condition_key = cond['condition_key']
            n_with_data = np.sum(~np.isnan(usage_arrays['sse'][:, i]))
            info['condition_coverage'][condition_key] = {
                'tissue': cond['tissue'],
                'timepoint': cond['timepoint'],
                'display_name': cond['display_name'],
                'n_samples_with_data': int(n_with_data),
                'coverage_fraction': float(n_with_data / info['n_samples']),
                'mean_sse': float(np.nanmean(usage_arrays['sse'][:, i])),
                'mean_alpha': float(np.nanmean(usage_arrays['alpha'][:, i])),
                'mean_beta': float(np.nanmean(usage_arrays['beta'][:, i]))
            }
                
        return info

    def get_usage_summary(self) -> pd.DataFrame:
        """
        Get summary of usage data coverage across genomes and conditions.
        
        Returns:
            DataFrame with usage statistics summary
        """
        summary_rows = []
        
        for genome_id, conditions_data in self.usage_data.items():
            for condition_key, usage_df in conditions_data.items():
                # Find condition metadata
                condition_info = next((c for c in self.usage_conditions if c['condition_key'] == condition_key), {})
                # Subset for chromosomes
                usage_df = usage_df[usage_df['chromosome'].isin(self.genomes[genome_id].chromosomes)]
                summary_rows.append({
                    'genome_id': genome_id,
                    'condition_key': condition_key,
                    'tissue': condition_info.get('tissue', 'unknown'),
                    'timepoint': condition_info.get('timepoint', 'NA'),
                    'display_name': condition_info.get('display_name', condition_key),
                    'n_sites': len(usage_df),
                    'mean_sse': usage_df['sse'].mean(),
                    'std_sse': usage_df['sse'].std(),
                    'mean_alpha': usage_df['alpha'].mean(),
                    'mean_beta': usage_df['beta'].mean()
                })
        
        return pd.DataFrame(summary_rows)

    def _encode_ohe_window(self, seq: str, window_start: int, window_length: int) -> np.ndarray:
        """
        Encode a portion of a sequence to one-hot encoding on-demand to save memory.
        
        Args:
            seq: The full sequence string
            window_start: Start position in the sequence
            window_length: Length of the window
            
        Returns:
            One-hot encoded array of shape (window_length, 4)
        """
        nuc_to_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
        ohe = np.zeros((window_length, 4), dtype=np.float32)
        
        window_end = min(window_start + window_length, len(seq))
        for i in range(window_start, window_end):
            nuc = seq[i]
            idx = nuc_to_idx.get(nuc, 4)
            if idx < 4:
                ohe[i - window_start, idx] = 1.0
        
        return ohe

    def _process_gene_windows(self,
                             genome_id: str,
                             gene_id: str,
                             df_gene: pd.DataFrame,
                             splice_site_index: Dict,
                             condition_to_idx: Dict[str, int],
                             n_conditions: int,
                             window_size: int,
                             context_size: int,
                             total_window: int,
                             genome_cache: Optional[Dict] = None) -> Tuple[List[np.ndarray], List[np.ndarray], List[Dict], List[Dict]]:
        """
        Process windows for a single gene (parallelizable).
        
        Memory-optimized version:
        - Encodes sequences on-demand per window instead of pre-computing full gene
        - Returns sparse usage data (coordinate list) instead of dense NaN-filled arrays
        
        Args:
            genome_cache: Optional pre-loaded genome to avoid repeated loading
        
        Returns:
            Tuple of (sequences, labels, usage_sparse_list, metadata_rows)
            usage_sparse_list: List of dicts with keys: sample_idx, position, condition_idx, alpha, beta, sse
        """
        sequences = []
        labels_list = []
        usage_sparse_list = []  # List of sparse coordinate dicts
        metadata_rows = []
        
        # Use cached genome if available, otherwise load it
        if genome_cache is not None and genome_id in genome_cache:
            genome = genome_cache[genome_id]
        else:
            genome = self.genomes[genome_id].load_genome()
        
        chrom = df_gene['chromosome'].iloc[0]
        strand = df_gene['strand'].iloc[0]
        
        # Get 5'-most and 3'-most positions
        orig_min_pos = df_gene['position'].min()
        orig_max_pos = df_gene['position'].max()
        min_pos = orig_min_pos
        max_pos = orig_max_pos

        # Always expand region so that the last window is padded with genomic sequence up to the end of the window
        gene_length = max_pos - min_pos + 1
        n_windows = ((gene_length - 1) // window_size) + 1
        expanded_length = n_windows * window_size
        pad_needed = expanded_length - gene_length
        pad_left = 0
        pad_right = pad_needed
        # For short genes, pad symmetrically
        if gene_length < window_size:
            pad_left = pad_needed // 2
            pad_right = pad_needed - pad_left
        min_pos = min_pos - pad_left
        max_pos = max_pos + pad_right

        # Extend the gene range for specified context
        requested_start = min_pos - context_size
        requested_end = max_pos + context_size
        
        # Adjust boundaries to valid range
        actual_start = max(0, requested_start)
        
        # Get chromosome length if available
        try:
            chrom_length = len(genome._genome[chrom])
            actual_end = min(requested_end, chrom_length)
        except (KeyError, AttributeError):
            actual_end = requested_end
        
        # Calculate additional padding needed if out of chr range
        left_pad = actual_start - requested_start
        right_pad = requested_end - actual_end
        
        # Get sequence with valid coordinates
        # Always get forward strand sequence - the windowing and labeling logic
        # works with genomic coordinates regardless of actual strand orientation
        try:
            seq = genome.get_seq(chrom, actual_start + 1, actual_end, rc=False)
        except Exception:
            return sequences, labels_list, usage_sparse_list, metadata_rows
        
        # Ensure seq is a string
        if not isinstance(seq, str):
            seq = str(seq).upper()
        else:
            seq = seq.upper()
        
        # For negative strand genes, apply complement
        # This preserves the biological meaning while keeping genomic coordinates
        if strand == '-':
            seq = complement_sequence(seq)  # Apply complement (not reverse complement)
        
        # Add padding with 'N' if out of chr range
        if left_pad > 0:
            seq = 'N' * left_pad + seq
        
        if right_pad > 0:
            seq = seq + 'N' * right_pad
        

        # Split gene into windows shifted by window_size, always pad last window with genomic sequence if needed
        n_windows = ((max_pos - min_pos + 1) - 1) // window_size + 1
        for window_idx in range(n_windows):
            window_start = window_idx * window_size
            seq_window = seq[window_start:window_start + total_window]

            # If at the end, pad with N if needed (this should not happen due to the way we calculated padding, but just in case)
            if len(seq_window) < total_window:
                n_n = total_window - len(seq_window)
                Warning(f"Window {window_idx} for gene {gene_id} is shorter than total_window. Padding with {n_n} Ns.")
                seq_window = seq_window + 'N' * (n_n)

            # Calculate the genomic position of this window
            window_genomic_start = requested_start + window_start
            window_genomic_end = window_genomic_start + total_window
            
            # The central window spans from context_size to context_size+window_size
            window_center_genomic_start = window_genomic_start + context_size
            window_center_genomic_end = window_center_genomic_start + window_size

            # Find splice sites in this window's central region
            sites_in_window = df_gene[
                (df_gene['position'] >= window_center_genomic_start) &
                (df_gene['position'] < window_center_genomic_end)
            ]

            # Skip windows with no splice sites
            if len(sites_in_window) == 0:
                continue

            # Encode sequence on-demand for this window only (memory optimization)
            ohe_seq = self._encode_ohe_window(seq_window, 0, total_window)

            # Initialize label array for this window
            labels = np.zeros(window_size, dtype=np.int8)

            # Use sparse dictionaries for usage arrays instead of full NaN arrays
            usage_alpha_dict = {}
            usage_beta_dict = {}
            usage_sse_dict = {}

            donor_sites = []
            acceptor_sites = []

            # Mark positions and add usage stats for each site
            for _, site_row in sites_in_window.iterrows():
                site_pos = site_row['position']
                site_type = site_row['site_type']

                # Calculate position within the window
                window_pos = site_pos - window_center_genomic_start

                if 0 <= window_pos < window_size:
                    lookup_key = (genome_id, chrom, site_pos, strand)
                    splice_site = splice_site_index.get(lookup_key)

                    if splice_site is not None:
                        labels[window_pos] = site_type
                        if site_type == 1 and window_pos not in donor_sites:
                            donor_sites.append(window_pos)
                        elif site_type == 2 and window_pos not in acceptor_sites:
                            acceptor_sites.append(window_pos)

                        # Look up usage stats on-demand instead of using pre-loaded dict
                        usage_stats = self.get_usage_stats(genome_id, chrom, site_pos, strand)

                        # Store only non-NaN usage stats (sparse representation)
                        for condition_key, stats in usage_stats.items():
                            if condition_key in condition_to_idx:
                                cond_idx = condition_to_idx[condition_key]
                                if window_pos not in usage_alpha_dict:
                                    usage_alpha_dict[window_pos] = {}
                                    usage_beta_dict[window_pos] = {}
                                    usage_sse_dict[window_pos] = {}
                                usage_alpha_dict[window_pos][cond_idx] = stats['alpha']
                                usage_beta_dict[window_pos][cond_idx] = stats['beta']
                                usage_sse_dict[window_pos][cond_idx] = stats['sse']

            # Store sparse usage coordinates (no dense array conversion)
            global_window_idx = len(sequences)  # Index of this window in the output

            for window_pos in usage_alpha_dict:

                # Use coords same as in AlphaGenome
                #if label == 1 and strand == '+':
                #    pos -= 2
                #if label == 2 and strand == '-':
                #    pos -= 2

                for cond_idx in usage_alpha_dict[window_pos].keys():
                    usage_sparse_list.append({
                        'sample_idx': global_window_idx,
                        'position': window_pos,
                        'strand': strand,
                        'condition_idx': cond_idx,
                        'alpha': usage_alpha_dict[window_pos][cond_idx],
                        'beta': usage_beta_dict[window_pos][cond_idx],
                        'sse': usage_sse_dict[window_pos][cond_idx]
                    })

            # Store sparse labels (only non-zero positions)
            labels_sparse_list = []
            for pos, label_val in enumerate(labels):
                if label_val != 0:  # Only store donors (1) and acceptors (2)
                    labels_sparse_list.append({
                        'sample_idx': global_window_idx,
                        'position': pos,
                        'strand': strand,
                        'label': label_val
                    })

            # Store this window's data
            sequences.append(ohe_seq)
            labels_list.extend(labels_sparse_list)

            # Track the actual transcript region (before padding) for this window
            # The overlap of [window_center_genomic_start, window_center_genomic_end) and [orig_min_pos, orig_max_pos+1)
            central_start = max(window_center_genomic_start, orig_min_pos)
            central_end = min(window_center_genomic_end, orig_max_pos + 1)

            metadata_row = {
                'genome_id': genome_id,
                'chromosome': chrom,
                'gene_id': gene_id,
                'strand': strand,
                'window_with_context_start': window_genomic_start,
                'window_with_context_end': window_genomic_end,
                'window_start': window_center_genomic_start,
                'window_end': window_center_genomic_end,
                'central_gene_start': central_start,
                'central_gene_end': central_end,
                'n_donor_sites': len(donor_sites),
                'n_acceptor_sites': len(acceptor_sites),
            }
            metadata_rows.append(metadata_row)

        # labels_list now contains sparse label coordinates, not dense arrays
        return sequences, labels_list, usage_sparse_list, metadata_rows

    def to_arrays(self,
                  window_size: int = 1000,
                  context_size: int = 4500,
                  alpha_threshold: Optional[int] = None,
                  n_workers: Optional[int] = None,
                  use_parallel: bool = True,
                  save_memmap: Optional[Union[str, Path]] = None) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Convert loaded data to arrays for ML training.
        
        Labels and usage data are saved in sparse parquet format. Use the module-level
        functions sparse_labels_to_dense_batch() and sparse_to_dense_batch() to reconstruct
        dense arrays for specific batches during training.
        
        Args:
            window_size: Size of the window containing splice sites
            context_size: Size of context on each side of the window
            alpha_threshold: Minimum alpha value (set lower values to 0)
            n_workers: Number of parallel workers (None for CPU count)
            use_parallel: Whether to use parallel processing for genes
            save_memmap: Optional path to save arrays as memmap files
        
        Returns:
            Tuple of (sequences, metadata_df)
            - sequences: memmap array if save_memmap specified, otherwise numpy array
            - metadata_df: DataFrame with sample metadata
        """
        if not self.loaded_data:
            raise ValueError("No data loaded. Call load_all_genomes_data() first.")

        if n_workers is None:
            n_workers = min(mp.cpu_count(), 8)

        # Pre-build index for O(1) splice site lookup
        print("Building splice site index...")
        splice_site_index = {}
        for site in self.loaded_data:
            key = (site.genome_id, site.chromosome, site.position, site.strand)
            splice_site_index[key] = site
        print(f"Indexed {len(splice_site_index)} splice sites")

        # Get all splice sites info
        df = self.get_dataframe()

        n_conditions = len(self.usage_conditions)
        condition_to_idx = {cond['condition_key']: idx for idx, cond in enumerate(self.usage_conditions)}

        loaded_genomes = df['genome_id'].unique()
        total_window = context_size + window_size + context_size

        # Fast window estimation from dataframe (no genome loading needed)
        if save_memmap:
            print("Estimating window count from splice site positions...")
            total_windows = 0
            
            for genome_id in loaded_genomes:
                df_genome = df[df['genome_id'] == genome_id]
                genes = df_genome['gene_id'].unique()
                
                for gene_id in genes:
                    df_gene = df_genome[df_genome['gene_id'] == gene_id]
                    
                    # Estimate windows from position range (no genome sequence needed)
                    min_pos = df_gene['position'].min()
                    max_pos = df_gene['position'].max()
                    gene_span = max_pos - min_pos + 2 * context_size
                    
                    # Count windows that would contain splice sites
                    n_windows_in_gene = max(1, (gene_span - total_window) // window_size + 1)
                    total_windows += n_windows_in_gene
            
            # Over-allocate by 10% to be safe (will trim at end)
            total_windows = int(total_windows * 1.1)
            print(f"Estimated windows (with 10% buffer): {total_windows}")
            
            # Pre-allocate memmap arrays
            save_memmap = Path(save_memmap)
            save_memmap.mkdir(parents=True, exist_ok=True)
            
            seq_shape = (total_windows, total_window, 4)
            
            print(f"Pre-allocating memmap arrays in {save_memmap}")
            sequences = np.memmap(
                save_memmap / 'sequences.mmap',
                dtype=np.float32,
                mode='w+',
                shape=seq_shape
            )
            
            # Note: Labels and usage data will be saved in sparse format (parquet), not memmap
            # This saves massive amounts of space and time
            all_labels_sparse = []  # Collect sparse label coordinates
            all_usage_sparse = []  # Collect sparse usage coordinates
            
            # Fill memmap arrays incrementally
            print("\nProcessing genes and writing to memmap...")
            current_idx = 0
            all_metadata_rows = []
            
        else:
            # Non-memmap path: collect in memory
            print("Collecting windows from all genes...")
            all_sequences = []
            all_labels_sparse = []  # Collect sparse label coordinates
            all_usage_sparse = []  # Collect sparse coordinate list
            all_metadata_rows = []

        for genome_id in loaded_genomes:
            print(f"Processing genome {genome_id}...")
            df_genome = df[df['genome_id'] == genome_id]
            genes = df_genome['gene_id'].unique()
            
            print(f"  Processing {len(genes)} genes...")
            
            # Pre-load genome once
            print(f"  Pre-loading genome...")
            genome_cache = {genome_id: self.genomes[genome_id].load_genome()}

            # Prepare gene data for parallel processing
            gene_tasks = []
            for gene_id in genes:
                df_gene = df_genome[df_genome['gene_id'] == gene_id]
                gene_tasks.append((genome_id, gene_id, df_gene))
            
            # Process genes
            if use_parallel and len(gene_tasks) > 10:
                print(f"  Using {n_workers} parallel workers...")
                
                from concurrent.futures import ThreadPoolExecutor
                
                with ThreadPoolExecutor(max_workers=n_workers) as executor:
                    future_to_gene = {}
                    for genome_id, gene_id, df_gene in gene_tasks:
                        future = executor.submit(
                            self._process_gene_windows,
                            genome_id, gene_id, df_gene,
                            splice_site_index, condition_to_idx, n_conditions,
                            window_size, context_size, total_window,
                            genome_cache
                        )
                        future_to_gene[future] = gene_id
                    
                    for future in tqdm(as_completed(future_to_gene), 
                                     total=len(gene_tasks),
                                     desc=f"  {genome_id} genes",
                                     unit="gene",
                                     mininterval=0.5):
                        try:
                            seq_list, labels_sparse, usage_sparse, metadata_rows = future.result(timeout=120)
                            
                            if save_memmap:
                                # Write incrementally to memmap (memory efficient)
                                n_windows = len(seq_list)
                                if n_windows > 0:
                                    end_idx = current_idx + n_windows
                                    if end_idx > len(sequences):
                                        print(f"\n  Warning: Exceeded estimated size, truncating at {current_idx}")
                                        break
                                    
                                    sequences[current_idx:end_idx] = np.array(seq_list, dtype=np.float32)
                                    
                                    # Adjust sample_idx in sparse labels and usage to match global indices
                                    for item in labels_sparse:
                                        item['sample_idx'] += current_idx
                                    for item in usage_sparse:
                                        item['sample_idx'] += current_idx
                                    all_labels_sparse.extend(labels_sparse)
                                    all_usage_sparse.extend(usage_sparse)
                                    
                                    all_metadata_rows.extend(metadata_rows)
                                    current_idx = end_idx
                                    
                                    # Periodic flush to disk
                                    if len(all_metadata_rows) % 500 == 0:
                                        sequences.flush()
                            else:
                                # Collect in memory for non-memmap mode
                                n_windows = len(seq_list)
                                # Adjust sample_idx to global indices
                                current_base = len(all_sequences)
                                for item in labels_sparse:
                                    item['sample_idx'] += current_base
                                for item in usage_sparse:
                                    item['sample_idx'] += current_base
                                
                                all_sequences.extend(seq_list)
                                all_labels_sparse.extend(labels_sparse)
                                all_usage_sparse.extend(usage_sparse)
                                all_metadata_rows.extend(metadata_rows)
                                
                        except Exception as e:
                            gene_id = future_to_gene[future]
                            print(f"\n  Warning: Failed to process gene {gene_id}: {e}")
            else:
                # Sequential processing
                for genome_id, gene_id, df_gene in tqdm(gene_tasks, 
                                                       desc=f"  {genome_id} genes",
                                                       unit="gene"):
                    try:
                        seq_list, labels_sparse, usage_sparse, metadata_rows = self._process_gene_windows(
                            genome_id, gene_id, df_gene,
                            splice_site_index, condition_to_idx, n_conditions,
                            window_size, context_size, total_window,
                            genome_cache
                        )
                        
                        if save_memmap:
                            # Write incrementally to memmap (memory efficient)
                            n_windows = len(seq_list)
                            if n_windows > 0:
                                end_idx = current_idx + n_windows
                                if end_idx > len(sequences):
                                    print(f"\n  Warning: Exceeded estimated size, stopping at {current_idx}")
                                    break
                                
                                sequences[current_idx:end_idx] = np.array(seq_list, dtype=np.float32)
                                
                                # Adjust sample_idx in sparse labels and usage to match global indices
                                for item in labels_sparse:
                                    item['sample_idx'] += current_idx
                                for item in usage_sparse:
                                    item['sample_idx'] += current_idx
                                all_labels_sparse.extend(labels_sparse)
                                all_usage_sparse.extend(usage_sparse)
                                
                                all_metadata_rows.extend(metadata_rows)
                                current_idx = end_idx
                        else:
                            # Collect in memory
                            n_windows = len(seq_list)
                            # Adjust sample_idx to global indices
                            current_base = len(all_sequences)
                            for item in labels_sparse:
                                item['sample_idx'] += current_base
                            for item in usage_sparse:
                                item['sample_idx'] += current_base
                            
                            all_sequences.extend(seq_list)
                            all_labels_sparse.extend(labels_sparse)
                            all_usage_sparse.extend(usage_sparse)
                            all_metadata_rows.extend(metadata_rows)
                    
                    except Exception as e:
                        print(f"Warning: Failed to process gene {gene_id}: {e}")
        
        # Finalize arrays
        if save_memmap:
            # Save sparse labels data as parquet
            print(f"Saving sparse labels data ({len(all_labels_sparse)} entries)...")
            labels_df = pd.DataFrame(all_labels_sparse)
            # Convert to appropriate dtypes for efficiency
            labels_df['sample_idx'] = labels_df['sample_idx'].astype(np.int64)
            labels_df['position'] = labels_df['position'].astype(np.int32)  # int32 supports up to 2.1 billion, enough for any sequence
            labels_df['label'] = labels_df['label'].astype(np.int8)
            # Use pyarrow engine to preserve dtypes and prevent automatic downcasting
            labels_df.to_parquet(save_memmap / 'labels.parquet', compression='snappy', index=False, engine='pyarrow')
            print(f"Sparse labels data saved to {save_memmap / 'labels.parquet'}")
            
            # Save sparse usage data as parquet
            print(f"Saving sparse usage data ({len(all_usage_sparse)} entries)...")
            usage_df = pd.DataFrame(all_usage_sparse)
            # Convert to appropriate dtypes for efficiency
            usage_df['sample_idx'] = usage_df['sample_idx'].astype(np.int64)
            usage_df['position'] = usage_df['position'].astype(np.int32)  # int32 supports up to 2.1 billion
            usage_df['condition_idx'] = usage_df['condition_idx'].astype(np.int16)  # Conditions are small numbers
            # Use pyarrow engine to preserve dtypes
            usage_df.to_parquet(save_memmap / 'usage.parquet', compression='snappy', index=False, engine='pyarrow')
            print(f"Sparse usage data saved to {save_memmap / 'usage.parquet'}")
            
            # Trim sequences array to actual size (we over-allocated by 10%)
            if current_idx < len(sequences):
                print(f"Trimming memmap arrays from {len(sequences)} to {current_idx} windows...")
                # Create properly-sized memmap
                sequences_trimmed = np.memmap(
                    save_memmap / 'sequences.mmap',
                    dtype=np.float32,
                    mode='r+',
                    shape=(current_idx, sequences.shape[1], sequences.shape[2])
                )
                
                # Replace reference
                sequences = sequences_trimmed
            
            # Final flush
            print("Flushing sequences memmap...")
            sequences.flush()
            
            n_samples = current_idx
            
            # Save metadata
            metadata_dict = {
                'sequences_shape': list(sequences.shape),
                'sequences_dtype': 'float32',
                'labels_format': 'sparse',
                'labels_sparse_entries': len(all_labels_sparse),
                'window_size': window_size,
                'context_size': context_size,
                'n_conditions': n_conditions,
                'usage_format': 'sparse',
                'usage_sparse_entries': len(all_usage_sparse)
            }
            
            with open(save_memmap / 'metadata.json', 'w') as f:
                json.dump(metadata_dict, f, indent=2)
            
            print(f"Memory-mapped files saved to: {save_memmap}")
            print(f"Sparse labels statistics:")
            print(f"  Total entries: {len(all_labels_sparse)}")
            print(f"Sparse usage statistics:")
            print(f"  Total entries: {len(all_usage_sparse)}")
            print(f"  Sparsity: {100 * len(all_usage_sparse) / (n_samples * window_size * n_conditions):.4f}%")
            
        else:
            # Create regular numpy arrays (no memmap)
            print("Converting to numpy arrays...")
            sequences = np.array(all_sequences, dtype=np.float32)
            n_samples = len(sequences)
            
            print(f"Collected {len(all_labels_sparse)} sparse label entries")
            print(f"Collected {len(all_usage_sparse)} sparse usage entries")
            if len(all_usage_sparse) > 0:
                sparsity = 100 * len(all_usage_sparse) / (n_samples * window_size * n_conditions)
                print(f"  Sparsity: {sparsity:.4f}%")
        
        metadata = pd.DataFrame(all_metadata_rows)
        
        # Save metadata to CSV if using memmap
        if save_memmap:
            metadata.to_csv(save_memmap / 'metadata.csv', index=False)
            print(f"Saved metadata to {save_memmap / 'metadata.csv'}")
        
        # Validation
        if save_memmap:
            assert sequences.shape[0] == n_samples, "Sequences shape mismatch"
        else:
            assert len(sequences) == n_samples, "Sequences count mismatch"
        assert len(metadata) == n_samples, "Metadata mismatch"

        print(f"Created {n_samples} windowed examples")
        print(f"  Sequence shape: {sequences.shape if save_memmap else (len(sequences), sequences[0].shape)}")
        
        # Calculate and display label statistics from sparse format
        total_donors = len([e for e in all_labels_sparse if e['label'] == 1])
        total_acceptors = len([e for e in all_labels_sparse if e['label'] == 2])
        
        if save_memmap:
            print(f"  Labels stored in sparse format: labels.parquet")
            print(f"  Usage stored in sparse format: usage.parquet")
        
        print(f"  Total donor sites: {total_donors}")
        print(f"  Total acceptor sites: {total_acceptors}")
        
        return sequences, metadata
