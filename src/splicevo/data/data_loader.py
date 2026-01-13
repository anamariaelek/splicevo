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
            sequences, labels, usage_arrays, metadata = self.to_arrays()

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
                             genome_cache: Optional[Dict] = None) -> Tuple[List[np.ndarray], List[np.ndarray], Dict[str, List[np.ndarray]], List[Dict]]:
        """
        Process windows for a single gene (parallelizable).
        
        Memory-optimized version:
        - Encodes sequences on-demand per window instead of pre-computing full gene
        - Stores sparse usage data instead of full NaN-filled arrays
        
        Args:
            genome_cache: Optional pre-loaded genome to avoid repeated loading
        
        Returns:
            Tuple of (sequences, labels, usage_dict, metadata_rows)
        """
        sequences = []
        labels_list = []
        usage_dict = {'alpha': [], 'beta': [], 'sse': []}
        metadata_rows = []
        
        # Use cached genome if available, otherwise load it
        if genome_cache is not None and genome_id in genome_cache:
            genome = genome_cache[genome_id]
        else:
            genome = self.genomes[genome_id].load_genome()
        
        chrom = df_gene['chromosome'].iloc[0]
        strand = df_gene['strand'].iloc[0]
        
        # Get 5'-most and 3'-most positions
        min_pos = df_gene['position'].min() 
        max_pos = df_gene['position'].max()

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
        
        # Calculate padding needed
        left_pad = actual_start - requested_start
        right_pad = requested_end - actual_end
        
        # Get sequence with valid coordinates
        # Always get forward strand sequence - the windowing and labeling logic
        # works with genomic coordinates regardless of actual strand orientation
        try:
            seq = genome.get_seq(chrom, actual_start + 1, actual_end, rc=False)
        except Exception:
            return sequences, labels_list, usage_dict, metadata_rows
        
        # Ensure seq is a string
        if not isinstance(seq, str):
            seq = str(seq).upper()
        else:
            seq = seq.upper()
        
        # For negative strand genes, reverse the sequence and apply complement
        # This preserves the biological meaning while keeping genomic coordinates
        if strand == '-':
            seq = complement_sequence(seq)  # Apply complement (not reverse complement)
        
        # Add padding with 'N' if needed
        if left_pad > 0:
            seq = 'N' * left_pad + seq
        
        if right_pad > 0:
            seq = seq + 'N' * right_pad
        
        # Split gene into windows shifted by window_size
        # Encode sequence on-demand per window to save memory
        for window_start in range(0, len(seq) - total_window + 1, window_size):
            # Calculate the genomic position of this window
            window_genomic_start = requested_start + window_start
            
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
            ohe_seq = self._encode_ohe_window(seq, window_start, total_window)
            
            # Initialize label array for this window
            labels = np.zeros(window_size, dtype=np.int8)
            
            # Use sparse dictionaries for usage arrays instead of full NaN arrays
            # This significantly reduces memory for genomes with many usage conditions
            usage_alpha_dict = {}
            usage_beta_dict = {}
            usage_sse_dict = {}
            
            n_donor_sites = 0
            n_acceptor_sites = 0
            
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
                    
                        if site_type == 1:
                            n_donor_sites += 1
                        elif site_type == 2:
                            n_acceptor_sites += 1
                        
                        # Look up usage stats on-demand instead of using pre-loaded dict
                        # This saves memory by not storing usage for all 500k sites during step 4
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
            
            # Convert sparse dicts to dense arrays with NaN (only at append time)
            # This defers allocation until necessary and keeps memory minimal during processing
            usage_alpha = np.full((window_size, n_conditions), np.nan, dtype=np.float32)
            usage_beta = np.full((window_size, n_conditions), np.nan, dtype=np.float32)
            usage_sse = np.full((window_size, n_conditions), np.nan, dtype=np.float32)
            
            for window_pos in usage_alpha_dict:
                for cond_idx, val in usage_alpha_dict[window_pos].items():
                    usage_alpha[window_pos, cond_idx] = val
            
            for window_pos in usage_beta_dict:
                for cond_idx, val in usage_beta_dict[window_pos].items():
                    usage_beta[window_pos, cond_idx] = val
                    
            for window_pos in usage_sse_dict:
                for cond_idx, val in usage_sse_dict[window_pos].items():
                    usage_sse[window_pos, cond_idx] = val
            
            # Store this window's data
            sequences.append(ohe_seq)
            labels_list.append(labels)
            usage_dict['alpha'].append(usage_alpha)
            usage_dict['beta'].append(usage_beta)
            usage_dict['sse'].append(usage_sse)

            # Create metadata for this window
            metadata_row = {
                'genome_id': genome_id,
                'chromosome': chrom,
                'gene_id': gene_id,
                'strand': strand,
                'window_start': window_center_genomic_start,
                'window_end': window_center_genomic_end,
                'n_donor_sites': n_donor_sites,
                'n_acceptor_sites': n_acceptor_sites
            }
            metadata_rows.append(metadata_row)
        
        return sequences, labels_list, usage_dict, metadata_rows

    def to_arrays(self,
                  window_size: int = 1000,
                  context_size: int = 4500,
                  alpha_threshold: Optional[int] = None,
                  n_workers: Optional[int] = None,
                  use_parallel: bool = True,
                  save_memmap: Optional[Union[str, Path]] = None) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], pd.DataFrame]:
        """
        Convert loaded data to arrays for ML training.
        
        Args:
            window_size: Size of the window containing splice sites
            context_size: Size of context on each side of the window
            alpha_threshold: Minimum alpha value (set lower values to 0)
            n_workers: Number of parallel workers (None for CPU count)
            use_parallel: Whether to use parallel processing for genes
            save_memmap: Optional path to save arrays as memmap files
        
        Returns:
            Tuple of (sequences, labels, usage_arrays, metadata_df)
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
            label_shape = (total_windows, window_size)
            usage_shape = (total_windows, window_size, n_conditions)
            
            print(f"Pre-allocating memmap arrays in {save_memmap}")
            sequences = np.memmap(
                save_memmap / 'sequences.mmap',
                dtype=np.float32,
                mode='w+',
                shape=seq_shape
            )
            
            labels = np.memmap(
                save_memmap / 'labels.mmap',
                dtype=np.int8,
                mode='w+',
                shape=label_shape
            )
            
            usage_arrays = {
                'alpha': np.memmap(
                    save_memmap / 'usage_alpha.mmap',
                    dtype=np.float32,
                    mode='w+',
                    shape=usage_shape
                ),
                'beta': np.memmap(
                    save_memmap / 'usage_beta.mmap',
                    dtype=np.float32,
                    mode='w+',
                    shape=usage_shape
                ),
                'sse': np.memmap(
                    save_memmap / 'usage_sse.mmap',
                    dtype=np.float32,
                    mode='w+',
                    shape=usage_shape
                )
            }
            
            # Initialize usage arrays with NaN
            print("Initializing usage arrays with NaN...")
            for key in ['alpha', 'beta', 'sse']:
                usage_arrays[key][:] = np.nan
                usage_arrays[key].flush()
            
            # Fill memmap arrays incrementally
            print("\nProcessing genes and writing to memmap...")
            current_idx = 0
            all_metadata_rows = []
            
        else:
            # Non-memmap path: collect in memory
            print("Collecting windows from all genes...")
            all_sequences = []
            all_labels = []
            all_usage = {'alpha': [], 'beta': [], 'sse': []}
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
                            seq_list, lbl_list, usage_dict, metadata_rows = future.result(timeout=120)
                            
                            if save_memmap:
                                # Write incrementally to memmap (memory efficient)
                                n_windows = len(seq_list)
                                if n_windows > 0:
                                    end_idx = current_idx + n_windows
                                    if end_idx > len(sequences):
                                        print(f"\n  Warning: Exceeded estimated size, truncating at {current_idx}")
                                        break
                                    
                                    sequences[current_idx:end_idx] = np.array(seq_list, dtype=np.float32)
                                    labels[current_idx:end_idx] = np.array(lbl_list, dtype=np.int8)
                                    
                                    for key in ['alpha', 'beta', 'sse']:
                                        usage_arrays[key][current_idx:end_idx] = np.array(usage_dict[key], dtype=np.float32)
                                    
                                    all_metadata_rows.extend(metadata_rows)
                                    current_idx = end_idx
                                    
                                    # Periodic flush to disk
                                    if len(all_metadata_rows) % 500 == 0:
                                        sequences.flush()
                                        labels.flush()
                                        for key in usage_arrays:
                                            usage_arrays[key].flush()
                            else:
                                # Collect in memory for non-memmap mode
                                all_sequences.extend(seq_list)
                                all_labels.extend(lbl_list)
                                all_usage['alpha'].extend(usage_dict['alpha'])
                                all_usage['beta'].extend(usage_dict['beta'])
                                all_usage['sse'].extend(usage_dict['sse'])
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
                        seq_list, lbl_list, usage_dict, metadata_rows = self._process_gene_windows(
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
                                labels[current_idx:end_idx] = np.array(lbl_list, dtype=np.int8)
                                
                                for key in ['alpha', 'beta', 'sse']:
                                    usage_arrays[key][current_idx:end_idx] = np.array(usage_dict[key], dtype=np.float32)
                                
                                all_metadata_rows.extend(metadata_rows)
                                current_idx = end_idx
                        else:
                            # Collect in memory
                            all_sequences.extend(seq_list)
                            all_labels.extend(lbl_list)
                            all_usage['alpha'].extend(usage_dict['alpha'])
                            all_usage['beta'].extend(usage_dict['beta'])
                            all_usage['sse'].extend(usage_dict['sse'])
                            all_metadata_rows.extend(metadata_rows)
                            
                    except Exception as e:
                        print(f"Warning: Failed to process gene {gene_id}: {e}")
        
        # Finalize arrays
        if save_memmap:
            # Trim arrays to actual size (we over-allocated by 10%)
            if current_idx < len(sequences):
                print(f"Trimming memmap arrays from {len(sequences)} to {current_idx} windows...")
                # Create properly-sized memmaps
                sequences_trimmed = np.memmap(
                    save_memmap / 'sequences.mmap',
                    dtype=np.float32,
                    mode='r+',
                    shape=(current_idx, sequences.shape[1], sequences.shape[2])
                )
                labels_trimmed = np.memmap(
                    save_memmap / 'labels.mmap',
                    dtype=np.int8,
                    mode='r+',
                    shape=(current_idx, labels.shape[1])
                )
                usage_arrays_trimmed = {}
                for key in ['alpha', 'beta', 'sse']:
                    usage_arrays_trimmed[key] = np.memmap(
                        save_memmap / f'usage_{key}.mmap',
                        dtype=np.float32,
                        mode='r+',
                        shape=(current_idx, usage_arrays[key].shape[1], usage_arrays[key].shape[2])
                    )
                
                # Replace references
                sequences = sequences_trimmed
                labels = labels_trimmed
                usage_arrays = usage_arrays_trimmed
            
            # Final flush
            print("Flushing memmap arrays...")
            sequences.flush()
            labels.flush()
            for key in usage_arrays:
                usage_arrays[key].flush()
            
            n_samples = current_idx
            
            # Save metadata
            metadata_dict = {
                'sequences_shape': list(sequences.shape),
                'labels_shape': list(labels.shape),
                'usage_shape': list(usage_arrays['alpha'].shape),
                'sequences_dtype': 'float32',
                'labels_dtype': 'int8',
                'usage_dtype': 'float32',
                'window_size': window_size,
                'context_size': context_size,
                'n_conditions': n_conditions
            }
            
            with open(save_memmap / 'metadata.json', 'w') as f:
                json.dump(metadata_dict, f, indent=2)
            
            print(f"Memory-mapped files saved to: {save_memmap}")

            # Print statistics
            for key in usage_arrays:
                n_nan = np.isnan(usage_arrays[key]).sum()
                n_nonzero = np.count_nonzero(usage_arrays[key])
                print(f"  {key}: {n_nan} NaN values, {n_nonzero} non-zero values")
            
            
        else:
            # Create regular numpy arrays
            print("Converting to numpy arrays...")
            sequences = np.array(all_sequences, dtype=np.float32)
            labels = np.array(all_labels, dtype=np.int8)
            usage_arrays = {
                'alpha': np.array(all_usage['alpha'], dtype=np.float32),
                'beta': np.array(all_usage['beta'], dtype=np.float32),
                'sse': np.array(all_usage['sse'], dtype=np.float32)
            }
            n_samples = len(sequences)
        
        # Apply alpha threshold if specified
        if alpha_threshold is not None:
            print(f"Applying alpha threshold: {alpha_threshold}")
            mask = usage_arrays['alpha'] < alpha_threshold
            for key in ['alpha', 'beta', 'sse']:
                usage_arrays[key][mask] = 0
                if save_memmap:
                    usage_arrays[key].flush()

        # Print statistics
        for key in usage_arrays:
            n_nan = np.isnan(usage_arrays[key]).sum()
            n_nonzero = np.count_nonzero(usage_arrays[key])
            print(f"  {key}: {n_nan} NaN values, {n_nonzero} non-zero values")
        
        metadata = pd.DataFrame(all_metadata_rows)
        
        # Validation
        assert labels.shape[0] == n_samples, "Labels shape mismatch"
        assert usage_arrays['alpha'].shape[0] == n_samples, "Usage arrays shape mismatch"
        assert len(metadata) == n_samples, "Metadata mismatch"

        print(f"Created {n_samples} windowed examples")
        print(f"  Sequence shape: {sequences.shape}")
        print(f"  Labels shape: {labels.shape}")
        print(f"  Usage arrays shape: {usage_arrays['alpha'].shape}")
        print(f"  Total donor sites: {(labels == 1).sum()}")
        print(f"  Total acceptor sites: {(labels == 2).sum()}")
        print(f"  Total no-site positions: {(labels == 0).sum()}")
        
        return sequences, labels, usage_arrays, metadata
