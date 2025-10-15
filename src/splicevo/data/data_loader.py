"""Data loading utilities for multi-genome splice site model training."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path

from ..io.genome import GenomeData
from ..io.gene_annotation import GTFProcessor, Transcript
from ..io.splice_sites import SpliceSite


class MultiGenomeDataLoader:
    """
    Data loader for multi-genome splice site prediction.
    
    This class handles loading and processing splice site data from multiple genomes,
    keeping track of genome, chromosome, and transcript origin for each example.
    Supports loading splice site usage statistics from tissue/cell-type specific files.
    """
    
    def __init__(self,
                 orthology_file: Optional[Union[str, Path]] = None):
        """
        Initialize the data loader.
        
        Args:
            orthology_file: Path to orthology TSV file with columns 
                            'ortholog_group', 'gene_id', 'genome_id'
        """
        self.genomes: Dict[str, GenomeData] = {}
        self.loaded_data: List[SpliceSite] = []
        self.orthology_table: Optional[pd.DataFrame] = None
        self.usage_data: Dict[str, Dict[str, pd.DataFrame]] = {}  # {genome_id: {condition_key: usage_df}}
        self.usage_conditions: List[Dict[str, str]] = []  # List of condition metadata
        self.condition_to_key: Dict[str, str] = {}  # Maps (tissue, timepoint) to condition_key
        # Load orthology file if provided
        if orthology_file is not None:
            self._load_orthology_file(orthology_file)
        
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

    def _load_orthology_file(self, orthology_file: Union[str, Path]) -> None:
        """
        Load orthology mapping from file.
        
        Args:
            orthology_file: Path to orthology file
            
        Expected format: TSV file with columns: ortholog_group, gene_id, genome_id
        """
        orthology_path = Path(orthology_file)
        
        if not orthology_path.exists():
            raise FileNotFoundError(f"Orthology file not found: {orthology_path}")
            
        print(f"Loading orthology file: {orthology_path}")
        
        try:
            # Read as tab-separated file
            df = pd.read_csv(orthology_path, sep='\t', header=0)
            
            # Check for required columns
            required_columns = {'ortholog_group', 'gene_id', 'genome_id'}
            available_columns = set(df.columns)
            
            if not required_columns.issubset(available_columns):
                missing_columns = required_columns - available_columns
                raise ValueError(
                    f"Orthology file missing required columns: {missing_columns}. "
                    f"Expected columns: {required_columns}, found: {available_columns}"
                )
            
            # Use provided table
            self.orthology_table = df[['ortholog_group', 'gene_id', 'genome_id']].copy()
            
            # Clean up the data
            self.orthology_table = self.orthology_table.dropna()

            # Make sure that each gene_id genome_id pair appears only once
            self.orthology_table = self.orthology_table.drop_duplicates(subset=['gene_id', 'genome_id'])

            # Ensure correct data types
            self.orthology_table['gene_id'] = self.orthology_table['gene_id'].astype(str)
            self.orthology_table['ortholog_group'] = self.orthology_table['ortholog_group'].astype(str)
            self.orthology_table['genome_id'] = self.orthology_table['genome_id'].astype(str)
            
            print(f"Loaded orthology mappings for {len(self.orthology_table)} genes from "
                  f"{self.orthology_table['genome_id'].nunique()} genomes in "
                  f"{self.orthology_table['ortholog_group'].nunique()} ortholog groups")
                  
        except Exception as e:
            print(f"Warning: Could not load orthology file {orthology_path}: {e}")
            self.orthology_table = None
            
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
        for batch_start in range(0, len(transcripts), batch_size):
            batch_end = min(batch_start + batch_size, len(transcripts))
            batch_transcripts = transcripts[batch_start:batch_end]
            
            if batch_start % 5000 == 0:
                print(f"Processing batch {batch_start//batch_size + 1}/{(len(transcripts) + batch_size - 1)//batch_size}")
            
            # Collect all positive positions for this batch
            batch_positives = []
            
            for transcript in batch_transcripts:
                # Skip transcripts without splice sites
                if len(transcript.splice_donor_sites) == 0 and len(transcript.splice_acceptor_sites) == 0:
                    continue
                    
                chrom = transcript.exons.iloc[0]['chrom']
                
                # Collect donor sites in bulk
                for donor_pos in transcript.splice_donor_sites:
                    usage_stats = self.get_usage_stats(genome_id, chrom, donor_pos, transcript.strand)
                    batch_positives.append({
                        'genome_id': genome_id,
                        'chromosome': chrom,
                        'transcript_id': transcript.transcript_id,
                        'gene_id': transcript.gene_id,
                        'position': donor_pos,
                        'site_type': 1,  # donor
                        'strand': transcript.strand,
                        'site_usage': usage_stats
                    })
                    
                # Collect acceptor sites in bulk
                for acceptor_pos in transcript.splice_acceptor_sites:
                    usage_stats = self.get_usage_stats(genome_id, chrom, acceptor_pos, transcript.strand)
                    batch_positives.append({
                        'genome_id': genome_id,
                        'chromosome': chrom,
                        'transcript_id': transcript.transcript_id,
                        'gene_id': transcript.gene_id,
                        'position': acceptor_pos,
                        'site_type': 2,  # acceptor
                        'strand': transcript.strand,
                        'site_usage': usage_stats
                    })
            
            # Bulk extend instead of individual appends
            all_positions.extend(batch_positives)
            positive_count += len(batch_positives)
        
        print(f"Collected {len(all_positions)} positions ({positive_count} positive)")
        return all_positions, positive_count
    
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
        print(f"Loading genome {genome_id}...")
        genome = genome_data.load_genome()
        
        print(f"Processing GTF annotations for {genome_id}...")
        gtf_processor = GTFProcessor(str(genome_data.gtf_path))
        transcripts = gtf_processor.process_gtf(chromosomes=genome_data.chromosomes)
        
        if max_transcripts is not None:
            transcripts = transcripts[:max_transcripts]
        
        # Collect all positions first (fast)
        positions_data, positive_count = self._collect_all_positions(
            transcripts, genome_id
        )
        
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
        summary = df.groupby(['genome_id', 'site_type']).agg({
            'chromosome': 'nunique',
            'strand': 'count'
        })
        
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
            'n_conditions': usage_arrays['alpha'].shape[1],
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

    def to_arrays(self,
                  window_size: int = 1000,
                  context_size: int = 4500) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], pd.DataFrame]:
        """
        Convert loaded data to arrays for ML training.
        
        Args:
            window_size: Size of the window containing splice sites
            context_size: Size of context on each side of the window
        
        Returns:
            Tuple of (sequences, labels, usage_arrays, metadata_df)
            - sequences: One-hot encoded DNA sequences of shape (n_samples, total_window, 4)
              where total_window = context_size + window_size + context_size
            - labels: Array of shape (n_samples, window_size) with values:
              0 = no splice site, 1 = donor site, 2 = acceptor site
            - usage_arrays: Dictionary with keys 'alpha', 'beta', 'sse', each containing
              arrays of shape (n_samples, window_size, n_conditions)
              Usage values are only meaningful where labels > 0
            - metadata_df: DataFrame with window metadata
        """
        if not self.loaded_data:
            raise ValueError("No data loaded. Call load_all_genomes_data() first.")

        # One-hot encoding mapping
        nuc_to_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
        
        # Pre-build index for O(1) splice site lookup - THIS IS THE KEY OPTIMIZATION
        print("Building splice site index...")
        splice_site_index = {}
        for site in self.loaded_data:
            key = (site.genome_id, site.chromosome, site.position, site.strand)
            splice_site_index[key] = site
        print(f"Indexed {len(splice_site_index)} splice sites")
        
        # Get all splice sites info
        df = self.get_dataframe()

        all_sequences = []
        all_labels = []
        all_usage = {'alpha': [], 'beta': [], 'sse': []}
        all_metadata_rows = []
        
        n_conditions = len(self.usage_conditions)
        condition_to_idx = {cond['condition_key']: idx for idx, cond in enumerate(self.usage_conditions)}
        
        # For each gene in each genome, get interval from 5'-most to 3'-most splice sites
        loaded_genomes = df['genome_id'].unique()
        total_window = context_size + window_size + context_size
        
        n_total_windows = 0
        n_skipped_windows = 0

        for genome_id in loaded_genomes:
            print(f"Processing genome {genome_id}...")
            genome = self.genomes[genome_id].load_genome()
            df_genome = df[df['genome_id'] == genome_id]
            genes = df_genome['gene_id'].unique()
            
            print(f"  Processing {len(genes)} genes...")

            for gene_idx, gene_id in enumerate(genes):
                if gene_idx % 100 == 0:
                    print(f"    Gene {gene_idx}/{len(genes)}")
                    
                df_gene = df_genome[df_genome['gene_id'] == gene_id]
                chrom = df_gene['chromosome'].iloc[0]
                strand = df_gene['strand'].iloc[0]
                
                # Get 5'-most and 3'-most positions
                min_pos = df_gene['position'].min() 
                max_pos = df_gene['position'].max()

                # Extend the gene range for specified context
                gene_start = min_pos - context_size
                gene_end = max_pos + context_size

                # Get sequence of the gene with context
                rc = strand == '-'
                seq = genome.get_seq(chrom, gene_start + 1, gene_end, rc)
                if rc:
                    seq = seq.complement
                
                # Ensure seq is a string
                if not isinstance(seq, str):
                    seq = str(seq).upper()
                else:
                    seq = seq.upper()
                
                # Pre-compute one-hot encoding for the entire gene sequence
                gene_ohe = np.zeros((len(seq), 4), dtype=np.float32)
                for i, nuc in enumerate(seq):
                    idx = nuc_to_idx.get(nuc, 4)
                    if idx < 4:
                        gene_ohe[i, idx] = 1.0
                
                # Split gene into windows shifted by window_size
                for window_start in range(0, len(seq) - total_window + 1, window_size):
                    n_total_windows += 1
                    
                    # Calculate the genomic position of this window
                    window_genomic_start = gene_start + window_start
                    
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
                        n_skipped_windows += 1
                        continue
                    
                    # Extract pre-computed one-hot encoding
                    ohe_seq = gene_ohe[window_start:window_start + total_window]
                    
                    # Initialize label array for this window (0=no site, 1=donor, 2=acceptor)
                    labels = np.zeros(window_size, dtype=np.int8)
                    
                    # Initialize usage arrays for this window
                    usage_alpha = np.full((window_size, n_conditions), np.nan, dtype=np.float32)
                    usage_beta = np.full((window_size, n_conditions), np.nan, dtype=np.float32)
                    usage_sse = np.full((window_size, n_conditions), np.nan, dtype=np.float32)
                    
                    n_donor_sites = 0
                    n_acceptor_sites = 0
                    
                    # Mark positions and add usage stats for each site
                    for _, site_row in sites_in_window.iterrows():
                        site_pos = site_row['position']
                        site_type = site_row['site_type']
                        
                        # Calculate position within the window (0 to window_size-1)
                        window_pos = site_pos - window_center_genomic_start
                        
                        if 0 <= window_pos < window_size:
                            lookup_key = (genome_id, chrom, site_pos, strand)
                            splice_site = splice_site_index.get(lookup_key)
                            
                            if splice_site is not None:
                                # Set label: 1 for donor, 2 for acceptor
                                labels[window_pos] = site_type
                                
                                if site_type == 1:
                                    n_donor_sites += 1
                                elif site_type == 2:
                                    n_acceptor_sites += 1
                                
                                # Fill usage stats for this position
                                for condition_key, usage_stats in splice_site.site_usage.items():
                                    if condition_key in condition_to_idx:
                                        cond_idx = condition_to_idx[condition_key]
                                        usage_alpha[window_pos, cond_idx] = usage_stats['alpha']
                                        usage_beta[window_pos, cond_idx] = usage_stats['beta']
                                        usage_sse[window_pos, cond_idx] = usage_stats['sse']
                    
                    # Store this window's data
                    all_sequences.append(ohe_seq)
                    all_labels.append(labels)
                    all_usage['alpha'].append(usage_alpha)
                    all_usage['beta'].append(usage_beta)
                    all_usage['sse'].append(usage_sse)
                    
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
                    all_metadata_rows.append(metadata_row)
        
        print("Converting to numpy arrays...")
        print(f"Skipped {n_skipped_windows}/{n_total_windows} windows with no splice sites ({100*n_skipped_windows/n_total_windows:.1f}%)")
        
        # Convert lists to arrays
        sequences = np.array(all_sequences, dtype=np.float32)  # (n_samples, total_window, 4)
        labels = np.array(all_labels, dtype=np.int8)  # (n_samples, window_size)
        
        usage_arrays = {
            'alpha': np.array(all_usage['alpha'], dtype=np.float32),  # (n_samples, window_size, n_conditions)
            'beta': np.array(all_usage['beta'], dtype=np.float32),
            'sse': np.array(all_usage['sse'], dtype=np.float32)
        }
        
        metadata = pd.DataFrame(all_metadata_rows)
        
        # Validation
        n_samples = len(sequences)
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
