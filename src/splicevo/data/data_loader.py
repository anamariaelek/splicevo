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
                 window_size: int = 200,
                 orthology_file: Optional[Union[str, Path]] = None):
        """
        Initialize the data loader.
        
        Args:
            window_size: Size of sequence window around splice sites
            orthology_file: Path to orthology TSV file with columns 
                            'ortholog_group', 'gene_id', 'genome_id'
        """
        self.window_size = window_size
        self.genomes: Dict[str, GenomeData] = {}
        self.loaded_data: List[SpliceSite] = []
        self.orthology_table: Optional[pd.DataFrame] = None
        self.usage_data: Dict[str, Dict[str, pd.DataFrame]] = {}  # {genome_id: {condition_key: usage_df}}
        self.usage_conditions: List[Dict[str, str]] = []  # List of condition metadata
        self.condition_to_key: Dict[str, str] = {}  # Maps (tissue, timepoint) to condition_key
        
        # Load orthology file if provided
        if orthology_file is not None:
            self._load_orthology_file(orthology_file)

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
        
    def generate_negative_examples(self, 
                                  transcript: Transcript, 
                                  genome,
                                  genome_id: str,
                                  n_negatives: int) -> List[SpliceSite]:
        """
        Generate negative examples from transcript sequence.
        
        Args:
            transcript: Transcript object
            genome: Loaded genome object
            genome_id: Genome identifier
            n_negatives: Number of negative examples to generate
            
        Returns:
            List of negative splice site examples
        """
        negative_examples = []
        
        # Get all splice site positions for this transcript
        all_splice_sites = transcript.splice_donor_sites.union(transcript.splice_acceptor_sites)
        
        # Get transcript span
        exons = transcript.exons.sort_values(by='start')
        chrom = exons.iloc[0]['chrom']
        tx_start = exons['start'].min()
        tx_end = exons['end'].max()
        
        # Generate random positions avoiding splice sites
        half_window = self.window_size // 2
        valid_positions = []
        
        for pos in range(tx_start + half_window, tx_end - half_window):
            if pos not in all_splice_sites:
                # Make sure we're not too close to splice sites
                too_close = any(abs(pos - ss) < 10 for ss in all_splice_sites)
                if not too_close:
                    valid_positions.append(pos)
        
        if len(valid_positions) < n_negatives:
            n_negatives = len(valid_positions)
            
        # Randomly sample positions
        sampled_positions = np.random.choice(valid_positions, size=n_negatives, replace=False)
        
        for pos in sampled_positions:
            try:
                # Get usage stats for this position
                usage_stats = self.get_usage_stats(genome_id, chrom, pos, transcript.strand)
                
                negative_site = SpliceSite.from_genomic_position(
                    genome_id=genome_id,
                    chromosome=chrom,
                    transcript_id=transcript.transcript_id,
                    gene_id=transcript.gene_id,
                    position=pos,
                    site_type=0,  # negative
                    strand=transcript.strand,
                    genome=genome,
                    window_size=self.window_size,
                    site_usage=usage_stats
                )
                negative_examples.append(negative_site)
            except Exception as e:
                # Skip if sequence extraction fails
                continue
                
        return negative_examples
        
    def _collect_all_positions(self, 
                             transcripts: List[Transcript],
                             genome_id: str,
                             negative_ratio: float) -> Tuple[List[Dict], int]:
        """
        Collect all splice site positions for batch processing using optimized bulk operations.
        
        Args:
            transcripts: List of Transcript objects
            genome_id: Genome identifier
            negative_ratio: Ratio of negative to positive examples
            
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
            batch_negatives = []
            
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
                
                # Generate negative positions efficiently
                n_negatives = int((len(transcript.splice_donor_sites) + len(transcript.splice_acceptor_sites)) * negative_ratio)
                if n_negatives > 0:
                    negative_positions = self._generate_negative_positions(transcript, n_negatives)
                    for neg_pos in negative_positions:
                        usage_stats = self.get_usage_stats(genome_id, chrom, neg_pos, transcript.strand)
                        batch_negatives.append({
                            'genome_id': genome_id,
                            'chromosome': chrom,
                            'transcript_id': transcript.transcript_id,
                            'gene_id': transcript.gene_id,
                            'position': neg_pos,
                            'site_type': 0,  # negative
                            'strand': transcript.strand,
                            'site_usage': usage_stats
                        })
            
            # Bulk extend instead of individual appends
            all_positions.extend(batch_positives)
            all_positions.extend(batch_negatives)
            positive_count += len(batch_positives)
        
        print(f"Collected {len(all_positions)} positions ({positive_count} positive)")
        return all_positions, positive_count
    
    def _generate_negative_positions(self, transcript: Transcript, n_negatives: int) -> List[int]:
        """Generate negative positions using optimized vectorized operations."""
        if n_negatives <= 0:
            return []
            
        # Get all splice site positions for this transcript
        all_splice_sites = transcript.splice_donor_sites.union(transcript.splice_acceptor_sites)
        
        # Get transcript span
        exons = transcript.exons.sort_values(by='start')
        tx_start = exons['start'].min()
        tx_end = exons['end'].max()
        
        # Use vectorized approach for position generation
        half_window = self.window_size // 2
        
        # Generate all possible positions at once
        all_possible_pos = np.arange(tx_start + half_window, tx_end - half_window)
        
        if len(all_possible_pos) == 0:
            return []
        
        # Convert splice sites to numpy array for faster operations
        splice_sites_array = np.array(list(all_splice_sites))
        
        # Vectorized filtering: remove positions too close to splice sites
        if len(splice_sites_array) > 0:
            # Use broadcasting to compute distances efficiently
            distances = np.abs(all_possible_pos[:, np.newaxis] - splice_sites_array[np.newaxis, :])
            min_distances = np.min(distances, axis=1)
            valid_mask = min_distances >= 10  # Not too close to splice sites
            valid_positions = all_possible_pos[valid_mask]
        else:
            valid_positions = all_possible_pos
        
        # Randomly sample positions
        if len(valid_positions) == 0:
            return []
        
        n_negatives = min(n_negatives, len(valid_positions))
        sampled_positions = np.random.choice(valid_positions, size=n_negatives, replace=False)
        return sampled_positions.tolist()
    
    def load_genome_data(self, 
                        genome_id: str, 
                        negative_ratio: float = 2.0,
                        max_transcripts: Optional[int] = None,
                        batch_size: int = 10000) -> List[SpliceSite]:
        """
        Load splice site data from a specific genome using optimized batch processing.
        
        Args:
            genome_id: ID of the genome to load
            negative_ratio: Ratio of negative to positive examples
            max_transcripts: Maximum number of transcripts to process (None for all)
            batch_size: Number of sequences to process in each batch
            
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
            transcripts, genome_id, negative_ratio
        )
        
        if not positions_data:
            print("No splice sites found")
            return []
        
        # Process sequences in smaller batches to reduce memory usage
        print(f"Batch processing {len(positions_data)} sequences in batches of {batch_size}...")
        all_examples = []
        
        for i in range(0, len(positions_data), batch_size):
            batch_end = min(i + batch_size, len(positions_data))
            batch_positions = positions_data[i:batch_end]
            
            if i % (batch_size * 5) == 0:
                print(f"Processing sequence batch {i//batch_size + 1}/{(len(positions_data) + batch_size - 1)//batch_size}")
            
            batch_examples = SpliceSite.from_positions_batch(
                batch_positions, genome, self.window_size
            )
            all_examples.extend(batch_examples)
            
            # Force garbage collection to free memory
            import gc
            gc.collect()
        
        print(f"Loaded {len(all_examples)} examples ({positive_count} positive, "
              f"{len(all_examples) - positive_count} negative)")
        return all_examples
     
    def load_all_genomes(self, 
                        negative_ratio: float = 2.0,
                        max_transcripts_per_genome: Optional[int] = None) -> None:
        """
        Load data from all added genomes using optimized batch processing.
        
        Args:
            negative_ratio: Ratio of negative to positive examples
            max_transcripts_per_genome: Maximum transcripts per genome
        """
        self.loaded_data = []
        
        for genome_id in self.genomes:
            genome_examples = self.load_genome_data(
                genome_id, 
                negative_ratio=negative_ratio,
                max_transcripts=max_transcripts_per_genome
            )
            self.loaded_data.extend(genome_examples)
            
        print(f"Total loaded examples: {len(self.loaded_data)}")
        
    def get_data_summary(self) -> pd.DataFrame:
        """Get summary statistics of loaded data."""
        if not self.loaded_data:
            return pd.DataFrame()
            
        summary_data = []
        
        for example in self.loaded_data:
            summary_data.append({
                'genome_id': example.genome_id,
                'chromosome': example.chromosome,
                'site_type': example.site_type,
                'strand': example.strand
            })
            
        df = pd.DataFrame(summary_data)
        
        # Generate summary statistics
        summary = df.groupby(['genome_id', 'site_type']).agg({
            'chromosome': 'nunique',
            'strand': 'count'
        })
        
        return summary
                
    def _add_ortholog_groups(self, metadata: pd.DataFrame) -> pd.DataFrame:
        """
        Add ortholog group information to metadata.
        
        Args:
            metadata: Metadata DataFrame
            
        Returns:
            Metadata DataFrame with ortholog_group column added
        """
        # Merge with orthology table
        metadata_with_orthologs = metadata.merge(
            self.orthology_table,
            on=['gene_id', 'genome_id'],
            how='left'
        )
        
        # Fill missing ortholog groups with unique identifiers
        # This ensures genes without orthologs get unique groups
        missing_mask = metadata_with_orthologs['ortholog_group'].isna()
        n_missing = missing_mask.sum()
        
        if n_missing > 0:
            # Create unique ortholog groups for genes without mappings
            unique_groups = [f"singleton_{i}" for i in range(n_missing)]
            metadata_with_orthologs.loc[missing_mask, 'ortholog_group'] = unique_groups
            
            print(f"Added {n_missing} singleton ortholog groups for genes without mappings")
        
        return metadata_with_orthologs

    def get_ortholog_groups_for_genes(self, gene_ids: List[str]) -> Dict[str, str]:
        """
        Get ortholog group mappings for a list of gene IDs.
        
        Args:
            gene_ids: List of gene IDs to look up
            
        Returns:
            Dictionary mapping gene_id to ortholog_group
        """
        if self.orthology_table is None:
            return {}
            
        # Filter orthology table for requested genes
        subset = self.orthology_table[self.orthology_table['gene_id'].isin(gene_ids)]
        return dict(zip(subset['gene_id'], subset['ortholog_group']))
        
    def get_genes_in_ortholog_group(self, ortholog_group: str) -> List[str]:
        """
        Get all gene IDs belonging to a specific ortholog group.
        
        Args:
            ortholog_group: Ortholog group identifier
            
        Returns:
            List of gene IDs in the ortholog group
        """
        if self.orthology_table is None:
            return []
            
        subset = self.orthology_table[self.orthology_table['ortholog_group'] == ortholog_group]
        return subset['gene_id'].tolist()
        
    def get_genes_in_ortholog_group_by_genome(self, ortholog_group: str, genome_id: str) -> List[str]:
        """
        Get gene IDs belonging to a specific ortholog group in a specific genome.
        
        Args:
            ortholog_group: Ortholog group identifier
            genome_id: Genome identifier
            
        Returns:
            List of gene IDs in the ortholog group for the specified genome
        """
        if self.orthology_table is None:
            return []
            
        subset = self.orthology_table[
            (self.orthology_table['ortholog_group'] == ortholog_group) &
            (self.orthology_table['genome_id'] == genome_id)
        ]
        return subset['gene_id'].tolist()
        
    def get_ortholog_groups_by_genome(self, genome_id: str) -> Dict[str, List[str]]:
        """
        Get all ortholog groups and their genes for a specific genome.
        
        Args:
            genome_id: Genome identifier
            
        Returns:
            Dictionary mapping ortholog_group to list of gene_ids for that genome
        """
        if self.orthology_table is None:
            return {}
            
        genome_subset = self.orthology_table[self.orthology_table['genome_id'] == genome_id]
        result = {}
        for ortholog_group in genome_subset['ortholog_group'].unique():
            genes = genome_subset[genome_subset['ortholog_group'] == ortholog_group]['gene_id'].tolist()
            result[ortholog_group] = genes
            
        return result
        
    def has_orthology_data(self) -> bool:
        """Check if orthology data is available."""
        return self.orthology_table is not None and len(self.orthology_table) > 0

    def to_arrays(self) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], pd.DataFrame]:
        """
        Convert loaded data to arrays suitable for ML training.
        
        Returns:
            Tuple of (sequences, labels, usage_arrays, metadata_df)
            - sequences: DNA sequences as strings
            - labels: Site type labels (0=negative, 1=donor, 2=acceptor)
            - usage_arrays: Dictionary with keys 'alpha', 'beta', 'sse', each containing
              arrays of shape (n_samples, n_conditions) for multi-task learning
            - metadata_df: DataFrame with additional metadata including condition info
        """
        if not self.loaded_data:
            raise ValueError("No data loaded. Call load_all_genomes() first.")
            
        sequences = np.array([example.sequence for example in self.loaded_data])
        labels = np.array([example.site_type for example in self.loaded_data])
        
        # Initialize usage arrays
        n_samples = len(self.loaded_data)
        n_conditions = len(self.usage_conditions)
        
        usage_arrays = {
            'alpha': np.full((n_samples, n_conditions), np.nan, dtype=np.float32),
            'beta': np.full((n_samples, n_conditions), np.nan, dtype=np.float32), 
            'sse': np.full((n_samples, n_conditions), np.nan, dtype=np.float32)
        }
        
        # Create condition index mapping for efficient lookup
        condition_to_idx = {cond['condition_key']: idx for idx, cond in enumerate(self.usage_conditions)}
        
        # Build metadata and fill usage arrays
        metadata_rows = []
        for sample_idx, example in enumerate(self.loaded_data):
            row = {
                'genome_id': example.genome_id,
                'chromosome': example.chromosome,
                'transcript_id': example.transcript_id,
                'gene_id': example.gene_id,
                'position': example.position,
                'gc_content': example.gc_content,
                'strand': example.strand,
                'n_conditions_with_usage': len(example.site_usage)
            }
            
            # Fill usage arrays for this sample
            for condition_key, usage_stats in example.site_usage.items():
                if condition_key in condition_to_idx:
                    condition_idx = condition_to_idx[condition_key]
                    usage_arrays['alpha'][sample_idx, condition_idx] = usage_stats['alpha']
                    usage_arrays['beta'][sample_idx, condition_idx] = usage_stats['beta']
                    usage_arrays['sse'][sample_idx, condition_idx] = usage_stats['sse']
            
            metadata_rows.append(row)
        
        metadata = pd.DataFrame(metadata_rows)
        
        # Add condition availability flags to metadata
        for cond in self.usage_conditions:
            condition_key = cond['condition_key']
            if condition_key in condition_to_idx:
                condition_idx = condition_to_idx[condition_key]
                metadata[f'has_{condition_key}_usage'] = ~np.isnan(usage_arrays['sse'][:, condition_idx])
        
        # Add ortholog group information if available
        if self.orthology_table is not None:
            metadata = self._add_ortholog_groups(metadata)
        
        # Ensure data and metadata are aligned
        assert len(sequences) == len(labels) == len(metadata), "Data and metadata lengths do not match"
        assert usage_arrays['alpha'].shape[0] == len(sequences), "Usage arrays and sequences length mismatch"

        return sequences, labels, usage_arrays, metadata
    
    def get_usage_array_info(self) -> Dict[str, any]:
        """
        Get information about usage arrays structure.
        
        Returns:
            Dictionary with usage array metadata including condition details
        """
        if not self.loaded_data:
            return {}
            
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
    
    def filter_by_usage_coverage(self, min_conditions: int = 1) -> 'MultiGenomeDataLoader':
        """
        Create a filtered version of the data loader with sites having usage data in at least min_conditions.
        
        Args:
            min_conditions: Minimum number of conditions with usage data required
            
        Returns:
            New MultiGenomeDataLoader instance with filtered data
        """
        if not self.loaded_data:
            raise ValueError("No data loaded. Call load_all_genomes() first.")
            
        filtered_data = []
        
        for example in self.loaded_data:
            if len(example.site_usage) >= min_conditions:
                filtered_data.append(example)
        
        # Create new instance with filtered data
        filtered_loader = MultiGenomeDataLoader(
            window_size=self.window_size
        )
        filtered_loader.genomes = self.genomes
        filtered_loader.orthology_table = self.orthology_table
        filtered_loader.usage_data = self.usage_data
        filtered_loader.usage_conditions = self.usage_conditions
        filtered_loader.condition_to_key = self.condition_to_key
        filtered_loader.loaded_data = filtered_data
        
        print(f"Filtered from {len(self.loaded_data)} to {len(filtered_data)} examples "
              f"(min_conditions={min_conditions})")
        
        return filtered_loader
