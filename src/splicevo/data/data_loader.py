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
    """
    
    def __init__(self, window_size: int = 200):
        """
        Initialize the data loader.
        
        Args:
            window_size: Size of sequence window around splice sites
        """
        self.window_size = window_size
        self.genomes: Dict[str, GenomeData] = {}
        self.loaded_data: List[SpliceSite] = []
        
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
                negative_site = SpliceSite.from_genomic_position(
                    genome_id=genome_id,
                    chromosome=chrom,
                    transcript_id=transcript.transcript_id,
                    gene_id=transcript.gene_id,
                    position=pos,
                    site_type=0,  # negative
                    strand=transcript.strand,
                    genome=genome,
                    window_size=self.window_size
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
                    batch_positives.append({
                        'genome_id': genome_id,
                        'chromosome': chrom,
                        'transcript_id': transcript.transcript_id,
                        'gene_id': transcript.gene_id,
                        'position': donor_pos,
                        'site_type': 1,  # donor
                        'strand': transcript.strand,
                        'site_usage': {}
                    })
                    
                # Collect acceptor sites in bulk
                for acceptor_pos in transcript.splice_acceptor_sites:
                    batch_positives.append({
                        'genome_id': genome_id,
                        'chromosome': chrom,
                        'transcript_id': transcript.transcript_id,
                        'gene_id': transcript.gene_id,
                        'position': acceptor_pos,
                        'site_type': 2,  # acceptor
                        'strand': transcript.strand,
                        'site_usage': {}
                    })
                
                # Generate negative positions efficiently
                n_negatives = int((len(transcript.splice_donor_sites) + len(transcript.splice_acceptor_sites)) * negative_ratio)
                if n_negatives > 0:
                    negative_positions = self._generate_negative_positions(transcript, n_negatives)
                    for neg_pos in negative_positions:
                        batch_negatives.append({
                            'genome_id': genome_id,
                            'chromosome': chrom,
                            'transcript_id': transcript.transcript_id,
                            'gene_id': transcript.gene_id,
                            'position': neg_pos,
                            'site_type': 0,  # negative
                            'strand': transcript.strand,
                            'site_usage': {}
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
                        max_transcripts: Optional[int] = None) -> List[SpliceSite]:
        """
        Load splice site data from a specific genome using optimized batch processing.
        
        Args:
            genome_id: ID of the genome to load
            negative_ratio: Ratio of negative to positive examples
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
            transcripts, genome_id, negative_ratio
        )
        
        if not positions_data:
            print("No splice sites found")
            return []
        
        # Batch process all sequences (this is the optimized part!)
        print(f"Batch processing {len(positions_data)} sequences...")
        examples = SpliceSite.from_positions_batch(
            positions_data, genome, self.window_size
        )
        
        print(f"Loaded {len(examples)} examples ({positive_count} positive, "
              f"{len(examples) - positive_count} negative)")
        return examples
     
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
        
    def to_arrays(self) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Convert loaded data to arrays suitable for ML training.
        
        Returns:
            Tuple of (sequences, labels, metadata_df)
        """
        if not self.loaded_data:
            raise ValueError("No data loaded. Call load_all_genomes() first.")
            
        sequences = np.array([example.sequence for example in self.loaded_data])
        labels = np.array([example.site_type for example in self.loaded_data])
        
        metadata = pd.DataFrame([
            {
                'genome_id': example.genome_id,
                'chromosome': example.chromosome,
                'transcript_id': example.transcript_id,
                'gene_id': example.gene_id,
                'position': example.position,
                'gc_content': example.gc_content,
                'strand': example.strand,
                'site_usage': example.site_usage
            }
            for example in self.loaded_data
        ])
        
        return sequences, labels, metadata