"""Utilities for processing the splice sites"""

import numpy as np
from typing import Dict, Tuple, Optional, Union, List
from dataclasses import dataclass, field
from pathlib import Path
from grelu.io.genome import CustomGenome
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp


@dataclass
class SpliceSite:
    """Individual splice site with context information."""
    genome_id: str
    chromosome: str
    transcript_id: str
    gene_id: str
    position: int
    site_type: int  # 0=negative, 1=donor, 2=acceptor
    strand: str
    site_usage: Dict[str, float] = field(default_factory=dict)  # For site-specific extension

    @classmethod
    def from_genomic_position(cls,
                            genome_id: str,
                            chromosome: str,
                            transcript_id: str,
                            gene_id: str,
                            position: int,
                            site_type: int,
                            strand: str,
                            sequence: np.ndarray,
                            site_usage: Optional[Dict[str, float]] = None) -> 'SpliceSite':
        """
        Create a SpliceSite instance with provided sequence.
        
        Args:
            genome_id: Genome identifier
            chromosome: Chromosome name
            transcript_id: Transcript identifier
            gene_id: Gene identifier
            position: Genomic position of splice site
            site_type: Type of site (0=negative, 1=donor, 2=acceptor)
            strand: Strand ('+' or '-')
            sequence: One-hot encoded sequence
            site_usage: Site usage dictionary
            
        Returns:
            SpliceSite instance
        """
        if site_usage is None:
            site_usage = {}
        
        return cls(
            genome_id=genome_id,
            chromosome=chromosome,
            transcript_id=transcript_id,
            gene_id=gene_id,
            position=position,
            site_type=site_type,
            sequence=sequence,
            strand=strand,
            site_usage=site_usage
        )
    
    @classmethod
    def from_positions_batch(cls, 
                            positions_data: List[Dict],
                            n_workers: Optional[int] = None,
                            use_processes: bool = False) -> List['SpliceSite']:
        """
        Create multiple SpliceSite instances with provided sequences.

        Args:
            positions_data: List of dicts with keys: genome_id, chromosome, transcript_id,
                          gene_id, position, site_type, strand, site_usage
            n_workers: Number of parallel workers. If None, uses CPU count
            use_processes: If True, use ProcessPoolExecutor (for CPU-bound), 
                         otherwise ThreadPoolExecutor (for I/O-bound)
            
        Returns:
            List of SpliceSite instances
        """
        if n_workers is None:
            n_workers = mp.cpu_count()
        
        # For small batches, parallel overhead isn't worth it
        if len(positions_data) < 100:
            results = []
            for data in tqdm(positions_data, desc="Creating splice sites", unit="site"):
                site_usage = data['site_usage'] if 'site_usage' in data else {}
                results.append(cls(
                    genome_id=data['genome_id'],
                    chromosome=data['chromosome'],
                    transcript_id=data['transcript_id'],
                    gene_id=data['gene_id'],
                    position=data['position'],
                    site_type=data['site_type'],
                    strand=data['strand'],
                    site_usage=site_usage
                ))
        
            return results
        
        # Helper function for parallel execution
        def create_site(data):
            site_usage = data['site_usage'] if 'site_usage' in data else {}
            return cls(
                genome_id=data['genome_id'],
                chromosome=data['chromosome'],
                transcript_id=data['transcript_id'],
                gene_id=data['gene_id'],
                position=data['position'],
                site_type=data['site_type'],
                strand=data['strand'],
                site_usage=site_usage
            )
        
        # Choose executor based on workload type
        ExecutorClass = ProcessPoolExecutor if use_processes else ThreadPoolExecutor
        
        with ExecutorClass(max_workers=n_workers) as executor:
            # Use tqdm with concurrent futures
            results = list(tqdm(
                executor.map(create_site, positions_data),
                total=len(positions_data),
                desc="Creating splice sites",
                unit="site"
            ))
        
        return results
    
    def get_site_type_name(self) -> str:
        """
        Get human-readable name for site type.
        
        Returns:
            Site type name
        """
        type_names = {0: "negative", 1: "donor", 2: "acceptor"}
        return type_names.get(self.site_type, "unknown")
    
    def is_positive_site(self) -> bool:
        """Check if this is a positive splice site (donor or acceptor)."""
        return self.site_type in [1, 2]
    
    def is_donor_site(self) -> bool:
        """Check if this is a donor splice site."""
        return self.site_type == 1
    
    def is_acceptor_site(self) -> bool:
        """Check if this is an acceptor splice site."""
        return self.site_type == 2
    
    def is_negative_site(self) -> bool:
        """Check if this is a negative example."""
        return self.site_type == 0
    
    def validate(self) -> bool:
        """
        Validate the splice site data.
        
        Returns:
            True if valid, False otherwise
        """
        # Check basic fields
        if not all([self.genome_id, self.chromosome, self.transcript_id, self.gene_id]):
            return False
            
        # Check site type
        if self.site_type not in [0, 1, 2]:
            return False
            
        # Check strand
        if self.strand not in ['+', '-']:
            return False
            
        # Check sequence shape
        if self.sequence.ndim != 2 or self.sequence.shape[1] != 4:
            return False
            
        return True
    
    def to_dict(self) -> Dict:
        """
        Convert SpliceSite to dictionary for serialization.
        
        Returns:
            Dictionary representation
        """
        return {
            'genome_id': self.genome_id,
            'chromosome': self.chromosome,
            'transcript_id': self.transcript_id,
            'gene_id': self.gene_id,
            'position': self.position,
            'site_type': self.site_type,
            'site_type_name': self.get_site_type_name(),
            'strand': self.strand,
            'site_usage': self.site_usage
        }

    def __str__(self) -> str:
        """String representation of the splice site."""
        return (f"SpliceSite({self.genome_id}:{self.chromosome}:{self.position} "
                f"{self.get_site_type_name()} {self.strand})")
    
    def __repr__(self) -> str:
        """Detailed representation of the splice site."""
        return (f"SpliceSite(genome_id='{self.genome_id}', "
                f"gene_id='{self.gene_id}', "
                f"transcript_id='{self.transcript_id}', "
                f"chromosome='{self.chromosome}', position={self.position}, "
                f"site_type={self.site_type}, strand='{self.strand}')")
