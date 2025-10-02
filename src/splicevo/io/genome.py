"""Utilities for working with genomes"""

from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from grelu.io.genome import CustomGenome
from tangermeme.utils import one_hot_encode

from .gene_annotation import Transcript


@dataclass
class GenomeData:
    """
    Container for genome data including sequences and annotations.
    
    This class encapsulates genome information and provides methods for
    loading genome sequences and encoding transcripts.
    """
    genome_id: str
    genome_path: Path
    gtf_path: Path
    chromosomes: Optional[List[str]] = None
    metadata: Dict = field(default_factory=dict)
    _genome: Optional[CustomGenome] = field(default=None, init=False, repr=False)
    
    def load_genome(self) -> CustomGenome:
        """
        Load the genome from the FASTA file.
        
        Returns:
            CustomGenome object for sequence access
        """
        if self._genome is None:
            # Ensure genome_path is a str
            fasta_path = str(self.genome_path)
            self._genome = CustomGenome(fasta_path)
        return self._genome
    
    @property
    def genome(self) -> CustomGenome:
        """Lazy-loaded genome property."""
        return self.load_genome()
    
    def encode_transcript(self, transcript: Transcript) -> tuple:
        """
        Given a Transcript object, return the one-hot encoded sequence and 
        a parallel vector with splice site annotations (0=not a splice site, 1=donor, 2=acceptor).
        
        Args:
            transcript: Transcript object to encode
            
        Returns:
            Tuple of (one-hot encoded sequence, annotation vector)
        """
        genome = self.load_genome()
        
        # Get transcript genomic span (from first exon start to last exon end)
        exons = transcript.exons.sort_values(by='start')
        chrom = exons.iloc[0]['chrom']
        tx_start = exons['start'].min()
        tx_end = exons['end'].max()
        rc = transcript.strand == '-'

        # Fetch full transcript sequence
        # Add 1 to start coordinate as genome.get_seq expects 1-based coordinates
        seq = genome.get_seq(chrom, tx_start + 1, tx_end, rc)
        # Ensure seq is a string (pyfaidx may return a Sequence object)
        if not isinstance(seq, str):
            seq = str(seq)

        # One-hot encode sequence
        ohe_seq = one_hot_encode(seq)

        # Build genomic position map for the full transcript
        if transcript.strand == '+':
            pos_map = list(range(tx_start, tx_end+1))
        else:
            pos_map = list(range(tx_end, tx_start-1, -1))

        # Build annotation vector: 0 for not a splice site, 1 for donor, 2 for acceptor
        ann_vec = [0] * len(pos_map)
        donor_sites = transcript.splice_donor_sites
        acceptor_sites = transcript.splice_acceptor_sites
        for i, gpos in enumerate(pos_map):
            if gpos in donor_sites:
                ann_vec[i] = 1
            elif gpos in acceptor_sites:
                ann_vec[i] = 2

        # Convert annotation vector to tensor
        try:
            import torch
            ann_vec = torch.tensor(ann_vec, dtype=torch.long)
        except ImportError:
            pass  # If torch is not available, return as list
        return ohe_seq, ann_vec
    