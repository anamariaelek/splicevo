"""Utilities for processing the splice sites"""

import numpy as np
from typing import Dict, Tuple, Optional, Union, List
from dataclasses import dataclass, field
from pathlib import Path
from grelu.io.genome import CustomGenome


@dataclass
class SpliceSite:
    """Individual splice site with context information."""
    genome_id: str
    chromosome: str
    transcript_id: str
    gene_id: str
    position: int
    site_type: int  # 0=negative, 1=donor, 2=acceptor
    sequence: np.ndarray  # One-hot encoded sequence
    gc_content: float
    strand: str
    site_usage: Dict[str, float] = field(default_factory=dict)  # For site-specific extension

    @staticmethod
    def calculate_gc_content(sequence: str) -> float:
        """
        Calculate GC content of a DNA sequence.
        
        Args:
            sequence: DNA sequence string
            
        Returns:
            GC content as fraction (0.0 to 1.0)
        """
        if len(sequence) == 0:
            return 0.0
        gc_count = sequence.upper().count('G') + sequence.upper().count('C')
        return gc_count / len(sequence)
    
    @classmethod
    def from_genomic_position(cls,
                            genome_id: str,
                            chromosome: str,
                            transcript_id: str,
                            gene_id: str,
                            position: int,
                            site_type: int,
                            strand: str,
                            genome: CustomGenome,
                            window_size: int = 200,
                            site_usage: Optional[Dict[str, float]] = None) -> 'SpliceSite':
        """
        Create a SpliceSite instance by extracting sequence from genomic coordinates.
        
        Args:
            genome_id: Genome identifier
            chromosome: Chromosome name
            transcript_id: Transcript identifier
            gene_id: Gene identifier
            position: Genomic position of splice site
            site_type: Type of site (0=negative, 1=donor, 2=acceptor)
            strand: Strand ('+' or '-')
            genome: Loaded CustomGenome object
            window_size: Size of sequence window around splice site
            site_usage: Site usage dictionary
            
        Returns:
            SpliceSite instance with extracted sequence
        """
        if site_usage is None:
            site_usage = {}
            
        # Extract sequence window
        half_window = window_size // 2
        start = position - half_window
        end = position + half_window
        
        # Adjust for 1-based coordinates in genome.get_seq
        seq = genome.get_seq(chromosome, start + 1, end, strand == '-')
        if not isinstance(seq, str):
            seq = str(seq)
        
        # Handle sequences that are shorter than expected window size
        # This happens when the window extends beyond chromosome boundaries
        if len(seq) < window_size:
            # Calculate how much padding is needed
            padding_needed = window_size - len(seq)
            
            # Pad with N's - distribute padding as evenly as possible
            left_pad = padding_needed // 2
            right_pad = padding_needed - left_pad
            
            seq = 'N' * left_pad + seq + 'N' * right_pad
            
        elif len(seq) > window_size:
            # Truncate if somehow we got too much sequence
            seq = seq[:window_size]
        
        # Check for too many N's
        n_count = seq.upper().count('N')
        if n_count > len(seq) * 0.8:  # Skip if more than 80% N's
            raise ValueError(f"Too many N's in sequence: {n_count}/{len(seq)}")
            
        # Calculate GC content
        gc_content = cls.calculate_gc_content(seq)
        
        # One-hot encode
        from tangermeme.utils import one_hot_encode
        ohe_seq = one_hot_encode(seq)
        
        return cls(
            genome_id=genome_id,
            chromosome=chromosome,
            transcript_id=transcript_id,
            gene_id=gene_id,
            position=position,
            site_type=site_type,
            sequence=ohe_seq,
            gc_content=gc_content,
            strand=strand,
            site_usage=site_usage
        )
    
    @classmethod
    def from_positions_batch(cls,
                           positions_data: List[Dict],
                           genome,
                           window_size: int = 200) -> List['SpliceSite']:
        """
        Create multiple SpliceSite instances efficiently by batching sequence extraction.
        
        This is much faster than calling from_genomic_position() repeatedly.
        
        Args:
            positions_data: List of dicts with keys: genome_id, chromosome, transcript_id,
                          gene_id, position, site_type, strand, site_usage
            genome: Loaded genome object
            window_size: Size of sequence window around splice sites
            
        Returns:
            List of SpliceSite instances
        """
        if not positions_data:
            return []
        
        # Group by chromosome for efficient extraction
        chr_groups = {}
        for i, data in enumerate(positions_data):
            chrom = data['chromosome']
            if chrom not in chr_groups:
                chr_groups[chrom] = []
            chr_groups[chrom].append((i, data))
        
        # Pre-allocate results
        results = [None] * len(positions_data)
        half_window = window_size // 2
        
        # Process each chromosome
        for chrom, chr_data in chr_groups.items():
            # Sort by position for potentially more efficient access
            chr_data.sort(key=lambda x: x[1]['position'])
            
            # Extract all sequences for this chromosome
            sequences = []
            gc_contents = []
            
            for _, data in chr_data:
                position = data['position']
                strand = data['strand']
                start = position - half_window
                end = position + half_window
                
                try:
                    # Extract sequence
                    seq = genome.get_seq(chrom, start + 1, end, strand == '-')
                    if not isinstance(seq, str):
                        seq = str(seq)
                    
                    # Handle sequences that are shorter than expected window size
                    # This happens when the window extends beyond chromosome boundaries
                    if len(seq) < window_size:
                        # Calculate how much padding is needed
                        padding_needed = window_size - len(seq)
                        
                        # Pad with N's - distribute padding as evenly as possible
                        left_pad = padding_needed // 2
                        right_pad = padding_needed - left_pad
                        
                        seq = 'N' * left_pad + seq + 'N' * right_pad
                        
                    elif len(seq) > window_size:
                        # Truncate if somehow we got too much sequence
                        seq = seq[:window_size]
                    
                    # Check for too many N's and skip if necessary
                    n_count = seq.upper().count('N')
                    if n_count > len(seq) * 0.8:  # Skip if more than 80% N's
                        raise ValueError(f"Too many N's: {n_count}/{len(seq)}")
                    
                    sequences.append(seq)
                    gc_contents.append(cls.calculate_gc_content(seq))
                    
                except Exception:
                    # Use empty sequence for failed extractions
                    sequences.append('N' * window_size)
                    gc_contents.append(0.0)
            
            # Batch one-hot encode all sequences for this chromosome
            try:
                from tangermeme.utils import one_hot_encode
                # Try to batch encode if possible
                if len(sequences) > 1:
                    # For now, encode individually but could be optimized further
                    ohe_sequences = [one_hot_encode(seq) for seq in sequences]
                else:
                    ohe_sequences = [one_hot_encode(sequences[0])]
            except Exception:
                # Fallback to individual encoding
                from tangermeme.utils import one_hot_encode
                ohe_sequences = []
                for seq in sequences:
                    try:
                        ohe_sequences.append(one_hot_encode(seq))
                    except Exception:
                        # Create dummy array for failed encoding
                        import numpy as np
                        ohe_sequences.append(np.zeros((window_size, 4)))
            
            # Create SpliceSite objects
            for j, (orig_idx, data) in enumerate(chr_data):
                site_usage = data.get('site_usage', {})
                
                results[orig_idx] = cls(
                    genome_id=data['genome_id'],
                    chromosome=data['chromosome'],
                    transcript_id=data['transcript_id'],
                    gene_id=data['gene_id'],
                    position=data['position'],
                    site_type=data['site_type'],
                    sequence=ohe_sequences[j],
                    gc_content=gc_contents[j],
                    strand=data['strand'],
                    site_usage=site_usage
                )
        
        return [r for r in results if r is not None]
    
    def get_sequence_string(self) -> str:
        """
        Convert one-hot encoded sequence back to string.
        
        Returns:
            DNA sequence as string
        """
        # Manual decode since tangermeme doesn't have one_hot_decode
        if hasattr(self.sequence, 'numpy'):
            seq_array = self.sequence.numpy()
        else:
            seq_array = self.sequence
            
        # Handle torch tensors
        if hasattr(seq_array, 'detach'):
            seq_array = seq_array.detach().numpy()
            
        mapping = ['A', 'C', 'G', 'T']
        decoded_chars = []
        
        for pos in seq_array:
            # Convert to numpy if needed
            if hasattr(pos, 'numpy'):
                pos = pos.numpy()
            elif hasattr(pos, 'detach'):
                pos = pos.detach().numpy()
                
            # Check if this is a valid position
            if len(pos) < 4:
                decoded_chars.append('N')
            elif pos[:4].sum() == 0:
                decoded_chars.append('N')  # All zeros in ATCG = unknown
            else:
                max_idx = pos[:4].argmax()  # Only consider ATCG
                decoded_chars.append(mapping[max_idx])
        
        return ''.join(decoded_chars)
    
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
    
    def get_consensus_score(self) -> float:
        """
        Calculate consensus sequence score for splice sites.
        
        Returns:
            Consensus score (higher = more canonical)
        """
        seq_str = self.get_sequence_string()
        center = len(seq_str) // 2
        
        if self.is_donor_site():
            # Check for GT dinucleotide at splice site
            if center + 1 < len(seq_str):
                dinuc = seq_str[center:center+2]
                if dinuc.upper() == "GT":
                    return 1.0
                elif dinuc.upper() == "GC":  # Alternative donor
                    return 0.5
            return 0.0
            
        elif self.is_acceptor_site():
            # Check for AG dinucleotide at splice site
            if center - 1 >= 0:
                dinuc = seq_str[center-1:center+1]
                if dinuc.upper() == "AG":
                    return 1.0
            return 0.0
            
        return 0.0  # Negative sites have no consensus
    
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
            
        # Check GC content range
        if not (0.0 <= self.gc_content <= 1.0):
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
            'gc_content': self.gc_content,
            'strand': self.strand,
            'site_usage': self.site_usage,
            'consensus_score': self.get_consensus_score(),
            'sequence_length': len(self.sequence)
        }

    def __str__(self) -> str:
        """String representation of the splice site."""
        return (f"SpliceSite({self.genome_id}:{self.chromosome}:{self.position} "
                f"{self.get_site_type_name()} {self.strand} "
                f"GC={self.gc_content:.2f})")
    
    def __repr__(self) -> str:
        """Detailed representation of the splice site."""
        return (f"SpliceSite(genome_id='{self.genome_id}', "
                f"chromosome='{self.chromosome}', position={self.position}, "
                f"site_type={self.site_type}, strand='{self.strand}')")
