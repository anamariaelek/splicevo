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

        """
        Get comprehensive information about a specific gene.
        
        Args:
            gene_id: Gene identifier to look up
            transcripts: List of Transcript objects to search through
            
        Returns:
            Dictionary containing gene information including:
            - gene_id: Gene identifier
            - chromosome: Chromosome location
            - strand: Genomic strand
            - start: Gene start position
            - end: Gene end position
            - transcript_count: Number of transcripts for this gene
            - transcripts: List of transcript IDs
            - total_exons: Total number of exons across all transcripts
            - splice_sites: Summary of splice sites
        """
        # Filter transcripts for this gene
        gene_transcripts = [t for t in transcripts if t.gene_id == gene_id]
        
        if not gene_transcripts:
            return {
                'gene_id': gene_id,
                'error': 'Gene not found',
                'transcript_count': 0
            }
        
        # Get basic gene information from first transcript
        first_transcript = gene_transcripts[0]
        chromosome = first_transcript.exons.iloc[0]['chrom']
        strand = first_transcript.strand
        
        # Calculate gene span (from earliest start to latest end across all transcripts)
        all_starts = []
        all_ends = []
        total_exons = 0
        transcript_ids = []
        
        all_donors = set()
        all_acceptors = set()
        
        for transcript in gene_transcripts:
            transcript_ids.append(transcript.transcript_id)
            exon_starts = transcript.exons['start'].tolist()
            exon_ends = transcript.exons['end'].tolist()
            all_starts.extend(exon_starts)
            all_ends.extend(exon_ends)
            total_exons += len(transcript.exons)
            
            # Collect splice sites
            all_donors.update(transcript.splice_donor_sites)
            all_acceptors.update(transcript.splice_acceptor_sites)
        
        gene_start = min(all_starts)
        gene_end = max(all_ends)
        
        return {
            'gene_id': gene_id,
            'chromosome': chromosome,
            'strand': strand,
            'start': gene_start,
            'end': gene_end,
            'length': gene_end - gene_start + 1,
            'transcript_count': len(gene_transcripts),
            'transcripts': transcript_ids,
            'total_exons': total_exons,
            'avg_exons_per_transcript': total_exons / len(gene_transcripts),
            'splice_sites': {
                'donor_count': len(all_donors),
                'acceptor_count': len(all_acceptors),
                'total_splice_sites': len(all_donors) + len(all_acceptors),
                'unique_positions': len(all_donors.union(all_acceptors))
            }
        }

