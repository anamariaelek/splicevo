import pandas as pd
from collections import defaultdict
import gzip
from typing import Dict, List, Tuple, Set
import re

class Transcript:
    def __init__(self, transcript_id: str, gene_id: str, strand: str, exons: pd.DataFrame):
        self.transcript_id = transcript_id
        self.gene_id = gene_id
        self.strand = strand
        self.exons = exons
        self.introns = self.calculate_introns()
        self.splice_donor_sites, self.splice_acceptor_sites = self.calculate_splice_sites()

    def calculate_introns(self) -> List[Tuple[int, int]]:
        """Calculate introns from exons"""
        introns = []
        for i in range(len(self.exons) - 1):
            intron_start = self.exons.iloc[i]['end']+1
            intron_end = self.exons.iloc[i + 1]['start']-1
            if intron_start < intron_end:
                introns.append((intron_start, intron_end))
        return introns

    def calculate_splice_sites(self) -> Tuple[Set[int], Set[int]]:
        """Calculate splice sites from introns"""
        splice_donor_sites = set()
        splice_acceptor_sites = set()

        # Skip introns shorter than 4 bp
        introns_filt = []
        for intron in self.introns:
            if intron[1] - intron[0] >= 4:
                introns_filt.append(intron)

        # Skip single-exon transcripts (no introns)
        if len(introns_filt) == 0:
            return splice_donor_sites, splice_acceptor_sites
        
        # Positive strand: donor = intron start, acceptor = intron end
        # Negative strand: donor = intron end, acceptor = intron start
        for intron in introns_filt:
            if self.strand == "+":
                splice_donor_sites.add(intron[0])
                splice_acceptor_sites.add(intron[1])
            else:
                splice_donor_sites.add(intron[1])
                splice_acceptor_sites.add(intron[0])

        return splice_donor_sites, splice_acceptor_sites

class GTFProcessor:
    def __init__(self, gtf_file: str):
        self.gtf_file = gtf_file
    
    def parse_gtf_attributes(self, attr_string: str) -> Dict[str, str]:
        """Parse GTF attribute string into dictionary"""
        attrs = {}
        for match in re.finditer(r'(\w+)\s+"([^"]+)"', attr_string):
            attrs[match.group(1)] = match.group(2)
        return attrs

    def load_gtf(self, chromosomes: List[str]) -> pd.DataFrame:
        """Load and parse GTF file"""
        print("Loading GTF file...")
        
        # Handle gzipped files
        opener = gzip.open if self.gtf_file.endswith('.gz') else open
        
        records = []
        with opener(self.gtf_file, 'rt') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                    
                fields = line.strip().split('\t')
                if len(fields) != 9:
                    continue
                
                # FIlter chromosomes
                if chromosomes is not None and fields[0] not in chromosomes:
                    continue

                # Parse attributes
                attrs = self.parse_gtf_attributes(fields[8])
                
                record = {
                    'chrom': fields[0],
                    'source': fields[1], 
                    'feature': fields[2],
                    'start': int(fields[3]),
                    'end': int(fields[4]),
                    'score': fields[5],
                    'strand': fields[6],
                    'frame': fields[7],
                    **attrs
                }
                records.append(record)
        
        df = pd.DataFrame(records)
        print(f"Loaded {len(df)} GTF records")
        return df
    
    def filter_exons(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter for exons"""
        filtered = df
        # Filter by 'feature'
        if 'feature' in filtered.columns:
            filtered = filtered[filtered['feature'] == 'exon']
        return filtered
    
    def filter_high_confidence(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter for high-confidence protein-coding transcripts"""
        filtered = df
        # Filter by 'gene_type' if present
        if 'gene_type' in filtered.columns:
            filtered = filtered[filtered['gene_biotype'] == 'protein_coding']
            print(f"Filtered to {len(filtered)} records from protein-coding genes")
        # Filter by 'transcript_type' if present
        if 'transcript_biotype' in filtered.columns:
            filtered = filtered[filtered['transcript_biotype'] == 'protein_coding']
            print(f"Filtered to {len(filtered)} records from protein-coding transcripts")
        # Additional quality filters if available
        if 'transcript_support_level' in filtered.columns:
            # TSL 1-2 are high confidence
            filtered = filtered[
                filtered['transcript_support_level'].isin(['1', '2', 'NA'])
            ]
            print(f"Filtered to {len(filtered)} records from high-support transcripts")
        return filtered

    def get_transcripts(self, df: pd.DataFrame, chromosomes: List[str] = None) -> List[Transcript]:
        """Get transcripts"""
        if chromosomes is not None:
            df = df[df['chrom'].isin(chromosomes)]
        transcripts = []
        for transcript_id in df['transcript_id'].unique():
            records = df[df['transcript_id'] == transcript_id].reset_index(drop=True).sort_values(by='start')
            gene_id = records['gene_id'].iloc[0]
            strand = records['strand'].iloc[0]
            exons = self.filter_exons(records)
            if exons is None or len(exons) == 0:
                continue
            transcript = Transcript(transcript_id, gene_id, strand, exons)
            transcripts.append(transcript)
        return transcripts

    def get_splice_sites(self, transcripts: List[Transcript]) -> Dict[str, Dict[str, list]]:
        """Get splice donor and acceptor sites from transcripts, grouped by chromosome, preserving order and removing duplicates."""
        print("Getting splice sites...")
        chrom_splice_sites = {}
        for transcript in transcripts:
            chrom = transcript.exons.iloc[0]['chrom']
            if chrom not in chrom_splice_sites:
                chrom_splice_sites[chrom] = {'splice_donor': [], 'splice_acceptor': []}
            # Extend lists in the order they appear in each transcript
            chrom_splice_sites[chrom]['splice_donor'].extend(sorted(transcript.splice_donor_sites))
            chrom_splice_sites[chrom]['splice_acceptor'].extend(sorted(transcript.splice_acceptor_sites))
        # Remove duplicates while preserving order
        for chrom in chrom_splice_sites:
            seen_donor = set()
            donor_no_dupes = []
            for x in chrom_splice_sites[chrom]['splice_donor']:
                if x not in seen_donor:
                    donor_no_dupes.append(x)
                    seen_donor.add(x)
            chrom_splice_sites[chrom]['splice_donor'] = donor_no_dupes

            seen_acceptor = set()
            acceptor_no_dupes = []
            for x in chrom_splice_sites[chrom]['splice_acceptor']:
                if x not in seen_acceptor:
                    acceptor_no_dupes.append(x)
                    seen_acceptor.add(x)
            chrom_splice_sites[chrom]['splice_acceptor'] = acceptor_no_dupes
        return chrom_splice_sites

    def process_gtf(self, chromosomes: List[str] = None) -> pd.DataFrame:
        """Process GTF file and return splicing events"""
        gtf_df = self.load_gtf(chromosomes=chromosomes)
        filtered_df = self.filter_high_confidence(gtf_df)
        transcripts = self.get_transcripts(filtered_df)
        return transcripts
