from pathlib import Path
import shutil
import gzip
from grelu.io.genome import CustomGenome
from tangermeme.utils import one_hot_encode

from .gene_annotation import Transcript

def load_genome(fasta_path: str | Path) -> CustomGenome:
	"""
	Load a genome from a FASTA file.
	"""
	# Ensure fasta_path is a str
	if isinstance(fasta_path, Path):
		fasta_path = str(fasta_path)
	return CustomGenome(fasta_path)

def encode_transcript(transcript: Transcript, genome: CustomGenome) -> tuple:
	"""
	Given a Transcript object and a genome, return the one-hot encoded sequence and a parallel vector
	with splice site annotations (0=not a splice site, 1=donor, 2=acceptor).
	"""
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


