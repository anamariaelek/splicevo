"""Split genome data into train/test sets based on chromosomes and orthology.

This script reads data from individual genome directories (created by data_load.py)
and combines them into train/test splits based on chromosome assignment and orthology.
"""

import time
import os
import numpy as np
from datetime import datetime
import json
import argparse
import sys
import pandas as pd
from pathlib import Path

parser = argparse.ArgumentParser(description="Split loaded data into train/test sets")
parser.add_argument("--input_dir", type=str, required=True, 
                    help="Base directory containing genome subdirectories")
parser.add_argument("--output_dir", type=str, required=True, 
                    help="Directory to save split results")
parser.add_argument("--orthology_file", type=str, required=True, 
                    help="Path to orthology file (TSV format)")
parser.add_argument("--pov_genome", type=str, default="human_GRCh37", 
                    help="Point-of-view genome for chromosome splitting")
parser.add_argument("--test_chromosomes", type=int, nargs='+', default=[20], 
                    help="Test chromosomes for POV genome (as integers)")
parser.add_argument("--n_cpus", type=int, default=2, help="Number of CPU cores to use")
parser.add_argument("--quiet", action='store_true', help="Suppress console output")
args = parser.parse_args()

input_dir = args.input_dir
output_dir = args.output_dir
pov_genome = args.pov_genome
test_chromosomes = set(str(c) for c in args.test_chromosomes)

# Start timing
script_start_time = time.time()

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Setup logging
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_file = os.path.join(output_dir, f'data_split_{timestamp}.txt')

if args.quiet:
    log_file_handle = open(log_file, 'a', buffering=1)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = log_file_handle
    sys.stderr = log_file_handle
    
    def log_print(msg):
        print(msg, flush=True) 
else:
    def log_print(msg):
        print(msg)
        with open(log_file, 'a') as f:
            f.write(msg + '\n')

def format_time(seconds):
    """Format seconds as hours, minutes, seconds."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours}h {minutes}m {secs}s"

log_print(f"Train/test splitting started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log_print(f"Input directory: {input_dir}")
log_print(f"Output directory: {output_dir}")
log_print(f"POV genome: {pov_genome}")
log_print(f"Test chromosomes: {sorted(test_chromosomes)}")
log_print("=" * 60)

# Step 1: Discover available genomes
log_print("\nStep 1: Discovering available genomes...")
step1_start = time.time()

genome_dirs = []
for item in os.listdir(input_dir):
    item_path = os.path.join(input_dir, item)
    if os.path.isdir(item_path):
        summary_file = os.path.join(item_path, 'summary.json')
        if os.path.exists(summary_file):
            genome_dirs.append(item)

log_print(f"  Found {len(genome_dirs)} genome directories:")
for gid in sorted(genome_dirs):
    log_print(f"    - {gid}")

step1_time = time.time() - step1_start
log_print(f"✓ Discovery completed in {step1_time:.2f} seconds\n")

# Step 2: Load orthology file
log_print("Step 2: Loading orthology file...")
step2_start = time.time()

# Read the orthology file - it's a TSV with columns: ortholog_group, genome_id, gene_id
orthology_df = pd.read_csv(args.orthology_file, sep='\t', dtype={'ortholog_group': str, 'genome_id': str, 'gene_id': str})

log_print(f"  Orthology file shape: {orthology_df.shape}")
log_print(f"  Columns: {list(orthology_df.columns)}")

# Create mappings
# gene_id -> ortholog_group_id
gene_to_group = {}
for _, row in orthology_df.iterrows():
    gene_id = row['gene_id']
    group_id = row['ortholog_group']
    gene_to_group[gene_id] = group_id

# genome_id -> set of gene_ids in that genome
genome_genes = {}
for _, row in orthology_df.iterrows():
    genome_id = row['genome_id']
    gene_id = row['gene_id']
    if genome_id not in genome_genes:
        genome_genes[genome_id] = set()
    genome_genes[genome_id].add(gene_id)

log_print(f"  Loaded {len(gene_to_group)} gene-to-group mappings")
n_groups = len(set(gene_to_group.values()))
log_print(f"  Total ortholog groups: {n_groups}")
log_print(f"  Genomes in orthology file: {sorted(genome_genes.keys())}")

step2_time = time.time() - step2_start
log_print(f"✓ Orthology loaded in {step2_time:.2f} seconds\n")

# Step 3: Determine train/test split based on POV genome chromosomes
log_print("Step 3: Determining train/test split...")
step3_start = time.time()

# Load POV genome metadata to get chromosome info
pov_genome_dir = os.path.join(input_dir, pov_genome)
if not os.path.exists(pov_genome_dir):
    log_print(f"ERROR: POV genome directory not found: {pov_genome_dir}")
    sys.exit(1)

pov_metadata = pd.read_csv(os.path.join(pov_genome_dir, 'metadata.csv'))
# Convert chromosome to string for consistent comparison
pov_metadata['chromosome'] = pov_metadata['chromosome'].astype(str)

log_print(f"  Loaded POV genome metadata: {len(pov_metadata)} entries")

# Identify test and train genes from POV genome
test_genes_pov = set()
train_genes_pov = set()

for _, row in pov_metadata.iterrows():
    gene_id = row['gene_id']
    chrom = str(row['chromosome'])
    
    if chrom in test_chromosomes:
        test_genes_pov.add(gene_id)
    else:
        train_genes_pov.add(gene_id)

log_print(f"  POV genome test genes (chrom {sorted(test_chromosomes)}): {len(test_genes_pov)}")
log_print(f"  POV genome train genes: {len(train_genes_pov)}")

# Map POV genes to ortholog groups
test_orthologs = set()

n_test_mapped = 0

for gene_id in test_genes_pov:
    group_id = gene_to_group.get(gene_id)
    if group_id:
        test_orthologs.add(group_id)
        n_test_mapped += 1

log_print(f"  Mapped POV genes to orthology: {n_test_mapped}/{len(test_genes_pov)}")
log_print(f"  Test ortholog groups: {len(test_orthologs)}")

# Create gene-to-split mapping for all genomes
# Strategy: By default, all genes go to train
# If a gene's ortholog group contains ANY test gene from POV, the entire group goes to test
gene_split_mapping = {}  # (genome_id, gene_id) -> 'train' or 'test'
total_assignments = 0

for gene_id, group_id in gene_to_group.items():
    # Determine split: test if in test group, otherwise train
    split = 'test' if group_id in test_orthologs else 'train'
    
    # Find which genome(s) this gene belongs to
    for genome_id in genome_dirs:
        if gene_id in genome_genes.get(genome_id, set()):
            gene_split_mapping[(genome_id, gene_id)] = split
            total_assignments += 1

test_assignments = sum(1 for v in gene_split_mapping.values() if v == 'test')
train_assignments = sum(1 for v in gene_split_mapping.values() if v == 'train')

log_print(f"  Total gene assignments across all genomes: {total_assignments}")
log_print(f"    Train: {train_assignments}")
log_print(f"    Test: {test_assignments}")

step3_time = time.time() - step3_start
log_print(f"✓ Split determined in {step3_time:.2f} seconds\n")

# Step 4: Load and split data from each genome
log_print("Step 4: Loading and splitting genome data...")
step4_start = time.time()

# Initialize lists for collecting data
train_sequences_list = []
train_labels_list = []
train_metadata_list = []

test_sequences_list = []
test_labels_list = []
test_metadata_list = []

train_usage_alpha_list = []
train_usage_beta_list = []
train_usage_sse_list = []

test_usage_alpha_list = []
test_usage_beta_list = []
test_usage_sse_list = []

total_train_seqs = 0
total_test_seqs = 0

for genome_id in sorted(genome_dirs):
    log_print(f"\n  Processing {genome_id}...")
    genome_dir = os.path.join(input_dir, genome_id)
    
    # Load summary
    with open(os.path.join(genome_dir, 'summary.json'), 'r') as f:
        summary = json.load(f)
    
    # Load metadata
    metadata = pd.read_csv(os.path.join(genome_dir, 'metadata.csv'))
    log_print(f"    Loaded metadata: {len(metadata)} entries")
    
    # Determine which sequences belong to train vs test
    train_indices = []
    test_indices = []
    
    for idx, row in metadata.iterrows():
        gene_id = row['gene_id']
        split = gene_split_mapping.get((genome_id, gene_id), None)
        
        if split == 'train':
            train_indices.append(idx)
        elif split == 'test':
            test_indices.append(idx)
    
    log_print(f"    Train sequences: {len(train_indices)}")
    log_print(f"    Test sequences: {len(test_indices)}")
    
    if len(train_indices) == 0 and len(test_indices) == 0:
        log_print(f"    Skipping {genome_id} - no sequences assigned")
        continue
    
    # Load sequences and labels
    seq_shape = tuple(summary['sequence_shape'])
    label_shape = tuple(summary['label_shape'])
    seq_dtype = summary['dtypes']['sequences']
    label_dtype = summary['dtypes']['labels']
    
    sequences = np.memmap(os.path.join(genome_dir, 'sequences.mmap'), 
                         dtype=seq_dtype, mode='r', shape=seq_shape)
    labels = np.memmap(os.path.join(genome_dir, 'labels.mmap'), 
                      dtype=label_dtype, mode='r', shape=label_shape)
    
    # Load usage arrays if they exist
    usage_alpha = None
    usage_beta = None
    usage_sse = None
    
    usage_alpha_path = os.path.join(genome_dir, 'usage_alpha.mmap')
    if os.path.exists(usage_alpha_path):
        usage_alpha_shape = tuple(summary['sequence_shape'][:1])  # Same first dimension as sequences
        usage_alpha_dtype = summary['dtypes'].get('usage_alpha', 'float32')
        usage_alpha = np.memmap(usage_alpha_path, dtype=usage_alpha_dtype, mode='r', shape=usage_alpha_shape)
    
    usage_beta_path = os.path.join(genome_dir, 'usage_beta.mmap')
    if os.path.exists(usage_beta_path):
        usage_beta_shape = tuple(summary['sequence_shape'][:1])
        usage_beta_dtype = summary['dtypes'].get('usage_beta', 'float32')
        usage_beta = np.memmap(usage_beta_path, dtype=usage_beta_dtype, mode='r', shape=usage_beta_shape)
    
    usage_sse_path = os.path.join(genome_dir, 'usage_sse.mmap')
    if os.path.exists(usage_sse_path):
        usage_sse_shape = tuple(summary['sequence_shape'][:1])
        usage_sse_dtype = summary['dtypes'].get('usage_sse', 'float32')
        usage_sse = np.memmap(usage_sse_path, dtype=usage_sse_dtype, mode='r', shape=usage_sse_shape)
    
    # Extract and save train sequences
    if train_indices:
        train_seqs = sequences[train_indices]
        train_lbls = labels[train_indices]
        train_sequences_list.append(train_seqs)
        train_labels_list.append(train_lbls)
        total_train_seqs += len(train_indices)
        
        if usage_alpha is not None:
            train_usage_alpha_list.append(usage_alpha[train_indices])
        if usage_beta is not None:
            train_usage_beta_list.append(usage_beta[train_indices])
        if usage_sse is not None:
            train_usage_sse_list.append(usage_sse[train_indices])
        
        # Add metadata entries
        for idx in train_indices:
            seq_metadata = metadata.iloc[idx].to_dict()
            train_metadata_list.append(seq_metadata)
    
    # Extract and save test sequences
    if test_indices:
        test_seqs = sequences[test_indices]
        test_lbls = labels[test_indices]
        test_sequences_list.append(test_seqs)
        test_labels_list.append(test_lbls)
        total_test_seqs += len(test_indices)
        
        if usage_alpha is not None:
            test_usage_alpha_list.append(usage_alpha[test_indices])
        if usage_beta is not None:
            test_usage_beta_list.append(usage_beta[test_indices])
        if usage_sse is not None:
            test_usage_sse_list.append(usage_sse[test_indices])
        
        # Add metadata entries
        for idx in test_indices:
            seq_metadata = metadata.iloc[idx].to_dict()
            test_metadata_list.append(seq_metadata)

log_print(f"\n  Total train sequences collected: {total_train_seqs}")
log_print(f"  Total test sequences collected: {total_test_seqs}")

step4_time = time.time() - step4_start
log_print(f"✓ Data loaded and split in {step4_time:.2f} seconds\n")

# Step 5: Combine and save train/test data
log_print("Step 5: Combining and saving train/test data...")
step5_start = time.time()

# Combine train data
log_print("  Combining train data...")
train_sequences = np.concatenate(train_sequences_list, axis=0) if train_sequences_list else np.array([])
train_labels = np.concatenate(train_labels_list, axis=0) if train_labels_list else np.array([])
train_usage_alpha = np.concatenate(train_usage_alpha_list, axis=0) if train_usage_alpha_list else None
train_usage_beta = np.concatenate(train_usage_beta_list, axis=0) if train_usage_beta_list else None
train_usage_sse = np.concatenate(train_usage_sse_list, axis=0) if train_usage_sse_list else None
train_metadata_df = pd.DataFrame(train_metadata_list) if train_metadata_list else pd.DataFrame()

log_print(f"    Train sequences shape: {train_sequences.shape}")
log_print(f"    Train labels shape: {train_labels.shape}")
if train_usage_alpha is not None:
    log_print(f"    Train usage_alpha shape: {train_usage_alpha.shape}")
if train_usage_beta is not None:
    log_print(f"    Train usage_beta shape: {train_usage_beta.shape}")
if train_usage_sse is not None:
    log_print(f"    Train usage_sse shape: {train_usage_sse.shape}")

# Combine test data
log_print("  Combining test data...")
test_sequences = np.concatenate(test_sequences_list, axis=0) if test_sequences_list else np.array([])
test_labels = np.concatenate(test_labels_list, axis=0) if test_labels_list else np.array([])
test_usage_alpha = np.concatenate(test_usage_alpha_list, axis=0) if test_usage_alpha_list else None
test_usage_beta = np.concatenate(test_usage_beta_list, axis=0) if test_usage_beta_list else None
test_usage_sse = np.concatenate(test_usage_sse_list, axis=0) if test_usage_sse_list else None
test_metadata_df = pd.DataFrame(test_metadata_list) if test_metadata_list else pd.DataFrame()

log_print(f"    Test sequences shape: {test_sequences.shape}")
log_print(f"    Test labels shape: {test_labels.shape}")
if test_usage_alpha is not None:
    log_print(f"    Test usage_alpha shape: {test_usage_alpha.shape}")
if test_usage_beta is not None:
    log_print(f"    Test usage_beta shape: {test_usage_beta.shape}")
if test_usage_sse is not None:
    log_print(f"    Test usage_sse shape: {test_usage_sse.shape}")

# Save train data
if len(train_sequences) > 0:
    train_dir = os.path.join(output_dir, 'train')
    os.makedirs(train_dir, exist_ok=True)
    
    log_print(f"  Saving train data to {train_dir}...")
    train_seq_mmap = np.memmap(os.path.join(train_dir, 'sequences.mmap'), 
                               dtype=train_sequences.dtype, mode='w+', 
                               shape=train_sequences.shape)
    train_seq_mmap[:] = train_sequences
    del train_seq_mmap
    
    train_lbl_mmap = np.memmap(os.path.join(train_dir, 'labels.mmap'), 
                               dtype=train_labels.dtype, mode='w+', 
                               shape=train_labels.shape)
    train_lbl_mmap[:] = train_labels
    del train_lbl_mmap
    
    if train_usage_alpha is not None:
        train_alpha_mmap = np.memmap(os.path.join(train_dir, 'usage_alpha.mmap'), 
                                     dtype=train_usage_alpha.dtype, mode='w+', 
                                     shape=train_usage_alpha.shape)
        train_alpha_mmap[:] = train_usage_alpha
        del train_alpha_mmap
    
    if train_usage_beta is not None:
        train_beta_mmap = np.memmap(os.path.join(train_dir, 'usage_beta.mmap'), 
                                    dtype=train_usage_beta.dtype, mode='w+', 
                                    shape=train_usage_beta.shape)
        train_beta_mmap[:] = train_usage_beta
        del train_beta_mmap
    
    if train_usage_sse is not None:
        train_sse_mmap = np.memmap(os.path.join(train_dir, 'usage_sse.mmap'), 
                                   dtype=train_usage_sse.dtype, mode='w+', 
                                   shape=train_usage_sse.shape)
        train_sse_mmap[:] = train_usage_sse
        del train_sse_mmap
    
    train_metadata_df.to_csv(os.path.join(train_dir, 'metadata.csv'), index=False)

# Save test data
if len(test_sequences) > 0:
    test_dir = os.path.join(output_dir, 'test')
    os.makedirs(test_dir, exist_ok=True)
    
    log_print(f"  Saving test data to {test_dir}...")
    test_seq_mmap = np.memmap(os.path.join(test_dir, 'sequences.mmap'), 
                              dtype=test_sequences.dtype, mode='w+', 
                              shape=test_sequences.shape)
    test_seq_mmap[:] = test_sequences
    del test_seq_mmap
    
    test_lbl_mmap = np.memmap(os.path.join(test_dir, 'labels.mmap'), 
                              dtype=test_labels.dtype, mode='w+', 
                              shape=test_labels.shape)
    test_lbl_mmap[:] = test_labels
    del test_lbl_mmap
    
    if test_usage_alpha is not None:
        test_alpha_mmap = np.memmap(os.path.join(test_dir, 'usage_alpha.mmap'), 
                                    dtype=test_usage_alpha.dtype, mode='w+', 
                                    shape=test_usage_alpha.shape)
        test_alpha_mmap[:] = test_usage_alpha
        del test_alpha_mmap
    
    if test_usage_beta is not None:
        test_beta_mmap = np.memmap(os.path.join(test_dir, 'usage_beta.mmap'), 
                                   dtype=test_usage_beta.dtype, mode='w+', 
                                   shape=test_usage_beta.shape)
        test_beta_mmap[:] = test_usage_beta
        del test_beta_mmap
    
    if test_usage_sse is not None:
        test_sse_mmap = np.memmap(os.path.join(test_dir, 'usage_sse.mmap'), 
                                  dtype=test_usage_sse.dtype, mode='w+', 
                                  shape=test_usage_sse.shape)
        test_sse_mmap[:] = test_usage_sse
        del test_sse_mmap
    
    test_metadata_df.to_csv(os.path.join(test_dir, 'metadata.csv'), index=False)

# Save summary
summary_data = {
    'pov_genome': pov_genome,
    'test_chromosomes': sorted(list(test_chromosomes)),
    'n_genomes': len(genome_dirs),
    'genomes': sorted(genome_dirs),
    'train': {
        'n_sequences': int(len(train_sequences)),
        'shape': list(train_sequences.shape),
    },
    'test': {
        'n_sequences': int(len(test_sequences)),
        'shape': list(test_sequences.shape),
    },
    'created_at': datetime.now().isoformat()
}

with open(os.path.join(output_dir, 'summary.json'), 'w') as f:
    json.dump(summary_data, f, indent=2)

step5_time = time.time() - step5_start
log_print(f"✓ Data saved in {step5_time:.2f} seconds\n")

# Final summary
total_time = time.time() - script_start_time
log_print("=" * 60)
log_print("SUMMARY")
log_print("=" * 60)
log_print(f"Total genomes processed: {len(genome_dirs)}")
log_print(f"Train sequences: {len(train_sequences)}")
log_print(f"Test sequences: {len(test_sequences)}")
log_print(f"Output directory: {output_dir}")
log_print(f"Total time: {format_time(total_time)}")
log_print("=" * 60)

log_print(f"\nData splitting completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if args.quiet:
    sys.stdout = original_stdout
    sys.stderr = original_stderr
    log_file_handle.close()
