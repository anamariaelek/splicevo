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
import psutil
import gc

parser = argparse.ArgumentParser(description="Split loaded data into train/test sets")
parser.add_argument("--input_dir", type=str, required=True, 
                    help="Base directory containing genome subdirectories")
parser.add_argument("--genome_ids", type=str, nargs='*', default=None,
                    help="Optional list of genome IDs to include (default: all found)")
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
genome_ids = args.genome_ids
output_dir = args.output_dir
pov_genome = args.pov_genome
test_chromosomes = set(str(c) for c in args.test_chromosomes)
orthology_file = args.orthology_file

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

def get_memory_usage():
    """Get current memory usage in GB."""
    process = psutil.Process()
    mem_info = process.memory_info()
    return mem_info.rss / (1024**3)  # Convert to GB

log_print(f"Train/test splitting started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log_print(f"Input directory: {input_dir}")
log_print(f"Output directory: {output_dir}")
log_print(f"Genomes to include: {genome_ids if genome_ids else 'All found'}")
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
        meta_file = os.path.join(item_path, 'metadata.json')
        if os.path.exists(meta_file):
            if genome_ids is None or item in genome_ids:
                genome_dirs.append(item)

log_print(f"  Found {len(genome_dirs)} genome directories:")
for gid in sorted(genome_dirs):
    log_print(f"    - {gid}")

step1_time = time.time() - step1_start
mem_after_step1 = get_memory_usage()
log_print(f"  Discovery completed in {step1_time:.2f} seconds")
log_print(f"  Memory usage: {mem_after_step1:.2f} GB\n")

# Step 2: Load orthology file
log_print("Step 2: Loading orthology file...")
step2_start = time.time()

# Read the orthology file - it's a TSV with columns: ortholog_group, genome_id, gene_id
orthology_df = pd.read_csv(orthology_file, sep='\t', dtype={'ortholog_group': str, 'genome_id': str, 'gene_id': str})

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
mem_after_step2 = get_memory_usage()
log_print(f"  Orthology loaded in {step2_time:.2f} seconds")
log_print(f"  Memory usage: {mem_after_step2:.2f} GB\n")

# Step 3: Determine train/test split based on POV genome chromosomes
log_print("Step 3: Determining train/test split...")
step3_start = time.time()

# Load POV genome metadata to get chromosome info
pov_genome_dir = os.path.join(input_dir, pov_genome)
if not os.path.exists(pov_genome_dir):
    log_print(f"ERROR: POV genome directory not found: {pov_genome_dir}")
    sys.exit(1)

pov_metadata = pd.read_csv(os.path.join(pov_genome_dir, 'metadata.csv'), dtype={'genome_id': str, 'gene_id': str, 'chromosome': str, 'window_start': int, 'window_end': int})
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

log_print(f"  Mapped test POV genes to orthology: {n_test_mapped}/{len(test_genes_pov)}")
log_print(f"  Test ortholog groups: {len(test_orthologs)}")

# Create gene-to-split mapping for all genomes
# Strategy: By default, all genes go to train
# If a gene's ortholog group contains ANY test gene from POV, the entire group goes to test
# If any gene from the test set overlaps with a gene in train set, reassign it to train set
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

log_print(f"  Initial gene assignments across all genomes: {total_assignments}")
log_print(f"    Train: {train_assignments}")
log_print(f"    Test: {test_assignments}")

# Load and cache genome metadata for overlap detection
log_print(f"  Loading genome metadata for overlap detection...")
genome_metadata_cache = {}
for genome_id in genome_dirs:
    genome_dir = os.path.join(input_dir, genome_id)
    genome_metadata_cache[genome_id] = pd.read_csv(
        os.path.join(genome_dir, 'metadata.csv'), 
        dtype={'genome_id': str, 'gene_id': str, 'chromosome': str, 'window_start': int, 'window_end': int}
    )
    genome_metadata_cache[genome_id]['chromosome'] = genome_metadata_cache[genome_id]['chromosome'].astype(str)

# Pre-index train genes by genome and chromosome for fast lookup
train_genes_by_genome_chrom = {}
for (genome_id, gene_id), split in gene_split_mapping.items():
    if split == 'train':
        if genome_id not in train_genes_by_genome_chrom:
            train_genes_by_genome_chrom[genome_id] = {}
        
        metadata_row = genome_metadata_cache[genome_id][
            genome_metadata_cache[genome_id]['gene_id'] == gene_id
        ]
        if not metadata_row.empty:
            chrom = str(metadata_row['chromosome'].values[0])
            if chrom not in train_genes_by_genome_chrom[genome_id]:
                train_genes_by_genome_chrom[genome_id][chrom] = []
            train_genes_by_genome_chrom[genome_id][chrom].append({
                'gene_id': gene_id,
                'start': metadata_row['window_start'].values[0],
                'end': metadata_row['window_end'].values[0]
            })

# Check for overlaps using pre-indexed train genes
reassigned_count = 0
for (genome_id, gene_id), split in list(gene_split_mapping.items()):
    if split == 'test':
        test_gene_row = genome_metadata_cache[genome_id][
            genome_metadata_cache[genome_id]['gene_id'] == gene_id
        ]
        if test_gene_row.empty:
            continue
        
        test_start = test_gene_row['window_start'].values[0]
        test_end = test_gene_row['window_end'].values[0]
        test_chrom = str(test_gene_row['chromosome'].values[0])
        
        # Fast lookup: only check train genes on same chromosome
        if genome_id in train_genes_by_genome_chrom and test_chrom in train_genes_by_genome_chrom[genome_id]:
            for train_gene in train_genes_by_genome_chrom[genome_id][test_chrom]:
                train_start = train_gene['start']
                train_end = train_gene['end']
                
                # Check for overlap
                if not (test_end < train_start or test_start > train_end):
                    gene_split_mapping[(genome_id, gene_id)] = 'train'
                    reassigned_count += 1
                    break

# Summarize assignments
test_assignments = sum(1 for v in gene_split_mapping.values() if v == 'test')
train_assignments = sum(1 for v in gene_split_mapping.values() if v == 'train')

log_print(f"  After reassignment of overlapping genes: {reassigned_count} genes moved to train")
log_print(f"    Train: {train_assignments}")
log_print(f"    Test: {test_assignments}")

step3_time = time.time() - step3_start

# Free memory from genome metadata cache (no longer needed)
del genome_metadata_cache, train_genes_by_genome_chrom
gc.collect()

mem_after_step3 = get_memory_usage()
log_print(f"  Split determined in {step3_time:.2f} seconds")
log_print(f"  Memory usage: {mem_after_step3:.2f} GB\n")

# Step 4: Identify common metadata columns and usage conditions
log_print("Step 4: Identifying common columns and conditions...")
step4_start = time.time()

# Scan all genomes to find common metadata columns
all_metadata_columns = []
for genome_id in sorted(genome_dirs):
    genome_dir = os.path.join(input_dir, genome_id)
    metadata = pd.read_csv(os.path.join(genome_dir, 'metadata.csv'), dtype={'genome_id': str, 'chromosome': str})
    all_metadata_columns.append(set(metadata.columns))

common_metadata_columns = sorted(list(set.intersection(*all_metadata_columns))) if all_metadata_columns else []
log_print(f"  Common metadata columns: {common_metadata_columns}")

# Identify all usage conditions across genomes
all_usage_conditions = set()
for genome_id in sorted(genome_dirs):
    genome_dir = os.path.join(input_dir, genome_id)
    with open(os.path.join(genome_dir, 'metadata.json'), 'r') as f:
        meta = json.load(f)
    usage_columns = meta.get('usage_conditions', [])
    all_usage_conditions.update(usage_columns)

common_usage_conditions = sorted(list(all_usage_conditions))
log_print(f"  Common usage conditions: {common_usage_conditions}")

step4_time = time.time() - step4_start
mem_after_step4 = get_memory_usage()
log_print(f"  Column identification completed in {step4_time:.2f} seconds")
log_print(f"  Memory usage: {mem_after_step4:.2f} GB\n")

# Step 5: Process genomes sequentially and write to memory-mapped files
log_print("Step 5: Processing genomes sequentially and writing to outputs...")
step5_start = time.time()

# Create output directories
train_dir = os.path.join(output_dir, 'train')
test_dir = os.path.join(output_dir, 'test')
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# First pass: count sequences to determine mmap shapes
train_count = 0
test_count = 0
for genome_id in sorted(genome_dirs):
    genome_dir = os.path.join(input_dir, genome_id)
    metadata = pd.read_csv(os.path.join(genome_dir, 'metadata.csv'), dtype={'genome_id': str, 'chromosome': str})
    
    for _, row in metadata.iterrows():
        gene_id = row['gene_id']
        split = gene_split_mapping.get((genome_id, gene_id), None)
        if split == 'train':
            train_count += 1
        elif split == 'test':
            test_count += 1

log_print(f"  Total train sequences: {train_count}")
log_print(f"  Total test sequences: {test_count}")

# Get shape information from first genome
first_genome_dir = os.path.join(input_dir, sorted(genome_dirs)[0])
with open(os.path.join(first_genome_dir, 'metadata.json'), 'r') as f:
    first_meta = json.load(f)

seq_dtype = first_meta['sequences_dtype']
label_dtype = first_meta['labels_dtype']
seq_window_size = first_meta['sequences_shape'][1]
label_window_size = first_meta['labels_shape'][1]

alpha_dtype = first_meta.get('alpha_dtype', 'float32')
beta_dtype = first_meta.get('beta_dtype', 'float32')
sse_dtype = first_meta.get('sse_dtype', 'float32')

n_usage_conditions = len(common_usage_conditions)

# Initialize memory-mapped output files
train_sequences_mmap = None
train_labels_mmap = None
train_species_ids_mmap = None
train_usage_alpha_mmap = None
train_usage_beta_mmap = None
train_usage_sse_mmap = None
train_condition_mask_mmap = None

test_sequences_mmap = None
test_labels_mmap = None
test_species_ids_mmap = None
test_usage_alpha_mmap = None
test_usage_beta_mmap = None
test_usage_sse_mmap = None
test_condition_mask_mmap = None

train_metadata_list = []
test_metadata_list = []

train_offset = 0
test_offset = 0

# Second pass: process each genome and write to mmap files
for genome_idx, genome_id in enumerate(sorted(genome_dirs)):
    log_print(f"\n  Processing genome {genome_idx}: {genome_id}...")
    genome_dir = os.path.join(input_dir, genome_id)
    
    # Load metadata
    with open(os.path.join(genome_dir, 'metadata.json'), 'r') as f:
        meta = json.load(f)
    
    metadata = pd.read_csv(os.path.join(genome_dir, 'metadata.csv'), dtype={'genome_id': str, 'chromosome': str})
    log_print(f"    Loaded metadata: {len(metadata)} entries")
    
    # Determine train/test split for this genome
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
    
    # Add metadata entries (always collect metadata)
    for idx in train_indices:
        seq_metadata = metadata.iloc[idx][common_metadata_columns].to_dict()
        seq_metadata['species_id'] = genome_idx
        train_metadata_list.append(seq_metadata)
    
    for idx in test_indices:
        seq_metadata = metadata.iloc[idx][common_metadata_columns].to_dict()
        seq_metadata['species_id'] = genome_idx
        test_metadata_list.append(seq_metadata)
    
    # Open memmap files without loading into memory (memory efficient)
    seq_shape = tuple(meta['sequences_shape'])
    labels_shape = tuple(meta['labels_shape'])
    
    log_print(f"    Opening sequences memmap (not loading into memory)...")
    sequences_mmap = np.memmap(os.path.join(genome_dir, 'sequences.mmap'), 
                              dtype=seq_dtype, mode='r', shape=seq_shape)
    
    log_print(f"    Opening labels memmap (not loading into memory)...")
    labels_mmap = np.memmap(os.path.join(genome_dir, 'labels.mmap'), 
                           dtype=label_dtype, mode='r', shape=labels_shape)
    
    # Open usage array memmaps if they exist
    usage_alpha_mmap = None
    usage_beta_mmap = None
    usage_sse_mmap = None
    genome_usage_conditions = meta.get('usage_conditions', [])
    
    usage_alpha_path = os.path.join(genome_dir, 'usage_alpha.mmap')
    if os.path.exists(usage_alpha_path):
        usage_alpha_shape = tuple(meta['alpha_shape'])
        log_print(f"    Opening usage_alpha memmap (not loading into memory)...")
        usage_alpha_mmap = np.memmap(usage_alpha_path, dtype=alpha_dtype, mode='r', shape=usage_alpha_shape)
    
    usage_beta_path = os.path.join(genome_dir, 'usage_beta.mmap')
    if os.path.exists(usage_beta_path):
        usage_beta_shape = tuple(meta['beta_shape'])
        log_print(f"    Opening usage_beta memmap (not loading into memory)...")
        usage_beta_mmap = np.memmap(usage_beta_path, dtype=beta_dtype, mode='r', shape=usage_beta_shape)
    
    usage_sse_path = os.path.join(genome_dir, 'usage_sse.mmap')
    if os.path.exists(usage_sse_path):
        usage_sse_shape = tuple(meta['sse_shape'])
        log_print(f"    Opening usage_sse memmap (not loading into memory)...")
        usage_sse_mmap = np.memmap(usage_sse_path, dtype=sse_dtype, mode='r', shape=usage_sse_shape)
    
    # Helper to reorder usage conditions
    def reorder_usage_to_common(usage_array, genome_conds, target_conds):
        """Reorder usage array columns to match target conditions."""
        if usage_array is None:
            return None
        n_seqs, window_size, _ = usage_array.shape
        reordered = np.zeros((n_seqs, window_size, len(target_conds)), dtype=usage_array.dtype)
        cond_to_idx = {cond: idx for idx, cond in enumerate(genome_conds)}
        for target_idx, target_cond in enumerate(target_conds):
            if target_cond in cond_to_idx:
                source_idx = cond_to_idx[target_cond]
                reordered[:, :, target_idx] = usage_array[:, :, source_idx]
        return reordered
    
    # Helper to build condition mask for genome
    def build_condition_mask(genome_conds, target_conds):
        """Build binary mask indicating which conditions are valid for this genome."""
        mask = np.zeros(len(target_conds), dtype=np.bool_)
        cond_to_idx = {cond: idx for idx, cond in enumerate(genome_conds)}
        for target_idx, target_cond in enumerate(target_conds):
            if target_cond in cond_to_idx:
                mask[target_idx] = True
        return mask
    
    # Process train sequences
    if train_indices:
        # Initialize mmap files on first write
        if train_sequences_mmap is None:
            log_print(f"    Initializing train mmap files...")
            train_sequences_mmap = np.memmap(os.path.join(train_dir, 'sequences.mmap'), 
                                            dtype=seq_dtype, mode='w+', 
                                            shape=(train_count, seq_window_size, 4))
            train_labels_mmap = np.memmap(os.path.join(train_dir, 'labels.mmap'), 
                                         dtype=label_dtype, mode='w+', 
                                         shape=(train_count, label_window_size))
            train_species_ids_mmap = np.memmap(os.path.join(train_dir, 'species_ids.mmap'), 
                                              dtype=np.int32, mode='w+', 
                                              shape=(train_count,))
            if usage_alpha is not None:
                train_usage_alpha_mmap = np.memmap(os.path.join(train_dir, 'usage_alpha.mmap'), 
                                                  dtype=alpha_dtype, mode='w+', 
                                                  shape=(train_count, label_window_size, n_usage_conditions))
            if usage_beta is not None:
                train_usage_beta_mmap = np.memmap(os.path.join(train_dir, 'usage_beta.mmap'), 
                                                 dtype=beta_dtype, mode='w+', 
                                                 shape=(train_count, label_window_size, n_usage_conditions))
            if usage_sse is not None:
                train_usage_sse_mmap = np.memmap(os.path.join(train_dir, 'usage_sse.mmap'), 
                                                dtype=sse_dtype, mode='w+', 
                                                shape=(train_count, label_window_size, n_usage_conditions))
            
            # Create condition mask array
            train_condition_mask_mmap = np.memmap(os.path.join(train_dir, 'condition_mask.mmap'),
                                                 dtype=np.bool_, mode='w+',
                                                 shape=(train_count, n_usage_conditions))
        
        # Write train data in batches to avoid memory spikes
        n_train = len(train_indices)
        batch_size = 1000  # Process 1000 sequences at a time
        
        log_print(f"    Writing {n_train} train sequences in batches of {batch_size}...")
        for batch_start in range(0, n_train, batch_size):
            batch_end = min(batch_start + batch_size, n_train)
            batch_indices = train_indices[batch_start:batch_end]
            batch_size_actual = len(batch_indices)
            
            # Copy batch directly from source memmap to dest memmap (memory efficient)
            train_sequences_mmap[train_offset:train_offset + batch_size_actual] = sequences_mmap[batch_indices]
            train_labels_mmap[train_offset:train_offset + batch_size_actual] = labels_mmap[batch_indices]
            train_species_ids_mmap[train_offset:train_offset + batch_size_actual] = genome_idx
            
            if usage_alpha_mmap is not None:
                train_alpha = reorder_usage_to_common(usage_alpha_mmap[batch_indices], 
                                                     genome_usage_conditions, 
                                                     common_usage_conditions)
                train_usage_alpha_mmap[train_offset:train_offset + batch_size_actual] = train_alpha
                del train_alpha
            
            if usage_beta_mmap is not None:
                train_beta = reorder_usage_to_common(usage_beta_mmap[batch_indices], 
                                                    genome_usage_conditions, 
                                                    common_usage_conditions)
                train_usage_beta_mmap[train_offset:train_offset + batch_size_actual] = train_beta
                del train_beta
            
            if usage_sse_mmap is not None:
                train_sse = reorder_usage_to_common(usage_sse_mmap[batch_indices], 
                                                   genome_usage_conditions, 
                                                   common_usage_conditions)
                train_usage_sse_mmap[train_offset:train_offset + batch_size_actual] = train_sse
                del train_sse
            
            # Write condition mask (same for all sequences from this genome)
            genome_mask = build_condition_mask(genome_usage_conditions, common_usage_conditions)
            train_condition_mask_mmap[train_offset:train_offset + batch_size_actual] = genome_mask
            
            train_offset += batch_size_actual
    
    # Process test sequences
    if test_indices:
        # Initialize mmap files on first write
        if test_sequences_mmap is None:
            log_print(f"    Initializing test mmap files...")
            test_sequences_mmap = np.memmap(os.path.join(test_dir, 'sequences.mmap'), 
                                           dtype=seq_dtype, mode='w+', 
                                           shape=(test_count, seq_window_size, 4))
            test_labels_mmap = np.memmap(os.path.join(test_dir, 'labels.mmap'), 
                                        dtype=label_dtype, mode='w+', 
                                        shape=(test_count, label_window_size))
            test_species_ids_mmap = np.memmap(os.path.join(test_dir, 'species_ids.mmap'), 
                                             dtype=np.int32, mode='w+', 
                                             shape=(test_count,))
            if usage_alpha is not None:
                test_usage_alpha_mmap = np.memmap(os.path.join(test_dir, 'usage_alpha.mmap'), 
                                                 dtype=alpha_dtype, mode='w+', 
                                                 shape=(test_count, label_window_size, n_usage_conditions))
            if usage_beta is not None:
                test_usage_beta_mmap = np.memmap(os.path.join(test_dir, 'usage_beta.mmap'), 
                                                dtype=beta_dtype, mode='w+', 
                                                shape=(test_count, label_window_size, n_usage_conditions))
            if usage_sse is not None:
                test_usage_sse_mmap = np.memmap(os.path.join(test_dir, 'usage_sse.mmap'), 
                                               dtype=sse_dtype, mode='w+', 
                                               shape=(test_count, label_window_size, n_usage_conditions))
            
            # Create condition mask array
            test_condition_mask_mmap = np.memmap(os.path.join(test_dir, 'condition_mask.mmap'),
                                                dtype=np.bool_, mode='w+',
                                                shape=(test_count, n_usage_conditions))
        
        # Write test data in batches to avoid memory spikes
        n_test = len(test_indices)
        batch_size = 1000  # Process 1000 sequences at a time
        
        log_print(f"    Writing {n_test} test sequences in batches of {batch_size}...")
        for batch_start in range(0, n_test, batch_size):
            batch_end = min(batch_start + batch_size, n_test)
            batch_indices = test_indices[batch_start:batch_end]
            batch_size_actual = len(batch_indices)
            
            # Copy batch directly from source memmap to dest memmap (memory efficient)
            test_sequences_mmap[test_offset:test_offset + batch_size_actual] = sequences_mmap[batch_indices]
            test_labels_mmap[test_offset:test_offset + batch_size_actual] = labels_mmap[batch_indices]
            test_species_ids_mmap[test_offset:test_offset + batch_size_actual] = genome_idx
            
            if usage_alpha_mmap is not None:
                test_alpha = reorder_usage_to_common(usage_alpha_mmap[batch_indices], 
                                                    genome_usage_conditions, 
                                                    common_usage_conditions)
                test_usage_alpha_mmap[test_offset:test_offset + batch_size_actual] = test_alpha
                del test_alpha
            
            if usage_beta_mmap is not None:
                test_beta = reorder_usage_to_common(usage_beta_mmap[batch_indices], 
                                                   genome_usage_conditions, 
                                                   common_usage_conditions)
                test_usage_beta_mmap[test_offset:test_offset + batch_size_actual] = test_beta
                del test_beta
            
            if usage_sse_mmap is not None:
                test_sse = reorder_usage_to_common(usage_sse_mmap[batch_indices], 
                                                  genome_usage_conditions, 
                                                  common_usage_conditions)
                test_usage_sse_mmap[test_offset:test_offset + batch_size_actual] = test_sse
                del test_sse
            
            # Write condition mask (same for all sequences from this genome)
            genome_mask = build_condition_mask(genome_usage_conditions, common_usage_conditions)
            test_condition_mask_mmap[test_offset:test_offset + batch_size_actual] = genome_mask
            
            test_offset += batch_size_actual
    
    log_print(f"    Genome {genome_id} processed")
    # Close memmap references to free file handles
    del sequences_mmap, labels_mmap, usage_alpha_mmap, usage_beta_mmap, usage_sse_mmap, metadata
    gc.collect()
    
    log_print(f"    Memory after processing: {get_memory_usage():.2f} GB")

# Flush and close mmap files
if train_sequences_mmap is not None:
    del train_sequences_mmap, train_labels_mmap, train_species_ids_mmap
    if train_usage_alpha_mmap is not None:
        del train_usage_alpha_mmap
    if train_usage_beta_mmap is not None:
        del train_usage_beta_mmap
    if train_usage_sse_mmap is not None:
        del train_usage_sse_mmap
    if train_condition_mask_mmap is not None:
        del train_condition_mask_mmap

if test_sequences_mmap is not None:
    del test_sequences_mmap, test_labels_mmap, test_species_ids_mmap
    if test_usage_alpha_mmap is not None:
        del test_usage_alpha_mmap
    if test_usage_beta_mmap is not None:
        del test_usage_beta_mmap
    if test_usage_sse_mmap is not None:
        del test_usage_sse_mmap
    if test_condition_mask_mmap is not None:
        del test_condition_mask_mmap

# Save metadata
log_print(f"\n  Saving metadata...")
if train_metadata_list:
    train_metadata_df = pd.DataFrame(train_metadata_list)
    train_metadata_df = train_metadata_df[common_metadata_columns + ['species_id']]
    train_metadata_df.to_csv(os.path.join(train_dir, 'metadata.csv'), index=False)
    log_print(f"    Train metadata: {len(train_metadata_df)} entries")

if test_metadata_list:
    test_metadata_df = pd.DataFrame(test_metadata_list)
    test_metadata_df = test_metadata_df[common_metadata_columns + ['species_id']]
    test_metadata_df.to_csv(os.path.join(test_dir, 'metadata.csv'), index=False)
    log_print(f"    Test metadata: {len(test_metadata_df)} entries")

# Helper function to create metadata.json
def create_split_metadata(output_split_dir, n_sequences, common_usage_conditions,
                         genome_dirs_list, input_dir, pov_genome):
    """Create metadata.json for a train/test split."""
    
    # Load window_size and context_size from POV genome config
    pov_summary_path = os.path.join(input_dir, pov_genome, 'metadata.json')
    with open(pov_summary_path, 'r') as f:
        pov_summary = json.load(f)
    
    window_size = pov_summary.get('window_size', 1000)
    context_size = pov_summary.get('context_size', 450)
    
    # Build consolidated metadata
    metadata = {
        'sequences_shape': [n_sequences, seq_window_size, 4],
        'labels_shape': [n_sequences, label_window_size],
        'sequences_dtype': seq_dtype,
        'labels_dtype': label_dtype,
        'usage_conditions': common_usage_conditions,
        'window_size': window_size,
        'context_size': context_size,
        'species_ids_dtype': 'int32',
        'species_ids_shape': [n_sequences],
    }
    
    # Add usage array info if present (check for file existence)
    if os.path.exists(os.path.join(output_split_dir, 'usage_alpha.mmap')):
        metadata['alpha_shape'] = [n_sequences, label_window_size, len(common_usage_conditions)]
        metadata['alpha_dtype'] = alpha_dtype
    if os.path.exists(os.path.join(output_split_dir, 'usage_beta.mmap')):
        metadata['beta_shape'] = [n_sequences, label_window_size, len(common_usage_conditions)]
        metadata['beta_dtype'] = beta_dtype
    if os.path.exists(os.path.join(output_split_dir, 'usage_sse.mmap')):
        metadata['sse_shape'] = [n_sequences, label_window_size, len(common_usage_conditions)]
        metadata['sse_dtype'] = sse_dtype
    if os.path.exists(os.path.join(output_split_dir, 'condition_mask.mmap')):
        metadata['condition_mask_shape'] = [n_sequences, len(common_usage_conditions)]
        metadata['condition_mask_dtype'] = 'bool'
    
    # Collect species mapping
    species_mapping = {}
    species_id_counter = 0
    for genome_id in sorted(genome_dirs_list):
        meta_path = os.path.join(input_dir, genome_id, 'metadata.json')
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        common_name = meta.get('genome_config', {}).get('common_name', genome_id)
        if common_name not in species_mapping:
            species_mapping[common_name] = species_id_counter
            species_id_counter += 1
    
    metadata['species_mapping'] = species_mapping
    metadata['n_species'] = len(species_mapping)
    
    return metadata

# Save metadata.json for splits
if train_offset > 0:
    train_meta = create_split_metadata(train_dir, train_offset, common_usage_conditions,
                                       genome_dirs, input_dir, pov_genome)
    with open(os.path.join(train_dir, 'metadata.json'), 'w') as f:
        json.dump(train_meta, f, indent=2)
    log_print(f"  Train metadata.json saved with {train_offset} sequences")

if test_offset > 0:
    test_meta = create_split_metadata(test_dir, test_offset, common_usage_conditions,
                                      genome_dirs, input_dir, pov_genome)
    with open(os.path.join(test_dir, 'metadata.json'), 'w') as f:
        json.dump(test_meta, f, indent=2)
    log_print(f"  Test metadata.json saved with {test_offset} sequences")

# Save summary
summary_data = {
    'pov_genome': pov_genome,
    'test_chromosomes': sorted(list(test_chromosomes)),
    'n_genomes': len(genome_dirs),
    'genomes': sorted(genome_dirs),
    'train': {
        'n_sequences': train_offset,
    },
    'test': {
        'n_sequences': test_offset,
    },
    'created_at': datetime.now().isoformat()
}

with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
    json.dump(summary_data, f, indent=2)

step5_time = time.time() - step5_start

# Final garbage collection
gc.collect()
mem_after_step5 = get_memory_usage()

log_print(f"  Data processed and saved in {step5_time:.2f} seconds")
log_print(f"  Final memory usage: {mem_after_step5:.2f} GB\n")

# Final summary
total_time = time.time() - script_start_time
log_print("=" * 60)
log_print("SUMMARY")
log_print("=" * 60)
log_print(f"Total genomes processed: {len(genome_dirs)}")
log_print(f"Train sequences: {train_offset}")
log_print(f"Test sequences: {test_offset}")
log_print(f"Output directory: {output_dir}")
log_print(f"Total time: {format_time(total_time)}")
log_print("=" * 60)
log_print("\nStep Timings:")
log_print(f"  Step 1 (Discovery): {format_time(step1_time)}")
log_print(f"  Step 2 (Orthology loading): {format_time(step2_time)}")
log_print(f"  Step 3 (Split determination): {format_time(step3_time)}")
log_print(f"  Step 4 (Data loading): {format_time(step4_time)}")
log_print(f"  Step 5 (Data saving): {format_time(step5_time)}")
log_print("=" * 60)

log_print(f"\nData splitting completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if args.quiet:
    sys.stdout = original_stdout
    sys.stderr = original_stderr
    log_file_handle.close()
