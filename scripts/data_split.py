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

# Natural sort for conditions (so Brain_5 comes before Brain_10)
import re
def natural_sort_key(s):
    """Sort strings with embedded numbers naturally."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

common_usage_conditions = sorted(list(all_usage_conditions), key=natural_sort_key)
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

# Track which species have which conditions (for condition masking)
species_condition_mapping = {}  # species_name -> list of condition indices

# Track which species have which conditions (for condition masking)
species_condition_mapping = {}  # species_name -> list of condition indices

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
seq_window_size = first_meta['sequences_shape'][1]
window_size = seq_window_size  # Same for sequences and labels
n_usage_conditions = len(common_usage_conditions)

# Initialize memory-mapped output files
train_sequences_mmap = None
test_sequences_mmap = None

train_metadata_list = []
test_metadata_list = []

# Lists to collect sparse labels and usage data
train_labels_list = []
test_labels_list = []
train_usage_sparse_list = []
test_usage_sparse_list = []

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
    
    log_print(f"    Opening sequences memmap (not loading into memory)...")
    sequences_mmap = np.memmap(os.path.join(genome_dir, 'sequences.mmap'), 
                              dtype=seq_dtype, mode='r', shape=seq_shape)
    
    # Load sparse labels data
    labels_sparse_path = os.path.join(genome_dir, 'labels.parquet')
    labels_sparse_df = None
    if os.path.exists(labels_sparse_path):
        log_print(f"    Loading sparse labels data...")
        labels_sparse_df = pd.read_parquet(labels_sparse_path)
        log_print(f"    Loaded {len(labels_sparse_df)} sparse label entries")
    else:
        log_print(f"    No labels data found for {genome_id}")
    
    # Load sparse usage data
    usage_sparse_path = os.path.join(genome_dir, 'usage.parquet')
    genome_usage_conditions = meta.get('usage_conditions', [])
    
    usage_sparse_df = None
    if os.path.exists(usage_sparse_path):
        log_print(f"    Loading sparse usage data...")
        usage_sparse_df = pd.read_parquet(usage_sparse_path)
        log_print(f"    Loaded {len(usage_sparse_df)} sparse usage entries")
    else:
        log_print(f"    No usage data found for {genome_id}")
    
    # Track which conditions this species has (map to global condition indices)
    genome_common_name = meta.get('genome_config', {}).get('common_name', genome_id)
    if genome_common_name not in species_condition_mapping:
        species_condition_mapping[genome_common_name] = []
        for genome_cond in genome_usage_conditions:
            if genome_cond in common_usage_conditions:
                global_idx = common_usage_conditions.index(genome_cond)
                if global_idx not in species_condition_mapping[genome_common_name]:
                    species_condition_mapping[genome_common_name].append(global_idx)
    
    # Track which conditions this species has (map to global condition indices)
    genome_common_name = meta.get('genome_config', {}).get('common_name', genome_id)
    if genome_common_name not in species_condition_mapping:
        species_condition_mapping[genome_common_name] = []
        for genome_cond in genome_usage_conditions:
            if genome_cond in common_usage_conditions:
                global_idx = common_usage_conditions.index(genome_cond)
                if global_idx not in species_condition_mapping[genome_common_name]:
                    species_condition_mapping[genome_common_name].append(global_idx)
    
    # Helper to remap sparse usage condition indices
    def remap_sparse_condition_indices(sparse_df, genome_conds, target_conds):
        """Remap condition_idx in sparse dataframe to match target conditions."""
        if sparse_df is None or len(sparse_df) == 0:
            return sparse_df
        # Create mapping from genome condition index to target condition index
        idx_mapping = {}
        for genome_idx, genome_cond in enumerate(genome_conds):
            if genome_cond in target_conds:
                target_idx = target_conds.index(genome_cond)
                idx_mapping[genome_idx] = target_idx
        # Filter and remap
        filtered = sparse_df[sparse_df['condition_idx'].isin(idx_mapping.keys())].copy()
        filtered['condition_idx'] = filtered['condition_idx'].map(idx_mapping)
        return filtered
    
    # Process train sequences
    if train_indices:
        # Initialize mmap files on first write
        if train_sequences_mmap is None:
            log_print(f"    Initializing train mmap files...")
            train_sequences_mmap = np.memmap(os.path.join(train_dir, 'sequences.mmap'), 
                                            dtype=seq_dtype, mode='w+', 
                                            shape=(train_count, seq_window_size, 4))
        
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
            
            # Handle sparse labels data
            if labels_sparse_df is not None:
                # Filter to batch samples, remap sample_idx to target indices
                batch_labels = labels_sparse_df[labels_sparse_df['sample_idx'].isin(batch_indices)].copy()
                # Create mapping from source to target indices
                idx_map = {src: train_offset + i for i, src in enumerate(batch_indices)}
                batch_labels['sample_idx'] = batch_labels['sample_idx'].map(idx_map)
                train_labels_list.append(batch_labels)
            
            # Handle sparse usage data
            if usage_sparse_df is not None:
                # Filter to batch samples and remap
                batch_sparse = usage_sparse_df[usage_sparse_df['sample_idx'].isin(batch_indices)].copy()
                # Remap condition indices to common conditions
                batch_sparse = remap_sparse_condition_indices(batch_sparse, 
                                                             genome_usage_conditions,
                                                             common_usage_conditions)
                # Adjust sample_idx to global train offset
                idx_map = {orig: new for new, orig in enumerate(batch_indices, start=train_offset)}
                batch_sparse['sample_idx'] = batch_sparse['sample_idx'].map(idx_map)
                train_usage_sparse_list.append(batch_sparse)
            
            train_offset += batch_size_actual
    
    # Process test sequences
    if test_indices:
        # Initialize mmap files on first write
        if test_sequences_mmap is None:
            log_print(f"    Initializing test mmap files...")
            test_sequences_mmap = np.memmap(os.path.join(test_dir, 'sequences.mmap'),
                                            dtype=seq_dtype, mode='w+', 
                                           shape=(test_count, seq_window_size, 4))
        
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
            
            # Handle sparse labels data
            if labels_sparse_df is not None:
                # Filter to batch samples, remap sample_idx to target indices
                batch_labels = labels_sparse_df[labels_sparse_df['sample_idx'].isin(batch_indices)].copy()
                # Create mapping from source to target indices
                idx_map = {src: test_offset + i for i, src in enumerate(batch_indices)}
                batch_labels['sample_idx'] = batch_labels['sample_idx'].map(idx_map)
                test_labels_list.append(batch_labels)
            
            # Handle sparse usage data
            if usage_sparse_df is not None:
                # Filter to batch samples and remap
                batch_sparse = usage_sparse_df[usage_sparse_df['sample_idx'].isin(batch_indices)].copy()
                # Remap condition indices to common conditions
                batch_sparse = remap_sparse_condition_indices(batch_sparse, 
                                                             genome_usage_conditions,
                                                             common_usage_conditions)
                # Adjust sample_idx to global test offset
                idx_map = {orig: new for new, orig in enumerate(batch_indices, start=test_offset)}
                batch_sparse['sample_idx'] = batch_sparse['sample_idx'].map(idx_map)
                test_usage_sparse_list.append(batch_sparse)
            
            test_offset += batch_size_actual
    
    log_print(f"    Genome {genome_id} processed")
    # Close references to free file handles
    del sequences_mmap, metadata
    if usage_sparse_df is not None:
        del usage_sparse_df
    gc.collect()
    
    log_print(f"    Memory after processing: {get_memory_usage():.2f} GB")

# Flush and close mmap files
if train_sequences_mmap is not None:
    del train_sequences_mmap

if test_sequences_mmap is not None:
    del test_sequences_mmap

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

# Save sparse labels data if any was collected
if train_labels_list:
    log_print(f"\n  Saving sparse labels data...")
    train_labels_sparse = pd.concat(train_labels_list, ignore_index=True)
    # Convert to efficient dtypes
    train_labels_sparse['sample_idx'] = train_labels_sparse['sample_idx'].astype(np.int32)
    train_labels_sparse['position'] = train_labels_sparse['position'].astype(np.int16)
    train_labels_sparse['label'] = train_labels_sparse['label'].astype(np.int8)
    train_labels_sparse.to_parquet(os.path.join(train_dir, 'labels.parquet'), 
                                   compression='snappy', index=False)
    log_print(f"    Train sparse labels: {len(train_labels_sparse)} entries")
    del train_labels_sparse

if test_labels_list:
    test_labels_sparse = pd.concat(test_labels_list, ignore_index=True)
    # Convert to efficient dtypes
    test_labels_sparse['sample_idx'] = test_labels_sparse['sample_idx'].astype(np.int32)
    test_labels_sparse['position'] = test_labels_sparse['position'].astype(np.int16)
    test_labels_sparse['label'] = test_labels_sparse['label'].astype(np.int8)
    test_labels_sparse.to_parquet(os.path.join(test_dir, 'labels.parquet'), 
                                  compression='snappy', index=False)
    log_print(f"    Test sparse labels: {len(test_labels_sparse)} entries")
    del test_labels_sparse

# Save sparse usage data if any was collected
if train_usage_sparse_list:
    log_print(f"\n  Saving sparse usage data...")
    train_usage_sparse = pd.concat(train_usage_sparse_list, ignore_index=True)
    # Convert to efficient dtypes
    train_usage_sparse['sample_idx'] = train_usage_sparse['sample_idx'].astype(np.int32)
    train_usage_sparse['position'] = train_usage_sparse['position'].astype(np.int16)
    train_usage_sparse['condition_idx'] = train_usage_sparse['condition_idx'].astype(np.int8)
    train_usage_sparse.to_parquet(os.path.join(train_dir, 'usage.parquet'), 
                                  compression='snappy', index=False)
    log_print(f"    Train sparse usage: {len(train_usage_sparse)} entries")
    sparsity = 100 * len(train_usage_sparse) / (train_offset * window_size * n_usage_conditions)
    log_print(f"    Train sparsity: {sparsity:.4f}%")
    del train_usage_sparse

if test_usage_sparse_list:
    test_usage_sparse = pd.concat(test_usage_sparse_list, ignore_index=True)
    # Convert to efficient dtypes
    test_usage_sparse['sample_idx'] = test_usage_sparse['sample_idx'].astype(np.int32)
    test_usage_sparse['position'] = test_usage_sparse['position'].astype(np.int16)
    test_usage_sparse['condition_idx'] = test_usage_sparse['condition_idx'].astype(np.int8)
    test_usage_sparse.to_parquet(os.path.join(test_dir, 'usage.parquet'), 
                                 compression='snappy', index=False)
    log_print(f"    Test sparse usage: {len(test_usage_sparse)} entries")
    sparsity = 100 * len(test_usage_sparse) / (test_offset * window_size * n_usage_conditions)
    log_print(f"    Test sparsity: {sparsity:.4f}%")
    del test_usage_sparse

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
    
    # Create usage condition mapping (index -> condition info)
    usage_condition_mapping = {}
    for idx, cond_key in enumerate(common_usage_conditions):
        usage_condition_mapping[str(idx)] = {
            'condition_key': cond_key,
            'display_name': cond_key
        }
    
    # Build consolidated metadata
    metadata = {
        'sequences_shape': [n_sequences, seq_window_size, 4],
        'sequences_dtype': seq_dtype,
        'labels_format': 'sparse',
        'window_size': window_size,
        'context_size': context_size,
        'usage_conditions': common_usage_conditions,
        'usage_condition_mapping': usage_condition_mapping,
        'species_condition_mapping': species_condition_mapping,
    }
    
    # Add labels info if present (sparse format only)
    labels_sparse_path = os.path.join(output_split_dir, 'labels.parquet')
    if os.path.exists(labels_sparse_path):
        labels_df = pd.read_parquet(labels_sparse_path)
        metadata['labels_sparse_entries'] = len(labels_df)
    
    # Add usage array info if present (sparse format only)
    usage_sparse_path = os.path.join(output_split_dir, 'usage.parquet')
    if os.path.exists(usage_sparse_path):
        metadata['usage_format'] = 'sparse'
        usage_df = pd.read_parquet(usage_sparse_path)
        metadata['usage_sparse_entries'] = len(usage_df)
    
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
