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
parser.add_argument("--folds_file", type=str, required=True, 
                    help="Path to folds TSV file with columns: split, genome_id, gene_id")
parser.add_argument("--n_cpus", type=int, default=2, help="Number of CPU cores to use")
parser.add_argument("--quiet", action='store_true', help="Suppress console output")
args = parser.parse_args()

input_dir = args.input_dir
genome_ids = args.genome_ids
output_dir = args.output_dir
folds_file = args.folds_file

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
log_print(f"Folds file: {folds_file}")
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

# Step 2: Load folds file
log_print("Step 2: Loading folds file...")
step2_start = time.time()

# Read the folds file - it's a TSV with columns: split, genome_id, gene_id
folds_df = pd.read_csv(folds_file, sep='\t', dtype={'split': str, 'genome_id': str, 'gene_id': str})

log_print(f"  Folds file shape: {folds_df.shape}")
log_print(f"  Columns: {list(folds_df.columns)}")

# Count splits
split_counts = folds_df['split'].value_counts()
log_print(f"  Split distribution:")
for split_name, count in split_counts.items():
    log_print(f"    {split_name}: {count}")

# Count genomes in folds file
genomes_in_folds = folds_df['genome_id'].unique()
log_print(f"  Genomes in folds file: {sorted(genomes_in_folds)}")

step2_time = time.time() - step2_start
mem_after_step2 = get_memory_usage()
log_print(f"  Orthology loaded in {step2_time:.2f} seconds")
log_print(f"  Memory usage: {mem_after_step2:.2f} GB\n")

# Step 3: Create gene-to-split mapping from folds file
log_print("Step 3: Creating gene-to-split mapping...")
step3_start = time.time()

# Create mapping: (genome_id, gene_id) -> 'train' or 'test'
# Combine 'train' and 'val' into 'train', keep 'test' as 'test'
gene_split_mapping = {}  # (genome_id, gene_id) -> 'train' or 'test'

for _, row in folds_df.iterrows():
    genome_id = row['genome_id']
    gene_id = row['gene_id']
    split = row['split']
    
    # Map val -> train, test -> test
    if split in ['train', 'val']:
        gene_split_mapping[(genome_id, gene_id)] = 'train'
    elif split == 'test':
        gene_split_mapping[(genome_id, gene_id)] = 'test'
    else:
        log_print(f"  WARNING: Unknown split '{split}' for {genome_id}:{gene_id}, skipping")

# Summarize assignments
train_assignments = sum(1 for v in gene_split_mapping.values() if v == 'train')
test_assignments = sum(1 for v in gene_split_mapping.values() if v == 'test')

log_print(f"  Gene assignments across all genomes: {len(gene_split_mapping)}")
log_print(f"    Train (including val): {train_assignments}")
log_print(f"    Test: {test_assignments}")

# Count by genome
genome_train_counts = {}
genome_test_counts = {}
for (genome_id, gene_id), split in gene_split_mapping.items():
    if split == 'train':
        genome_train_counts[genome_id] = genome_train_counts.get(genome_id, 0) + 1
    else:
        genome_test_counts[genome_id] = genome_test_counts.get(genome_id, 0) + 1

log_print(f"  Per-genome distribution:")
for genome_id in sorted(set(list(genome_train_counts.keys()) + list(genome_test_counts.keys()))):
    train_cnt = genome_train_counts.get(genome_id, 0)
    test_cnt = genome_test_counts.get(genome_id, 0)
    log_print(f"    {genome_id}: train={train_cnt}, test={test_cnt}")

step3_time = time.time() - step3_start
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

train_dir = os.path.join(output_dir, 'train')
test_dir = os.path.join(output_dir, 'test')
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)
species_condition_mapping = {}  # species_name -> list of condition indices
test_count = 0
test_sequences_mmap = None
test_metadata_list = []
test_labels_list = []
test_usage_sparse_list = []
test_offset = 0

log_print("Step 5: Processing genomes sequentially and writing to outputs...")
step5_start = time.time()

# Create output directories
train_dir = os.path.join(output_dir, 'train')
test_dir = os.path.join(output_dir, 'test')
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Build split DataFrame once for all genomes
split_df_all = pd.DataFrame(
    [(*k, v) for k, v in gene_split_mapping.items()],
    columns=["genome_id", "gene_id", "split"]
)

# First pass: count sequences to determine mmap shapes
train_count = 0
test_count = 0
for genome_id in sorted(genome_dirs):
    genome_dir = os.path.join(input_dir, genome_id)
    metadata = pd.read_csv(os.path.join(genome_dir, 'metadata.csv'), dtype={'genome_id': str, 'chromosome': str})
    # Merge metadata with split info for this genome
    split_df = split_df_all[split_df_all["genome_id"] == genome_id]
    metadata_merged = metadata.merge(split_df, on=["genome_id", "gene_id"], how="inner")
    train_count += (metadata_merged["split"] == "train").sum()
    test_count += (metadata_merged["split"] == "test").sum()

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
"""
First, collect all split assignments (including default assignments) for all genomes to determine the true train/test counts.
"""
all_train_indices = []
all_test_indices = []
all_metadata_merged = {}
for genome_idx, genome_id in enumerate(sorted(genome_dirs)):
    genome_dir = os.path.join(input_dir, genome_id)
    metadata = pd.read_csv(os.path.join(genome_dir, 'metadata.csv'), dtype={'genome_id': str, 'chromosome': str})
    split_df = split_df_all[split_df_all["genome_id"] == genome_id]
    metadata_merged = metadata.merge(split_df, on=["genome_id", "gene_id"], how="left")
    missing_split_mask = metadata_merged["split"].isna()
    metadata_merged.loc[missing_split_mask, "split"] = "train"
    train_mask = metadata_merged["split"] == "train"
    test_mask = metadata_merged["split"] == "test"
    all_metadata_merged[genome_id] = metadata_merged
    all_train_indices.append((genome_id, metadata_merged.index[train_mask].to_numpy()))
    all_test_indices.append((genome_id, metadata_merged.index[test_mask].to_numpy()))
true_train_count = sum(len(indices) for _, indices in all_train_indices)
true_test_count = sum(len(indices) for _, indices in all_test_indices)

log_print(f"  Recomputed total train sequences: {true_train_count}")
log_print(f"  Recomputed total test sequences: {true_test_count}")

# Now process as before, but use the recomputed counts for memmap shapes
train_sequences_mmap = None
test_sequences_mmap = None
train_metadata_list = []
test_metadata_list = []
train_labels_list = []
test_labels_list = []
train_usage_sparse_list = []
test_usage_sparse_list = []
train_offset = 0
test_offset = 0

for genome_idx, genome_id in enumerate(sorted(genome_dirs)):
    log_print(f"\n  Processing genome {genome_idx}: {genome_id}...")
    genome_dir = os.path.join(input_dir, genome_id)
    with open(os.path.join(genome_dir, 'metadata.json'), 'r') as f:
        meta = json.load(f)
    metadata_merged = all_metadata_merged[genome_id]
    train_mask = metadata_merged["split"] == "train"
    test_mask = metadata_merged["split"] == "test"
    log_print(f"    Loaded metadata: {len(metadata_merged)} entries")
    n_missing = (metadata_merged["split"] == "train").sum() + (metadata_merged["split"] == "test").sum() - len(metadata_merged)
    if n_missing > 0:
        log_print(f"    WARNING: {n_missing} metadata entries have no split assignment for {genome_id}, moving to train by default.")
    log_print(f"    Train sequences: {train_mask.sum()}")
    log_print(f"    Test sequences: {test_mask.sum()}")
    if not train_mask.any() and not test_mask.any():
        log_print(f"    Skipping {genome_id} - no sequences assigned")
        continue
    # Add metadata entries (vectorized)
    if train_mask.any():
        train_meta_df = metadata_merged.loc[train_mask, common_metadata_columns].copy()
        train_meta_df['species_id'] = genome_idx
        train_metadata_list.extend(train_meta_df.to_dict(orient='records'))
    if test_mask.any():
        test_meta_df = metadata_merged.loc[test_mask, common_metadata_columns].copy()
        test_meta_df['species_id'] = genome_idx
        test_metadata_list.extend(test_meta_df.to_dict(orient='records'))
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
    # Helper to remap sparse usage condition indices
    def remap_sparse_condition_indices(sparse_df, genome_conds, target_conds):
        if sparse_df is None or len(sparse_df) == 0:
            return sparse_df
        idx_mapping = {genome_idx: target_conds.index(genome_cond)
                       for genome_idx, genome_cond in enumerate(genome_conds) if genome_cond in target_conds}
        filtered = sparse_df[sparse_df['condition_idx'].isin(idx_mapping.keys())].copy()
        filtered['condition_idx'] = filtered['condition_idx'].map(idx_mapping)
        return filtered
    # Process train sequences (optimized batch filtering)
    train_indices = metadata_merged.index[train_mask].to_numpy()
    if len(train_indices) > 0:
        if train_sequences_mmap is None:
            log_print(f"    Initializing train mmap files...")
            train_sequences_mmap = np.memmap(os.path.join(train_dir, 'sequences.mmap'), 
                                            dtype=seq_dtype, mode='w+', 
                                            shape=(true_train_count, seq_window_size, 4))
        n_train = len(train_indices)
        batch_size = 1000
        log_print(f"    Writing {n_train} train sequences in batches of {batch_size}...")
        # Pre-filter and index labels/usage for train_indices
        if labels_sparse_df is not None and not labels_sparse_df.empty:
            train_labels_df = labels_sparse_df[labels_sparse_df['sample_idx'].isin(train_indices)].copy()
            train_labels_df.set_index('sample_idx', inplace=True)
        else:
            train_labels_df = None
        if usage_sparse_df is not None and not usage_sparse_df.empty:
            train_usage_df = usage_sparse_df[usage_sparse_df['sample_idx'].isin(train_indices)].copy()
            train_usage_df.set_index('sample_idx', inplace=True)
        else:
            train_usage_df = None
        for batch_start in range(0, n_train, batch_size):
            batch_end = min(batch_start + batch_size, n_train)
            batch_indices = train_indices[batch_start:batch_end]
            train_sequences_mmap[train_offset:train_offset + len(batch_indices)] = sequences_mmap[batch_indices]
            if train_labels_df is not None and not train_labels_df.empty:
                # Use .loc for fast access
                batch_labels = train_labels_df.loc[train_labels_df.index.intersection(batch_indices)].copy()
                idx_map = {src: train_offset + i for i, src in enumerate(batch_indices)}
                batch_labels.reset_index(inplace=True)
                batch_labels['sample_idx'] = batch_labels['sample_idx'].map(idx_map)
                train_labels_list.append(batch_labels)
            if train_usage_df is not None and not train_usage_df.empty:
                batch_sparse = train_usage_df.loc[train_usage_df.index.intersection(batch_indices)].copy()
                batch_sparse.reset_index(inplace=True)
                batch_sparse = remap_sparse_condition_indices(batch_sparse, genome_usage_conditions, common_usage_conditions)
                idx_map = {orig: new for new, orig in enumerate(batch_indices, start=train_offset)}
                batch_sparse['sample_idx'] = batch_sparse['sample_idx'].map(idx_map)
                train_usage_sparse_list.append(batch_sparse)
            train_offset += len(batch_indices)
    # Process test sequences (optimized batch filtering)
    test_indices = metadata_merged.index[test_mask].to_numpy()
    if len(test_indices) > 0:
        if test_sequences_mmap is None:
            log_print(f"    Initializing test mmap files...")
            test_sequences_mmap = np.memmap(os.path.join(test_dir, 'sequences.mmap'),
                                            dtype=seq_dtype, mode='w+', 
                                            shape=(true_test_count, seq_window_size, 4))
        n_test = len(test_indices)
        batch_size = 1000
        log_print(f"    Writing {n_test} test sequences in batches of {batch_size}...")
        # Pre-filter and index labels/usage for test_indices
        if labels_sparse_df is not None and not labels_sparse_df.empty:
            test_labels_df = labels_sparse_df[labels_sparse_df['sample_idx'].isin(test_indices)].copy()
            test_labels_df.set_index('sample_idx', inplace=True)
        else:
            test_labels_df = None
        if usage_sparse_df is not None and not usage_sparse_df.empty:
            test_usage_df = usage_sparse_df[usage_sparse_df['sample_idx'].isin(test_indices)].copy()
            test_usage_df.set_index('sample_idx', inplace=True)
        else:
            test_usage_df = None
        for batch_start in range(0, n_test, batch_size):
            batch_end = min(batch_start + batch_size, n_test)
            batch_indices = test_indices[batch_start:batch_end]
            test_sequences_mmap[test_offset:test_offset + len(batch_indices)] = sequences_mmap[batch_indices]
            if test_labels_df is not None and not test_labels_df.empty:
                batch_labels = test_labels_df.loc[test_labels_df.index.intersection(batch_indices)].copy()
                idx_map = {src: test_offset + i for i, src in enumerate(batch_indices)}
                batch_labels.reset_index(inplace=True)
                batch_labels['sample_idx'] = batch_labels['sample_idx'].map(idx_map)
                test_labels_list.append(batch_labels)
            if test_usage_df is not None and not test_usage_df.empty:
                batch_sparse = test_usage_df.loc[test_usage_df.index.intersection(batch_indices)].copy()
                batch_sparse.reset_index(inplace=True)
                batch_sparse = remap_sparse_condition_indices(batch_sparse, genome_usage_conditions, common_usage_conditions)
                idx_map = {orig: new for new, orig in enumerate(batch_indices, start=test_offset)}
                batch_sparse['sample_idx'] = batch_sparse['sample_idx'].map(idx_map)
                test_usage_sparse_list.append(batch_sparse)
            test_offset += len(batch_indices)
    log_print(f"    Genome {genome_id} processed")
    del sequences_mmap
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
                         genome_dirs_list, input_dir):
    """Create metadata.json for a train/test split."""
    
    # Load window_size and context_size from first genome config
    first_genome_path = os.path.join(input_dir, sorted(genome_dirs_list)[0], 'metadata.json')
    with open(first_genome_path, 'r') as f:
        first_genome_summary = json.load(f)
    
    window_size = first_genome_summary.get('window_size', 1000)
    context_size = first_genome_summary.get('context_size', 450)
    
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
                                       genome_dirs, input_dir)
    with open(os.path.join(train_dir, 'metadata.json'), 'w') as f:
        json.dump(train_meta, f, indent=2)
    log_print(f"  Train metadata.json saved with {train_offset} sequences")

if test_offset > 0:
    test_meta = create_split_metadata(test_dir, test_offset, common_usage_conditions,
                                      genome_dirs, input_dir)
    with open(os.path.join(test_dir, 'metadata.json'), 'w') as f:
        json.dump(test_meta, f, indent=2)
    log_print(f"  Test metadata.json saved with {test_offset} sequences")

# Save summary
summary_data = {
    'folds_file': folds_file,
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
