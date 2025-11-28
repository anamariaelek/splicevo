"""Load splice site data from a single genome and save as intermediate format.

This script processes one genome at a time to minimize memory usage.
Run this script separately for each genome you want to process.
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

from splicevo.data import MultiGenomeDataLoader

parser = argparse.ArgumentParser(description="Load splicing data from a single genome")
parser.add_argument("--output_dir", type=str, required=True, help="Base directory to save results (genome will be saved to output_dir/genome_id/)")
parser.add_argument("--config", type=str, required=True, help="Path to genome configuration JSON file")
parser.add_argument("--genome_id", type=str, required=True, help="Genome ID to process (must match config)")
parser.add_argument("--window_size", type=int, default=1000, help="Window size for sequences")
parser.add_argument("--context_size", type=int, default=450, help="Context size on each side")
parser.add_argument("--alpha_threshold", type=int, default=5, help="Minimum alpha value threshold")
parser.add_argument("--n_cpus", type=int, default=8, help="Number of CPU cores to use")
parser.add_argument("--quiet", action='store_true', help="Suppress console output")
args = parser.parse_args()

genome_id = args.genome_id
output_dir = args.output_dir
n_cpus = args.n_cpus
window_size = args.window_size
context_size = args.context_size
alpha_threshold = args.alpha_threshold

# Start timing
script_start_time = time.time()

# Create genome-specific output directory
genome_output_dir = os.path.join(output_dir, genome_id)
os.makedirs(genome_output_dir, exist_ok=True)

# Setup logging
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_file = os.path.join(genome_output_dir, f'data_load_{timestamp}.txt')

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

log_print(f"Data loading started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log_print(f"Genome ID: {genome_id}")
log_print(f"Output directory: {genome_output_dir}")
log_print("=" * 60)

# Load configuration
log_print("\nStep 1: Loading genome configuration...")
step1_start = time.time()

with open(args.config, 'r') as f:
    config = json.load(f)

genomes_config = config.get('genomes', [])
usage_files_config = config.get('usage_files', {})

# Find the specific genome in config
genome_config = None
for g in genomes_config:
    if g['genome_id'] == genome_id:
        genome_config = g
        break

if genome_config is None:
    log_print(f"ERROR: Genome {genome_id} not found in config file!")
    sys.exit(1)

log_print(f"  Found genome configuration for {genome_id}")
log_print(f"  Genome path: {genome_config['genome_path']}")
log_print(f"  GTF path: {genome_config['gtf_path']}")

step1_time = time.time() - step1_start
log_print(f"✓ Configuration loaded in {step1_time:.2f} seconds\n")

# Step 2: Initialize loader and load this genome
log_print("Step 2: Loading genome data...")
step2_start = time.time()

loader = MultiGenomeDataLoader(orthology_file=None)
loader.add_genome(
    genome_id=genome_id,
    genome_path=genome_config['genome_path'],
    gtf_path=genome_config['gtf_path'],
    chromosomes=genome_config.get('chromosomes', None),
    metadata=genome_config.get('metadata', {}),
    common_name=genome_config.get('common_name', None)
)

# Load the genome data
log_print(f"  Loading splice sites from {genome_id}...")
loader.load_all_genomes_data()

log_print(f"  Loaded {len(loader.loaded_data)} splice sites")

step2_time = time.time() - step2_start
log_print(f"✓ Genome loaded in {step2_time:.2f} seconds\n")

# Step 3: Add usage files from config
log_print("Step 3: Loading usage data...")
step3_start = time.time()

usage_times = {}
for genome_id, usage_config in usage_files_config.items():
    # Skip if genome not in loader
    if genome_id not in loader.genomes:
        continue
    
    usage_start = time.time()
    log_print(f"  Adding usage files for {genome_id}...")
    
    usage_pattern = usage_config.get('pattern', None)
    usage_list = usage_config.get('files', [])
    
    usage_count = 0
    
    if usage_pattern:
        # Use pattern-based loading
        tissues = usage_config.get('tissues', [])
        timepoints = usage_config.get('timepoints', [])
        
        log_print(f"    pattern: {usage_pattern}")
        log_print(f"    tissues: {tissues}")
        log_print(f"    timepoints: {timepoints}")
        
        for tissue in tissues:
            for timepoint in timepoints:
                usage_file = usage_pattern.format(tissue=tissue, timepoint=timepoint)
                
                if not os.path.exists(usage_file):
                    log_print(f"    Warning: {usage_file} not found")
                    continue
                
                try:
                    loader.add_usage_file(
                        genome_id=genome_id,
                        usage_file=usage_file,
                        tissue=tissue,
                        timepoint=str(timepoint)
                    )
                    usage_count += 1
                except Exception as e:
                    log_print(f"    Error loading {usage_file}: {e}")
    else:
        # Use explicit file list
        for usage_entry in usage_list:
            usage_file = usage_entry['file']
            
            if not os.path.exists(usage_file):
                log_print(f"    Warning: {usage_file} not found")
                continue
            
            try:
                loader.add_usage_file(
                    genome_id=genome_id,
                    usage_file=usage_file,
                    tissue=usage_entry['tissue'],
                    timepoint=usage_entry.get('timepoint', None)
                )
                usage_count += 1
            except Exception as e:
                log_print(f"    Error loading {usage_file}: {e}")
    
    usage_time = time.time() - usage_start
    usage_times[genome_id] = usage_time
    log_print(f"  ✓ {usage_count} usage files added for {genome_id} in {usage_time:.2f} seconds")

step3_time = time.time() - step3_start
log_print(f"✓ Usage files processed in {step3_time:.2f} seconds")

conditions_df = loader.get_available_conditions()
log_print(f"  Available conditions: {len(conditions_df)}")
log_print("")

# Step 4: Extract sequences and labels
log_print("Step 4: Extracting sequences and labels...")
step4_start = time.time()

# Get all sequences and save directly to memmap
log_print(f"  Converting to arrays and saving to memmap (window={window_size}, context={context_size})...")
sequences, labels, usage_arrays, metadata, species_ids = loader.to_arrays(
    window_size=window_size,
    context_size=context_size,
    alpha_threshold=alpha_threshold,
    n_workers=n_cpus,
    save_memmap=genome_output_dir
)

log_print(f"  Generated {len(sequences)} sequence windows")
log_print(f"  Sequence shape: {sequences.shape}")
log_print(f"  Label shape: {labels.shape}")
if usage_arrays:
    for key, arr in usage_arrays.items():
        log_print(f"  Usage array '{key}' shape: {arr.shape}")

step4_time = time.time() - step4_start
log_print(f"✓ Arrays extracted and saved in {step4_time:.2f} seconds\n")

# Step 5: Save data to disk
log_print("Step 5: Saving metadata...")
step5_start = time.time()

# Data is already saved as memmap in step 4, just save metadata and summary
seq_path = os.path.join(genome_output_dir, 'sequences.mmap')
labels_path = os.path.join(genome_output_dir, 'labels.mmap')

# Save metadata as CSV
metadata_path = os.path.join(genome_output_dir, 'metadata.csv')
log_print(f"  Saving metadata to {metadata_path}")
metadata.to_csv(metadata_path, index=False)

# Build usage_paths from the returned usage_arrays
usage_paths = {}
if usage_arrays:
    for key in usage_arrays.keys():
        usage_paths[key] = os.path.join(genome_output_dir, f'usage_{key}.mmap')

# Save summary metadata as JSON
summary = {
    'genome_id': genome_id,
    'n_sequences': int(len(sequences)),
    'n_splice_sites': int(len(loader.loaded_data)),
    'window_size': window_size,
    'context_size': context_size,
    'alpha_threshold': alpha_threshold,
    'sequence_shape': list(sequences.shape),
    'label_shape': list(labels.shape),
    'usage_conditions': list(usage_paths.keys()),
    'files': {
        'sequences': seq_path,
        'labels': labels_path,
        'metadata': metadata_path,
        'usage_arrays': usage_paths
    },
    'dtypes': {
        'sequences': str(sequences.dtype),
        'labels': str(labels.dtype),
    },
    'genome_config': genome_config,
    'created_at': datetime.now().isoformat()
}

if usage_arrays:
    for key, arr in usage_arrays.items():
        summary['dtypes'][f'usage_{key}'] = str(arr.dtype)

summary_path = os.path.join(genome_output_dir, 'summary.json')
log_print(f"  Saving summary to {summary_path}")
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2)

step5_time = time.time() - step5_start
log_print(f"✓ Metadata and summary saved in {step5_time:.2f} seconds\n")

# Final summary
total_time = time.time() - script_start_time
log_print("=" * 60)
log_print("SUMMARY")
log_print("=" * 60)
log_print(f"Genome ID: {genome_id}")
log_print(f"Total splice sites: {len(loader.loaded_data)}")
log_print(f"Total sequences: {len(sequences)}")
log_print(f"Output directory: {genome_output_dir}")
log_print(f"Total time: {format_time(total_time)}")
log_print("=" * 60)

log_print(f"\nData loading completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if args.quiet:
    sys.stdout = original_stdout
    sys.stderr = original_stderr
    log_file_handle.close()
