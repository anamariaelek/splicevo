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
import psutil
import gc

from splicevo.data import MultiGenomeDataLoader

parser = argparse.ArgumentParser(description="Load splicing data from a single genome")
parser.add_argument("--output_dir", type=str, required=True, help="Base directory to save results (genome will be saved to output_dir/genome_id/)")
parser.add_argument("--config", type=str, required=True, help="Path to genome configuration JSON file")
parser.add_argument("--genome_id", type=str, required=True, help="Genome ID to process (must match config)")
parser.add_argument("--window_size", type=int, default=1000, help="Window size for sequences")
parser.add_argument("--context_size", type=int, default=450, help="Context size on each side")
parser.add_argument("--alpha_threshold", type=int, default=5, help="Minimum alpha value threshold")
parser.add_argument("--n_cpus", type=int, default=8, help="Number of CPU cores to use")
parser.add_argument("--dry_run", action='store_true', help="Don't extract arrrays, just metadata")
parser.add_argument("--quiet", action='store_true', help="Suppress console output")
args = parser.parse_args()

config= args.config
genome_id = args.genome_id
output_dir = args.output_dir
n_cpus = args.n_cpus
window_size = args.window_size
context_size = args.context_size
alpha_threshold = args.alpha_threshold
dry_run = args.dry_run
quiet = args.quiet

# For debugging
debugging = False
if debugging:
    home = "/home/elek/"
    config=os.path.join(home, "projects/splicevo/configs/genomes_small.json")
    genome_id = "mouse_GRCm38"
    output_dir=os.path.join(home, "sds/sd17d003/Anamaria/splicevo/data/processed_small/")
    n_cpus = 4
    window_size = 1000
    context_size = 450
    alpha_threshold = 5
    dry_run = False
    quiet = False

# Start timing
script_start_time = time.time()

# Create genome-specific output directory
genome_output_dir = os.path.join(output_dir, genome_id)
os.makedirs(genome_output_dir, exist_ok=True)

# Setup logging
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_file = os.path.join(genome_output_dir, f'data_load_{timestamp}.txt')

if quiet:
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

log_print(f"Data loading started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log_print(f"Genome ID: {genome_id}")
log_print(f"Output directory: {genome_output_dir}")
log_print("=" * 60)

# Load configuration
log_print("\nStep 1: Loading genome configuration...")
step1_start = time.time()

with open(config, 'r') as f:
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

log_print(f"Found genome configuration for {genome_id}")
log_print(f"Genome path: {genome_config['genome_path']}")
log_print(f"GTF path: {genome_config['gtf_path']}")

step1_time = time.time() - step1_start
mem_after_step1 = get_memory_usage()
log_print(f"Configuration loaded in {step1_time:.2f} seconds")
log_print(f"Memory usage: {mem_after_step1:.2f} GB\n")

# Step 2: Initialize loader and add this genome
log_print( "\nStep 2: Adding genome data...")
step2_start = time.time()

loader = MultiGenomeDataLoader()
loader.add_genome(
    genome_id=genome_id,
    genome_path=genome_config['genome_path'],
    gtf_path=genome_config['gtf_path'],
    chromosomes=genome_config.get('chromosomes', None),
    metadata=genome_config.get('metadata', {})
)

step2_time = time.time() - step2_start
mem_after_step2 = get_memory_usage()
log_print(f"Genome added in {step2_time:.2f} seconds")
log_print(f"Memory usage: {mem_after_step2:.2f} GB\n")

# Step 3: Add usage files from config
log_print( "\nStep 3: Adding usage data...")
step3_start = time.time()

usage_times = {}
for genome_name, usage_config in usage_files_config.items():
    # Skip if genome not in loader
    if genome_name not in loader.genomes:
        continue
    
    usage_start = time.time()
    log_print(f"Adding usage files for {genome_name}...")
    
    usage_pattern = usage_config.get('pattern', None)
    usage_list = usage_config.get('files', [])
    
    usage_count = 0
    
    if usage_pattern:
        # Use pattern-based loading
        tissues = usage_config.get('tissues', [])
        timepoints = usage_config.get('timepoints', [])
        
        log_print(f"  pattern: {usage_pattern}")
        log_print(f"  tissues: {tissues}")
        log_print(f"  timepoints: {timepoints}")
        
        for tissue in tissues:
            for timepoint in timepoints:
                usage_file = usage_pattern.format(tissue=tissue, timepoint=timepoint)
                
                if not os.path.exists(usage_file):
                    log_print(f"  Warning: {usage_file} not found")
                    continue
                
                try:
                    loader.add_usage_file(
                        genome_id=genome_name,
                        usage_file=usage_file,
                        tissue=tissue,
                        timepoint=str(timepoint)
                    )
                    usage_count += 1
                except Exception as e:
                    log_print(f"  Error loading {usage_file}: {e}")
    else:
        # Use explicit file list
        for usage_entry in usage_list:
            usage_file = usage_entry['file']
            
            if not os.path.exists(usage_file):
                log_print(f"  Warning: {usage_file} not found")
                continue
            
            try:
                loader.add_usage_file(
                    genome_id=genome_name,
                    usage_file=usage_file,
                    tissue=usage_entry['tissue'],
                    timepoint=usage_entry.get('timepoint', None)
                )
                usage_count += 1
            except Exception as e:
                log_print(f"  Error loading {usage_file}: {e}")
    
    usage_time = time.time() - usage_start
    usage_times[genome_name] = usage_time
    log_print(f"  {usage_count} usage files added for {genome_name} in {usage_time:.2f} seconds")

step3_time = time.time() - step3_start
mem_after_step3 = get_memory_usage()
log_print(f"Usage files processed in {step3_time:.2f} seconds")
log_print(f"Memory usage: {mem_after_step3:.2f} GB")

# Step 4: Load the genome data
log_print(f"\nStep 4: Loading splice sites from {genome_id}...")
step4_start = time.time()
loader.load_all_genomes_data()
loader_df = loader.get_dataframe()
loader_df.to_csv(os.path.join(genome_output_dir, 'splice_sites.csv'), index=False)

log_print(f"Loaded {len(loader.loaded_data)} splice sites")

# Get available usage conditions
conditions_df = loader.get_available_conditions()
log_print(f"Available conditions: {len(conditions_df)}")

# Add summary of loaded usage data
usage_summary = loader.get_usage_summary()
if len(usage_summary) > 0:
    log_print(f"\n  Usage data loaded:")
    for _, row in usage_summary.iterrows():
        log_print(f"  {row['genome_id']} - {row['display_name']}: {row['n_sites']} sites")
    usage_summary.to_csv(os.path.join(genome_output_dir, 'usage_summary.csv'), index=False)
else:
    log_print("WARNING: No usage data loaded!")

step4_time = time.time() - step4_start

# Force garbage collection and log memory
gc.collect()
mem_after_step4 = get_memory_usage()
log_print(f"Loading splice sites completed in {step4_time:.2f} seconds")
log_print(f"Memory usage after step 4: {mem_after_step4:.2f} GB\n")

# Step 5: Extract sequences and labels
log_print( "\nStep 5: Extracting sequences and labels to arrays...")
step5_start = time.time()

# If not a dry run, extract arrays
if not dry_run:
    # Get all sequences and save directly to memmap
    log_print(f"Converting to arrays and saving to memmap (window={window_size}, context={context_size})...")
    sequences, labels, usage_arrays, metadata = loader.to_arrays(
        window_size=window_size,
        context_size=context_size,
        alpha_threshold=alpha_threshold,
        n_workers=n_cpus,
        use_parallel=True,
        save_memmap=genome_output_dir
    )
    log_print(f"Generated {len(sequences)} sequence windows")
    log_print(f"Sequence shape: {sequences.shape}")
    log_print(f"Label shape: {labels.shape}")
    if usage_arrays:
        for key, arr in usage_arrays.items():
            log_print(f"Usage array '{key}' shape: {arr.shape}")
            # Check if any values are not NaN
            if np.all(np.isnan(arr)):
                log_print(f"  Warning: All values in usage array '{key}' are NaN")
            else:
                # Add info about loaded usage data
                usage_info = loader.get_usage_array_info(usage_arrays = usage_arrays)
        
    step5_time = time.time() - step5_start
    
    # Force garbage collection and log memory
    gc.collect()
    mem_after_step5 = get_memory_usage()
    log_print(f"Arrays extracted and saved in {step5_time:.2f} seconds")
    log_print(f"Memory usage after step 5: {mem_after_step5:.2f} GB")
    log_print(f"Memory increase from step 4 to 5: {mem_after_step5 - mem_after_step4:.2f} GB\n")
else:
    log_print("Dry run specified, skipping array extraction.")
    sequences = np.array([])
    labels = np.array([])
    usage_arrays = {}
    step5_time = time.time() - step5_start
    log_print(f"Dry run completed in {step5_time:.2f} seconds\n")

# Step 6: Save data to disk
log_print( "\nStep 6: Saving metadata...")
step6_start = time.time()

# Keep track of usage condition mapping 
usage_condition_mapping = {}
if len(conditions_df) > 0:
    for idx, row in conditions_df.iterrows():
        cond = row['condition_key']
        usage_condition_mapping[cond] = {
            'idx': int(idx),
            'tissue': row.get('tissue', ''),
            'timepoint': row.get('timepoint', ''),
            'display_name': row.get('display_name', cond)
        }

# Save consolidated metadata.json (includes both summary and metadata info)
metadata_json = {
    # Summary information
    'genome_id': genome_id,
    'n_sequences': int(len(sequences)),
    'n_splice_sites': int(len(loader.loaded_data)),
    'window_size': window_size,
    'context_size': context_size,
    'alpha_threshold': alpha_threshold,
    'genome_config': genome_config,
    'created_at': datetime.now().isoformat(),
    
    # Array shapes and dtypes
    'sequences_shape': list(sequences.shape),
    'labels_shape': list(labels.shape),
    'sequences_dtype': str(sequences.dtype),
    'labels_dtype': str(labels.dtype),
    'usage_conditions': list(usage_condition_mapping.keys()),
    'usage_condition_mapping': usage_condition_mapping,
    
    # File paths
    'files': {
        'sequences': os.path.join(genome_output_dir, 'sequences.mmap'),
        'labels': os.path.join(genome_output_dir, 'labels.mmap'),
        'metadata': os.path.join(genome_output_dir, 'metadata.csv'),
    }
}

# Add usage array info if present
if usage_arrays:
    for key, arr in usage_arrays.items():
        metadata_json[f'{key}_shape'] = list(arr.shape)
        metadata_json[f'{key}_dtype'] = str(arr.dtype)
        metadata_json['files'][f'{key}'] = os.path.join(genome_output_dir, f'usage_{key}.mmap')

metadata_json_path = os.path.join(genome_output_dir, 'metadata.json')
log_print(f"Saving metadata to {metadata_json_path}")
with open(metadata_json_path, 'w') as f:
    json.dump(metadata_json, f, indent=2)

# Save metadata.csv
metadata_path = os.path.join(genome_output_dir, 'metadata.csv')
if not dry_run:
    log_print(f"Saving sequence metadata to {metadata_path}")
    metadata.to_csv(metadata_path, index=False)

step6_time = time.time() - step6_start
log_print(f"Metadata saved in {step6_time:.2f} seconds\n")

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
log_print("\nStep Timings:")
log_print(f"Step 1 (Configuration): {format_time(step1_time)}")
log_print(f"Step 2 (Genome loading): {format_time(step2_time)}")
log_print(f"Step 3 (Usage files): {format_time(step3_time)}")
log_print(f"Step 4 (Splice site loading): {format_time(step4_time)}")
log_print(f"Step 5 (Array extraction): {format_time(step5_time)}")
log_print(f"Step 6 (Metadata saving): {format_time(step6_time)}")
log_print("=" * 60)

log_print(f"\nData loading completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if args.quiet:
    sys.stdout = original_stdout
    sys.stderr = original_stderr
    log_file_handle.close()
