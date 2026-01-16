"""Create condition masks for existing data splits.

This script retroactively adds condition_mask.mmap to train/test directories
that were created before the masking feature was implemented.
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
import time
from datetime import datetime

parser = argparse.ArgumentParser(description="Create condition masks for existing splits")
parser.add_argument("--split_dir", type=str, required=True,
                    help="Directory containing train/ and test/ subdirectories")
parser.add_argument("--genome_config", type=str, required=True,
                    help="Path to genome config JSON file with usage_conditions")
parser.add_argument("--quiet", action='store_true', help="Suppress console output")
args = parser.parse_args()

split_dir = args.split_dir
genome_config_path = args.genome_config

# Setup logging
def log_print(msg):
    if not args.quiet:
        print(msg)
        sys.stdout.flush()

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_file = os.path.join(split_dir, f"create_mask_{timestamp}.log")

def log_to_file(msg):
    with open(log_file, 'a') as f:
        f.write(msg + '\n')

def log_both(msg):
    log_print(msg)
    log_to_file(msg)

log_both(f"Create Condition Mask Script")
log_both(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log_both(f"Split directory: {split_dir}")
log_both(f"Genome config: {genome_config_path}\n")

script_start = time.time()

# Step 1: Load genome configuration and determine actual per-genome conditions
log_both("Step 1: Loading genome configuration and per-genome conditions...")
step1_start = time.time()

# First, try to load actual per-genome conditions from the processed data directories
# Look for metadata in processed genome directories
processed_dir = Path(split_dir).parent.parent / 'processed_full_1kb'  # Go up to data dir
if not processed_dir.exists():
    processed_dir = Path(split_dir).parent / 'processed_full_1kb'
if not processed_dir.exists():
    # Try without 1kb/5kb suffix
    processed_dir = Path(split_dir).parent.parent / 'processed'
    
log_both(f"  Looking for processed data in: {processed_dir}")

genome_usage_conditions = {}

# Try to find actual per-genome conditions from processed data
for potential_genome_id in ['human_GRCh37', 'mouse_GRCm38', 'rat_Rnor_5.0']:
    genome_dir = processed_dir / potential_genome_id
    if genome_dir.exists():
        # Try to load metadata.json from processed genome directory
        metadata_file = genome_dir / 'metadata.json'
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                genome_metadata = json.load(f)
            
            # Look for usage_conditions in metadata
            if 'usage_conditions' in genome_metadata:
                genome_usage_conditions[potential_genome_id] = genome_metadata['usage_conditions']
                log_both(f"  Found {potential_genome_id}: {len(genome_metadata['usage_conditions'])} conditions from processed data")
            elif 'alpha' in genome_metadata and 'conditions' in genome_metadata['alpha']:
                genome_usage_conditions[potential_genome_id] = genome_metadata['alpha']['conditions']
                log_both(f"  Found {potential_genome_id}: {len(genome_metadata['alpha']['conditions'])} conditions from processed data")

# If we couldn't find conditions from processed data, fall back to config file
if not genome_usage_conditions:
    log_both(f"  Could not find conditions in processed data, using genome config...")
    with open(genome_config_path, 'r') as f:
        genome_config = json.load(f)

    # Extract usage conditions for each genome from usage_files
    if 'usage_files' in genome_config:
        # New format with usage_files
        for genome_id, usage_info in genome_config['usage_files'].items():
            if 'tissues' in usage_info and 'timepoints' in usage_info:
                # Build condition names
                conditions = []
                for tissue in usage_info['tissues']:
                    for timepoint in usage_info['timepoints']:
                        conditions.append(f"{tissue}_{timepoint}")
                genome_usage_conditions[genome_id] = conditions
            else:
                log_both(f"  WARNING: No tissues/timepoints for {genome_id}")
    elif 'genomes' in genome_config:
        # Try to extract from genomes list
        genomes = genome_config['genomes']
        if isinstance(genomes, list):
            for genome in genomes:
                if 'usage_conditions' in genome:
                    genome_usage_conditions[genome['genome_id']] = genome['usage_conditions']
    else:
        # Assume direct genome_id -> config mapping
        for genome_id, config in genome_config.items():
            if 'usage_conditions' in config:
                genome_usage_conditions[genome_id] = config['usage_conditions']
            else:
                log_both(f"  WARNING: No usage_conditions for {genome_id}")

if not genome_usage_conditions:
    log_both(f"  ERROR: Could not extract usage conditions")
    sys.exit(1)

log_both(f"  Loaded usage conditions for {len(genome_usage_conditions)} genomes:")
for genome_id, conditions in sorted(genome_usage_conditions.items()):
    log_both(f"    {genome_id}: {len(conditions)} conditions")

step1_time = time.time() - step1_start
log_both(f"  Completed in {step1_time:.2f} seconds\n")

# Helper function to build condition mask
def build_condition_mask(genome_conds, target_conds):
    """Build binary mask indicating which conditions are valid for this genome."""
    mask = np.zeros(len(target_conds), dtype=np.bool_)
    cond_to_idx = {cond: idx for idx, cond in enumerate(genome_conds)}
    for target_idx, target_cond in enumerate(target_conds):
        if target_cond in cond_to_idx:
            mask[target_idx] = True
    return mask

# Process each split (train and test)
for split_name in ['train', 'test']:
    split_path = os.path.join(split_dir, split_name)
    
    if not os.path.exists(split_path):
        log_both(f"\nSkipping {split_name}: directory not found")
        continue
    
    log_both(f"\nStep 2.{1 if split_name == 'train' else 2}: Processing {split_name} split...")
    step_start = time.time()
    
    # Check if mask already exists
    mask_path = os.path.join(split_path, 'condition_mask.mmap')
    if os.path.exists(mask_path):
        log_both(f"  WARNING: condition_mask.mmap already exists, skipping {split_name}")
        continue
    
    # Load metadata
    metadata_path = os.path.join(split_path, 'metadata.json')
    if not os.path.exists(metadata_path):
        log_both(f"  ERROR: metadata.json not found in {split_path}")
        continue
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Get number of sequences - handle different metadata formats
    if 'sequences' in metadata:
        n_sequences = metadata['sequences']['shape'][0]
    elif 'sequences_shape' in metadata:
        n_sequences = metadata['sequences_shape'][0]
    else:
        log_both(f"  ERROR: Could not find sequences shape in metadata")
        continue
        
    log_both(f"  Number of sequences: {n_sequences}")
    
    # Get usage condition info
    usage_conditions = metadata.get('usage_conditions', None)
    if usage_conditions is None:
        log_both(f"  ERROR: No usage_conditions in metadata")
        continue
    
    n_conditions = len(usage_conditions)
    log_both(f"  Number of conditions (union): {n_conditions}")
    log_both(f"  Conditions: {usage_conditions[:5]}...")
    
    # Load species_ids to map sequences to genomes
    species_ids_path = os.path.join(split_path, 'species_ids.mmap')
    if not os.path.exists(species_ids_path):
        log_both(f"  ERROR: species_ids.mmap not found")
        continue
    
    # Get species_ids metadata - handle different formats
    if 'species_ids' in metadata:
        species_dtype = np.dtype(metadata['species_ids']['dtype'])
        species_shape = tuple(metadata['species_ids']['shape'])
    elif 'species_ids_dtype' in metadata and 'species_ids_shape' in metadata:
        species_dtype = np.dtype(metadata['species_ids_dtype'])
        species_shape = tuple(metadata['species_ids_shape'])
    else:
        log_both(f"  ERROR: Could not find species_ids metadata")
        continue
        
    species_ids = np.memmap(species_ids_path, dtype=species_dtype, mode='r',
                           shape=species_shape)
    
    # Get unique genomes
    unique_genomes = np.unique(species_ids)
    
    # Get species mapping if available (maps short name -> ID)
    species_mapping = metadata.get('species_mapping', {})
    
    # Create reverse mapping (ID -> full genome_id)
    id_to_genome = {}
    if species_mapping:
        # species_mapping is like {'human': 0, 'mouse': 1, 'rat': 2}
        # We need to match these to genome IDs like human_GRCh37, mouse_GRCm38, etc.
        for short_name, species_id in species_mapping.items():
            # Find matching genome in config
            for genome_id in genome_usage_conditions.keys():
                if short_name in genome_id.lower():
                    id_to_genome[species_id] = genome_id
                    break
    
    log_both(f"  Unique genome IDs in split: {sorted(unique_genomes)}")
    log_both(f"  Species mapping: {species_mapping}")
    log_both(f"  ID to genome mapping: {id_to_genome}")
    
    # Create condition mask
    log_both(f"  Creating condition mask array ({n_sequences}, {n_conditions})...")
    condition_mask = np.memmap(mask_path, dtype=np.bool_, mode='w+',
                              shape=(n_sequences, n_conditions))
    
    # Process in batches to manage memory
    batch_size = 10000
    n_batches = (n_sequences + batch_size - 1) // batch_size
    log_both(f"  Processing {n_batches} batches of up to {batch_size} sequences...")
    
    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, n_sequences)
        batch_species = species_ids[start_idx:end_idx]
        
        # Build mask for each sequence using species ID
        for i, species_id in enumerate(batch_species):
            # species_id is an integer like 0, 1, 2
            genome_id = id_to_genome.get(int(species_id), None)
            
            if genome_id is None or genome_id not in genome_usage_conditions:
                if (batch_idx == 0 and i == 0):  # Log warning once only
                    log_both(f"  WARNING: No genome mapping for species_id {species_id}, setting all False")
                condition_mask[start_idx + i, :] = False
            else:
                genome_conds = genome_usage_conditions[genome_id]
                mask = build_condition_mask(genome_conds, usage_conditions)
                condition_mask[start_idx + i, :] = mask
        
        if (batch_idx + 1) % 10 == 0 or batch_idx == n_batches - 1:
            log_both(f"    Processed {end_idx}/{n_sequences} sequences...")
    
    # Flush to disk
    condition_mask.flush()
    del condition_mask
    
    # Verify and report statistics
    log_both(f"  Verifying mask...")
    condition_mask_verify = np.memmap(mask_path, dtype=np.bool_, mode='r',
                                      shape=(n_sequences, n_conditions))
    
    total_valid = condition_mask_verify.sum()
    avg_valid_per_seq = total_valid / n_sequences
    log_both(f"  Total valid conditions: {total_valid:,}")
    log_both(f"  Average valid per sequence: {avg_valid_per_seq:.1f}")
    
    # Report per-genome statistics
    for species_id in unique_genomes:
        genome_id = id_to_genome.get(int(species_id), f"Unknown({species_id})")
        genome_mask = species_ids[:] == species_id
        genome_seqs = genome_mask.sum()
        genome_valid = condition_mask_verify[genome_mask].sum()
        genome_avg = genome_valid / genome_seqs if genome_seqs > 0 else 0
        
        expected = len(genome_usage_conditions.get(genome_id, []))
        log_both(f"    {genome_id} (ID={species_id}): {genome_seqs:,} sequences, "
                f"{genome_avg:.1f} valid/seq (expected: {expected})")
    
    del condition_mask_verify
    
    # Update metadata.json
    log_both(f"  Updating metadata.json...")
    metadata['condition_mask'] = {
        'shape': [n_sequences, n_conditions],
        'dtype': 'bool',
        'description': 'Binary mask indicating valid conditions for each sequence'
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    step_time = time.time() - step_start
    log_both(f"  {split_name.capitalize()} split completed in {step_time:.2f} seconds")

# Summary
total_time = time.time() - script_start
log_both(f"\n{'='*60}")
log_both(f"Script completed successfully!")
log_both(f"Total time: {total_time:.2f} seconds")
log_both(f"Log file: {log_file}")
log_both(f"{'='*60}")
