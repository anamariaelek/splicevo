"""Split loaded data into train/test sets based on orthology."""

import time
import os
import numpy as np
from datetime import datetime
import json
import argparse
import sys
import pandas as pd
import pickle

from splicevo.data import MultiGenomeDataLoader

parser = argparse.ArgumentParser(description="Split loaded data into train/test sets")
parser.add_argument("--input_dir", type=str, required=True, help="Directory with loaded data")
parser.add_argument("--output_dir", type=str, required=True, help="Directory to save split results")
parser.add_argument("--n_cpus", type=int, default=8, help="Number of CPU cores to use")
parser.add_argument("--quiet", action='store_true', help="Suppress console output")
parser.add_argument("--pov_genome", type=str, default="human_GRCh37", 
                    help="Point-of-view genome for chromosome splitting")
parser.add_argument("--test_chromosomes", type=str, nargs='+', default=['1', '3', '5'], 
                    help="Test chromosomes for POV genome")
parser.add_argument("--window_size", type=int, default=1000, help="Window size for sequences")
parser.add_argument("--context_size", type=int, default=450, help="Context size on each side")
parser.add_argument("--alpha_threshold", type=int, default=5, help="Minimum alpha value threshold")
parser.add_argument("--sequential", action='store_true', help="Process train and test splits sequentially to reduce memory usage")
parser.add_argument("--process-by-genome", action='store_true', help="Process one genome at a time to reduce memory usage (most scalable)")
args = parser.parse_args()

input_dir = args.input_dir
output_dir = args.output_dir
n_cpus = args.n_cpus
pov_genome = args.pov_genome
test_chromosomes = args.test_chromosomes

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
log_print(f"Test chromosomes: {test_chromosomes}")
log_print("=" * 60)

# Step 1: Load intermediate data
log_print("Step 1: Loading intermediate data...")
step1_start = time.time()

loader_file = os.path.join(input_dir, "loader_state.pkl")
log_print(f"  Loading loader state from {loader_file}")
with open(loader_file, 'rb') as f:
    loader_state = pickle.load(f)

# Reconstruct loader
log_print("  Reconstructing loader...")
loader = MultiGenomeDataLoader()
loader.loaded_data = loader_state['loaded_data']
loader.usage_conditions = loader_state['usage_conditions']
loader.genome_to_species = loader_state['genome_to_species']
loader.species_to_id = loader_state['species_to_id']
loader.orthology_table = loader_state['orthology_table']

# Reconstruct genomes dict (paths only, don't load actual genomes yet)
from splicevo.io.genome import GenomeData
from pathlib import Path
for gid, gdata in loader_state['genomes'].items():
    loader.genomes[gid] = GenomeData(
        genome_id=gdata['genome_id'],
        genome_path=Path(gdata['genome_path']),
        gtf_path=Path(gdata['gtf_path']),
        chromosomes=gdata['chromosomes'],
        metadata=gdata['metadata']
    )

log_print(f"  Loaded {len(loader.loaded_data)} splice sites")
log_print(f"  Loaded {len(loader.genomes)} genomes")

step1_time = time.time() - step1_start
log_print(f"  Intermediate data loaded in {step1_time:.2f} seconds")
log_print("")

# Step 2: Determine train/test gene split
log_print("Step 2: Determining train/test split based on orthology...")
step2_start = time.time()

df = loader.get_dataframe()
ortho_df = loader.orthology_table

# Add diagnostic logging for loaded data
log_print(f"\n  Loaded data breakdown:")
for genome_id in df['genome_id'].unique():
    genome_df = df[df['genome_id'] == genome_id]
    log_print(f"    {genome_id}: {len(genome_df)} sites, {genome_df['gene_id'].nunique()} genes")
    log_print(f"      Chromosomes: {sorted(genome_df['chromosome'].unique())}")

if ortho_df is None:
    log_print("WARNING: No orthology table loaded. Cannot split by orthology.")
    log_print("Falling back to chromosome-based split for POV genome only.")
    
    df['split'] = 'train'
    pov_mask = df['genome_id'] == pov_genome
    test_chr_mask = df['chromosome'].isin(test_chromosomes)
    df.loc[pov_mask & test_chr_mask, 'split'] = 'test'
    
    test_genes = set(df[df['split'] == 'test']['gene_id'].unique())
    train_genes = set(df[df['split'] == 'train']['gene_id'].unique())
    test_ortholog_groups = []
    
else:
    # Find genes on test chromosomes in POV genome
    pov_test_genes = df[
        (df['genome_id'] == pov_genome) & 
        (df['chromosome'].isin(test_chromosomes))
    ]['gene_id'].unique()
    
    log_print(f"\n  POV genome ({pov_genome}) test chromosomes: {test_chromosomes}")
    log_print(f"  Found {len(pov_test_genes)} genes on test chromosomes in POV genome")
    
    # Find ortholog groups for these test genes
    test_ortholog_groups = ortho_df[
        (ortho_df['genome_id'] == pov_genome) & 
        (ortho_df['gene_id'].isin(pov_test_genes))
    ]['ortholog_group'].unique()
    
    log_print(f"  These genes belong to {len(test_ortholog_groups)} ortholog groups")
    
    # Get all genes in these ortholog groups (across all species)
    test_genes_all_species = ortho_df[
        ortho_df['ortholog_group'].isin(test_ortholog_groups)
    ]['gene_id'].unique()
    
    log_print(f"  Total genes in test ortholog groups (all species): {len(test_genes_all_species)}")
    
    # Diagnostic: show distribution across genomes in orthology
    log_print(f"\n  Test genes by genome (from orthology):")
    test_ortho_by_genome = ortho_df[ortho_df['ortholog_group'].isin(test_ortholog_groups)]
    for genome_id in test_ortho_by_genome['genome_id'].unique():
        n_genes = test_ortho_by_genome[test_ortho_by_genome['genome_id'] == genome_id]['gene_id'].nunique()
        log_print(f"    {genome_id}: {n_genes} genes")
    
    # Get genes actually present in loaded data
    loaded_gene_ids = set(df['gene_id'].unique())
    
    test_genes = set(test_genes_all_species) & loaded_gene_ids
    train_genes = loaded_gene_ids - test_genes
    
    # Diagnostic: show which genes were filtered out
    filtered_out = set(test_genes_all_species) - test_genes
    if filtered_out:
        log_print(f"\n  Warning: {len(filtered_out)} test genes from orthology not found in loaded data")
        # Show breakdown by genome
        filtered_by_genome = ortho_df[ortho_df['gene_id'].isin(filtered_out)]
        for genome_id in filtered_by_genome['genome_id'].unique():
            n_filtered = filtered_by_genome[filtered_by_genome['genome_id'] == genome_id]['gene_id'].nunique()
            log_print(f"    {genome_id}: {n_filtered} genes filtered out")
    
    log_print(f"\n  Train genes (in loaded data): {len(train_genes)}")
    log_print(f"  Test genes (in loaded data): {len(test_genes)}")
    
    # Diagnostic: show test genes by genome
    log_print(f"\n  Test genes by genome (in loaded data):")
    for genome_id in df['genome_id'].unique():
        genome_test_genes = df[(df['genome_id'] == genome_id) & (df['gene_id'].isin(test_genes))]['gene_id'].unique()
        log_print(f"    {genome_id}: {len(genome_test_genes)} genes")
    
    df['split'] = df['gene_id'].apply(lambda x: 'test' if x in test_genes else 'train')

# Count splice sites in each split
train_sites = df[df['split'] == 'train']
test_sites = df[df['split'] == 'test']

log_print(f"\nSplit statistics:")
log_print(f"  Train: {len(train_sites)} splice sites from {len(train_genes)} genes")
log_print(f"  Test: {len(test_sites)} splice sites from {len(test_genes)} genes")

# Show distribution by genome and site type
for split_name, split_df in [('Train', train_sites), ('Test', test_sites)]:
    log_print(f"\n  {split_name} split by genome:")
    for genome_id in split_df['genome_id'].unique():
        genome_sites = split_df[split_df['genome_id'] == genome_id]
        n_donors = (genome_sites['site_type'] == 1).sum()
        n_acceptors = (genome_sites['site_type'] == 2).sum()
        log_print(f"    {genome_id}: {len(genome_sites)} sites ({n_donors} donors, {n_acceptors} acceptors)")

step2_time = time.time() - step2_start
log_print(f"  \nGTrain/test split determined in {step2_time:.2f} seconds")
log_print("")

# Step 3: Convert to arrays for each split
log_print("Step 3: Creating train and test datasets...")
step3_start = time.time()

original_loaded_data = loader.loaded_data

if args.sequential:
    log_print("  Using sequential processing mode (lower memory usage)")
if args.process_by_genome:
    log_print("  Using genome-by-genome processing mode (most scalable)")

for split in ['train', 'test']:
    log_print(f"\n  Processing {split} split...")
    split_start = time.time()
    
    split_genes = train_genes if split == 'train' else test_genes
    
    # Filter loaded_data
    split_sites = [
        site for site in original_loaded_data 
        if site.gene_id in split_genes
    ]
    
    log_print(f"    Filtered to {len(split_sites)} splice sites")
    
    mmap_dir = os.path.join(output_dir, f'memmap_{split}')
    
    if args.process_by_genome:
        # Process genome by genome to minimize memory usage
        log_print(f"    Processing genome-by-genome...")
        
        # Group sites by genome
        sites_by_genome = {}
        for site in split_sites:
            if site.genome_id not in sites_by_genome:
                sites_by_genome[site.genome_id] = []
            sites_by_genome[site.genome_id].append(site)
        
        log_print(f"    Found {len(sites_by_genome)} genomes to process")
        for gid, sites in sites_by_genome.items():
            log_print(f"      {gid}: {len(sites)} sites")
        
        all_metadata = []
        total_windows = 0
        first_genome = True
        
        for genome_idx, (genome_id, genome_sites) in enumerate(sites_by_genome.items(), 1):
            genome_time_start = time.time()
            log_print(f"\n    [{genome_idx}/{len(sites_by_genome)}] Processing {genome_id} ({len(genome_sites)} sites)...")
            
            # Set loader to this genome's sites only
            loader.loaded_data = genome_sites
            
            # Convert to arrays for this genome
            sequences, labels, usage_arrays, metadata, species_ids = loader.to_arrays(
                window_size=args.window_size,
                context_size=args.context_size,
                alpha_threshold=args.alpha_threshold,
                n_workers=n_cpus,
                save_memmap=None  # We'll handle memmap ourselves
            )
            
            n_windows = len(sequences)
            log_print(f"        Created {n_windows} windows from {genome_id}")
            
            if first_genome:
                # First genome: create new memmap files
                os.makedirs(mmap_dir, exist_ok=True)
                
                # Save initial arrays as memmap
                seq_shape = (n_windows,) + sequences.shape[1:]
                label_shape = (n_windows,) + labels.shape[1:]
                species_shape = (n_windows,)
                
                seq_mmap = np.lib.format.open_memmap(
                    os.path.join(mmap_dir, 'sequences.npy'),
                    mode='w+',
                    dtype=sequences.dtype,
                    shape=seq_shape
                )
                seq_mmap[:] = sequences
                del seq_mmap
                
                label_mmap = np.lib.format.open_memmap(
                    os.path.join(mmap_dir, 'labels.npy'),
                    mode='w+',
                    dtype=labels.dtype,
                    shape=label_shape
                )
                label_mmap[:] = labels
                del label_mmap
                
                species_mmap = np.lib.format.open_memmap(
                    os.path.join(mmap_dir, 'species_ids.npy'),
                    mode='w+',
                    dtype=species_ids.dtype,
                    shape=species_shape
                )
                species_mmap[:] = species_ids
                del species_mmap
                
                # Save usage arrays
                for i, usage_array in enumerate(usage_arrays):
                    usage_shape = (n_windows,) + usage_array.shape[1:]
                    usage_mmap = np.lib.format.open_memmap(
                        os.path.join(mmap_dir, f'usage_{i}.npy'),
                        mode='w+',
                        dtype=usage_array.dtype,
                        shape=usage_shape
                    )
                    usage_mmap[:] = usage_array
                    del usage_mmap
                
                total_windows = n_windows
                first_genome = False
                
            else:
                # Subsequent genomes: append to existing memmap files
                new_total = total_windows + n_windows
                
                # Resize and append sequences
                seq_mmap = np.lib.format.open_memmap(
                    os.path.join(mmap_dir, 'sequences.npy'),
                    mode='r+'
                )
                seq_mmap_resized = np.lib.format.open_memmap(
                    os.path.join(mmap_dir, 'sequences_temp.npy'),
                    mode='w+',
                    dtype=seq_mmap.dtype,
                    shape=(new_total,) + seq_mmap.shape[1:]
                )
                seq_mmap_resized[:total_windows] = seq_mmap[:]
                seq_mmap_resized[total_windows:new_total] = sequences
                del seq_mmap, seq_mmap_resized
                os.replace(os.path.join(mmap_dir, 'sequences_temp.npy'), 
                          os.path.join(mmap_dir, 'sequences.npy'))
                
                # Resize and append labels
                label_mmap = np.lib.format.open_memmap(
                    os.path.join(mmap_dir, 'labels.npy'),
                    mode='r+'
                )
                label_mmap_resized = np.lib.format.open_memmap(
                    os.path.join(mmap_dir, 'labels_temp.npy'),
                    mode='w+',
                    dtype=label_mmap.dtype,
                    shape=(new_total,) + label_mmap.shape[1:]
                )
                label_mmap_resized[:total_windows] = label_mmap[:]
                label_mmap_resized[total_windows:new_total] = labels
                del label_mmap, label_mmap_resized
                os.replace(os.path.join(mmap_dir, 'labels_temp.npy'), 
                          os.path.join(mmap_dir, 'labels.npy'))
                
                # Resize and append species IDs
                species_mmap = np.lib.format.open_memmap(
                    os.path.join(mmap_dir, 'species_ids.npy'),
                    mode='r+'
                )
                species_mmap_resized = np.lib.format.open_memmap(
                    os.path.join(mmap_dir, 'species_ids_temp.npy'),
                    mode='w+',
                    dtype=species_mmap.dtype,
                    shape=(new_total,)
                )
                species_mmap_resized[:total_windows] = species_mmap[:]
                species_mmap_resized[total_windows:new_total] = species_ids
                del species_mmap, species_mmap_resized
                os.replace(os.path.join(mmap_dir, 'species_ids_temp.npy'), 
                          os.path.join(mmap_dir, 'species_ids.npy'))
                
                # Resize and append usage arrays
                for i, usage_array in enumerate(usage_arrays):
                    usage_mmap = np.lib.format.open_memmap(
                        os.path.join(mmap_dir, f'usage_{i}.npy'),
                        mode='r+'
                    )
                    usage_mmap_resized = np.lib.format.open_memmap(
                        os.path.join(mmap_dir, f'usage_{i}_temp.npy'),
                        mode='w+',
                        dtype=usage_mmap.dtype,
                        shape=(new_total,) + usage_mmap.shape[1:]
                    )
                    usage_mmap_resized[:total_windows] = usage_mmap[:]
                    usage_mmap_resized[total_windows:new_total] = usage_array
                    del usage_mmap, usage_mmap_resized
                    os.replace(os.path.join(mmap_dir, f'usage_{i}_temp.npy'), 
                              os.path.join(mmap_dir, f'usage_{i}.npy'))
                
                total_windows = new_total
            
            all_metadata.append(metadata)
            
            genome_time = time.time() - genome_time_start
            log_print(f"        {genome_id} processed in {genome_time:.2f} seconds")
            
            # Clear memory after each genome
            del sequences, labels, usage_arrays, species_ids, metadata
            import gc
            gc.collect()
        
        # Restore loader
        loader.loaded_data = split_sites
        
        # Combine all metadata
        metadata = pd.concat(all_metadata, ignore_index=True)
        
        log_print(f"\n    Total windows created: {total_windows}")
        log_print(f"    Combined metadata rows: {len(metadata)}")
        
        # Reload final arrays for info display
        sequences = np.load(os.path.join(mmap_dir, 'sequences.npy'), mmap_mode='r')
        labels = np.load(os.path.join(mmap_dir, 'labels.npy'), mmap_mode='r')
        species_ids = np.load(os.path.join(mmap_dir, 'species_ids.npy'), mmap_mode='r')
        usage_arrays = [np.load(os.path.join(mmap_dir, f'usage_{i}.npy'), mmap_mode='r') 
                       for i in range(len(loader.usage_conditions))]
        
    else:
        # Original processing: all genomes at once
        loader.loaded_data = split_sites
        
        sequences, labels, usage_arrays, metadata, species_ids = loader.to_arrays(
            window_size=args.window_size,
            context_size=args.context_size,
            alpha_threshold=args.alpha_threshold,
            n_workers=n_cpus,
            save_memmap=mmap_dir
        )
    
    log_print(f"    Created {len(sequences)} windows")
    log_print(f"    Sequences shape: {sequences.shape}")
    log_print(f"    Labels shape: {labels.shape}")
    log_print(f"    Species IDs shape: {species_ids.shape}")
    
    # Save metadata
    metadata.to_csv(
        os.path.join(output_dir, f"metadata_{split}.csv.gz"), 
        index=False, 
        compression='gzip'
    )
    
    # Save species mapping
    species_info = {
        'species_mapping': loader.species_to_id,
        'genome_to_species': loader.genome_to_species
    }
    with open(os.path.join(output_dir, f"species_info_{split}.json"), 'w') as f:
        json.dump(species_info, f, indent=2)
    
    # Save usage info
    usage_info = loader.get_usage_array_info(usage_arrays)
    with open(os.path.join(output_dir, f"usage_info_{split}.json"), 'w') as f:
        json.dump(usage_info, f, indent=2, default=str)
    
    usage_summary = loader.get_usage_summary()
    usage_summary.to_csv(
        os.path.join(output_dir, f"usage_summary_{split}.csv"), 
        index=False
    )
    
    split_time = time.time() - split_start
    log_print(f"    {split.capitalize()} split completed in {split_time:.2f} seconds")
    
    # Clear memory if sequential mode
    if args.sequential or args.process_by_genome:
        log_print(f"    Clearing memory after {split} split...")
        del sequences, labels, usage_arrays, metadata, species_ids
        import gc
        gc.collect()
        log_print(f"    Memory cleared")

loader.loaded_data = original_loaded_data

step3_time = time.time() - step3_start
log_print(f"\n  All splits created in {step3_time:.2f} seconds")
log_print("")

# Step 4: Save gene lists and split info
log_print("Step 4: Saving gene lists...")
step4_start = time.time()

log_print(f"  Saving {len(train_genes)} train genes")
log_print(f"  Saving {len(test_genes)} test genes")

with open(os.path.join(output_dir, "train_genes.txt"), 'w') as f:
    for gene in sorted(train_genes):
        f.write(f"{gene}\n")

with open(os.path.join(output_dir, "test_genes.txt"), 'w') as f:
    for gene in sorted(test_genes):
        f.write(f"{gene}\n")

split_info = {
    'pov_genome': pov_genome,
    'test_chromosomes': test_chromosomes,
    'n_train_genes': len(train_genes),
    'n_test_genes': len(test_genes),
    'n_test_ortholog_groups': len(test_ortholog_groups),
    'orthology_based_split': ortho_df is not None,
    'window_size': args.window_size,
    'context_size': args.context_size,
    'alpha_threshold': args.alpha_threshold
}

with open(os.path.join(output_dir, "split_info.json"), 'w') as f:
    json.dump(split_info, f, indent=2)

step4_time = time.time() - step4_start
log_print(f"  Gene lists saved in {step4_time:.2f} seconds")

# Summary timing report
total_time = time.time() - script_start_time
log_print("")
log_print("=" * 60)
log_print("Timing Summary")
log_print("=" * 60)
log_print(f"Step 1 - Load intermediate:     {format_time(step1_time):>12} ({100*step1_time/total_time:5.1f}%)")
log_print(f"Step 2 - Determine split:       {format_time(step2_time):>12} ({100*step2_time/total_time:5.1f}%)")
log_print(f"Step 3 - Create datasets:       {format_time(step3_time):>12} ({100*step3_time/total_time:5.1f}%)")
log_print(f"Step 4 - Save gene lists:       {format_time(step4_time):>12} ({100*step4_time/total_time:5.1f}%)")
log_print("-" * 60)
log_print(f"Total time:                     {format_time(total_time):>12}")
log_print(f"\nSplitting completed at {datetime.now()}")
log_print("=" * 60)
log_print(f"Results saved to: {os.path.abspath(output_dir)}")

if args.quiet:
    sys.stdout = original_stdout
    sys.stderr = original_stderr
    log_file_handle.close()
    print(f"Splitting complete. Log saved to: {log_file}")
