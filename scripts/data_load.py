"""Load splice site data from multiple genomes and save as intermediate format."""

import time
import os
import numpy as np
from datetime import datetime
import json
import argparse
import sys
import pickle
import pandas as pd
from pathlib import Path

from splicevo.data import MultiGenomeDataLoader

parser = argparse.ArgumentParser(description="Load splicing data from multiple genomes")
parser.add_argument("--output_dir", type=str, required=True, help="Directory to save results")
parser.add_argument("--config", type=str, required=True, help="Path to genome configuration JSON file")
parser.add_argument("--n_cpus", type=int, default=8, help="Number of CPU cores to use")
parser.add_argument("--quiet", action='store_true', help="Suppress console output")
parser.add_argument("--append", action='store_true', help="Append to existing data instead of starting fresh")
parser.add_argument("--overwrite", type=str, nargs='+', help="Genome IDs to overwrite (re-load even if already present)")
parser.add_argument("--sequential", action='store_true', help="Load genomes sequentially (one at a time) to reduce memory usage")
args = parser.parse_args()

output_dir = args.output_dir
n_cpus = args.n_cpus

# Start timing
script_start_time = time.time()

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Setup logging
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_file = os.path.join(output_dir, f'data_load_{timestamp}.txt')

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
log_print("=" * 60)

# Load configuration
log_print("Loading genome configuration...")
with open(args.config, 'r') as f:
    config = json.load(f)

orthology_file = config.get('orthology_file', None)
genomes_config = config.get('genomes', [])
usage_files_config = config.get('usage_files', {})

# Step 1: Initialize or load existing loader
log_print("Step 1: Initializing MultiGenomeDataLoader...")
step1_start = time.time()

loader_file = os.path.join(output_dir, "loader_state.pkl")
existing_genomes = set()
existing_orthology_sources = set()
genomes_to_overwrite = set(args.overwrite) if args.overwrite else set()

# CHANGED: Load existing state if --overwrite is specified OR --append is specified
if (args.append or genomes_to_overwrite) and os.path.exists(loader_file):
    log_print(f"  Loading existing loader state from {loader_file}")
    with open(loader_file, 'rb') as f:
        loader_state = pickle.load(f)
    
    # Initialize loader WITHOUT orthology file first
    loader = MultiGenomeDataLoader(orthology_file=None)
    
    # Filter out genomes to be overwritten
    if genomes_to_overwrite:
        log_print(f"  Will overwrite genomes: {', '.join(genomes_to_overwrite)}")
        
        # Count splice sites per genome before removal
        sites_before = {}
        for site in loader_state['loaded_data']:
            sites_before[site.genome_id] = sites_before.get(site.genome_id, 0) + 1
        
        log_print(f"  Splice sites before overwrite: {sites_before}")
        
        # Remove splice sites from genomes to be overwritten
        loader.loaded_data = [
            site for site in loader_state['loaded_data'] 
            if site.genome_id not in genomes_to_overwrite
        ]
        
        # Count after removal
        sites_after = {}
        for site in loader.loaded_data:
            sites_after[site.genome_id] = sites_after.get(site.genome_id, 0) + 1
        
        removed_count = len(loader_state['loaded_data']) - len(loader.loaded_data)
        log_print(f"  Removed {removed_count} splice sites from genomes to overwrite")
        log_print(f"  Splice sites after removal: {sites_after}")
    else:
        loader.loaded_data = loader_state['loaded_data']
    
    # Load usage_data but exclude overwritten genomes
    loader.usage_data = {}
    if 'usage_data' in loader_state:
        for gid, data in loader_state['usage_data'].items():
            if gid not in genomes_to_overwrite:
                loader.usage_data[gid] = data
        if genomes_to_overwrite:
            log_print(f"  Cleared usage data for {len(genomes_to_overwrite)} overwritten genome(s)")
    
    loader.usage_conditions = loader_state.get('usage_conditions', [])
    loader.genome_to_species = loader_state.get('genome_to_species', {})
    loader.species_to_id = loader_state.get('species_to_id', {})
    
    # Handle orthology table merging
    existing_ortho = loader_state.get('orthology_table', None)
    existing_orthology_sources = set(loader_state.get('orthology_sources', []))
    
    if orthology_file is not None:
        # Get orthology file identifier (basename without extension)
        ortho_file_id = Path(orthology_file).stem
        
        if ortho_file_id not in existing_orthology_sources:
            log_print(f"  Merging new orthology file: {orthology_file}")
            
            # Load new orthology file
            new_ortho = pd.read_csv(orthology_file, sep='\t', header=0)
            new_ortho = new_ortho[['ortholog_group', 'gene_id', 'genome_id']].copy()
            new_ortho = new_ortho.dropna()
            new_ortho = new_ortho.drop_duplicates(subset=['gene_id', 'genome_id'])
            
            # Add file identifier to ortholog groups to avoid conflicts
            new_ortho['ortholog_group'] = ortho_file_id + '_' + new_ortho['ortholog_group'].astype(str)
            
            # Ensure correct data types
            new_ortho['gene_id'] = new_ortho['gene_id'].astype(str)
            new_ortho['ortholog_group'] = new_ortho['ortholog_group'].astype(str)
            new_ortho['genome_id'] = new_ortho['genome_id'].astype(str)
            
            if existing_ortho is not None and len(existing_ortho) > 0:
                # Merge with existing orthology table
                loader.orthology_table = pd.concat([existing_ortho, new_ortho], ignore_index=True)
                loader.orthology_table = loader.orthology_table.drop_duplicates(subset=['gene_id', 'genome_id'])
                
                log_print(f"    Added {len(new_ortho)} new orthology entries")
                log_print(f"    Total orthology entries: {len(loader.orthology_table)}")
                log_print(f"    Total ortholog groups: {loader.orthology_table['ortholog_group'].nunique()}")
            else:
                # No existing orthology, use new one
                loader.orthology_table = new_ortho
                log_print(f"    Loaded {len(new_ortho)} orthology entries")
            
            # Track this orthology source
            existing_orthology_sources.add(ortho_file_id)
        else:
            log_print(f"  Orthology file {orthology_file} already loaded, skipping")
            loader.orthology_table = existing_ortho
    else:
        # No new orthology file, keep existing
        loader.orthology_table = existing_ortho
        log_print(f"  Keeping existing orthology table with {len(existing_ortho) if existing_ortho is not None else 0} entries")
    
    from splicevo.io.genome import GenomeData
    from pathlib import Path
    for gid, gdata in loader_state['genomes'].items():
        # Skip genomes that will be overwritten
        if gid in genomes_to_overwrite:
            log_print(f"  Skipping {gid} from existing state (will be overwritten)")
            continue
            
        loader.genomes[gid] = GenomeData(
            genome_id=gdata['genome_id'],
            genome_path=Path(gdata['genome_path']),
            gtf_path=Path(gdata['gtf_path']),
            chromosomes=gdata['chromosomes'],
            metadata=gdata['metadata']
        )
        existing_genomes.add(gid)
    
    log_print(f"  Loaded existing data with {len(loader.loaded_data)} splice sites from {len(existing_genomes)} genomes")
    if len(existing_genomes) > 0:
        log_print(f"  Existing genomes: {', '.join(existing_genomes)}")
else:
    log_print("  Creating new loader")
    loader = MultiGenomeDataLoader(orthology_file=None)
    
    # Load orthology file with file identifier prefix
    if orthology_file is not None:
        log_print(f"  Loading orthology file: {orthology_file}")
        ortho_file_id = Path(orthology_file).stem
        
        # Load orthology manually to add prefix
        ortho_df = pd.read_csv(orthology_file, sep='\t', header=0)
        ortho_df = ortho_df[['ortholog_group', 'gene_id', 'genome_id']].copy()
        ortho_df = ortho_df.dropna()
        ortho_df = ortho_df.drop_duplicates(subset=['gene_id', 'genome_id'])
        
        # Add file identifier prefix to ortholog groups
        ortho_df['ortholog_group'] = ortho_file_id + '_' + ortho_df['ortholog_group'].astype(str)
        
        # Ensure correct data types
        ortho_df['gene_id'] = ortho_df['gene_id'].astype(str)
        ortho_df['ortholog_group'] = ortho_df['ortholog_group'].astype(str)
        ortho_df['genome_id'] = ortho_df['genome_id'].astype(str)
        
        loader.orthology_table = ortho_df
        existing_orthology_sources.add(ortho_file_id)
        
        log_print(f"    Loaded {len(ortho_df)} orthology entries from {ortho_df['genome_id'].nunique()} genomes")
        log_print(f"    Ortholog groups: {ortho_df['ortholog_group'].nunique()}")

step1_time = time.time() - step1_start
log_print(f"  Loader initialized in {step1_time:.2f} seconds")
log_print("")

# Step 2: Add genomes from config
log_print("Step 2: Adding genomes from configuration")
step2_start = time.time()

genome_times = {}
for genome_config in genomes_config:
    genome_id = genome_config['genome_id']
    
    # Skip if already loaded AND not in overwrite list
    if genome_id in existing_genomes:
        log_print(f"  Skipping {genome_id} (already loaded)")
        continue
    
    # Check if this is an overwrite operation
    is_overwrite = genome_id in genomes_to_overwrite
    
    genome_start = time.time()
    if is_overwrite:
        log_print(f"  Overwriting {genome_id}")
    else:
        log_print(f"  Adding {genome_id}")
    
    loader.add_genome(
        genome_id=genome_id,
        genome_path=genome_config['genome_path'],
        gtf_path=genome_config['gtf_path'],
        chromosomes=genome_config.get('chromosomes', None),
        metadata=genome_config.get('metadata', {}),
        common_name=genome_config.get('common_name', None)  # CHANGED: species -> common_name
    )
    
    genome_time = time.time() - genome_start
    genome_times[genome_id] = genome_time
    log_print(f"  ✓ {genome_id} added in {genome_time:.2f} seconds")

step2_time = time.time() - step2_start
log_print(f"✓ Genomes processed in {step2_time:.2f} seconds")
log_print("")

# Step 3: Add usage files from config
log_print("Step 3: Adding usage files from configuration...")
step3_start = time.time()

usage_times = {}
for genome_id, usage_config in usage_files_config.items():
    # Skip if genome not in loader
    if genome_id not in loader.genomes:
        log_print(f"  Skipping usage files for {genome_id} (genome not loaded)")
        continue
    
    # Skip if already loaded AND not being overwritten
    if genome_id in existing_genomes:
        log_print(f"  Skipping usage files for {genome_id} (already loaded)")
        continue
    
    # Check if this is an overwrite operation
    is_overwrite = genome_id in genomes_to_overwrite
    
    usage_start = time.time()
    if is_overwrite:
        log_print(f"  Reloading usage files for {genome_id} (overwrite mode)...")
    else:
        log_print(f"  Adding usage files for {genome_id}...")
    
    usage_count = 0
    usage_pattern = usage_config.get('pattern', None)
    usage_list = usage_config.get('files', [])
    
    if usage_pattern:
        # Use pattern-based loading
        tissues = usage_config.get('tissues', [])
        timepoints = usage_config.get('timepoints', [])
        
        for tissue in tissues:
            for timepoint in timepoints:
                try:
                    usage_file = usage_pattern.format(tissue=tissue, timepoint=timepoint)
                    loader.add_usage_file(
                        genome_id=genome_id,
                        usage_file=usage_file,
                        tissue=tissue,
                        timepoint=str(timepoint)
                    )
                    usage_count += 1
                except FileNotFoundError:
                    log_print(f"    Warning: {usage_file} not found")
                    pass
    else:
        # Use explicit file list
        for usage_entry in usage_list:
            try:
                loader.add_usage_file(
                    genome_id=genome_id,
                    usage_file=usage_entry['file'],
                    tissue=usage_entry['tissue'],
                    timepoint=usage_entry.get('timepoint', None)
                )
                usage_count += 1
            except FileNotFoundError:
                log_print(f"    Warning: {usage_entry['file']} not found")
    
    usage_time = time.time() - usage_start
    usage_times[genome_id] = usage_time
    log_print(f"  ✓ {usage_count} usage files added for {genome_id} in {usage_time:.2f} seconds")

step3_time = time.time() - step3_start
log_print(f"✓ Usage files processed in {step3_time:.2f} seconds")

conditions_df = loader.get_available_conditions()
log_print(f"  Available conditions: {len(conditions_df)}")
log_print("")

# Step 4: Load genome data (only for new genomes)
log_print("Step 4: Loading genome data...")
step4_start = time.time()

# New genomes are those not in existing_genomes (which already excludes overwritten ones)
new_genomes = [gid for gid in loader.genomes.keys() if gid not in existing_genomes]

if new_genomes:
    log_print(f"  Loading data for {len(new_genomes)} genome(s): {', '.join(new_genomes)}")
    
    # Store current loaded data count
    prev_count = len(loader.loaded_data)
    
    if args.sequential:
        # Load genomes one at a time to reduce memory usage
        log_print(f"  Using sequential loading mode (lower memory usage)")
        all_genomes = loader.genomes.copy()
        
        for i, genome_id in enumerate(new_genomes, 1):
            genome_load_start = time.time()
            log_print(f"  [{i}/{len(new_genomes)}] Loading {genome_id}...")
            
            # Temporarily set to single genome
            loader.genomes = {genome_id: all_genomes[genome_id]}
            
            # Load data for this genome only
            loader.load_all_genomes_data()
            
            genome_load_time = time.time() - genome_load_start
            current_count = len(loader.loaded_data) - prev_count
            log_print(f"      ✓ {genome_id} loaded {current_count} sites in {genome_load_time:.2f} seconds")
            prev_count = len(loader.loaded_data)
        
        # Restore all genomes
        loader.genomes = all_genomes
    else:
        # Parallel loading
        # Temporarily filter to only new genomes
        all_genomes = loader.genomes.copy()
        loader.genomes = {gid: all_genomes[gid] for gid in new_genomes}
        
        # Load only new data
        loader.load_all_genomes_data()
        
        # Restore all genomes
        loader.genomes = all_genomes
    
    new_count = len(loader.loaded_data) - prev_count
    log_print(f"  Loaded {new_count} new splice sites")
else:
    log_print("  No new genomes to load")

step4_time = time.time() - step4_start
log_print(f"✓ Genome data loaded in {step4_time:.2f} seconds")

summary = loader.get_summary()
log_print("\nData summary:")
log_print(str(summary))
log_print("")

# Step 5: Save intermediate data
log_print("Step 5: Saving intermediate data...")
step5_start = time.time()

# Count splice sites per genome before saving
sites_per_genome = {}
for site in loader.loaded_data:
    sites_per_genome[site.genome_id] = sites_per_genome.get(site.genome_id, 0) + 1

log_print(f"  Splice sites per genome to be saved:")
for gid, count in sorted(sites_per_genome.items()):
    log_print(f"    {gid}: {count} sites")
log_print(f"  Total: {len(loader.loaded_data)} splice sites")

# Save the entire loader object
loader_file = os.path.join(output_dir, "loader_state.pkl")
log_print(f"  Saving loader state to {loader_file}")
with open(loader_file, 'wb') as f:
    pickle.dump({
        'loaded_data': loader.loaded_data,
        'genomes': {gid: {'genome_id': gdata.genome_id, 
                         'genome_path': str(gdata.genome_path),
                         'gtf_path': str(gdata.gtf_path),
                         'chromosomes': gdata.chromosomes,
                         'metadata': gdata.metadata}
                   for gid, gdata in loader.genomes.items()},
        'usage_data': loader.usage_data,
        'usage_conditions': loader.usage_conditions,
        'genome_to_species': loader.genome_to_species,
        'species_to_id': loader.species_to_id,
        'orthology_table': loader.orthology_table,
        'orthology_sources': list(existing_orthology_sources)
    }, f)

# Save dataframe of splice sites
df = loader.get_dataframe()
log_print(f"  DataFrame has {len(df)} rows")
log_print(f"  DataFrame genomes: {df['genome_id'].unique().tolist() if len(df) > 0 else 'empty'}")

df_file = os.path.join(output_dir, "splice_sites.csv.gz")
log_print(f"  Saving splice sites dataframe to {df_file}")
df.to_csv(df_file, index=False, compression='gzip')

# Save usage summary
usage_summary = loader.get_usage_summary()
usage_summary.to_csv(
    os.path.join(output_dir, "usage_summary.csv"), 
    index=False
)

# Save species info
species_info = {
    'species_mapping': loader.species_to_id,
    'genome_to_species': loader.genome_to_species,
    'n_genomes': len(loader.genomes),
    'genomes': list(loader.genomes.keys())
}
with open(os.path.join(output_dir, "species_info.json"), 'w') as f:
    json.dump(species_info, f, indent=2)

# Save conditions info
conditions_info = {
    'n_conditions': len(loader.usage_conditions),
    'conditions': loader.usage_conditions
}
with open(os.path.join(output_dir, "conditions_info.json"), 'w') as f:
    json.dump(conditions_info, f, indent=2)

step5_time = time.time() - step5_start
log_print(f"✓ Intermediate data saved in {step5_time:.2f} seconds")

# Summary timing report
total_time = time.time() - script_start_time
log_print("")
log_print("=" * 60)
log_print("Timing Summary")
log_print("=" * 60)
log_print(f"Step 1 - Initialize loader:     {format_time(step1_time):>12} ({100*step1_time/total_time:5.1f}%)")
log_print(f"Step 2 - Add genomes:           {format_time(step2_time):>12} ({100*step2_time/total_time:5.1f}%)")
log_print(f"Step 3 - Add usage files:       {format_time(step3_time):>12} ({100*step3_time/total_time:5.1f}%)")
log_print(f"Step 4 - Load genome data:      {format_time(step4_time):>12} ({100*step4_time/total_time:5.1f}%)")
log_print(f"Step 5 - Save intermediate data: {format_time(step5_time):>12} ({100*step5_time/total_time:5.1f}%)")
log_print("-" * 60)
log_print(f"Total time:                     {format_time(total_time):>12} (100.0%)")
log_print("=" * 60)
log_print(f"Data loading completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
if args.quiet:
    sys.stdout = original_stdout
    sys.stderr = original_stderr
    log_file_handle.close()
    print(f"Data loading complete. Log saved to: {log_file}")
    