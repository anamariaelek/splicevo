import time
import pickle
import os
import numpy as np
from datetime import datetime
import json
import argparse
import sys

from splicevo.data import MultiGenomeDataLoader

parser = argparse.ArgumentParser(description="Process splicing data with MultiGenomeDataLoader")
parser.add_argument("--group", choices=["train", "test", "subset"], required=True, help="Data group: 'train' or 'test' or 'subset'")
parser.add_argument("--output_dir", type=str, required=True, help="Directory to save results")
parser.add_argument("--n_cpus", type=int, default=8, help="Number of CPU cores to use for processing")
parser.add_argument("--quiet", action='store_true', help="Suppress console output (log to file only)")
args = parser.parse_args()

group = args.group
output_dir = args.output_dir
n_cpus = args.n_cpus

# Start timing
script_start_time = time.time()

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Setup logging
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_file = os.path.join(output_dir, f'splicevo_process_data_{group}_{timestamp}.txt')
mmap_dir = os.path.join(output_dir, f'memmap_{group}')

# Redirect stdout and stderr to log file if quiet mode
if args.quiet:
    # Open log file with line buffering
    log_file_handle = open(log_file, 'a', buffering=1)
    # Save original stdout/stderr
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    # Redirect
    sys.stdout = log_file_handle
    sys.stderr = log_file_handle
    
    def log_print(msg):
        """Only write to file (stdout already redirected)."""
        print(msg, flush=True) 
else:
    def log_print(msg):
        """Print and write to log file."""
        print(msg)
        with open(log_file, 'a') as f:
            f.write(msg + '\n')

def format_time(seconds):
    """Format seconds as hours, minutes, seconds."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours}h {minutes}m {secs}s"

log_print(f"Processing started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log_print(f"Group: {args.group}")
log_print("=" * 60)

# Step 1: Initialize loader
log_print("Step 1: Initializing MultiGenomeDataLoader...")
step1_start = time.time()
loader = MultiGenomeDataLoader()
step1_time = time.time() - step1_start
log_print(f"✓ Loader initialized in {step1_time:.2f} seconds")
log_print("")

# Step 2: Add genomes
log_print("Step 2: Adding genomes")
log_print("  Adding human genome")
step2_start = time.time()
if group == "train":
    human_chromosomes = ['2', '4', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', 'X', 'Y', 'MT']
elif group == "test":
    human_chromosomes = ['1', '3', '5']
elif group == "subset":
    human_chromosomes = ['20', '21']

human_start = time.time()
log_print("  (chrs " + ", ".join(human_chromosomes) + ")")
loader.add_genome(
    genome_id="human_GRCh37",
    genome_path="/home/elek/sds/sd17d003/Anamaria/genomes/mazin/fasta/Homo_sapiens.fa.gz", 
    gtf_path="/home/elek/sds/sd17d003/Anamaria/genomes/mazin/gtf/Homo_sapiens.gtf.gz",
    chromosomes=human_chromosomes,
    metadata={"species": "homo_sapiens", "assembly": "GRCh37"}
)
human_time = time.time() - human_start
log_print(f"  ✓ Human genome added in {human_time:.2f} seconds")

step2_time = time.time() - step2_start
log_print(f"✓ All genomes added in {step2_time:.2f} seconds")
log_print("")

# Step 3: Add usage files
log_print("Step 3: Adding usage files...")
step3_start = time.time()

# Add human usage files if they exist
log_print("  Adding human usage files...")
human_usage_start = time.time()
for tissue in ["Brain"]: # , "Cerebellum", "Heart", "Kidney", "Liver", "Ovary", "Testis"
    for timepoint_int in range(1, 16):
        try:
            timepoint = str(timepoint_int)
            loader.add_usage_file(
                genome_id="human_GRCh37", 
                usage_file=f"/home/elek/projects/splicing/results/spliser/Homo_sapiens/Human.{tissue}.{timepoint}.combined.tsv",
                tissue=tissue,
                timepoint=timepoint
            )
        except FileNotFoundError as e:
            log_print(f"  Human usage files not found: {e}")
    human_usage_time = time.time() - human_usage_start
    log_print(f"  ✓ Human usage files added in {human_usage_time:.2f} seconds")

step3_time = time.time() - step3_start
log_print(f"✓ Usage files processed in {step3_time:.2f} seconds")

# Show available conditions
conditions_df = loader.get_available_conditions()
log_print("Available conditions:")
log_print(str(conditions_df))
log_print("")

# Step 4: Load all genome data
log_print("Step 4: Loading all genome data...")
step4_start = time.time()

loader.load_all_genomes_data()

step4_time = time.time() - step4_start
log_print(f"✓ All genome data loaded in {step4_time:.2f} seconds")

# Show summary statistics
log_print("\nData summary:")
summary = loader.get_summary()
log_print(str(summary))
log_print("")

# Step 5: Convert to arrays
log_print("Step 5: Converting to arrays with windowing...")
step5_start = time.time()

sequences, labels, usage_arrays, metadata = loader.to_arrays(
    window_size=1000,
    context_size=450,
    alpha_threshold=5,
    n_workers=n_cpus,
    save_memmap=mmap_dir
)

step5_time = time.time() - step5_start
log_print(f"✓ Data converted to arrays in {step5_time:.2f} seconds")
log_print(f"  Shape of sequences: {sequences.shape}")
log_print(f"  Shape of labels: {labels.shape}")
log_print(f"    Labels format: [:, :, 0] = donor sites, [:, :, 1] = acceptor sites")
log_print(f"  Shape of usage_arrays['alpha']: {usage_arrays['alpha'].shape}")
log_print(f"  Shape of usage_arrays['beta']: {usage_arrays['beta'].shape}")
log_print(f"  Shape of usage_arrays['sse']: {usage_arrays['sse'].shape}")
log_print(f"  Shape of metadata: {metadata.shape}")

# Get usage array info
usage_info = loader.get_usage_array_info(usage_arrays)
log_print(f"  Available conditions: {[c['display_name'] for c in usage_info['conditions']]}")
log_print("")

# Step 6: Save metadata
log_print("Step 6: Saving metadata...")
save_start = time.time()

metadata.to_csv(os.path.join(output_dir, f"metadata_{group}.csv.gz"), index=False, compression='gzip')

# Save usage info
with open(os.path.join(output_dir, f"usage_info_{group}.json"), 'w') as f:
    json.dump(usage_info, f, indent=2, default=str)

usage_summary = loader.get_usage_summary()
usage_summary.to_csv(os.path.join(output_dir, f"usage_summary_{group}.csv"), index=False)

save_time = time.time() - save_start

# Finish timing
total_time = time.time() - script_start_time

# Summary timing report
log_print("=" * 60)
log_print("Timing Summary")
log_print("=" * 60)
total_time = time.time() - step1_start
log_print(f"Step 1 - Initialize loader:     {format_time(step1_time):>12} ({100*step1_time/total_time:5.1f}%)")
log_print(f"Step 2 - Add genomes:           {format_time(step2_time):>12} ({100*step2_time/total_time:5.1f}%)")
log_print(f"  - Human genome:               {format_time(human_time):>12} ({100*human_time/step2_time:5.1f}%)")
log_print(f"Step 3 - Add usage files:       {format_time(step3_time):>12} ({100*step3_time/total_time:5.1f}%)")
log_print(f"  - Human usage files:          {format_time(human_usage_time):>12} ({100*human_usage_time/step3_time:5.1f}%)")
log_print(f"Step 4 - Load genome data:      {format_time(step4_time):>12} ({100*step4_time/total_time:5.1f}%)")
log_print(f"Step 5 - Convert to arrays:     {format_time(step5_time):>12} ({100*step5_time/total_time:5.1f}%)")
log_print(f"Step 6 - Save metadata:         {format_time(save_time):>12} ({100*save_time/total_time:5.1f}%)")
log_print("-" * 60)
log_print(f"Total time:                     {format_time(total_time):>12}")
log_print(f"\nProcessing completed at {datetime.now()}")
log_print("=" * 60)
log_print(f"Results saved to: {os.path.abspath(output_dir)}")

# Restore stdout/stderr if quiet mode
if args.quiet:
    sys.stdout = original_stdout
    sys.stderr = original_stderr
    log_file_handle.close()
    print(f"Processing complete. Log saved to: {log_file}")