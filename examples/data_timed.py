import time
import pickle
import os
from datetime import datetime

from splicevo.data import MultiGenomeDataLoader

# Create output directory for results
output_dir = "data_processing_results"
os.makedirs(output_dir, exist_ok=True)

print(f"Starting data processing at {datetime.now()}")
print("=" * 60)

# Step 1: Initialize loader
print("Step 1: Initializing MultiGenomeDataLoader...")
step1_start = time.time()
loader = MultiGenomeDataLoader(
    window_size=200, 
    orthology_file='../splicing/data/mazin/ortholog_groups.tsv'
)
step1_time = time.time() - step1_start
print(f"✓ Loader initialized in {step1_time:.2f} seconds")
print()

# Step 2: Add genomes
print("Step 2: Adding genomes...")
step2_start = time.time()

print("  Adding human genome...")
human_start = time.time()
loader.add_genome(
    genome_id="human_GRCh37",
    genome_path="../../sds/sd17d003/Anamaria/genomes/mazin/fasta/Homo_sapiens.fa.gz", 
    gtf_path="../../sds/sd17d003/Anamaria/genomes/mazin/gtf/Homo_sapiens.gtf.gz",
    chromosomes=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', 'X', 'Y'],
    metadata={"species": "homo_sapiens", "assembly": "GRCh37"}
)
human_time = time.time() - human_start
print(f"  ✓ Human genome added in {human_time:.2f} seconds")

print("  Adding mouse genome...")
mouse_start = time.time()
loader.add_genome(
    genome_id="mouse_GRCm38",
    genome_path="../../sds/sd17d003/Anamaria/genomes/mazin/fasta/Mus_musculus.fa.gz",
    gtf_path="../../sds/sd17d003/Anamaria/genomes/mazin/gtf/Mus_musculus.gtf.gz",
    chromosomes=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', 'X', 'Y'],
    metadata={"species": "mus_musculus", "assembly": "GRCm38"}
)
mouse_time = time.time() - mouse_start
print(f"  ✓ Mouse genome added in {mouse_time:.2f} seconds")

step2_time = time.time() - step2_start
print(f"✓ All genomes added in {step2_time:.2f} seconds")
print()

# Step 3: Load all data
print("Step 3: Loading all genome data...")
step3_start = time.time()
loader.load_all_genomes(negative_ratio=2.0)
step3_time = time.time() - step3_start
print(f"✓ All genome data loaded in {step3_time:.2f} seconds")
print()

# Step 4: Convert to arrays
print("Step 4: Converting to ML arrays...")
step4_start = time.time()
X, y, metadata = loader.to_arrays()
step4_time = time.time() - step4_start
print(f"✓ Data converted to arrays in {step4_time:.2f} seconds")
print(f"  Shape of X: {X.shape}")
print(f"  Shape of y: {y.shape}")
print(f"  Shape of metadata: {metadata.shape}")
print()

# Save the raw data
print("Saving raw data...")
save_start = time.time()
with open(os.path.join(output_dir, "raw_data.pkl"), "wb") as f:
    pickle.dump({"X": X, "y": y, "metadata": metadata}, f)
metadata.to_csv(os.path.join(output_dir, "metadata.csv"), index=False)
save_time = time.time() - save_start
print(f"✓ Raw data saved in {save_time:.2f} seconds")
print()

# Step 5: Initialize splitter
print("Step 5: Initializing StratifiedGCSplitter...")
step5_start = time.time()
from splicevo.data import StratifiedGCSplitter

splitter = StratifiedGCSplitter(
    test_size=0.2,
    val_size=0.2, 
    gc_bins=10,
    random_state=42
)
step5_time = time.time() - step5_start
print(f"✓ Splitter initialized in {step5_time:.2f} seconds")
print()

# Step 6: Stratified split by GC content and class
print("Step 6: Performing stratified split by GC content and class...")
step6_start = time.time()
split_data_gc = splitter.stratified_split(X, y, metadata, stratify_by='gc_class')
step6_time = time.time() - step6_start
print(f"✓ GC stratified split completed in {step6_time:.2f} seconds")

stats_gc = splitter.get_split_statistics(split_data_gc)
print("Split statistics (GC stratified):")
print(stats_gc)
print()

# Save GC split results
print("Saving GC split results...")
save_gc_start = time.time()
with open(os.path.join(output_dir, "split_data_gc.pkl"), "wb") as f:
    pickle.dump(split_data_gc, f)
stats_gc.to_csv(os.path.join(output_dir, "split_stats_gc.csv"), index=False)
save_gc_time = time.time() - save_gc_start
print(f"✓ GC split results saved in {save_gc_time:.2f} seconds")
print()

# Step 7: Balanced class split with undersampling
print("Step 7: Performing balanced class split with undersampling...")
step7_start = time.time()
split_data_balanced = splitter.balanced_class_split(X, y, metadata, balance_method='undersample')
step7_time = time.time() - step7_start
print(f"✓ Balanced split completed in {step7_time:.2f} seconds")

stats_balanced = splitter.get_split_statistics(split_data_balanced)
print("Split statistics (Balanced):")
print(stats_balanced)
print()

# Save balanced split results
print("Saving balanced split results...")
save_balanced_start = time.time()
with open(os.path.join(output_dir, "split_data_balanced.pkl"), "wb") as f:
    pickle.dump(split_data_balanced, f)
stats_balanced.to_csv(os.path.join(output_dir, "split_stats_balanced.csv"), index=False)
save_balanced_time = time.time() - save_balanced_start
print(f"✓ Balanced split results saved in {save_balanced_time:.2f} seconds")
print()

# Step 8: Chromosome-aware split with ortholog exclusion
print("Step 8: Performing chromosome-aware split with ortholog exclusion...")
step8_start = time.time()
test_chromosomes = {'human_GRCh37': ['1'], 'mouse_GRCm38': ['1']}
split_data_ortholog = splitter.chromosome_aware_split(
    X, y, metadata, test_chromosomes=test_chromosomes,
)
step8_time = time.time() - step8_start
print(f"✓ Chromosome-aware split completed in {step8_time:.2f} seconds")

stats_ortholog = splitter.get_split_statistics(split_data_ortholog)
print("Split statistics (Chromosome-aware with ortholog exclusion):")
print(stats_ortholog)
print()

# Save chromosome-aware split results
print("Saving chromosome-aware split results...")
save_ortholog_start = time.time()
with open(os.path.join(output_dir, "split_data_ortholog.pkl"), "wb") as f:
    pickle.dump(split_data_ortholog, f)
stats_ortholog.to_csv(os.path.join(output_dir, "split_stats_ortholog.csv"), index=False)
save_ortholog_time = time.time() - save_ortholog_start
print(f"✓ Chromosome-aware split results saved in {save_ortholog_time:.2f} seconds")
print()

# Summary timing report
print("=" * 60)
print("TIMING SUMMARY")
print("=" * 60)
total_time = time.time() - step1_start
print(f"Step 1 - Initialize loader:     {step1_time:8.2f} seconds")
print(f"Step 2 - Add genomes:           {step2_time:8.2f} seconds")
print(f"  - Human genome:               {human_time:8.2f} seconds")
print(f"  - Mouse genome:               {mouse_time:8.2f} seconds")
print(f"Step 3 - Load all data:         {step3_time:8.2f} seconds")
print(f"Step 4 - Convert to arrays:     {step4_time:8.2f} seconds")
print(f"Step 5 - Initialize splitter:   {step5_time:8.2f} seconds")
print(f"Step 6 - GC stratified split:   {step6_time:8.2f} seconds")
print(f"Step 7 - Balanced split:        {step7_time:8.2f} seconds")
print(f"Step 8 - Chromosome split:      {step8_time:8.2f} seconds")
print(f"Data saving time:               {save_time + save_gc_time + save_balanced_time + save_ortholog_time:8.2f} seconds")
print("-" * 60)
print(f"TOTAL TIME:                     {total_time:8.2f} seconds")
print("=" * 60)

print(f"\nProcessing completed at {datetime.now()}")
print(f"Results saved to: {os.path.abspath(output_dir)}")
print("\nSaved files:")
print(f"  - raw_data.pkl: Raw X, y, metadata arrays")
print(f"  - metadata.csv: Metadata in CSV format")
print(f"  - split_data_gc.pkl: GC stratified split results")
print(f"  - split_stats_gc.csv: GC split statistics")
print(f"  - split_data_balanced.pkl: Balanced split results")
print(f"  - split_stats_balanced.csv: Balanced split statistics")
print(f"  - split_data_ortholog.pkl: Chromosome-aware split results")
print(f"  - split_stats_ortholog.csv: Chromosome-aware split statistics")