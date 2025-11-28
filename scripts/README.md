# Memory-Efficient Data Loading and Splitting Approach

## Overview

The refactored approach processes one genome at a time to minimize memory usage and enables parallel/distributed processing of genomes.

Make sure to have the following variables set:

```bash
# Where SplicEvo is located
export SPLICEVO_DIR=/home/elek/projects/splicevo
```

## Two-Step Process

### Step 1: data_load.py - Process Individual Genomes

Processes **one genome at a time** and saves its data in a dedicated directory.

**Usage:**
```bash
python ${SPLICEVO_DIR}/scripts/data_load.py \
    --config configs/genomes.json \
    --genome_id human_GRCh37 \
    --output_dir data/processed \
    --window_size 1000 \
    --context_size 450 \
    --alpha_threshold 5 \
    --n_cpus 8 &
```

**What it does:**
1. Loads only the specified genome
2. Extracts sequences, labels, and usage arrays
3. Maps all genes/splice sites to their sequence
4. Saves to `output_dir/genome_id/`:
   - `sequences.mmap` - sequences
   - `labels.mmap` - labels for each sequence
   - `usage_alpha.mmap`, `usage_beta.mmap`, `usage_sse.mmap` - usage arrays
   - `metadata.csv` - links sequences to genes/sites
   - `summary.json` - metadata about the genome data

**Run for each genome:**
```bash
for genome in human_GRCh37 mouse_GRCm38 rat_Rnor_5.0; do
    python ${SPLICEVO_DIR}/scripts/data_load.py \
        --config configs/genomes.json \
        --genome_id $genome \
        --output_dir data/processed \
        --window_size 1000 \
        --context_size 450 \
        --quiet \
        --n_cpus 8 &
done
```

### Step 2: data_split.py - Combine into Train/Test Sets

Reads individual genome directories and combines them into train/test splits based on orthology.

**Usage:**
```bash
SPLICEVO_DIR=/home/elek/projects/splicevo
ORTHOLOGY_FILE=/home/elek/sds/sd17d003/Anamaria/genomes/mazin/ortholog_groups.tsv
python ${SPLICEVO_DIR}/scripts/data_split.py \
    --input_dir data/processed \
    --output_dir data/splits/run1 \
    --orthology_file ${ORTHOLOGY_FILE} \
    --pov_genome human_GRCh37 \
    --test_chromosomes 1 3 5 \
    --quiet &
```

**What it does:**
1. Discovers all genome directories in `input_dir`
2. Loads orthology file
3. Determines train/test split based on POV genome chromosomes
4. Expands split to ortholog groups
5. For each genome:
   - Loads only sequences assigned to train or test
   - Avoids loading entire genome data
6. Combines across genomes and saves to:
   - `output_dir/train/` - training data
   - `output_dir/test/` - test data
   - Each with sequences.mmap, labels.mmap, usage_*.mmap, metadata.csv

## Key Improvements

1. **Memory Efficiency**: Only one genome loaded at a time during data_load
2. **Deduplication**: Same sequence saved once per genome, linked to all overlapping genes
3. **Parallel Processing**: Can run data_load.py for different genomes in parallel
4. **Incremental**: Can process additional genomes later without reprocessing existing ones
5. **Flexible Splitting**: Can create multiple train/test splits from the same loaded data

## Directory Structure

```
data/
  processed/
    human_GRCh37/
      sequences.mmap
      labels.mmap
      usage_alpha.mmap
      usage_beta.mmap
      usage_sse.mmap
      metadata.csv
      sequence_mapping.csv
      summary.json
    mouse_GRCm38/
      ...
    rat_Rnor_6.0/
      ...
  splits/
    run1/
      train/
        sequences.mmap
        labels.mmap
        usage_*.mmap
        metadata.csv
      test/
        ...
      summary.json
```
