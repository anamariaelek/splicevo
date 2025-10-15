# Multi-Genome Splice Site Data Processing

This module provides comprehensive functionality for loading, processing, and splitting splice site data from multiple genomes for deep learning model training. It addresses key challenges in splice site prediction including class imbalance, GC content bias, and cross-genome generalization.

## Key Features

### 1. Multi-Genome Data Loading
- Load splice site annotations from multiple genomes simultaneously
- Track genome, chromosome, and transcript origin for each example
- Extract sequence windows around splice sites with configurable size
- Generate balanced negative examples from non-splice site positions
- Calculate GC content for each sequence window
- Load tissue-specific splice site usage statistics (alpha, beta, SSE values)

## Core Classes

### MultiGenomeDataLoader
Main class for loading splice site data from multiple genomes with usage statistics.

```python
from splicevo.data import MultiGenomeDataLoader

# Initialize loader
loader = MultiGenomeDataLoader(
    orthology_file='../splicing/data/mazin/ortholog_groups.tsv'
)

# Add genomes
loader.add_genome(
    genome_id="human_GRCh37",
    genome_path="../../sds/sd17d003/Anamaria/genomes/mazin/fasta/Homo_sapiens.fa.gz", 
    gtf_path="../../sds/sd17d003/Anamaria/genomes/mazin/gtf/Homo_sapiens.gtf.gz",
    chromosomes=["21", "20"],
    metadata={"species": "homo_sapiens", "assembly": "GRCh37"}
)

loader.add_genome(
    genome_id="mouse_GRCm38",
    genome_path="../../sds/sd17d003/Anamaria/genomes/mazin/fasta/Mus_musculus.fa.gz",
    gtf_path="../../sds/sd17d003/Anamaria/genomes/mazin/gtf/Mus_musculus.gtf.gz",
    chromosomes=['18', '19'],
    metadata={"species": "mus_musculus", "assembly": "GRCm38"}
)

# Add usage files individually for different tissues and timepoints
loader.add_usage_file(
    genome_id="human_GRCh37", 
    usage_file="../splicing/results/spliser/usage_stats/Human.Cerebellum.29ypb.combined.nochr.tsv",
    tissue="Cerebellum",
    timepoint="29ypb"
)

loader.add_usage_file(
    genome_id="human_GRCh37",
    usage_file="../splicing/results/spliser/usage_stats/Human.Cerebellum.0dpb.combined.nochr.tsv",
    tissue="Cerebellum",
    timepoint="0dpb"
)

loader.add_usage_file(
    genome_id="human_GRCh37",
    usage_file="../splicing/results/spliser/usage_stats/Human.Heart.0dpb.combined.nochr.tsv", 
    tissue="Heart",
    timepoint="0dpb"
)

loader.add_usage_file(
    genome_id="mouse_GRCm38",
    usage_file="../splicing/results/spliser/usage_stats/Mouse.Cerebellum.0dpb.combined.nochr.tsv", 
    tissue="Cerebellum",
    timepoint="0dpb"
)

loader.add_usage_file(
    genome_id="mouse_GRCm38",
    usage_file="../splicing/results/spliser/usage_stats/Mouse.Heart.0dpb.combined.nochr.tsv", 
    tissue="Heart",
    timepoint="0dpb"
)

# Check available conditions
conditions_df = loader.get_available_conditions()
print(conditions_df)

# Load all data
loader.load_all_genomes_data()

# Dataframe with all splice sites from loaded genomes
df = loader.get_dataframe().head()
df.head()

# Summarize
loader.get_summary()

# Convert to arrays for ML
sequences, labels, usage_arrays, metadata = loader.to_arrays(
    window_size=1000,
    context_size=4500
)

# Get usage information
usage_info = loader.get_usage_array_info(usage_arrays=usage_arrays)
print(f"Available conditions: {[c['display_name'] for c in usage_info['conditions']]}")
print(f"Coverage per condition: {usage_info['condition_coverage']}")

# Save arrays to disk as .npz compressed file
import numpy as np
np.savez_compressed('splicevo_data.npz', sequences=sequences, labels=labels, usage_arrays=usage_arrays)
metadata.to_csv('splicevo_metadata.csv', index=False)


```

### StratifiedGCSplitter
Advanced data splitter that handles class imbalance and GC content stratification.

```python
from splicevo.data import StratifiedGCSplitter

splitter = StratifiedGCSplitter(
    test_size=0.2,
    val_size=0.2, 
    gc_bins=10,
    random_state=42
)

# Stratified split by GC content and class (acceptor, donor, none)
split_data_gc = splitter.stratified_split(sequences, labels, metadata, stratify_by='gc_class')
splitter.get_split_statistics(split_data_gc)

# Balanced class (acceptor, donor, none) split with undersampling
split_data_balanced = splitter.balanced_class_split(sequences, labels, metadata, balance_method='undersample', stratify_by='gc_class')
splitter.get_split_statistics(split_data_balanced)

# Chromosome-aware split with ortholog exclusion
test_chromosomes = {'human_GRCh37': ['21'], 'mouse_GRCm38': ['19']}
split_data_ortholog = splitter.chromosome_aware_split(
    sequences, labels, metadata, test_chromosomes=test_chromosomes
)
splitter.get_split_statistics(split_data_ortholog)
```
