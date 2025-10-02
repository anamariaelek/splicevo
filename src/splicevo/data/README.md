# Multi-Genome Splice Site Data Processing

This module provides comprehensive functionality for loading, processing, and splitting splice site data from multiple genomes for deep learning model training. It addresses key challenges in splice site prediction including class imbalance, GC content bias, and cross-genome generalization.

## Key Features

### 1. Multi-Genome Data Loading
- Load splice site annotations from multiple genomes simultaneously
- Track genome, chromosome, and transcript origin for each example
- Extract sequence windows around splice sites with configurable size
- Generate balanced negative examples from non-splice site positions
- Calculate GC content for each sequence window

### 2. Advanced Data Splitting
- **Stratified splitting** by GC content and class to maintain balanced representation
- **Genome-aware splitting** to prevent data leakage between related species
- **Class balancing** to handle positive/negative example imbalance
- **Site-specific extensions** for modeling tissue-specific, condition-specific, or stage-specific splice site usage

### 3. GC Content Analysis & Normalization
- Comprehensive GC content bias detection
- Statistical analysis of GC distributions across classes and genomes
- Multiple normalization methods (standard, min-max, quantile)
- Visualization tools for GC content analysis

## Core Classes

### MultiGenomeDataLoader
Main class for loading splice site data from multiple genomes.

```python
from splicevo.data import MultiGenomeDataLoader

# Initialize loader
loader = MultiGenomeDataLoader(
    window_size=200, 
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
    chromosomes=["19", "18"],
    metadata={"species": "mus_musculus", "assembly": "GRCm38"}
)

# Load all data (uses fast batch processing by default)
loader.load_all_genomes(negative_ratio=2.0)

# Convert to arrays for ML
X, y, metadata = loader.to_arrays()
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
split_data_gc = splitter.stratified_split(X, y, metadata, stratify_by='gc_class')
splitter.get_split_statistics(split_data_gc)

# Balanced class (acceptor, donor, none) split with undersampling
split_data_balanced = splitter.balanced_class_split(X, y, metadata, balance_method='undersample', stratify_by='gc_class')
splitter.get_split_statistics(split_data_balanced)

# Chromosome-aware split with ortholog exclusion
test_chromosomes = {'human_GRCh37': ['21'], 'mouse_GRCm38': ['19']}
split_data_ortholog = splitter.chromosome_aware_split(
    X, y, metadata, test_chromosomes=test_chromosomes,
)
splitter.get_split_statistics(split_data_ortholog)
```
