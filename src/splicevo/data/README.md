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
loader = MultiGenomeDataLoader(window_size=200)

# Add genomes
loader.add_genome(
    genome_id="human_hg38",
    genome_path="../../sds/sd17d003/Anamaria/genomes/mazin/fasta/Homo_sapiens.fa.gz", 
    gtf_path="../../sds/sd17d003/Anamaria/genomes/mazin/gtf/Homo_sapiens.gtf.gz",
    chromosomes=["1", "2"],
    metadata={"species": "homo_sapiens", "assembly": "hg38"}
)

loader.add_genome(
    genome_id="mouse_mm10",
    genome_path="../../sds/sd17d003/Anamaria/genomes/mazin/fasta/Mus_musculus.fa.gz",
    gtf_path="../../sds/sd17d003/Anamaria/genomes/mazin/gtf/Mus_musculus.gtf.gz",
    chromosomes=["1", "2"],
    metadata={"species": "mus_musculus", "assembly": "mm10"}
)

# Load all data (uses fast batch processing by default)
loader.load_all_genomes(negative_ratio=2.0)

# Convert to arrays for ML
X, y, metadata = loader.to_arrays()

```
    