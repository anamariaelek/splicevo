# Flexible Attributions API

This document describes the new flexible API for calculating attributions (input gradients) for splice site and usage predictions.

## Overview

The API provides a unified interface for computing attributions with flexible input specification:

- **Window indices**: Explicitly specify which sequences to analyze
- **Genomic coordinates**: Specify genomic regions and automatically find overlapping windows
- **Defaults**: Automatically select evenly-spaced windows

## Core Modules

### 1. `window_utils.py` - Window and Coordinate Utilities

Located in `src/splicevo/utils/window_utils.py`, provides core utilities for window matching and genomic coordinate resolution:

#### `resolve_window_indices()`
Flexible resolution of window indices from various input formats:

```python
from splicevo.utils.window_utils import resolve_window_indices
import numpy as np

# Option 1: Explicit indices
indices = resolve_window_indices(meta_df, window_indices=np.array([0, 5, 10]))

# Option 2: Genomic coordinates
coords = [
    ('human_GRCh37', '3', 142740160, 142740259, '+'),
    ('mouse_GRCm38', '5', 124543487, 124543507, '+'),
]
indices = resolve_window_indices(meta_df, genomic_coords=coords)

# Option 3: Default evenly-spaced windows
indices = resolve_window_indices(meta_df, default_n=10)
```

#### `find_overlapping_windows()`
Find all windows that overlap with specified genomic coordinates:

```python
from splicevo.utils.window_utils import find_overlapping_windows

coords = [('human_GRCh37', '3', 142740160, 142740259, '+')]
overlapping_indices = find_overlapping_windows(meta_df, coords)
```

#### `filter_splice_sites_by_genomic_coords()`
Filter splice site positions to only those within specified coordinate ranges:

```python
from splicevo.utils.window_utils import filter_splice_sites_by_genomic_coords

donor_pos, acceptor_pos = filter_splice_sites_by_genomic_coords(
    donor_positions=donor_pos,
    acceptor_positions=acceptor_pos,
    window_start=window_start,
    window_strand='+',
    genomic_coords_dict=coords_dict,
    genome_id='human_GRCh37',
    chromosome='3'
)
```

### 2. `attributions.py` - Flexible Attribution Calculator

Located in `src/splicevo/attributions/attributions.py`, provides the `AttributionCalculator` class:

```python
from splicevo.attributions.attributions import AttributionCalculator
from splicevo.model import SplicevoModel

# Initialize calculator
model = SplicevoModel.load(model_path)
calc = AttributionCalculator(model, device='cuda', verbose=True)

# Compute attributions for splice sites
result = calc.compute_splice_attributions(
    sequences, labels, meta_df,
    window_indices=np.array([0, 5, 10])
)

# Compute attributions for usage conditions
result = calc.compute_usage_attributions(
    sequences, labels, usage, meta_df,
    genomic_coords=[('human_GRCh37', '3', 142740160, 142740259, '+')]
)
```

#### Attribution Results

Both methods return a dictionary with:

```python
{
    'attributions': {
        'seq_idx_position': {
            'id': 'seq_idx_position',
            'seq_idx': int,
            'position': int,
            'site_class': int (1=donor, 2=acceptor),
            'site_type': str ('donor' or 'acceptor'),
            'attribution': np.ndarray (seq_len, 4) or (seq_len, 4, n_conditions),
            'metadata': {
                'genome_id': str,
                'chromosome': str,
                'window_start': int,
                'window_end': int,
                'strand': str
            }
        },
        # ... more contributions
    },
    'metadata': {
        'task': str ('splice_classification' or 'usage_prediction'),
        'device': str,
        'model_type': str,
        'window_indices': list,
        'genomic_coords_provided': bool
    },
    'summary': {
        'total_processed': int,
        'total_skipped': int,
        'n_windows': int
    }
}
```

### 3. Convenience Functions in `compute.py`

High-level wrapper functions for easy access:

```python
from splicevo.attributions.compute import (
    compute_attributions_splice,
    compute_attributions_usage
)

# Splice site attributions
result = compute_attributions_splice(
    model, sequences, labels, meta_df,
    genomic_coords=[('human_GRCh37', '3', 142740160, 142740259, '+')]
)

# Usage attributions
result = compute_attributions_usage(
    model, sequences, labels, usage, meta_df,
    window_indices=np.array([0, 5, 10]),
    condition_names=['Brain', 'Heart', 'Liver']
)
```

## Calculate Attributions

Load data

```python
from splicevo.model import SplicevoModel
from splicevo.utils.model_utils import load_model_and_config
from splicevo.utils.data_utils import load_processed_data, load_predictions
from splicevo.attributions.compute import compute_attributions_splice, compute_attributions_usage
import numpy as np
import pandas as pd

# Load data and model
model = load_model_and_config('path/to/model.pt')
sequences, labels, _, _, usage, _ = load_processed_data('path/to/processed_data')
label_predictions, label_probabilities, usage_predictioins, _, _, _ = load_predictions('path/to/predictions')
metadata = pd.read_csv('path/to/processed_data/metadata.csv')
```

### Example 1: Compute Attributions for Specific Sequence in Data

```python
# Compute attributions for sequences 0, 5, and 10
result = compute_attributions_splice(
    model, sequences, labels, metadata,
    window_indices=np.array([0, 5, 10]),
    device='cuda'
)

# Access results
for site_id, attr_data in result['attributions'].items():
    print(f"Site {site_id}: type={attr_data['site_type']}, "
          f"Total attribution={attr_data['attribution'].sum():.4f}")
```

### Example 2: Compute Attributions for Genomic Regions

Genomic coordinates are specified as tuples: `(genome_id, chromosome, start, end, strand)`

- `genome_id`: Identifier of the genome (e.g., 'human_GRCh37', 'mouse_GRCm38')
- `chromosome`: Chromosome identifier (e.g., '3', 'X', 'MT')
- `start`: 1-based inclusive start coordinate
- `end`: 1-based inclusive end coordinate
- `strand`: '+', '-', or '*' (where '*' means match any strand)

```python
# Define genomic regions of interest
genomic_targets = [
    ('human_GRCh37', '1', 32304501, 32305501, '+'),
    ('mouse_GRCm38', '2', 3114266, 3115266, '+'),
    ('rat_Rnor_5.0', '5', 177157211, 177158211, '-'),
]

# Compute attributions
result = compute_attributions_splice(
    model, sequences, labels, metadata,
    genomic_coords=genomic_targets,
    verbose=True
)

# Access results
print(f"Processed {result['summary']['total_processed']} splice sites")
for site_id, attr_data in result['attributions'].items():
    meta = attr_data['metadata']
    print(f"{meta['genome_id']}:{meta['chromosome']}"
          f":{meta['window_start']}-{meta['window_end']}"
          f" Total attribution={attr_data['attribution'].sum():.4f}")
```

### Example 3: Compute Attributions for Specific Splice Sites

You can specify exact splice site positions as `(seq_idx, position)` tuples:

```python
# Compute attributions for specific splice sites
positions = [
    (0, 50),    # sequence 0, position 50
    (0, 150),   # sequence 0, position 150
    (5, 95),    # sequence 5, position 95
    (10, 120)   # sequence 10, position 120
]

result = compute_attributions_splice(
    model, sequences, labels, metadata,
    positions=positions
)

# Access results - only the specified positions will be present
for site_id, attr_data in result['attributions'].items():
    print(f"Site {site_id}: type={attr_data['site_type']}, "
          f"Total attribution={attr_data['attribution'].sum():.4f}")
```

### Example 4: Filter Attributions by Prediction Correctness

```python
# Compute attributions only for correctly predicted sites
result = compute_attributions_splice(
    model, sequences, labels, meta_df,
    genomic_coords=genomic_targets,
    predictions=label_predictions,
    filter_by_correct=True
)

print(f"Total: {result['summary']['total_processed']}, "
      f"Skipped: {result['summary']['total_skipped']}")
```

### Example 5: Compute Usage Attributions

```python
# Compute attributions for usage (condition) predictions
result = compute_attributions_usage(
    model, sequences, labels, usage, meta_df,
    window_indices=np.array([0, 5, 10]),
    condition_names=['Brain', 'Heart', 'Liver', 'Kidney', 'Ovary', 'Testis', 'Cerebellum'],
    verbose=True
)

# Access per-condition attributions
for site_id, attr_data in result['attributions'].items():
    attr = attr_data['attribution']  # Shape: (seq_len, 4, n_conditions)
    print(f"Site {site_id} total attribution: {np.nansum(attr):.4f}")
```
