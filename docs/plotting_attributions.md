# Plotting Attributions with the Flexible API

The plotting functions work seamlessly with the attributions API. Here's how to use them:

## Quick Start

```python
from splicevo.attributions.plot import plot_attributions_splice, plot_attributions_usage
from splicevo.attributions.compute import compute_attributions_splice

# 1. Compute attributions
result = compute_attributions_splice(
    model, sequences, labels, metadata,
    window_indices=np.array([0, 5, 10])
)

# 2. Convert to plotting format
def convert_attributions_to_attrs_dict(result):
    attrs_dict = {}
    for site_id, attr_data in result['attributions'].items():
        attrs_dict[site_id] = {
            'id': attr_data['id'],
            'seq_idx': attr_data['seq_idx'],
            'position': attr_data['position'],
            'sequence': attr_data['sequence'],
            'attr': attr_data['attribution'],
            'site_class': attr_data['site_class'],
            'site_type': attr_data['site_type'],
            'strand': attr_data['metadata'].get('strand', '?')
        }
    return attrs_dict

attrs_dict = convert_attributions_to_attrs_dict(result)

# 3. Plot
fig = plot_attributions_splice(attrs_dict, model_config, max_plots=5)
plt.show()
```

## Plotting Functions

### `plot_attributions_splice()`

Visualize splice site attributions as sequence logos.

**Parameters:**
```python
plot_attributions_splice(
    attrs_dict: Dict,           # Attribution data (from convert function)
    model_config: Dict,         # Must contain 'context_len'
    max_plots: int = 5,         # Max number of sites to show
    window: int = 100,          # Bases around splice site
    figsize: Optional[Tuple] = None,  # (width, height)
    ylim: Optional = None       # Y-axis limits
)
```

**Y-axis Options:**
- `ylim=None` - Auto-scale each plot independently
- `ylim='auto'` - Auto-scale globally across all plots
- `ylim=(-0.5, 0.5)` - Fixed limits for all plots

**Example:**
```python
attrs_dict = convert_attributions_to_attrs_dict(result)
fig = plot_attributions_splice(
    attrs_dict, 
    model_config,
    max_plots=5,
    window=100,
    figsize=(12, 8),
    ylim='auto'  # Global auto-scaling
)
plt.show()
```

### `plot_attributions_usage()`

Visualize usage/condition-specific attributions as sequence logos.

**Parameters:**
```python
plot_attributions_usage(
    attrs_dict: Dict,                    # Attribution data
    model_config: Dict,                  # Must contain 'context_len'
    conditions: Optional[list] = None,   # Condition names
    conditions_to_plot: Optional[list] = None,  # Indices to include
    max_plots: int = 100,                # Max total plots
    window: int = 100,                   # Bases around splice site
    figsize: Optional[Tuple] = None,     # (width, height)
    ylim: Optional = None                # Y-axis limits
)
```

**Note:** Each splice site is plotted once per condition, so max_plots limits the total number of plots (site × condition combinations).

**Example:**
```python
result = compute_attributions_usage(
    model, sequences, labels, usage, metadata,
    window_indices=np.array([0, 5, 10]),
    condition_names=['Brain', 'Heart', 'Liver', ...]
)

attrs_dict = convert_attributions_to_attrs_dict(result)
fig = plot_attributions_usage(
    attrs_dict,
    model_config,
    conditions=['Brain', 'Heart', 'Liver', ...],
    conditions_to_plot=[0, 1, 2],  # Plot first 3 conditions only
    max_plots=20,
    window=100,
    ylim='auto'
)
plt.show()
```

## Integration Patterns

### Pattern 1: Window Indices + Plotting
```python
result = compute_attributions_splice(
    model, sequences, labels, metadata,
    window_indices=np.array([0, 5, 10])
)
attrs_dict = convert_attributions_to_attrs_dict(result)
fig = plot_attributions_splice(attrs_dict, model_config, ylim='auto')
plt.show()
```

### Pattern 2: Genomic Coordinates + Plotting
```python
coords = [('human_GRCh37', '3', 142740160, 142740259, '+')]
result = compute_attributions_splice(
    model, sequences, labels, metadata,
    genomic_coords=coords
)
attrs_dict = convert_attributions_to_attrs_dict(result)
fig = plot_attributions_splice(attrs_dict, model_config, window=150)
plt.show()
```

### Pattern 3: Filter Correct + Plotting
```python
result = compute_attributions_splice(
    model, sequences, labels, metadata,
    genomic_coords=coords,
    predictions=model_predictions,
    filter_by_correct=True
)
attrs_dict = convert_attributions_to_attrs_dict(result)
fig = plot_attributions_splice(attrs_dict, model_config, ylim='auto')
plt.show()
```

### Pattern 4: Usage Attributions + Plotting
```python
result = compute_attributions_usage(
    model, sequences, labels, usage, metadata,
    window_indices=np.array([0, 5, 10]),
    condition_names=['Brain', 'Heart', 'Liver', ...]
)
attrs_dict = convert_attributions_to_attrs_dict(result)
fig = plot_attributions_usage(
    attrs_dict,
    model_config,
    conditions=['Brain', 'Heart', 'Liver', ...],
    max_plots=30,
    ylim='auto'
)
plt.show()
```

## Visualization Details

### Sequence Logos
- **Height** of each letter = importance (absolute value of attribution)
- **Color**: A=green, C=blue, G=orange, T=red
- **Dashed line**: Marks the exact splice site position

### Plot Titles
Include:
- Sequence index and position
- Site type (donor/acceptor)
- Strand orientation
- Total attribution around the site

### Auto-scaling with `ylim='auto'`
- Determines global min/max across all plots
- Ensures consistent visual comparison
- Useful for identifying high vs low attribution sites
- Better than individual scaling for comparing multiple sites

## Tips

1. **Use `ylim='auto'`** for comparing multiple attributions
2. **Increase `window`** for more context around the splice site
3. **Set `max_plots`** to limit output for large results
4. **Specify `conditions_to_plot`** in usage plotting to focus on relevant conditions
5. **Save figures** for inclusion in reports:
   ```python
   fig.savefig('attributions.pdf', dpi=300, bbox_inches='tight')
   ```
