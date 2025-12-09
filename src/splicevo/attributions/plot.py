"""
Visualization utilities for sequence attributions using sequence logos.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logomaker
from typing import Optional, Dict, Tuple


BASE_COLORS = {
    0: 'green',    # A
    1: 'blue',     # C
    2: 'orange',   # G
    3: 'red'       # T
}

BASE_NAMES = {
    0: 'A',
    1: 'C',
    2: 'G',
    3: 'T'
}


def extract_attribution_values(
    attr: np.ndarray,
    sequence: np.ndarray,
    start: int,
    end: int,
    condition_idx: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract attribution values for actual bases in a sequence region.
    
    Args:
        attr: Attribution array (seq_len, 4) or (seq_len, 4, n_conditions)
        sequence: One-hot encoded sequence (seq_len, 4)
        start: Start position
        end: End position (exclusive)
        condition_idx: Condition index if attr has shape (seq_len, 4, n_conditions)
        
    Returns:
        Tuple of:
        - attr_values: Attribution values for each position
        - base_indices: Indices of actual bases
        - base_names: Names of actual bases
    """
    if condition_idx is not None:
        attr_slice = attr[start:end, :, condition_idx]
    else:
        attr_slice = attr[start:end, :]
    
    # Attribution for the actual base at each position
    attr_base = attr_slice * sequence[start:end, :]
    
    attr_values = []
    base_indices = []
    base_names_list = []
    
    for i in range(attr_base.shape[0]):
        base_idx = np.argmax(sequence[start + i])
        val = attr_base[i, base_idx]
        if np.isnan(val):
            val = 0.0
        attr_values.append(val)
        base_indices.append(base_idx)
        base_names_list.append(BASE_NAMES[base_idx])
    
    return np.array(attr_values), np.array(base_indices), np.array(base_names_list)


def create_attribution_logo(
    attr_values: np.ndarray,
    base_names: np.ndarray,
    ax: plt.Axes,
    x_positions: Optional[np.ndarray] = None,
    title: Optional[str] = None,
    xlabel: str = 'Position',
    ylabel: str = 'Input x Gradient'
) -> None:
    """
    Create a sequence logo with attribution values.
    
    Args:
        attr_values: Attribution values for each position
        base_names: Base names for each position
        ax: Matplotlib axes to plot on
        x_positions: X-axis positions (if None, use indices)
        title: Title for the plot
        xlabel: Label for x-axis
        ylabel: Label for y-axis
    """
    if x_positions is None:
        x_positions = np.arange(len(attr_values))
    
    # Create DataFrame for logomaker
    df_data = []
    for i, (attr_val, base) in enumerate(zip(attr_values, base_names)):
        row = {'A': 0, 'C': 0, 'G': 0, 'T': 0}
        row[base] = attr_val
        df_data.append(row)
    
    df = pd.DataFrame(df_data, index=x_positions)
    
    # Create logo
    logomaker.Logo(
        df,
        ax=ax,
        color_scheme={'A': 'green', 'C': 'blue', 'G': 'orange', 'T': 'red'}
    )
    
    if title:
        ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.axvline(0, color='red', linestyle='--', linewidth=1, alpha=0.7, label='Target position')


def plot_attributions_splice(
    attrs_dict: Dict,
    model_config: Dict,
    max_plots: int = 5,
    window: int = 100,
    figsize: Optional[Tuple[int, int]] = None
) -> plt.Figure:
    """
    Plot attributions for splice site predictions.
    
    Args:
        attrs_dict: Dictionary of attribution data from compute_attributions_for_sequence
        model_config: Model configuration dict with 'context_len'
        max_plots: Maximum number of plots to show
        window: Window size around splice site
        figsize: Figure size (width, height). Default: (12, 1.5 * max_plots)
        
    Returns:
        matplotlib Figure object
    """
    if figsize is None:
        figsize = (12, min(1.5 * max_plots, 1.5 * len(attrs_dict)))
    
    nrows = min(max_plots, len(attrs_dict))
    fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=figsize, squeeze=False)
    
    context = model_config.get('context_len', 50)
    plot_count = 0
    
    for idx, (id, info) in enumerate(list(attrs_dict.items())):
        if plot_count >= nrows:
            break
        
        sequence = info['sequence']
        attr = info['attr']
        seq_idx = info['seq_idx']
        pos = info['position']
        site_class = info['site_class']
        site_type = info['site_type']
        strand = info.get('strand', '?')
        
        position_coord = context + pos
        start = max(0, position_coord - window)
        end = min(sequence.shape[0], position_coord + window + 1)
        x_vals = np.arange(start, end) - position_coord
        
        # Extract attribution values for actual bases
        attr_values, _, base_names = extract_attribution_values(
            attr, sequence, start, end
        )
        
        # Sum attributions around splice site (+/- 2 bp)
        attr_total = np.sum(attr_values[window - 2:window + 3])
        
        ax = axes[plot_count, 0]
        
        # Create logo plot
        title = f'seq {seq_idx} pos {pos} - {site_type} ({site_class}) on {strand} strand; attr={attr_total:.4f}'
        create_attribution_logo(
            attr_values, base_names, ax,
            x_positions=x_vals,
            title=title,
            xlabel='Distance from splice site',
            ylabel='Input x Gradient'
        )
        ax.set_xticks(np.arange(x_vals[0], x_vals[-1] + 1, 10))
        
        plot_count += 1
    
    fig.tight_layout()
    return fig


def plot_attributions_usage(
    attrs_dict: Dict,
    model_config: Dict,
    conditions: Optional[list] = None,
    max_plots: int = 100,
    window: int = 100,
    figsize: Optional[Tuple[int, int]] = None
) -> plt.Figure:
    """
    Plot attributions for usage (condition-specific) predictions.
    
    Each splice site is plotted for each condition, creating multiple rows per site.
    
    Args:
        attrs_dict: Dictionary of attribution data from compute_attributions_for_sequence
        model_config: Model configuration dict with 'context_len'
        conditions: List of condition names. If None, uses generic names.
        max_plots: Maximum number of plots to show
        window: Window size around splice site
        figsize: Figure size (width, height). Default: (12, 1.5 * max_plots)
        
    Returns:
        matplotlib Figure object
    """
    # First pass: determine n_conditions
    if len(attrs_dict) == 0:
        return None
    
    first_attr = next(iter(attrs_dict.values()))['attr']
    n_conditions = first_attr.shape[2] if len(first_attr.shape) == 3 else 1
    
    # Calculate actual number of rows needed (each site * each condition)
    total_plots = min(max_plots, len(attrs_dict) * n_conditions)
    
    if figsize is None:
        figsize = (12, max(8, min(1.5 * total_plots, 1.5 * 50)))
    
    fig, axes = plt.subplots(nrows=total_plots, ncols=1, figsize=figsize, squeeze=False)
    
    context = model_config.get('context_len', 50)
    plot_count = 0
    
    for idx, (id, info) in enumerate(list(attrs_dict.items())):
        if plot_count >= total_plots:
            break
        
        sequence = info['sequence']
        attr = info['attr']  # Shape: (seq_len, 4, n_conditions)
        seq_idx = info['seq_idx']
        pos = info['position']
        site_class = info['site_class']
        site_type = info['site_type']
        strand = info.get('strand', '?')
        
        position_coord = context + pos
        start = max(0, position_coord - window)
        end = min(sequence.shape[0], position_coord + window + 1)
        x_vals = np.arange(start, end) - position_coord
        
        # Plot each condition for this site
        for cond_idx in range(n_conditions):
            if plot_count >= total_plots:
                break
            
            ax = axes[plot_count, 0]
            
            # Extract attribution values
            attr_values, _, base_names = extract_attribution_values(
                attr, sequence, start, end, condition_idx=cond_idx
            )
            
            # Sum attributions around splice site
            attr_total = np.sum(attr_values[window - 2:window + 3])
            
            # Create condition name
            cond_name = conditions[cond_idx] if conditions else f"condition {cond_idx}"
            
            # Create logo plot
            title = f'seq {seq_idx} pos {pos} - {site_type} ({site_class}) on {strand}; {cond_name}; attr={attr_total:.4f}'
            create_attribution_logo(
                attr_values, base_names, ax,
                x_positions=x_vals,
                title=title,
                xlabel='Distance from splice site',
                ylabel='Input x Gradient'
            )
            ax.set_xticks(np.arange(x_vals[0], x_vals[-1] + 1, 10))
            
            plot_count += 1
    
    fig.tight_layout()
    return fig
