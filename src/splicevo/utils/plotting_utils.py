"""Plotting utilities for splice site usage visualization."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Optional
from matplotlib.lines import Line2D


def plot_splice_site_usage(
    analysis_results: List[Dict],
    site_type: Optional[str] = None,
    tissue_colors: Optional[Dict[str, str]] = None,
    plot_zeroes: Optional[bool] = True,
    figsize: tuple = (7, 5),
    title_suffix: str = '',
    grid_cols: Optional[int] = None
) -> plt.Figure:
    """
    Plot SSE usage across tissues (lines) and timepoints (x-axis) for splice sites.
    Creates a grid of subplots, one for each splice site.
    
    Args:
        analysis_results: List of analysis data dicts from analyze_splice_sites(return_data=True)
        site_type: 'donors', 'acceptors', or None for all splice sites
        tissue_colors: Optional dict mapping tissue names to colors
        figsize: Figure size (width, height)
        title_suffix: Additional text for the title
        grid_cols: Number of columns in the subplot grid (None for auto, max 3)
        
    Returns:
        Tuple of (matplotlib Figure object, dataframe)
    """
    if not analysis_results or not analysis_results[0].get('usage_data'):
        print("No usage data available for plotting.")
        return None, None
    
    # Collect all usage data across windows
    all_sites_data = []
    site_info = []  # Track unique sites with metadata
    
    for window_result in analysis_results:
        window_idx = window_result['window_index']
        gene_id = window_result['gene_id']
        chrom = window_result['chromosome']
        genomic_start = window_result['start']
        genomic_end = window_result['end']
        strand = window_result['strand']
        genome_id = window_result.get('genome_id', 'unknown')  # Add genome_id extraction
        
        usage_data = window_result.get('usage_data', {})
        
        # Determine which site types to include
        site_types_to_process = []
        if site_type is None:
            site_types_to_process = ['donors', 'acceptors']
        else:
            site_types_to_process = [site_type]
        
        for current_site_type in site_types_to_process:
            sites = usage_data.get(current_site_type, [])
            
            for site_data in sites:
                pos = site_data['position']
                # Genomic position of this splice site
                genomic_pos = genomic_start + pos
                
                site_id = f"{chrom}:{genomic_pos}:{strand}:{current_site_type[0].upper()}:{window_idx}:{pos}"
                
                for cond_key, cond_data in site_data.get('conditions', {}).items():
                    all_sites_data.append({
                        'site_id': site_id,
                        'window_index': window_idx,
                        'gene_id': gene_id,
                        'chromosome': chrom,
                        'genomic_start': genomic_start,
                        'genomic_end': genomic_end,
                        'genomic_position': genomic_pos,
                        'data_position': pos,
                        'strand': strand,
                        'site_type': current_site_type[0].upper(),  # 'D' or 'A'
                        'tissue': cond_data['tissue'],
                        'timepoint': cond_data['timepoint'],
                        'display_name': cond_data['display_name'],
                        'sse': cond_data['sse']
                    })
                
                # Store site info
                site_info_dict = {
                    'site_id': site_id,
                    'genome_id': genome_id,
                    'gene_id': gene_id,
                    'chromosome': chrom,
                    'genomic_position': genomic_pos,
                    'data_position': pos,
                    'strand': strand,
                    'window_index': window_idx,
                    'site_type': current_site_type[0].upper()
                }
                
                if site_info_dict not in site_info:
                    site_info.append(site_info_dict)
    
    if not all_sites_data:
        print(f"No splice site data found for plotting.")
        return None, None
    
    df = pd.DataFrame(all_sites_data)
    
    # Get unique tissues
    tissues = sorted(df['tissue'].unique())
    
    # Convert timepoints to numeric
    def convert_timepoint(x):
        x_str = str(x)
        try:
            return int(x_str)
        except ValueError:
            try:
                return float(x_str)
            except ValueError:
                return np.nan
    
    df['timepoint_numeric'] = df['timepoint'].apply(convert_timepoint)
    
    timepoints_numeric = sorted(df['timepoint_numeric'].unique())
    # Convert to integers for plotting
    timepoints_numeric = np.array(timepoints_numeric, dtype=int)
    
    # Create grid of subplots
    n_sites = len(site_info)
    n_cols = grid_cols if grid_cols is not None else min(3, n_sites)  # Use provided cols or auto (max 3)
    n_rows = (n_sites + n_cols - 1) // n_cols  # Ceiling division
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    # Flatten axes array for easier iteration
    if n_sites == 1:
        axes = np.array([axes])
    else:
        axes = axes.flatten()
    
    # Color map for tissues
    colors = plt.cm.Set2(np.linspace(0, 1, len(tissues)))
    if tissue_colors is None:
        tissue_colors = {tissue: colors[i] for i, tissue in enumerate(tissues)}
    
    # Plot each site
    for idx, site_info_dict in enumerate(site_info):
        ax = axes[idx]
        site_id = site_info_dict['site_id']
        site_df = df[df['site_id'] == site_id]
        
        # Plot each tissue as a line with all SSE values
        for tissue in tissues:
            tissue_data = site_df[site_df['tissue'] == tissue]
            
            # Collect all SSE values for each timepoint
            timepoint_data = {}
            
            for tp_num in timepoints_numeric:
                tp_data = tissue_data[tissue_data['timepoint_numeric'] == int(tp_num)]['sse'].values
                # For plotting, only include SSE values > 0
                if not plot_zeroes:
                    tp_data = tp_data[tp_data > 0]
                if len(tp_data) > 0:
                    timepoint_data[tp_num] = tp_data

            if timepoint_data:
                # Plot all individual points for each timepoint
                for tp_num, sse_values in timepoint_data.items():
                    # Add jitter to x-axis to avoid overlapping points
                    x_jitter = np.random.normal(tp_num, 0.05, size=len(sse_values))
                    ax.scatter(x_jitter, sse_values, 
                              color=tissue_colors[tissue], alpha=0.6, s=20)
                
                # Connect tissue with a line for visual clarity
                tp_nums = sorted(timepoint_data.keys())
                means = [np.mean(timepoint_data[tp]) for tp in tp_nums]
                ax.plot(tp_nums, means, label=tissue, color=tissue_colors[tissue], 
                       linewidth=2, zorder=10)
        
        # Create detailed title
        genome_id = site_info_dict['genome_id']
        chrom = site_info_dict['chromosome']
        genomic_pos = site_info_dict['genomic_position']
        strand = site_info_dict['strand']
        window_idx = site_info_dict['window_index']
        data_pos = site_info_dict['data_position']
        site_type_label = site_info_dict['site_type']
        
        title_text = f'{genome_id} {chrom}:{genomic_pos}({strand})\nidx:{int(window_idx)} pos:{int(data_pos)} | {site_type_label}'
        
        # Formatting for each subplot
        ax.set_xlabel('Timepoint', fontsize=10)
        ax.set_ylabel('SSE', fontsize=10)
        ax.set_title(title_text, fontsize=9)
        ax.set_ylim(-0.05, 1.05)
        
        # Dynamically adjust x-axis labels to fit
        n_timepoints = len(timepoints_numeric)
        if n_timepoints <= 5:
            # Show all labels
            tick_step = 1
        elif n_timepoints <= 10:
            # Show every other label
            tick_step = 2
        elif n_timepoints <= 20:
            # Show every 3rd label
            tick_step = 3
        else:
            # Show every 5th label
            tick_step = 5
        
        # Set all timepoints as ticks to ensure consistent x-axis across subplots
        ax.set_xticks(timepoints_numeric)
        # Set x-axis limits to show full range
        if len(timepoints_numeric) > 0:
            ax.set_xlim(timepoints_numeric[0] - 1, timepoints_numeric[-1] + 1)
        
        # Create tick labels with stepping
        tick_labels = [str(int(tp)) for tp in timepoints_numeric]
        visible_labels = [tick_labels[i] if i % tick_step == 0 else '' for i in range(len(tick_labels))]
        ax.set_xticklabels(visible_labels)
        ax.grid(True, alpha=0.3, axis='y')
    
    # Hide unused subplots
    for idx in range(n_sites, len(axes)):
        axes[idx].set_visible(False)
    
    # Add overall title
    site_type_label = site_type.capitalize() if site_type else ''
    fig.suptitle(f'SSE Usage Across Tissues and Timepoints {title_suffix}', 
                fontsize=14, y=0.995)
    
    # Create common legend outside plots
    handles = [plt.Line2D([0], [0], color=tissue_colors[tissue], linewidth=2, label=tissue) 
               for tissue in tissues]
    fig.legend(handles=handles, loc='center left', bbox_to_anchor=(1.0, 0.5), 
              title='Tissue', framealpha=0.9, fontsize=9)
    
    plt.tight_layout()
    return fig, df
