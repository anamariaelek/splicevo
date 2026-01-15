"""
Visualization utilities for sequence attributions using sequence logos.
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import logomaker
from typing import Optional, Dict, Tuple, Union, List
from matplotlib.patches import Rectangle
from splicevo.io.gene_annotation import GTFProcessor

# Fix font rendering: use ASCII minus instead of Unicode to avoid missing glyph warnings
matplotlib.rcParams['axes.unicode_minus'] = False


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
    ylabel: str = 'Input x Gradient',
    ylim: Optional[Tuple[float, float]] = None
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
        ylim: Y-axis limits (min, max). If None, auto-scale.
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
    
    # Validate and apply y-axis limits
    if ylim is not None:
        y_min, y_max = ylim
        # Ensure min and max are different to avoid singular transformation
        if y_min == y_max:
            if y_min == 0:
                ylim = (-0.1, 0.1)
            else:
                # Add 10% padding around the value
                padding = abs(y_min) * 0.1 if y_min != 0 else 0.1
                ylim = (y_min - padding, y_max + padding)
        ax.set_ylim(ylim)
    
    ax.axvline(0, color='red', linestyle='--', linewidth=1, alpha=0.7, label='Target position')


def get_overlapping_transcripts(gtf_source, genome_id: str, chromosome: str, start: int, end: int, strand: str):
    """Get transcripts overlapping a genomic region.
    
    Args:
        gtf_source: Either a path to GTF annotation file (str), a GTFProcessor object, or a list of transcripts
        genome_id: Genome identifier
        chromosome: Chromosome name
        start: Start position (genomic coordinates)
        end: End position (genomic coordinates)
        strand: Strand ('+' or '-')
        
    Returns:
        List of overlapping transcripts
    """
    # Handle both file paths and pre-loaded objects
    if isinstance(gtf_source, str):
        # File path - load and process
        processor = GTFProcessor(gtf_source)
        transcripts = processor.process_gtf(chromosomes=[chromosome])
    elif isinstance(gtf_source, GTFProcessor):
        # GTFProcessor object - call process_gtf
        transcripts = gtf_source.process_gtf(chromosomes=[chromosome])
    elif isinstance(gtf_source, list):
        # Already a list of transcripts - use directly
        transcripts = gtf_source
    else:
        # Try to iterate it (might be a generator or other iterable)
        transcripts = list(gtf_source)
    
    # Ensure we have a list
    if not isinstance(transcripts, list):
        transcripts = list(transcripts)
    
    # Filter transcripts that overlap the region on the correct chromosome
    overlapping = []
    for transcript in transcripts:
        # Check chromosome match
        if hasattr(transcript, 'chromosome') and transcript.chromosome != chromosome:
            continue
        
        # Check strand match
        if transcript.strand != strand:
            continue
        
        # Get transcript boundaries from exons
        transcript_start = transcript.exons['start'].min()
        transcript_end = transcript.exons['end'].max()
        
        # Check for overlap
        if not (transcript_end < start or transcript_start > end):
            overlapping.append(transcript)
    
    return overlapping


def plot_gene_track(ax: plt.Axes, transcripts: list, start: int, end: int, strand: str, max_transcripts: Optional[int] = 5):
    """Plot gene track with exons and introns.
    
    Args:
        ax: Matplotlib axes to plot on
        transcripts: List of Transcript objects to plot
        start: Start position of the region (genomic coordinates)
        end: End position of the region (genomic coordinates)
        strand: Strand ('+' or '-')
        max_transcripts: Maximum number of transcripts to display (default 5, None for all)
    """
    if not transcripts:
        ax.text(0.5, 0.5, 'No genes in region', ha='center', va='center', transform=ax.transAxes)
        ax.set_ylim(0, 1)
        ax.set_xlim(start, end)
        ax.set_yticks([])
        return
    
    # Limit number of transcripts if specified
    if max_transcripts is not None and len(transcripts) > max_transcripts:
        transcripts = transcripts[:max_transcripts]
    
    # Plot each transcript on a separate row
    n_transcripts = len(transcripts)
    y_step = 1.0 / (n_transcripts + 1)
    
    # Make exon height adaptive: larger for fewer transcripts
    exon_height = min(0.4, 0.8 / (n_transcripts + 1))
    half_height = exon_height / 2
    
    for i, transcript in enumerate(transcripts):
        y_pos = 1.0 - (i + 1) * y_step
        
        # Draw intron line spanning the transcript
        transcript_start = transcript.exons['start'].min()
        transcript_end = transcript.exons['end'].max()
        ax.plot([transcript_start, transcript_end], [y_pos, y_pos], 
                color='gray', linewidth=2, zorder=1)
        
        # Draw exons as rectangles
        for _, exon in transcript.exons.iterrows():
            exon_start = exon['start']
            exon_end = exon['end']
            exon_width = exon_end - exon_start
            
            rect = Rectangle((exon_start, y_pos - half_height), exon_width, exon_height,
                           facecolor='blue', edgecolor='darkblue', linewidth=1, zorder=2)
            ax.add_patch(rect)
        
        # Add transcript label with adaptive positioning
        label_x = transcript_start if strand == '+' else transcript_end
        label_offset = max(0.03, exon_height * 0.6)
        ax.text(label_x, y_pos + label_offset, transcript.transcript_id, 
               fontsize=8, ha='left' if strand == '+' else 'right', va='bottom')
    
    ax.set_xlim(start, end)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel('Genomic Position')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)


def plot_attributions_splice(
    attrs_dict: Dict,
    model_config: Dict,
    max_plots: Optional[int] = None,
    window: int = 100,
    figsize: Optional[Tuple[int, int]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    gtf: Optional[Union[str, Dict, GTFProcessor]] = None,
    max_transcripts: Optional[int] = 5
) -> plt.Figure:
    """
    Plot attributions for splice site predictions.

    Args:
        attrs_dict: Dictionary of attribution data from compute_attributions_for_sequence
        model_config: Model configuration dict with 'context_len'
        max_plots: Maximum number of plots to show
        window: Window size around splice site
        figsize: Figure size (width, height). Default: (12, 1.5 * max_plots)
        ylim: Y-axis limits (min, max). If None (default), no limits are set. If 'auto', limits are determined based on data.
        gtf: Optional GTF annotation(s) for plotting gene tracks below attribution plots.
             Can be:
             - A string path to a GTF file (single genome)
             - A GTFProcessor object (pre-loaded, single genome)
             - A dict mapping genome_id to GTF paths or GTFProcessor objects (multi-genome)
             Example: {'human_GRCh37': 'human.gtf', 'mouse_GRCm38': processor_obj}
        max_transcripts: Maximum number of transcripts to display per gene track (default 5, None for all)

    Returns:
        matplotlib Figure object
    """
    if max_plots is None:
        max_plots = len(attrs_dict)

    nrows_base = min(max_plots, len(attrs_dict))
    
    if figsize is None:
        # Standard width, taller when gene tracks included (2 rows per site)
        width = 12
        if gtf:
            height = min(4.0 * nrows_base, 4.0 * len(attrs_dict))
        else:
            height = min(1.5 * nrows_base, 1.5 * len(attrs_dict))
        figsize = (width, height)

    # Create subplot grid with attribution and optional gene tracks stacked vertically
    if gtf:
        # Each site gets 2 rows: one for attribution, one for gene track
        nrows = nrows_base * 2
        fig = plt.figure(figsize=figsize)
        # Use gridspec for flexible subplot sizing (gene tracks shorter)
        import matplotlib.gridspec as gridspec
        # Create height ratios: [attribution, gene, attribution, gene, ...]
        height_ratios = [2, 1] * nrows_base
        gs = gridspec.GridSpec(nrows, 1, height_ratios=height_ratios, figure=fig, hspace=0.7)
        axes = np.empty((nrows_base, 2), dtype=object)
        for i in range(nrows_base):
            axes[i, 0] = fig.add_subplot(gs[i * 2])      # Attribution plot
            axes[i, 1] = fig.add_subplot(gs[i * 2 + 1])  # Gene track below
    else:
        fig, axes = plt.subplots(nrows=nrows_base, ncols=1, figsize=figsize, squeeze=False)

    context = model_config.get('context_len', 50)

    # First pass: collect all attribution values to determine global y-axis range
    all_attr_values = []
    plot_data = []

    for idx, (id, info) in enumerate(list(attrs_dict.items())):
        if idx >= nrows:
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

        all_attr_values.extend(attr_values)

        # Sum attributions around splice site (+/- 2 bp)
        attr_total = np.sum(attr_values[window - 2:window + 3])
        
        # Extract additional metadata if available
        metadata = info.get('metadata', {})
        genomic_coord = metadata.get('genomic_coord')
        chromosome = metadata.get('chromosome')
        genome_id = metadata.get('genome_id')
        
        plot_data.append({
            'attr_values': attr_values,
            'base_names': base_names,
            'x_vals': x_vals,
            'seq_idx': seq_idx,
            'pos': pos,
            'site_type': site_type,
            'site_class': site_class,
            'strand': strand,
            'attr_total': attr_total,
            'genomic_coord': genomic_coord,
            'chromosome': chromosome,
            'genome_id': genome_id
        })
    # Calculate global y-axis limits if not provided
    if ylim == 'auto':
        y_min = np.min(all_attr_values) if all_attr_values else 0
        y_max = np.max(all_attr_values) if all_attr_values else 1
        # Ensure min and max are different
        if y_min == y_max:
            if y_min == 0:
                ylim = (-0.1, 0.1)
            else:
                padding = abs(y_min) * 0.1 if y_min != 0 else 0.1
                ylim = (y_min - padding, y_max + padding)
        else:
            # Add 10% padding
            y_padding = (y_max - y_min) * 0.1
            ylim = (y_min - y_padding, y_max + y_padding)
    
    # Second pass: create plots with consistent y-axis
    for plot_count, data in enumerate(plot_data):
        ax = axes[plot_count, 0]
        
        # Create logo plot
        genomic_info = ""
        if 'genomic_coord' in data and data['genomic_coord'] is not None:
            genome_id = data.get('genome_id', '?')
            chr_name = data.get('chromosome', '?')
            genomic_coord = data['genomic_coord']
            strand_symbol = '+' if data['strand'] == '+' else '-'
            genomic_info = f" | {genome_id} {chr_name}:{genomic_coord} ({strand_symbol})"
        
        site_info = f"{data['site_type']} ({data['site_class']})"
        title = f"seq {data['seq_idx']} pos {data['pos']}{genomic_info} | {site_info} | attr={data['attr_total']:.4f}"
        create_attribution_logo(
            data['attr_values'], data['base_names'], ax,
            x_positions=data['x_vals'],
            title=title,
            xlabel='Distance from splice site',
            ylabel='Input x Gradient',
            ylim=ylim
        )
        # Adaptive tick spacing to avoid overcrowding
        x_range = data['x_vals'][-1] - data['x_vals'][0]
        if x_range <= 100:
            tick_step = 10
        elif x_range <= 200:
            tick_step = 20
        elif x_range <= 400:
            tick_step = 50
        else:
            tick_step = 100
        ax.set_xticks(np.arange(data['x_vals'][0], data['x_vals'][-1] + 1, tick_step))
        
        # Add gene track if gtf provided
        if gtf and data['genomic_coord'] is not None:
            ax_gene = axes[plot_count, 1]
            genome_id = data.get('genome_id', '')
            chromosome = data.get('chromosome', '')
            
            # Determine which GTF source to use
            if isinstance(gtf, dict):
                # Multi-genome support: lookup by genome_id
                current_gtf = gtf.get(genome_id)
                if current_gtf is None:
                    ax_gene.text(0.5, 0.5, f'No GTF for\\ngenome: {genome_id}', 
                               ha='center', va='center', transform=ax_gene.transAxes, fontsize=8)
                    ax_gene.set_xlim(0, 1)
                    ax_gene.set_ylim(0, 1)
                    ax_gene.set_xticks([])
                    ax_gene.set_yticks([])
                    continue
            else:
                # Single GTF source for all genomes
                current_gtf = gtf
            
            # Calculate genomic region based on window
            position_coord = context + data['pos']
            start_offset = max(0, position_coord - window)
            end_offset = min(len(data['attr_values']), position_coord + window + 1)
            
            # Map to genomic coordinates
            genomic_start = data['genomic_coord'] - (position_coord - start_offset)
            genomic_end = data['genomic_coord'] + (end_offset - position_coord)
            
            try:
                transcripts = get_overlapping_transcripts(
                    current_gtf, genome_id, chromosome, 
                    genomic_start, genomic_end, data['strand']
                )
                plot_gene_track(ax_gene, transcripts, genomic_start, genomic_end, data['strand'], max_transcripts)
            except Exception as e:
                ax_gene.text(0.5, 0.5, f'Error loading genes:\\n{str(e)}', 
                           ha='center', va='center', transform=ax_gene.transAxes, fontsize=8)
                ax_gene.set_xlim(0, 1)
                ax_gene.set_ylim(0, 1)
                ax_gene.set_xticks([])
                ax_gene.set_yticks([])
    
    #fig.tight_layout()
    return fig


def plot_attributions_usage(
    attrs_dict: Dict,
    model_config: Dict,
    conditions: Optional[list] = None,
    conditions_to_plot: Optional[list] = None,
    max_plots: Optional[int] = None,
    window: int = 100,
    figsize: Optional[Tuple[int, int]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    gtf: Optional[Union[str, Dict, GTFProcessor]] = None,
    max_transcripts: Optional[int] = 5
) -> plt.Figure:
    """
    Plot attributions for usage (condition-specific) predictions.
    
    Each splice site is plotted for each condition, creating multiple rows per site.
    
    Args:
        attrs_dict: Dictionary of attribution data from compute_attributions_for_sequence
        model_config: Model configuration dict with 'context_len'
        conditions: List of condition names. If None, uses generic names.
        conditions_to_plot: List of condition indices to plot. If None, plots all conditions.
        max_plots: Maximum number of plots to show
        window: Window size around splice site
        figsize: Figure size (width, height). Default: (12, 1.5 * max_plots)
        ylim: Y-axis limits (min, max). If None (default), no limits are set. If 'auto', auto-scales based on data.
        gtf: Optional GTF annotation(s) for plotting gene tracks below attribution plots.
             Can be:
             - A string path to a GTF file (single genome)
             - A GTFProcessor object (pre-loaded, single genome)
             - A dict mapping genome_id to GTF paths or GTFProcessor objects (multi-genome)
             Example: {'human_GRCh37': 'human.gtf', 'mouse_GRCm38': processor_obj}
        max_transcripts: Maximum number of transcripts to display per gene track (default 5, None for all)
        
    Returns:
        matplotlib Figure object
    """
    # First pass: determine n_conditions
    if len(attrs_dict) == 0:
        return None
    
    first_attr = next(iter(attrs_dict.values()))['attr']
    n_conditions = first_attr.shape[2] if len(first_attr.shape) == 3 else 1
    
    # Determine which conditions to plot
    if conditions_to_plot is None:
        conditions_to_plot = list(range(n_conditions))
    
    # Calculate actual number of rows needed (each site * each condition)
    total_plots = min(max_plots, len(attrs_dict) * len(conditions_to_plot))
    
    if figsize is None:
        # Standard width, taller when gene tracks included
        width = 8
        if gtf:
            height = max(8, min(4.0 * total_plots, 4.0 * 50))
        else:
            height = max(8, min(1.5 * total_plots, 1.5 * 50))
        figsize = (width, height)
    
    # Create subplot grid with attribution and optional gene tracks stacked vertically
    if gtf:
        # Each plot gets 2 rows: one for attribution, one for gene track
        nrows = total_plots * 2
        fig = plt.figure(figsize=figsize)
        import matplotlib.gridspec as gridspec
        # Create height ratios: [attribution, gene, attribution, gene, ...]
        height_ratios = [2, 1] * total_plots
        gs = gridspec.GridSpec(nrows, 1, height_ratios=height_ratios, figure=fig, hspace=0.7)
        axes = np.empty((total_plots, 2), dtype=object)
        for i in range(total_plots):
            axes[i, 0] = fig.add_subplot(gs[i * 2])      # Attribution plot
            axes[i, 1] = fig.add_subplot(gs[i * 2 + 1])  # Gene track below
    else:
        fig, axes = plt.subplots(nrows=total_plots, ncols=1, figsize=figsize, squeeze=False)
    
    context = model_config.get('context_len', 50)
    
    # First pass: collect all attribution values to determine global y-axis range
    all_attr_values = []
    plot_data = []
    
    for idx, (id, info) in enumerate(list(attrs_dict.items())):
        if len(plot_data) >= total_plots:
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
        
        # Collect data for each condition for this site
        for cond_idx in conditions_to_plot:
            if len(plot_data) >= total_plots:
                break
            
            # Extract attribution values
            attr_values, _, base_names = extract_attribution_values(
                attr, sequence, start, end, condition_idx=cond_idx
            )
            
            all_attr_values.extend(attr_values)
            
            # Sum attributions around splice site
            attr_total = np.sum(attr_values[window - 2:window + 3])
            
            # Create condition name
            cond_name = conditions[cond_idx] if conditions else f"condition {cond_idx}"
            
            # Extract additional metadata if available
            metadata = info.get('metadata', {})
            genomic_coord = metadata.get('genomic_coord')
            chromosome = metadata.get('chromosome')
            genome_id = metadata.get('genome_id')
            
            plot_data.append({
                'attr_values': attr_values,
                'base_names': base_names,
                'x_vals': x_vals,
                'seq_idx': seq_idx,
                'pos': pos,
                'site_type': site_type,
                'site_class': site_class,
                'strand': strand,
                'cond_name': cond_name,
                'attr_total': attr_total,
                'genomic_coord': genomic_coord,
                'chromosome': chromosome,
                'genome_id': genome_id
            })
    
    # Calculate global y-axis limits if not provided
    if ylim == 'auto':
        y_min = np.min(all_attr_values) if all_attr_values else 0
        y_max = np.max(all_attr_values) if all_attr_values else 1
        # Ensure min and max are different
        if y_min == y_max:
            if y_min == 0:
                ylim = (-0.1, 0.1)
            else:
                padding = abs(y_min) * 0.1 if y_min != 0 else 0.1
                ylim = (y_min - padding, y_max + padding)
        else:
            # Add 10% padding
            y_padding = (y_max - y_min) * 0.1
            ylim = (y_min - y_padding, y_max + y_padding)
    
    # Second pass: create plots with consistent y-axis
    for plot_count, data in enumerate(plot_data):
        ax = axes[plot_count, 0]
        
        # Create logo plot
        genomic_info = ""
        if 'genomic_coord' in data and data['genomic_coord'] is not None:
            genome_id = data.get('genome_id', '?')
            chr_name = data.get('chromosome', '?')
            genomic_coord = data['genomic_coord']
            strand_symbol = '+' if data['strand'] == '+' else '-'
            genomic_info = f" | {genome_id} {chr_name}:{genomic_coord} ({strand_symbol})"
        
        site_info = f"{data['site_type']} ({data['site_class']})"
        title = f"seq {data['seq_idx']} pos {data['pos']}{genomic_info} | {site_info} | {data['cond_name']} | attr={data['attr_total']:.4f}"
        create_attribution_logo(
            data['attr_values'], data['base_names'], ax,
            x_positions=data['x_vals'],
            title=title,
            xlabel='Distance from splice site',
            ylabel='Input x Gradient',
            ylim=ylim
        )
        # Adaptive tick spacing to avoid overcrowding
        x_range = data['x_vals'][-1] - data['x_vals'][0]
        if x_range <= 100:
            tick_step = 10
        elif x_range <= 200:
            tick_step = 20
        elif x_range <= 400:
            tick_step = 50
        else:
            tick_step = 100
        ax.set_xticks(np.arange(data['x_vals'][0], data['x_vals'][-1] + 1, tick_step))
        
        # Add gene track if gtf provided
        if gtf and data['genomic_coord'] is not None:
            ax_gene = axes[plot_count, 1]
            genome_id = data.get('genome_id', '')
            chromosome = data.get('chromosome', '')
            
            # Determine which GTF source to use
            if isinstance(gtf, dict):
                # Multi-genome support: lookup by genome_id
                current_gtf = gtf.get(genome_id)
                if current_gtf is None:
                    ax_gene.text(0.5, 0.5, f'No GTF for\\ngenome: {genome_id}', 
                               ha='center', va='center', transform=ax_gene.transAxes, fontsize=8)
                    ax_gene.set_xlim(0, 1)
                    ax_gene.set_ylim(0, 1)
                    ax_gene.set_xticks([])
                    ax_gene.set_yticks([])
                    continue
            else:
                # Single GTF source for all genomes
                current_gtf = gtf
            
            # Calculate genomic region based on window
            position_coord = context + data['pos']
            start_offset = max(0, position_coord - window)
            end_offset = min(first_attr.shape[0], position_coord + window + 1)
            
            # Map to genomic coordinates
            genomic_start = data['genomic_coord'] - (position_coord - start_offset)
            genomic_end = data['genomic_coord'] + (end_offset - position_coord)
            
            try:
                transcripts = get_overlapping_transcripts(
                    current_gtf, genome_id, chromosome, 
                    genomic_start, genomic_end, data['strand']
                )
                plot_gene_track(ax_gene, transcripts, genomic_start, genomic_end, data['strand'], max_transcripts)
            except Exception as e:
                ax_gene.text(0.5, 0.5, f'Error loading genes:\n{str(e)}', 
                           ha='center', va='center', transform=ax_gene.transAxes, fontsize=8)
                ax_gene.set_xlim(0, 1)
                ax_gene.set_ylim(0, 1)
                ax_gene.set_xticks([])
                ax_gene.set_yticks([])
    
    fig.tight_layout()
    return fig


def plot_attributions_splice_from_result(
    result: Dict,
    model_config: Dict,
    max_plots: Optional[int] = None,
    window: int = 100,
    figsize: Optional[Tuple[int, int]] = None,
    ylim: Optional[Union[str, Tuple[float, float]]] = None,
    gtf: Optional[Union[str, Dict, GTFProcessor]] = None,
    max_transcripts: Optional[int] = 5
) -> plt.Figure:
    """
    Plot splice site attributions directly from flexible API result.
    
    This function accepts the result dictionary from compute_attributions_splice()
    directly, without needing to convert to the legacy attrs_dict format.
    
    Args:
        result: Result dictionary from compute_attributions_splice() with 'attributions' key
        model_config: Model configuration dict with 'context_len'
        max_plots: Maximum number of plots to show
        window: Window size around splice site
        figsize: Figure size (width, height)
        ylim: Y-axis limits ('auto', None, or tuple)
        gtf: Optional GTF annotation(s) for plotting gene tracks below attribution plots.
             Can be:
             - A string path to a GTF file (single genome)
             - A GTFProcessor object (pre-loaded, single genome)
             - A dict mapping genome_id to GTF paths or GTFProcessor objects (multi-genome)
             Example: {'human_GRCh37': 'human.gtf', 'mouse_GRCm38': processor_obj}
        max_transcripts: Maximum number of transcripts to display per gene track (default 5, None for all)
        
    Returns:
        matplotlib Figure object
    """
    # Convert flexible API format to legacy format
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
            'strand': attr_data['metadata'].get('strand', '?'),
            'metadata': attr_data['metadata']  # Include full metadata for genomic coordinates
        }
    
    return plot_attributions_splice(attrs_dict, model_config, max_plots, window, figsize, ylim, gtf, max_transcripts)


def plot_attributions_usage_from_result(
    result: Dict,
    model_config: Dict,
    conditions: Optional[list] = None,
    conditions_to_plot: Optional[list] = None,
    sites_to_plot: Optional[Union[str, List[str], List[int], range]] = None,
    max_plots: int = 100,
    window: int = 100,
    figsize: Optional[Tuple[int, int]] = None,
    ylim: Optional[Union[str, Tuple[float, float]]] = None,
    gtf: Optional[Union[str, Dict, GTFProcessor]] = None,
    max_transcripts: Optional[int] = 5
) -> plt.Figure:
    """
    Plot usage attributions directly from flexible API result.
    
    This function accepts the result dictionary from compute_attributions_usage()
    directly, without needing to convert to the legacy attrs_dict format.
    
    Args:
        result: Result dictionary from compute_attributions_usage() with 'attributions' key
        model_config: Model configuration dict with 'context_len'
        conditions: List of condition names (optional)
        conditions_to_plot: List of condition indices to plot
        sites_to_plot: Sites to plot - can be:
            - Single site ID (string like '279_106')
            - List of site IDs (strings like ['279_106', '324_281'])
            - Range/list of indices (e.g., range(5) to plot first 5 sites)
            - None (default) to plot all sites
        max_plots: Maximum number of plots to show
        window: Window size around splice site
        figsize: Figure size (width, height)
        ylim: Y-axis limits ('auto', None, or tuple)
        gtf: Optional GTF annotation(s) for plotting gene tracks below attribution plots.
             Can be:
             - A string path to a GTF file (single genome)
             - A GTFProcessor object (pre-loaded, single genome)
             - A dict mapping genome_id to GTF paths or GTFProcessor objects (multi-genome)
             Example: {'human_GRCh37': 'human.gtf', 'mouse_GRCm38': processor_obj}
        max_transcripts: Maximum number of transcripts to display per gene track (default 5, None for all)
        
    Returns:
        matplotlib Figure object
    """
    # Convert flexible API format to legacy format
    attrs_dict = {}
    site_ids = list(result['attributions'].keys())
    
    # Handle sites_to_plot
    if sites_to_plot is not None:
        # Handle single string (site ID)
        if isinstance(sites_to_plot, str):
            site_ids = [sites_to_plot] if sites_to_plot in result['attributions'] else []
        elif isinstance(sites_to_plot, (list, range)):
            # Check if it's a range of indices or site IDs
            if len(sites_to_plot) > 0:
                first_item = sites_to_plot[0] if isinstance(sites_to_plot, list) else next(iter(sites_to_plot))
                if isinstance(first_item, int):
                    # It's indices - select by position
                    site_ids = [site_ids[i] for i in sites_to_plot if i < len(site_ids)]
                else:
                    # It's site IDs - filter to only those provided
                    site_ids = [sid for sid in site_ids if sid in sites_to_plot]
    
    # Build attrs_dict with selected sites
    for site_id in site_ids:
        if site_id in result['attributions']:
            attr_data = result['attributions'][site_id]
            attrs_dict[site_id] = {
                'id': attr_data['id'],
                'seq_idx': attr_data['seq_idx'],
                'position': attr_data['position'],
                'sequence': attr_data['sequence'],
                'attr': attr_data['attribution'],
                'site_class': attr_data['site_class'],
                'site_type': attr_data['site_type'],
                'strand': attr_data['metadata'].get('strand', '?'),
                'metadata': attr_data['metadata']  # Include full metadata for genomic coordinates
            }
    
    return plot_attributions_usage(attrs_dict, model_config, conditions, 
                                  conditions_to_plot, max_plots, window, figsize, ylim, gtf, max_transcripts)