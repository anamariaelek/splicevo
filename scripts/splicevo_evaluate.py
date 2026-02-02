"""Model evaluation script for splice site classification and usage prediction."""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import json
from typing import Dict, Optional, Tuple
import sys
import time
from datetime import datetime
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, auc, mean_squared_error, confusion_matrix
from matplotlib import colors as mpl_colors
from scipy.stats import pearsonr, spearmanr

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from splicevo.utils.data_utils import load_processed_data, load_predictions


def setup_logging(log_file: Optional[str] = None, quiet: bool = False):
    """Setup logging to file and stdout."""
    def log_fn(msg: str):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted_msg = f"[{timestamp}] {msg}"
        if not quiet:
            print(formatted_msg)
        if log_file:
            with open(log_file, 'a') as f:
                f.write(formatted_msg + '\n')
    return log_fn


def evaluate_splice_site_classification(
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    pred_probs: np.ndarray,
    output_dir: Path,
    log_fn=print
) -> Dict:
    """Evaluate splice site classification performance."""
    log_fn("="*60)
    log_fn("SPLICE SITE CLASSIFICATION EVALUATION")
    log_fn("="*60)
    
    results = {}
    
    # PR-AUC scores
    log_fn("Calculating Precision-Recall AUC scores...")
    pr_auc_scores = {}
    class_labels = {0: 'no splice site', 1: 'donor', 2: 'acceptor'}
    class_colors = {0: 'tab:blue', 1: 'tab:orange', 2: 'tab:green'}
    
    total_positions = true_labels.shape[0] * true_labels.shape[1]
    
    plt.figure(figsize=(5, 4))
    for class_idx in range(3):
        color = class_colors[class_idx]
        label = class_labels[class_idx]
        
        if class_idx == 0:
            y_true = (true_labels != class_idx).astype(int).reshape(-1)
            y_scores = 1 - pred_probs[:, :, class_idx].reshape(-1)
            freq = np.sum(true_labels != class_idx) / total_positions
        else:
            y_true = (true_labels == class_idx).astype(int).reshape(-1)
            y_scores = pred_probs[:, :, class_idx].reshape(-1)
            freq = np.sum(true_labels == class_idx) / total_positions
        
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        pr_auc = auc(recall, precision)
        pr_auc_scores[class_idx] = pr_auc
        
        plt.plot(
            recall, precision,
            label=f"{label} (AUC = {pr_auc:.3f})",
            linewidth=2,
            color=color
        )
    
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title("Precision–Recall Curves for Splice Site Classes", fontsize=14)
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    pr_auc_plot = output_dir / "splice_site_pr_auc.png"
    plt.savefig(pr_auc_plot, dpi=150)
    plt.close()
    log_fn(f"Saved PR-AUC plot to {pr_auc_plot}")
    
    for c, auc_score in pr_auc_scores.items():
        log_fn(f"  PR-AUC for {class_labels[c]}: {auc_score:.4f}")
    
    results['pr_auc'] = pr_auc_scores
    
    # Confusion matrix
    log_fn("Generating confusion matrix...")
    cm = confusion_matrix(true_labels.flatten(), pred_labels.flatten(), labels=[0, 1, 2])
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['Non-splice', 'Donor', 'Acceptor'],
        yticklabels=['Non-splice', 'Donor', 'Acceptor'],
        norm=mpl_colors.LogNorm()
    )
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix for Splice Site Predictions')
    plt.tight_layout()
    
    confusion_plot = output_dir / "confusion_matrix.png"
    plt.savefig(confusion_plot, dpi=150)
    plt.close()
    log_fn(f"Saved confusion matrix to {confusion_plot}")
    
    # Probability distributions for correct/incorrect predictions
    log_fn("Plotting probability distributions...")
    correct_donor_probs = pred_probs[(true_labels == 1) & (pred_labels == 1), 1]
    incorrect_donor_probs = pred_probs[(true_labels != 1) & (pred_labels == 1), 1]
    correct_acceptor_probs = pred_probs[(true_labels == 2) & (pred_labels == 2), 2]
    incorrect_acceptor_probs = pred_probs[(true_labels != 2) & (pred_labels == 2), 2]
    
    plt.figure(figsize=(9, 4))
    plt.subplot(1, 2, 1)
    sns.kdeplot(correct_donor_probs, label=f'TP Donors (n={len(correct_donor_probs):,})', color='g')
    sns.kdeplot(incorrect_donor_probs, label=f'FP Donors (n={len(incorrect_donor_probs):,})', color='r')
    plt.title('Donor Splice Site Prediction Probabilities')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Density')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    sns.kdeplot(correct_acceptor_probs, label=f'TP Acceptors (n={len(correct_acceptor_probs):,})', color='g')
    sns.kdeplot(incorrect_acceptor_probs, label=f'FP Acceptors (n={len(incorrect_acceptor_probs):,})', color='r')
    plt.title('Acceptor Splice Site Prediction Probabilities')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    
    prob_dist_plot = output_dir / "probability_distributions.png"
    plt.savefig(prob_dist_plot, dpi=150)
    plt.close()
    log_fn(f"Saved probability distributions to {prob_dist_plot}")
    
    # Top-k accuracy
    log_fn("Calculating top-k accuracy...")
    def top_k_accuracy(y_true, y_scores):
        k = np.sum(y_true)
        if k == 0:
            return 0.0
        threshold = np.sort(y_scores)[-k]
        y_pred = (y_scores >= threshold).astype(int)
        accuracy = np.sum((y_pred == 1) & (y_true == 1)) / k
        return accuracy
    
    top_k_acc = {}
    for class_idx in range(3):
        if class_idx == 0:
            y_true = (true_labels != class_idx).astype(int)
            y_scores = 1 - pred_probs[:, :, class_idx]
        else:
            y_true = (true_labels == class_idx).astype(int)
            y_scores = pred_probs[:, :, class_idx]
        
        y_true = y_true.reshape(-1)
        y_scores = y_scores.reshape(-1)
        acc = top_k_accuracy(y_true, y_scores)
        top_k_acc[class_idx] = acc
        log_fn(f"  Top-k accuracy for {class_labels[class_idx]}: {acc:.4f}")
    
    results['top_k_acc'] = top_k_acc
    
    return results


def evaluate_splice_usage_regression(
    true_sse: np.ndarray,
    pred_sse: np.ndarray,
    species_ids: np.ndarray,
    species_id_to_name: Dict,
    meta: Dict,
    output_dir: Path,
    log_fn=print
) -> Dict:
    """Evaluate splice usage (SSE) prediction performance."""
    log_fn("="*60)
    log_fn("SPLICE USAGE REGRESSION EVALUATION")
    log_fn("="*60)
    
    results = {}
    
    # Calculate MSE for each species and condition
    log_fn("Calculating MSE for each species and condition...")
    conds = meta['conditions']
    mse_dict = {}
    
    for sp in set(species_ids):
        sp_indices = [i for i, s in enumerate(species_ids) if s == sp]
        sp_n = species_id_to_name[sp]
        
        for i, _ in enumerate(conds):
            id = f"{sp_n}_{conds[i]}"
            true_sse_ = true_sse[sp_indices, :, i]
            pred_sse_ = pred_sse[sp_indices, :, i]
            
            # Replace NaN values
            true_sse_ = np.nan_to_num(true_sse_)
            mask = ~np.isnan(true_sse_)
            
            true_sse_vals = true_sse_[mask]
            pred_sse_vals = pred_sse_[mask]
            
            if len(true_sse_vals) > 0:
                mse = mean_squared_error(true_sse_vals, pred_sse_vals)
                mse_dict[id] = mse
    
    # Create MSE dataframe
    mse_df = pd.DataFrame(list(mse_dict.items()), columns=['Sample', 'MSE'])
    mse_df[['Species', 'Tissue', 'Timepoint']] = mse_df['Sample'].str.split('_', expand=True)
    mse_df['Timepoint'] = mse_df['Timepoint'].astype(int)
    mse_df.sort_values(['Species', 'Tissue', "Timepoint"], ascending=True, inplace=True)
    
    for _, row in mse_df.iterrows():
        rmse = np.sqrt(row['MSE'])
        log_fn(f"  {row['Sample']}: RMSE = {rmse:.4f}")
    
    results['mse_df'] = mse_df
    
    # Plot RMSE by species
    log_fn("Plotting RMSE by species...")
    plt.figure(figsize=(9, 3))
    species_list = mse_df['Species'].unique()
    
    for i, species in enumerate(species_list):
        plt.subplot(1, len(species_list), i + 1)
        for tissue in mse_df[mse_df['Species'] == species]['Tissue'].unique():
            tissue_data = mse_df[(mse_df['Species'] == species) & (mse_df['Tissue'] == tissue)]
            plt.plot(
                tissue_data['Timepoint'].astype(int),
                np.sqrt(tissue_data['MSE']),
                marker='o',
                label=tissue
            )
        
        plt.title(f'SSE prediction: {species}', fontsize=12)
        plt.xlabel('Timepoint', fontsize=10)
        plt.ylabel('RMSE', fontsize=10)
        plt.ylim(0, np.sqrt(mse_df['MSE']).max() * 1.1)
        plt.grid()
    
    plt.tight_layout()
    plt.legend(title='Tissue', fontsize=8, bbox_to_anchor=(1.05, 0.5), loc='center left')
    
    rmse_plot = output_dir / "splice_usage_rmse.png"
    plt.savefig(rmse_plot, dpi=150, bbox_inches='tight')
    plt.close()
    log_fn(f"Saved RMSE plot to {rmse_plot}")
    
    return results


def prepare_matched_positions(
    true_labels: np.ndarray,
    true_sse: np.ndarray,
    pred_sse: np.ndarray,
    condition_mask: np.ndarray,
    meta_df: pd.DataFrame,
    species_id_to_name: Dict,
    meta: Dict,
    log_fn=print
) -> pd.DataFrame:
    """Prepare matched SSE positions across timepoints."""
    log_fn("Preparing matched SSE positions...")
    
    num_sequences = true_sse.shape[0]
    num_positions = true_sse.shape[1]
    conds = meta['conditions']
    matched_positions = {}
    
    for seq_idx in range(num_sequences):
        if seq_idx % 100 == 0:
            log_fn(f"  Processing sequence {seq_idx + 1}/{num_sequences}", end='\r')
        
        # Check if this sequence has any valid conditions
        valid_conditions = condition_mask[seq_idx, :]
        if not valid_conditions.any():
            continue
        
        # For positions where the sequence has valid conditions, check for splice sites
        for pos in range(num_positions):
            # Check if this position is a splice site
            if true_labels[seq_idx, pos] == 0:
                continue
            
            # Get true and pred values for valid conditions only
            true_vals = true_sse[seq_idx, pos, valid_conditions]
            pred_vals = pred_sse[seq_idx, pos, valid_conditions]
            
            # Skip if no valid SSE values (all NaN)
            if np.all(np.isnan(true_vals)):
                continue
            
            # Store species information for each matched position
            species_idx = meta_df.iloc[seq_idx]['species_id']
            species_name = species_id_to_name[species_idx]
            
            # Add to dict
            matched_positions[(seq_idx, pos)] = {
                "species": species_name,
                "true": true_vals,
                "pred": pred_vals
            }
    
    log_fn(f"\n\n  Found {len(matched_positions)} matched positions")
    
    # Convert to dataframe
    all_data = []
    for (seq_idx, pos), vals in matched_positions.items():
        # Get species information for this sequence
        species_idx = meta_df.iloc[seq_idx]['species_id']
        species_name = species_id_to_name[species_idx]
        
        # Get valid conditions for this sequence
        valid_mask = condition_mask[seq_idx, :]
        valid_indices = np.where(valid_mask)[0]
        
        # Iterate through valid conditions
        for idx, cond_idx in enumerate(valid_indices):
            # Get true and pred values (indexed by position in filtered array)
            true_val = vals['true'][idx]
            pred_val = vals['pred'][idx]
            
            # Skip if true value is NaN
            if np.isnan(true_val):
                continue
            
            # Get condition name and split into tissue and timepoint
            condition = conds[cond_idx]
            parts = condition.rsplit('_', 1)  # Split from right to handle multi-word tissues
            tissue = parts[0]
            timepoint = int(parts[1]) if len(parts) > 1 else 0
            
            # Add to data list
            all_data.append({
                'species': species_name,
                'sequence': seq_idx,
                'position': pos,
                'condition': condition,
                'tissue': tissue,
                'timepoint': timepoint,
                'true_SSE': true_val,
                'pred_SSE': pred_val
            })
    
    all_data_df = pd.DataFrame(all_data)
    log_fn(f"  Created DataFrame with {len(all_data_df)} rows")
    log_fn(f"  Species: {all_data_df['species'].unique()}")
    log_fn(f"  Tissues: {sorted(all_data_df['tissue'].unique())}")
    
    return all_data_df


def calculate_tissue_correlations(all_data_df: pd.DataFrame, log_fn=print) -> pd.DataFrame:
    """Calculate correlation between predicted and true SSE values."""
    log_fn("Calculating correlations...")
    
    correlation_results = []
    for (species, seq_idx, pos, tissue), group in all_data_df.groupby(['species', 'sequence', 'position', 'tissue']):
        true_vals = group['true_SSE'].values
        pred_vals = group['pred_SSE'].values
        # Replace true nan with 0
        true_vals = np.nan_to_num(true_vals, nan=0)
        # Skip if less than 2 values
        n_timepoints = len(true_vals)
        if n_timepoints < 2:
            continue
        correlation = np.corrcoef(true_vals, pred_vals)[0, 1]
        correlation_results.append({
            'species': species,
            'sequence': seq_idx,
            'position': pos,
            'tissue': tissue,
            'correlation': correlation,
            'n_timepoints': n_timepoints,
            'avg_true_SSE': np.mean(true_vals),
            'avg_pred_SSE': np.mean(pred_vals)
        })
    
    correlation_df = pd.DataFrame(correlation_results)
    
    # Summary statistics
    nas = correlation_df['correlation'].isna()
    log_fn(f"  {nas.sum()} ({nas.mean() * 100:.2f}%) correlations are NaN")
    log_fn(f"  Mean correlation: {correlation_df['correlation'].mean():.4f}")
    log_fn(f"  Median correlation: {correlation_df['correlation'].median():.4f}")
    log_fn(f"  Std correlation: {correlation_df['correlation'].std():.4f}")
    
    return correlation_df


def plot_sse_density(
    all_data_df: pd.DataFrame,
    group: str,
    output_dir: Path,
    log_fn=print,
    max_points_per_group: int = 10000,
    thresh: float = 0.05
):
    """Plot density of predicted vs true SSE values.
    
    Args:
        all_data_df: DataFrame with SSE data
        group: Column name to group by ('species' or 'tissue')
        output_dir: Directory to save output
        log_fn: Logging function
        max_points_per_group: Maximum points to use for KDE (for speed)
        thresh: Threshold for KDE plot to skip low-density regions
    """
    log_fn(f"Plotting SSE density by {group}...")
    
    samples = all_data_df[group].unique()
    num_samples = len(samples)
    
    num_cols = 1
    num_rows = (num_samples + num_cols - 1) // num_cols
    
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(num_cols * 4, num_rows * 4), squeeze=False)
    
    for i, sample in enumerate(samples):
        log_fn(f"  Processing {group} {i+1}/{num_samples}: {sample}")
        sample_data = all_data_df[all_data_df[group] == sample]
        
        if len(sample_data) > max_points_per_group:
            log_fn(f"    Subsampling to {max_points_per_group} points for faster KDE...")
            sample_data_plot = sample_data.sample(n=max_points_per_group, random_state=42)
        else:
            sample_data_plot = sample_data
        
        row = i // num_cols
        col = i % num_cols
        ax = axs[row, col]
        
        r_value = np.corrcoef(sample_data['true_SSE'], sample_data['pred_SSE'])[0, 1]
        
        # 2D density plot
        sns.kdeplot(
            x=sample_data_plot['true_SSE'],
            y=sample_data_plot['pred_SSE'],
            levels=5,
            fill=True,
            cmap="rocket_r",
            thresh=thresh,
            bw_adjust=0.8,
            ax=ax
        )
        
        # Top histogram (True SSE) - use full data
        ax_histx = ax.inset_axes([0, 1.05, 1, 0.2], sharex=ax)
        ax_histx.hist(sample_data['true_SSE'], bins=30, color='gray', alpha=0.7)
        ax_histx.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
        
        # Right histogram (Pred SSE) - use full data
        ax_histy = ax.inset_axes([1.05, 0, 0.2, 1], sharey=ax)
        ax_histy.hist(sample_data['pred_SSE'], bins=30, orientation='horizontal', color='gray', alpha=0.7)
        ax_histy.tick_params(axis="y", which="both", left=False, right=False, labelleft=False)
        
        mean_true = sample_data['true_SSE'].mean()
        mean_pred = sample_data['pred_SSE'].mean()
        ax.axvline(mean_true, color='red', linestyle='--', linewidth=1, label='Mean True')
        ax.axhline(mean_pred, color='red', linestyle='--', linewidth=1, label='Mean Pred')
        
        ax.text(0.05, 0.95, f'r = {r_value:.3f}',
                transform=ax.transAxes, fontsize=12, fontweight='bold',
                verticalalignment='top')
        
        ax.set_title(sample, fontsize=12, fontweight='bold')
        ax.set_xlabel('True SSE', fontsize=10)
        ax.set_ylabel('Predicted SSE', fontsize=10)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid()
    
    plt.tight_layout()
    density_plot = output_dir / f"sse_density_{group}.png"
    plt.savefig(density_plot, dpi=150, bbox_inches='tight')
    plt.close()
    log_fn(f"  Saved density plot to {density_plot}")


def plot_correlation_distribution(correlation_df: pd.DataFrame, output_dir: Path, log_fn=print):
    """Plot distribution of correlations."""
    log_fn("Plotting correlation distribution...")
    
    sorted_corr = np.sort(correlation_df['correlation'].dropna())
    cdf = np.arange(1, len(sorted_corr) + 1) / len(sorted_corr)
    
    fig, ax1 = plt.subplots(figsize=(5, 3.5))
    ax1.hist(correlation_df['correlation'].dropna(), bins=50, color='#66c2a5', edgecolor='black')
    ax1.set_xlabel('Correlation Coefficient', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12, color='#4daf4a')
    ax1.tick_params(axis='y', labelcolor='#4daf4a')
    
    ax2 = ax1.twinx()
    ax2.plot(sorted_corr, cdf, color='#fc8d62', linewidth=2, label='CDF')
    ax2.set_ylabel('CDF', fontsize=12, color='#ff7f00')
    ax2.tick_params(axis='y', labelcolor='#ff7f00')
    
    plt.title('Distribution of Correlation\nTrue vs Predicted SSE', fontsize=14)
    ax1.grid(alpha=0.3)
    fig.tight_layout()
    
    corr_plot = output_dir / "correlation_distribution.png"
    plt.savefig(corr_plot, dpi=150, bbox_inches='tight')
    plt.close()
    log_fn(f"Saved correlation distribution plot to {corr_plot}")


def plot_correlation_by_n_timepoints(correlation_df: pd.DataFrame, output_dir: Path, log_fn=print):
    """Plot correlation distribution by number of timepoints available."""
    log_fn("Plotting correlations by number of timepoints...")
    
    n_timepoint_bins = [(2, 4), (5, 7), (8, 10), (11, 14)]
    n_timepoint_labels = ['2-4 timepoints', '5-7 timepoints', '8-10 timepoints', '11-14 timepoints']
    
    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    axes = axes.flatten()
    
    for idx, ((min_n, max_n), label) in enumerate(zip(n_timepoint_bins, n_timepoint_labels)):
        ax = axes[idx]
        
        # Filter by number of timepoints
        filtered_corr = correlation_df[
            (correlation_df['n_timepoints'] >= min_n) &
            (correlation_df['n_timepoints'] <= max_n)
        ]['correlation'].dropna()
        
        # Cumulative distribution
        sorted_corr_filtered = np.sort(filtered_corr)
        cdf_filtered = np.arange(1, len(sorted_corr_filtered) + 1) / len(sorted_corr_filtered)
        
        # Plot histogram
        ax.hist(filtered_corr, bins=50, color='#66c2a5', edgecolor='black', alpha=0.7)
        ax.set_xlabel('Correlation Coefficient', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11, color='#4daf4a')
        ax.tick_params(axis='y', labelcolor='#4daf4a')
        ax.set_xlim(-1, 1)
        
        # Plot CDF on twin axis
        ax2 = ax.twinx()
        ax2.plot(sorted_corr_filtered, cdf_filtered, color='#fc8d62', linewidth=2)
        ax2.set_ylabel('CDF', fontsize=11, color='#ff7f00')
        ax2.tick_params(axis='y', labelcolor='#ff7f00')
        ax2.set_ylim(0, 1)
        
        # Title with count
        ax.set_title(f'{label}\\n(n={len(filtered_corr):,})', fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    timepoint_plot = output_dir / "correlation_by_n_timepoints.png"
    plt.savefig(timepoint_plot, dpi=150, bbox_inches='tight')
    plt.close()
    log_fn(f"Saved timepoint correlation plot to {timepoint_plot}")


def plot_sse_violin(all_data_df: pd.DataFrame, output_dir: Path, log_fn=print):
    """Plot violin plot of predicted SSE values per true SSE bin."""
    log_fn("Plotting SSE violin plot...")
    
    # Bin true SSE values
    all_data_df_copy = all_data_df.copy()
    all_data_df_copy['true_SSE_bin'] = pd.cut(
        all_data_df_copy['true_SSE'],
        bins=[0.0, 0.1, 0.5, 0.9, 1.0],
        right=False
    )
    
    plt.figure(figsize=(5, 4))
    sns.violinplot(x='true_SSE_bin', y='pred_SSE', data=all_data_df_copy)
    plt.xlabel('True SSE', fontsize=12)
    plt.ylabel('Predicted SSE', fontsize=12)
    plt.ylim(0, 1)
    plt.tight_layout()
    
    violin_plot = output_dir / "sse_violin_plot.png"
    plt.savefig(violin_plot, dpi=150, bbox_inches='tight')
    plt.close()
    log_fn(f"Saved violin plot to {violin_plot}")


def calculate_overall_sse_correlation(all_data_df: pd.DataFrame, log_fn=print):
    """Calculate overall correlation between true and predicted SSE."""
    log_fn("Calculating overall SSE correlation...")
    
    pearson_corr, _ = pearsonr(all_data_df['true_SSE'], all_data_df['pred_SSE'])
    spearman_corr, _ = spearmanr(all_data_df['true_SSE'], all_data_df['pred_SSE'])
    
    log_fn(f"  Overall Pearson correlation: {pearson_corr:.3f}")
    log_fn(f"  Overall Spearman correlation: {spearman_corr:.3f}")
    
    return {'pearson': pearson_corr, 'spearman': spearman_corr}


def plot_correlation_by_tissue(
    correlation_df: pd.DataFrame,
    output_dir: Path,
    log_fn=print
):
    """Plot correlation distribution by tissue."""
    log_fn("Plotting correlations by tissue...")
    
    tissue_colors = {
        'Brain': '#3399cc',
        'Cerebellum': '#34ccff',
        'Heart': '#cc0100',
        'Kidney': '#cc9900',
        'Liver': '#339900',
        'Ovary': '#cc329a',
        'Testis': '#ff6600'
    }
    
    unique_tissues = correlation_df['tissue'].unique()
    num_tissues = len(unique_tissues)
    num_cols = 3
    num_rows = (num_tissues + num_cols - 1) // num_cols
    
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(num_cols * 3, num_rows * 2), squeeze=False)
    
    for i, tissue in enumerate(unique_tissues):
        row = i // num_cols
        col = i % num_cols
        ax = axs[row, col]
        
        tissue_corrs = correlation_df[correlation_df['tissue'] == tissue]['correlation'].dropna()
        tissue_color = tissue_colors.get(tissue, '#000000')
        
        ax.hist(tissue_corrs, bins=30, color=tissue_color, edgecolor='black')
        ax.set_title(f'Tissue: {tissue}', fontsize=12)
        ax.set_xlabel("Pearson's Correlation", fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.set_xlim(-1, 1)
        ax.grid(alpha=0.3)
    
    # Remove extra subplots
    for j in range(i + 1, num_rows * num_cols):
        row = j // num_cols
        col = j % num_cols
        fig.delaxes(axs[row][col])
    
    plt.tight_layout()
    tissue_plot = output_dir / "correlation_by_tissue.png"
    plt.savefig(tissue_plot, dpi=150)
    plt.close()
    log_fn(f"Saved tissue correlation plot to {tissue_plot}")


def plot_top_correlation_examples(
    all_data_df: pd.DataFrame,
    correlation_df: pd.DataFrame,
    output_dir: Path,
    n_examples: int = 10,
    log_fn=print
):
    """Plot top correlation examples showing SSE trends across timepoints."""
    log_fn(f"Plotting top {n_examples} correlation examples...")
    
    # Define tissue colors
    tissue_colors = {
        'Brain': '#3399cc',
        'Cerebellum': '#34ccff',
        'Heart': '#cc0100',
        'Kidney': '#cc9900',
        'Liver': '#339900',
        'Ovary': '#cc329a',
        'Testis': '#ff6600'
    }
    
    # Select top positions across all tissues
    correlation_df['correlation_across_tissues'] = correlation_df.groupby(['species', 'sequence', 'position'])['correlation'].transform('mean')
    correlation_df['avg_sse_across_tissues'] = correlation_df.groupby(['species', 'sequence', 'position'])['avg_true_SSE'].transform('mean')
    correlation_df['n_tissues'] = correlation_df.groupby(['species', 'sequence', 'position'])['correlation'].transform('count')
    correlation_df['n_timepoints_across_tissues'] = correlation_df.groupby(['species', 'sequence', 'position'])['n_timepoints'].transform('min')
    
    top_examples = correlation_df.sort_values(by='correlation_across_tissues', ascending=False)
    top_examples = top_examples[(top_examples['n_timepoints_across_tissues'] >= 5) & (top_examples['n_tissues'] >= 3)]
    top_examples = top_examples[top_examples['avg_sse_across_tissues'] >= 0.2]
    top_examples = top_examples[['sequence', 'position']].drop_duplicates().head(n_examples)
    
    if len(top_examples) == 0:
        log_fn("  No examples to plot")
        return
    
    # Filter data for top examples
    plot_df = all_data_df[
        (all_data_df['sequence'].isin(top_examples['sequence'])) &
        (all_data_df['position'].isin(top_examples['position']))
    ].copy()
    
    plot_df['site'] = plot_df['sequence'].astype(str) + '_' + plot_df['position'].astype(str)
    
    unique_sites = plot_df['site'].unique()
    timepoint_order = sorted(plot_df['timepoint'].unique())
    plot_df['timepoint'] = pd.Categorical(plot_df['timepoint'], categories=timepoint_order, ordered=True)
    
    n_sites = len(unique_sites)
    unique_tissues_plot = plot_df['tissue'].unique()
    n_tissues = len(unique_tissues_plot)
    
    fig, axes = plt.subplots(
        n_tissues, n_sites,
        figsize=(2.5 * n_sites, 2.5 * n_tissues),
        sharex='col',
        squeeze=False
    )
    
    legend_handles, legend_labels = [], []
    
    for col_idx, site in enumerate(unique_sites):
        site_data = plot_df[plot_df['site'] == site]
        for tissue in site_data['tissue'].unique():
            row_idx = list(unique_tissues_plot).index(tissue)
            tissue_data = site_data[site_data['tissue'] == tissue].sort_values('timepoint')
            color = tissue_colors.get(tissue, '#000000')
            
            x_values = tissue_data['timepoint'].cat.codes
            y_sse_true = tissue_data['true_SSE'].values
            y_sse_pred = tissue_data['pred_SSE'].values
            
            # Calculate correlation only if we have at least 2 data points
            if len(y_sse_true) >= 2:
                correlation = np.corrcoef(y_sse_true, y_sse_pred)[0, 1]
            else:
                correlation = np.nan
            
            # True SSE: solid line
            line_true, = axes[row_idx, col_idx].plot(
                x_values, y_sse_true,
                label=f'{tissue} True',
                color=color,
                linewidth=2,
                linestyle='-'
            )
            axes[row_idx, col_idx].scatter(x_values, y_sse_true, color=color, marker='o', s=36)
            
            # Predicted SSE: dashed line
            line_pred, = axes[row_idx, col_idx].plot(
                x_values, y_sse_pred,
                label=f'{tissue} Pred',
                color=color,
                linewidth=2,
                linestyle='--'
            )
            axes[row_idx, col_idx].scatter(x_values, y_sse_pred, color=color, marker='x', s=36)
            
            if col_idx == 0:
                legend_handles.extend([line_true, line_pred])
                legend_labels.extend([f'{tissue} True', f'{tissue} Pred'])
            
            axes[row_idx, col_idx].set_ylim(0, 1)
            
            # Highlight high correlation with bold title and colored background
            if correlation > 0.9:
                axes[row_idx, col_idx].set_title(
                    f'{site}\\n{correlation:.3f}',
                    fontsize=14,
                    fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='gold', alpha=0.3)
                )
                axes[row_idx, col_idx].patch.set_edgecolor('gold')
                axes[row_idx, col_idx].patch.set_linewidth(3)
            else:
                axes[row_idx, col_idx].set_title(f'{site}\\n{correlation:.3f}', fontsize=14)
            
            axes[row_idx, col_idx].set_xticks(range(len(timepoint_order)))
            axes[row_idx, col_idx].set_xticklabels(timepoint_order, rotation=90)
            axes[row_idx, col_idx].grid(True, alpha=0.3)
    
    # Set y labels for first column
    for row_idx in range(n_tissues):
        axes[row_idx, 0].set_ylabel('SSE', fontsize=12, fontweight='bold')
    
    # Set x labels for last row
    for col_idx in range(n_sites):
        axes[-1, col_idx].set_xlabel('Timepoint', fontsize=12, fontweight='bold')
        axes[-1, col_idx].set_xticklabels(timepoint_order, rotation=90)
    
    fig.legend(
        legend_handles, legend_labels,
        loc='lower center',
        bbox_to_anchor=(1.05, 0.5),
        title='Tissue',
        title_fontsize=14,
        fontsize=12,
        ncol=1
    )
    
    plt.tight_layout(rect=[0, 0, 0.95, 1])
    
    top_examples_plot = output_dir / "top_correlation_examples.png"
    plt.savefig(top_examples_plot, dpi=150, bbox_inches='tight')
    plt.close()
    log_fn(f"Saved top correlation examples plot to {top_examples_plot}")


def save_results_summary(
    eval_results: Dict,
    output_dir: Path,
    log_fn=print
):
    """Save evaluation results summary to JSON."""
    log_fn("Saving results summary...")
    
    # Convert numpy types to Python native types for JSON serialization
    summary = {
        'pr_auc': {
            str(k): float(v) for k, v in eval_results.get('pr_auc', {}).items()
        },
        'top_k_acc': {
            str(k): float(v) for k, v in eval_results.get('top_k_acc', {}).items()
        },
        'overall_sse_correlation': {
            'pearson': float(eval_results.get('overall_sse_correlation', {}).get('pearson', 0)),
            'spearman': float(eval_results.get('overall_sse_correlation', {}).get('spearman', 0))
        },
        'timestamp': datetime.now().isoformat()
    }
    
    results_file = output_dir / "evaluation_results.json"
    with open(results_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    log_fn(f"Saved results summary to {results_file}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate splice site and usage predictions")
    parser.add_argument(
        "--test-data",
        required=True,
        help="Path to test data directory with sequences and labels"
    )
    parser.add_argument(
        "--predictions",
        required=True,
        help="Path to predictions directory"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for evaluation results"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress output messages"
    )
    
    args = parser.parse_args()
    
    # Setup paths
    test_data_dir = Path(args.test_data)
    pred_dir = Path(args.predictions)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    log_file = output_dir / "evaluation.log"
    log_fn = setup_logging(str(log_file), quiet=args.quiet)
    
    log_fn("="*60)
    log_fn("SPLICEVO MODEL EVALUATION")
    log_fn("="*60)
    log_fn(f"Test data directory: {test_data_dir}")
    log_fn(f"Predictions directory: {pred_dir}")
    log_fn(f"Output directory: {output_dir}")
    log_fn(f"Start time: {datetime.now().isoformat()}")
    
    try:
        # Load test data
        log_fn("Loading test data...")
        test_seq, test_labels, test_alpha, test_beta, test_sse, test_species, test_condition_mask = load_processed_data(str(test_data_dir))
        log_fn(f"  Sequences shape: {test_seq.shape}")
        log_fn(f"  Labels shape: {test_labels.shape}")
        log_fn(f"  SSE shape: {test_sse.shape}")
        if test_condition_mask is not None:
            log_fn(f"  Condition mask shape: {test_condition_mask.shape}")
        
        # Load metadata
        meta_fn = test_data_dir / "metadata.json"
        with open(meta_fn, "r") as f:
            test_meta = json.load(f)
        
        # Load test windows metadata
        meta_csv_fn = test_data_dir / "metadata.csv"
        meta_df = pd.read_csv(meta_csv_fn)
        log_fn(f"  Loaded metadata for {len(meta_df)} sequences")
        
        # Create species mapping
        species_name_to_id = test_meta['species_mapping']
        species_id_to_name = {v: k for k, v in species_name_to_id.items()}
        
        # Load predictions
        log_fn("Loading predictions...")
        pred_labels, pred_probs, pred_sse, meta, true_labels, true_sse, condition_mask = load_predictions(str(pred_dir))
        log_fn(f"  Predicted labels shape: {pred_labels.shape}")
        log_fn(f"  Predicted probs shape: {pred_probs.shape}")
        log_fn(f"  Predicted SSE shape: {pred_sse.shape}")
        
        # Use condition_mask from test data, not predictions
        if test_condition_mask is not None:
            condition_mask = test_condition_mask
            log_fn(f"  Using condition mask from test data")
        
        # Evaluate splice site classification
        eval_results = evaluate_splice_site_classification(
            true_labels, pred_labels, pred_probs, output_dir, log_fn
        )
        
        # Evaluate splice usage regression
        species_ids = meta_df['species_id'].values
        eval_results.update(
            evaluate_splice_usage_regression(
                true_sse, pred_sse, species_ids, species_id_to_name, meta, output_dir, log_fn
            )
        )
        
        # Prepare matched positions
        all_data_df = prepare_matched_positions(
            true_labels, true_sse, pred_sse, condition_mask,
            meta_df, species_id_to_name, meta, log_fn
        )
        
        # Save matched positions
        matched_file = output_dir / "matched_sse_positions.csv"
        all_data_df.to_csv(matched_file, index=False)
        log_fn(f"Saved matched positions to {matched_file}")
        
        # Calculate overall SSE correlations
        overall_corr = calculate_overall_sse_correlation(all_data_df, log_fn)
        eval_results['overall_sse_correlation'] = overall_corr
        
        # Calculate correlations per position
        correlation_df = calculate_tissue_correlations(all_data_df, log_fn)
        
        # Save correlation table
        correlation_file = output_dir / "sse_correlation_per_position.csv"
        correlation_df.to_csv(correlation_file, index=False)
        log_fn(f"Saved correlation table to {correlation_file}")
        
        # Generate plots
        log_fn("="*60)
        log_fn("GENERATING PLOTS")
        log_fn("="*60)
        
        # SSE density plots by species and tissue
        plot_sse_density(all_data_df, 'species', output_dir, log_fn)
        plot_sse_density(all_data_df, 'tissue', output_dir, log_fn)
        
        # Correlation plots
        plot_correlation_distribution(correlation_df, output_dir, log_fn)
        plot_correlation_by_n_timepoints(correlation_df, output_dir, log_fn)
        plot_correlation_by_tissue(correlation_df, output_dir, log_fn)
        
        # SSE violin plot
        plot_sse_violin(all_data_df, output_dir, log_fn)
        
        # Top correlation examples
        plot_top_correlation_examples(all_data_df, correlation_df, output_dir, log_fn=log_fn)
        
        # Save results summary
        save_results_summary(eval_results, output_dir, log_fn)
        
        log_fn("="*60)
        log_fn("EVALUATION COMPLETED SUCCESSFULLY")
        log_fn("="*60)
        log_fn(f"End time: {datetime.now().isoformat()}")
        
    except Exception as e:
        log_fn(f"ERROR: {str(e)}")
        import traceback
        log_fn(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
