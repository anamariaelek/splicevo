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
from sklearn.metrics import precision_recall_curve, auc, mean_squared_error

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
    log_fn("\n" + "="*60)
    log_fn("SPLICE SITE CLASSIFICATION EVALUATION")
    log_fn("="*60)
    
    results = {}
    
    # PR-AUC scores
    log_fn("\nCalculating Precision-Recall AUC scores...")
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
    
    # Top-k accuracy
    log_fn("\nCalculating top-k accuracy...")
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
    log_fn("\n" + "="*60)
    log_fn("SPLICE USAGE REGRESSION EVALUATION")
    log_fn("="*60)
    
    results = {}
    
    # Calculate MSE for each species and condition
    log_fn("\nCalculating MSE for each species and condition...")
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
    log_fn("\nPlotting RMSE by species...")
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
    true_sse: np.ndarray,
    pred_sse: np.ndarray,
    meta: Dict,
    log_fn=print
) -> pd.DataFrame:
    """Prepare matched SSE positions across timepoints."""
    log_fn("\nPreparing matched SSE positions...")
    
    num_sequences = true_sse.shape[0]
    conds = meta['conditions']
    matched_positions = {}
    
    for seq_idx in range(num_sequences):
        # Find positions where all tissues are not NaN
        valid_positions = np.where(np.all(~np.isnan(true_sse[seq_idx, :, :]), axis=1))[0]
        
        if len(valid_positions) == 0:
            continue
        
        for pos in valid_positions:
            true_vals = true_sse[seq_idx, pos, :]
            pred_vals = pred_sse[seq_idx, pos, :]
            true_vals = np.nan_to_num(true_vals)
            matched_positions[(seq_idx, pos)] = {"true": true_vals, "pred": pred_vals}
    
    log_fn(f"  Found {len(matched_positions)} matched positions")
    
    # Convert to dataframe
    all_data = []
    for (seq_idx, pos), vals in matched_positions.items():
        for tissue_idx in range(len(conds)):
            all_data.append({
                'sequence': seq_idx,
                'position': pos,
                'group': conds[tissue_idx],
                'true_SSE': vals['true'][tissue_idx],
                'pred_SSE': vals['pred'][tissue_idx]
            })
    
    all_data_df = pd.DataFrame(all_data)
    all_data_df['Tissue'] = all_data_df['group'].apply(lambda x: x.split('_')[0])
    all_data_df['Timepoint'] = all_data_df['group'].apply(lambda x: int(x.split('_')[1]))
    
    return all_data_df


def calculate_tissue_correlations(all_data_df: pd.DataFrame, log_fn=print) -> pd.DataFrame:
    """Calculate correlation between predicted and true SSE values."""
    log_fn("\nCalculating correlations...")
    
    correlation_results = []
    for (seq_idx, pos, tissue), group in all_data_df.groupby(['sequence', 'position', 'Tissue']):
        true_vals = group['true_SSE'].values
        pred_vals = group['pred_SSE'].values
        
        if len(true_vals) < 2:
            continue
        
        correlation = np.corrcoef(true_vals, pred_vals)[0, 1]
        correlation_results.append({
            'sequence': seq_idx,
            'position': pos,
            'tissue': tissue,
            'correlation': correlation
        })
    
    correlation_df = pd.DataFrame(correlation_results)
    
    # Summary statistics
    nas = correlation_df['correlation'].isna()
    log_fn(f"  {nas.sum()} ({nas.mean() * 100:.2f}%) correlations are NaN")
    log_fn(f"  Mean correlation: {correlation_df['correlation'].mean():.4f}")
    log_fn(f"  Median correlation: {correlation_df['correlation'].median():.4f}")
    log_fn(f"  Std correlation: {correlation_df['correlation'].std():.4f}")
    
    return correlation_df


def plot_sse_density(all_data_df: pd.DataFrame, output_dir: Path, log_fn=print):
    """Plot density of predicted vs true SSE values."""
    log_fn("\nPlotting SSE density...")
    
    samples = all_data_df['group'].unique()[0:1]  # Only first timepoint
    num_tissues = len(samples)
    num_cols = 1
    num_rows = (num_tissues + num_cols - 1) // num_cols
    
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(num_cols * 4, num_rows * 4), squeeze=False)
    
    for i, tissue in enumerate(samples):
        tissue_data = all_data_df[all_data_df['group'] == tissue]
        
        row = i // num_cols
        col = i % num_cols
        ax = axs[row, col]
        
        # 2D density plot
        sns.kdeplot(
            x=tissue_data['true_SSE'],
            y=tissue_data['pred_SSE'],
            levels=10,
            fill=True,
            cmap="rocket_r",
            ax=ax
        )
        
        # Top histogram (True SSE)
        ax_histx = ax.inset_axes([0, 1.05, 1, 0.2], sharex=ax)
        ax_histx.hist(tissue_data['true_SSE'], bins=30, color='gray', alpha=0.7)
        ax_histx.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
        
        # Right histogram (Pred SSE)
        ax_histy = ax.inset_axes([1.05, 0, 0.2, 1], sharey=ax)
        ax_histy.hist(tissue_data['pred_SSE'], bins=30, orientation='horizontal', color='gray', alpha=0.7)
        ax_histy.tick_params(axis="y", which="both", left=False, right=False, labelleft=False)
        
        # Add mean lines
        mean_true = tissue_data['true_SSE'].mean()
        mean_pred = tissue_data['pred_SSE'].mean()
        ax.axvline(mean_true, color='red', linestyle='--', linewidth=1, label='Mean True')
        ax.axhline(mean_pred, color='red', linestyle='--', linewidth=1, label='Mean Pred')
        
        ax.set_xlabel('True SSE', fontsize=10)
        ax.set_ylabel('Predicted SSE', fontsize=10)
        ax.set_xlim(0, 1)
        ax.grid()
    
    plt.tight_layout()
    density_plot = output_dir / "sse_density.png"
    plt.savefig(density_plot, dpi=150)
    plt.close()
    log_fn(f"Saved density plot to {density_plot}")


def plot_correlation_distribution(correlation_df: pd.DataFrame, output_dir: Path, log_fn=print):
    """Plot distribution of correlations."""
    log_fn("\nPlotting correlation distribution...")
    
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
    plt.savefig(corr_plot, dpi=150)
    plt.close()
    log_fn(f"Saved correlation distribution plot to {corr_plot}")


def plot_correlation_by_tissue(
    correlation_df: pd.DataFrame,
    output_dir: Path,
    log_fn=print
):
    """Plot correlation distribution by tissue."""
    log_fn("\nPlotting correlations by tissue...")
    
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


def save_results_summary(
    eval_results: Dict,
    output_dir: Path,
    log_fn=print
):
    """Save evaluation results summary to JSON."""
    log_fn("\nSaving results summary...")
    
    # Convert numpy types to Python native types for JSON serialization
    summary = {
        'pr_auc': {
            str(k): float(v) for k, v in eval_results.get('pr_auc', {}).items()
        },
        'top_k_acc': {
            str(k): float(v) for k, v in eval_results.get('top_k_acc', {}).items()
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
        log_fn("\nLoading test data...")
        test_seq, test_labels, test_alpha, test_beta, test_sse, test_species = load_processed_data(str(test_data_dir))
        log_fn(f"  Sequences shape: {test_seq.shape}")
        log_fn(f"  Labels shape: {test_labels.shape}")
        log_fn(f"  SSE shape: {test_sse.shape}")
        
        # Load metadata
        meta_fn = test_data_dir / "metadata.json"
        with open(meta_fn, "r") as f:
            test_meta = json.load(f)
        
        # Load predictions
        log_fn("\nLoading predictions...")
        pred_labels, pred_probs, pred_sse, meta, true_labels, true_sse = load_predictions(str(pred_dir))
        log_fn(f"  Predicted labels shape: {pred_labels.shape}")
        log_fn(f"  Predicted probs shape: {pred_probs.shape}")
        log_fn(f"  Predicted SSE shape: {pred_sse.shape}")
        
        # Create species mapping
        species_ids = np.array([test_meta['species_mapping'][sp] for sp in test_meta['species_mapping'].keys()])
        species_id_to_name = {v: k for k, v in test_meta['species_mapping'].items()}
        
        # Evaluate splice site classification
        eval_results = evaluate_splice_site_classification(
            true_labels, pred_labels, pred_probs, output_dir, log_fn
        )
        
        # Evaluate splice usage regression
        eval_results.update(
            evaluate_splice_usage_regression(
                true_sse, pred_sse, species_ids, species_id_to_name, meta, output_dir, log_fn
            )
        )
        
        # Prepare matched positions
        all_data_df = prepare_matched_positions(true_sse, pred_sse, meta, log_fn)
        
        # Save matched positions
        matched_file = output_dir / "matched_sse_positions.csv"
        all_data_df.to_csv(matched_file, index=False)
        log_fn(f"Saved matched positions to {matched_file}")
        
        # Calculate correlations
        correlation_df = calculate_tissue_correlations(all_data_df, log_fn)
        
        # Generate plots
        plot_sse_density(all_data_df, output_dir, log_fn)
        plot_correlation_distribution(correlation_df, output_dir, log_fn)
        plot_correlation_by_tissue(correlation_df, output_dir, log_fn)
        
        # Save results summary
        save_results_summary(eval_results, output_dir, log_fn)
        
        log_fn("\n" + "="*60)
        log_fn("EVALUATION COMPLETED SUCCESSFULLY")
        log_fn("="*60)
        log_fn(f"End time: {datetime.now().isoformat()}")
        
    except Exception as e:
        log_fn(f"\nERROR: {str(e)}")
        import traceback
        log_fn(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
