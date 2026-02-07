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

from splicevo.data.data_loader import load_memmap_data


def load_sparse_predictions(pred_dir: Path, test_data_dir: Path, log_fn=print) -> Tuple:
    """
    Load predictions and ground truth from sparse parquet format.
    
    Args:
        pred_dir: Directory containing prediction parquet files
        test_data_dir: Directory containing test data
        log_fn: Logging function
    
    Returns:
        Tuple of (pred_labels, pred_probs, pred_usage, metadata, true_labels, true_usage)
        All as dense numpy arrays for compatibility with evaluation functions
    """
    log_fn("Loading sparse predictions...")
    
    # Load metadata
    meta_file = pred_dir / 'predictions_metadata.json'
    with open(meta_file, 'r') as f:
        pred_meta = json.load(f)
    
    # Load test data metadata to get dimensions
    test_meta_file = test_data_dir / 'metadata.json'
    with open(test_meta_file, 'r') as f:
        test_meta = json.load(f)
    
    # Get dimensions from test data
    sequences, labels_true_sparse, usage_true_sparse, metadata = load_memmap_data(
        test_data_dir,
        load_labels=True,
        load_usage=True
    )
    
    n_samples = sequences.shape[0]
    seq_len = sequences.shape[1]
    context_len = test_meta.get('context_len', 450)
    central_len = seq_len - 2 * context_len
    
    # Determine number of conditions from usage data
    if usage_true_sparse is not None and len(usage_true_sparse) > 0:
        if 'condition_idx' in usage_true_sparse.columns:
            n_conditions = usage_true_sparse['condition_idx'].max() + 1
        else:
            # Try to infer from test metadata
            n_conditions = len(test_meta.get('condition_names', []))
            if n_conditions == 0:
                # Default fallback
                n_conditions = 8
                log_fn(f"  Warning: Could not determine n_conditions from data, using default: {n_conditions}")
    else:
        # Try to infer from test metadata
        n_conditions = len(test_meta.get('condition_names', []))
        if n_conditions == 0:
            n_conditions = 0
            log_fn(f"  Warning: No usage data found, n_conditions set to 0")
    
    log_fn(f"  Dimensions: {n_samples} samples, central length: {central_len}, conditions: {n_conditions}")
    
    # Initialize dense arrays
    pred_labels = np.zeros((n_samples, central_len), dtype=np.int8)
    pred_probs = np.zeros((n_samples, central_len, 3), dtype=np.float32)  # 3 classes
    
    if n_conditions > 0:
        pred_usage = np.zeros((n_samples, central_len, n_conditions), dtype=np.float32)
        true_usage = np.full((n_samples, central_len, n_conditions), np.nan, dtype=np.float32)
    else:
        pred_usage = np.zeros((n_samples, central_len, 0), dtype=np.float32)
        true_usage = np.full((n_samples, central_len, 0), np.nan, dtype=np.float32)
    
    # Load predicted labels
    labels_pred_file = pred_dir / 'labels_pred.parquet'
    if labels_pred_file.exists():
        labels_pred_df = pd.read_parquet(labels_pred_file)
        pred_labels = labels_pred_df
    
    # Load predicted probabilities
    probs_pred_file = pred_dir / 'probs_pred.parquet'
    if probs_pred_file.exists():
        probs_pred_df = pd.read_parquet(probs_pred_file)
        log_fn(f"  Loaded {len(probs_pred_df)} predicted probability entries")
        pred_probs = probs_pred_df
    
    # Load predicted usage
    usage_pred_file = pred_dir / 'usage_pred.parquet'
    if usage_pred_file.exists() and n_conditions > 0:
        usage_pred_df = pd.read_parquet(usage_pred_file)
        log_fn(f"  Loaded {len(usage_pred_df)} predicted usage entries")
        pred_usage = usage_pred_df
    
    return pred_labels, pred_probs, pred_usage, test_meta, metadata


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


def evaluate_splice_site_classification(true_labels, pred_probs, output_dir: Path, log_fn=print) -> Dict:
    """
    Evaluate splice site classification performance.
    
    Args:
        true_labels: DataFrame with columns (sample_idx, position, label) or numpy array
        pred_probs: DataFrame with columns (sample_idx, position, class_id, probability) or numpy array
        output_dir: Output directory for plots
        log_fn: Logging function
    """
    
    results = {}
    
    # Handle DataFrame inputs efficiently
    if isinstance(pred_probs, pd.DataFrame):
        log_fn("Processing sparse probability predictions...")
        
        # Pivot pred_probs to have class_id as columns
        pred_wide = pred_probs.pivot_table(
            index=['sample_idx', 'position'],
            columns='class_id',
            values='probability',
            fill_value=0.0
        )
        pred_wide.columns = [f'prob_class_{int(c)}' for c in pred_wide.columns]
        pred_wide = pred_wide.reset_index()
        
        if isinstance(true_labels, pd.DataFrame):
            
            # Merge true labels with predictions
            merged = pred_wide.merge(
                true_labels[['sample_idx', 'position', 'label']],
                on=['sample_idx', 'position'],
                how='left'
            )
            merged['label'] = merged['label'].fillna(0).astype(int)
        else:
            # true_labels is numpy array - need to convert positions to labels
            log_fn("Converting numpy true labels to match predictions...")
            merged = pred_wide.copy()
            merged['label'] = 0  # default background
            # This would need custom logic to map from array indices
            
        log_fn(f"Evaluating {len(merged)} positions...")
        
        # Build evaluation arrays for each class
        y_true_by_class = {}
        y_scores_by_class = {}
        
        for class_idx in range(3):
            prob_col = f'prob_class_{class_idx}'
            
            if class_idx == 0:
                # Background: positive if NOT this class
                y_true_by_class[class_idx] = (merged['label'] != class_idx).astype(int).values
                y_scores_by_class[class_idx] = (1 - merged[prob_col]).values
            else:
                # Splice sites: positive if IS this class
                y_true_by_class[class_idx] = (merged['label'] == class_idx).astype(int).values
                y_scores_by_class[class_idx] = merged[prob_col].values
    
    else:
        # Handle numpy array inputs (backwards compatibility)
        log_fn("Processing dense array predictions...")
        
        if isinstance(true_labels, np.ndarray):
            y_true_by_class = {}
            y_scores_by_class = {}
            
            for class_idx in range(3):
                if class_idx == 0:
                    y_true_by_class[class_idx] = (true_labels != class_idx).astype(int).reshape(-1)
                    y_scores_by_class[class_idx] = (1 - pred_probs[:, :, class_idx]).reshape(-1)
                else:
                    y_true_by_class[class_idx] = (true_labels == class_idx).astype(int).reshape(-1)
                    y_scores_by_class[class_idx] = pred_probs[:, :, class_idx].reshape(-1)
        else:
            raise ValueError("Unsupported combination of input types")
    
    # Calculate PR-AUC scores
    log_fn("Calculating Precision-Recall AUC scores...")
    pr_auc_scores = {}
    class_labels = {0: 'no splice site', 1: 'donor', 2: 'acceptor'}
    class_colors = {0: 'tab:blue', 1: 'tab:orange', 2: 'tab:green'}
    
    plt.figure(figsize=(5, 4))
    for class_idx in range(3):
        color = class_colors[class_idx]
        label = class_labels[class_idx]
        
        y_true = y_true_by_class[class_idx]
        y_scores = y_scores_by_class[class_idx]
        
        if len(y_true) > 0 and np.sum(y_true) > 0:
            precision, recall, _ = precision_recall_curve(y_true, y_scores)
            pr_auc = auc(recall, precision)
            pr_auc_scores[class_idx] = pr_auc
            
            plt.plot(
                recall, precision,
                label=f"{label} (AUC = {pr_auc:.3f})",
                linewidth=2,
                color=color
            )
        else:
            pr_auc_scores[class_idx] = 0.0
            log_fn(f"  Warning: No positive samples for {label}")
    
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
    log_fn("Calculating top-k accuracy...")
    def top_k_accuracy(y_true, y_scores):
        k = int(np.sum(y_true))
        if k == 0:
            return 0.0
        threshold = np.sort(y_scores)[-k] if k <= len(y_scores) else y_scores.min()
        y_pred = (y_scores >= threshold).astype(int)
        accuracy = np.sum((y_pred == 1) & (y_true == 1)) / k
        return accuracy
    
    top_k_acc = {}
    for class_idx in range(3):
        y_true = y_true_by_class[class_idx]
        y_scores = y_scores_by_class[class_idx]
        
        if len(y_true) > 0:
            acc = top_k_accuracy(y_true, y_scores)
            top_k_acc[class_idx] = acc
            log_fn(f"  Top-k accuracy for {class_labels[class_idx]}: {acc:.4f}")
        else:
            top_k_acc[class_idx] = 0.0
    
    results['top_k_acc'] = top_k_acc
    
    return results


def calculate_condition_correlations(all_data_df: pd.DataFrame, group_by=['tissue'], log_fn=print) -> pd.DataFrame:
    """Calculate correlation between predicted and true SSE values."""
    log_fn("Calculating correlations...")
    
    # Get valid positions where we have both true and predicted SSE
    valid_data = all_data_df.dropna(subset=['true_sse', 'pred_sse'])
    log_fn(f"  Valid positions with both true and predicted SSE: {len(valid_data)} (%{len(valid_data) / len(all_data_df) * 100:.2f}%)")

    # Add condition annotation if available
    if 'condition_idx' not in valid_data.columns:
        log_fn("  Warning: 'condition_idx' column not found, adding default condition")
        valid_data['condition_idx'] = 0

    # Group by specified columns to get correlations
    correlation_results = []
    for group_var, group in valid_data.groupby(group_by):
        true_vals = group['true_sse'].values
        pred_vals = group['pred_sse'].values
        
        if len(true_vals) < 2:
            continue
        
        correlation = np.corrcoef(true_vals, pred_vals)[0, 1]
        
        # Build result dict with proper column names
        result = {}
        if len(group_by) == 1:
            result[group_by[0]] = group_var
        else:
            for i, col in enumerate(group_by):
                result[col] = group_var[i]
        result['correlation'] = correlation
        result['num_positions'] = len(group)
        
        correlation_results.append(result)
    
    # Create DataFrame with proper handling of empty results
    if len(correlation_results) == 0:
        # Return empty DataFrame with expected columns
        cols = group_by + ['correlation', 'num_positions']
        correlation_df = pd.DataFrame(columns=cols)
        log_fn("  Warning: No correlations calculated (all groups have < 2 samples)")
        return correlation_df
    
    correlation_df = pd.DataFrame(correlation_results)
    
    # Summary statistics
    nas = correlation_df['correlation'].isna()
    log_fn(f"  {nas.sum()} ({nas.mean() * 100:.2f}%) correlations are NaN")
    log_fn(f"  Mean correlation: {correlation_df['correlation'].mean():.4f}")
    log_fn(f"  Median correlation: {correlation_df['correlation'].median():.4f}")
    log_fn(f"  Std correlation: {correlation_df['correlation'].std():.4f}")
    
    return correlation_df


def plot_sse_density(all_data_df: pd.DataFrame, output_dir: Path, group_by=['tissue'], log_fn=print):
    """Plot density of predicted vs true SSE values."""
    log_fn("Plotting SSE density...")
    
    samples = all_data_df[group_by[0]].unique()
    num_tissues = len(samples)
    num_cols = 1
    num_rows = (num_tissues + num_cols - 1) // num_cols
    
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(num_cols * 4, num_rows * 4), squeeze=False)
    
    for i, condition in enumerate(samples):
        condition_data = all_data_df[all_data_df[group_by[0]] == condition]
        
        row = i // num_cols
        col = i % num_cols
        ax = axs[row, col]
        
        # 2D density plot
        sns.kdeplot(
            x=condition_data['true_sse'],
            y=condition_data['pred_sse'],
            levels=10,
            fill=True,
            cmap="rocket_r",
            ax=ax
        )
        
        # Top histogram (True SSE)
        ax_histx = ax.inset_axes([0, 1.05, 1, 0.2], sharex=ax)
        ax_histx.hist(condition_data['true_sse'], bins=30, color='gray', alpha=0.7)
        ax_histx.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
        
        # Right histogram (Pred SSE)
        ax_histy = ax.inset_axes([1.05, 0, 0.2, 1], sharey=ax)
        ax_histy.hist(condition_data['pred_sse'], bins=30, orientation='horizontal', color='gray', alpha=0.7)
        ax_histy.tick_params(axis="y", which="both", left=False, right=False, labelleft=False)
        
        # Add mean lines
        mean_true = condition_data['true_sse'].mean()
        mean_pred = condition_data['pred_sse'].mean()
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
    
    log_fn("Starting evaluation of model predictions")
    log_fn(f"Test data directory: {test_data_dir}")
    log_fn(f"Predictions directory: {pred_dir}")
    log_fn(f"Output directory: {output_dir}")
    log_fn(f"Start time: {datetime.now().isoformat()}")
    
    try:

        # Load true labels and usage (sparse format)
        log_fn("Loading true labels and usage...")
        sequences, true_labels, true_sse, true_metadata = load_memmap_data(
            test_data_dir,
            load_labels=True,
            load_usage=True
        )
        log_fn(f"  Sequences shape: {sequences.shape}")
        log_fn(f"  True labels shape: {true_labels.shape}")
        log_fn(f"  True SSE shape: {true_sse.shape}")

        # Load predictions (sparse format)
        log_fn("Loading predictions...")
        pred_labels, pred_probs, pred_sse, pred_meta, pred_metadata = load_sparse_predictions(
            pred_dir, test_data_dir, log_fn
        )
        log_fn(f"  Predicted labels shape: {pred_labels.shape}")
        log_fn(f"  Predicted probs shape: {pred_probs.shape}")
        log_fn(f"  Predicted SSE shape: {pred_sse.shape}")
        
        # Get species info from test metadata
        if 'species_mapping' in pred_meta:
            species_mapping = pred_meta['species_mapping']
            species_ids = np.array(list(species_mapping.values()))
            species_id_to_name = {v: k for k, v in species_mapping.items()}
        else:
            # Default: single species
            species_ids = np.array([0])
            species_id_to_name = {0: 'unknown'}
        log_fn(f"  Species: {species_id_to_name}")
        
        # Evaluate splice site classification
        splice_eval = evaluate_splice_site_classification(
            true_labels, pred_probs, output_dir, log_fn
        )
        
        # Combine true and predicted SSE dataframes on sample_idx, position_idx, and condition_idx
        log_fn("Combining true and predicted SSE data for correlation analysis...")
        # Combine true and predicted usage into one dataframe
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            all_sse = pd.merge(
                true_sse, pred_sse, how='outer', on=['sample_idx', 'position', 'condition_idx']
            )
            # Rename 'value' to predicted_usage' and 'sse' to 'true_usage' for clarity
            all_sse = all_sse.rename(columns={'value': 'pred_sse', 'sse': 'true_sse'})

        # Add condition names
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            condition_mapping = pred_meta.get('usage_condition_mapping', {})
            all_sse['condition_name'] = all_sse['condition_idx'].apply(
                lambda idx: condition_mapping.get(str(idx), {}).get('condition_key', None) if pd.notna(idx) else None
            )

        # Extract tissue and timepoint from condition_name column (format: Tissue_timepoint)
        # Only process rows where condition_name is not None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            all_sse['tissue'] = all_sse['condition_name'].str.split('_').str[0]
            all_sse['timepoint'] = all_sse['condition_name'].str.split('_').str[1]
            # Convert timepoint to int, handling NaN values
            all_sse['timepoint'] = pd.to_numeric(all_sse['timepoint'], errors='coerce').astype('Int64')
        # Convert timepoint to int, handling NaN values
        all_sse['timepoint'] = pd.to_numeric(all_sse['timepoint'], errors='coerce').astype('Int64')
        
        # Save matched positions
        matched_file = output_dir / "matched_sse_positions.csv"
        all_sse.to_csv(matched_file, index=False)
        log_fn(f"Saved matched positions to {matched_file}")
        
        # Calculate correlations
        correlation_df = calculate_condition_correlations(all_sse, log_fn=log_fn)
        
        # Save correlation table
        correlation_file = output_dir / "sse_correlation_per_position.csv"
        correlation_df.to_csv(correlation_file, index=False)
        log_fn(f"Saved correlation table to {correlation_file}")
        
        # Generate correlation plots
        plot_sse_density(all_sse, output_dir, log_fn=log_fn)
        
        # Save results summary
        save_results_summary(splice_eval, output_dir, log_fn)
        
        log_fn("Evaluation completed successfully")
        log_fn(f"End time: {datetime.now().isoformat()}")
        
    except Exception as e:
        log_fn(f"ERROR: {str(e)}")
        import traceback
        log_fn(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
