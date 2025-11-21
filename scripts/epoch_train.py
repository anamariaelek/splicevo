"""Manual training epoch to understand model behavior.

This script provides detailed insights into the training process by:
- Running multiple epochs on a small data subset
- Analyzing per-position losses and distributions
- Tracking gradient norms and loss changes
- Comparing predictions vs targets for usage values

Usage:
    # Basic usage with default settings (100 samples, 3 batches, 2 epochs)
    python epoch_train.py --config /path/to/config.yaml

    # Custom number of samples and epochs
    python epoch_train.py --config /path/to/config.yaml --n-samples 200 --n-epochs 5

    # More batches per epoch for better statistics
    python epoch_train.py --config /path/to/config.yaml --n-batches 10

Example:
    python scripts/epoch_train.py \
        --config configs/training_full.yaml \
        --n-samples 50 \
        --n-batches 5 \
        --n-epochs 3

Output:
    - Data loading statistics (original and normalized)
    - Batch composition analysis
    - Per-position loss breakdown for each usage type (SSE, alpha, beta)
    - Target and prediction distributions
    - Gradient norm statistics
    - Epoch-by-epoch loss comparison
"""

import torch
import numpy as np
from pathlib import Path
import sys
import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from splicevo.model import SplicevoModel
from splicevo.training.dataset import SpliceDataset
from splicevo.training.losses import WeightedMSELoss
from splicevo.utils.data_utils import load_processed_data
from torch.utils.data import DataLoader


def load_small_data(data_path: str, n_samples: int = 100):
    """Load a small subset of data for debugging."""
    print(f"Loading {n_samples} samples from {data_path}...")
    
    # Use the utility function to load data
    sequences, labels, alpha, beta, sse = load_processed_data(data_path)
    
    # Take only n_samples
    sequences = sequences[:n_samples]
    labels = labels[:n_samples]
    
    # Only use SSE for training
    if sse is None:
        raise ValueError("SSE array not found in data")
    
    usage_sse = sse[:n_samples]
    
    # SSE is already in [0,1] range, no normalization needed
    print(f"\nSSE array stats (no normalization needed):")
    sse_clean = sse[:n_samples][~np.isnan(sse[:n_samples])]
    print(f"  usage_sse: min={sse_clean.min():.3f}, max={sse_clean.max():.3f}, mean={sse_clean.mean():.3f}, std={sse_clean.std():.3f}")
    print(f"    - 25th percentile: {np.percentile(sse_clean, 25):.3f}")
    print(f"    - 50th percentile: {np.percentile(sse_clean, 50):.3f}")
    print(f"    - 75th percentile: {np.percentile(sse_clean, 75):.3f}")
        
    print(f"\nData shapes:")
    print(f"  sequences: {sequences.shape}")
    print(f"  labels: {labels.shape}")
    print(f"  usage_sse: {usage_sse.shape}")
    
    return sequences, labels, usage_sse

def inspect_batch(batch, batch_idx):
    """Inspect a single batch."""

    # Print keys
    print(f"Batch {batch_idx} keys:")
    for key in batch.keys():
        print(f"  - {key}")

    sequences = batch['sequences']
    labels = batch['splice_labels']
    usage = batch['usage_targets']

    print(f"\n{'-'*60}")
    print(f"BATCH {batch_idx}")
    print(f"{'-'*60}")
    print(f"Sequences shape: {sequences.shape}")
    print(f"\nLabels shape: {labels.shape}")
    print(f"  - dtype: {labels.dtype}")
    print(f"  - unique values: {torch.unique(labels).tolist()}")
    print(f"  - class distribution:")
    for cls in torch.unique(labels):
        count = (labels == cls).sum().item()
        pct = 100 * count / labels.numel()
        print(f"    Class {cls}: {count:6d} ({pct:5.1f}%)")
    print(f"\nUsage arrays:")
    print(f"  usage type: {usage.type()}")
    print(f"  usage shape: {usage.shape}")
    
    # Only inspect SSE (first and only usage type)
    usage_type_names = ['sse']  # Only SSE
    for i, key in enumerate(usage_type_names): 
        if i >= usage.shape[-1]:
            break
        arr = usage[..., i]
        arr_no_nan = arr[~torch.isnan(arr)]
        print(f"  {key}: {arr.shape}")
        if arr_no_nan.numel() > 0:
            print(f"    - num valid data points (non-NaN): {arr_no_nan.numel()}")
            print(f"    - min: {arr_no_nan.min().item():.3f}")
            print(f"    - max: {arr_no_nan.max().item():.3f}")
            print(f"    - mean: {arr_no_nan.mean().item():.3f}")
            print(f"    - std: {arr_no_nan.std().item():.3f}")
            print(f"    - 25th percentile: {np.percentile(arr_no_nan.cpu().numpy(), 25):.3f}")
            print(f"    - 50th percentile: {np.percentile(arr_no_nan.cpu().numpy(), 50):.3f}")
            print(f"    - 75th percentile: {np.percentile(arr_no_nan.cpu().numpy(), 75):.3f}")
        else:
            print(f"    - No valid data points (all values are NaN)")


def manual_training_step(model, batch, optimizer, criterion_splice, criterion_usage, device, epoch_num, track_extremes=False):
    """Manually run one training step with detailed output."""
    sequences = batch['sequences']
    labels = batch['splice_labels']
    usage = batch['usage_targets']  # (batch, positions, n_conditions)

    # Move to device
    sequences = sequences.to(device)
    labels = labels.to(device)
    usage = usage.to(device)

    print(f"\n{'='*60}")
    print(f"Training step (epoch {epoch_num})")
    print(f"{'='*60}")

    # Zero gradients
    optimizer.zero_grad()

    # Forward pass
    model.train()

    # Output
    output = model(sequences)
    splice_logits = output['splice_logits']
    usage_preds = output['usage_predictions']

    # Compute losses
    splice_pred = splice_logits
    splice_true = labels

    # Only consider positions for splice sites
    splice_mask = (splice_true > 0)  # shape: [batch, positions]

    # Flatten for CrossEntropyLoss
    splice_pred_flat = splice_pred.reshape(-1, splice_pred.size(-1))  # [batch*positions, num_classes]
    splice_true_flat = splice_true.reshape(-1)                        # [batch*positions]

    splice_loss = criterion_splice(splice_pred_flat, splice_true_flat)

    # Detailed per-position loss analysis
    print(f"\nPer-position loss analysis:")
    print(f"  Total positions: {splice_mask.numel()}")
    print(f"  Splice site positions: {splice_mask.sum().item()}")
    print(f"  Non-splice positions: {(~splice_mask).sum().item()}")
    
    usage_losses = {}
    usage_losses_per_position = {}
    
    if splice_mask.sum() > 0:
        mask_flat = splice_mask.reshape(-1)  # [batch*positions]
        
        # SSE only
        usage_targets = usage  # [batch, positions, n_conditions]
        usage_preds_i = usage_preds  # [batch, positions, n_conditions]

        # Replace NaNs in the targets with zeros
        usage_targets_clean = torch.nan_to_num(usage_targets, nan=0.0)
        usage_preds_clean = torch.nan_to_num(usage_preds_i, nan=0.0)

        # Flatten batch and positions, keep n_conditions
        usage_targets_flat = usage_targets_clean.reshape(-1, usage_targets_clean.shape[-1])
        usage_preds_flat = usage_preds_clean.reshape(-1, usage_preds_clean.shape[-1])

        # Get masked values
        masked_preds = usage_preds_flat[mask_flat]  # [num_valid, n_conditions]
        masked_targets = usage_targets_flat[mask_flat]  # [num_valid, n_conditions]
        
        # Compute loss
        usage_loss = criterion_usage(masked_preds, masked_targets)
        
        # Per-position loss (without reduction)
        if hasattr(criterion_usage, 'reduction'):
            # Temporarily change reduction to 'none'
            original_reduction = criterion_usage.reduction
            criterion_usage.reduction = 'none'
            per_pos_loss = criterion_usage(masked_preds, masked_targets)  # [num_valid, n_conditions]
            criterion_usage.reduction = original_reduction
            
            # Statistics
            usage_losses_per_position = {
                'mean': per_pos_loss.mean(dim=1).mean().item(),
                'std': per_pos_loss.mean(dim=1).std().item(),
                'min': per_pos_loss.mean(dim=1).min().item(),
                'max': per_pos_loss.mean(dim=1).max().item(),
                'median': per_pos_loss.mean(dim=1).median().item()
            }
            
            # Target distribution
            print(f"\n  SSE targets at splice sites:")
            print(f"    - mean: {masked_targets.mean().item():.4f}")
            print(f"    - std: {masked_targets.std().item():.4f}")
            print(f"    - min: {masked_targets.min().item():.4f}")
            print(f"    - max: {masked_targets.max().item():.4f}")
            print(f"    - values near 0 (< 0.05): {(masked_targets < 0.05).sum().item() / masked_targets.numel() * 100:.1f}%")
            print(f"    - values near 1 (> 0.95): {(masked_targets > 0.95).sum().item() / masked_targets.numel() * 100:.1f}%")
            print(f"    - middle values (0.05-0.95): {((masked_targets >= 0.05) & (masked_targets <= 0.95)).sum().item() / masked_targets.numel() * 100:.1f}%")
            
            print(f"  SSE predictions at splice sites:")
            print(f"    - mean: {masked_preds.mean().item():.4f}")
            print(f"    - std: {masked_preds.std().item():.4f}")
            print(f"    - min: {masked_preds.min().item():.4f}")
            print(f"    - max: {masked_preds.max().item():.4f}")
            
            # Print tables per condition showing first 10 positions
            n_conditions = 1 # masked_targets.shape[1]
            n_positions = masked_targets.shape[0]
            n_show = min(10, n_positions)
            
            for cond_idx in range(n_conditions):
                print(f"\n  Condition {cond_idx}:")
                print(f"    Position | Target   | Pred     | Loss    ")
                print(f"    ---------|----------|----------|----------")
                for i in range(n_show):
                    t = masked_targets[i, cond_idx].item()
                    p = masked_preds[i, cond_idx].item()
                    l = per_pos_loss[i, cond_idx].item()
                    print(f"    {i:8d} | {t:8.4f} | {p:8.4f} | {l:8.6f}")
                
                if n_positions > n_show:
                    print(f"    ... ({n_positions - n_show} more positions)")
            
            print(f"\n  SSE per-position loss statistics (averaged over all conditions):")
            print(f"    - mean: {usage_losses_per_position['mean']:.6f}")
            print(f"    - std: {usage_losses_per_position['std']:.6f}")
            print(f"    - min: {usage_losses_per_position['min']:.6f}")
            print(f"    - max: {usage_losses_per_position['max']:.6f}")
            print(f"    - median: {usage_losses_per_position['median']:.6f}")
        
        # Track extreme values if requested
        extreme_losses = None
        if track_extremes:
            # Find positions with extreme SSE values (near 0 or 1) and middle values
            is_zero = (masked_targets < 0.05)  # [num_valid, n_conditions]
            is_one = (masked_targets > 0.95)
            is_middle = (masked_targets >= 0.05) & (masked_targets <= 0.95)
            
            # Get per-position losses (already computed above)
            if hasattr(criterion_usage, 'reduction'):
                original_reduction = criterion_usage.reduction
                criterion_usage.reduction = 'none'
                per_pos_loss = criterion_usage(masked_preds, masked_targets)
                criterion_usage.reduction = original_reduction
                
                # Extract losses for zeros, ones, and middle
                zero_losses = per_pos_loss[is_zero]
                one_losses = per_pos_loss[is_one]
                middle_losses = per_pos_loss[is_middle]
                
                extreme_losses = {
                    'zeros': zero_losses.detach().cpu().numpy() if zero_losses.numel() > 0 else np.array([]),
                    'ones': one_losses.detach().cpu().numpy() if one_losses.numel() > 0 else np.array([]),
                    'middle': middle_losses.detach().cpu().numpy() if middle_losses.numel() > 0 else np.array([]),
                    'n_zeros': is_zero.sum().item(),
                    'n_ones': is_one.sum().item(),
                    'n_middle': is_middle.sum().item(),
                    # Store predictions and targets for scatter plot
                    'targets': masked_targets.detach().cpu().numpy(),
                    'predictions': masked_preds.detach().cpu().numpy()
                }
        
    else:
        usage_loss = torch.tensor(0.0, device=device)
        extreme_losses = None

    # Weighted total loss
    splice_weight = 1.0
    usage_weight = 0.5
    total_loss = splice_weight * splice_loss + usage_weight * usage_loss

    print(f"\nCombined losses:")
    print(f"  splice_loss: {splice_loss.item():.6f}")
    print(f"  usage_loss_sse: {usage_loss.item():.6f}")
    print(f"  total_loss: {total_loss.item():.6f}")

    # Backward pass
    print(f"\nBackward pass...")
    total_loss.backward()

    # Check gradients
    grad_norms = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_norms.append(grad_norm)

    print(f"  gradient norms: min={min(grad_norms):.6f}, max={max(grad_norms):.6f}, mean={np.mean(grad_norms):.6f}")

    # Optimizer step
    optimizer.step()
    print(f"  optimizer step complete")

    return total_loss.item(), extreme_losses


def plot_losses(epoch_extreme_losses, output_dir=None):
    """Plot how losses changed for SSE values across epochs."""
    
    # Collect data
    epochs = []
    zero_means = []
    zero_stds = []
    one_means = []
    one_stds = []
    middle_means = []
    middle_stds = []
    
    for epoch_num, losses in epoch_extreme_losses.items():
        epochs.append(epoch_num)
        
        if len(losses['zeros']) > 0:
            zero_means.append(np.mean(losses['zeros']))
            zero_stds.append(np.std(losses['zeros']))
        else:
            zero_means.append(np.nan)
            zero_stds.append(np.nan)
        
        if len(losses['ones']) > 0:
            one_means.append(np.mean(losses['ones']))
            one_stds.append(np.std(losses['ones']))
        else:
            one_means.append(np.nan)
            one_stds.append(np.nan)
        
        if len(losses['middle']) > 0:
            middle_means.append(np.mean(losses['middle']))
            middle_stds.append(np.std(losses['middle']))
        else:
            middle_means.append(np.nan)
            middle_stds.append(np.nan)
    
    epochs = np.array(epochs)
    zero_means = np.array(zero_means)
    zero_stds = np.array(zero_stds)
    one_means = np.array(one_means)
    one_stds = np.array(one_stds)
    middle_means = np.array(middle_means)
    middle_stds = np.array(middle_stds)
    
    # Create single plot with all three lines
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Plot all three on same axes
    ax.errorbar(epochs, zero_means, yerr=zero_stds, marker='o', label='SSE ≈ 0 (< 0.05)', capsize=5, color='blue', alpha=0.7)
    ax.errorbar(epochs, one_means, yerr=one_stds, marker='s', label='SSE ≈ 1 (> 0.95)', capsize=5, color='orange', alpha=0.7)
    ax.errorbar(epochs, middle_means, yerr=middle_stds, marker='^', label='SSE middle (0.05-0.95)', capsize=5, color='green', alpha=0.7)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Loss progression for different SSE value ranges', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    
    if output_dir:
        output_path = Path(output_dir) / 'sse_loss_progression.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nLoss progression plot saved to: {output_path}")
    
    plt.show()
    
    return fig


def plot_scatter(epoch_extreme_losses, output_dir=None):
    """Plot scatter plots of predicted vs target SSE values for each epoch."""
    
    epochs = sorted(epoch_extreme_losses.keys())
    n_epochs = len(epochs)
    
    # Create subplots: one for each epoch
    n_cols = min(3, n_epochs)
    n_rows = (n_epochs + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    if n_epochs == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for idx, epoch in enumerate(epochs):
        ax = axes[idx]
        losses = epoch_extreme_losses[epoch]
        
        if 'targets' in losses and 'predictions' in losses:
            targets = losses['targets'].flatten()
            predictions = losses['predictions'].flatten()
            
            # Create scatter plot
            ax.scatter(targets, predictions, alpha=0.3, s=10, color='blue', edgecolors='none')
            
            # Add diagonal line (perfect prediction)
            min_val = min(targets.min(), predictions.min())
            max_val = max(targets.max(), predictions.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction')
            
            # Add threshold lines for extreme values
            ax.axvline(x=0.05, color='gray', linestyle=':', alpha=0.5, linewidth=1, label='Extreme thresholds')
            ax.axvline(x=0.95, color='gray', linestyle=':', alpha=0.5, linewidth=1)
            ax.axhline(y=0.05, color='gray', linestyle=':', alpha=0.5, linewidth=1)
            ax.axhline(y=0.95, color='gray', linestyle=':', alpha=0.5, linewidth=1)
            
            # Calculate correlation
            correlation = np.corrcoef(targets, predictions)[0, 1]
            
            ax.set_xlabel('Target SSE', fontsize=10)
            ax.set_ylabel('Predicted SSE', fontsize=10)
            ax.set_title(f'Epoch {epoch} (r={correlation:.3f})', fontsize=11)
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')
            
            if idx == 0:
                ax.legend(fontsize=8, loc='upper left')
    
    # Hide unused subplots
    for idx in range(n_epochs, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if output_dir:
        output_path = Path(output_dir) / 'sse_scatter_plots.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Scatter plots saved to: {output_path}")
    
    plt.show()
    
    return fig


def print_loss_summary(epoch_extreme_losses):
    """Print summary of how losses changed for SSE values."""
    
    print(f"\n{'='*60}")
    print(f"EXTREME SSE VALUE LOSS TRACKING")
    print(f"{'='*60}")
    
    epochs = sorted(epoch_extreme_losses.keys())
    
    print(f"\nLoss changes for SSE ≈ 0 (< 0.05):")
    print(f"  Epoch | N positions | Mean Loss | Std Loss")
    print(f"  ------|-------------|-----------|----------")
    
    for epoch in epochs:
        losses = epoch_extreme_losses[epoch]
        n_zeros = losses['n_zeros']
        if len(losses['zeros']) > 0:
            mean_loss = np.mean(losses['zeros'])
            std_loss = np.std(losses['zeros'])
            print(f"  {epoch:5d} | {n_zeros:11d} | {mean_loss:9.6f} | {std_loss:8.6f}")
        else:
            print(f"  {epoch:5d} | {n_zeros:11d} | {'N/A':>9s} | {'N/A':>8s}")
    
    print(f"\nLoss changes for SSE ≈ 1 (> 0.95):")
    print(f"  Epoch | N positions | Mean Loss | Std Loss")
    print(f"  ------|-------------|-----------|----------")
    
    for epoch in epochs:
        losses = epoch_extreme_losses[epoch]
        n_ones = losses['n_ones']
        if len(losses['ones']) > 0:
            mean_loss = np.mean(losses['ones'])
            std_loss = np.std(losses['ones'])
            print(f"  {epoch:5d} | {n_ones:11d} | {mean_loss:9.6f} | {std_loss:8.6f}")
        else:
            print(f"  {epoch:5d} | {n_ones:11d} | {'N/A':>9s} | {'N/A':>8s}")
    
    print(f"\nLoss changes for middle SSE (0.05-0.95):")
    print(f"  Epoch | N positions | Mean Loss | Std Loss")
    print(f"  ------|-------------|-----------|----------")
    
    for epoch in epochs:
        losses = epoch_extreme_losses[epoch]
        n_middle = losses['n_middle']
        if len(losses['middle']) > 0:
            mean_loss = np.mean(losses['middle'])
            std_loss = np.std(losses['middle'])
            print(f"  {epoch:5d} | {n_middle:11d} | {mean_loss:9.6f} | {std_loss:8.6f}")
        else:
            print(f"  {epoch:5d} | {n_middle:11d} | {'N/A':>9s} | {'N/A':>8s}")
    
    # Print improvement
    if len(epochs) > 1:
        first_epoch = epochs[0]
        last_epoch = epochs[-1]
        
        print(f"\nImprovement from epoch {first_epoch} to {last_epoch}:")
        
        if len(epoch_extreme_losses[first_epoch]['zeros']) > 0 and len(epoch_extreme_losses[last_epoch]['zeros']) > 0:
            first_zero_loss = np.mean(epoch_extreme_losses[first_epoch]['zeros'])
            last_zero_loss = np.mean(epoch_extreme_losses[last_epoch]['zeros'])
            improvement = first_zero_loss - last_zero_loss
            pct_improvement = (improvement / first_zero_loss) * 100
            print(f"  SSE ≈ 0: {first_zero_loss:.6f} → {last_zero_loss:.6f} ({improvement:+.6f}, {pct_improvement:+.1f}%)")
        
        if len(epoch_extreme_losses[first_epoch]['ones']) > 0 and len(epoch_extreme_losses[last_epoch]['ones']) > 0:
            first_one_loss = np.mean(epoch_extreme_losses[first_epoch]['ones'])
            last_one_loss = np.mean(epoch_extreme_losses[last_epoch]['ones'])
            improvement = first_one_loss - last_one_loss
            pct_improvement = (improvement / first_one_loss) * 100
            print(f"  SSE ≈ 1: {first_one_loss:.6f} → {last_one_loss:.6f} ({improvement:+.6f}, {pct_improvement:+.1f}%)")
        
        if len(epoch_extreme_losses[first_epoch]['middle']) > 0 and len(epoch_extreme_losses[last_epoch]['middle']) > 0:
            first_middle_loss = np.mean(epoch_extreme_losses[first_epoch]['middle'])
            last_middle_loss = np.mean(epoch_extreme_losses[last_epoch]['middle'])
            improvement = first_middle_loss - last_middle_loss
            pct_improvement = (improvement / first_middle_loss) * 100
            print(f"  Middle: {first_middle_loss:.6f} → {last_middle_loss:.6f} ({improvement:+.6f}, {pct_improvement:+.1f}%)")


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Debug training with detailed per-position loss analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings (100 samples, 3 batches, 2 epochs)
  python %(prog)s --config configs/training_full.yaml

  # Run more epochs on more data
  python %(prog)s --config configs/training_full.yaml --n-samples 200 --n-epochs 5

  # Process more batches per epoch
  python %(prog)s --config configs/training_full.yaml --n-batches 10

  # Quick test with minimal data
  python %(prog)s --config configs/training_full.yaml --n-samples 20 --n-batches 1 --n-epochs 1

  # Output details:
  - Original and normalized usage statistics
  - Batch composition (splice sites vs non-sites)
  - Per-position loss for SSE, alpha, beta
  - Target distributions (near 0, near 1, middle values)
  - Prediction statistics
  - Gradient norms
  - Loss changes across epochs
        """
    )
    parser.add_argument('--config', type=str, required=True,
                        help='Path to training configuration file (YAML format)')
    parser.add_argument('--n-samples', type=int, default=100,
                        help='Number of samples to load from dataset (default: 100)')
    parser.add_argument('--n-batches', type=int, default=3,
                        help='Number of batches to process per epoch (default: 3)')
    parser.add_argument('--n-epochs', type=int, default=2,
                        help='Number of training epochs to run (default: 2)')
    args = parser.parse_args()
    
    # Validate arguments
    if args.n_samples < 1:
        parser.error("--n-samples must be at least 1")
    if args.n_batches < 1:
        parser.error("--n-batches must be at least 1")
    if args.n_epochs < 1:
        parser.error("--n-epochs must be at least 1")
    
    print(f"{'='*60}")
    print(f"EPOCH TRAINING DEBUG SCRIPT")
    print(f"{'='*60}")
    print(f"Configuration:")
    print(f"  Config file: {args.config}")
    print(f"  Samples: {args.n_samples}")
    print(f"  Batches per epoch: {args.n_batches}")
    print(f"  Epochs: {args.n_epochs}")
    print(f"{'='*60}\n")
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup device
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    print(f"Device: {device}\n")
    
    # Load data
    data_path = config['data']['path']
    sequences, labels, usage_sse = load_small_data(data_path, args.n_samples)
    
    # Create dataset
    dataset = SpliceDataset(sequences, labels, usage_sse)
    
    # Create dataloader
    batch_size = min(32, args.n_samples)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    print(f"\nDataLoader created:")
    print(f"  batch_size: {batch_size}")
    print(f"  n_batches: {len(loader)}")
    
    # Create model
    n_conditions = usage_sse.shape[2]
    model_config = config['model']
    
    model = SplicevoModel(
        embed_dim=model_config.get('embed_dim', 128),
        num_resblocks=model_config.get('num_resblocks', 8),
        num_classes=model_config.get('num_classes', 3),
        n_conditions=n_conditions,
        context_len=model_config.get('context_len', 4500),
        dropout=model_config.get('dropout', 0.5)
    )
    model = model.to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel created with {n_params:,} parameters")
    
    # Create optimizer and loss functions
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    criterion_splice = torch.nn.CrossEntropyLoss()
    
    # Use weighted MSE for usage prediction
    use_weighted_mse = config['training'].get('usage_loss_type', 'mse') == 'weighted_mse'
    if use_weighted_mse:
        extreme_low = config['training'].get('weighted_mse_extreme_low', 0.05)
        extreme_high = config['training'].get('weighted_mse_extreme_high', 0.95)
        extreme_weight = config['training'].get('weighted_mse_extreme_weight', 10.0)
        print(f"\nUsing weighted MSE loss:")
        print(f"  extreme_low: {extreme_low}")
        print(f"  extreme_high: {extreme_high}")
        print(f"  extreme_weight: {extreme_weight}x")
        criterion_usage = WeightedMSELoss(
            extreme_low_threshold=extreme_low,
            extreme_high_threshold=extreme_high,
            extreme_weight=extreme_weight,
            reduction='mean'
        )
    else:
        print(f"\nUsing standard MSE loss")
        criterion_usage = torch.nn.MSELoss()
    
    # Process batches for multiple epochs
    print(f"")
    print(f"Starting training: {args.n_epochs} epochs")
    print(f"")
    
    epoch_losses = []
    epoch_extreme_losses = {}  # Track losses for extreme SSE values
    
    for epoch in range(args.n_epochs):
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch + 1}/{args.n_epochs}")
        print(f"{'='*60}")
        
        batch_losses = []
        batch_extreme_losses = {
            'zeros': [], 
            'ones': [], 
            'middle': [], 
            'n_zeros': 0, 
            'n_ones': 0, 
            'n_middle': 0,
            'targets': [],  # Accumulate targets
            'predictions': []  # Accumulate predictions
        }
        
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= args.n_batches:
                break

            # Inspect batch (only for first epoch, first batch)
            if epoch == 0 and batch_idx == 0:
                inspect_batch(batch, batch_idx)

            # Manual training step (track extremes for all epochs)
            loss, extreme_losses = manual_training_step(
                model, batch, optimizer,
                criterion_splice, criterion_usage, device,
                epoch_num=epoch + 1,
                track_extremes=True
            )
            batch_losses.append(loss)
            
            # Accumulate extreme losses
            if extreme_losses is not None:
                if len(extreme_losses['zeros']) > 0:
                    batch_extreme_losses['zeros'].extend(extreme_losses['zeros'].tolist())
                if len(extreme_losses['ones']) > 0:
                    batch_extreme_losses['ones'].extend(extreme_losses['ones'].tolist())
                if len(extreme_losses['middle']) > 0:
                    batch_extreme_losses['middle'].extend(extreme_losses['middle'].tolist())
                batch_extreme_losses['n_zeros'] += extreme_losses['n_zeros']
                batch_extreme_losses['n_ones'] += extreme_losses['n_ones']
                batch_extreme_losses['n_middle'] += extreme_losses['n_middle']
                
                # Accumulate targets and predictions
                if 'targets' in extreme_losses and 'predictions' in extreme_losses:
                    batch_extreme_losses['targets'].append(extreme_losses['targets'])
                    batch_extreme_losses['predictions'].append(extreme_losses['predictions'])
        
        # Store extreme losses for this epoch
        epoch_extreme_losses[epoch + 1] = {
            'zeros': np.array(batch_extreme_losses['zeros']),
            'ones': np.array(batch_extreme_losses['ones']),
            'middle': np.array(batch_extreme_losses['middle']),
            'n_zeros': batch_extreme_losses['n_zeros'],
            'n_ones': batch_extreme_losses['n_ones'],
            'n_middle': batch_extreme_losses['n_middle'],
            # Concatenate all targets and predictions from all batches
            'targets': np.concatenate(batch_extreme_losses['targets']) if batch_extreme_losses['targets'] else np.array([]),
            'predictions': np.concatenate(batch_extreme_losses['predictions']) if batch_extreme_losses['predictions'] else np.array([])
        }
        
        avg_epoch_loss = np.mean(batch_losses)
        epoch_losses.append(avg_epoch_loss)
        
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch + 1} SUMMARY")
        print(f"{'='*60}")
        print(f"  Batches processed: {len(batch_losses)}")
        print(f"  Average loss: {avg_epoch_loss:.6f}")
        print(f"  Loss std: {np.std(batch_losses):.6f}")
        print(f"  Min loss: {np.min(batch_losses):.6f}")
        print(f"  Max loss: {np.max(batch_losses):.6f}")
    
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Epoch losses:")
    for i, loss in enumerate(epoch_losses):
        print(f"  Epoch {i+1}: {loss:.6f}")
    
    # Print and plot loss tracking
    print_loss_summary(epoch_extreme_losses)
    
    # Create plots - save to script directory
    script_dir = Path(__file__).parent
    plot_losses(epoch_extreme_losses, output_dir=script_dir)
    plot_scatter(epoch_extreme_losses, output_dir=script_dir)


if __name__ == '__main__':
    main()