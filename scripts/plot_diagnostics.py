"""Plot diagnostic plots from saved training data."""

import argparse
import sys
import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


def load_training_data(checkpoint_dir: Path):
    """Load training data from checkpoint directory."""
    # Load training history
    config_path = checkpoint_dir / 'training_config.json'
    if not config_path.exists():
        raise FileNotFoundError(f"Training config not found: {config_path}")
    
    with open(config_path, 'r') as f:
        full_config = json.load(f)
    
    config = full_config.get('config', {})
    
    # Load checkpoint to get history
    checkpoint_files = list(checkpoint_dir.glob('best_model.pt'))
    if not checkpoint_files:
        checkpoint_files = list(checkpoint_dir.glob('checkpoint_epoch_*.pt'))
        if checkpoint_files:
            # Sort by epoch number
            checkpoint_files.sort(key=lambda x: int(x.stem.split('_')[-1]))
    
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")
    
    import torch
    checkpoint = torch.load(checkpoint_files[-1], map_location='cpu')
    history = checkpoint.get('history', {})
    
    # Try to load SSE loss tracking if available
    npz_path = checkpoint_dir / 'diagnostics' / 'loss_tracking.npz'
    sse_tracking = None
    if npz_path.exists():
        sse_data = np.load(npz_path, allow_pickle=True)
        sse_tracking = {
            'train': sse_data['train_tracking'].item(),
            'val': sse_data['val_tracking'].item()
        }
    
    # Extract usage loss type from config
    usage_loss_type = config.get('training', {}).get('usage_loss_type', 'weighted_mse')
    
    return history, sse_tracking, usage_loss_type, config


def plot_loss_progression(history: dict, output_dir: Path, figsize: tuple = (16, 12)):
    """Plot loss progression across epochs."""
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Plot 1: Combined loss
    ax = axes[0, 0]
    epochs = np.arange(1, len(history['train_loss']) + 1)
    ax.plot(epochs, history['train_loss'], marker='o', label='Train', color='blue', alpha=0.7)
    if history.get('val_loss'):
        ax.plot(epochs, history['val_loss'], marker='s', label='Val', color='orange', alpha=0.7)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Combined Loss', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    # Plot 2: Splice loss
    ax = axes[0, 1]
    ax.plot(epochs, history['train_splice_loss'], marker='o', label='Train', color='blue', alpha=0.7)
    if history.get('val_splice_loss'):
        ax.plot(epochs, history['val_splice_loss'], marker='s', label='Val', color='orange', alpha=0.7)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Splice Classification Loss', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    # Plot 3: Usage loss
    ax = axes[1, 0]
    ax.plot(epochs, history['train_usage_loss'], marker='o', label='Train', color='blue', alpha=0.7)
    if history.get('val_usage_loss'):
        ax.plot(epochs, history['val_usage_loss'], marker='s', label='Val', color='orange', alpha=0.7)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Usage Prediction Loss', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    # Plot 4: Learning rate
    ax = axes[1, 1]
    ax.plot(epochs, history['learning_rate'], marker='o', color='green', alpha=0.7)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Learning Rate', fontsize=12)
    ax.set_title('Learning Rate Schedule', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'loss_progression.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_dir / 'loss_progression.png'}")


def plot_weighted_mse_losses(sse_tracking: dict, output_dir: Path, figsize: tuple = (16, 6)):
    """Plot weighted MSE losses for different SSE ranges."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    for split_idx, (split, ax) in enumerate([('train', ax1), ('val', ax2)]):
        tracking = sse_tracking[split]
        if not tracking:
            ax.text(0.5, 0.5, f'No {split} data', ha='center', va='center', transform=ax.transAxes)
            continue
        
        epochs = sorted(tracking.keys())
        
        zero_means = []
        one_means = []
        middle_means = []
        
        for epoch in epochs:
            data = tracking[epoch]
            if len(data['zeros']) > 0:
                zero_means.append(np.mean(data['zeros']))
            else:
                zero_means.append(np.nan)
            
            if len(data['ones']) > 0:
                one_means.append(np.mean(data['ones']))
            else:
                one_means.append(np.nan)
            
            if len(data['middle']) > 0:
                middle_means.append(np.mean(data['middle']))
            else:
                middle_means.append(np.nan)
        
        epochs = np.array(epochs)
        
        ax.plot(epochs, zero_means, marker='o', label='SSE ≈ 0 (< 0.05)', color='blue', alpha=0.7)
        ax.plot(epochs, one_means, marker='s', label='SSE ≈ 1 (> 0.95)', color='orange', alpha=0.7)
        ax.plot(epochs, middle_means, marker='^', label='SSE middle (0.05-0.95)', color='green', alpha=0.7)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title(f'{split.capitalize()} Weighted MSE Losses', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'weighted_mse_losses.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_dir / 'weighted_mse_losses.png'}")


def plot_splice_losses(checkpoint_dir: Path, output_dir: Path, figsize: tuple = (16, 6)):
    """Plot splice classification losses per class."""
    # Load splice class tracking
    tracking_file = checkpoint_dir / 'diagnostics' / 'splice_class_tracking.json'
    if not tracking_file.exists():
        print(f"Note: Splice class tracking not found at {tracking_file}")
        return
    
    with open(tracking_file, 'r') as f:
        splice_tracking = json.load(f)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    for split_idx, (split, ax) in enumerate([('train', ax1), ('val', ax2)]):
        tracking = splice_tracking.get(split, {})
        if not tracking:
            ax.text(0.5, 0.5, f'No {split} data', ha='center', va='center', transform=ax.transAxes)
            continue
        
        # Convert string keys to integers for sorting
        epochs = sorted([int(k) for k in tracking.keys()])
        
        class_0_losses = [tracking[str(e)]['class_0'] for e in epochs]
        class_1_losses = [tracking[str(e)]['class_1'] for e in epochs]
        class_2_losses = [tracking[str(e)]['class_2'] for e in epochs]
        
        epochs = np.array(epochs)
        
        ax.plot(epochs, class_0_losses, marker='o', label='Class 0 (No splice)', color='blue', alpha=0.7)
        ax.plot(epochs, class_1_losses, marker='s', label='Class 1 (Acceptor)', color='orange', alpha=0.7)
        ax.plot(epochs, class_2_losses, marker='^', label='Class 2 (Donor)', color='green', alpha=0.7)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title(f'{split.capitalize()} Splice Class Losses', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'splice_class_losses.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_dir / 'splice_class_losses.png'}")


def plot_hybrid_losses(checkpoint_dir: Path, output_dir: Path, figsize: tuple = (16, 12)):
    """Plot hybrid loss components (regression and classification) by SSE range."""
    # Load hybrid loss tracking
    tracking_file = checkpoint_dir / 'diagnostics' / 'hybrid_loss_tracking.json'
    if not tracking_file.exists():
        print(f"Note: Hybrid loss tracking not found at {tracking_file}")
        return
    
    with open(tracking_file, 'r') as f:
        hybrid_tracking = json.load(f)
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Plot 1: Overall components (train)
    ax = axes[0, 0]
    tracking = hybrid_tracking.get('train', {})
    if tracking:
        epochs = sorted([int(k) for k in tracking.keys()])
        reg_losses = [tracking[str(e)]['regression'] for e in epochs]
        class_losses = [tracking[str(e)]['classification'] for e in epochs]
        epochs = np.array(epochs)
        
        ax.plot(epochs, reg_losses, marker='o', label='Regression', color='blue', alpha=0.7)
        ax.plot(epochs, class_losses, marker='s', label='Classification', color='orange', alpha=0.7)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Train: Hybrid Loss Components', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
    else:
        ax.text(0.5, 0.5, 'No train data', ha='center', va='center', transform=ax.transAxes)
    
    # Plot 2: Overall components (val)
    ax = axes[0, 1]
    tracking = hybrid_tracking.get('val', {})
    if tracking:
        epochs = sorted([int(k) for k in tracking.keys()])
        reg_losses = [tracking[str(e)]['regression'] for e in epochs]
        class_losses = [tracking[str(e)]['classification'] for e in epochs]
        epochs = np.array(epochs)
        
        ax.plot(epochs, reg_losses, marker='o', label='Regression', color='blue', alpha=0.7)
        ax.plot(epochs, class_losses, marker='s', label='Classification', color='orange', alpha=0.7)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Val: Hybrid Loss Components', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
    else:
        ax.text(0.5, 0.5, 'No val data', ha='center', va='center', transform=ax.transAxes)
    
    # Plot 3: Classification by range (train)
    ax = axes[1, 0]
    tracking = hybrid_tracking.get('train', {})
    if tracking:
        epochs = sorted([int(k) for k in tracking.keys()])
        zeros_losses = [tracking[str(e)]['class_zeros'] for e in epochs]
        ones_losses = [tracking[str(e)]['class_ones'] for e in epochs]
        middle_losses = [tracking[str(e)]['class_middle'] for e in epochs]
        epochs = np.array(epochs)
        
        ax.plot(epochs, zeros_losses, marker='o', label='SSE ≈ 0 (< 0.05)', color='blue', alpha=0.7)
        ax.plot(epochs, ones_losses, marker='s', label='SSE ≈ 1 (> 0.95)', color='orange', alpha=0.7)
        ax.plot(epochs, middle_losses, marker='^', label='SSE middle (0.05-0.95)', color='green', alpha=0.7)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Classification Loss', fontsize=12)
        ax.set_title('Train: Classification Loss by SSE Range', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
    else:
        ax.text(0.5, 0.5, 'No train data', ha='center', va='center', transform=ax.transAxes)
    
    # Plot 4: Classification by range (val)
    ax = axes[1, 1]
    tracking = hybrid_tracking.get('val', {})
    if tracking:
        epochs = sorted([int(k) for k in tracking.keys()])
        zeros_losses = [tracking[str(e)]['class_zeros'] for e in epochs]
        ones_losses = [tracking[str(e)]['class_ones'] for e in epochs]
        middle_losses = [tracking[str(e)]['class_middle'] for e in epochs]
        epochs = np.array(epochs)
        
        ax.plot(epochs, zeros_losses, marker='o', label='SSE ≈ 0 (< 0.05)', color='blue', alpha=0.7)
        ax.plot(epochs, ones_losses, marker='s', label='SSE ≈ 1 (> 0.95)', color='orange', alpha=0.7)
        ax.plot(epochs, middle_losses, marker='^', label='SSE middle (0.05-0.95)', color='green', alpha=0.7)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Classification Loss', fontsize=12)
        ax.set_title('Val: Classification Loss by SSE Range', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
    else:
        ax.text(0.5, 0.5, 'No val data', ha='center', va='center', transform=ax.transAxes)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'hybrid_loss_components.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_dir / 'hybrid_loss_components.png'}")


def main():
    parser = argparse.ArgumentParser(description='Plot diagnostic plots from training data')
    parser.add_argument('--checkpoint-dir', type=str, required=True,
                       help='Directory containing training checkpoints')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for plots (default: checkpoint_dir/diagnostics)')
    parser.add_argument('--figsize', type=float, nargs=2, default=[8, 6],
                       metavar=('WIDTH', 'HEIGHT'),
                       help='Figure size in inches (default: 8 6)')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress output messages')
    
    args = parser.parse_args()
    
    checkpoint_dir = Path(args.checkpoint_dir)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
    
    output_dir = Path(args.output_dir) if args.output_dir else checkpoint_dir / 'diagnostics'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    figsize = tuple(args.figsize)
    figsize_2col = (figsize[0], figsize[1] / 2)  # For 2-column plots
    
    def log_print(msg):
        if not args.quiet:
            print(msg)
    
    log_print(f"Loading training data from: {checkpoint_dir}")
    history, sse_tracking, usage_loss_type, config = load_training_data(checkpoint_dir)
    
    log_print(f"Usage loss type: {usage_loss_type}")
    log_print(f"Plotting diagnostics to: {output_dir}")
    log_print(f"Figure size: {figsize[0]:.1f} x {figsize[1]:.1f} inches")
    
    # Plot 1: Loss progression
    log_print("\nPlotting loss progression...")
    plot_loss_progression(history, output_dir, figsize=figsize)

    # Plot 2: splice predictions 
    log_print("\nPlotting splice predictions...")
    plot_splice_losses(checkpoint_dir, output_dir, figsize=figsize_2col)

    # Plot 3: SSE-specific plots (if available)
    if sse_tracking is not None:
        if usage_loss_type in ['weighted_mse']:
            log_print("Plotting weighted MSE losses...")
            plot_weighted_mse_losses(sse_tracking, output_dir, figsize=figsize_2col)
        elif usage_loss_type in ['hybrid']:
            log_print("Plotting hybrid losses...")
            plot_hybrid_losses(checkpoint_dir, output_dir, figsize=figsize)
    else:
        log_print("No SSE tracking data found, skipping SSE-specific plots")
    
    log_print(f"\nDiagnostic plots saved to: {output_dir}")


if __name__ == '__main__':
    main()
