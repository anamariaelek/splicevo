"""Manual training epoch to understand model behavior."""

import torch
import numpy as np
from pathlib import Path
import sys
import yaml
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from splicevo.model import SplicevoModel
from splicevo.training.dataset import SpliceDataset
from splicevo.training.normalization import normalize_usage_arrays
from torch.utils.data import DataLoader


def load_small_data(data_path: str, n_samples: int = 100):
    """Load a small subset of data for debugging."""
    print(f"Loading {n_samples} samples from {data_path}...")
    
    with np.load(data_path) as data:
        sequences = data['sequences'][:n_samples]
        labels = data['labels'][:n_samples]
        usage_arrays = {
            'alpha': data['usage_alpha'][:n_samples],
            'beta': data['usage_beta'][:n_samples],
            'sse': data['usage_sse'][:n_samples],
        }
    
    # Normalize
    normalized_usage, stats = normalize_usage_arrays(usage_arrays, method='per_sample_cpm')

    # Original data distribution stats
    print(f"\nOriginal usage array stats:")
    for k, v in usage_arrays.items():
        v=v[~np.isnan(v)]  # Ignore NaNs
        print(f"  usage_{k}: min={v.min():.3f}, max={v.max():.3f}, mean={v.mean():.3f}, std={v.std():.3f}")
        print(f"    - 25th percentile: {np.percentile(v, 25):.3f}")
        print(f"    - 50th percentile: {np.percentile(v, 50):.3f}")
        print(f"    - 75th percentile: {np.percentile(v, 75):.3f}")
    
    # Normalized data distribution stats
    print(f"\nNormalized usage array stats:")
    for k, v in normalized_usage.items():
        v=v[~np.isnan(v)]  # Ignore NaNs if any
        print(f"  usage_{k}: min={v.min():.3f}, max={v.max():.3f}, mean={v.mean():.3f}, std={v.std():.3f}")
        print(f"    - 25th percentile: {np.percentile(v, 25):.3f}")
        print(f"    - 50th percentile: {np.percentile(v, 50):.3f}")
        print(f"    - 75th percentile: {np.percentile(v, 75):.3f}")
        
    print(f"Data shapes:")
    print(f"  sequences: {sequences.shape}")
    print(f"  labels: {labels.shape}")
    for k, v in normalized_usage.items():
        print(f"  usage_{k}: {v.shape}")
    
    return sequences, labels, normalized_usage

def inspect_batch(batch, batch_idx):
    """Inspect a single batch."""

    # Print keys
    print(f"Batch {batch_idx} keys:")
    for key in batch.keys():
        print(f"  - {key}")

    sequences = batch['sequences']
    labels = batch['splice_labels']
    usage = batch['usage_targets']

    print(f"\n{'='*60}")
    print(f"BATCH {batch_idx}")
    print(f"{'='*60}")
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
    for i, key in enumerate(['sse', 'alpha', 'beta']): 
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


def manual_training_step(model, batch, optimizer, criterion_splice, criterion_usage, device):
    """Manually run one training step with detailed output."""
    sequences = batch['sequences']
    labels = batch['splice_labels']
    usage = batch['usage_targets']

    # Move to device
    sequences = sequences.to(device)
    labels = labels.to(device)
    usage_keys = ['alpha', 'beta', 'sse']
    if isinstance(usage, dict):
        usage = torch.stack([usage[k].to(device) for k in usage_keys], dim=1)  # shape: (batch, 3, ...)
    else:
        usage = usage.to(device)

    print(f"\n{'='*60}")
    print(f"TRAINING STEP")
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

    usage_losses = {}
    if splice_mask.sum() > 0:
        mask_flat = splice_mask.reshape(-1)  # [batch*positions]
        for i, key in enumerate(['sse', 'alpha', 'beta']):  # SSE first
            usage_targets = usage[..., i]    # [batch, positions, n_conditions]
            usage_preds_i = usage_preds[..., i]  # [batch, positions, n_conditions]

            # Replace NaNs in the targets with zeros
            usage_targets = torch.nan_to_num(usage_targets, nan=0.0)
            usage_preds_i = torch.nan_to_num(usage_preds_i, nan=0.0)

            # Flatten batch and positions, keep n_conditions
            usage_targets_flat = usage_targets.reshape(-1, usage_targets.shape[-1])  # [batch*positions, n_conditions]
            usage_preds_flat = usage_preds_i.reshape(-1, usage_preds_i.shape[-1])    # [batch*positions, n_conditions]

            # Mask valid positions for all conditions
            usage_losses[key] = criterion_usage(
                usage_preds_flat[mask_flat],    # [num_valid, n_conditions]
                usage_targets_flat[mask_flat]   # [num_valid, n_conditions]
            )
        usage_loss = sum(usage_losses.values()) / len(usage_losses)
    else:
        usage_loss = torch.tensor(0.0, device=device)

    # Weighted total loss
    splice_weight = 1.0
    usage_weight = 0.5
    total_loss = splice_weight * splice_loss + usage_weight * usage_loss

    print(f"\nLosses:")
    print(f"  splice_loss: {splice_loss.item():.6f}")
    for key, loss in usage_losses.items():
        print(f"  usage_loss_{key}: {loss.item():.6f}")
    print(f"  usage_loss (avg): {usage_loss.item():.6f}")
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

    return total_loss.item()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Config file')
    parser.add_argument('--n-samples', type=int, default=100, help='Number of samples to load')
    parser.add_argument('--n-batches', type=int, default=3, help='Number of batches to process')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup device
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    print(f"Device: {device}\n")
    
    # Load data
    data_path = config['data']['path']
    sequences, labels, normalized_usage = load_small_data(data_path, args.n_samples)
    
    # Create dataset
    dataset = SpliceDataset(sequences, labels, normalized_usage)
    
    # Create dataloader
    batch_size = min(32, args.n_samples)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    print(f"\nDataLoader created:")
    print(f"  batch_size: {batch_size}")
    print(f"  n_batches: {len(loader)}")
    
    # Create model
    n_conditions = normalized_usage['alpha'].shape[2]
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
    criterion_usage = torch.nn.MSELoss()
    
    # Process batches
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= args.n_batches:
            break

        # Inspect batch
        inspect_batch(batch, batch_idx)

        # Manual training step
        loss = manual_training_step(
            model, batch, optimizer,
            criterion_splice, criterion_usage, device
        )
    
    print(f"\n{'='*60}")
    print(f"DEBUG COMPLETE")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()