"""Prediction script for splice site prediction models."""

import torch
import numpy as np
from pathlib import Path
import argparse
import json
from typing import Dict, Optional
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from splicevo.model import SplicevoModel
from splicevo.training.normalization import denormalize_usage, load_normalization_stats


def load_model_and_config(checkpoint_path: str, device: str = 'cuda') -> tuple:
    """Load trained model from checkpoint."""
    print(f"Loading checkpoint from {checkpoint_path}...")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Try to find config file in same directory
    checkpoint_dir = Path(checkpoint_path).parent
    
    # Look for training config files
    config_files = list(checkpoint_dir.glob('training_config_*.json'))
    if not config_files:
        config_files = list(checkpoint_dir.glob('config_*.json'))
    
    model_config = None
    if config_files:
        # Use most recent config
        config_files.sort()
        config_path = config_files[-1]
        print(f"Loading model config from {config_path}...")
        with open(config_path, 'r') as f:
            full_config = json.load(f)
            # Handle both config formats
            if 'config' in full_config and 'model' in full_config['config']:
                model_config = full_config['config']['model']
            elif 'model_params' in full_config:
                model_config = full_config['model_params']
    
    if model_config is None:
        print("Warning: Could not find model config, using defaults")
        model_config = {
            'embed_dim': 128,
            'num_resblocks': 8,
            'dilation_strategy': 'alternating',
            'alternate': 2,
            'num_classes': 3,
            'context_len': 4500,
            'dropout': 0.5
        }
    
    # Infer n_conditions from checkpoint state_dict
    state_dict = checkpoint['model_state_dict']
    n_conditions = 1  # Default
    
    # Look for usage_predictor weight to infer n_conditions
    # Shape is [n_conditions * 3, embed_dim, 1]
    # Because we predict 3 values (alpha, beta, sse) per condition
    for key in state_dict.keys():
        if 'usage_predictor.weight' in key:
            usage_predictor_shape = state_dict[key].shape
            # First dimension is n_conditions * 3
            n_conditions = usage_predictor_shape[0] // 3
            print(f"Inferred n_conditions from checkpoint: {n_conditions}")
            print(f"  (usage_predictor shape: {usage_predictor_shape})")
            break
    
    # Override with config value if available
    if 'n_conditions' in model_config:
        n_conditions = model_config['n_conditions']
        print(f"Using n_conditions from config: {n_conditions}")
    
    # Create model
    model_params = {
        'embed_dim': model_config.get('embed_dim', 128),
        'num_resblocks': model_config.get('num_resblocks', 8),
        'dilation_strategy': model_config.get('dilation_strategy', 'alternating'),
        'alternate': model_config.get('alternate', 2),
        'num_classes': model_config.get('num_classes', 3),
        'n_conditions': n_conditions,
        'context_len': model_config.get('context_len', 4500),
        'dropout': model_config.get('dropout', 0.5)
    }
    
    print(f"\nModel configuration:")
    for key, value in model_params.items():
        print(f"  {key}: {value}")
    
    model = SplicevoModel(**model_params)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    # Report checkpoint info
    print(f"\n{'='*60}")
    print("CHECKPOINT INFORMATION:")
    print(f"{'='*60}")
    print(f"  Epoch: {checkpoint['epoch'] + 1}")
    print(f"  Best validation loss: {checkpoint.get('best_val_loss', 'N/A'):.4f}")
    
    if 'history' in checkpoint:
        history = checkpoint['history']
        n_epochs_trained = len(history['train_loss'])
        print(f"  Total epochs trained: {n_epochs_trained}")
        
        if len(history['train_loss']) > 0:
            print(f"  Initial train loss: {history['train_loss'][0]:.4f}")
            print(f"  Final train loss: {history['train_loss'][-1]:.4f}")
        
        if 'val_loss' in history and len(history['val_loss']) > 0:
            print(f"  Initial val loss: {history['val_loss'][0]:.4f}")
            print(f"  Final val loss: {history['val_loss'][-1]:.4f}")
            best_epoch = history['val_loss'].index(min(history['val_loss'])) + 1
            print(f"  Best epoch: {best_epoch} (val_loss: {min(history['val_loss']):.4f})")
    
    print(f"{'='*60}\n")
    print("Model loaded successfully!")
    
    return model, model_params


def load_test_data(data_path: str) -> Dict[str, np.ndarray]:
    """Load test dataset."""
    print(f"\nLoading test data from {data_path}...")
    data = np.load(data_path)
    
    print(f"Available keys: {list(data.keys())}")
    print(f"Sequences shape: {data['sequences'].shape}")
    print(f"Splice labels shape: {data['labels'].shape}")
    
    if 'usage_alpha' in data:
        print(f"Usage alpha shape: {data['usage_alpha'].shape}")
        print(f"Usage beta shape: {data['usage_beta'].shape}")
        print(f"Usage SSE shape: {data['usage_sse'].shape}")
    
    return {key: data[key] for key in data.keys()}


def denormalize_predictions(
    predictions: Dict[str, np.ndarray],
    normalization_stats_path: str,
    n_samples: int
) -> Dict[str, np.ndarray]:
    """
    Denormalize usage predictions back to original scale.
    
    Args:
        predictions: Dictionary with model predictions
        normalization_stats_path: Path to normalization stats JSON
        n_samples: Number of samples (for per-sample denormalization)
    
    Returns:
        Dictionary with denormalized predictions
    """
    print(f"\nDenormalizing predictions using stats from {normalization_stats_path}...")
    
    stats = load_normalization_stats(normalization_stats_path)
    
    usage_preds = predictions['usage_predictions']  # (n_samples, seq_len, n_conditions, 3)
    
    # Denormalize each component (alpha, beta, sse)
    denorm_alpha = np.zeros_like(usage_preds[..., 0])
    denorm_beta = np.zeros_like(usage_preds[..., 1])
    denorm_sse = np.zeros_like(usage_preds[..., 2])
    
    for i in range(n_samples):
        # Alpha (acceptor usage)
        denorm_alpha[i] = denormalize_usage(
            usage_preds[i, :, :, 0],
            stats,
            key='alpha',
            sample_idx=i if stats['method'] == 'per_sample_cpm' else None
        )
        
        # Beta (donor usage)
        denorm_beta[i] = denormalize_usage(
            usage_preds[i, :, :, 1],
            stats,
            key='beta',
            sample_idx=i if stats['method'] == 'per_sample_cpm' else None
        )
        
        # SSE (usually identity transform, but apply anyway)
        denorm_sse[i] = denormalize_usage(
            usage_preds[i, :, :, 2],
            stats,
            key='sse',
            sample_idx=i if stats['method'] == 'per_sample_cpm' else None
        )
        
        if (i + 1) % 100 == 0:
            print(f"  Denormalized {i+1}/{n_samples} samples...")
    
    denormalized = {
        'usage_alpha_original': denorm_alpha,
        'usage_beta_original': denorm_beta,
        'usage_sse_original': denorm_sse
    }
    
    print(f"Denormalization complete!")
    print(f"  Alpha range: [{denorm_alpha.min():.2f}, {denorm_alpha.max():.2f}]")
    print(f"  Beta range: [{denorm_beta.min():.2f}, {denorm_beta.max():.2f}]")
    print(f"  SSE range: [{denorm_sse.min():.2f}, {denorm_sse.max():.2f}]")
    
    return denormalized


def main():
    parser = argparse.ArgumentParser(description='Make predictions with trained splice site model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--test-data', type=str, required=True,
                        help='Path to test data (.npz file)')
    parser.add_argument('--normalization-stats', type=str, required=True,
                        help='Path to normalization stats (.json file)')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to save predictions (.npz file)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for inference')
    parser.add_argument('--save-logits', action='store_true',
                        help='Save raw logits (large file)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load model
    device = args.device if torch.cuda.is_available() else 'cpu'
    model, model_config = load_model_and_config(args.checkpoint, device=device)
    
    # Load test data
    test_data = load_test_data(args.test_data)
    sequences = test_data['sequences']
    n_samples = sequences.shape[0]
    
    # Make predictions using model's predict method
    print(f"\nMaking predictions on {n_samples} samples...")
    print(f"Using batch size: {args.batch_size}")
    
    # Convert to tensor but keep on CPU - model.predict will handle device transfers
    if isinstance(sequences, np.ndarray):
        sequences_tensor = torch.from_numpy(sequences).float()
    else:
        sequences_tensor = sequences
    
    # Model.predict should handle batching and device transfers internally
    predictions = model.predict(
        sequences_tensor,
        batch_size=args.batch_size
    )
    
    print(f"Predictions complete!")
    print(f"  Splice logits shape: {predictions['splice_logits'].shape}")
    print(f"  Splice probs shape: {predictions['splice_probs'].shape}")
    print(f"  Splice predictions shape: {predictions['splice_predictions'].shape}")
    print(f"  Usage predictions shape: {predictions['usage_predictions'].shape}")
    
    # Extract alpha, beta, sse from usage predictions
    # Shape: (n_samples, seq_len, n_conditions, 3) where last dim is [alpha, beta, sse]
    usage_preds = predictions['usage_predictions']
    usage_alpha = usage_preds[..., 0]  # (n_samples, seq_len, n_conditions)
    usage_beta = usage_preds[..., 1]   # (n_samples, seq_len, n_conditions)
    usage_sse = usage_preds[..., 2]    # (n_samples, seq_len, n_conditions)
    
    print(f"\nExtracted usage components:")
    print(f"  Alpha shape: {usage_alpha.shape}")
    print(f"  Beta shape: {usage_beta.shape}")
    print(f"  SSE shape: {usage_sse.shape}")
    
    # Denormalize usage predictions if needed
    # denormalized = denormalize_predictions(
    #     predictions,
    #     args.normalization_stats,
    #     n_samples
    # )
    
    # Prepare output
    output_data = {
        'splice_predictions': predictions['splice_predictions'],
        'splice_probs': predictions['splice_probs'],
        'usage_alpha': usage_alpha,
        'usage_beta': usage_beta,
        'usage_sse': usage_sse,
        # 'usage_alpha_denormalized': denormalized['usage_alpha_original'],
        # 'usage_beta_denormalized': denormalized['usage_beta_original'],
        # 'usage_sse_denormalized': denormalized['usage_sse_original']
    }
    
    if args.save_logits:
        output_data['splice_logits'] = predictions['splice_logits']
    
    # Add ground truth if available
    if 'labels' in test_data:
        output_data['labels_true'] = test_data['labels']
    if 'usage_alpha' in test_data:
        output_data['usage_alpha_true'] = test_data['usage_alpha']
        output_data['usage_beta_true'] = test_data['usage_beta']
        output_data['usage_sse_true'] = test_data['usage_sse']
    
    # Save predictions
    print(f"\nSaving predictions to {args.output}...")
    np.savez_compressed(args.output, **output_data)
    
    print("\nPrediction Summary:")
    print(f"  Total samples: {n_samples}")
    print(f"  Output file: {args.output}")
    print(f"  File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    print("\nDone!")


if __name__ == '__main__':
    main()
